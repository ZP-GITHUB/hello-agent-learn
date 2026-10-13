"""FastAPI 入口：把 DeepResearchAgent 以 HTTP（非流式）与 SSE（流式）两种方式暴露出去。

一次 ``POST /research/stream`` 请求的完整阶段：

    ① 接入层  main.py                  建配置 → 建 Agent → 返回 StreamingResponse
    ② 编排层  agent.run_stream()       建事件队列、按任务起线程、统一出口 enqueue()
    ③ 规划层  planner.plan_todo_list() 把研究主题拆成 3~5 个 TodoItem
    ④ 执行层  agent._execute_task()    N 个线程并行：检索 → 整理上下文 → 流式总结
    ⑤ 汇总层  reporter.generate_report() → _persist_final_report() 落盘 note

SSE 帧格式：每个事件序列化成一帧 ``data: {json}\\n\\n``（见 event_iterator）。
注意状态码语义：建流**之前**的配置错误会抛 HTTP 400；建流**之后**（即整个研究过程）
的异常一律被 event_iterator 捕获成 ``{"type": "error"}`` 事件，HTTP 状态码仍是 200。
"""

from __future__ import annotations

import json
import sys
from typing import Any, Dict, Iterator, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from loguru import logger
from pydantic import BaseModel, Field

from config import Configuration, SearchAPI
from agent import DeepResearchAgent

# 添加控制台日志处理程序
logger.add(
    sys.stderr,
    level="INFO",
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <4}</level> | <cyan>using_function:{function}</cyan> | <cyan>{file}:{line}</cyan> | <level>{message}</level>",
    colorize=True,
)


# 添加错误日志文件处理程序
logger.add(
    sink=sys.stderr,
    level="ERROR",
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <4}</level> | <cyan>using_function:{function}</cyan> | <cyan>{file}:{line}</cyan> | <level>{message}</level>",
    colorize=True,
)


class ResearchRequest(BaseModel):
    """研究请求体：两个接口（/research 与 /research/stream）共用。

    topic      —— 研究主题，必填。
    search_api —— 可选。填了会作为 override 覆盖 env 里的 SEARCH_API；
                  不填则完全走 Configuration.from_env() 的默认值。
    """

    topic: str = Field(..., description="Research topic supplied by the user")
    search_api: SearchAPI | None = Field(
        default=None,
        description="Override the default search backend configured via env",
    )


class ResearchResponse(BaseModel):
    """非流式接口 POST /research 的响应体（流式接口不返回它）。"""

    report_markdown: str = Field(
        ..., description="Markdown-formatted research report including sections"
    )
    todo_items: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Structured TODO items with summaries and sources",
    )


def _mask_secret(value: Optional[str], visible: int = 4) -> str:
    """Mask sensitive tokens while keeping leading and trailing characters."""
    if not value:
        return "unset"

    if len(value) <= visible * 2:
        return "*" * len(value)

    return f"{value[:visible]}...{value[-visible:]}"


def _build_config(payload: ResearchRequest) -> Configuration:
    """把「env 配置 + 请求体覆盖」合成一份 Configuration。

    先由 Configuration.from_env() 读全量环境变量，再把请求体里显式给出的字段
    作为 overrides 盖上去。两个接口在建 Agent 前都会先走这一步。
    """

    overrides: Dict[str, Any] = {}

    if payload.search_api is not None:
        overrides["search_api"] = payload.search_api

    return Configuration.from_env(overrides=overrides)


def create_app() -> FastAPI:
    """组装 FastAPI 应用：CORS、启动日志、/healthz、/research、/research/stream。"""

    app = FastAPI(title="HelloAgents Deep Researcher")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.on_event("startup")
    def log_startup_configuration() -> None:
        """启动时把生效配置打进日志（API Key 走 _mask_secret 脱敏）。"""

        config = Configuration.from_env()

        if config.llm_provider == "ollama":
            base_url = config.sanitized_ollama_url()
        elif config.llm_provider == "lmstudio":
            base_url = config.lmstudio_base_url
        else:
            base_url = config.llm_base_url or "unset"

        logger.info(
            "DeepResearch configuration loaded: provider=%s model=%s base_url=%s search_api=%s "
            "max_loops=%s fetch_full_page=%s tool_calling=%s strip_thinking=%s api_key=%s",
            config.llm_provider,
            config.resolved_model() or "unset",
            base_url,
            (config.search_api.value if isinstance(config.search_api, SearchAPI) else config.search_api),
            config.max_web_research_loops,
            config.fetch_full_page,
            config.use_tool_calling,
            config.strip_thinking_tokens,
            _mask_secret(config.llm_api_key),
        )

    @app.get("/healthz")
    def health_check() -> Dict[str, str]:
        """存活探针，不碰 LLM/搜索，恒返回 ok。"""

        return {"status": "ok"}

    @app.post("/research", response_model=ResearchResponse)
    def run_research(payload: ResearchRequest) -> ResearchResponse:
        """非流式入口：一次请求跑完全流程，最后统一返回报告 + 任务列表。

        与流式入口的差别：
        - 全程无事件输出，前端只能等最终响应；
        - 走 DeepResearchAgent.run()，内部对 _execute_task() 必须显式迭代
          （生成器函数只调用不迭代 = 一行都不执行，见 agent.py 的 [本地修复]）。
        """

        try:
            config = _build_config(payload)
            agent = DeepResearchAgent(config=config)
            result = agent.run(payload.topic)
        except ValueError as exc:  # Likely due to unsupported configuration
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:  # pragma: no cover - defensive guardrail
            raise HTTPException(status_code=500, detail="Research failed") from exc

        # TodoItem（dataclass）→ 可 JSON 序列化的 dict，字段与前端 todo_list 事件保持一致
        todo_payload = [
            {
                "id": item.id,
                "title": item.title,
                "intent": item.intent,
                "query": item.query,
                "status": item.status,
                "summary": item.summary,
                "sources_summary": item.sources_summary,
                "note_id": item.note_id,
                "note_path": item.note_path,
            }
            for item in result.todo_items
        ]

        return ResearchResponse(
            # report_markdown 为空时退到 running_summary，再退到空串
            report_markdown=(result.report_markdown or result.running_summary or ""),
            todo_items=todo_payload,
        )

    @app.post("/research/stream")
    def stream_research(payload: ResearchRequest) -> StreamingResponse:
        """流式入口：把 run_stream() 产出的每个事件即时推给前端（Server-Sent Events）。

        三段式：
        1) 建配置 + 建 Agent —— 只在这里可能抛 HTTP 400；
        2) event_iterator() 包装成同步生成器，逐帧 yield ``data: {...}\\n\\n``；
        3) 交给 StreamingResponse（media_type=text/event-stream），并关掉缓存。
        """

        try:
            config = _build_config(payload)
            agent = DeepResearchAgent(config=config)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        def event_iterator() -> Iterator[str]:
            """把 run_stream() 的事件流翻译成 SSE 帧。

            这是**同步生成器**，Starlette 会把它放进线程池里逐帧 next()，
            所以内部的阻塞（LLM 调用、网络搜索）不会卡死事件循环。
            研究过程中的任何异常都在这里被吞掉并转成一个 error 事件——
            前端拿到的是 HTTP 200 + error 事件，而不是 5xx。
            """

            try:
                for event in agent.run_stream(payload.topic):
                    yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
            except Exception as exc:  # pragma: no cover - defensive guardrail
                logger.exception("Streaming research failed")
                error_payload = {"type": "error", "detail": str(exc)}
                yield f"data: {json.dumps(error_payload, ensure_ascii=False)}\n\n"

        return StreamingResponse(
            event_iterator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            },
        )

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
