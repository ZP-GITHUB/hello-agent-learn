"""④ 执行层的检索模块：调 SearchTool 抓网页，并把结果整理成能给模型看的上下文。

两个函数分别在 _execute_task() 的前、中两段被调用：
    dispatch_search()           → 跑检索，返回标准化后的结果；把 notices 交给上层发 status 事件
    prepare_research_context()  → 把结果去重、截断，产出「来源清单 + 长上下文」两份文本

注意模块级的 _GLOBAL_SEARCH_TOOL 是**导入期创建的单例**，多个 worker 线程共用它，
所以它内部的 SearchTool 会被并发调用。这也要求 config.py 在导入期就 load_dotenv()，
否则这里读不到 TAVILY_API_KEY。
"""

from __future__ import annotations

import logging
from typing import Any, Optional, Tuple

from hello_agents.tools import SearchTool

from config import Configuration
from utils import (
    deduplicate_and_format_sources,
    format_sources,
    get_config_value,
)

logger = logging.getLogger(__name__)

MAX_TOKENS_PER_SOURCE = 2000
_GLOBAL_SEARCH_TOOL = SearchTool(backend="hybrid")


def dispatch_search(
    query: str,
    config: Configuration,
    loop_count: int,
) -> Tuple[dict[str, Any] | None, list[str], Optional[str], str]:
    """执行一次检索，把 SearchTool 的返回值标准化成四元组。

    返回 (payload, notices, answer_text, backend_label)：
        payload        —— 结构化结果字典（含 results / answer / backend / notices）
        notices        —— 提示语列表，上层会逐条转成 status 事件给前端
        answer_text    —— 搜索后端直接给出的 AI 答案（可能为 None）
        backend_label  —— 实际生效的后端名（可能被 SearchTool 内部改写）

    特例：raw_response 是字符串时（例如「Tavily 未配置」这类纯提示），
    会包装成「空 results + notices」返回，让上层走 skipped 分支而不是抛异常。
    """

    search_api = get_config_value(config.search_api)

    try:
        raw_response = _GLOBAL_SEARCH_TOOL.run(
            {
                "input": query,
                "backend": search_api,
                "mode": "structured",
                "fetch_full_page": config.fetch_full_page,
                "max_results": 5,
                "max_tokens_per_source": MAX_TOKENS_PER_SOURCE,
                "loop_count": loop_count,
            }
        )
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.exception("Search backend %s failed: %s", search_api, exc)
        raise

    if isinstance(raw_response, str):
        notices = [raw_response]
        logger.warning("Search backend %s returned text notice: %s", search_api, raw_response)
        payload: dict[str, Any] = {
            "results": [],
            "backend": search_api,
            "answer": None,
            "notices": notices,
        }
    else:
        payload = raw_response
        notices = list(payload.get("notices") or [])

    backend_label = str(payload.get("backend") or search_api)
    answer_text = payload.get("answer")
    results = payload.get("results", [])

    if notices:
        for notice in notices:
            logger.info("Search notice (%s): %s", backend_label, notice)

    logger.info(
        "Search backend=%s resolved_backend=%s answer=%s results=%s",
        search_api,
        backend_label,
        bool(answer_text),
        len(results),
    )

    return payload, notices, answer_text, backend_label


def prepare_research_context(
    search_result: dict[str, Any] | None,
    answer_text: Optional[str],
    config: Configuration,
) -> tuple[str, str]:
    """把检索结果整理成下游总结 Agent 要用的两份文本。

    返回 (sources_summary, context)：
        sources_summary —— 简短来源清单，存进 task.sources_summary，供前端与报告引用
        context         —— 去重 + 截断后的长上下文，喂给总结 Agent；
                           若搜索后端给了 AI 直答，会以「AI直接答案：」前置
    """

    sources_summary = format_sources(search_result)
    context = deduplicate_and_format_sources(
        search_result or {"results": []},
        max_tokens_per_source=MAX_TOKENS_PER_SOURCE,
        fetch_full_page=config.fetch_full_page,
    )

    if answer_text:
        context = f"AI直接答案：\n{answer_text}\n\n{context}"

    return sources_summary, context
