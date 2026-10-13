"""编排层：协调「规划 → 并行检索+总结 → 汇总报告」的深度研究工作流。

整体链路（`POST /research/stream`）：

    main.stream_research()
      └─ DeepResearchAgent(config)                ← 本文件：构造全部零件
      └─ agent.run_stream(topic)                  ← 本文件：事件生成器（流式主入口）
           ├─ planner.plan_todo_list()            ③ 规划层：主题 → 3~5 个 TodoItem
           ├─ Thread × N → worker()               ④ 执行层：每任务一线程，互不等待
           │    └─ _execute_task()                单任务流水线：检索 → 上下文 → 总结
           │         ├─ search.dispatch_search()
           │         ├─ search.prepare_research_context()
           │         └─ summarizer.stream_task_summary()
           ├─ reporter.generate_report()          ⑤ 汇总层：把 N 份任务总结拼成报告
           └─ _persist_final_report()             报告落盘为 note

并发模型（排障必读）：
- 事件出口只有一个 —— run_stream() 内嵌的 ``enqueue()``，由它统一补 ``step`` / ``stream_token``；
- 每个任务起一个 daemon 线程，**没有并发上限**（任务数 = 线程数 = 并发 LLM 请求数）；
- 线程通过 ``Queue`` 把事件交回主生成器，``__task_done__`` 是只有内部才认的哨兵，
  计数到任务总数就收尾，永远不会发给前端；
- 工具调用事件走**旁路双通道**：``_set_tool_event_sink()`` 打开后，框架回调会在
  worker 线程里实时把事件塞进同一个队列。此时 ``_drain_tool_events()`` 恒返回 ``[]``
  （见其实现），避免同一条事件被推送两次。非流式路径才靠 drain() 批量取。
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from queue import Empty, Queue
from threading import Lock, Thread
from typing import Any, Callable, Iterator

from hello_agents import HelloAgentsLLM, ToolAwareSimpleAgent
from hello_agents.tools import ToolRegistry
from hello_agents.tools.builtin.note_tool import NoteTool

from config import Configuration
from prompts import (
    report_writer_instructions,
    task_summarizer_instructions,
    todo_planner_system_prompt,
)
from models import SummaryState, SummaryStateOutput, TodoItem
from services.planner import PlanningService
from services.reporter import ReportingService
from services.search import dispatch_search, prepare_research_context
from services.summarizer import SummarizationService
from services.tool_events import ToolCallTracker

logger = logging.getLogger(__name__)


class DeepResearchAgent:
    """Coordinator orchestrating TODO-based research workflow using HelloAgents."""

    def __init__(self, config: Configuration | None = None) -> None:
        """初始化编排器：一次构造、全程复用所有零件。

        构造顺序与各自用途：
        1) ``_init_llm()``              统一 LLM 客户端（三个 Agent 共用一个）
        2) ``NoteTool`` + ``ToolRegistry``  笔记工具及其注册表（enable_notes=False 时都为空）
        3) ``ToolCallTracker``          工具调用的旁路收集器（供 SSE 推送与 task↔note 关联）
        4) 三个 ``ToolAwareSimpleAgent`` 规划 / 报告 / 总结（注意：报告那个 use_tools=False）
        5) 三个 Service                  PlanningService / SummarizationService / ReportingService
        """

        self.config = config or Configuration.from_env()
        self.llm = self._init_llm()

        self.note_tool = (
            NoteTool(workspace=self.config.notes_workspace)
            if self.config.enable_notes
            else None
        )
        self.tools_registry: ToolRegistry | None = None
        if self.note_tool:
            registry = ToolRegistry()
            registry.register_tool(self.note_tool)
            self.tools_registry = registry

        self._tool_tracker = ToolCallTracker(
            self.config.notes_workspace if self.config.enable_notes else None
        )
        self._tool_event_sink_enabled = False
        self._state_lock = Lock()

        self.todo_agent = self._create_tool_aware_agent(
            name="研究规划专家",
            system_prompt=todo_planner_system_prompt.strip(),
        )
        self.report_agent = self._create_tool_aware_agent(
            name="报告撰写专家",
            system_prompt=report_writer_instructions.strip(),
            # [本地修复] 报告正文所需的任务总结已在提示词中内联，无需工具；
            # 关闭工具能力可保证 run() 的返回值就是完整报告正文。
            use_tools=False,
        )

        self._summarizer_factory: Callable[[], ToolAwareSimpleAgent] = lambda: self._create_tool_aware_agent(  # noqa: E501
            name="任务总结专家",
            system_prompt=task_summarizer_instructions.strip(),
        )

        self.planner = PlanningService(self.todo_agent, self.config)
        self.summarizer = SummarizationService(self._summarizer_factory, self.config)
        self.reporting = ReportingService(self.report_agent, self.config)
        self._last_search_notices: list[str] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def _init_llm(self) -> HelloAgentsLLM:
        """按配置构造 HelloAgentsLLM（temperature 固定 0.0，研究场景不需要发散）。

        按 provider 分三种取 base_url/api_key 的策略：
        - ``ollama``   ：base_url 走 sanitized_ollama_url()（自动补 /v1），无 Key 时填占位 "ollama"
        - ``lmstudio`` ：base_url 走 lmstudio 专用字段
        - 其他（含 custom）：直接用 llm_base_url / llm_api_key
        """

        llm_kwargs: dict[str, Any] = {"temperature": 0.0}

        model_id = self.config.llm_model_id or self.config.local_llm
        if model_id:
            llm_kwargs["model"] = model_id

        provider = (self.config.llm_provider or "").strip()
        if provider:
            llm_kwargs["provider"] = provider

        if provider == "ollama":
            llm_kwargs["base_url"] = self.config.sanitized_ollama_url()
            if self.config.llm_api_key:
                llm_kwargs["api_key"] = self.config.llm_api_key
            else:
                llm_kwargs["api_key"] = "ollama"
        elif provider == "lmstudio":
            llm_kwargs["base_url"] = self.config.lmstudio_base_url
            if self.config.llm_api_key:
                llm_kwargs["api_key"] = self.config.llm_api_key
        else:
            if self.config.llm_base_url:
                llm_kwargs["base_url"] = self.config.llm_base_url
            if self.config.llm_api_key:
                llm_kwargs["api_key"] = self.config.llm_api_key

        return HelloAgentsLLM(**llm_kwargs)

    def _create_tool_aware_agent(
        self, *, name: str, system_prompt: str, use_tools: bool = True
    ) -> ToolAwareSimpleAgent:
        """Instantiate a ToolAwareSimpleAgent sharing tool registry and tracker.

        [本地修复] 新增 use_tools 开关：报告撰写 Agent 直接基于内联的任务总结生成正文，
        无需调用工具。关闭其工具能力后，模型无法再于"报告正文 + 保存笔记"同轮输出，
        从根上避免 run() 只返回最后一句交付摘要、报告正文丢失的问题。
        """
        registry = self.tools_registry if use_tools else None
        return ToolAwareSimpleAgent(
            name=name,
            llm=self.llm,
            system_prompt=system_prompt,
            enable_tool_calling=registry is not None,
            tool_registry=registry,
            tool_call_listener=self._tool_tracker.record,
        )

    def _set_tool_event_sink(self, sink: Callable[[dict[str, Any]], None] | None) -> None:
        """打开/关闭工具事件的「实时推送」开关。

        传 callable 时：ToolCallTracker.record() 每记录一条就立刻回调 sink
        （在 worker 子线程里执行），事件因此能实时进队；
        传 None 时：恢复成「攒着等 drain() 批量取」的模式（非流式路径用）。

        这也是 ``_drain_tool_events()`` 在流式期间恒返回 [] 的原因——通道二选一，防止重复。
        """

        self._tool_event_sink_enabled = sink is not None
        self._tool_tracker.set_event_sink(sink)

    def run(self, topic: str) -> SummaryStateOutput:
        """非流式主流程：跑完所有任务后一次性返回报告（对应 POST /research）。

        与 run_stream() 的差异：无事件、无线程（顺序执行）、_execute_task 必须显式迭代。
        """

        state = SummaryState(research_topic=topic)
        state.todo_items = self.planner.plan_todo_list(state)
        self._drain_tool_events(state)

        if not state.todo_items:
            logger.info("No TODO items generated; falling back to single task")
            state.todo_items = [self.planner.create_fallback_task(state)]

        for task in state.todo_items:
            # [本地修复] _execute_task 是生成器函数：只"调用"而不迭代，函数体一行都不会执行。
            # 上游此处直接调用（未迭代），导致非流式 POST /research 静默"假成功"——
            # 返回 HTTP 200 和一份像模像样的报告，但检索/总结全没跑、所有任务停留在 pending。
            # 这里显式迭代消费生成器，使任务真正执行（emit_stream=False 时不产生事件）。
            for _ in self._execute_task(state, task, emit_stream=False):
                pass

        report = self.reporting.generate_report(state)
        self._drain_tool_events(state)
        state.structured_report = report
        state.running_summary = report
        self._persist_final_report(state, report)

        return SummaryStateOutput(
            running_summary=report,
            report_markdown=report,
            todo_items=state.todo_items,
        )

    def run_stream(self, topic: str) -> Iterator[dict[str, Any]]:
        """流式主入口：边跑边 yield 事件（对应 POST /research/stream）。

        它是本项目的调度中枢，按顺序做这几件事：

            ① 建全局状态 state = SummaryState(...)
            ② 发 status，通知前端「开始初始化」
            ③ planner.plan_todo_list() 拆任务（阻塞，一次 LLM）
            ④ drain 规划阶段产生的 tool_call 事件并转发
            ⑤ 任务为空则兜底一个任务（create_fallback_task）
            ⑥ 建 channel_map：task.id → {step, token}，并写回 task.stream_token
            ⑦ 发 todo_list，前端此时就能画出全部任务卡片
            ⑧ 建 Queue + enqueue()（唯一出口）+ 打开工具事件实时通道
            ⑨ 每个任务起一个 daemon 线程跑 worker()
            ⑩ 消费队列直到收齐 N 个 __task_done__，再排空残余
            ⑪ 生成报告 → 落盘 note → 发 final_report + done

        注意第 ⑪ 步的报告生成是**同步阻塞**的单次 LLM 调用，期间不产生任何事件，
        前端会表现为「所有任务都 completed 了但页面卡住若干十秒」，这是正常现象。
        """

        # ① 全局状态袋：topic / todo_items / web_research_results / sources_gathered /
        #    research_loop_count / report_note_id 等，全流程共享
        state = SummaryState(research_topic=topic)
        logger.debug("Starting streaming research: topic=%s", topic)
        # ② 第一个事件：前端据此把界面切到「运行中」
        yield {"type": "status", "message": "初始化研究流程"}

        # ③ 规划层：拆出 3~5 个可并行的 TodoItem（阻塞，内部一次 LLM 调用）
        state.todo_items = self.planner.plan_todo_list(state)
        # ④ 把规划过程中模型对 note 工具的调用事件转发给前端
        for event in self._drain_tool_events(state, step=0):
            yield event
        # ⑤ 兜底：规划失败或产出空数组时至少给一个任务，保证流程能跑完
        if not state.todo_items:
            state.todo_items = [self.planner.create_fallback_task(state)]

        # ⑥ 给每个任务编上「第几步」和一个流式 token：之后所有事件靠 task_id 反查这两个值，
        #    前端才能判断某条 chunk 属于哪张任务卡
        channel_map: dict[int, dict[str, Any]] = {}
        for index, task in enumerate(state.todo_items, start=1):
            token = f"task_{task.id}"
            task.stream_token = token
            channel_map[task.id] = {"step": index, "token": token}

        # ⑦ 一次性把任务清单推给前端（此时所有任务都还是 pending）
        yield {
            "type": "todo_list",
            "tasks": [self._serialize_task(t) for t in state.todo_items],
            "step": 0,
        }

        # ⑧ 事件队列：worker 线程往里塞，主生成器从这里取并 yield 给 FastAPI。
        #    所有写入都必须经过 enqueue() —— 它是事件的唯一出口。
        event_queue: Queue[dict[str, Any]] = Queue()

        def enqueue(
            event: dict[str, Any],
            *,
            task: TodoItem | None = None,
            step_override: int | None = None,
        ) -> None:
            """事件唯一出口：统一补齐 task_id / step / stream_token 后入队。

            task          —— 显式指定事件归属的任务（worker 内部调用时必传）
            step_override —— 覆写 step（收尾阶段用它把报告相关事件标记为「任务数+1」）
            """

            payload = dict(event)
            target_task_id = payload.get("task_id")
            if task is not None:
                target_task_id = task.id
                payload["task_id"] = task.id

            channel = channel_map.get(target_task_id) if target_task_id is not None else None
            if channel:
                payload.setdefault("step", channel["step"])
                payload["stream_token"] = channel["token"]
            if step_override is not None:
                payload["step"] = step_override
            event_queue.put(payload)

        def tool_event_sink(event: dict[str, Any]) -> None:
            """框架回调入口：模型每次调用工具都会走到这里（在 worker 子线程中执行）。"""

            enqueue(event)

        # 打开实时通道：此后 _drain_tool_events() 恒返回 []，事件全部由上面的 sink 即时入队
        self._set_tool_event_sink(tool_event_sink)

        threads: list[Thread] = []

        def worker(task: TodoItem, step: int) -> None:
            """单任务线程体：发「开始」→ 跑 _execute_task → 发「完成/失败」→ 发哨兵。

            无论成功失败，finally 都会发出 __task_done__，主循环据此计数收尾；
            单个任务失败只会把它自己变成 task_status: failed，不会中断整条流。
            """

            try:
                enqueue(
                    {
                        "type": "task_status",
                        "task_id": task.id,
                        "status": "in_progress",
                        "title": task.title,
                        "intent": task.intent,
                        "note_id": task.note_id,
                        "note_path": task.note_path,
                    },
                    task=task,
                )

                for event in self._execute_task(state, task, emit_stream=True, step=step):
                    enqueue(event, task=task)
            except Exception as exc:  # pragma: no cover - defensive guardrail
                logger.exception("Task execution failed", exc_info=exc)
                enqueue(
                    {
                        "type": "task_status",
                        "task_id": task.id,
                        "status": "failed",
                        "detail": str(exc),
                        "title": task.title,
                        "intent": task.intent,
                        "note_id": task.note_id,
                        "note_path": task.note_path,
                    },
                    task=task,
                )
            finally:
                enqueue({"type": "__task_done__", "task_id": task.id})

        # ⑨ 每个任务各起一个 daemon 线程 —— 全部并行开跑，没有并发上限。
        #    规划出 5 个任务就是 5 路并发打同一个模型端点（上游有限流时会集中报错）。
        for task in state.todo_items:
            step = channel_map.get(task.id, {}).get("step", 0)
            thread = Thread(target=worker, args=(task, step), daemon=True)
            threads.append(thread)
            thread.start()

        active_workers = len(state.todo_items)
        finished_workers = 0

        # ⑩ 主循环：阻塞等待事件。__task_done__ 只用来计数，绝不出流给前端
        try:
            while finished_workers < active_workers:
                event = event_queue.get()
                if event.get("type") == "__task_done__":
                    finished_workers += 1
                    continue
                yield event

            # 收尾：把残余事件排空（non-blocking，取不到就退出）
            while True:
                try:
                    event = event_queue.get_nowait()
                except Empty:
                    break
                if event.get("type") != "__task_done__":
                    yield event
        finally:
            # 关掉实时通道，并等所有 worker 真正结束，避免线程泄漏到下一次请求
            self._set_tool_event_sink(None)
            for thread in threads:
                thread.join()

        # ⑪ 汇总层：把 N 份任务总结拼成一份结构化报告（同步阻塞，期间不产生任何事件）
        report = self.reporting.generate_report(state)
        final_step = len(state.todo_items) + 1
        for event in self._drain_tool_events(state, step=final_step):
            yield event
        state.structured_report = report
        state.running_summary = report

        # 报告落盘为 note（note_type=conclusion）；成功则补一条 report_note 事件
        note_event = self._persist_final_report(state, report)
        if note_event:
            yield note_event

        # 最后一帧报告正文；done 是前端用来关闭流的终止信号
        yield {
            "type": "final_report",
            "report": report,
            "note_id": state.report_note_id,
            "note_path": state.report_note_path,
        }
        yield {"type": "done"}

    # ------------------------------------------------------------------
    # Execution helpers
    # ------------------------------------------------------------------
    def _execute_task(
        self,
        state: SummaryState,
        task: TodoItem,
        *,
        emit_stream: bool,
        step: int | None = None,
    ) -> Iterator[dict[str, Any]]:
        """Run search + summarization for a single task.

        单任务流水线（④ 执行层，每个 worker 线程跑一份）：

            1) task.status = "in_progress"
            2) search.dispatch_search()          检索，拿 (payload, notices, answer, backend)
            3) notices 逐条发 status 事件         例如「Tavily 未配置」这类提示
            4) 结果为空 → task.status="skipped"，发事件后直接 return
            5) search.prepare_research_context() 按 URL 去重 + 截断，拼出 sources 与 context
            6) 加锁把 context / sources 累加进 state，并让 research_loop_count += 1
            7) 流式：发 sources → stream_task_summary 逐 chunk 发 task_summary_chunk → completed
               非流式：summarize_task 一次性拿结果，不发任何事件
            8) task.summary 赋值、task.status = "completed"

        整个方法是生成器：emit_stream=False 时也必须被显式迭代，否则一行都不会执行。
        """

        task.status = "in_progress"

        search_result, notices, answer_text, backend = dispatch_search(
            task.query,
            self.config,
            state.research_loop_count,
        )
        self._last_search_notices = notices
        task.notices = notices

        if emit_stream:
            for event in self._drain_tool_events(state, step=step):
                yield event
        else:
            self._drain_tool_events(state)

        if notices and emit_stream:
            for notice in notices:
                if notice:
                    yield {
                        "type": "status",
                        "message": notice,
                        "task_id": task.id,
                        "step": step,
                    }

        if not search_result or not search_result.get("results"):
            task.status = "skipped"
            if emit_stream:
                for event in self._drain_tool_events(state, step=step):
                    yield event
                yield {
                    "type": "task_status",
                    "task_id": task.id,
                    "status": "skipped",
                    "title": task.title,
                    "intent": task.intent,
                    "note_id": task.note_id,
                    "note_path": task.note_path,
                    "step": step,
                }
            else:
                self._drain_tool_events(state)
            return
        else:
            if not emit_stream:
                self._drain_tool_events(state)

        sources_summary, context = prepare_research_context(
            search_result,
            answer_text,
            self.config,
        )

        task.sources_summary = sources_summary

        with self._state_lock:
            state.web_research_results.append(context)
            state.sources_gathered.append(sources_summary)
            state.research_loop_count += 1

        summary_text: str | None = None

        if emit_stream:
            for event in self._drain_tool_events(state, step=step):
                yield event
            yield {
                "type": "sources",
                "task_id": task.id,
                "latest_sources": sources_summary,
                "raw_context": context,
                "step": step,
                "backend": backend,
                "note_id": task.note_id,
                "note_path": task.note_path,
            }

            summary_stream, summary_getter = self.summarizer.stream_task_summary(state, task, context)
            try:
                for event in self._drain_tool_events(state, step=step):
                    yield event
                for chunk in summary_stream:
                    if chunk:
                        yield {
                            "type": "task_summary_chunk",
                            "task_id": task.id,
                            "content": chunk,
                            "note_id": task.note_id,
                            "step": step,
                        }
                    for event in self._drain_tool_events(state, step=step):
                        yield event
            finally:
                summary_text = summary_getter()
        else:
            summary_text = self.summarizer.summarize_task(state, task, context)
            self._drain_tool_events(state)

        task.summary = summary_text.strip() if summary_text else "暂无可用信息"
        task.status = "completed"

        if emit_stream:
            for event in self._drain_tool_events(state, step=step):
                yield event
            yield {
                "type": "task_status",
                "task_id": task.id,
                "status": "completed",
                "summary": task.summary,
                "sources_summary": task.sources_summary,
                "note_id": task.note_id,
                "note_path": task.note_path,
                "step": step,
            }
        else:
            self._drain_tool_events(state)

    def _drain_tool_events(
        self,
        state: SummaryState,
        *,
        step: int | None = None,
    ) -> list[dict[str, Any]]:
        """取出尚未消费的工具调用事件（双通道开关在这里生效）。

        流式期间（sink 已打开）：工具事件已由 worker 线程实时入队，这里直接返回 []
        以**避免同一条事件被推送两次**；但 drain() 仍会被调用，因为它还负责把
        事件里的 note_id 回填到对应的 TodoItem。

        非流式期间（sink 未打开）：返回自上次调用以来的增量事件，由调用方 yield 出去。
        """

        events = self._tool_tracker.drain(state, step=step)
        if self._tool_event_sink_enabled:
            return []
        return events

    @property
    def _tool_call_events(self) -> list[dict[str, Any]]:
        """Expose recorded tool events for legacy integrations."""
        return self._tool_tracker.as_dicts()

    def _serialize_task(self, task: TodoItem) -> dict[str, Any]:
        """Convert task dataclass to serializable dict for frontend."""
        return {
            "id": task.id,
            "title": task.title,
            "intent": task.intent,
            "query": task.query,
            "status": task.status,
            "summary": task.summary,
            "sources_summary": task.sources_summary,
            "note_id": task.note_id,
            "note_path": task.note_path,
            "stream_token": task.stream_token,
        }

    def _persist_final_report(self, state: SummaryState, report: str) -> dict[str, Any] | None:
        """把最终报告写进 NoteTool，并返回一条 report_note 事件（失败则 None）。

        流程：先找历史同名报告笔记 → 有就 update → 失败或没有就 create →
        从返回文本里正则抠出 note_id → 回写 state.report_note_id / note_path。

        [本地修复] 上游让「报告 Agent 自己调 note 工具保存」，但那会污染 run() 的返回值
        （模型先写正文、再调工具、最后补一句交付摘要，正文丢失）。现在改为**代码侧落盘**，
        报告 Agent 只负责输出正文。
        """

        if not self.note_tool or not report or not report.strip():
            return None

        note_title = f"研究报告：{state.research_topic}".strip() or "研究报告"
        tags = ["deep_research", "report"]
        content = report.strip()

        note_id = self._find_existing_report_note_id(state)
        response = ""

        if note_id:
            response = self.note_tool.run(
                {
                    "action": "update",
                    "note_id": note_id,
                    "title": note_title,
                    "note_type": "conclusion",
                    "tags": tags,
                    "content": content,
                }
            )
            if response.startswith("❌"):
                note_id = None

        if not note_id:
            response = self.note_tool.run(
                {
                    "action": "create",
                    "title": note_title,
                    "note_type": "conclusion",
                    "tags": tags,
                    "content": content,
                }
            )
            note_id = self._extract_note_id_from_text(response)

        if not note_id:
            return None

        state.report_note_id = note_id
        if self.config.notes_workspace:
            note_path = Path(self.config.notes_workspace) / f"{note_id}.md"
            state.report_note_path = str(note_path)
        else:
            note_path = None

        payload = {
            "type": "report_note",
            "note_id": note_id,
            "title": note_title,
            "content": content,
        }
        if note_path:
            payload["note_path"] = str(note_path)

        return payload

    def _find_existing_report_note_id(self, state: SummaryState) -> str | None:
        if state.report_note_id:
            return state.report_note_id

        for event in reversed(self._tool_tracker.as_dicts()):
            if event.get("tool") != "note":
                continue

            parameters = event.get("parsed_parameters") or {}
            if not isinstance(parameters, dict):
                continue

            action = parameters.get("action")
            if action not in {"create", "update"}:
                continue

            note_type = parameters.get("note_type")
            if note_type != "conclusion":
                title = parameters.get("title")
                if not (isinstance(title, str) and title.startswith("研究报告")):
                    continue

            note_id = parameters.get("note_id")
            if not note_id:
                note_id = self._tool_tracker._extract_note_id(event.get("result", ""))  # type: ignore[attr-defined]

            if note_id:
                return note_id

        return None

    @staticmethod
    def _extract_note_id_from_text(response: str) -> str | None:
        if not response:
            return None

        match = re.search(r"ID:\s*([^\n]+)", response)
        if not match:
            return None

        return match.group(1).strip()


def run_deep_research(topic: str, config: Configuration | None = None) -> SummaryStateOutput:
    """Convenience function mirroring the class-based API."""
    agent = DeepResearchAgent(config=config)
    return agent.run(topic)
