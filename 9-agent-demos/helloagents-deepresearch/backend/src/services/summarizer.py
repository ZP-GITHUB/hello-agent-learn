"""④ 执行层的总结模块：把一次检索的上下文压成任务总结。

两条路径共用同一份提示词（_build_prompt）：
    summarize_task()       —— 非流式，一次性拿结果，不发事件
    stream_task_summary()  —— 流式，返回 (生成器, 取值函数)，边生成边推 chunk

[设计要点] 每调用一次就通过 _agent_factory 新建一个 ToolAwareSimpleAgent，
用完立刻 clear_history()。目的是让任务之间彼此隔离，避免上下文互相污染。
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import Tuple

from hello_agents import ToolAwareSimpleAgent

from models import SummaryState, TodoItem
from config import Configuration
from utils import strip_thinking_tokens
from services.notes import build_note_guidance
from services.text_processing import strip_tool_calls


class SummarizationService:
    """任务总结服务：提供一次性总结与流式总结两种形态。"""

    def __init__(
        self,
        summarizer_factory: Callable[[], ToolAwareSimpleAgent],
        config: Configuration,
    ) -> None:
        self._agent_factory = summarizer_factory
        self._config = config

    def summarize_task(self, state: SummaryState, task: TodoItem, context: str) -> str:
        """非流式总结：阻塞拿到完整结果后统一清理并返回。

        清理顺序有意义：先 strip_thinking_tokens（剥 <think> 思考块），
        再 strip_tool_calls（剥 [TOOL_CALL:...] 指令），最后兜底「暂无可用信息」。
        """

        prompt = self._build_prompt(state, task, context)

        agent = self._agent_factory()
        try:
            response = agent.run(prompt)
        finally:
            agent.clear_history()

        summary_text = response.strip()
        if self._config.strip_thinking_tokens:
            summary_text = strip_thinking_tokens(summary_text)

        summary_text = strip_tool_calls(summary_text).strip()

        return summary_text or "暂无可用信息"

    def stream_task_summary(
        self, state: SummaryState, task: TodoItem, context: str
    ) -> Tuple[Iterator[str], Callable[[], str]]:
        """流式总结：返回 (生成器, 取值函数) 二元组。

        为什么要返回两个东西：生成器在 yield 的过程中逐段交给上游发 SSE，
        而「完整总结」只有等流走完才知道——所以额外给一个 get_summary() 回调，
        由调用方在 finally 里调用，把清洗后的全文写进 task.summary。
        """

        prompt = self._build_prompt(state, task, context)
        remove_thinking = self._config.strip_thinking_tokens
        raw_buffer = ""
        visible_output = ""
        emit_index = 0
        agent = self._agent_factory()

        def flush_visible() -> Iterator[str]:
            """把 raw_buffer 里「已确定不属于 <think> 块」的部分逐段吐出。

            这是流式剥离思考块的难点：标签可能跨 chunk 到来，所以用 emit_index
            游标**滞后**推进——只有 </think> 已出现、或确认后面不再可能是标签开头时，
            才把游标推过去，避免把半个 "<thi" 当正文发出去。
            """

            nonlocal emit_index, raw_buffer
            while True:
                start = raw_buffer.find("<think>", emit_index)
                if start == -1:
                    if emit_index < len(raw_buffer):
                        segment = raw_buffer[emit_index:]
                        emit_index = len(raw_buffer)
                        if segment:
                            yield segment
                    break

                if start > emit_index:
                    segment = raw_buffer[emit_index:start]
                    emit_index = start
                    if segment:
                        yield segment

                end = raw_buffer.find("</think>", start)
                if end == -1:
                    break
                emit_index = end + len("</think>")

        def generator() -> Iterator[str]:
            """主生成器：包住 agent.stream_run()，边收 chunk 边过滤思考块并 yield。"""

            nonlocal raw_buffer, visible_output, emit_index
            try:
                for chunk in agent.stream_run(prompt):
                    raw_buffer += chunk
                    if remove_thinking:
                        for segment in flush_visible():
                            visible_output += segment
                            if segment:
                                yield segment
                    else:
                        visible_output += chunk
                        if chunk:
                            yield chunk
            finally:
                if remove_thinking:
                    for segment in flush_visible():
                        visible_output += segment
                        if segment:
                            yield segment
                agent.clear_history()

        def get_summary() -> str:
            """惰性取值函数：流结束后才能拿到完整总结，故以回调形式交给调用方。"""

            if remove_thinking:
                cleaned = strip_thinking_tokens(visible_output)
            else:
                cleaned = visible_output

            return strip_tool_calls(cleaned).strip()

        return generator(), get_summary

    def _build_prompt(self, state: SummaryState, task: TodoItem, context: str) -> str:
        """拼任务总结的提示词（流式、非流式共用）。

        六段拼接：任务主题 / 任务名称 / 任务目标 / 检索查询 / 任务上下文（网页正文），
        再加上 build_note_guidance(task) 注入的 note 工具协作指令。
        最后一句要求模型「先同步笔记，再输出面向用户的 Markdown 总结」——
        这条指令是笔记机制能被触发的关键。
        """

        return (
            f"任务主题：{state.research_topic}\n"
            f"任务名称：{task.title}\n"
            f"任务目标：{task.intent}\n"
            f"检索查询：{task.query}\n"
            f"任务上下文：\n{context}\n"
            f"{build_note_guidance(task)}\n"
            "请按照以上协作要求先同步笔记，然后返回一份面向用户的 Markdown 总结（仍遵循任务总结模板）。"
        )
