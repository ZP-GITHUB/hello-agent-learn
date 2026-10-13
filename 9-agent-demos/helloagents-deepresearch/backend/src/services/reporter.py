"""⑤ 汇总层：把所有任务的总结合并成一份结构化研究报告。

在整条链路中的位置：run_stream() 的收尾阶段调用 generate_report()，
这是**同步阻塞**的单次 LLM 调用，期间不产生任何 SSE 事件——
前端会表现为「任务都 completed 了但页面卡住若干十秒」，属正常现象。
"""

from __future__ import annotations

from hello_agents import ToolAwareSimpleAgent

from models import SummaryState
from config import Configuration
from utils import strip_thinking_tokens
from services.text_processing import strip_tool_calls


class ReportingService:
    """报告生成服务：只负责产出正文，落盘由 agent 侧完成。"""

    def __init__(self, report_agent: ToolAwareSimpleAgent, config: Configuration) -> None:
        self._agent = report_agent
        self._config = config

    def generate_report(self, state: SummaryState) -> str:
        """基于已完成的任务生成最终报告正文，返回清洗后的 Markdown。

        步骤：
        1) 遍历 state.todo_items，把每个任务的意图 / 查询 / 状态 / 总结 / 来源拼成 tasks_block；
        2) 拼一个 note 索引（仅供来源追溯，明确告诉模型「不需要去读」）；
        3) agent.run(prompt) —— 该 Agent 是 use_tools=False 构造的，无法调用工具；
        4) strip_thinking_tokens + strip_tool_calls 清理，兜底一句失败提示。

        返回值就是报告正文本身；工具调用与落盘都不再经过模型。
        """

        tasks_block = []
        for task in state.todo_items:
            summary_block = task.summary or "暂无可用信息"
            sources_block = task.sources_summary or "暂无来源"
            tasks_block.append(
                f"### 任务 {task.id}: {task.title}\n"
                f"- 任务目标：{task.intent}\n"
                f"- 检索查询：{task.query}\n"
                f"- 执行状态：{task.status}\n"
                f"- 任务总结：\n{summary_block}\n"
                f"- 来源概览：\n{sources_block}\n"
            )

        note_references = []
        for task in state.todo_items:
            if task.note_id:
                note_references.append(
                    f"- 任务 {task.id}《{task.title}》：note_id={task.note_id}"
                )

        notes_section = "\n".join(note_references) if note_references else "- 暂无可用任务笔记"

        # [本地修复] 原提示词要求报告 Agent 先读笔记、写正文、再调 note 工具保存结论。
        # 由于 run() 只返回"最后一条不含工具调用的消息"，模型常先输出正文再调工具、最后
        # 补一句"报告已生成"的交付摘要 → 最终拿到的正文丢失（只剩几百字摘要）。
        # 现在：任务总结已在上面内联提供，直接要求输出报告正文、禁止任何工具调用；
        # 报告落盘由 DeepResearchAgent._persist_final_report 在代码侧完成。
        # （对应地，报告 Agent 已在 agent.py 中以 use_tools=False 构造，模型无法调用工具。）
        prompt = (
            f"研究主题：{state.research_topic}\n"
            f"任务概览：\n{''.join(tasks_block)}\n"
            f"任务笔记索引（仅用于来源追溯，无需读取）：\n{notes_section}\n"
            "请直接撰写并一次性输出完整的研究报告正文，严格遵循报告模板的分节结构。\n"
            "注意：你的回复本身就是最终报告，必须包含全部章节与细节；"
            "禁止输出任何 [TOOL_CALL:...] 指令，也禁止只回复「已完成」「报告已生成」之类的交付说明。"
        )

        response = self._agent.run(prompt)
        self._agent.clear_history()

        report_text = response.strip()
        if self._config.strip_thinking_tokens:
            report_text = strip_thinking_tokens(report_text)

        report_text = strip_tool_calls(report_text).strip()

        return report_text or "报告生成失败，请检查输入。"

