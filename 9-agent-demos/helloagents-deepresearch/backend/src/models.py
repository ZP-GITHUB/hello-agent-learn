"""状态数据模型：TodoItem（并行单元）+ SummaryState（全流程共享状态）+ 输入输出外壳。

数据怎样流动：
    planner.plan_todo_list()    → 往 state.todo_items 里塞 3~5 个 TodoItem
    每个 worker 线程            → 只改自己那一个 TodoItem（status / summary / sources_summary）
    tool_events.ToolCallTracker → drain() 时把 note_id / note_path 回填到对应 TodoItem
    reporter.generate_report()  → 遍历 state.todo_items 拼出最终报告
    SummaryStateOutput          → 非流式接口 POST /research 的返回值外壳

[并发提示] `Annotated[list, operator.add]` 是 LangGraph 风格的 reducer 标注，在普通
dataclass 上**不会自动生效**。本项目靠 agent._state_lock 手动加锁累加，别误以为它在自动合并。
"""

import operator
from dataclasses import dataclass, field
from typing import List, Optional

from typing_extensions import Annotated


@dataclass(kw_only=True)
class TodoItem:
    """单个待办任务项，同时是流式事件里 task_id / step / stream_token 的载体。

    id / title / intent / query —— 规划层产出，此后不再变
    status                      —— pending → in_progress → completed / skipped
    summary                     —— 总结 Agent 的产出，是最终报告的主要素材
    sources_summary             —— 本任务检索到的来源清单（`* 标题 : URL`）
    notices                     —— 检索后端返回的提示语，会转成 status 事件给前端
    note_id / note_path         —— 由 ToolCallTracker 从工具事件里反查回填
    stream_token                —— 形如 task_3，供前端把 chunk 归到正确的任务卡
    """

    id: int
    title: str
    intent: str
    query: str
    status: str = field(default="pending")
    summary: Optional[str] = field(default=None)
    sources_summary: Optional[str] = field(default=None)
    notices: list[str] = field(default_factory=list)
    note_id: Optional[str] = field(default=None)
    note_path: Optional[str] = field(default=None)
    stream_token: Optional[str] = field(default=None)


@dataclass(kw_only=True)
class SummaryState:
    """全流程共享的状态袋，在一次研究请求的整个生命周期里被反复读写。

    [并发提示] web_research_results / sources_gathered / research_loop_count 会被 N 个
    worker 线程同时写入，agent._execute_task() 用 self._state_lock 保护，勿移除该锁。
    """

    research_topic: str = field(default=None)  # Report topic
    search_query: str = field(default=None)  # Deprecated placeholder
    web_research_results: Annotated[list, operator.add] = field(default_factory=list)
    sources_gathered: Annotated[list, operator.add] = field(default_factory=list)
    research_loop_count: int = field(default=0)  # Research loop count
    running_summary: str = field(default=None)  # Legacy summary field
    todo_items: Annotated[list, operator.add] = field(default_factory=list)
    structured_report: Optional[str] = field(default=None)
    report_note_id: Optional[str] = field(default=None)
    report_note_path: Optional[str] = field(default=None)


@dataclass(kw_only=True)
class SummaryStateInput:
    """非流式流程的入参外壳（沿用上游命名，当前代码未直接实例化）。"""

    research_topic: str = field(default=None)  # Report topic


@dataclass(kw_only=True)
class SummaryStateOutput:
    """非流式接口的返回值外壳：报告正文 + 全部任务的最终状态。"""

    running_summary: str = field(default=None)  # Backward-compatible文本
    report_markdown: Optional[str] = field(default=None)
    todo_items: List[TodoItem] = field(default_factory=list)

