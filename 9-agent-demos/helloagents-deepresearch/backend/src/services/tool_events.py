"""工具调用事件的旁路收集器：框架回调 → 事件记录 → SSE 载荷。

要解决的问题：模型对 note 工具的调用写在 LLM 响应文本里（形如 `[TOOL_CALL:note:{...}]`）。
框架执行完工具后，会通过 tool_call_listener 回调出来。我们既要把它展示在前端，
又要把 note_id 关联回对应的 TodoItem，于是有了这个 Tracker。

两种消费方式（二选一，由 agent._set_tool_event_sink() 切换）：
    实时模式 —— 已注册 event_sink：record() 里立刻回调，事件即时进 SSE 队列（流式用）
    批量模式 —— 没有 sink：事件先攒在 _events 里，由 drain() 按游标增量取走（非流式用）

并发：record() 由多个 worker 子线程调用，drain() 由主生成器调用，全部走 Lock 保护。
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Any, Callable, Optional

from models import SummaryState, TodoItem

logger = logging.getLogger(__name__)


@dataclass
class ToolCallEvent:
    """一条工具调用事件的内部表示（在 record() 里构造）。"""

    id: int
    agent: str
    tool: str
    raw_parameters: str
    parsed_parameters: dict[str, Any]
    result: str
    task_id: Optional[int]
    note_id: Optional[str]


class ToolCallTracker:
    """收集工具调用事件并转换成前端可消费的 SSE 载荷。

    内部用 _events 列表 + _cursor 游标实现「只取增量」的 drain 语义：
    流式期间游标几乎不动（事件靠 sink 直推），非流式期间靠它一次次把列表搬空。
    """

    def __init__(self, notes_workspace: Optional[str]) -> None:
        self._notes_workspace = notes_workspace
        self._events: list[ToolCallEvent] = []
        self._cursor = 0
        self._lock = Lock()
        self._event_sink: Optional[Callable[[dict[str, Any]], None]] = None

    def record(self, payload: dict[str, Any]) -> None:
        """记录一次工具调用（由框架的 tool_call_listener 在 worker 子线程中回调）。

        五步：
        1) 从 payload 取 agent_name / tool_name / 原始参数 / 解析后参数 / 执行结果；
        2) _infer_task_id() 反推这条调用属于哪个任务；
        3) 若是 note 工具，取出 note_id（参数里没有就从结果文本里正则抠）；
        4) 加锁追加到 _events 并打日志；
        5) 若注册了 event_sink，立刻把 SSE 载荷推出去（实时模式）。
        """

        agent_name = str(payload.get("agent_name") or "unknown")
        tool_name = str(payload.get("tool_name") or "unknown")
        raw_parameters = str(payload.get("raw_parameters") or "")
        parsed_parameters = payload.get("parsed_parameters") or {}
        result_text = str(payload.get("result") or "")

        if not isinstance(parsed_parameters, dict):
            parsed_parameters = {}

        task_id = self._infer_task_id(parsed_parameters)
        note_id: Optional[str] = None

        if tool_name == "note":
            note_id = parsed_parameters.get("note_id")
            if note_id is None:
                note_id = self._extract_note_id(result_text)

        event = ToolCallEvent(
            id=len(self._events) + 1,
            agent=agent_name,
            tool=tool_name,
            raw_parameters=raw_parameters,
            parsed_parameters=parsed_parameters,
            result=result_text,
            task_id=task_id,
            note_id=note_id,
        )

        with self._lock:
            self._events.append(event)

        logger.info(
            "Tool call recorded: agent=%s tool=%s task_id=%s note_id=%s parsed_parameters=%s",
            agent_name,
            tool_name,
            task_id,
            note_id,
            parsed_parameters,
        )

        sink = self._event_sink
        if sink:
            sink(self._build_payload(event, step=None))

    # ------------------------------------------------------------------
    # Draining helpers
    # ------------------------------------------------------------------
    def drain(self, state: SummaryState, *, step: Optional[int] = None) -> list[dict[str, Any]]:
        """取走尚未消费的事件，并顺手把 note_id / note_path 回填到对应 TodoItem。

        返回值是自上次 drain 以来的**增量**事件（靠 _cursor 保证不重复）。
        注意：即使调用方打算丢弃返回值（流式期间就是这样），也**仍然需要调用它** ——
        因为 task ↔ note_id 的关联就是在这里建立的。
        """

        with self._lock:
            if self._cursor >= len(self._events):
                return []
            new_events = self._events[self._cursor :]
            self._cursor = len(self._events)

        if state.todo_items:
            for event in new_events:
                task_id = event.task_id
                note_id = event.note_id
                if task_id is None or not note_id:
                    continue
                self._attach_note_to_task(state.todo_items, task_id, note_id)

        payloads: list[dict[str, Any]] = []
        for event in new_events:
            payload = self._build_payload(event, step=step)
            payloads.append(payload)

        return payloads

    def reset(self) -> None:
        """清空已记录的事件并重置游标（多轮研究之间复用同一个实例时用）。"""

        with self._lock:
            self._events.clear()
            self._cursor = 0

    def as_dicts(self) -> list[dict[str, Any]]:
        """导出全部事件的原始快照（加锁），供 _find_existing_report_note_id 等反查使用。"""

        with self._lock:
            return [
                {
                    "id": event.id,
                    "agent": event.agent,
                    "tool": event.tool,
                    "raw_parameters": event.raw_parameters,
                    "parsed_parameters": event.parsed_parameters,
                    "result": event.result,
                    "task_id": event.task_id,
                    "note_id": event.note_id,
                }
                for event in self._events
            ]

    def set_event_sink(self, sink: Optional[Callable[[dict[str, Any]], None]]) -> None:
        """注册/注销实时回调（传 None 即切回批量模式）。"""

        self._event_sink = sink

    def _build_payload(self, event: ToolCallEvent, step: Optional[int]) -> dict[str, Any]:
        """把内部事件转成前端消费的 dict（type=tool_call，note 工具额外带 note_path）。"""

        payload = {
            "type": "tool_call",
            "event_id": event.id,
            "agent": event.agent,
            "tool": event.tool,
            "parameters": event.parsed_parameters,
            "result": event.result,
            "task_id": event.task_id,
            "note_id": event.note_id,
        }
        if event.note_id and self._notes_workspace:
            note_path = Path(self._notes_workspace) / f"{event.note_id}.md"
            payload["note_path"] = str(note_path)
        if step is not None:
            payload["step"] = step
        return payload

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _attach_note_to_task(self, tasks: list[TodoItem], task_id: int, note_id: str) -> None:
        """把 note_id / note_path 写到 id 匹配的 TodoItem 上（找不到就静默跳过）。"""

        for task in tasks:
            if task.id != task_id:
                continue

            if task.note_id != note_id:
                task.note_id = note_id
                if self._notes_workspace:
                    task.note_path = str(Path(self._notes_workspace) / f"{note_id}.md")
            elif task.note_path is None and self._notes_workspace:
                task.note_path = str(Path(self._notes_workspace) / f"{note_id}.md")
            break

    def _infer_task_id(self, parameters: dict[str, Any]) -> Optional[int]:
        """从工具参数反推任务 id，三条线索依次尝试：

        1) 参数里直接有 task_id；
        2) tags 里含形如 task_3 的标签；
        3) title 里含「任务 3」这样的前缀。
        都识别不出就返回 None —— 事件仍会推给前端，只是归不到某张任务卡上。
        """

        if not parameters:
            return None

        if "task_id" in parameters:
            try:
                return int(parameters["task_id"])
            except (TypeError, ValueError):
                pass

        tags = parameters.get("tags")
        if isinstance(tags, list):
            for tag in tags:
                match = re.search(r"task_(\d+)", str(tag))
                if match:
                    return int(match.group(1))

        title = parameters.get("title")
        if isinstance(title, str):
            match = re.search(r"任务\s*(\d+)", title)
            if match:
                return int(match.group(1))

        return None

    def _extract_note_id(self, response: str) -> Optional[str]:
        """从 note 工具的执行结果文本里抠出 note_id（结果形如 "... ID: xxx"）。"""

        if not response:
            return None

        match = re.search(r"ID:\s*([^\n]+)", response)
        if match:
            return match.group(1).strip()
        return None
