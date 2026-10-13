"""③ 规划层：把研究主题拆解为一组可并行执行的待办任务（TodoItem）。

在整条链路中的位置：
    agent.run_stream() 第 ③ 步调用 plan_todo_list()，产出 3~5 个 TodoItem；
    之后 run_stream() 会为**每个 TodoItem 起一个线程**（agent.py 第 ⑨ 步），
    所以这里拆出几个任务，后面就是几路并发。

一次规划 = 一次 LLM 调用（框架内部最多再补 3 轮工具调用）。
模型被要求回严格 JSON，但实际输出常夹杂 <think> 思考块与 [TOOL_CALL:note:...] 指令，
因此 _extract_tasks() 做了三级容错解析（见各方法注释）。
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, List, Optional

from hello_agents import ToolAwareSimpleAgent

from models import SummaryState, TodoItem
from config import Configuration
from prompts import get_current_date, todo_planner_instructions
from utils import strip_thinking_tokens

logger = logging.getLogger(__name__)

TOOL_CALL_PATTERN = re.compile(
    r"\[TOOL_CALL:(?P<tool>[^:]+):(?P<body>[^\]]+)\]",
    re.IGNORECASE,
)

class PlanningService:
    """包装规划 Agent：输入研究主题，输出结构化 TodoItem 列表。"""

    def __init__(self, planner_agent: ToolAwareSimpleAgent, config: Configuration) -> None:
        self._agent = planner_agent
        self._config = config

    def plan_todo_list(self, state: SummaryState) -> List[TodoItem]:
        """调规划 Agent 拆任务，返回 TodoItem 列表并同步写回 state.todo_items。

        步骤：
        1) 用 todo_planner_instructions 填入当前日期与研究主题，拼出提示词；
        2) agent.run() —— 阻塞调用，框架侧最多 3 轮「LLM → 执行工具 → 回灌结果」；
        3) clear_history() —— 必须做，否则多次研究之间上下文会串味；
        4) _extract_tasks() 三级容错解析出任务字典；
        5) 逐条构造 TodoItem，id 从 1 开始按顺序编号（后续线程与事件都靠它定位）。

        解析失败时返回空列表，由调用方（run_stream / run）用兜底任务接手。
        """

        prompt = todo_planner_instructions.format(
            current_date=get_current_date(),
            research_topic=state.research_topic,
        )

        response = self._agent.run(prompt)
        self._agent.clear_history()

        logger.info("Planner raw output (truncated): %s", response[:500])

        tasks_payload = self._extract_tasks(response)
        todo_items: List[TodoItem] = []

        for idx, item in enumerate(tasks_payload, start=1):
            title = str(item.get("title") or f"任务{idx}").strip()
            intent = str(item.get("intent") or "聚焦主题的关键问题").strip()
            query = str(item.get("query") or state.research_topic).strip()

            if not query:
                query = state.research_topic

            task = TodoItem(
                id=idx,
                title=title,
                intent=intent,
                query=query,
            )
            todo_items.append(task)

        state.todo_items = todo_items

        titles = [task.title for task in todo_items]
        logger.info("Planner produced %d tasks: %s", len(todo_items), titles)
        return todo_items

    @staticmethod
    def create_fallback_task(state: SummaryState) -> TodoItem:
        """兜底任务：规划失败或返回空数组时使用，保证流程不会空转。

        固定 id=1、标题「基础背景梳理」，查询词为主题 + " 最新进展"。
        """

        return TodoItem(
            id=1,
            title="基础背景梳理",
            intent="收集主题的核心背景与最新动态",
            query=f"{state.research_topic} 最新进展" if state.research_topic else "基础背景梳理",
        )

    # ------------------------------------------------------------------
    # Parsing helpers
    # ------------------------------------------------------------------
    def _extract_tasks(self, raw_response: str) -> List[dict[str, Any]]:
        """三级容错解析：剥思考块 → 找 JSON → 再兜底找 TOOL_CALL 里的负载。

        第 1 级 _extract_json_payload()：正常的 {"tasks": [...]} 输出；
        第 2 级 _extract_tool_payload()：模型把任务塞进 [TOOL_CALL:note:{...}] 参数里的情况；
        都没有就返回空列表，触发上游的兜底任务。
        """

        text = raw_response.strip()
        if self._config.strip_thinking_tokens:
            text = strip_thinking_tokens(text)

        json_payload = self._extract_json_payload(text)
        tasks: List[dict[str, Any]] = []

        if isinstance(json_payload, dict):
            candidate = json_payload.get("tasks")
            if isinstance(candidate, list):
                for item in candidate:
                    if isinstance(item, dict):
                        tasks.append(item)
        elif isinstance(json_payload, list):
            for item in json_payload:
                if isinstance(item, dict):
                    tasks.append(item)

        if not tasks:
            tool_payload = self._extract_tool_payload(text)
            if tool_payload and isinstance(tool_payload.get("tasks"), list):
                for item in tool_payload["tasks"]:
                    if isinstance(item, dict):
                        tasks.append(item)

        return tasks

    def _extract_json_payload(self, text: str) -> Optional[dict[str, Any] | list]:
        """从模型输出里找出可解析的 JSON 对象/数组——规划结果的容错解析。

        优先级：带 tasks 列表的对象 → 元素为对象的数组 → 任意可解析的对象/数组。

        [本地修复] 原实现取「第一个 { 到最后一个 }」再 json.loads。当模型输出里夹带
        [TOOL_CALL:note:{"action":...}] 时，工具调用里的花括号会与后面的任务 JSON 连成
        一整片，json.loads 必然失败 → 静默回退成单任务（表现为"随机只剩 1 个任务"）。
        改为用 JSONDecoder.raw_decode 从每个 { / [ 位置逐个尝试解析，收集所有可解析的
        载荷，并优先返回带 "tasks" 的那个。
        """

        decoder = json.JSONDecoder()
        candidates: List[Any] = []
        cursor = 0
        length = len(text)

        while cursor < length:
            positions = [
                pos
                for pos in (text.find("{", cursor), text.find("[", cursor))
                if pos != -1
            ]
            if not positions:
                break

            start = min(positions)
            try:
                value, end = decoder.raw_decode(text, start)
            except json.JSONDecodeError:
                cursor = start + 1
                continue

            candidates.append(value)
            cursor = max(end, start + 1)

        # 优先：包含 tasks 列表的对象（规划器的标准输出）
        for value in candidates:
            if isinstance(value, dict) and isinstance(value.get("tasks"), list):
                return value

        # 其次：元素为对象的数组
        for value in candidates:
            if isinstance(value, list) and any(isinstance(item, dict) for item in value):
                return value

        # 兜底：任意可解析的对象/数组（保持与原实现相同的宽松度）
        for value in candidates:
            if isinstance(value, (dict, list)):
                return value

        return None

    def _extract_tool_payload(self, text: str) -> Optional[dict[str, Any]]:
        """解析输出里第一处 [TOOL_CALL:xxx:...] 的参数部分。

        先按 JSON 解析；失败则退回「k=v,k=v」的形式手工切分。
        用于兜住「模型没按 JSON 模板回复，而是把任务写进了工具调用参数」的情况。
        """

        match = TOOL_CALL_PATTERN.search(text)
        if not match:
            return None

        body = match.group("body")

        try:
            payload = json.loads(body)
            if isinstance(payload, dict):
                return payload
        except json.JSONDecodeError:
            pass

        parts = [segment.strip() for segment in body.split(",") if segment.strip()]
        payload: dict[str, Any] = {}
        for part in parts:
            if "=" not in part:
                continue
            key, value = part.split("=", 1)
            payload[key.strip()] = value.strip().strip('"').strip("'")

        return payload or None
