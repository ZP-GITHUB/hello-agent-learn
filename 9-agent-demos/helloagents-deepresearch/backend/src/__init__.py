"""HelloAgents Deep Research —— 基于 HelloAgents 的深度研究助手（包入口）。

模块地图：
    main.py        FastAPI 入口：/healthz、/research（非流式）、/research/stream（SSE）
    config.py      配置层：env → Configuration（导入期即 load_dotenv）
    models.py      数据模型：TodoItem / SummaryState / SummaryStateOutput
    agent.py       编排层：DeepResearchAgent.run() / run_stream() / _execute_task()
    prompts.py     三份提示词：规划、任务总结、报告撰写
    utils.py       文本清理与检索结果格式化
    services/      领域服务（规划 / 检索 / 总结 / 报告 / 笔记 / 工具事件收集）

注意：本项目使用**扁平导入**（`from utils import ...`），运行入口的当前工作目录必须是
`backend/src`，否则会 ModuleNotFoundError。
"""

__version__ = "0.0.1"

from .agent import DeepResearchAgent
from .config import Configuration, SearchAPI
from .models import SummaryState, SummaryStateInput, SummaryStateOutput, TodoItem

__all__ = [
    "DeepResearchAgent",
    "Configuration",
    "SearchAPI",
    "SummaryState",
    "SummaryStateInput",
    "SummaryStateOutput",
    "TodoItem",
]

