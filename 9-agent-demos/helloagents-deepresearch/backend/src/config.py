"""配置层：把环境变量映射成强类型的 Configuration（唯一配置来源）。

【重要】load_dotenv() 放在**模块顶层**而不是 from_env() 里，且必须保持这样：
services/search.py 在 import 阶段就构造 SearchTool 并读取 TAVILY_API_KEY，
若等到 from_env() 才加载 .env，那时已经来不及，会一直报「TAVILY_API_KEY 未设置」。

取值优先级：请求体 overrides > 环境变量 > 字段默认值（见 from_env）。
"""

import os
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field

# [本地修复] 上游整个后端没有任何 load_dotenv() 调用，导致 backend/.env 完全不生效
# （LLM_API_KEY / SEARCH_API / TAVILY_API_KEY 全部读不到）。
# 必须放在**模块导入期**执行：services/search.py 在 import 阶段就构造 SearchTool 并读取
# TAVILY_API_KEY，若只在 from_env() 里加载就来不及，会一直报 "TAVILY_API_KEY 未设置"。
from dotenv import load_dotenv

load_dotenv()


class SearchAPI(Enum):
    PERPLEXITY = "perplexity"
    TAVILY = "tavily"
    DUCKDUCKGO = "duckduckgo"
    SEARXNG = "searxng"
    ADVANCED = "advanced"


class Configuration(BaseModel):
    """Configuration options for the deep research assistant."""

    max_web_research_loops: int = Field(
        default=3,
        title="Research Depth",
        description="Number of research iterations to perform",
    )
    local_llm: str = Field(
        default="llama3.2",
        title="Local Model Name",
        description="Name of the locally hosted LLM (Ollama/LMStudio)",
    )
    llm_provider: str = Field(
        default="ollama",
        title="LLM Provider",
        description="Provider identifier (ollama, lmstudio, or custom)",
    )
    search_api: SearchAPI = Field(
        default=SearchAPI.DUCKDUCKGO,
        title="Search API",
        description="Web search API to use",
    )
    enable_notes: bool = Field(
        default=True,
        title="Enable Notes",
        description="Whether to store task progress in NoteTool",
    )
    notes_workspace: str = Field(
        default="./notes",
        title="Notes Workspace",
        description="Directory for NoteTool to persist task notes",
    )
    fetch_full_page: bool = Field(
        default=True,
        title="Fetch Full Page",
        description="Include the full page content in the search results",
    )
    ollama_base_url: str = Field(
        default="http://localhost:11434",
        title="Ollama Base URL",
        description="Base URL for Ollama API (without /v1 suffix)",
    )
    lmstudio_base_url: str = Field(
        default="http://localhost:1234/v1",
        title="LMStudio Base URL",
        description="Base URL for LMStudio OpenAI-compatible API",
    )
    strip_thinking_tokens: bool = Field(
        default=True,
        title="Strip Thinking Tokens",
        description="Whether to strip <think> tokens from model responses",
    )
    use_tool_calling: bool = Field(
        default=False,
        title="Use Tool Calling",
        description="Use tool calling instead of JSON mode for structured output",
    )
    llm_api_key: Optional[str] = Field(
        default=None,
        title="LLM API Key",
        description="Optional API key when using custom OpenAI-compatible services",
    )
    llm_base_url: Optional[str] = Field(
        default=None,
        title="LLM Base URL",
        description="Optional base URL when using custom OpenAI-compatible services",
    )
    llm_model_id: Optional[str] = Field(
        default=None,
        title="LLM Model ID",
        description="Optional model identifier for custom OpenAI-compatible services",
    )

    @classmethod
    def from_env(cls, overrides: Optional[dict[str, Any]] = None) -> "Configuration":
        """从环境变量 + overrides 构造配置；overrides 优先级最高。

        取三轮：
        1) 按字段名大写（SEARCH_API / LLM_PROVIDER …）扫一遍 os.environ；
        2) 再按 env_aliases 里列的别名补一轮（用 setdefault，不覆盖第 1 轮的值）；
        3) 最后用 overrides 强制覆盖（即 POST body 里显式给的 search_api）。

        字符串到枚举/布尔的转换由 pydantic 完成，所以 env 里要写枚举的字面值
        （例如 SEARCH_API=tavily、ENABLE_NOTES=true）。
        """

        raw_values: dict[str, Any] = {}

        # Load values from environment variables based on field names
        for field_name in cls.model_fields.keys():
            env_key = field_name.upper()
            if env_key in os.environ:
                raw_values[field_name] = os.environ[env_key]

        # Additional mappings for explicit env names
        env_aliases = {
            "local_llm": os.getenv("LOCAL_LLM"),
            "llm_provider": os.getenv("LLM_PROVIDER"),
            "llm_api_key": os.getenv("LLM_API_KEY"),
            "llm_model_id": os.getenv("LLM_MODEL_ID"),
            "llm_base_url": os.getenv("LLM_BASE_URL"),
            "lmstudio_base_url": os.getenv("LMSTUDIO_BASE_URL"),
            "ollama_base_url": os.getenv("OLLAMA_BASE_URL"),
            "max_web_research_loops": os.getenv("MAX_WEB_RESEARCH_LOOPS"),
            "fetch_full_page": os.getenv("FETCH_FULL_PAGE"),
            "strip_thinking_tokens": os.getenv("STRIP_THINKING_TOKENS"),
            "use_tool_calling": os.getenv("USE_TOOL_CALLING"),
            "search_api": os.getenv("SEARCH_API"),
            "enable_notes": os.getenv("ENABLE_NOTES"),
            "notes_workspace": os.getenv("NOTES_WORKSPACE"),
        }

        for key, value in env_aliases.items():
            if value is not None:
                raw_values.setdefault(key, value)

        if overrides:
            for key, value in overrides.items():
                if value is not None:
                    raw_values[key] = value

        return cls(**raw_values)

    def sanitized_ollama_url(self) -> str:
        """给 Ollama 地址补上 OpenAI 客户端要求的 /v1 后缀（幂等，可重复调用）。"""

        base = self.ollama_base_url.rstrip("/")
        if not base.endswith("/v1"):
            base = f"{base}/v1"
        return base

    def resolved_model(self) -> Optional[str]:
        """返回实际使用的模型名：优先 llm_model_id（云服务），否则退回 local_llm（本地模型）。"""

        return self.llm_model_id or self.local_llm

