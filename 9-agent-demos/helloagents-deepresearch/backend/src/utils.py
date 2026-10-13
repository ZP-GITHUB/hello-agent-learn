"""通用工具：文本清理 + 检索结果格式化 + 配置取值。

调用关系：
    strip_thinking_tokens()            ← planner / summarizer / reporter 三处复用
    deduplicate_and_format_sources()   ← 只在 services/search.py 里用（拼总结用的长上下文）
    format_sources()                   ← 只在 services/search.py 里用（任务卡的「来源概览」）
    get_config_value()                 ← 只在 services/search.py 里用（枚举转字符串）
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Union

CHARS_PER_TOKEN = 4

logger = logging.getLogger(__name__)


def get_config_value(value: Any) -> str:
    """把配置项统一成字符串：SearchAPI 枚举取 .value，str 原样返回。"""

    return value if isinstance(value, str) else value.value


def strip_thinking_tokens(text: str) -> str:
    """循环剥离模型回复里的 <think>...</think> 思考块。

    用 while 而不是单次替换：MiniMax-M3 这类模型一次可能输出多段思考块。
    只删**成对**的标签，落单的 <think> 保持原样，避免误删正文。
    """

    while "<think>" in text and "</think>" in text:
        start = text.find("<think>")
        end = text.find("</think>") + len("</think>")
        text = text[:start] + text[end:]
    return text


def deduplicate_and_format_sources(
    search_response: Dict[str, Any] | List[Dict[str, Any]],
    max_tokens_per_source: int,
    *,
    fetch_full_page: bool = False,
) -> str:
    """把检索结果去重后拼成给总结 Agent 看的上下文长文本。

    - 去重键是 url：后出现的相同 url 直接丢弃；
    - fetch_full_page=True 时额外拼上 raw_content 全文，并按
      max_tokens_per_source * CHARS_PER_TOKEN 截断（字符数粗估，不做真分词）；
    - 输出形如「信息来源 / URL / 信息内容」的段落序列。
    """

    if isinstance(search_response, dict):
        sources_list = search_response.get("results", [])
    else:
        sources_list = search_response

    unique_sources: dict[str, Dict[str, Any]] = {}
    for source in sources_list:
        url = source.get("url")
        if not url:
            continue
        if url not in unique_sources:
            unique_sources[url] = source

    formatted_parts: List[str] = []
    for source in unique_sources.values():
        title = source.get("title") or source.get("url", "")
        content = source.get("content", "")
        formatted_parts.append(f"信息来源: {title}\n\n")
        formatted_parts.append(f"URL: {source.get('url', '')}\n\n")
        formatted_parts.append(f"信息内容: {content}\n\n")

        if fetch_full_page:
            raw_content = source.get("raw_content")
            if raw_content is None:
                logger.debug("raw_content missing for %s", source.get("url", ""))
                raw_content = ""
            char_limit = max_tokens_per_source * CHARS_PER_TOKEN
            if len(raw_content) > char_limit:
                raw_content = f"{raw_content[:char_limit]}... [truncated]"
            formatted_parts.append(
                f"详细信息内容限制为 {max_tokens_per_source} 个 token: {raw_content}\n\n"
            )

    return "".join(formatted_parts).strip()


def format_sources(search_results: Dict[str, Any] | None) -> str:
    """生成简短的来源清单（`* 标题 : URL`），用于任务卡的「来源概览」与报告引用。"""

    if not search_results:
        return ""

    results = search_results.get("results", [])
    return "\n".join(
        f"* {item.get('title', item.get('url', ''))} : {item.get('url', '')}"
        for item in results
        if item.get("url")
    )
