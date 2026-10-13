"""生成文本的清洗工具：把模型回复里不该给用户看的内容剔掉。

目前只有 strip_tool_calls()。它与 utils.strip_thinking_tokens() 是配对使用的两步：
先剥 <think> 思考块，再剥 [TOOL_CALL:...] 工具调用指令，剩下的才是给用户看的正文。
"""

from __future__ import annotations

import re


def strip_tool_calls(text: str) -> str:
    """删除文本里所有 [TOOL_CALL:...] 指令。

    正则用 ``[^\\]]+``（遇到第一个 ] 就停），因此参数里若含右方括号会被截断——
    实际场景中 note 工具的参数是 JSON，不含 `]`，所以够用。
    """

    if not text:
        return text

    pattern = re.compile(r"\[TOOL_CALL:[^\]]+\]")
    return pattern.sub("", text)

