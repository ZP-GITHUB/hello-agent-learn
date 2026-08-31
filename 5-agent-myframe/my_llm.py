# my_llm.py
import os
from typing import Optional, Iterator
from openai import OpenAI
from hello_agents import HelloAgentsLLM, HelloAgentsException


class MyLLM(HelloAgentsLLM):
    def __init__(
            self,
            model: Optional[str] = None,
            api_key: Optional[str] = None,
            base_url: Optional[str] = None,
            provider: Optional[str] = "auto",
            **kwargs
    ):
        # 检查provider是否为我们想处理的'minimax'
        if provider == "minimax":
            print("正在使用自定义的 Minimax Provider")
            self.provider = "minimax"

            # 解析 Minimax 的凭证
            self.api_key = api_key or os.getenv("MINIMAX_API_KEY")
            self.base_url = base_url or "https://api.minimaxi.com/v1"

            # 验证凭证是否存在
            if not self.api_key:
                raise ValueError("Minimax API key not found. Please set MINIMAX_API_KEY environment variable.")

            # 设置默认模型和其他参数（优先使用 Minimax 专属环境变量，避免与其他 provider 的 LLM_MODEL_ID 冲突）
            self.model = model or os.getenv("MINIMAX_MODEL_ID") or "MiniMax-M2.5"
            self.temperature = kwargs.get('temperature', 0.7)
            self.max_tokens = kwargs.get('max_tokens')
            self.timeout = kwargs.get('timeout', 60)

            # 使用获取的参数创建OpenAI客户端实例
            self._client = OpenAI(api_key=self.api_key, base_url=self.base_url, timeout=self.timeout)

        else:
            # 如果不是 minimax, 则完全使用父类的原始逻辑来处理
            super().__init__(model=model, api_key=api_key, base_url=base_url, provider=provider, **kwargs)

    def think(self, messages: list, temperature: Optional[float] = None) -> Iterator[str]:
        """
        重写父类的 think 方法，修复流式响应中的 IndexError 问题。

        问题背景：
          OpenAI 兼容 API 的流式响应最后一个数据块通常只携带 usage 统计信息，
          此时 chunk.choices 是空列表 []。
          父类直接访问 chunk.choices[0] 会抛出 IndexError: list index out of range。

        修复方式：
          在访问 choices[0] 前先判断 choices 是否为空，为空则跳过该数据块。

        Args:
            messages: 消息列表
            temperature: 温度参数，如果未提供则使用初始化时的值

        Yields:
            str: 流式响应的文本片段
        """
        print(f"🧠 正在调用 {self.model} 模型...")
        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature if temperature is not None else self.temperature,
                max_tokens=self.max_tokens,
                stream=True,
            )

            print("✅ 大语言模型响应成功:")
            for chunk in response:
                # 关键修复：跳过 choices 为空的数据块（通常是流末尾的 usage 统计块）
                if not chunk.choices:
                    continue
                content = chunk.choices[0].delta.content or ""
                if content:
                    print(content, end="", flush=True)
                    yield content
            print()  # 在流式输出结束后换行

        except Exception as e:
            print(f"❌ 调用LLM API时发生错误: {e}")
            raise HelloAgentsException(f"LLM调用失败: {str(e)}")