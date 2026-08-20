import os
from openai import OpenAI
from dotenv import load_dotenv
from typing import List, Dict

# 加载项目根目录下的 .env 文件，读取其中定义的环境变量（如 API 密钥、模型地址等）
load_dotenv()


class HelloAgentsLLM:
    """
    为本书 "Hello Agents" 定制的 LLM 客户端。
    它用于调用任何兼容 OpenAI 接口的服务（如 OpenAI、DeepSeek、本地 Ollama 等），
    并默认使用流式响应（streaming），实现类似"打字机"的逐字输出效果。
    """

    def __init__(self, model: str = None, apiKey: str = None, baseUrl: str = None, timeout: int = None):
        """
        初始化 LLM 客户端。
        参数优先级：传入的参数 > .env 文件中的环境变量。
        如果两者都未提供，则会抛出 ValueError 异常。
        """
        # 模型名称：优先使用传入参数，否则从环境变量 LLM_MODEL_ID 读取
        self.model = model or os.getenv("LLM_MODEL_ID")
        # API 密钥：用于身份验证，防止未授权访问
        apiKey = apiKey or os.getenv("LLM_API_KEY")
        # 服务地址：API 的 base URL，可以是 OpenAI 官方地址，也可以是第三方兼容服务的地址
        baseUrl = baseUrl or os.getenv("LLM_BASE_URL")
        # 超时时间（秒）：防止请求无限等待，默认 60 秒
        timeout = timeout or int(os.getenv("LLM_TIMEOUT", 60))

        # 校验三个必要参数是否都已提供，任一缺失则抛出异常
        if not all([self.model, apiKey, baseUrl]):
            raise ValueError("模型ID、API密钥和服务地址必须被提供或在.env文件中定义。")

        # 创建 OpenAI 客户端实例，后续所有 API 调用都通过这个 client 进行
        self.client = OpenAI(api_key=apiKey, base_url=baseUrl, timeout=timeout)

    def think(self, messages: List[Dict[str, str]], temperature: float = 0) -> str:
        """
        调用大语言模型进行"思考"，并返回其文本响应。
        
        参数:
            messages: 对话消息列表，每个消息是一个字典，包含 "role" 和 "content" 字段。
                     例如：[{"role": "user", "content": "你好"}]
            temperature: 控制输出的随机性，0 表示最确定性输出，值越大越随机。
        
        返回:
            模型生成的完整文本内容（str），如果出错则返回 None。
        """
        print(f"🧠 正在调用 {self.model} 模型...")
        try:
            # 调用 OpenAI 兼容的 chat completions 接口
            # stream=True 表示启用流式输出：模型不会一次性返回完整结果，
            # 而是返回一个迭代器（openai.Stream 对象），逐块（chunk）产出响应数据
            response = self.client.chat.completions.create(
                model=self.model,        # 指定使用的模型名称
                messages=messages,        # 对话历史消息
                temperature=temperature,  # 温度参数，控制创造性
                stream=True,              # 关键参数：启用流式响应
            )

            # ========== 处理流式响应 ==========
            # 此时 response 是一个 openai.Stream 对象（迭代器），
            # 它内部持有底层的 HTTP 响应（response.response），但对外提供的是解析后的 ChatCompletionChunk 对象
            print("✅ 大语言模型响应成功:")
            
            # 用一个列表收集所有文本片段，最后拼接成完整回答
            collected_content = []
            
            # 遍历流式响应的每一个数据块（chunk）
            # 每个 chunk 的类型是 ChatCompletionChunk，包含 model、choices、created 等字段
            for chunk in response:
                # 有些 chunk 可能不包含 choices（如流结束时的终止信号），直接跳过
                if not chunk.choices:
                    continue
                
                # 从 chunk 中提取文本内容：
                # - chunk.choices[0]: 取第一个候选回答（通常只有一个）
                # - .delta: 表示"增量"，即相比上一个 chunk 新增的内容
                # - .content: 新增的文本内容，可能为 None（用 or "" 兜底）
                content = chunk.choices[0].delta.content or ""
                
                # 实时打印到屏幕：
                # - end="": 不在末尾加换行符，让文字连续显示在同一行
                # - flush=True: 强制立即刷新输出缓冲区，实现"逐字出现"的打字机效果
                #   （如果不加 flush，Python 会缓冲输出，用户看不到实时效果）
                print(content, end="", flush=True)
                
                # 同时把这段文本存入列表，用于最后拼接完整回答
                collected_content.append(content)
            
            # 流式输出结束后，打印一个换行符，让终端光标跳到下一行
            print()
            
            # 将所有文本片段拼接成完整字符串，返回给调用者
            return "".join(collected_content)

        except Exception as e:
            # 捕获所有异常（如网络错误、API 密钥无效、模型不存在等），打印错误信息并返回 None
            print(f"❌ 调用LLM API时发生错误: {e}")
            return None


# ========== 客户端使用示例 ==========
# 当直接运行此脚本时（而非被其他模块导入时），执行以下测试代码
if __name__ == '__main__':
    try:
        # 创建 LLM 客户端实例，配置从 .env 文件自动加载
        llmClient = HelloAgentsLLM()

        # 构造对话消息列表
        # - system 消息：设定 AI 的角色和行为准则
        # - user 消息：用户的具体问题或指令
        exampleMessages = [
            {"role": "system", "content": "You are a helpful assistant that writes Python code."},
            {"role": "user", "content": "写一个快速排序算法"}
        ]

        print("--- 调用LLM ---")
        # 调用 think 方法，获取模型响应（流式输出会在控制台实时显示）
        responseText = llmClient.think(exampleMessages)
        
        # 如果响应成功（非 None），再次打印完整内容（此时是完整文本，非流式）
        if responseText:
            print("\n\n--- 完整模型响应 ---")
            print(responseText)

    except ValueError as e:
        # 捕获初始化时的参数缺失异常
        print(e)