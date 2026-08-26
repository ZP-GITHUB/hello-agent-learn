"""
数字电子书协作撰写模块

基于 CAMEL 框架的 RolePlaying（角色扮演）机制，让两个 AI 智能体分别扮演"心理学家"和"作家"，
通过多轮自主对话协作完成一本关于"拖延症心理学"的短篇电子书。

核心流程：
1. 从 .env 加载 API 配置，通过 ModelFactory 创建 OpenAI 兼容的大模型实例
2. 定义协作任务提示词，明确电子书的主题、风格和要求
3. 初始化 RolePlaying 会话，让两个智能体进入各自角色
4. 进入循环对话，每轮由两个智能体交替发言，逐步推进任务
5. 检测到任务完成标志（CAMEL_TASK_DONE）或达到轮次上限时结束

CAMEL 框架的核心理念是 "角色扮演自主协作"：
- user_role（作家）负责提出需求和方向
- assistant_role（心理学家）负责提供专业内容和建议
- 两个智能体在无需人工干预的情况下自主推进任务，直到完成
"""

# ========== 依赖导入 ==========
from colorama import Fore                          # 终端彩色输出库，用于区分不同角色的发言
from camel.societies import RolePlaying            # CAMEL 核心类：角色扮演会话，管理两个智能体的协作对话
from camel.utils import print_text_animated        # CAMEL 工具函数：逐字动画打印，模拟打字效果
from camel.models import ModelFactory              # CAMEL 模型工厂：根据平台类型统一创建大模型实例
from camel.types import ModelPlatformType          # 模型平台类型枚举，指定使用哪种 API 协议
from dotenv import load_dotenv                     # 环境变量加载器，从 .env 文件读取敏感配置
import os

# ========== 配置加载 ==========
# 从当前目录的 .env 文件中加载 API 密钥、接口地址、模型名称等配置
# 这样避免将密钥硬编码在代码中，便于不同环境切换
load_dotenv()
LLM_API_KEY = os.getenv("LLM_API_KEY")             # API 访问密钥
LLM_BASE_URL = os.getenv("LLM_BASE_URL")           # API 基础地址（OpenAI 兼容格式）
LLM_MODEL_ID = os.getenv("LLM_MODEL_ID")           # 模型标识符（如 deepseek-v4-flash-0731）

# ========== 模型创建 ==========
# 使用 ModelFactory 工厂方法创建模型实例
# - OPENAI_COMPATIBLE_MODEL：表示目标 API 遵循 OpenAI 的请求/响应格式（兼容模式）
#   这样无论底层是 DeepSeek、Qwen 还是其他模型，只要接口兼容 OpenAI 格式即可通用
# - model_type：具体的模型名称，从 .env 读取，方便随时切换不同模型
# - url / api_key：连接到哪个 API 服务以及认证凭据
model = ModelFactory.create(
    model_platform=ModelPlatformType.OPENAI_COMPATIBLE_MODEL,
    model_type=LLM_MODEL_ID,
    url=LLM_BASE_URL,
    api_key=LLM_API_KEY
)

# ========== 协作任务定义 ==========
# task_prompt 是驱动两个智能体协作的核心指令
# 它会被注入到两个智能体的系统提示词中，引导它们围绕这个任务展开对话
# 任务描述越具体，智能体的输出质量和方向性越好
task_prompt = """
创作一本关于"拖延症心理学"的短篇电子书，目标读者是对心理学感兴趣的普通大众。
要求：
1. 内容科学严谨，基于实证研究
2. 语言通俗易懂，避免过多专业术语
3. 包含实用的改善建议和案例分析
4. 篇幅控制在8000-10000字
5. 结构清晰，包含引言、核心章节和总结
"""

print(Fore.YELLOW + f"协作任务:\n{task_prompt}\n")

# ========== 角色扮演会话初始化 ==========
# RolePlaying 是 CAMEL 的核心机制，它创建了一个双智能体协作环境：
# - assistant_role_name="心理学家"：助手智能体扮演心理学家，负责提供专业内容
# - user_role_name="作家"：用户智能体扮演作家，负责提出创作需求并整合内容
# - task_prompt：共享的协作任务描述，两个智能体都围绕此任务工作
# - model：两个智能体共用同一个大模型实例（也可以分别指定不同模型）
#
# 初始化后，CAMEL 会自动为两个智能体生成各自的角色系统提示词，
# 让它们"进入角色"并开始自主对话
role_play_session = RolePlaying(
    assistant_role_name="心理学家",
    user_role_name="作家",
    task_prompt=task_prompt,
    model=model
)

# 打印经过 CAMEL 细化后的实际任务描述（框架可能会在原始 task_prompt 基础上补充角色信息）
print(Fore.CYAN + f"具体任务描述:\n{role_play_session.task_prompt}\n")

# ========== 协作对话循环 ==========
# chat_turn_limit：最大对话轮次，防止智能体无限循环对话浪费 API 额度
# n：当前轮次计数器
# init_chat()：发起第一轮对话的初始消息，触发智能体开始协作
chat_turn_limit, n = 30, 0
input_msg = role_play_session.init_chat()

while n < chat_turn_limit:
    n += 1
    # step() 是每轮对话的核心调用：
    # 它会让 assistant（心理学家）和 user（作家）各生成一次回复
    # 返回两个响应对象，分别包含各自的消息内容
    assistant_response, user_response = role_play_session.step(input_msg)

    # 逐字动画打印两个角色的发言内容，用不同颜色区分：
    # - 蓝色：作家（user_role）的发言
    # - 绿色：心理学家（assistant_role）的发言
    print_text_animated(Fore.BLUE + f"作家:\n\n{user_response.msg.content}\n")
    print_text_animated(Fore.GREEN + f"心理学家:\n\n{assistant_response.msg.content}\n")

    # 检查任务完成标志：
    # CAMEL 框架约定，当智能体认为任务已完成时，会在消息中插入 "CAMEL_TASK_DONE" 标记
    # 检测到该标记后立即退出循环，避免多余的 API 调用
    if "CAMEL_TASK_DONE" in user_response.msg.content:
        print(Fore.MAGENTA + "✅ 电子书创作完成！")
        break

    # 将心理学家的回复作为下一轮对话的输入，推动协作继续
    input_msg = assistant_response.msg

# 打印最终的对话轮次统计
print(Fore.YELLOW + f"总共进行了 {n} 轮协作对话")