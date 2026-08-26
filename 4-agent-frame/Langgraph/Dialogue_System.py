"""
智能搜索助手 - 基于 LangGraph + Tavily API 的真实搜索系统

本模块实现了一个基于 LangGraph 状态图（StateGraph）的智能搜索助手，
通过三阶段流水线完成用户查询的理解、搜索和回答生成：
    1. 理解用户需求 —— 利用 LLM 分析用户意图并提取最优搜索关键词
    2. 使用Tavily API真实搜索信息 —— 调用外部搜索 API 获取实时数据
    3. 生成基于搜索结果的回答 —— 综合搜索结果，由 LLM 生成结构化回答

核心依赖：
    - LangGraph：用于构建有状态的节点-边工作流图
    - LangChain：提供 LLM 调用封装和消息类型定义
    - Tavily API：提供真实互联网搜索能力
    - InMemorySaver：LangGraph 内置的内存检查点，支持多轮对话的会话隔离

使用方式：
    1. 在 .env 文件中配置 LLM_API_KEY、LLM_BASE_URL、LLM_MODEL_ID 和 TAVILY_API_KEY
    2. 运行本脚本即可启动交互式搜索助手
"""

import asyncio
from typing import TypedDict, Annotated
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import InMemorySaver
import os
from dotenv import load_dotenv
from tavily import TavilyClient

# ========== 环境变量加载 ==========
# 从 .env 文件中读取 API 密钥、模型 ID 等配置项
# 必须在初始化 LLM 和 Tavily 客户端之前调用
load_dotenv()


# ========== 状态定义 ==========
# SearchState 定义了工作流图中每个节点共享的状态结构
# 使用 TypedDict 确保类型安全，Annotated[list, add_messages] 使消息列表
# 在节点间传递时自动追加（而非覆盖），这是 LangGraph 的消息合并机制
class SearchState(TypedDict):
    messages: Annotated[list, add_messages]  # 对话消息列表，自动追加合并
    user_query: str       # 用户原始查询（经 LLM 理解后的总结）
    search_query: str     # 优化后的搜索关键词，用于调用 Tavily API
    search_results: str   # Tavily 搜索返回的原始结果文本
    final_answer: str     # 最终生成的回答内容
    step: str             # 当前工作流所处步骤标记（start -> understood -> searched/search_failed -> completed）


# ========== 模型与客户端初始化 ==========
# 使用 ChatOpenAI 通过 OpenAI 兼容接口调用 LLM
# 所有配置均从 .env 环境变量中读取，确保密钥不硬编码在代码中
llm = ChatOpenAI(
    model=os.getenv("LLM_MODEL_ID", "gpt-4o-mini"),      # 模型名称，默认 gpt-4o-mini
    api_key=os.getenv("LLM_API_KEY"),                     # API 密钥（必填）
    base_url=os.getenv("LLM_BASE_URL", "https://api.openai.com/v1"),  # API 基础地址
    temperature=0.7  # 温度参数：控制输出随机性，0.7 在创造性和准确性之间取得平衡
)

# 初始化 Tavily 搜索客户端，用于执行真实的互联网搜索
# Tavily 专为 AI Agent 设计，返回结构化的搜索结果
tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))


# ========== 节点函数定义 ==========
# 以下三个函数分别对应工作流图中的三个节点，
# 每个节点接收当前状态 SearchState，返回需要更新的字段字典，
# LangGraph 会自动将返回值合并到全局状态中。

def understand_query_node(state: SearchState) -> SearchState:
    """
    节点1：理解用户查询并生成搜索关键词

    工作流程：
        1. 从消息列表中逆序查找最后一条用户消息（HumanMessage）
        2. 构造提示词，让 LLM 同时完成「需求理解」和「搜索词提取」
        3. 解析 LLM 输出，提取 "搜索词：" 或 "搜索关键词：" 后的内容作为实际搜索词
        4. 返回更新后的状态，包括用户查询总结、搜索关键词和步骤标记

    设计意图：
        直接用用户原始输入搜索往往效果不佳（如口语化、含冗余信息），
        通过 LLM 预处理可以提取出更精准的搜索关键词，提高搜索命中率。

    Args:
        state: 当前工作流状态，必须包含 messages 字段

    Returns:
        包含 user_query、search_query、step、messages 更新项的字典
    """

    # 从消息列表中逆序查找最新的用户消息
    # 使用逆序遍历是因为最新消息在列表末尾，避免遍历整个列表
    user_message = ""
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage):
            user_message = msg.content
            break

    # 构造理解提示词：让 LLM 同时输出需求总结和搜索关键词
    # 使用 f-string 将用户消息嵌入提示词模板
    understand_prompt = f"""分析用户的查询："{user_message}"

请完成两个任务：
1. 简洁总结用户想要了解什么
2. 生成最适合搜索的关键词（中英文均可，要精准）

格式：
理解：[用户需求总结]
搜索词：[最佳搜索关键词]"""

    # 调用 LLM 进行查询理解，使用 SystemMessage 包裹提示词
    response = llm.invoke([SystemMessage(content=understand_prompt)])

    # ========== 解析 LLM 输出，提取搜索关键词 ==========
    # 优先尝试匹配 "搜索词：" 格式，其次匹配 "搜索关键词：" 格式
    # 如果都无法匹配，则降级使用用户原始消息作为搜索词（防御性编程）
    response_text = response.content
    search_query = user_message  # 默认值：使用原始查询作为兜底

    if "搜索词：" in response_text:
        # 按 "搜索词：" 分割，取后半部分并去除首尾空白
        search_query = response_text.split("搜索词：")[1].strip()
    elif "搜索关键词：" in response_text:
        # 兼容 LLM 可能输出 "搜索关键词：" 的情况
        search_query = response_text.split("搜索关键词：")[1].strip()

    # 返回状态更新字典，LangGraph 会将其合并到全局状态
    # messages 字段由于使用了 add_messages reducer，会自动追加而非覆盖
    return {
        "user_query": response.content,       # 保存 LLM 对用户需求的理解总结
        "search_query": search_query,          # 保存提取出的搜索关键词
        "step": "understood",                  # 标记当前步骤为「已理解」
        "messages": [AIMessage(content=f"我理解您的需求：{response.content}")]
    }


def tavily_search_node(state: SearchState) -> SearchState:
    """
    节点2：使用 Tavily API 进行真实互联网搜索

    工作流程：
        1. 从状态中获取上一步生成的搜索关键词
        2. 调用 Tavily API 执行搜索，获取结构化结果
        3. 优先提取 Tavily 的 AI 综合答案，再拼接前 3 条具体结果
        4. 如果搜索失败，返回错误信息并标记步骤为 search_failed

    设计意图：
        Tavily 返回的 response 包含两部分有价值内容：
        - answer：Tavily 自身对搜索结果的 AI 综合总结
        - results：具体的搜索结果列表（含标题、摘要、URL）
        优先展示 answer 可以提供更直接的信息，results 作为补充来源。

    Args:
        state: 当前工作流状态，必须包含 search_query 字段

    Returns:
        包含 search_results、step、messages 更新项的字典
    """

    search_query = state["search_query"]

    try:
        print(f"🔍 正在搜索: {search_query}")

        # 调用 Tavily 搜索 API
        # 参数说明：
        #   search_depth="basic"  - 搜索深度，basic 速度快，advanced 更全面但更慢
        #   include_answer=True   - 让 Tavily 返回 AI 综合答案
        #   include_raw_content=False - 不返回网页原始内容（减少 token 消耗）
        #   max_results=5         - 最多返回 5 条搜索结果
        response = tavily_client.search(
            query=search_query,
            search_depth="basic",
            include_answer=True,
            include_raw_content=False,
            max_results=5
        )

        # ========== 处理搜索结果 ==========
        search_results = ""

        # 优先使用 Tavily 的 AI 综合答案（如果有）
        if response.get("answer"):
            search_results = f"综合答案：\n{response['answer']}\n\n"

        # 拼接前 3 条具体搜索结果，包含标题、内容摘要和来源 URL
        if response.get("results"):
            search_results += "相关信息：\n"
            for i, result in enumerate(response["results"][:3], 1):
                title = result.get("title", "")    # 网页标题
                content = result.get("content", "")  # 内容摘要
                url = result.get("url", "")          # 来源链接
                search_results += f"{i}. {title}\n{content}\n来源：{url}\n\n"

        # 兜底处理：如果搜索结果完全为空
        if not search_results:
            search_results = "抱歉，没有找到相关信息。"

        # 返回搜索成功状态
        return {
            "search_results": search_results,  # 拼接后的搜索结果文本
            "step": "searched",                  # 标记步骤为「已搜索」
            "messages": [AIMessage(content=f"✅ 搜索完成！找到了相关信息，正在为您整理答案...")]
        }

    except Exception as e:
        # ========== 搜索异常处理 ==========
        # 捕获所有异常，避免程序崩溃，降级为基于 LLM 自身知识回答
        error_msg = f"搜索时发生错误: {str(e)}"
        print(f"❌ {error_msg}")

        return {
            "search_results": f"搜索失败：{error_msg}",
            "step": "search_failed",
            "messages": [AIMessage(content="❌ 搜索遇到问题，我将基于已有知识为您回答")]
        }


def generate_answer_node(state: SearchState) -> SearchState:
    """
    节点3：基于搜索结果生成最终答案

    工作流程：
        1. 检查上一步的搜索状态（step 字段）
        2. 如果搜索失败（search_failed），走降级路径：直接用 LLM 知识回答
        3. 如果搜索成功，将搜索结果和用户问题一起传给 LLM 生成结构化回答

    设计意图：
        采用「降级策略」保证即使搜索 API 不可用，用户也能得到回答，
        提升系统的鲁棒性和用户体验。

    Args:
        state: 当前工作流状态，包含 search_results、user_query、step 等字段

    Returns:
        包含 final_answer、step、messages 更新项的字典
    """

    # ========== 降级路径：搜索失败时基于 LLM 自身知识回答 ==========
    if state["step"] == "search_failed":
        # 构造降级提示词：明确告知 LLM 搜索不可用，要求其基于自身知识回答
        fallback_prompt = f"""搜索API暂时不可用，请基于您的知识回答用户的问题：

用户问题：{state['user_query']}

请提供一个有用的回答，并说明这是基于已有知识的回答。"""

        response = llm.invoke([SystemMessage(content=fallback_prompt)])

        return {
            "final_answer": response.content,
            "step": "completed",
            "messages": [AIMessage(content=response.content)]
        }

    # ========== 正常路径：基于搜索结果生成答案 ==========
    # 将用户问题和搜索结果一起传给 LLM，让其综合生成回答
    answer_prompt = f"""基于以下搜索结果为用户提供完整、准确的答案：

用户问题：{state['user_query']}

搜索结果：
{state['search_results']}

请要求：
1. 综合搜索结果，提供准确、有用的回答
2. 如果是技术问题，提供具体的解决方案或代码
3. 引用重要信息的来源
4. 回答要结构清晰、易于理解
5. 如果搜索结果不够完整，请说明并提供补充建议"""

    response = llm.invoke([SystemMessage(content=answer_prompt)])

    return {
        "final_answer": response.content,
        "step": "completed",
        "messages": [AIMessage(content=response.content)]
    }


# ========== 工作流构建 ==========
def create_search_assistant():
    """
    创建并编译智能搜索助手的工作流图

    工作流结构（线性流水线）：
        START -> understand -> search -> answer -> END

    各节点职责：
        - understand：理解用户查询，提取搜索关键词
        - search：调用 Tavily API 执行互联网搜索
        - answer：基于搜索结果生成最终回答

    Returns:
        编译后的 LangGraph CompiledGraph 对象，可通过 invoke/astream 调用
    """
    # 创建状态图，指定状态类型为 SearchState
    workflow = StateGraph(SearchState)

    # 添加三个处理节点，每个节点绑定对应的处理函数
    workflow.add_node("understand", understand_query_node)  # 理解节点
    workflow.add_node("search", tavily_search_node)          # 搜索节点
    workflow.add_node("answer", generate_answer_node)        # 回答节点

    # 设置边（线性流程）：定义节点间的执行顺序
    workflow.add_edge(START, "understand")      # 入口 -> 理解
    workflow.add_edge("understand", "search")   # 理解 -> 搜索
    workflow.add_edge("search", "answer")       # 搜索 -> 回答
    workflow.add_edge("answer", END)            # 回答 -> 结束

    # ========== 编译并添加记忆检查点 ==========
    # InMemorySaver 提供内存级会话持久化，通过 thread_id 隔离不同会话
    # 这使得同一个会话中的多轮对话可以共享上下文
    memory = InMemorySaver()
    app = workflow.compile(checkpointer=memory)

    return app


async def main():
    """
    主函数：运行智能搜索助手的交互式循环

    功能说明：
        1. 启动前检查 Tavily API 密钥是否已配置
        2. 创建搜索助手工作流实例
        3. 进入交互式循环，接收用户输入并流式展示各阶段的处理结果
        4. 每次查询使用独立的 thread_id 实现会话隔离

    退出方式：
        输入 quit / q / 退出 / exit 即可退出程序
    """

    # ========== 启动前检查 ==========
    # 确保 Tavily API 密钥已配置，否则搜索功能无法使用
    if not os.getenv("TAVILY_API_KEY"):
        print("❌ 错误：请在.env文件中配置TAVILY_API_KEY")
        return

    # 创建搜索助手工作流（编译后的 LangGraph 图）
    app = create_search_assistant()

    print("🔍 智能搜索助手启动！")
    print("我会使用Tavily API为您搜索最新、最准确的信息")
    print("支持各种问题：新闻、技术、知识问答等")
    print("(输入 'quit' 退出)\n")

    # 会话计数器，用于生成唯一的 thread_id
    # 每次新的用户查询都会分配新的 thread_id，实现会话隔离
    session_count = 0

    while True:
        user_input = input("🤔 您想了解什么: ").strip()

        # 检查退出命令（支持多种退出方式）
        if user_input.lower() in ['quit', 'q', '退出', 'exit']:
            print("感谢使用！再见！👋")
            break

        # 跳过空输入
        if not user_input:
            continue

        # 递增会话计数器，生成当前查询的唯一 thread_id
        session_count += 1
        config = {"configurable": {"thread_id": f"search-session-{session_count}"}}

        # ========== 构造初始状态 ==========
        # 所有字段初始化为空值，后续由各个节点逐步填充
        initial_state = {
            "messages": [HumanMessage(content=user_input)],  # 用户消息作为初始输入
            "user_query": "",       # 待 understand 节点填充
            "search_query": "",     # 待 understand 节点填充
            "search_results": "",   # 待 search 节点填充
            "final_answer": "",     # 待 answer 节点填充
            "step": "start"         # 初始步骤标记
        }

        try:
            print("\n" + "=" * 60)

            # ========== 流式执行工作流 ==========
            # astream 异步流式输出每个节点的执行结果
            # output 的结构为 {节点名: 该节点返回的状态更新字典}
            async for output in app.astream(initial_state, config=config):
                # 遍历每个节点的输出（每次迭代通常只有一个节点）
                for node_name, node_output in output.items():
                    # 检查节点输出中是否包含消息
                    if "messages" in node_output and node_output["messages"]:
                        # 取最后一条消息（即该节点最新生成的消息）
                        latest_message = node_output["messages"][-1]
                        if isinstance(latest_message, AIMessage):
                            # 根据节点名称展示不同阶段的输出
                            if node_name == "understand":
                                print(f"🧠 理解阶段: {latest_message.content}")
                            elif node_name == "search":
                                print(f"🔍 搜索阶段: {latest_message.content}")
                            elif node_name == "answer":
                                # 最终回答单独展示，加换行使输出更清晰
                                print(f"\n💡 最终回答:\n{latest_message.content}")

            print("\n" + "=" * 60 + "\n")

        except Exception as e:
            # 捕获工作流执行过程中的异常，提示用户重新输入
            print(f"❌ 发生错误: {e}")
            print("请重新输入您的问题。\n")


# ========== 程序入口 ==========
# 使用 asyncio.run 启动异步主函数
if __name__ == "__main__":
    asyncio.run(main())