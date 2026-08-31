# my_simple_agent.py
# 本文件演示如何基于 hello_agents 框架的 SimpleAgent 基类，
# 扩展实现一个支持「工具调用」能力的自定义 Agent。
#
# 核心思路：
#   1. 继承 SimpleAgent，复用其对话管理和历史消息维护能力
#   2. 引入 ToolRegistry（工具注册表），让 Agent 可以动态挂载/卸载工具
#   3. 通过在系统提示词中注入工具描述和调用格式约定，引导 LLM 输出特定标记
#   4. 解析 LLM 输出中的 [TOOL_CALL:工具名:参数] 标记，自动执行工具并将结果回传给 LLM
#   5. 支持多轮工具调用循环，直到 LLM 给出最终回答（无工具标记）为止

from typing import Optional, Iterator
# SimpleAgent: 框架提供的简单对话 Agent 基类，提供 run/stream_run/add_message/get_history 等方法
# HelloAgentsLLM: 框架的统一 LLM 客户端，支持多种 provider
# Config: 框架的配置类，控制 temperature、max_tokens 等参数
# Message: 框架的消息数据类，用于存储单条对话记录（role + content）
from hello_agents import SimpleAgent, HelloAgentsLLM, Config, Message
import re


class MySimpleAgent(SimpleAgent):
    """
    重写的简单对话Agent - 在原版 SimpleAgent 基础上增加了工具调用能力。

    与父类的区别：
      - 父类 SimpleAgent.run() 只做「一问一答」，LLM 输出什么就返回什么
      - 本类 MySimpleAgent.run() 增加了「工具调用循环」：
        如果 LLM 的输出中包含 [TOOL_CALL:xxx:yyy] 标记，
        会自动解析、执行工具、把结果喂回 LLM，直到 LLM 不再请求工具为止。

    展示如何基于框架基类构建自定义Agent
    """

    def __init__(
            self,
            name: str,
            llm: HelloAgentsLLM,
            system_prompt: Optional[str] = None,
            config: Optional[Config] = None,
            tool_registry: Optional['ToolRegistry'] = None,
            # 是否启用工具调用；只有同时满足「enable_tool_calling=True」和「传入了 tool_registry」时才真正启用
            enable_tool_calling: bool = True
    ):
        # 调用父类初始化，完成 name/llm/system_prompt/config/_history 等基础属性的设置
        super().__init__(name, llm, system_prompt, config)
        # 工具注册表：持有所有已注册的工具实例，负责工具的查找和执行
        self.tool_registry = tool_registry
        # 工具调用的实际启用状态：必须同时满足两个条件
        self.enable_tool_calling = enable_tool_calling and tool_registry is not None
        print(f"✅ {name} 初始化完成，工具调用: {'启用' if self.enable_tool_calling else '禁用'}")

    def run(self, input_text: str, max_tool_iterations: int = 3, **kwargs) -> str:
        """
        重写的运行方法 - 实现简单对话逻辑，支持可选工具调用

        执行流程：
          1. 构建消息列表（系统提示词 + 历史消息 + 当前用户输入）
          2. 如果未启用工具 → 直接调用 LLM 并返回结果（与父类行为一致）
          3. 如果启用了工具 → 进入 _run_with_tools 的多轮循环：
             LLM 输出 → 解析是否有工具调用 → 执行工具 → 结果回传 LLM → 重复直到无工具调用

        Args:
            input_text: 用户输入的文本
            max_tool_iterations: 最大工具调用轮数，防止无限循环，默认 3 轮
            **kwargs: 透传给 LLM 的额外参数（如 temperature）
        """
        print(f"🤖 {self.name} 正在处理: {input_text}")

        # ---- 第一步：构建消息列表 ----
        messages = []

        # 添加系统消息（可能包含工具信息）
        # _get_enhanced_system_prompt 会在原始系统提示词后追加「可用工具列表」和「调用格式约定」
        enhanced_system_prompt = self._get_enhanced_system_prompt()
        messages.append({"role": "system", "content": enhanced_system_prompt})

        # 添加历史消息：遍历 _history 中所有之前的对话记录，保持上下文连续性
        for msg in self._history:
            messages.append({"role": msg.role, "content": msg.content})

        # 添加当前用户消息
        messages.append({"role": "user", "content": input_text})

        # ---- 第二步：根据是否启用工具，走不同分支 ----

        # 分支 A：未启用工具调用 → 简单对话，与父类 SimpleAgent.run() 逻辑等价
        if not self.enable_tool_calling:
            response = self.llm.invoke(messages, **kwargs)
            # 将本轮对话（用户输入 + AI 回复）保存到历史记录
            self.add_message(Message(input_text, "user"))
            self.add_message(Message(response, "assistant"))
            print(f"✅ {self.name} 响应完成")
            return response

        # 分支 B：启用工具调用 → 进入多轮工具执行循环
        return self._run_with_tools(messages, input_text, max_tool_iterations, **kwargs)

    def _get_enhanced_system_prompt(self) -> str:
        """
        构建增强的系统提示词，在原始提示词后追加工具使用说明。

        为什么需要这样做？
          因为 LLM 本身不知道自己有哪些工具可用，也不知道该用什么格式请求工具。
          通过在系统提示词中注入「工具列表」和「调用格式约定」，LLM 就能：
            1. 知道有哪些工具可以使用
            2. 按照约定格式输出工具调用请求（如 [TOOL_CALL:calculator:2+3*4]）
          这样 Agent 才能从 LLM 的输出中正确解析出工具调用意图。

        Returns:
            增强后的系统提示词字符串
        """
        # 获取基础系统提示词，如果未设置则使用默认值
        base_prompt = self.system_prompt or "你是一个有用的AI助手。"

        # 如果未启用工具或没有注册任何工具，直接返回原始提示词
        if not self.enable_tool_calling or not self.tool_registry:
            return base_prompt

        # 从工具注册表获取所有工具的描述信息（格式如 "- calculator: 执行数学计算..."）
        tools_description = self.tool_registry.get_tools_description()
        if not tools_description or tools_description == "暂无可用工具":
            return base_prompt

        # 拼接工具说明段落：告诉 LLM 有哪些工具可用
        tools_section = "\n\n## 可用工具\n"
        tools_section += "你可以使用以下工具来帮助回答问题：\n"
        tools_section += tools_description + "\n"

        # 拼接调用格式约定：告诉 LLM 用什么格式请求工具
        # 约定格式为 [TOOL_CALL:工具名:参数]，例如 [TOOL_CALL:search:Python编程]
        tools_section += "\n## 工具调用格式\n"
        tools_section += "当需要使用工具时，请使用以下格式：\n"
        tools_section += "`[TOOL_CALL:{tool_name}:{parameters}]`\n"
        tools_section += "例如：`[TOOL_CALL:search:Python编程]` 或 `[TOOL_CALL:memory:recall=用户信息]`\n\n"
        tools_section += "工具调用结果会自动插入到对话中，然后你可以基于结果继续回答。\n"

        # 将工具说明追加到基础提示词后面
        return base_prompt + tools_section

    def _run_with_tools(self, messages: list, input_text: str, max_tool_iterations: int, **kwargs) -> str:
        """
        支持工具调用的核心运行逻辑 - 实现「LLM → 工具执行 → LLM」的多轮循环。

        循环流程图解：
          ┌─────────────────────────────────────────────┐
          │  第 1 轮：调用 LLM                          │
          │    ↓                                        │
          │  LLM 输出包含 [TOOL_CALL:xxx:yyy] ？        │
          │    ├─ 是 → 解析并执行工具                    │
          │    │     → 将工具结果作为新消息喂回 LLM      │
          │    │     → 进入下一轮循环                    │
          │    └─ 否 → 这就是最终回答，跳出循环          │
          └─────────────────────────────────────────────┘

        Args:
            messages: 完整的消息列表（系统提示 + 历史 + 当前输入）
            input_text: 用户原始输入，用于保存历史
            max_tool_iterations: 最大工具调用轮数，防止死循环
            **kwargs: 透传给 LLM 的额外参数

        Returns:
            LLM 的最终文本回答
        """
        current_iteration = 0  # 当前工具调用轮数计数器
        final_response = ""    # 最终回答

        # 核心循环：每轮调用 LLM，检查是否需要执行工具
        while current_iteration < max_tool_iterations:
            # 调用 LLM 获取响应（非流式，一次性返回完整结果）
            response = self.llm.invoke(messages, **kwargs)

            # 用正则从 LLM 输出中解析所有 [TOOL_CALL:工具名:参数] 标记
            tool_calls = self._parse_tool_calls(response)

            if tool_calls:
                # LLM 请求了工具 → 执行所有工具调用
                print(f"🔧 检测到 {len(tool_calls)} 个工具调用")
                tool_results = []
                clean_response = response

                for call in tool_calls:
                    # 逐个执行工具，收集结果
                    result = self._execute_tool_call(call['tool_name'], call['parameters'])
                    tool_results.append(result)
                    # 从响应文本中移除工具调用标记，保持消息干净
                    clean_response = clean_response.replace(call['original'], "")

                # 将 LLM 的干净回复作为 assistant 消息加入上下文
                messages.append({"role": "assistant", "content": clean_response})

                # 将所有工具执行结果拼接后作为 user 消息喂回 LLM
                # 提示 LLM 基于这些结果继续回答
                tool_results_text = "\n\n".join(tool_results)
                messages.append(
                    {"role": "user", "content": f"工具执行结果：\n{tool_results_text}\n\n请基于这些结果给出完整的回答。"})

                # 轮数 +1，继续下一轮循环
                current_iteration += 1
                continue

            # LLM 没有请求工具 → 说明它已经给出了最终回答，跳出循环
            final_response = response
            break

        # 安全兜底：如果达到最大轮数但还没拿到最终回答，强制让 LLM 做一次总结
        if current_iteration >= max_tool_iterations and not final_response:
            final_response = self.llm.invoke(messages, **kwargs)

        # 将本轮完整对话（用户输入 + AI 最终回答）保存到历史记录
        self.add_message(Message(input_text, "user"))
        self.add_message(Message(final_response, "assistant"))
        print(f"✅ {self.name} 响应完成")

        return final_response

    def _parse_tool_calls(self, text: str) -> list:
        """
        用正则从 LLM 输出文本中解析所有工具调用标记。

        匹配格式：[TOOL_CALL:工具名:参数]
        示例：
          输入: "我来帮你搜索 [TOOL_CALL:search:Python编程] 的结果"
          输出: [{'tool_name': 'search', 'parameters': 'Python编程', 'original': '[TOOL_CALL:search:Python编程]'}]

        正则说明：
          \[TOOL_CALL:     匹配字面量前缀 [TOOL_CALL:
          ([^:]+)          第1个捕获组：工具名（匹配到下一个冒号为止，非贪婪）
          :                分隔工具和参数的冒号
          ([^\]]+)         第2个捕获组：参数（匹配到右方括号 ] 为止）
          \]               匹配字面量右方括号

        Args:
            text: LLM 输出的完整文本

        Returns:
            工具调用字典列表，每个字典包含 tool_name、parameters、original 三个字段
        """
        pattern = r'\[TOOL_CALL:([^:]+):([^\]]+)\]'
        matches = re.findall(pattern, text)

        tool_calls = []
        for tool_name, parameters in matches:
            tool_calls.append({
                'tool_name': tool_name.strip(),
                'parameters': parameters.strip(),
                # 保留原始标记文本，后续用于从 LLM 输出中清除工具调用痕迹
                'original': f'[TOOL_CALL:{tool_name}:{parameters}]'
            })

        return tool_calls

    def _execute_tool_call(self, tool_name: str, parameters: str) -> str:
        """
        执行单个工具调用。

        根据工具类型采用不同的参数传递策略：
          - calculator（计算器）：参数本身就是数学表达式字符串，直接传入
          - 其他工具：需要先通过 _parse_tool_parameters 将字符串解析为字典

        Args:
            tool_name: 要执行的工具名称
            parameters: 工具参数字符串（如 "2+3*4" 或 "action=search,query=Python"）

        Returns:
            带状态前缀的工具执行结果字符串
        """
        # 安全检查：确保工具注册表已初始化
        if not self.tool_registry:
            return f"❌ 错误：未配置工具注册表"

        try:
            # ---- 智能参数解析：根据工具类型选择传参方式 ----
            if tool_name == 'calculator':
                # 计算器工具比较特殊：它的参数就是一个数学表达式字符串（如 "2+3*4"）
                # ToolRegistry.execute_tool 会把它包装成 {"input": "2+3*4"} 传给 CalculatorTool.run()
                result = self.tool_registry.execute_tool(tool_name, parameters)
            else:
                # 其他工具需要更精细的参数解析（如 key=value 格式）
                param_dict = self._parse_tool_parameters(tool_name, parameters)
                # 从注册表中获取工具对象
                tool = self.tool_registry.get_tool(tool_name)
                if not tool:
                    return f"❌ 错误：未找到工具 '{tool_name}'"
                # 直接调用工具的 run 方法，传入解析后的参数字典
                result = tool.run(param_dict)

            return f"🔧 工具 {tool_name} 执行结果：\n{result}"

        except Exception as e:
            return f"❌ 工具调用失败：{str(e)}"

    def _parse_tool_parameters(self, tool_name: str, parameters: str) -> dict:
        """
        将工具参数字符串智能解析为字典格式。

        支持三种参数格式：
          1. 多参数键值对："action=search,query=Python,limit=3" → {'action': 'search', 'query': 'Python', 'limit': '3'}
          2. 单参数键值对："query=Python" → {'query': 'Python'}
          3. 纯值（无等号）：根据工具类型自动推断参数名
             - search 工具 → {'query': 'Python编程'}
             - memory 工具 → {'action': 'search', 'query': '用户信息'}
             - 其他工具   → {'input': '参数值'}

        Args:
            tool_name: 工具名称，用于纯值场景下的参数名推断
            parameters: 原始参数字符串

        Returns:
            解析后的参数字典，可直接传给 Tool.run(parameters)
        """
        param_dict = {}

        if '=' in parameters:
            # 参数中包含等号 → 键值对格式
            if ',' in parameters:
                # 多个键值对，用逗号分隔：action=search,query=Python,limit=3
                pairs = parameters.split(',')
                for pair in pairs:
                    if '=' in pair:
                        key, value = pair.split('=', 1)  # split('=', 1) 只按第一个等号分割
                        param_dict[key.strip()] = value.strip()
            else:
                # 单个键值对：key=value
                key, value = parameters.split('=', 1)
                param_dict[key.strip()] = value.strip()
        else:
            # 参数中没有等号 → 纯值，根据工具类型智能推断参数名
            if tool_name == 'search':
                # 搜索工具的参数名约定为 query
                param_dict = {'query': parameters}
            elif tool_name == 'memory':
                # 记忆工具默认执行 search 动作
                param_dict = {'action': 'search', 'query': parameters}
            else:
                # 通用兜底：参数名约定为 input
                param_dict = {'input': parameters}

        return param_dict

    def stream_run(self, input_text: str, **kwargs) -> Iterator[str]:
        """
        自定义的流式运行方法 - 逐字输出 LLM 响应（类似打字机效果）。

        注意：流式模式不支持工具调用循环，因为工具调用需要拿到完整输出才能解析。
        如果需要工具能力，请使用 run() 方法（非流式）。

        与父类 stream_run 的区别：
          - 增加了更丰富的日志输出（处理开始/完成的提示）
          - 核心逻辑一致：构建消息 → 流式调用 LLM → 逐块 yield → 保存历史

        Args:
            input_text: 用户输入
            **kwargs: 透传给 LLM 的额外参数

        Yields:
            str: LLM 响应的文本片段（逐字产出）
        """
        print(f"🌊 {self.name} 开始流式处理: {input_text}")

        # 构建消息列表（系统提示 + 历史 + 当前输入）
        messages = []

        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})

        for msg in self._history:
            messages.append({"role": msg.role, "content": msg.content})

        messages.append({"role": "user", "content": input_text})

        # 流式调用 LLM：逐块接收响应
        # 注意：不需要在这里 print(chunk)，因为 HelloAgentsLLM.think() 内部
        # 已经打印了每个片段（见 llm.py 的 think 方法）。如果再打印一次，
        # 同一内容会被输出两遍，导致终端文本交错重复的乱码现象。
        full_response = ""
        print("📝 实时响应: ", end="")
        for chunk in self.llm.stream_invoke(messages, **kwargs):
            full_response += chunk       # 拼接完整响应，用于后续保存
            # print(chunk, end="", flush=True)  # 已禁用：think() 内部会打印，避免重复输出
            yield chunk                  # 将当前片段产出给调用者

        print()  # 流式输出结束后换行

        # 将完整对话保存到历史记录
        self.add_message(Message(input_text, "user"))
        self.add_message(Message(full_response, "assistant"))
        print(f"✅ {self.name} 流式响应完成")

    def add_tool(self, tool) -> None:
        """
        添加工具到 Agent（便利方法）。

        这是对外暴露的便捷接口，让调用者可以一行代码给 Agent 挂载工具。
        如果 Agent 初始化时没有传入 tool_registry，这里会自动创建一个。

        使用示例：
            agent = MySimpleAgent(name="助手", llm=llm, system_prompt="...")
            agent.add_tool(CalculatorTool())   # 挂载计算器工具
            response = agent.run("请计算 2+3*4")  # Agent 会自动调用计算器

        Args:
            tool: Tool 子类的实例（如 CalculatorTool），必须继承自 hello_agents.tools.base.Tool
        """
        # 如果之前没有工具注册表，自动创建一个并启用工具调用
        if not self.tool_registry:
            from hello_agents import ToolRegistry
            self.tool_registry = ToolRegistry()
            self.enable_tool_calling = True

        # 将工具注册到注册表中
        self.tool_registry.register_tool(tool)
        print(f"🔧 工具 '{tool.name}' 已添加")

    def has_tools(self) -> bool:
        """检查 Agent 是否已启用工具且拥有可用工具"""
        return self.enable_tool_calling and self.tool_registry is not None

    def remove_tool(self, tool_name: str) -> bool:
        """
        移除指定名称的工具。

        Args:
            tool_name: 要移除的工具名称

        Returns:
            True 表示移除成功，False 表示工具注册表不存在
        """
        if self.tool_registry:
            # 调用注册表的 unregister 方法移除工具
            self.tool_registry.unregister(tool_name)
            return True
        return False

    def list_tools(self) -> list:
        """
        列出当前所有已注册的工具名称。

        Returns:
            工具名称列表，如 ['calculator', 'search']；若无工具则返回空列表
        """
        if self.tool_registry:
            return self.tool_registry.list_tools()
        return []