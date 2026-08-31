# my_plan_solve_agent.py
# 本文件演示如何基于 hello_agents 框架的 PlanAndSolveAgent 基类，
# 扩展实现一个支持「自定义提示词模板 + 增强计划解析 + 丰富日志」的 Plan-and-Solve Agent。
#
# 核心思路：
#   1. 继承 PlanAndSolveAgent，复用其「规划 → 逐步执行」的两阶段工作流
#   2. 增强计划解析逻辑：当 LLM 未按 ```python 格式输出时，尝试用正则从文本中提取列表
#   3. 支持通过 custom_prompts 参数传入自定义提示词模板，适配不同场景（通用、数学等）
#   4. 在关键步骤添加丰富的日志输出，方便学习理解 Plan-and-Solve 范式的工作流程

import re
import ast
from typing import Optional, Dict
# PlanAndSolveAgent: 框架提供的规划执行 Agent 基类，提供「先规划，再执行」的两阶段流程
# HelloAgentsLLM: 框架的统一 LLM 客户端，支持多种 provider
# Config: 框架的配置类，控制 temperature、max_tokens 等参数
# Message: 框架的消息数据类，用于存储单条对话记录（role + content）
from hello_agents import PlanAndSolveAgent, HelloAgentsLLM, Config, Message


# 自定义默认规划器提示词
# 与框架内置的 DEFAULT_PLANNER_PROMPT 相比，增加了更明确的格式要求说明
MY_PLANNER_PROMPT = """
你是一个顶级的AI规划专家。你的任务是将用户提出的复杂问题分解成一个由多个简单步骤组成的行动计划。
请确保计划中的每个步骤都是一个独立的、可执行的子任务，并且严格按照逻辑顺序排列。
你的输出必须是一个Python列表，其中每个元素都是一个描述子任务的字符串。

问题: {question}

请严格按照以下格式输出你的计划:
```python
["步骤1", "步骤2", "步骤3", ...]
```
"""

# 自定义默认执行器提示词
# 与框架内置的 DEFAULT_EXECUTOR_PROMPT 基本一致，保持通用化设计
MY_EXECUTOR_PROMPT = """
你是一位顶级的AI执行专家。你的任务是严格按照给定的计划，一步步地解决问题。
你将收到原始问题、完整的计划、以及到目前为止已经完成的步骤和结果。
请你专注于解决"当前步骤"，并仅输出该步骤的最终答案，不要输出任何额外的解释或对话。

# 原始问题:
{question}

# 完整计划:
{plan}

# 历史步骤与结果:
{history}

# 当前步骤:
{current_step}

请仅输出针对"当前步骤"的回答:
"""


class MyPlanAndSolveAgent(PlanAndSolveAgent):
    """
    重写的 Plan-and-Solve Agent - 在原版基础上增强了计划解析和日志输出。

    与父类的区别：
      - 父类 PlanAndSolveAgent 的计划解析只支持 ```python 代码块格式
        如果 LLM 没有按此格式输出（如直接输出 ["步骤1", "步骤2"]），解析会失败返回空列表
      - 本类 MyPlanAndSolveAgent 增加了多级降级解析策略：
        1. 优先尝试 ```python 代码块提取（与父类一致）
        2. 失败后用正则从文本中搜索列表模式 [...]
        3. 最终兜底尝试 ast.literal_eval 直接解析
      - 支持通过 custom_prompts 参数替换规划器和执行器的提示词模板
      - 增加了更丰富的分步日志输出，方便学习和调试

    工作流程图解：
      ┌──────────────────────────────────────┐
      │  第 1 阶段：规划（Plan）              │
      │    → 调用 LLM 将问题拆解为步骤列表   │
      │    → 增强解析：提取 Python 列表       │
      └──────────────┬───────────────────────┘
                     ↓
          ┌──────────────────────┐
          │ 计划是否为空？        │
          ├─ 是 → 任务终止       │
          └─ 否 ↓                │
      ┌──────────────────────────────────────┐
      │  第 2 阶段：执行（Solve）             │
      │    → 逐步执行每个子任务               │
      │    → 每步携带历史上下文               │
      │    → 最终汇总得到答案                 │
      └──────────────────────────────────────┘

    使用示例：
        # 使用默认提示词
        agent = MyPlanAndSolveAgent(name="规划助手", llm=llm)
        result = agent.run("一个复杂的多步问题...")

        # 使用自定义数学提示词
        math_prompts = {
            "planner": "你是数学问题规划专家，请将数学问题分解为计算步骤:\\n问题: {question}",
            "executor": "你是数学计算专家，请计算当前步骤:\\n当前步骤: {current_step}"
        }
        math_agent = MyPlanAndSolveAgent(name="数学助手", llm=llm, custom_prompts=math_prompts)
    """

    def __init__(
            self,
            name: str,
            llm: HelloAgentsLLM,
            system_prompt: Optional[str] = None,
            config: Optional[Config] = None,
            custom_prompts: Optional[Dict[str, str]] = None
    ):
        """
        初始化 MyPlanAndSolveAgent

        Args:
            name: Agent 名称
            llm: LLM 实例
            system_prompt: 系统提示词（可选）
            config: 配置对象（可选）
            custom_prompts: 自定义提示词模板字典，需包含 "planner" 和 "executor" 两个键
                           如果未提供，则使用 MY_PLANNER_PROMPT / MY_EXECUTOR_PROMPT
        """
        # 调用父类初始化
        # 父类会根据 custom_prompts 创建 Planner 和 Executor 实例
        super().__init__(name, llm, system_prompt, config, custom_prompts)

        # 如果用户没有传入自定义提示词，则用我们自己的默认模板覆盖父类的
        # 父类构造函数中，custom_prompts 为 None 时 Planner/Executor 会使用框架内置的 DEFAULT_*_PROMPT
        # 这里替换为我们自定义的 MY_*_PROMPT
        if custom_prompts is None:
            from hello_agents.agents.plan_solve_agent import Planner, Executor
            self.planner = Planner(self.llm, MY_PLANNER_PROMPT)
            self.executor = Executor(self.llm, MY_EXECUTOR_PROMPT)

        print(f"✅ {name} 初始化完成")

    def run(self, input_text: str, **kwargs) -> str:
        """
        重写的运行方法 - 实现「先规划，再执行」的两阶段工作流。

        与父类 run() 的核心区别：
          - 增强了计划解析逻辑（多级降级策略）
          - 增加了更丰富的分步日志输出

        执行流程：
          1. 调用规划器（Planner）将问题拆解为步骤列表
          2. 如果计划为空，尝试用增强解析从 LLM 原始输出中提取
          3. 如果仍为空，任务终止
          4. 调用执行器（Executor）按计划逐步执行
          5. 返回最终答案并保存对话历史

        Args:
            input_text: 要解决的问题
            **kwargs: 透传给 LLM 的额外参数

        Returns:
            最终答案
        """
        print(f"\n🤖 {self.name} 开始处理问题: {input_text}")

        # ---- 第 1 阶段：生成计划 ----
        print("\n📋 === 第 1 阶段：规划 ===")
        plan = self.planner.plan(input_text, **kwargs)

        # 增强解析：如果父类的标准解析失败了，尝试从 LLM 原始输出中恢复
        if not plan:
            print("\n⚠️ 标准解析失败，尝试增强解析...")
            plan = self._enhanced_plan_parse(input_text, **kwargs)

        # 计划仍为空，任务终止
        if not plan:
            final_answer = "无法生成有效的行动计划，任务终止。"
            print(f"\n--- 任务终止 ---\n{final_answer}")
            self.add_message(Message(input_text, "user"))
            self.add_message(Message(final_answer, "assistant"))
            return final_answer

        # 打印计划摘要
        print(f"\n📋 计划共 {len(plan)} 个步骤:")
        for i, step in enumerate(plan, 1):
            print(f"   {i}. {step}")

        # ---- 第 2 阶段：执行计划 ----
        print(f"\n🔧 === 第 2 阶段：执行 ===")
        final_answer = self.executor.execute(input_text, plan, **kwargs)
        print(f"\n--- 任务完成 ---\n最终答案: {final_answer}")

        # 保存对话历史（继承自 Agent 基类的能力）
        self.add_message(Message(input_text, "user"))
        self.add_message(Message(final_answer, "assistant"))

        return final_answer

    def _enhanced_plan_parse(self, question: str, **kwargs) -> list:
        """
        增强版计划解析 - 当标准 ```python 代码块解析失败时的降级策略。

        为什么需要增强？
          父类 Planner.plan() 的解析逻辑是：
            response_text.split("```python")[1].split("```")[0].strip()
          这要求 LLM 必须按 ```python ... ``` 格式输出代码块。
          但实际中 LLM 可能：
            - 直接输出 ["步骤1", "步骤2"]（没有代码块包裹）
            - 用 ``` 而不是 ```python（缺少语言标识）
            - 在列表前后加了多余文字

        降级策略（按优先级依次尝试）：
          1. 重新调用 LLM，明确要求只输出列表
          2. 用正则从响应文本中搜索 [...] 列表模式
          3. 用 ast.literal_eval 尝试直接解析

        Args:
            question: 原始问题
            **kwargs: 透传给 LLM 的额外参数

        Returns:
            解析后的步骤列表，解析失败返回空列表
        """
        # 重新构造一个更简洁的规划请求
        simple_prompt = f"""请将以下问题分解为步骤列表，只输出Python列表格式，不要其他内容：

问题: {question}

输出格式: ["步骤1", "步骤2", ...]"""

        messages = [{"role": "user", "content": simple_prompt}]
        response_text = self.llm.invoke(messages, **kwargs) or ""

        # --- 策略 1：正则搜索列表模式 ---
        # 匹配 [ 开头、] 结尾的内容，尝试提取列表
        list_match = re.search(r'\[.*\]', response_text, re.DOTALL)
        if list_match:
            try:
                plan = ast.literal_eval(list_match.group())
                if isinstance(plan, list) and len(plan) > 0:
                    print(f"✅ 增强解析成功（正则提取），共 {len(plan)} 个步骤")
                    return plan
            except (ValueError, SyntaxError):
                pass

        # --- 策略 2：直接尝试 ast.literal_eval 解析整段文本 ---
        try:
            plan = ast.literal_eval(response_text.strip())
            if isinstance(plan, list) and len(plan) > 0:
                print(f"✅ 增强解析成功（直接解析），共 {len(plan)} 个步骤")
                return plan
        except (ValueError, SyntaxError):
            pass

        print("❌ 增强解析也失败了")
        return []
