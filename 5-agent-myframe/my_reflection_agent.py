# my_reflection_agent.py
# 本文件演示如何基于 hello_agents 框架的 ReflectionAgent 基类，
# 扩展实现一个支持「自定义提示词模板 + 增强停止检测」的 Reflection Agent。
#
# 核心思路：
#   1. 继承 ReflectionAgent，复用其「初始执行 → 反思 → 优化」的迭代循环
#   2. 支持通过 custom_prompts 参数传入自定义提示词模板，适配不同场景（通用文本、代码生成等）
#   3. 增强停止检测逻辑：使用正则排除否定语境（如"不能认为无需改进"），避免误判
#   4. 在关键步骤添加丰富的日志输出，方便学习理解 Reflection 范式的工作流程

import re
from typing import Optional, Dict
# ReflectionAgent: 框架提供的反思 Agent 基类，提供「执行→反思→优化」迭代循环
# HelloAgentsLLM: 框架的统一 LLM 客户端，支持多种 provider
# Config: 框架的配置类，控制 temperature、max_tokens 等参数
# Message: 框架的消息数据类，用于存储单条对话记录（role + content）
from hello_agents import ReflectionAgent, HelloAgentsLLM, Config, Message


# 自定义的默认提示词模板
# 与框架内置的 DEFAULT_PROMPTS 相比，措辞略有调整，使其更通用化
MY_DEFAULT_PROMPTS = {
    "initial": """
请根据以下要求完成任务:

任务: {task}

请提供一个完整、准确的回答。
""",
    "reflect": """
请仔细审查以下回答，并找出可能的问题或改进空间:

# 原始任务:
{task}

# 当前回答:
{content}

请分析这个回答的质量，指出不足之处，并提出具体的改进建议。
如果回答已经很好，请回答"无需改进"。
""",
    "refine": """
请根据反馈意见改进你的回答:

# 原始任务:
{task}

# 上一轮回答:
{last_attempt}

# 反馈意见:
{feedback}

请提供一个改进后的回答。
"""
}


class MyReflectionAgent(ReflectionAgent):
    """
    重写的 Reflection Agent - 在原版基础上增强了停止检测和日志输出。

    与父类的区别：
      - 父类 ReflectionAgent 的停止检测只做简单的字符串包含判断（"无需改进" in feedback）
      - 本类 MyReflectionAgent 增加了正则排除逻辑，防止否定语境被误判为"无需改进"
        例如："不能认为无需改进"、"并非无需改进" 等不应触发停止
      - 增加了更丰富的分步日志输出，方便学习和调试

    工作流程图解：
      ┌──────────────────────────────────────┐
      │  第 1 步：执行初始任务                │
      │    → 生成初始回答                     │
      └──────────────┬───────────────────────┘
                     ↓
      ┌──────────────────────────────────────┐
      │  第 2 步：反思（Review）              │
      │    → 审查回答质量，找出改进空间       │
      └──────────────┬───────────────────────┘
                     ↓
          ┌──────────────────────┐
          │ 回答是否"无需改进"？  │
          │  （增强版检测逻辑）   │
          ├─ 是 → 输出最终结果   │
          └─ 否 ↓                │
      ┌──────────────────────────────────────┐
      │  第 3 步：优化（Refine）              │
      │    → 根据反馈改进回答                 │
      └──────────────┬───────────────────────┘
                     ↓
            回到第 2 步（循环）
            最多迭代 max_iterations 次

    使用示例：
        # 使用默认通用提示词
        agent = MyReflectionAgent(name="反思助手", llm=llm)
        result = agent.run("写一篇关于AI的文章")

        # 使用自定义代码生成提示词
        code_prompts = {
            "initial": "你是Python专家，请编写函数:{task}",
            "reflect": "请审查代码的算法效率:\\n任务:{task}\\n代码:{content}",
            "refine": "请根据反馈优化代码:\\n任务:{task}\\n反馈:{feedback}"
        }
        code_agent = MyReflectionAgent(name="代码助手", llm=llm, custom_prompts=code_prompts)
    """

    def __init__(
            self,
            name: str,
            llm: HelloAgentsLLM,
            system_prompt: Optional[str] = None,
            config: Optional[Config] = None,
            max_iterations: int = 3,
            custom_prompts: Optional[Dict[str, str]] = None
    ):
        """
        初始化 MyReflectionAgent

        Args:
            name: Agent 名称
            llm: LLM 实例
            system_prompt: 系统提示词（可选）
            config: 配置对象（可选）
            max_iterations: 最大迭代次数，默认 3 次
            custom_prompts: 自定义提示词模板字典，需包含 "initial"、"reflect"、"refine" 三个键
                           如果未提供，则使用 MY_DEFAULT_PROMPTS
        """
        # 调用父类初始化
        # 父类会设置 name/llm/system_prompt/config，并初始化 Memory 和 prompts
        super().__init__(name, llm, system_prompt, config, max_iterations, custom_prompts)

        # 如果用户没有传入自定义提示词，则使用我们自己的默认模板
        # 注意：父类构造函数中已经做了 custom_prompts if custom_prompts else DEFAULT_PROMPTS 的判断
        # 所以这里需要在我们自己的初始化中覆盖为 MY_DEFAULT_PROMPTS
        if custom_prompts is None:
            self.prompts = MY_DEFAULT_PROMPTS

        print(f"✅ {name} 初始化完成，最大迭代次数: {max_iterations}")

    def run(self, input_text: str, **kwargs) -> str:
        """
        重写的运行方法 - 实现「初始执行 → 反思 → 优化」的迭代循环。

        与父类 run() 的核心区别：
          - 增强了「是否需要停止」的检测逻辑
          - 父类只做简单的 "无需改进" in feedback 判断
          - 本类使用正则排除否定语境，避免误判

        执行流程：
          1. 重置记忆，确保每次 run() 都是全新的迭代过程
          2. 调用 LLM 完成初始任务
          3. 进入迭代循环：
             a. 调用 LLM 对当前结果进行反思
             b. 检查反思结果是否认为"无需改进"（增强版检测）
             c. 如果仍需改进，调用 LLM 根据反馈优化结果
          4. 返回最终优化后的结果

        Args:
            input_text: 任务描述文本
            **kwargs: 透传给 LLM 的额外参数（如 temperature）

        Returns:
            经过迭代优化后的最终结果
        """
        print(f"\n🤖 {self.name} 开始处理任务: {input_text}")

        # ---- 第一步：重置记忆 ----
        # 每次 run() 都从空白开始，避免上一次运行的记忆干扰当前任务
        from hello_agents.agents.reflection_agent import Memory
        self.memory = Memory()

        # ---- 第二步：初始执行 ----
        print("\n--- 正在进行初始尝试 ---")
        initial_prompt = self.prompts["initial"].format(task=input_text)
        initial_result = self._get_llm_response(initial_prompt, **kwargs)
        self.memory.add_record("execution", initial_result)
        print(f"📄 初始回答长度: {len(initial_result)} 字符")

        # ---- 第三步：迭代循环 - 反思与优化 ----
        for i in range(self.max_iterations):
            print(f"\n--- 第 {i + 1}/{self.max_iterations} 轮迭代 ---")

            # a. 反思：让 LLM 审查当前结果
            print("\n-> 正在进行反思...")
            last_result = self.memory.get_last_execution()
            reflect_prompt = self.prompts["reflect"].format(
                task=input_text,
                content=last_result
            )
            feedback = self._get_llm_response(reflect_prompt, **kwargs)
            self.memory.add_record("reflection", feedback)

            # b. 检查是否需要停止（增强版检测逻辑）
            if self._should_stop(feedback):
                print("\n✅ 反思认为结果已无需改进，任务完成。")
                break

            # c. 优化：根据反馈改进结果
            print("\n-> 正在进行优化...")
            refine_prompt = self.prompts["refine"].format(
                task=input_text,
                last_attempt=last_result,
                feedback=feedback
            )
            refined_result = self._get_llm_response(refine_prompt, **kwargs)
            self.memory.add_record("execution", refined_result)
            print(f"📄 优化后回答长度: {len(refined_result)} 字符")

        # ---- 第四步：获取最终结果并保存 ----
        final_result = self.memory.get_last_execution()
        print(f"\n--- 任务完成 ---")

        # 将本轮对话保存到历史记录（继承自 Agent 基类的能力）
        self.add_message(Message(input_text, "user"))
        self.add_message(Message(final_result, "assistant"))

        return final_result

    def _should_stop(self, feedback: str) -> bool:
        """
        增强版的停止检测逻辑 - 判断反思结果是否认为"无需改进"。

        为什么需要增强？
          父类的检测逻辑是简单的字符串包含判断：
            "无需改进" in feedback
          但这会产生误判。例如以下反馈：
            - "不能认为无需改进，仍存在性能问题"  → 实际含义是需要改进
            - "并非无需改进"                        → 实际含义是需要改进
          这些情况下，简单的包含判断会错误地触发停止。

        本方法的检测策略：
          1. 先检查中文标记 "无需改进"
             - 用正则检查是否存在否定前缀（如"不"、"未"、"没"、"非"等）
             - 如果没有否定前缀 → 确认为"无需改进"
          2. 再检查英文标记 "no need for improvement"
             - 同理检查否定前缀（如"not"、"no"、"still"等）
          3. 任一语言检测到真正的"无需改进"→ 返回 True

        Args:
            feedback: LLM 的反思反馈文本

        Returns:
            True 表示应该停止迭代（结果已足够好），False 表示继续迭代
        """
        # --- 中文检测 ---
        # 首先检查是否包含"无需改进"关键词
        if "无需改进" in feedback:
            # 用正则排除否定语境
            # 匹配模式：(不|未|没|非|无法|不能|不可|难以|远非) + 0~6个字符 + "无需改进"
            # 例如："不能认为无需改进"、"并非无需改进"、"远非无需改进"
            has_negative_context = re.search(
                r'(不|未|没|非|无法|不能|不可|难以|远非).{0,6}无需改进',
                feedback
            ) is not None
            if not has_negative_context:
                return True

        # --- 英文检测 ---
        # 检查英文 "no need for improvement" 关键词
        if "no need for improvement" in feedback.lower():
            # 同理排除否定语境
            has_negative_context = re.search(
                r'(not|no|still|cannot|hardly).{0,10}no need for improvement',
                feedback.lower()
            ) is not None
            if not has_negative_context:
                return True

        return False
