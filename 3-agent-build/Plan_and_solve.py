# os: 环境变量读取（dotenv 依赖它来注入环境变量）
import os
# ast: 提供 literal_eval，用于安全地将字符串解析为 Python 对象（如列表、字典）
import ast
# HelloAgentsLLM: 封装了 LLM API 调用的客户端类，来自同目录下的 llm_client.py
from llm_client import HelloAgentsLLM
# load_dotenv: 从 .env 文件加载环境变量到系统环境中，避免在代码里硬编码 API Key 等敏感信息
from dotenv import load_dotenv
# List, Dict: 类型提示工具（当前文件未直接使用，属于预留导入）
from typing import List, Dict

# 加载 .env 文件中的环境变量，处理文件不存在异常
try:
    load_dotenv()
except FileNotFoundError:
    print("警告：未找到 .env 文件，将使用系统环境变量。")
except Exception as e:
    print(f"警告：加载 .env 文件时出错: {e}")

# ===================== 2. 规划器 (Planner) =====================
# 规划器的职责：将用户的复杂问题拆解为多个有序的子步骤
# 提示词模板，{question} 会在运行时被替换为用户的实际问题
# 模板中用 ```python ... ``` 包裹输出，是为了后续能用字符串分割精确提取列表内容
PLANNER_PROMPT_TEMPLATE = """
你是一个顶级的AI规划专家。你的任务是将用户提出的复杂问题分解成一个由多个简单步骤组成的行动计划。
请确保计划中的每个步骤都是一个独立的、可执行的子任务，并且严格按照逻辑顺序排列。
你的输出必须是一个Python列表，其中每个元素都是一个描述子任务的字符串。

问题: {question}

请严格按照以下格式输出你的计划，```python与```作为前后缀是必要的:
```python
["步骤1", "步骤2", "步骤3", ...]
```
"""


class Planner:
    """规划器：调用 LLM 将复杂问题拆解为一个有序的步骤列表。"""

    def __init__(self, llm_client: HelloAgentsLLM):
        self.llm_client = llm_client  # 持有 LLM 客户端实例，用于发送请求

    def plan(self, question: str) -> list[str]:
        """
        核心方法：接收用户问题，返回 LLM 生成的步骤列表。
        返回示例: ["计算周一和周二的苹果数", "计算周三的苹果数", "求总和"]
        若解析失败则返回空列表。
        """
        # 1) 将用户问题填入提示词模板，构造完整的 prompt
        prompt = PLANNER_PROMPT_TEMPLATE.format(question=question)
        # 2) 构造 OpenAI 格式的消息列表（role=user 表示用户消息）
        messages = [{"role": "user", "content": prompt}]

        print("--- 正在生成计划 ---")
        # 3) 调用 LLM 获取回复；think() 返回字符串或 None，用 or "" 兜底为空字符串
        response_text = self.llm_client.think(messages=messages) or ""
        print(f"✅ 计划已生成:\n{response_text}")

        # 4) 从 LLM 的原始文本中提取 Python 列表
        try:
            # 4a) 第一步：字符串分割提取代码块内容
            #     LLM 输出格式为: ```python\n["步骤1", "步骤2"]\n```
            #     split("```python")[1]  → 取 ```python 之后的部分
            #     split("```")[0]         → 再取 ``` 之前的部分
            #     strip()                → 去除首尾空白字符
            #     最终得到纯字符串: '["步骤1", "步骤2"]'
            plan_str = response_text.split("```python")[1].split("```")[0].strip()

            # 4b) 第二步：安全地将字符串解析为 Python 对象
            #     使用 ast.literal_eval 而非 eval，原因是：
            #     - eval() 会执行任意 Python 代码，存在安全风险
            #     - json.loads() 不支持单引号，而 LLM 经常输出单引号
            #     - ast.literal_eval 只解析字面量（字符串/列表/字典/数字等），安全且兼容性好
            plan = ast.literal_eval(plan_str)

            # 4c) 第三步：防御性校验，确保解析结果确实是列表类型
            #     如果 LLM 输出了非列表格式（如字典），则返回空列表兜底
            return plan if isinstance(plan, list) else []

        except (ValueError, SyntaxError, IndexError) as e:
            # ValueError/SyntaxError: literal_eval 解析失败（如格式不合法）
            # IndexError: split 分割后索引越界（如 LLM 没有按 ```python 格式输出）
            print(f"❌ 解析计划时出错: {e}")
            print(f"原始响应: {response_text}")
            return []
        except Exception as e:
            # 兜底捕获其他未知异常，防止程序崩溃
            print(f"❌ 解析计划时发生未知错误: {e}")
            return []


# ===================== 3. 执行器 (Executor) =====================
# 执行器的职责：接收一个步骤列表，按顺序逐步调用 LLM 执行每个子任务
# 每一步执行时都会把「原始问题 + 完整计划 + 历史步骤结果 + 当前步骤」一起传给 LLM
# 这样 LLM 能利用前面步骤的结果来回答当前步骤，实现上下文的逐步传递
EXECUTOR_PROMPT_TEMPLATE = """
你是一位顶级的AI执行专家。你的任务是严格按照给定的计划，一步步地解决问题。
你将收到原始问题、完整的计划、以及到目前为止已经完成的步骤和结果。
请你专注于解决“当前步骤”，并仅输出该步骤的最终答案，不要输出任何额外的解释或对话。

# 原始问题:
{question}

# 完整计划:
{plan}

# 历史步骤与结果:
{history}

# 当前步骤:
{current_step}

请仅输出针对“当前步骤”的回答:
"""


class Executor:
    """执行器：按计划逐步执行每个子任务，每步都携带历史上下文。"""

    def __init__(self, llm_client: HelloAgentsLLM):
        self.llm_client = llm_client  # 持有 LLM 客户端实例

    def execute(self, question: str, plan: list[str]) -> str:
        """
        按计划顺序逐步执行，返回最后一步的结果作为最终答案。
        :param question: 用户的原始问题
        :param plan: Planner 生成的步骤列表
        :return: 最后一步的执行结果（字符串）
        """
        # history: 累积所有已完成步骤的结果，供后续步骤作为上下文参考
        history = ""
        # final_answer: 记录每一步的结果，循环结束后保存的是最后一步的结果
        final_answer = ""

        print("\n--- 正在执行计划 ---")
        # enumerate(plan, 1) 从 1 开始编号，方便打印 "步骤 1/N"
        for i, step in enumerate(plan, 1):
            print(f"\n-> 正在执行步骤 {i}/{len(plan)}: {step}")
            # 将原始问题、完整计划、历史结果、当前步骤填入执行器提示词模板
            # history 为空时显示"无"，避免 LLM 看到空白上下文
            prompt = EXECUTOR_PROMPT_TEMPLATE.format(
                question=question, plan=plan, history=history if history else "无", current_step=step
            )
            messages = [{"role": "user", "content": prompt}]

            # 调用 LLM 获取当前步骤的执行结果
            response_text = self.llm_client.think(messages=messages) or ""

            # 将当前步骤及其结果追加到历史记录中，供下一步使用
            history += f"步骤 {i}: {step}\n结果: {response_text}\n\n"
            # 更新最终答案为当前步骤结果（最后一步的结果即为整体最终答案）
            final_answer = response_text
            print(f"✅ 步骤 {i} 已完成，结果: {final_answer}")

        return final_answer


# ===================== 4. 智能体 (PlanAndSolveAgent) =====================
# 整合 Planner 和 Executor，实现「先规划，再执行」的两阶段工作流
class PlanAndSolveAgent:
    """Plan-and-Solve 智能体：先调用 Planner 生成计划，再调用 Executor 逐步执行。"""

    def __init__(self, llm_client: HelloAgentsLLM):
        self.llm_client = llm_client
        # 组合模式：Agent 内部持有 Planner 和 Executor 实例
        self.planner = Planner(self.llm_client)
        self.executor = Executor(self.llm_client)

    def run(self, question: str):
        """
        智能体主运行流程：
        1. 调用 Planner 将问题拆解为步骤列表
        2. 若计划为空（解析失败），提前终止
        3. 调用 Executor 逐步执行计划，输出最终答案
        """
        print(f"\n--- 开始处理问题 ---\n问题: {question}")

        # 阶段一：规划
        plan = self.planner.plan(question)
        if not plan:
            # 规划失败时直接返回，不进入执行阶段
            print("\n--- 任务终止 --- \n无法生成有效的行动计划。")
            return

        # 阶段二：执行
        final_answer = self.executor.execute(question, plan)
        print(f"\n--- 任务完成 ---\n最终答案: {final_answer}")


# ===================== 5. 主函数入口 =====================
# __name__ == '__main__' 确保只有直接运行本文件时才执行，被 import 时不会触发
if __name__ == '__main__':
    try:
        # 1) 初始化 LLM 客户端（内部从环境变量读取 API Key 和 Base URL）
        llm_client = HelloAgentsLLM()
        # 2) 用客户端创建 Plan-and-Solve 智能体
        agent = PlanAndSolveAgent(llm_client)
        # 3) 定义测试问题（多步数学应用题，需要拆解后逐步求解）
        question = "一个水果店周一卖出了15个苹果。周二卖出的苹果数量是周一的两倍。周三卖出的数量比周二少了5个。请问这三天总共卖出了多少个苹果？"
        # 4) 运行智能体：先规划 → 再执行 → 输出最终答案
        agent.run(question)
    except ValueError as e:
        # 捕获 HelloAgentsLLM 初始化时可能抛出的环境变量缺失异常
        print(e)