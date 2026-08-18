import os
import re
import json
from OpenAICompatibleClient import OpenAICompatibleClient
from get_weather import get_weather
from get_attraction import get_attraction

# --- 1. 加载配置文件 ---
# 加载 API 配置
with open(os.path.join(os.path.dirname(__file__), 'config.json'), 'r', encoding='utf-8') as f:
    api_config = json.load(f)

API_KEY = api_config['API_KEY']
BASE_URL = api_config['BASE_URL']
MODEL_ID = api_config['MODEL_ID']
TAVILY_API_KEY = api_config['TAVILY_API_KEY']
os.environ['TAVILY_API_KEY'] = TAVILY_API_KEY

# 加载系统提示词
with open(os.path.join(os.path.dirname(__file__), 'prompt_config.txt'), 'r', encoding='utf-8') as f:
    AGENT_SYSTEM_PROMPT = f.read()


# 将所有工具函数放入一个字典，方便后续调用
available_tools = {
    "get_weather": get_weather,
    "get_attraction": get_attraction,
}


llm = OpenAICompatibleClient(
    model=MODEL_ID,
    api_key=API_KEY,
    base_url=BASE_URL
)

# --- 2. 初始化 ---
user_prompt = "你好，请帮我查询一下今天北京的天气，然后根据天气推荐一个合适的旅游景点。"
prompt_history = [f"用户请求: {user_prompt}"]

print(f"用户输入: {user_prompt}\n" + "=" * 40)

# --- 3. 运行主循环 ---
for i in range(5):  # 设置最大循环次数
    print(f"--- 循环 {i + 1} ---\n")

    # 3.1. 构建Prompt
    full_prompt = "\n".join(prompt_history)

    # 3.2. 调用LLM进行思考
    llm_output = llm.generate(full_prompt, system_prompt=AGENT_SYSTEM_PROMPT)
    # 模型可能会输出多余的Thought-Action，需要截断
    match = re.search(r'(Thought:.*?Action:.*?)(?=\n\s*(?:Thought:|Action:|Observation:)|\Z)', llm_output, re.DOTALL)
    if match:
        truncated = match.group(1).strip()
        if truncated != llm_output.strip():
            llm_output = truncated
            print("已截断多余的 Thought-Action 对")
    print(f"模型输出:\n{llm_output}\n")
    prompt_history.append(llm_output)

    # 3.3. 解析并执行行动
    action_match = re.search(r"Action: (.*)", llm_output, re.DOTALL)
    if not action_match:
        observation = "错误: 未能解析到 Action 字段。请确保你的回复严格遵循 'Thought: ... Action: ...' 的格式。"
        observation_str = f"Observation: {observation}"
        print(f"{observation_str}\n" + "=" * 40)
        prompt_history.append(observation_str)
        continue
    action_str = action_match.group(1).strip()

    if action_str.startswith("Finish"):
        finish_match = re.match(r"Finish\[(.*)\]", action_str)
        if finish_match:
            final_answer = finish_match.group(1)
            print(f"任务完成，最终答案: {final_answer}")
            break
        else:
            observation = "错误: Finish 格式不正确，请使用 Action: Finish[最终答案] 格式。"
            observation_str = f"Observation: {observation}"
            print(f"{observation_str}\n" + "=" * 40)
            prompt_history.append(observation_str)
            continue

    tool_match = re.search(r"(\w+)\((.*)\)", action_str)
    if not tool_match:
        observation = "错误: 无法解析工具调用，请确保格式为 function_name(arg=\"value\")。"
        observation_str = f"Observation: {observation}"
        print(f"{observation_str}\n" + "=" * 40)
        prompt_history.append(observation_str)
        continue

    tool_name = tool_match.group(1)
    args_str = tool_match.group(2)
    kwargs = dict(re.findall(r'(\w+)="([^"]*)"', args_str))

    if tool_name in available_tools:
        observation = available_tools[tool_name](**kwargs)
    else:
        observation = f"错误:未定义的工具 '{tool_name}'"

    # 3.4. 记录观察结果
    observation_str = f"Observation: {observation}"
    print(f"{observation_str}\n" + "=" * 40)
    prompt_history.append(observation_str)
