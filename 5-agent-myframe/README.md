# 5-agent-myframe

基于 [HelloAgents](https://github.com/jjyaoao/HelloAgents) 多智能体框架的自定义扩展学习项目。通过继承框架基类并重写关键方法，实现并演示了 **Simple、ReAct、Reflection、Plan-and-Solve** 四种主流智能体范式，同时构建了自定义 LLM 客户端、计算器/搜索工具、工具链管理与异步执行等配套能力。

## 目录结构

| 文件 | 说明 |
| --- | --- |
| `my_llm.py` | 自定义 LLM 客户端 `MyLLM`：支持 Minimax Provider，修复流式响应空 choices 数据块导致的 IndexError |
| `my_main.py` | 演示 `MyLLM` 以 `minimax` Provider 发起流式对话 |
| `my_simple_agent.py` | 自定义 `MySimpleAgent`：在框架 SimpleAgent 基础上增加 `[TOOL_CALL:xxx:yyy]` 工具调用循环 |
| `my_react_agent.py` | 自定义 `MyReActAgent`：自定义提示词模板的「思考-行动」推理智能体 |
| `my_reflection_agent.py` | 自定义 `MyReflectionAgent`：增强停止检测（正则排除否定语境）的反思智能体 |
| `my_plan_solve_agent.py` | 自定义 `MyPlanAndSolveAgent`：多级降级计划解析的「先规划后执行」智能体 |
| `my_calculator_tool.py` | 自定义计算器工具，基于 `ToolRegistry` 注册 |
| `my_advanced_search.py` | 多源搜索引擎工具：自动探测并整合 Tavily / SerpApi 搜索源 |
| `tool_chain_manager.py` | 工具链管理器：将多个工具按步骤串联执行，支持模板变量引用 |
| `async_tool_executor.py` | 异步工具执行器：基于线程池并行执行多个工具 |
| `simpleHelloAgents.py` | 框架原生 `SimpleAgent` 入门示例 |
| `test_simple_agent.py` | 测试：基础对话 / 工具增强 / 流式响应 / 动态工具管理 |
| `test_react_agent.py` | 测试：ReAct 推理循环与工具调用 |
| `test_reflection_agent.py` | 测试：反思迭代与停止检测 |
| `test_plan_solve_agent.py` | 测试：两阶段规划执行 |
| `test_my_calculator.py` | 测试：自定义计算器工具 |
| `test_advanced_search.py` | 测试：多源搜索工具 |

## 环境准备

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置环境变量

复制 `.env_example` 为 `.env` 并填入真实配置：

```bash
# 通用 LLM 配置（provider="auto" 时使用）
LLM_MODEL_ID="YOUR-MODEL"
LLM_API_KEY="YOUR-API-KEY"
LLM_BASE_URL="YOUR-URL"

# Minimax 专属配置（provider="minimax" 时使用）
MINIMAX_API_KEY="YOUR-MINIMAX-API-KEY"

# 搜索 API 密钥（二选一，用于 my_advanced_search.py）
SERPAPI_API_KEY="YOUR-SERPAPI-API-KEY"
TAVILY_API_KEY="YOUR-TAVILY-API-KEY"
```

常用变量说明：

| 环境变量 | 用途 |
| --- | --- |
| `LLM_MODEL_ID` / `LLM_API_KEY` / `LLM_BASE_URL` | 框架自动检测 Provider 时读取的通用 LLM 配置 |
| `MINIMAX_API_KEY` | `MyLLM(provider="minimax")` 的 API 密钥，base_url 默认为 `https://api.minimaxi.com/v1` |
| `SERPAPI_API_KEY` | SerpApi 搜索源密钥，获取地址 https://serpapi.com/ |
| `TAVILY_API_KEY` | Tavily 搜索源密钥，获取地址 https://tavily.com/ |

> 搜索密钥无需全部配置：`my_advanced_search.py` 会自动探测可用搜索源，配置任意一个即可；同时启用时优先尝试 Tavily。使用 SerpApi 请确保 `serpapi>=1.1.0`（新版 `Client.search` API），使用 Tavily 请额外安装 `tavily-python`（见 requirements.txt 中的可选依赖）。

## 快速开始

配置好 `.env` 后，依次运行各测试脚本观察效果：

```bash
# 框架原生 SimpleAgent 入门
python simpleHelloAgents.py

# 自定义 SimpleAgent：工具调用循环
python test_simple_agent.py

# ReAct 推理智能体
python test_react_agent.py

# Reflection 反思智能体
python test_reflection_agent.py

# Plan-and-Solve 规划智能体
python test_plan_solve_agent.py

# 自定义工具测试
python test_my_calculator.py
python test_advanced_search.py

# Minimax Provider 流式调用演示
python my_main.py
```

## 功能模块

### MyLLM（my_llm.py）

继承 `HelloAgentsLLM` 的自定义客户端：

- `provider="minimax"` 时使用 Minimax 专属凭证（`MINIMAX_API_KEY` / `MINIMAX_MODEL_ID`），默认模型 `MiniMax-M2.5`；
- 其他 provider 完全走父类原始逻辑（如 `provider="auto"` 自动读取 `LLM_*` 环境变量）；
- 重写 `think()` 方法：在访问 `chunk.choices[0]` 前判空，跳过流末尾仅携带 usage 统计的空 choices 数据块，修复 `IndexError: list index out of range`。

### MySimpleAgent（my_simple_agent.py）

在原版 `SimpleAgent` 一问一答的基础上增加工具调用能力：通过在系统提示词中注入工具描述与 `[TOOL_CALL:工具名:参数]` 调用格式，解析 LLM 输出并执行工具，将结果回传 LLM，支持多轮循环（默认最多 3 轮，`max_tool_iterations` 可调）。

### MyReActAgent（my_react_agent.py）

基于自定义提示词模板 `MY_REACT_PROMPT` 的推理-行动智能体：每轮输出 `Thought` / `Action`，通过 `工具名[参数]` 调用工具、`Finish[答案]` 结束，最多执行 `max_steps` 步（默认 5）。

### MyReflectionAgent（my_reflection_agent.py）

「初始执行 → 反思 → 优化」迭代循环，相比父类的两点增强：

- 停止检测升级：父类仅做 `"无需改进" in feedback` 子串匹配，本类增加正则排除否定语境（如"不能认为无需改进"），避免误判；
- 支持 `custom_prompts` 自定义 `initial` / `reflect` / `refine` 三段提示词模板。

### MyPlanAndSolveAgent（my_plan_solve_agent.py）

「先规划、后执行」两阶段工作流，相比父类的两点增强：

- 计划解析降级策略：优先解析 ```python 代码块格式，失败后用正则提取列表，最终兜底 `ast.literal_eval`；
- 支持 `custom_prompts` 自定义 `planner` / `executor` 提示词模板。

### 工具生态

- **MyCalculatorTool**：基于 Python `ast` 的安全表达式计算器；
- **MyAdvancedSearchTool**：多源搜索，自动探测 Tavily / SerpApi 可用性并择优返回；
- **ToolChainManager**：将「搜索 → 计算」等多个工具按模板串联执行，步骤间通过 `{output_key}` 引用上一步结果；
- **AsyncToolExecutor**：基于 `ThreadPoolExecutor` 并行执行多个工具任务。

## 依赖说明

- 核心依赖：`hello-agents`（框架）、`openai`（OpenAI 兼容客户端）、`python-dotenv`（环境变量）；
- 可选依赖：`serpapi>=1.1.0` 或 `tavily-python`（搜索工具，按需安装其一）。