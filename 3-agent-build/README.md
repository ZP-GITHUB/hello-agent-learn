# 3-Agent-Build: ReAct 智能体实现

## 项目简介

本项目实现了基于 **ReAct（Reasoning and Acting）** 模式的智能体，能够调用外部工具（如搜索引擎）来回答需要实时信息的问题。

ReAct 模式的核心思想是：**思考（Thought）→ 行动（Action）→ 观察（Observation）** 的循环，直到获得最终答案。

---

## 文件结构

```
3-agent-build/
├── .env              # 环境变量配置文件（API 密钥等，不要提交到 Git）
├── .env_example      # 环境变量配置示例
├── ReAct.py          # ReAct 智能体主程序
├── llm_client.py     # LLM 客户端封装（支持 OpenAI 兼容接口）
└── tools.py          # 工具定义与执行器（SerpApi 搜索引擎）
```

---

## 依赖安装

```bash
pip install openai python-dotenv serpapi
```

---

## 配置说明

复制 `.env_example` 为 `.env`，并填入你的配置：

```env
LLM_API_KEY="你的API密钥"
LLM_MODEL_ID="你的模型ID"
LLM_BASE_URL="你的API地址"
SERPAPI_API_KEY="你的SerpApi密钥"
```

---

## 使用方法

### 运行 ReAct 智能体

```bash
python ReAct.py
```

### 运行工具测试

```bash
python tools.py
```

### 运行 LLM 客户端测试

```bash
python llm_client.py
```

---

## 核心组件说明

### 1. HelloAgentsLLM（llm_client.py）

封装了 OpenAI 兼容的 LLM 客户端，支持流式输出。

**关键特性：**
- 支持任何兼容 OpenAI 接口的服务（OpenAI、DeepSeek、Ollama 等）
- 默认启用流式响应（`stream=True`），实现"打字机"效果
- 参数优先级：传入参数 > `.env` 环境变量

### 2. ToolExecutor（tools.py）

工具执行器，负责管理和执行外部工具。

**当前支持的工具：**
- `Search`: 基于 SerpApi 的网页搜索引擎

**扩展方法：**
```python
tool_executor.registerTool("工具名", "工具描述", 工具函数)
```

### 3. ReActAgent（ReAct.py）

ReAct 智能体核心实现，包含以下关键方法：

| 方法 | 功能 |
|------|------|
| `run(question)` | 执行 ReAct 循环，返回最终答案 |
| `_parse_output(text)` | 从 LLM 输出中解析 Thought 和 Action |
| `_parse_action(action_text)` | 解析 Action 格式：`tool_name[tool_input]` |
| `_parse_action_input(action_text)` | 从 Finish 指令中提取最终答案 |

---

## ReAct 工作流程

```
用户问题
   ↓
┌─────────────────────────────────────
│  第 N 步                             │
│  1. 构造 Prompt（包含工具描述+历史）  │
│  2. 调用 LLM 获取响应                │
│  3. 解析 Thought 和 Action           │
│  4. 判断 Action 类型：               │
│     - Finish[答案] → 返回最终答案     │
│     - Tool[输入]   → 执行工具         │
│  5. 记录 Action 和 Observation       │
─────────────────────────────────────┘
   ↓
达到最大步数或获得答案
```

---

## 已知问题与解决方案

### 问题：模型输出过长导致无限循环

#### 问题现象

运行 `ReAct.py` 时，程序陷入无限循环，控制台不断输出内容，无法自动结束。最终需要手动按 `Ctrl+C` 中断程序。

**控制台输出特征：**
```
--- 第 1 步 ---
🧠 正在调用 deepseek-v4-flash-0731 模型...
✅ 大语言模型响应成功:
Thought: 用户询问华为最新手机型号及主要卖点...
Action: Search[华为最新款手机 2026年发布 主要卖点]  
Need wait for tool result. But as AI in this environment...
...（大量自我对话和推理内容）...
Ok.
Ok.
Ok.
...（数千个 "Ok."）...
```

#### 根本原因

**1. 模型行为问题**

模型在输出 `Action: Search[...]` 后没有停止生成，而是继续输出了大量的内部推理内容（包括自我对话、犹豫、多次尝试输出 `Finish` 但又自我否定），最终陷入重复输出 "Ok." 的循环。

**2. 代码解析逻辑缺陷**

`_parse_output` 方法中的正则表达式存在问题：

```python
# 问题代码
action_match = re.search(r"Action:\s*(.*?)$", text, re.DOTALL)
```

这个正则表达式会匹配从**第一个 `Action:`** 到**文本末尾**的所有内容。由于模型输出了大量后续内容，`action` 变量实际上包含了：

```
Search[华为最新款手机 2026年发布 主要卖点]  
Need wait for tool result. But as AI in this environment...
...（数千行的推理内容）...
Action: Finish[...]
...（更多内容）...
Ok.
Ok.
...（数千个 Ok.）
```

**3. 循环机制**

由于 `action` 以 `Search` 开头而不是 `Finish`，代码会执行搜索工具，然后将超长的 `action` 存入 `history`。下一步模型会收到包含超长历史的 prompt，再次返回超长文本，形成无限循环。

#### 解决方案

**方案 1：修复解析逻辑（代码层面）**

修改 `_parse_output` 方法，只提取第一个 `Action:` 字段的内容：

```python
# 修改前
action_match = re.search(r"Action:\s*(.*?)$", text, re.DOTALL)

# 修改后
action_match = re.search(r"Action:\s*(.*?)(?=\nThought:|\nAction:|$)", text, re.DOTALL)
```

**改进说明：**
- 使用前瞻断言 `(?=\nThought:|\nAction:|$)` 限制匹配范围
- 只匹配到下一个 `Thought:`、`Action:` 或文本末尾
- 即使模型继续输出大量内容，也能正确提取第一个 Action

**方案 2：增强提示词（引导层面）**

在 `REACT_PROMPT_TEMPLATE` 中添加强调停止输出的说明：

```
重要：一旦你输出了 Action: 字段，必须立即停止生成，不要继续输出任何内容！
```

**改进说明：**
- 明确告知模型在输出 `Action:` 后必须停止
- 从提示词层面减少模型过度生成的可能性

#### 预防措施

如果未来再次遇到类似问题，可以考虑：

1. **添加输出长度限制**：在 `llm_client.py` 的 `think` 方法中添加最大长度限制
2. **添加超时机制**：为整个 ReAct 循环添加超时控制
3. **监控 history 大小**：如果 history 过长，主动截断或终止循环

---

## 注意事项

1. **API 密钥安全**：`.env` 文件包含敏感信息，不要提交到 Git 仓库
2. **模型选择**：建议使用指令遵循能力较强的模型（如 GPT-4、Claude 等）
3. **步数限制**：默认最大步数为 5，可根据问题复杂度调整
4. **工具扩展**：可以在 `tools.py` 中添加更多工具（如计算器、数据库查询等）

---

## 参考资料

- [ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629)
- [OpenAI API Documentation](https://platform.openai.com/docs)
- [SerpApi Documentation](https://serpapi.com/docs)
