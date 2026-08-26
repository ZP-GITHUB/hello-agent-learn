# 智能体框架开发实践

> 基于 [Hello-Agents](https://hello-agents.datawhale.cc/) 第六章「框架开发实践」的学习实践。

## 目录简介

本目录包含 **4 个主流智能体框架** 的实践项目，每个项目独立运行，覆盖不同的智能体应用场景：

| 框架 | 项目目录 | 场景 | 智能体数量 |
|---|---|---|---|
| **AgentScope** | `AgentScopeDemo/` | 三国狼人杀游戏 | 6 个 AI 智能体 |
| **AutoGen** | `AutoGenDemo/` | 虚拟软件开发团队 | 4 个 AI 智能体 |
| **CAMEL** | `CAMEL/` | 协作撰写数字电子书 | 2 个 AI 智能体 |
| **LangGraph** | `Langgraph/` | 智能搜索助手 | 三阶段工作流 |

## 项目概览

### 1. AgentScope —— 三国狼人杀

使用阿里巴巴 [AgentScope](https://github.com/modelscope/agentscope) 框架，6 位三国角色（刘备、关羽、曹操等）被随机分配身份，按照夜晚行动 → 白天讨论 → 投票淘汰的流程进行博弈，直到某一阵营达成胜利条件。

- **核心机制**：ReActAgent、MsgHub 消息广播、结构化输出（Pydantic）
- **运行入口**：`python AgentScopeDemo/main_cn.py`
- [详细文档 →](AgentScopeDemo/README.md)

### 2. AutoGen —— 虚拟软件开发团队

使用微软 [AutoGen](https://github.com/microsoft/autogen) 框架，4 个 AI 智能体组成开发团队，围绕开发需求按照需求分析 → 编码实现 → 代码审查 → 用户验收的流程轮询讨论。

- **核心机制**：RoundRobinGroupChat 轮询群聊、TextMentionTermination 终止条件
- **运行入口**：`python AutoGenDemo/autogen_software_team.py`
- [详细文档 →](AutoGenDemo/README.md)

### 3. CAMEL —— 数字电子书协作撰写

使用 [CAMEL](https://github.com/camel-ai/camel) 框架的角色扮演（RolePlaying）机制，两个 AI 智能体分别扮演"心理学家"和"作家"，通过多轮自主对话协作完成一本关于"拖延症心理学"的短篇电子书。

- **核心机制**：RolePlaying 双智能体协作、ModelFactory 模型工厂
- **运行入口**：`python CAMEL/DigitalBookWriting.py`
- [详细文档 →](CAMEL/README.md)

### 4. LangGraph —— 智能搜索助手

使用 [LangGraph](https://github.com/langchain-ai/langgraph) 框架构建状态图工作流，通过理解 → 搜索 → 回答的三阶段流水线，结合 Tavily API 完成真实的互联网搜索与回答生成。

- **核心机制**：StateGraph 状态图、InMemorySaver 会话检查点、astream 流式输出
- **运行入口**：`python Langgraph/Dialogue_System.py`
- [详细文档 →](Langgraph/README.md)

## 目录结构

```
4-agent-frame/
├── AgentScopeDemo/            # AgentScope 三国狼人杀
│   ├── main_cn.py             #   主程序：游戏主循环与流程控制
│   ├── prompt_cn.py           #   中文提示词模板
│   ├── game_roles.py          #   角色定义
│   ├── structured_output_cn.py #  结构化输出（Pydantic 模型）
│   ├── utils_cn.py            #   工具函数
│   └── requirements.txt
│
├── AutoGenDemo/               # AutoGen 虚拟开发团队
│   ├── autogen_software_team.py # 主程序：多智能体团队协作
│   ├── output.py              #   智能体协作产出的示例代码
│   └── requirements.txt
│
├── CAMEL/                     # CAMEL 数字电子书撰写
│   ├── DigitalBookWriting.py  #   主程序：角色扮演协作撰写
│   └── requirements.txt
│
├── Langgraph/                 # LangGraph 智能搜索助手
│   ├── Dialogue_System.py     #   主程序：搜索助手工作流
│   └── requirements.txt
│
└── README.md                  # 本文件
```

> 每个子项目目录中都包含 `.env`（实际配置，不提交到 Git）和 `.env_example`（配置模板）。

## 快速开始

### 通用配置

所有子项目共享相同的环境变量命名规范。每个子项目独立维护自己的 `.env` 文件，配置方式相同：

```bash
# 进入对应子项目目录
cd AgentScopeDemo   # 或 AutoGenDemo / CAMEL / Langgraph

# 从模板创建配置文件
cp .env_example .env
```

通用 `.env` 配置项：

| 配置项 | 说明 | 示例 |
|---|---|---|
| `LLM_MODEL_ID` | 模型名称 | `deepseek-v4-flash-0731` |
| `LLM_API_KEY` | API 密钥 | `sk-xxx` |
| `LLM_BASE_URL` | API 地址（兼容 OpenAI 格式） | `https://your-api.com/v1` |

> 支持任何兼容 OpenAI API 格式的服务，如 OpenAI、DeepSeek、阿里云百炼等。
> LangGraph 项目额外需要 `TAVILY_API_KEY`，可在 [tavily.com](https://tavily.com/) 免费获取。

### 运行各项目

每个子项目可独立运行，按需安装各自依赖：

```bash
# 1. AgentScope 三国狼人杀
pip install -r AgentScopeDemo/requirements.txt
python AgentScopeDemo/main_cn.py

# 2. AutoGen 虚拟开发团队
pip install -r AutoGenDemo/requirements.txt
python AutoGenDemo/autogen_software_team.py

# 3. CAMEL 数字电子书撰写
pip install -r CAMEL/requirements.txt
python CAMEL/DigitalBookWriting.py

# 4. LangGraph 智能搜索助手
pip install -r Langgraph/requirements.txt
python Langgraph/Dialogue_System.py
```

## 框架对比

| 特性 | AgentScope | AutoGen | CAMEL | LangGraph |
|---|---|---|---|---|
| **开发方** | 阿里巴巴 | 微软 | CAMEL-AI | LangChain |
| **核心模式** | 多智能体消息广播 | 轮询式群聊 | 双智能体角色扮演 | 状态图工作流 |
| **智能体数量** | 6 | 4 | 2 | 3 节点 |
| **适用场景** | 多人博弈 / 社交模拟 | 团队协作 / 软件开发 | 双人协作 / 知识创作 | 流水线任务 / RAG 搜索 |
| **会话管理** | 内置消息历史 | 群聊自动轮转 | 自动多轮对话 | InMemorySaver 检查点 |
| **输出约束** | Pydantic 结构化输出 | TERMINATE 终止机制 | CAMEL_TASK_DONE | TypedDict 状态定义 |

## 注意事项

- 各子项目的依赖版本相互独立，建议在同级虚拟环境（`.venv`）中统一安装
- 所有 `.env` 文件包含 API 密钥等敏感信息，已在 `.gitignore` 中排除
- 多智能体项目涉及大量 API 调用，运行时间取决于 API 响应速度和限流策略
- 各框架版本可能存在兼容性要求，详见各子项目的 README
