# 3-agent-build：智能体经典范式构建

> 本模块是 [Hello-Agents](https://hello-agents.datawhale.cc/) 第四章的学习实践项目，实现了三种经典的 LLM 智能体（Agent）范式：**ReAct**、**Reflection** 和 **Plan-and-Solve**。

## 📖 项目概述

本项目基于 Datawhale 的 Hello-Agents 教程，动手实现了三种智能体设计模式。每种模式代表了不同的推理策略，帮助你理解如何构建一个能"思考"、"行动"和"自我改进"的 AI 智能体。

| 范式 | 核心思想 | 对应文件 |
|------|---------|---------|
| **ReAct** | 推理（Reasoning）与行动（Acting）交替进行，通过调用外部工具获取实时信息 | `ReAct.py` |
| **Reflection** | 智能体自我反思、自我改进，通过"生成 → 评审 → 优化"的迭代循环提升输出质量 | `Reflection.py` |
| **Plan-and-Solve** | 先将复杂问题拆解为有序步骤，再逐步执行，实现"先规划，后执行" | `Plan_and_solve.py` |

## 📁 项目结构

```
3-agent-build/
├── llm_client.py          # LLM 客户端封装（兼容 OpenAI 接口，支持流式响应）
├── tools.py               # 工具管理模块（ToolExecutor 工具执行器 + SerpApi 搜索工具）
├── ReAct.py               # ReAct 智能体实现
├── Reflection.py          # Reflection 智能体实现
├── Plan_and_solve.py      # Plan-and-Solve 智能体实现
├── .env                   # 环境变量配置（API 密钥等，已加入 .gitignore）
├── .env_example           # 环境变量示例文件
└── README.md              # 本文件
```

## 🔧 环境准备

### 1. 安装依赖

```bash
pip install openai python-dotenv serpapi
```

### 2. 配置环境变量

复制 `.env_example` 为 `.env`，并填入你的 API 密钥：

```bash
cp .env_example .env
```

`.env` 文件需要配置以下变量：

| 变量名 | 说明 | 使用模块 |
|--------|------|---------|
| `LLM_API_KEY` | LLM 服务的 API 密钥 | 全部 |
| `LLM_MODEL_ID` | 模型名称（如 `gpt-4o`、`deepseek-chat`） | 全部 |
| `LLM_BASE_URL` | LLM 服务的 API 地址 | 全部 |
| `SERPAPI_API_KEY` | [SerpApi](https://serpapi.com/) 搜索 API 密钥 | ReAct |

## 🚀 运行示例

### ReAct 智能体

ReAct（Reasoning + Acting）让 LLM 在推理过程中交替进行思考和工具调用，适合需要实时信息的问答场景。

```bash
python ReAct.py
```

默认会查询"华为最新的手机是哪一款？它的主要卖点是什么？"，通过 SerpApi 搜索引擎获取实时信息并给出回答。

**工作流程：**
```
Thought → Action(工具调用) → Observation → Thought → Action(Finish) → 最终答案
```

### Reflection 智能体

Reflection 通过"生成 → 反思 → 优化"的迭代循环，让智能体自我审查并改进输出。

```bash
python Reflection.py
```

默认任务是"编写一个 Python 函数，找出 1 到 n 之间所有的素数"，智能体会先写出初始代码，然后通过自我评审不断优化算法效率。

**工作流程：**
```
初始代码 → 反思评审 → 优化代码 → 再反思 → 再优化 → ... → 无需改进 → 输出最终代码
```

### Plan-and-Solve 智能体

Plan-and-Solve 将复杂问题先拆解为多个子步骤，再逐步执行，适合多步推理类任务。

```bash
python Plan_and_solve.py
```

默认会求解一道多步数学应用题，Planner 负责拆解步骤，Executor 负责逐步求解。

**工作流程：**
```
用户问题 → Planner（拆解步骤） → Executor（逐步执行） → 最终答案
```

## 🏗️ 核心模块说明

### `llm_client.py` — LLM 客户端

封装了 `HelloAgentsLLM` 类，兼容所有 OpenAI 接口的服务（OpenAI、DeepSeek、Ollama 等），支持流式响应（打字机效果）。

### `tools.py` — 工具模块

- **`ToolExecutor`**：工具管理器，负责工具的注册、查询和描述生成。
- **`search`**：基于 SerpApi 的网页搜索工具，智能解析搜索结果，优先返回直接答案。

## 📚 参考资料

- [Hello-Agents 教程 - 第四章：智能体经典范式构建](https://hello-agents.datawhale.cc/#/./chapter4/第四章%20智能体经典范式构建)
- [ReAct 论文](https://arxiv.org/abs/2210.03629)：Synergizing Reasoning and Acting in Language Models
- [Plan-and-Solve 论文](https://arxiv.org/abs/2305.04091)：Plan-and-Solve Prompting
