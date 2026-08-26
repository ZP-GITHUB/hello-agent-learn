# LangGraph 智能搜索助手

> 基于 [Hello-Agents](https://hello-agents.datawhale.cc/) 第六章「框架开发实践」—— LangGraph 框架部分的学习实践。

## 项目简介

本项目使用 [LangGraph](https://github.com/langchain-ai/langgraph) 框架（v1.0.0a3），构建了一个基于 **状态图（StateGraph）** 的智能搜索助手。系统通过三阶段流水线——**理解 → 搜索 → 回答**——完成用户的查询请求：

1. **理解阶段**：利用 LLM 分析用户意图，提取精准的搜索关键词
2. **搜索阶段**：调用 [Tavily API](https://tavily.com/) 执行真实互联网搜索，获取实时数据
3. **回答阶段**：综合搜索结果，由 LLM 生成结构化、可引用的最终回答

程序启动后，用户可以交互式地输入各种问题，系统会实时展示每个阶段的处理进展，最终输出基于搜索结果的完整回答。

## 核心概念

- **StateGraph**：LangGraph 的核心结构，通过定义节点（Node）和边（Edge）构建有状态的工作流图
- **TypedDict 状态**：使用 `SearchState` 定义全局状态结构，各节点共享并逐步填充状态字段
- **add_messages reducer**：LangGraph 的消息合并机制，使消息列表在节点间传递时自动追加（而非覆盖）
- **InMemorySaver**：内置的内存检查点组件，通过 `thread_id` 实现多会话隔离和上下文持久化
- **astream 流式输出**：异步流式执行工作流，实时展示每个节点的输出，提升交互体验
- **降级策略**：当搜索 API 不可用时，自动降级为基于 LLM 自身知识回答，保证系统鲁棒性

## 工作流架构

```
START
  ↓
🧠 understand（理解节点）
  │  分析用户意图 → 提取搜索关键词
  ↓
🔍 search（搜索节点）
  │  调用 Tavily API → 获取搜索结果
  ↓
💡 answer（回答节点）
  │  综合搜索结果 → 生成最终回答（搜索失败时走降级路径）
  ↓
END
```

## 文件说明

```
Langgraph/
├── Dialogue_System.py    # 主程序：智能搜索助手工作流
├── .env                  # 环境变量配置（不提交到 Git）
├── .env_example          # 环境变量配置模板
├── requirements.txt      # Python 依赖
└── README.md             # 本文件
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

> `requirements.txt` 中指定了 `langgraph==1.0.0a3`（alpha 版本），如需使用稳定版可自行调整。

### 2. 配置环境变量

复制 `.env_example` 为 `.env`，填入你的模型配置：

```bash
cp .env_example .env
```

`.env` 配置项说明：

| 配置项 | 说明 | 示例 |
|---|---|---|
| `LLM_MODEL_ID` | 模型名称 | `gpt-4o-mini` |
| `LLM_API_KEY` | API 密钥 | `sk-xxx` |
| `LLM_BASE_URL` | API 地址（兼容 OpenAI 格式） | `https://your-api.com/v1` |
| `TAVILY_API_KEY` | Tavily 搜索 API 密钥 | `tvly-xxx` |

> LLM 支持任何兼容 OpenAI API 格式的服务，如 OpenAI、DeepSeek、阿里云百炼等。
> Tavily API Key 可在 [tavily.com](https://tavily.com/) 免费注册获取。

### 3. 运行

```bash
python Dialogue_System.py
```

## 运行效果

程序启动后，进入交互式循环，实时展示三阶段的处理过程：

```
🔍 智能搜索助手启动！
我会使用Tavily API为您搜索最新、最准确的信息
支持各种问题：新闻、技术、知识问答等
(输入 'quit' 退出)

🤔 您想了解什么: LangGraph 最新版本有什么新特性？

============================================================
🧠 理解阶段: 我理解您的需求：用户想了解 LangGraph 的最新版本和新特性...
🔍 正在搜索: LangGraph latest version new features
🔍 搜索阶段: ✅ 搜索完成！找到了相关信息，正在为您整理答案...

💡 最终回答:
根据搜索结果，LangGraph 最新版本的主要新特性包括：
1. 增强的状态图功能...
2. 改进的检查点机制...
...
============================================================
```

## 工作流程

```
用户输入问题
  │
  ├── understand 节点
  │   ├── 提取最后一条 HumanMessage
  │   ├── LLM 分析意图 → 输出需求总结 + 搜索关键词
  │   └── 解析 "搜索词：" 字段（兜底使用原始输入）
  │
  ├── search 节点
  │   ├── 调用 Tavily API（search_depth=basic, max_results=5）
  │   ├── 优先提取 AI 综合答案（answer）
  │   ├── 拼接前 3 条具体结果（标题 + 摘要 + URL）
  │   └── 异常时标记 search_failed，不中断流程
  │
  └── answer 节点
      ├── 搜索成功 → 基于搜索结果 + 用户问题生成结构化回答
      └── 搜索失败 → 降级为 LLM 自身知识回答
```

## 注意事项

- 每次查询涉及 3 个节点的串行 API 调用（至少 2 次 LLM 调用 + 1 次 Tavily 调用），响应速度取决于各 API 延迟
- 搜索失败时系统不会崩溃，而是自动降级为基于 LLM 知识回答，但回答时效性会降低
- `InMemorySaver` 仅在内存中保存会话状态，程序重启后会话上下文将丢失
- Tavily API 免费版有每日调用次数限制，频繁测试时请注意配额
- `.env` 文件包含 API 密钥等敏感信息，已在 `.gitignore` 中排除，请勿提交到版本控制
