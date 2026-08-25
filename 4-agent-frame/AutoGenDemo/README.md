# AutoGen 软件开发团队协作案例

> 基于 [Hello-Agents](https://hello-agents.datawhale.cc/) 第六章「框架开发实践」—— AutoGen 框架部分的学习实践。

## 项目简介

本项目使用微软 [AutoGen](https://github.com/microsoft/autogen) 框架（v0.4+），构建了一个由 4 个 AI 智能体组成的**虚拟软件开发团队**，通过多智能体协作完成软件开发任务。

程序启动后，4 个智能体会围绕一个开发需求（比特币价格显示应用），按照 **需求分析 → 编码实现 → 代码审查 → 用户验收** 的流程进行轮询讨论，直到任务完成。

## 智能体角色

| 智能体 | 角色 | 职责 |
|---|---|---|
| **ProductManager** | 产品经理 | 需求分析、功能模块划分、技术选型建议 |
| **Engineer** | 软件工程师 | 编写完整的可运行代码实现 |
| **CodeReviewer** | 代码审查员 | 审查代码质量、安全性、最佳实践 |
| **UserProxy** | 用户代理 | 模拟用户验收，判断是否满足需求 |

## 核心概念

- **RoundRobinGroupChat**：轮询式群聊，智能体按固定顺序依次发言
- **TextMentionTermination**：当某位智能体说出 `TERMINATE` 时自动结束对话
- **AssistantAgent**：由 LLM 驱动的智能体，通过 `system_message` 定义角色行为
- **OpenAIChatCompletionClient**：兼容 OpenAI API 的模型客户端，支持第三方模型服务

## 文件说明

```
AutoGenDemo/
├── autogen_software_team.py   # 主程序：多智能体团队协作
├── output.py                  # 智能体协作产出的比特币价格应用代码
├── .env                       # 环境变量配置（不提交到 Git）
├── .env_example               # 环境变量配置模板
├── requirements.txt           # Python 依赖
└── README.md                  # 本文件
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置环境变量

复制 `.env_example` 为 `.env`，填入你的模型配置：

```bash
cp .env_example .env
```

`.env` 配置项说明：

| 配置项 | 说明 | 示例 |
|---|---|---|
| `LLM_MODEL_ID` | 模型名称 | `deepseek-v4-flash-0731` |
| `LLM_API_KEY` | API 密钥 | `sk-xxx` |
| `LLM_BASE_URL` | API 地址（兼容 OpenAI 格式） | `https://your-api.com/v1` |
| `LLM_TIMEOUT` | 请求超时时间（秒） | `60` |

> 支持任何兼容 OpenAI API 格式的服务，如 OpenAI、DeepSeek、阿里云百炼等。
>
> 使用非 OpenAI 官方模型时，程序会通过 `model_info` 参数声明模型能力，无需额外配置。

### 3. 运行

```bash
python autogen_software_team.py
```

## 运行效果

程序启动后，终端会实时输出智能体之间的对话过程：

```
🔧 正在初始化模型客户端...
👥 正在创建智能体团队...
🚀 启动 AutoGen 软件开发团队协作...
============================================================
---------- TextMessage (user) ----------
我们需要开发一个比特币价格显示应用...

---------- TextMessage (ProductManager) ----------
【需求分析、功能模块划分、技术选型建议...】

---------- TextMessage (Engineer) ----------
【完整的 Streamlit 代码实现...】

---------- TextMessage (CodeReviewer) ----------
【代码审查意见和改进建议...】

---------- TextMessage (UserProxy) ----------
【验收结果...TERMINATE】
============================================================
✅ 团队协作完成！
```

## 注意事项

- 整个协作流程涉及多次 API 调用（4 个智能体 × 最多 20 轮），运行时间取决于 API 响应速度
- 如遇 API 限流（503 错误），程序会自动等待并重试（最多 3 次）
- 对话最多进行 20 轮，或直到某位智能体回复包含 `TERMINATE` 时结束
