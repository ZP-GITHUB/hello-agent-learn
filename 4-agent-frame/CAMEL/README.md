# CAMEL 数字电子书协作撰写

> 基于 [Hello-Agents](https://hello-agents.datawhale.cc/) 第六章「框架开发实践」—— CAMEL 框架部分的学习实践。

## 项目简介

本项目使用 [CAMEL](https://github.com/camel-ai/camel) 框架（v0.2.75），基于其核心的 **RolePlaying（角色扮演）** 机制，让两个 AI 智能体分别扮演"心理学家"和"作家"，通过多轮自主对话协作完成一本关于"拖延症心理学"的短篇电子书。

程序启动后，两个智能体会围绕电子书创作任务，按照 **作家提出需求 → 心理学家提供专业内容 → 作家整合润色** 的模式自主推进，直到任务完成或达到轮次上限。

## 智能体角色

| 智能体 | 角色 | 职责 |
|---|---|---|
| **作家**（user_role） | 内容创作者 | 提出创作需求、整合内容、把控结构和风格 |
| **心理学家**（assistant_role） | 领域专家 | 提供拖延症相关的心理学知识、实证研究和改善建议 |

## 核心概念

- **RolePlaying**：CAMEL 的核心机制，创建双智能体协作环境，自动为两个角色生成系统提示词并驱动自主对话
- **ModelFactory**：模型工厂，根据平台类型统一创建大模型实例，支持多种 API 平台
- **OPENAI_COMPATIBLE_MODEL**：OpenAI 兼容模型平台类型，适用于任何遵循 OpenAI API 格式的第三方服务
- **CAMEL_TASK_DONE**：框架约定的任务完成标志，智能体认为任务完成时会在消息中插入此标记
- **print_text_animated**：逐字动画打印工具，模拟打字效果，提升终端交互体验

## 文件说明

```
CAMEL/
├── DigitalBookWriting.py   # 主程序：角色扮演协作撰写电子书
├── .env                    # 环境变量配置（不提交到 Git）
├── .env_example            # 环境变量配置模板
├── requirements.txt        # Python 依赖
└── README.md               # 本文件
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

> 支持任何兼容 OpenAI API 格式的服务，如 OpenAI、DeepSeek、阿里云百炼等。

### 3. 运行

```bash
python DigitalBookWriting.py
```

## 运行效果

程序启动后，终端会以彩色动画逐字输出两个智能体的对话过程：

```
协作任务:
创作一本关于"拖延症心理学"的短篇电子书...

具体任务描述:
（CAMEL 细化后的任务描述）

作家:
（蓝色文字）关于电子书结构和引言的创作思路...

心理学家:
（绿色文字）关于拖延症心理学的专业知识和研究引用...

作家:
（蓝色文字）整合内容并推进下一章节...

...
✅ 电子书创作完成！
总共进行了 15 轮协作对话
```

## 工作流程

```
启动程序
  ├── 加载 .env 配置，创建大模型实例
  ├── 定义协作任务提示词（电子书主题与要求）
  └── 初始化 RolePlaying 会话
        ↓
  循环对话（最多 30 轮）
  ├── 作家（user_role）发言 → 提出需求、整合内容
  ├── 心理学家（assistant_role）发言 → 提供专业内容
  ├── 检测 CAMEL_TASK_DONE 标志 → 完成则退出
  └── 未完成则将回复传入下一轮
        ↓
  输出轮次统计，程序结束
```

## 注意事项

- 整个协作流程涉及 2 个智能体的多轮对话，每轮产生 2 次 API 调用，运行时间取决于 API 响应速度
- 对话最多进行 30 轮（`chat_turn_limit`），或直到智能体输出包含 `CAMEL_TASK_DONE` 时提前结束
- 使用 `OPENAI_COMPATIBLE_MODEL` 平台类型，可灵活切换不同模型而无需修改代码
- `.env` 文件包含 API 密钥等敏感信息，已在 `.gitignore` 中排除，请勿提交到版本控制
