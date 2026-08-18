# 1-simple-agent - 初识智能体

本项目是一个基于 ReAct 模式的简单智能体（Agent）示例，参考 [Datawhale Hello-Agents 教程](https://hello-agents.datawhale.cc/#/./chapter1/%E7%AC%AC%E4%B8%80%E7%AB%A0%20%E5%88%9D%E8%AF%86%E6%99%BA%E8%83%BD%E4%BD%93?id=_13-%e5%8a%a8%e6%89%8b%e4%bd%93%e9%aa%8c%ef%bc%9a5-%e5%88%86%e9%92%9f%e5%ae%9e%e7%8e%b0%e7%ac%ac%e4%b8%80%e4%b8%aa%e6%99%ba%e8%83%bd%e4%bd%93) 实现。

## 功能

智能体能够根据用户请求，自动调用工具完成任务。示例场景：
- 查询城市实时天气
- 根据天气推荐旅游景点

## 项目结构

```
1-simple-agent/
├── SimpleAgent.py              # 主程序入口
├── OpenAICompatibleClient.py   # LLM 客户端封装
├── get_weather.py              # 天气查询工具
├── get_attraction.py           # 景点推荐工具
├── config_example.json         # API 示例配置文件（需自行填写，填写完后重命名为 config.json）
└── prompt_config.txt           # 系统提示词配置
```

## 快速开始

### 1. 安装依赖

```bash
pip install openai requests tavily-python
```

### 2. 配置 API 密钥

编辑 `config.json`，填入你的 API 密钥：

```json
{
    "API_KEY": "YOUR_API_KEY",
    "BASE_URL": "YOUR_BASE_URL",
    "MODEL_ID": "YOUR_MODEL_ID",
    "TAVILY_API_KEY": "YOUR_TAVILY_API_KEY"
}
```

### 3. 运行

```bash
python SimpleAgent.py
```

## 工作原理

智能体采用 **ReAct（Reasoning + Acting）** 模式，通过循环执行以下步骤完成任务：

1. **Thought** - 思考当前情况，决定下一步行动
2. **Action** - 调用工具或输出最终答案（Finish）
3. **Observation** - 获取工具返回结果，继续下一轮循环
