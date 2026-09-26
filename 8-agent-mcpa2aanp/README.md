# 第 8 章《智能体通信协议：MCP、A2A 与 ANP》· 运行手册

对应教材第十章（chapter10）的本地代码目录：上游 22 个示例脚本 + `weather-mcp-server/` 发布样板，全部已在本机逐个实测。

| | |
|---|---|
| 上游教程 | <https://hello-agents.datawhale.cc/#/./chapter10/第十章%20智能体通信协议> |
| 上游代码 | <https://github.com/datawhalechina/hello-agents/tree/main/code/chapter10> |
| 本环境 | 项目根 `.venv`（Python 3.12.13，第 1–8 章共用）+ 本章专用依赖 `requirements.txt` |
| 外部依赖 | Node.js（npx 拉起社区 MCP 服务器）；LLM 走 `.env` 配置的 OpenAI 兼容端点 |
| 实测结论 | **14 个可执行脚本 12 个全绿**；`03`/`06` 只差一个 GitHub token；详见 [学习指南](学习指南_01-14.md) |

> 详细的逐文件分级、6 阶段学习顺序、11 条实测坑位 → **[学习指南_01-14.md](学习指南_01-14.md)**（本手册的权威扩展，先读它再跑代码）。

---

## 快速开始

```bash
# 0) 装本章依赖（在项目根 .venv 里，一次性）
../.venv/Scripts/python.exe -m pip install -r requirements.txt

# 1) 配 .env：复制 .env.example 填入 LLM_MODEL_ID / LLM_API_KEY / LLM_BASE_URL
#    （本仓库的 .env 直接复用第 7 章配置即可）

# 2) ⚠️ 本章最大坑：脚本会用 "python"/"npx" 字符串 spawn MCP 子进程，
#    子进程继承 PATH 而非当前解释器 —— 跑任何脚本前必须先把 venv 的 Scripts 顶进 PATH：
export PATH="<项目根>/.venv/Scripts:$PATH"   # Git Bash；验证 which python 指向 .venv

# 3) 第一个示例：三协议最小体验（纯本地，不耗 LLM token）
../.venv/Scripts/python.exe 01_TestConnect.py
```

Windows + PyCharm 用户注意两点（2026-09-20 实测踩坑，详见学习指南坑 11）：

- Run/Debug 配置里 **Working directory 必须设为本目录**（脚本里 `"."`、`my_README.md` 都是相对路径）；
- 子进程 spawn 的是 PATH 里的 `python`/`npx` —— 在 Environment variables 里把 `.venv\Scripts` 加到 `Path` 最前面。

## 文件清单

**MCP（10.2）**：`my_mcp_server.py`（FastMCP 全要素样板，先读它）→ `01_TestConnect.py` → `02_Connect2MCP.py`（裸 MCPClient 四次曲）→ `03_GitHubMCP.py`（需 GitHub token）→ `04_MCPTransport.py`（Memory/Stdio 传输对比）→ `05_UseMCPToolInAgent.py`（Agent 集成拐点）→ `06_MultiAgentDocumentAssist.py`（双 Agent 协作，产物见 `report.md`）

**A2A（10.3）**：`07_SimpleA2AAgent.py` / `08_CustomA2AAgent.py`（技能装饰器，进程内直调不走网络）→ `09_A2A_Server.py` + `09_A2A_Client.py`（双终端 HTTP 通信，端口 5000）→ `09_A2A_Network.py`（三服务流水线 5000–5002）→ `09_A2A_WithAgent.py`（手搓 A2ATool 封装，最佳教材）→ `10_A2ATool_Simple.py` / `10_CustomerService.py`（框架自带 A2ATool 版）/ `10_AgentNegotiation.py`（Agent 协商 7000/7001）

**ANP（10.4）**：`11_ANPInit.py`（注册/发现/组网，零依赖）→ `12_ANPTaskDistribution.py`（ANP+LLM 任务调度，全章 LLM 调协议工具质量最高）→ `13_ANPLoadBalancing.py`（负载均衡）

**自建服务器（10.5）**：`14_weather_mcp_server.py`（真实天气 stdio 服务器）→ `14_test_weather_server.py`（MCPClient 验证，wttr.in 实数据）→ `14_weather_agent.py demo`（Agent 全链路）

**其他**：`weather-mcp-server/` = 10.5.2 Smithery 发布物样板（HTTP 传输 + Dockerfile + smithery.yaml），**章内无脚本依赖它，只读**；`a2a_document_*.md` / `report.md` / `output.txt` / `my_README.md` = 运行产物或测试夹具。

## 本地改动说明（与上游的差异）

| 文件 | 改动 | 原因 |
|---|---|---|
| `05_UseMCPToolInAgent.py` | 补 `load_dotenv()`（锚定脚本目录 `.env`） | 上游未加载，构造 LLM 即抛异常 |
| `06_MultiAgentDocumentAssist.py` | `load_dotenv("../HelloAgents/.env")` → 脚本同目录 | 原路径是上游作者本机的死路径 |
| `09_A2A_WithAgent.py` | 补 `load_dotenv()` | 同 05 |
| `requirements.txt` | 新增（fastmcp 2.14.7 / a2a-sdk 0.3.26 / protobuf 5.29.6 / flask） | 本章 [protocols] extras 依赖 |

源码内均有 `# [本地修复]` 标注。**版本硬约束**：`a2a-sdk` 必须 `<1.0`（1.x 删了顶层 `A2AClient`，框架探测失败会误报"未安装"）；`fastmcp` 必须 `<3`；装完 a2a-sdk 需回钉 `protobuf~=5.29.3`（否则砸第 4 章 autogen-core）。

## 已知坑（速查，全 11 条见学习指南第五节）

1. **PATH 决定一切**：忘 `export PATH` → 子进程用 PATH 里的其他 python/npx，报误导性错误；
2. **端口互斥**：09 系 5000–5002、客服类 6000/6001、协商 7000/7001，上一个没退干净下一个 BindError；
3. **FastMCP 启动横幅很长**、且提示"3.0 coming / 4.0.4 available"——**别升级 fastmcp**；
4. 本章数据**不落盘**（ANP/A2A 全内存态），可放心反复跑，与第 6/7 章跨轮累积相反。
