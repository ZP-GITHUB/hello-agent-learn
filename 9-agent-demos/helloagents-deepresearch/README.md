# 第 14 章《自动化深度研究智能体》· 运行手册

对应教材第十四章，本地代码目录 `9-agent-demos/helloagents-deepresearch/`，取自上游 `code/chapter14`，共 31 个文件（后端 17 + 前端 14）。2026-10-13 在本机完成前后端联调，真实联网跑通。

| | |
|---|---|
| 上游教程 | <https://hello-agents.datawhale.cc/#/./chapter14/第十四章%20自动化深度研究智能体> |
| 上游代码 | <https://github.com/datawhalechina/hello-agents/tree/main/code/chapter14> |
| 本环境 | 项目根 `.venv`（Python 3.12.13，与第 1–9 章共用；已补装 `fastapi` / `uvicorn[standard]` / `loguru` / `ddgs`） |
| 外部依赖 | Node.js ≥ 18（前端 Vite 6）；一个可用的 LLM 端点 + 一个可用的搜索后端 |
| 实测结论 | **前后端全链路跑通**：流式 138.5s / 5 任务全 `completed` / 报告 8751 字符；非流式 416.3s / 11451 字符。修复了 3 个上游 bug + 1 个环境缺失，详见 [学习指南](学习指南.md) |

> 逐文件四档分级、对应教材 14.1–14.7 的 7 阶段学习顺序、6 条实测坑、三个上游 bug 的定位与修复 → **[学习指南.md](学习指南.md)**（先读它再跑代码）。

---

## 快速开始

```bash
# 1) 依赖已在项目根 .venv 里（一次性）
cd <项目根>/9-agent-demos/helloagents-deepresearch/backend
# 若换机器需重装：../../.venv/Scripts/python.exe -m pip install -e .

# 2) 后端 .env（含密钥，已被根 .gitignore 排除）
#    至少填：LLM_PROVIDER / LLM_MODEL_ID / LLM_API_KEY / LLM_BASE_URL
#            SEARCH_API（duckduckgo 或 tavily）+ 对应 API Key
#    ⚠️ LLM_TIMEOUT 建议 300：报告撰写那一步耗时可达 60s+，默认 60 会静默超时
cd <项目根>/9-agent-demos/helloagents-deepresearch/backend/src
../../../.venv/Scripts/python.exe main.py          # 监听 0.0.0.0:8000

# 3) 前端（另开一个终端）
cd <项目根>/9-agent-demos/helloagents-deepresearch/frontend
npm install        # 首次
npm run dev        # http://localhost:5174
```

浏览器打开 <http://localhost:5174>，输入研究主题即可。前端只会请求 `VITE_API_BASE_URL`（默认 `http://localhost:8000`）的 `/research/stream`（SSE）。

Windows + PyCharm 用户注意（2026-10-13 实测）：

- **后端工作目录必须是 `backend/src`**：`main.py` 里是 `from config import ...`、`from agent import ...` 的平铺导入，在 `backend/` 下跑会 `ModuleNotFoundError`；
- 停止后端**不能只杀主进程**：`main.py` 用 `uvicorn.run(..., reload=True)`，会额外 fork 出 worker 子进程并继承监听套接字。只杀父进程后 8000 端口仍被占用（详见学习指南坑 1）。

## 文件清单

**后端 `backend/src/`（17 文件）**

- 入口与配置：`main.py`（FastAPI + `/healthz` `/research` `/research/stream` 三端点、CORS、启动日志）、`config.py`（`Configuration` + `SearchAPI` 枚举，环境变量读取）、`models.py`（`TodoItem` / `SummaryState` / `SummaryStateOutput` 三个 dataclass）、`utils.py`（`strip_thinking_tokens`、来源去重与格式化）、`prompts.py`（三个 Agent 的中文系统提示词 + 报告模板）、`agent.py`（**核心编排器** `DeepResearchAgent`）
- 服务层 `services/`：`planner.py`（规划，含 JSON 抽取）、`summarizer.py`（任务总结，同步 + 流式两套）、`reporter.py`（报告生成）、`search.py`（搜索调度，全局 `SearchTool(backend="hybrid")`）、`notes.py` / `tool_events.py` / `text_processing.py`（笔记指引、工具事件追踪、`[TOOL_CALL:...]` 清洗）

**前端 `frontend/`（14 文件，去 node_modules）**

- `src/App.vue`（**唯一大文件**：居中输入卡片 → 全屏左右分栏，任务清单 + 时间线 + 来源 + 工具调用 + 最终报告，全部逻辑内含）、`src/services/api.ts`（手写 SSE 读取器，按 `\n\n` 切事件，无 axios 依赖）、`src/main.ts` / `env.d.ts` / `style.css`、`vite.config.ts`（**端口 5174**）、`index.html`、两份 tsconfig、`package.json`

> 教材 14.1.2 画的目录树里有 `components/ResearchModal.vue`、`composables/useResearch.ts`、`workspace/`，**上游实际代码里并不存在**：UI 全部塞在 `App.vue` 单文件里，笔记目录名是 `notes/`。以代码为准。

## 本地改动说明（与上游的差异）

| 文件 | 改动 | 原因 |
|---|---|---|
| `backend/src/config.py` | 模块顶层新增 `load_dotenv()` | 上游整个后端**没有任何** `load_dotenv()`，`.env` 完全不生效 |
| `backend/src/agent.py` | `run()` 中显式迭代 `_execute_task(...)` 生成器 | **上游 bug 1**：只调用不迭代 → 非流式接口静默假成功 |
| `backend/src/agent.py` | `_create_tool_aware_agent()` 新增 `use_tools` 开关，`report_agent` 传 `False` | **上游 bug 3**：报告 Agent 有工具能力时正文会丢失 |
| `backend/src/services/planner.py` | `_extract_json_payload()` 改用 `JSONDecoder.raw_decode` 逐位置扫描 | **上游 bug 2**：`[TOOL_CALL:note:{...}]` 的花括号污染 JSON，随机只剩 1 个任务 |
| `backend/src/services/reporter.py` / `prompts.py` | 提示词改为"回复即最终报告、禁止工具调用"，任务总结改为内联提供 | 配合 bug 3 修复，落盘改由代码侧完成 |
| `backend/.env` / `frontend/.env.local` | 本地生成（含密钥，不入库） | 上游只提供 `.env.example` |
| 根 `.gitignore` | 新增 `/9-agent-demos/**/.env`、`node_modules/`、`__pycache__/`、`dist/`、`notes/` | 兜底，防止密钥与产物进公开仓库 |

源码内均有 `# [本地修复]` 标注。**别把 bug 1/2/3 当成"框架问题"改回去**——三处都是上游代码缺陷，修复前后行为差异见学习指南第五节。

## 已知坑（速查，全 6 条见学习指南第六节）

1. **`uvicorn.run(reload=True)` 会 fork worker**：停后端要连子进程一起杀，否则 8000 端口仍被占用（`netstat -ano | findstr :8000` 找 PID）；
2. **后端 cwd 必须是 `backend/src`**，否则平铺 import 直接失败；
3. **DuckDuckGo 在本机被网络阻断**（`duckduckgo.com` 超时，代理未开时 ddgs 所有后端全挂）→ 本机把 `SEARCH_API` 切到 `tavily` 并配 Key；
4. **`LLM_TIMEOUT` 默认 60 不够**：带三段上下文后报告撰写会超时并静默降级 → 已改 300；
5. **Vite 端口**：`vite.config.ts` 写死 5174，被占用会自动 +1（本次实际跑在 5175）；访问用 `localhost` 而非 `127.0.0.1`（Vite 默认绑 IPv6 `::1`）；
6. **`notes/` 是运行产物**（`NoteTool` 的 `NOTES_WORKSPACE`），每次研究会新增 md，已 gitignore，不要提交。
