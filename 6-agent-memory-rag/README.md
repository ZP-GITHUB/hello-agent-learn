# 第 6 章《记忆与检索》· 运行手册

hello-agents 第八章 11 个示例（`01`–`11`）的本地运行环境、配置与坑位记录。

| | |
|---|---|
| 上游教程 | <https://hello-agents.datawhale.cc/#/./chapter8/第八章%20记忆与检索> |
| 上游代码 | <https://github.com/datawhalechina/hello-agents/tree/main/code/chapter8> |
| 本环境 | 项目根 `.venv`（Python 3.12.13，第 1–6 章共用） |
| 模型资源 | `huggingface-models/`（约 1.3 GB，已从 C 盘 HF 缓存迁出，见 2.5） |
| 外部依赖 | Qdrant + Neo4j（WSL2 Docker）；LLM 走任意 OpenAI 兼容端点 |
| 已实测跑通 | `01`、`03`、`04`；LLM 走 `.env` 配置的 MiniMax 端点（`05`–`08` `10` `11` 同链路，未逐个跑） |

> 本手册的依赖关系不是照抄文档，而是逐个读取示例源码 + 框架源码（`hello_agents/memory/*`、`hello_agents/tools/builtin/{memory_tool,rag_tool}.py`）后确认的实际行为。

---

## 快速开始

```bash
# 1) 装依赖（在项目根执行，一次性覆盖第 1–6 章）
cd <项目根>                                          # 即 hello-agent-learn 所在目录
.venv/Scripts/python.exe -m pip install -r requirements.txt

# 2) 起数据库（WSL2 Docker：Qdrant + Neo4j）
bash 6-agent-memory-rag/start-databases.sh

# 3) 体检：库连通性 + 嵌入维度一致性（改完 .env 就跑它）
cd 6-agent-memory-rag
../.venv/Scripts/python.exe _check_env.py

# 4) 跑第一个示例 —— 纯内存，不需要数据库、不需要 API Key
../.venv/Scripts/python.exe 03_WorkingMemory_Implementation.py
```

**目录**

- [一、环境档位与依赖矩阵](#一环境档位与依赖矩阵)
- [二、环境准备](#二环境准备)
- [三、坑位清单](#三坑位清单18-条全部源码实测)
- [四、框架已知问题](#四框架已知问题实测确认)
- [五、本机实测结论](#五本机实测结论)
- [六、目录结构](#六目录结构)

---

## 一、环境档位与依赖矩阵

### 三档环境，按需选

| 档位 | 覆盖示例 | 需要什么 | 预估体积 |
|---|---|---|---|
| **A. 零依赖档** | `03`（工作记忆） | Python + `hello-agents[memory]`，**不需要数据库、不需要 API Key** | ~150 MB |
| **B. 记忆全功能档** | `01` `02` `06` `09` | A + **Qdrant + Neo4j**（Docker） | + ~800 MB 镜像 |
| **C. RAG 全功能档** | `04` `05` `07` `08` `10` `11` | B + **LLM API Key** + `rag` 依赖（torch / sentence-transformers / markitdown）+ `gradio` | + ~3–6 GB |

先跑通 A 档，再补 B、C，能避免一上来被 torch 体积和数据库连接问题劝退。

### 逐示例依赖矩阵（源码实测）

| 示例 | 记忆类型 / 工具 | Qdrant | Neo4j | LLM Key | 文档解析依赖 |
|---|---|---|---|---|---|
| `01_MemoryTool_Basic_Operations` | working + episodic + semantic + perceptual | ✅ | ✅ | — | — |
| `02_MemoryTool_Architecture` | 可配置；第二个 demo 用 working + semantic | ✅ | ✅ | — | — |
| `03_WorkingMemory_Implementation` | **仅 working（纯内存）** | ❌ | ❌ | ❌ | ❌ |
| `04_RAGTool_MarkItDown_Pipeline` | RAGTool | ✅ | ❌ | ✅ | markitdown / pypdf |
| `05_RAGTool_Advanced_Search` | RAGTool（MQE / HyDE） | ✅ | ❌ | ✅ | — |
| `06_Memory_Consolidation_Demo` | working + episodic + semantic + perceptual | ✅ | ✅ | — | — |
| `07_RAGTool_Intelligent_QA` | RAGTool | ✅ | ❌ | ✅ | — |
| `08_Agent_Tool_Integration` | MemoryTool(4 类型) + RAGTool + SimpleAgent | ✅ | ✅ | ✅ | — |
| `09_Memory_Types_Deep_Dive` | 四个独立 MemoryTool（每种记忆单独实例） | ✅ | ✅ | — | — |
| `10_RAG_Pipeline_Complete` | RAGTool（完整管道） | ✅ | ❌ | ✅ | markitdown（示例自带文本） |
| `11_Q&A_Assistant` | MemoryTool(默认 3 类型) + RAGTool + **Gradio** | ✅ | ✅ | ✅ | markitdown（需自备 PDF） |

**关键点：`MemoryTool` 全程不调用大模型**（框架源码里没有任何 LLM 引用），所以纯记忆类示例只要有数据库就能跑；只有 `RAGTool` 在初始化时构造 `HelloAgentsLLM`，没配 `LLM_*` 会直接初始化失败。

---

## 二、环境准备

### 2.1 Python 环境

> **当前状态（2026-09-15 起）：已统一为项目根的 `.venv`，本章不再单独建环境。**
>
> 第 1–6 章共用 `hello-agent-learn/.venv`（Python 3.12.13，204 个包），PyCharm 直接指向它即可，
> 无需按章切换解释器。清单在**项目根目录**：

| 文件 | 内容 |
|---|---|
| `requirements.txt` | 第 1、2、3、5、6 章全部依赖（含本章） |
| `requirements-frameworks.txt` | 仅第 4 章四个对比框架演示（AgentScope / AutoGen / CAMEL / LangGraph） |

```bash
cd <项目根>                                          # 即 hello-agent-learn 所在目录

# 一次性装好全部章节的依赖
.venv/Scripts/python.exe -m pip install -r requirements.txt
.venv/Scripts/python.exe -m pip install -r requirements-frameworks.txt   # 第 4 章演示才需要
```

**为什么必须 Python 3.12**：`numpy 2.5.3` 要求 `>=3.12`，`pandas 3.0.5` / `scikit-learn 1.9.1` 要求 `>=3.11`。项目根原先的 Python 3.10 环境装不上这三个包（实测 pip 直接报 `No matching distribution found for scikit-learn==1.9.1`），因此已整体重建。

**备用：只想给本章单独建环境**

本目录的 `requirements.txt` 保留为此用途，或直接用 `setup-env.sh`（会建本章独立 `.venv` + 装依赖 + 拉 spaCy 模型）。注意 `setup-env.sh` 走的是"单章独立环境"的老路径，与现在的统一环境不冲突但不能混用。

```bash
cd <项目根>/6-agent-memory-rag

# 1) 装 3.12（uv 会自动下载，单文件隔离，不污染系统 Python）
uv python install 3.12
uv venv --python 3.12 .venv
source .venv/Scripts/activate        # Git Bash

# 2) 按清单安装全部依赖（已内含 CPU 版 torch 与 qdrant-client<1.16 的约束）
.venv/Scripts/python.exe -m pip install -r requirements.txt
```

`requirements.txt` 已覆盖：框架本体、Qdrant/Neo4j 客户端、嵌入模型三件套、spaCy 中英模型、RAG 文档解析、gradio 界面。分组注释里标明了每一组服务于哪些示例、哪些可以裁掉。

**手工方式（想自己控制每一步时）**

```bash
# 1) 先装 CPU 版 torch，避免默认拉 CUDA 版（能省 2 GB+）
uv pip install torch --index-url https://download.pytorch.org/whl/cpu

# 2) 框架从 PyPI 装，锁 0.2.9；不要带 [all]，理由见下
uv pip install "hello-agents==0.2.9" gradio

# 3) ⚠️ 必须把 qdrant-client 压回 1.16 以下，否则记忆示例起不来（详见坑位 #1）
uv pip install "qdrant-client<1.16.0"
```

**为什么锁 `hello-agents==0.2.9` 且绝不能带 `[all]`（实测确认）：**

- PyPI 上 `hello-agents` **最新已是 1.0.0**，API 与书中示例不一致，直接装最新会报错；必须锁版本。
- `0.2.9` 的 `[all]` 会展开 `rl` 组（`trl` / `bitsandbytes` / `wandb` / `tensorboard` / `accelerate` / `peft` 等，约 1–2 GB，是第 11 章 RL 训练才用的）。更糟的是两个隐性冲突：`rl` 组要求 `torch>=2.0.0`（无 CPU 标记，会把 CPU 版 torch 覆盖成 CUDA 版），`evaluation` 组要求 `gradio<5.0.0`（会把 gradio 6.x 降级到 4.x）。
- `requirements.txt` 已显式列出 memory/rag 所需的全部依赖且版本更精确，所以不带任何 extras 即可。
- 注：本地如有 hello-agents 的 fork 仓库可编辑安装便于调试框架源码，但**分享依赖清单时不要写入本地路径**，统一用 PyPI 版本号表达。

可选（仅当用阿里云百炼的云端 Embedding）：

```bash
uv pip install dashscope
```

### 2.2 spaCy 语言模型（可选，不装只告警不报错）

```bash
python -m spacy download zh_core_web_sm
python -m spacy download en_core_web_sm
```

用于 `09` 示例的实体抽取。缺失时框架只打印 `⚠️ spaCy不可用，实体提取将受限`，不会中断。

> ⚠️ **国内直连 GitHub 极慢**（48 MB 的中文模型十几分钟下不完，反复 `ReadTimeoutError`）。
> 用代理前缀 + 官方 release 直链，实测 2 分钟下完，且 **sha256 与官方一致**：
>
> ```bash
> curl -sL -o zh_core_web_sm-3.8.0-py3-none-any.whl \
>   "https://ghproxy.net/https://github.com/explosion/spacy-models/releases/download/zh_core_web_sm-3.8.0/zh_core_web_sm-3.8.0-py3-none-any.whl"
> .venv/Scripts/python.exe -m pip install ./zh_core_web_sm-3.8.0-py3-none-any.whl
> ```
>
> **wheel 文件名必须规范**（`zh_core_web_sm-3.8.0-py3-none-any.whl`），改名成 `_zh.whl` 会报 `Invalid wheel filename`。

### 2.3 启动 Qdrant 与 Neo4j

```bash
# Qdrant（向量库）—— 版本与 qdrant-client<1.16.0 对齐
wsl.exe -e bash -lc "docker run -d --name qdrant -p 6333:6333 -p 6334:6334 \
  -v qdrant_storage:/qdrant/storage --restart unless-stopped qdrant/qdrant:v1.15.1"

# Neo4j（图库）—— 密码必须与 .env 的 NEO4J_PASSWORD 一致
wsl.exe -e bash -lc "docker run -d --name neo4j -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/hello-agents-password \
  -v neo4j_data:/data --restart unless-stopped neo4j:5"
```

验证：

```bash
curl http://127.0.0.1:6333/collections     # Qdrant，应返回 JSON
# 浏览器打开 http://127.0.0.1:7474 ，用 neo4j / hello-agents-password 登录
```

> ⚠️ **务必用 `127.0.0.1`，不要用 `localhost`**。WSL2 的端口转发只监听 IPv4，而 `localhost` 会先解析到 IPv6 的 `::1` 并一直等到超时（bolt 7687 稳定复现，HTTP 7474 却正常，极易误判为网络抖动）。详见坑位 #17。
>
> 国内拉镜像慢可先给 WSL 的 Docker 配镜像加速。

### 2.4 配置 `.env`

把本目录 `code/.env.example` 复制到**运行脚本所在目录**（示例用 `load_dotenv()` 读当前工作目录），或直接用本目录已预填好的 `.env` 模板。至少填这几项：

```ini
# LLM —— 任何 OpenAI 兼容端点即可（可用本机的 One API / sub2api 网关）
LLM_MODEL_ID=你的模型名
LLM_API_KEY=你的key
LLM_BASE_URL=http://127.0.0.1:3000/v1
LLM_TIMEOUT=60

# Qdrant（本地 Docker）—— 用 127.0.0.1，不要用 localhost
QDRANT_URL=http://127.0.0.1:6333
QDRANT_API_KEY=
QDRANT_COLLECTION=hello_agents_vectors
QDRANT_VECTOR_SIZE=384
QDRANT_DISTANCE=cosine
QDRANT_TIMEOUT=30

# Neo4j（本地 Docker）—— 用 127.0.0.1，不要用 localhost
NEO4J_URI=bolt://127.0.0.1:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=hello-agents-password
NEO4J_DATABASE=neo4j

# Embedding：dashscope(需key) / local(需torch，首跑下载模型)
EMBED_MODEL_TYPE=local
HF_ENDPOINT=https://hf-mirror.com
```

**Embedding 三档怎么选**（框架实际逻辑：`dashscope → local → tfidf` 逐级回退，只在初始化时探测一次）：

| 取值 | 需要 | 实测结论 |
|---|---|---|
| `dashscope`（默认） | `dashscope` 包 + `EMBED_API_KEY` | 阿里云百炼 `text-embedding-v3`。填了 `EMBED_BASE_URL` 则走 OpenAI 兼容 REST，可指向任意网关 |
| **`local`（推荐，本机在用）** | `torch` + `sentence-transformers` | 默认 `all-MiniLM-L6-v2`。**本机已把模型放到 `huggingface-models/` 并在 `EMBED_MODEL_NAME` 填本地路径**，离线加载、不联网、不占 C 盘（见 2.5）。若留空则用模型 ID，首跑会联网下载 ~90 MB，国内需配 `HF_ENDPOINT=https://hf-mirror.com`。**这是无 key 时唯一可用的方案** |
| ~~`tfidf`~~ | 仅 `scikit-learn` | ⚠️ **实测不可用**：`TFIDFEmbedding` 必须先 `fit()` 才能 `transform()`，而框架在语义记忆路径里从不调用 `fit()`，会持续报 `❌ 添加语义记忆失败: TF-IDF模型未训练，请先调用fit()方法` 且检索恒返回 0 条。**还会顺手把 Qdrant 集合建成 1000 维**（坑位 #18）。只适合做"能启动"的冒烟测试 |

> 实测在 `EMBED_MODEL_TYPE=tfidf` 下，Qdrant / Neo4j / spaCy 全部连通正常，但语义记忆的写入与检索全部失败——**回退链的最后一档是名义上的兜底，不能真用**。请配 `dashscope` 或 `local`。

### 2.5 本地模型资源（`huggingface-models/`，约 1.3 GB）

2026-09-15 已把系统默认的 HuggingFace 缓存目录（`~/.cache/huggingface/hub`）里的三个模型**扁平化迁移**到本目录（拆掉缓存特有的 blobs/symlink 结构，每个模型一个自包含目录），缓存盘已释放 1.26 GB。`from_pretrained()` 接受本地路径，`.env` 里已改用路径加载，**初始化不再联网解析模型 ID**（也顺带省掉了每次启动时对 HF Hub 的元数据校验请求，冷启动快约 20 秒）。

| 目录 | 上游模型 | 大小 | 作用 | 被谁使用 |
|---|---|---|---|---|
| `all-MiniLM-L6-v2/` | sentence-transformers/all-MiniLM-L6-v2 | 88 MB | **文本嵌入**：把文本编码成 384 维向量，语义相近的文本向量也相近。情景/语义/感知记忆的向量检索（Qdrant 相似度搜索）全靠它——没有它"搜记忆"就退化成关键词匹配 | `MemoryTool` 初始化即加载（`EMBED_MODEL_NAME`）；`RAGTool` 文档向量化 |
| `clip-vit-base-patch32/` | openai/clip-vit-base-patch32 | 581 MB | **图像编码**：把图片编码成 512 维向量，支持"以文搜图"的跨模态对齐 | 感知记忆 `modality="image"`（`CLIP_MODEL`，可缺省——缺失时自动降级为哈希向量，示例仍能跑） |
| `clap-htsat-unfused/` | laion/clap-htsat-unfused | 590 MB | **音频编码**：把音频编码成 512 维向量，支持"以文搜声" | 感知记忆 `modality="audio"`（`CLAP_MODEL`，可缺省，同上） |

三个模型共用一次加载、常驻内存（约 200 MB）；CLIP/CLAP 只在示例 `09` 用到图像/音频记忆时才加载。模型已入 `.gitignore`，不会提交。

**换机/重装时的恢复方式**（二选一）：

```bash
# 方式 A：整个目录随项目一起拷贝（推荐，零网络）
# 方式 B：从零下载 —— 保持 .env 里 EMBED_MODEL_NAME 留空、HF_ENDPOINT 配镜像，
#         跑一次示例 09 会自动把三个模型下到 C 盘默认缓存，再按迁移方法挪过来
```

> 提示：MiniLM 是英文为主的模型，中文内容能用但语义区分度打折（框架用 spaCy 中文模型做实体抽取来补短板）。若日后要换中文嵌入模型（如 `BAAI/bge-small-zh-v1.5`，512 维），**必须先删 Qdrant 集合**——维度变了，坑位 #18 同理。

### 2.6 建议的跑通顺序

```bash
cd code && python 03_WorkingMemory_Implementation.py   # 零依赖，先验证框架装对了
```

再按 A → B → C 顺序推进：`01` `02` `06` `09` → `04` `05` `07` `10` → `08` `11`

`11_Q&A_Assistant.py` 是**唯一带 Web 界面**的示例，启动后访问 <http://localhost:7860>：

```bash
cd 6-agent-memory-rag && ../.venv/Scripts/python.exe 11_Q\&A_Assistant.py
# 启动后浏览器打开 http://127.0.0.1:7860
```

页面操作顺序（四步，缺一步问答就返回"请先加载文档"）：**初始化助手 → 上传 PDF 并点"加载文档" → （可选）去"学习笔记"页记笔记 → 在"智能问答"页提问**。

运行前必须满足两个前提，否则会"看起来成功但答不出"：

1. **`markitdown[pdf]` 已装**（坑位 #20）——否则 PDF 被当二进制入库，界面上显示"✅ 加载成功"，但提问只能检索到乱码；
2. **gradio 版本与脚本代码匹配**（坑位 #21）——本仓库已按 gradio 6.27 修好，若自行降级 gradio 到 4.x 需把 `chat()` 改回列表格式。

语料自备即可：任何含文字层的 PDF 都能用（扫描件无文本层则提取不到内容）。本机目录里有一份 `Happy-LLM-0727.pdf` 可直接拿来做测试，但它属于第三方教材，**已加入 `.gitignore` 不随仓库分发**。

---

## 三、坑位清单（21 条，全部源码实测）

| # | 坑 | 现象 | 处理 |
|---|---|---|---|
| 19 | **Qdrant 容器文件描述符耗尽（假死）** | 容器 `Up` 但所有请求 `timed out`（先打"成功连接"再"连接失败"），WSL 内部 `curl localhost:6333` 也超时；日志持续刷 `Too many open files (os error 24)`。根因：Docker 默认容器 nofile 上限仅 **1024**，Qdrant 按 CPU 数起 worker（本机 27 个）+ 反复跑示例的连接堆积，几小时即耗尽 | 临时：`docker restart qdrant`。根治：run 时加 `--ulimit nofile=65535:65535`（`start-databases.sh` 已内置，2026-09-15 已按此重建，数据卷不丢）。验证：`docker exec qdrant sh -c 'ulimit -n'` |
| 1 | **`qdrant-client` 必须锁 <1.16.0** | PyPI 默认装到 1.19.0，而框架仍在 `from qdrant_client.http.models import ... SearchRequest`；该类在 1.16.0 被移除，导入异常被 `except ImportError` 静默吞掉，最终报**误导性的** `ImportError: qdrant-client未安装` | `pip install "qdrant-client<1.16.0"`（实测 1.15.1 可用） |
| 2 | 客户端与服务端版本差过大 | `UserWarning: Qdrant client version 1.15.1 is incompatible with server version 1.19.1` | 让服务端与客户端同档：镜像改用 `qdrant/qdrant:v1.15.1` |
| 3 | PyPI 最新是 1.0.0 | 装最新版后 `from hello_agents.tools import MemoryTool` 报错/行为不符 | 锁 `==0.2.9` |
| 4 | Qdrant 连不上**没有降级** | `❌ Qdrant连接失败` 后直接 `raise`，`MemoryTool` 初始化抛异常 | 先 `curl 127.0.0.1:6333/collections` 确认容器起来了 |
| 5 | Neo4j 密码不匹配 | Semantic 记忆初始化抛异常（同样无降级） | `NEO4J_AUTH` 的密码必须等于 `.env` 的 `NEO4J_PASSWORD` |
| 6 | Neo4j 启动慢 | 容器 `Up` 但 7474/7687 还没监听，立即跑脚本会连不上 | 首次启动等 40–60 秒；`start-databases.sh` 已内置双服务就绪轮询 |
| 7 | `tfidf` 嵌入不可用 | `❌ 添加语义记忆失败: TF-IDF模型未训练` | 改 `dashscope` 或 `local`，见 2.4 |
| 8 | 首次 `local` 嵌入卡住 | 在下载 HuggingFace 模型 | 设 `HF_ENDPOINT=https://hf-mirror.com`，或先手动预热 |
| 9 | **装了 `rag` extras 后会尝试下载 CLIP/CLAP** | 感知记忆初始化时 `CLIPModel.from_pretrained(...)`，`transformers` 已装即会联网拉模型（CLIP ~600 MB、CLAP ~1.5 GB） | 本机已本地化到 `huggingface-models/` 并在 `.env` 指定路径（见 2.5），不再触发下载；换新机器时设 `HF_HUB_OFFLINE=1` 可跳过（自动降级为哈希向量） |
| 10 | spaCy 模型缺失 | 仅告警，实体抽取受限 | 按 2.2 装 |
| 11 | RAGTool 没配 `LLM_*` | 所有调用返回 `❌ RAG工具未正确初始化，请检查配置` | 补 2.4 的 LLM 三项 |
| 12 | 相对路径产物到处生成 | 各示例在当前目录建 `./xxx_kb`、`memory_data/memory.db` | 已在 `.gitignore` 中忽略；建议分别在各自目录运行 |
| 13 | 示例 `03` 自身小瑕疵 | 三处 `{"action":"stats"}` 缺 `user_id`，打印 `❌ 参数验证失败：缺少必需的参数` | 上游示例问题，不影响主流程；加 `"user_id"` 即可 |
| 14 | uv 传 Git Bash 路径 | `/d/xxx` 被当成相对路径，在 `D:\d\xxx` 下乱建 venv | 脚本已用 `pwd -W` 转成 `D:/xxx` 再传给 uv |
| 15 | Windows 连 WSL 容器 | **`localhost` 对 Neo4j bolt 稳定失败**（HTTP 端口看似正常，容易误判为"偶发"） | 不是偶发，是 IPv6 解析问题，见坑位 #17：统一用 `127.0.0.1` |
| 16 | Windows 原生 `curl.exe` + Git Bash 的 `/dev/null` | `curl -o /dev/null` 输出正确但**退出码 23**（CURLE_WRITE_ERROR），把 `\|\| echo 000` 之类的兜底逻辑带偏，导致误判服务未就绪 | 别用 `-o /dev/null`；改成 `curl -s -w '\n%{http_code}' <url> \| tail -1` 解析状态码 |
| 17 | **`NEO4J_URI` 写 `localhost` 必挂** | `ServiceUnavailable: Timed out trying to establish connection to ResolvedIPv6Address(('::1', 7687))`，即使 HTTP 7474 能通；等待 15 秒后整体失败 | **改成 `bolt://127.0.0.1:7687`**。WSL2 的端口转发只监听 IPv4，`localhost` 会先解析到 `::1` 并吃满超时；`127.0.0.1` 实测 0.02 秒连通（neo4j driver 6.3.0 已验证）。`QDRANT_URL` 同理写 `http://127.0.0.1:6333` |
| 18 | **`tfidf` 兜底会污染 Qdrant** | tfidf 的 `max_features=1000` ⇒ 集合被建成 **1000 维**（正常 `local` 是 384 维）；换回 `local` 后框架只在集合**不存在**时才创建，于是复用这个 1000 维集合，写入 384 维向量直接失败 | 删集合重建：`curl -X DELETE http://127.0.0.1:6333/collections/hello_agents_vectors`（perceptual 的 3 个也要删）。注意 Qdrant 集合维度**创建后不可改**，只能删了重建 |
| 20 | **裸装 `markitdown` 缺 PDF 解析器，PDF 被当二进制乱码入库** | `add_document` 传 `.pdf` 时**不报错**，`load_document()` 返回"✅ 加载成功"，但 Qdrant 里存的 payload 是 `%PDF-1.4\n1 0 obj\n<< /Type /Catalog ...` 这种 PDF 源码，检索出来的片段是垃圾，问答自然答不出（示例 11 的主要入口就是 PDF） | 装带 extra 的版本：`pip install "markitdown[pdf]==0.1.7"`（会带上 `pdfplumber`）。**必须核对点**：markitdown 的 `_pdf_converter.py` 同时 import `pdfminer` / `pdfminer.high_level` / `pdfplumber` 三者，缺任一即抛 `MissingDependencyException`；框架 `_convert_to_markdown()` 捕获该异常后降级调用 `_fallback_text_reader()`（`open(path, errors='ignore')` 直读），因为读到的是非空字符串，整条链路**不会报警**。自检：把 PDF 加进库后查 payload，含 `%PDF-1.` 即命中此坑 |
| 21 | **示例 11 的 Gradio 代码是按 gradio 4.x 写的，与本机 6.27 不兼容** | 三个点：¹ `gr.Chatbot(bubble_full_width=...)` → `TypeError: unexpected keyword argument`（该参数 5.0 起移除）；² `gr.Blocks(theme=...)` → `UserWarning: 参数已移到 launch()`；³ `chat()` 用 `history.append([user, bot])` 旧列表格式 → 请求时抛 `Data incompatible with messages format`，问答直接不可用 | 已在本仓库修复：删 `bubble_full_width`、`theme` 移到 `demo.launch()`、`chat()` 改用 `{"role": ..., "content": ...}` 字典列表。**注意**：本仓库当前装的是 gradio 6.27，与 `requirements.txt` 已对齐；若降级 gradio 到 4.x 则字典格式又不兼容，二者只能选一边 |

---

## 四、框架已知问题（实测确认）

跑示例时容易把下面这些当成"自己代码写错了"，其实都是框架行为：

| 表现 | 机制 | 影响 |
|---|---|---|
| `search` 传 `min_importance` 不生效 | 四种记忆的 `retrieve()` 签名里都**没有**这个参数，被 `**kwargs` 吞掉后丢弃；episodic 读的是 `importance_threshold`，参数名对不上 | 低于阈值的结果照样返回，是真 bug |
| `limit=3` 时每种类型只返回 1 条 | `per_type_limit = max(1, limit // len(memory_types))`，`3 // 4 = 0` 被兜成 1 | "找到 3 条"= 3 个类型各 1 条，不是全局 top 3；想多给几条 `limit` 需 ≥8 |
| 搜索结果**不按相似度**排序 | 各类型内部算了相关度，但跨类型合并时 `sort(key=importance)`，相关度被丢弃 | 排序只反映重要性，不反映匹配度 |
| 摘要条数与实际库内数据对不上 | `get_stats()` 数的是**内存列表**，`retrieve()` 查的是 **Qdrant / SQLite 持久库**，两者口径不同 | 摘要显示 4 条，库里可能已累积几十条 |
| 反复运行后数据不断堆叠 | `consolidate` 把工作记忆搬进情景记忆时**不查重**，且 `importance *= 1.1`（所以会出现 0.77 这种数） | 要干净复现需先清库，见下方命令 |

**清库复现命令**

```bash
# Qdrant 集合
curl -X DELETE http://127.0.0.1:6333/collections/hello_agents_vectors
curl -X DELETE http://127.0.0.1:6333/collections/hello_agents_vectors_perceptual_text
# SQLite
rm 6-agent-memory-rag/memory_data/memory.db
# Neo4j（可选，实体节点会累积）
wsl.exe -e bash -lc "docker exec neo4j cypher-shell -u neo4j -p hello-agents-password 'MATCH (n) DETACH DELETE n'"
```

---

## 五、本机实测结论

2026-09-15 在本机实跑验证，记录如下：

| 验证项 | 结果 |
|---|---|
| Python | 3.12.13（项目根 `.venv`，第 1–6 章共用，204 个包）✅ |
| hello-agents | 0.2.9（PyPI）✅ |
| qdrant-client | 1.15.1（<1.16.0，避开坑位 #1）✅ |
| neo4j driver | 6.3.0；torch 2.14.0+cpu；sentence-transformers 6.0.1 ✅ |
| Qdrant 容器 | `qdrant/qdrant:v1.15.1`，`127.0.0.1:6333` 连通 ✅ |
| Neo4j 容器 | `neo4j:5`（5.26.30），`bolt://127.0.0.1:7687` 连通（0.02s）✅ |
| `EMBED_MODEL_TYPE=local` | `LocalTransformerEmbedding` 加载成功，**384 维**；模型经 `HF_ENDPOINT=https://hf-mirror.com` 下载 ✅ |
| 示例 `01`（四种记忆类型） | ✅ 跑通：Qdrant 写入向量、Neo4j 写入 Entity 节点、记忆整合/遗忘均正常 |
| 示例 `03`（纯内存工作记忆） | ✅ 跑通，无需数据库与 API Key |
| 示例 `04`（MarkItDown 管道） | ✅ 跑通：4 种格式文档统一转 md 入库，LLM 链路正常 |
| 示例 `07`（智能问答） | ✅ 跑通：单次 `ask` 14–26 秒（含 MQE/HyDE 两次 LLM 扩展 + 一次生成） |
| 示例 `11`（Gradio 问答助手） | ✅ 跑通：页面 HTTP 200、7 个 API 端点全在，`初始化→加载 PDF→提问→回顾→出报告` 五步闭环实测通过（检索 10.5s + 生成 6.7s，命中 PDF 原文相似度 0.714） |
| `markitdown[pdf]` | ⚠️ 原清单漏了 PDF extra，导致 PDF 静默乱码入库 —— 已补进 `requirements.txt`（坑位 #20） |
| gradio | 6.27.0；示例 `11` 原代码按 4.x 写，三处不兼容 —— 已修复（坑位 #21） |
| Windows → WSL2 容器 | ⚠️ 必须用 `127.0.0.1`，`localhost` 对 bolt 7687 稳定超时（坑位 #17） |

跑完 `01` 后 Qdrant 中的集合形态（可作为"配置正确"的对照）：

```
hello_agents_vectors                    size=384  ← 与 all-MiniLM-L6-v2 对齐
hello_agents_vectors_perceptual_text    size=384
hello_agents_vectors_perceptual_image   size=512  ← 未配 CLIP，降级哈希向量
hello_agents_vectors_perceptual_audio   size=512  ← 未配 CLAP，降级哈希向量
```

> ⚠️ 若之后下载了 CLIP/CLAP（或打开 `HF_HUB_OFFLINE=1` 切换编码器），image/audio 两个集合的维度会变化，需先删除旧集合，否则同样触发坑位 #18 的维度冲突。

**已验证 LLM 链路（2026-09-15）**：`04` 端到端跑通（10/10 文件、16 分块），LLM 走 `.env` 配置的 MiniMax OpenAI 兼容端点。`07`、`11` 也已在同一条 RAGTool 链路上实跑通过；`05`、`06`、`08`、`09`、`10` 与之共用同一批组件。

> 注意：`.env` 里的 MiniMax 是推理模型，`llm.invoke()` 的返回值**不过滤 `<think>` 标签**，思考过程会直接混进答案正文（示例 07/11 都能看到）。做预览截断（如 `answer[:300]`）或长度评分时要把这段算进去，否则结论会被带偏。

### 首次运行的耗时分布与日志读法

`01` 全流程实测 **40 秒**（退出码 0），耗时分布如下（用 `python -u` 逐行打时间戳测得）：

| 时刻 | 事件 | 说明 |
|---|---|---|
| 1.6s | SQLite 建表完成，`MemoryTool` 还在构造中 | 脚本自身的 `print` 到这里为止 |
| **1.6s → 6.3s** | **静默约 4.7 秒** | **不是卡住**：首次使用才触发的 `import torch` / `import sentence_transformers`，纯导入开销 |
| 6.3s | `No device provided, using cpu` | 未指定 device，默认 CPU。我们的 torch 是 CPU 版，属预期 |
| **7.6s** | 首次联网请求 HF Hub | 即使模型已缓存，`huggingface_hub` 仍会发一次元数据请求校验 |
| 7.7s | `Loading SentenceTransformer model from ...` | 开始加载权重（本地缓存 88 MB） |
| 11.1s | `Loading weights: 103/103` | 权重装载完成 |
| **11.1s → 17.5s** | **静默约 6.4 秒** | CPU 上首次推理的预热（线程池/算子初始化），无 GPU 可加速 |
| 17.6s | `✅ 成功连接到Qdrant服务` / `使用现有Qdrant集合` | 集合已存在则直接复用（**这就是坑位 #18 的触发点**） |
| 17.9s | `✅ 嵌入模型就绪，维度: 384` + Neo4j 连接 + 健康检查 | 语义记忆初始化完成 |
| 18.9s | 加载 spaCy 中/英模型 | |
| 40.1s | `🎉 演示完成` | |

**日志行序 ≠ 执行顺序**（重要）。框架用 `logging`（写 stderr，默认不缓冲），脚本用 `print`（写 stdout）。当 stdout 重定向或不是终端时，Python 会对它做**块缓冲**——实测同一份代码，缓冲模式下模型加载的日志会**排在 SQLite 那条 `print` 之前**，与真实顺序相反；PyCharm 控制台因为接近终端语义，顺序才恰好正确。**要准确判断先后，用 `python -u` 跑。**

下面这些看着可疑、实际全部无害：

| 输出 | 性质 |
|---|---|
| `No device provided, using cpu` | sentence-transformers 的 INFO，CPU 推理是预期行为 |
| `You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN ...` | 未配 `HF_TOKEN`，匿名请求速率受限。**只是提醒**。同一句出现两遍，是因为它同时经 `huggingface_hub` 与 root logger 输出 |
| `Loading weights: 100%\|...\| 3928.89it/s` / `Batches: 100%\|...` | tqdm 进度条，走 stderr |
| `[Qdrant] add_vectors start / upsert begin / upsert done` | 框架的调试级日志 |

---

## 六、目录结构

```
6-agent-memory-rag/
├── README.md                # 本文档
├── requirements.txt         # 本章依赖清单（备用；日常用项目根的同名文件）
├── .env                     # 实际生效的环境配置（已在 .gitignore 中）
├── .env.example             # 配置模板
├── .gitignore
├── huggingface-models/      # 本地模型目录（~1.3 GB，见 2.5；已在 .gitignore 中）
│   ├── all-MiniLM-L6-v2/        # 文本嵌入 384 维（记忆/RAG 检索核心）
│   ├── clip-vit-base-patch32/   # 图像编码 512 维（示例 09 感知记忆）
│   └── clap-htsat-unfused/      # 音频编码 512 维（示例 09 感知记忆）
├── _check_env.py            # 环境自检脚本（连通性 + 嵌入维度一致性）
├── setup-env.sh             # 一键：装 3.12 + 本章独立 venv + 依赖 + spaCy 模型
├── start-databases.sh       # 一键：WSL 里拉起 Qdrant + Neo4j 并做连通性自检
├── 01_…11_.py               # 上游 code/chapter8 全部示例
├── 学习指南_06-10.md         # 示例 06~10 的由浅到深学习指南（本仓库补充）
├── Happy-LLM-0727.pdf       # RAG 语料（示例 11 用；第三方教材，已在 .gitignore 中）
├── memory_data/             # 运行产物：SQLite 文档库（首次运行自动创建，已在 .gitignore 中）
└── *_kb/、knowledge_base/   # 运行产物：RAG 知识库目录（同上，已忽略）
```

环境与依赖清单在项目根：`.venv/`（Python 3.12.13）、`requirements.txt`、`requirements-frameworks.txt`。

### 环境自检脚本

`_check_env.py` 可一键体检 `.env` 配置，会依次验证 Qdrant 连通性、Neo4j 连通性、实际 embedding 实现与维度，并给出维度一致性结论：

```bash
cd 6-agent-memory-rag
../.venv/Scripts/python.exe _check_env.py
```

> 示例代码版权归 [datawhalechina/hello-agents](https://github.com/datawhalechina/hello-agents)（CC-BY-NC-SA-4.0），仅作本地学习运行使用。
