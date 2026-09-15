#!/usr/bin/env bash
# ============================================================================
# 第八章《记忆与检索》示例 —— Python 环境一键安装
# 用法：在 Git Bash 中执行  bash setup-env.sh
# 说明：使用 uv 创建独立 .venv，不污染系统 Python；可重复执行
# ============================================================================
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# ⚠️ uv 是 Windows 原生程序，必须喂 Windows 风格路径（D:/xxx），
#    否则它会把 /d/xxx 当成相对路径，在 D:\d\xxx 下乱建目录
HERE_WIN="$(cd "$HERE" && pwd -W)"
FRAMEWORK_DIR="${FRAMEWORK_DIR:-$(cd "$HERE/.." && pwd -W)}"
PY_VER="${PY_VER:-3.12}"

echo "==> 工作目录: $HERE_WIN"
echo "==> 框架目录: $FRAMEWORK_DIR"
echo "==> Python 版本: $PY_VER"

command -v uv >/dev/null 2>&1 || { echo "❌ 未找到 uv，请先安装：pip install uv"; exit 1; }

# 1) 安装指定版本 Python（已存在则跳过）
echo
echo "==> [1/5] 准备 Python $PY_VER"
uv python install "$PY_VER"

# 2) 创建虚拟环境（--seed 会装上 pip，后面 python -m spacy download 需要它）
echo
echo "==> [2/5] 创建虚拟环境 .venv"
if [ ! -f "$HERE_WIN/.venv/Scripts/python.exe" ]; then
  uv venv --seed --python "$PY_VER" "$HERE_WIN/.venv"
else
  echo "    已存在，跳过"
fi
PY="$HERE_WIN/.venv/Scripts/python.exe"
"$PY" -V

# 3) 先装 CPU 版 torch，避免默认拉取 CUDA 版（可省 2GB+）
echo
echo "==> [3/5] 安装 CPU 版 torch"
uv pip install --python "$PY" torch --index-url https://download.pytorch.org/whl/cpu

# 4) 安装框架（可编辑模式，指向本地 0.2.9 仓库）+ 第八章需要的 extras
echo
echo "==> [4/5] 安装 hello-agents[memory,rag] + gradio"
uv pip install --python "$PY" -e "$FRAMEWORK_DIR[memory,rag]" gradio \
  || {
    echo "⚠️  可编辑安装失败，改为从 PyPI 安装锁定版本 0.2.9"
    uv pip install --python "$PY" "hello-agents[memory,rag]==0.2.9" gradio
  }

# 4b) ⚠️ 关键：qdrant-client 必须 <1.16.0
#     1.16.0 移除了 SearchRequest，而框架仍在导入它；导入失败被 except ImportError
#     静默吞掉，最终报出误导性的 "qdrant-client未安装"
echo
echo "==> [4b/5] 锁定 qdrant-client<1.16.0"
uv pip install --python "$PY" "qdrant-client<1.16.0"

# 5) spaCy 语言模型（可选，失败不影响示例运行，只影响实体抽取质量）
echo
echo "==> [5/5] 下载 spaCy 语言模型（可选）"
"$PY" -m spacy download zh_core_web_sm || echo "⚠️  zh_core_web_sm 安装失败，可跳过"
"$PY" -m spacy download en_core_web_sm || echo "⚠️  en_core_web_sm 安装失败，可跳过"

# 生成 .env
if [ ! -f "$HERE/.env" ]; then
  cp "$HERE/.env.example" "$HERE/.env"
  echo
  echo "✅ 已生成 $HERE/.env —— 请填入 LLM_MODEL_ID / LLM_API_KEY / LLM_BASE_URL"
fi

echo
echo "============================================================"
echo "✅ 环境安装完成"
echo "下一步："
echo "  bash start-databases.sh                                   # 启动 Qdrant + Neo4j"
echo "  source .venv/Scripts/activate && cd code && python 03_WorkingMemory_Implementation.py"
echo "============================================================"
