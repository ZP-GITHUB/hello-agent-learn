#!/usr/bin/env bash
# ============================================================================
# 第八章《记忆与检索》示例 —— Qdrant / Neo4j 数据库一键启动（WSL2 Docker）
# 用法：在 Git Bash 中执行  bash start-databases.sh
# 说明：容器已存在则启动，不存在则创建；末尾自动做连通性自检
# ============================================================================
set -uo pipefail

QDRANT_NAME=qdrant
NEO4J_NAME=neo4j
# Qdrant 服务端版本与 qdrant-client 保持同档，避免版本不兼容告警
# （框架依赖的 SearchRequest 在 client 1.16.0 被移除，故 client 锁 <1.16.0）
QDRANT_IMAGE="${QDRANT_IMAGE:-qdrant/qdrant:v1.15.1}"
NEO4J_PASSWORD="${NEO4J_PASSWORD:-hello-agents-password}"

# 统一通过 WSL 调用 docker（Windows 侧可能没有 docker CLI）
dex() { wsl.exe -e bash -lc "$1" 2>&1 | tr -d '\r'; }

echo "==> 检查 WSL 与 Docker"
dex "docker version --format '{{.Server.Version}}'" || { echo "❌ WSL 中 Docker 不可用"; exit 1; }

ensure_container() {
  local name="$1"; shift
  local run_cmd="$1"
  if dex "docker ps -a --format '{{.Names}}'" | grep -qx "$name"; then
    # 若已在运行但镜像与预期不一致，则重建（数据卷保留）
    local cur
    cur="$(dex "docker inspect -f '{{.Config.Image}}' $name")"
    local want
    want="$(echo "$run_cmd" | grep -oE '[a-z0-9./_-]+:[a-zA-Z0-9._-]+' | tail -1)"
    if [ -n "$want" ] && [ "$cur" != "$want" ]; then
      echo "    容器 $name 镜像为 $cur，与预期 $want 不一致，重建中"
      dex "docker rm -f $name" >/dev/null || true
      dex "$run_cmd"
      return
    fi
    echo "    容器 $name 已存在，尝试启动"
    dex "docker start $name" >/dev/null || true
  else
    echo "    创建容器 $name（首次会拉取镜像，可能较慢）"
    dex "$run_cmd"
  fi
}

echo
echo "==> [1/2] Qdrant ($QDRANT_IMAGE) 端口 6333 / 6334"
# --ulimit nofile=65535：Docker 默认容器 FD 上限仅 1024，Qdrant 多 worker（默认按 CPU 数）
# + 频繁客户端连接会把 FD 耗尽，表现为容器 Up 但所有请求 timed out、日志刷
# "Too many open files (os error 24)"。调大后根治（见 README 坑位 #19）
ensure_container "$QDRANT_NAME" \
  "docker run -d --name $QDRANT_NAME -p 6333:6333 -p 6334:6334 -v qdrant_storage:/qdrant/storage --restart unless-stopped --ulimit nofile=65535:65535 $QDRANT_IMAGE"

echo
echo "==> [2/2] Neo4j (7474 / 7687)  密码: $NEO4J_PASSWORD"
ensure_container "$NEO4J_NAME" \
  "docker run -d --name $NEO4J_NAME -p 7474:7474 -p 7687:7687 -e NEO4J_AUTH=neo4j/$NEO4J_PASSWORD -v neo4j_data:/data --restart unless-stopped neo4j:5"

echo
echo "==> 等待服务就绪（Neo4j 首次启动较慢，约 40-60 秒）"
q_ok=0; n_ok=0
for i in $(seq 1 40); do
  if [ "$q_ok" = 0 ]; then
    q=$(curl -s --max-time 3 http://localhost:6333/collections 2>/dev/null || true)
    if echo "$q" | grep -q "result"; then echo "    ✅ Qdrant 已就绪"; q_ok=1; fi
  fi
  if [ "$n_ok" = 0 ]; then
    # 注意：不要用 `curl -o /dev/null`，Git Bash 下 Windows 原生 curl.exe 写 /dev/null
    # 会返回退出码 23(CURLE_WRITE_ERROR)，把就绪判断带偏。这里改为解析 -w 输出的状态码
    nc="$(curl -s --max-time 3 -w '\n%{http_code}' http://localhost:7474/ 2>/dev/null | tail -1)"
    [ -z "$nc" ] && nc=000
    if [ "$nc" = "200" ]; then echo "    ✅ Neo4j 已就绪"; n_ok=1; fi
  fi
  [ "$q_ok" = 1 ] && [ "$n_ok" = 1 ] && break
  sleep 3
done
[ "$q_ok" = 0 ] && echo "    ⚠️ Qdrant 尚未响应，请查看: docker logs $QDRANT_NAME"
[ "$n_ok" = 0 ] && echo "    ⚠️ Neo4j 尚未响应，请查看: docker logs $NEO4J_NAME"

echo
echo "==> 连通性自检"
printf "  Qdrant  http://localhost:6333/collections -> "
curl -s --max-time 5 http://localhost:6333/collections || echo "❌ 无响应"
echo
printf "  Neo4j   http://localhost:7474 (浏览器登录)  容器状态: "
dex "docker inspect -f '{{.State.Status}}' $NEO4J_NAME"

echo
echo "------------------------------------------------------------"
echo "若 Windows 侧 Python 连不上 localhost，请改用 WSL 的 IP："
echo "    wsl.exe -e bash -lc \"hostname -I\""
echo "然后把 .env 中的 QDRANT_URL / NEO4J_URI 里的 localhost 换成该 IP。"
echo "------------------------------------------------------------"
echo
echo "当前容器列表："
dex "docker ps --filter name=qdrant --filter name=neo4j --format 'table {{.Names}}\t{{.Image}}\t{{.Status}}\t{{.Ports}}'"
