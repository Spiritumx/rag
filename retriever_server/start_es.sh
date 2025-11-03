#!/bin/bash

set -euo pipefail

# 目录与用户设置（使用项目内目录，避免 /var 与 /etc 与 /usr）
BASE_DIR="$(cd "$(dirname "$0")" && pwd)"
DATA_DIR="$BASE_DIR/es_data"
LOG_DIR="$BASE_DIR/es_logs"
RUN_DIR="$BASE_DIR/es_run"
CONFIG_DIR="$BASE_DIR/es_config"
DIST_DIR="$BASE_DIR/es_dist"
ES_VERSION="8.15.0"
ES_HOME="$DIST_DIR/elasticsearch-$ES_VERSION"
ES_BIN="$ES_HOME/bin/elasticsearch"
ES_USER=elasticsearch
PID_FILE=$RUN_DIR/elasticsearch.pid

# 准备目录与权限（需要 sudo；若容器无 sudo，请去掉 sudo 再执行）
mkdir -p "$DATA_DIR" "$LOG_DIR" "$RUN_DIR" "$CONFIG_DIR" "$DIST_DIR"
chown -R "$ES_USER":"$ES_USER" "$DATA_DIR" "$LOG_DIR" "$RUN_DIR" 2>/dev/null || true

# 若本地 ES 二进制不存在，则尝试自动下载
if [ ! -x "$ES_BIN" ]; then
  TGZ="elasticsearch-$ES_VERSION-linux-x86_64.tar.gz"
  URL="https://artifacts.elastic.co/downloads/elasticsearch/$TGZ"
  echo "Elasticsearch binary not found. Downloading $TGZ ..."
  curl -fSL "$URL" -o "$DIST_DIR/$TGZ"
  tar -xzf "$DIST_DIR/$TGZ" -C "$DIST_DIR"
fi

# 若配置不存在，生成最小配置，确保 path.data 与 path.logs 设置到项目目录
if [ ! -f "$CONFIG_DIR/elasticsearch.yml" ]; then
  cat > "$CONFIG_DIR/elasticsearch.yml" <<'EOF'
cluster.name: es-docker-cluster
node.name: es-node
discovery.type: single-node
xpack.security.enabled: false
xpack.security.enrollment.enabled: false
network.host: 0.0.0.0
http.port: 9200
transport.port: 9300
path.data: ./es_data
path.logs: ./es_logs
EOF
fi

# 已在运行则跳过
if [ -f "$PID_FILE" ] && ps -p "$(cat "$PID_FILE" 2>/dev/null)" >/dev/null 2>&1; then
  echo "Elasticsearch already running with PID $(cat "$PID_FILE")."
  exit 0
fi

# 后台启动（写入 PID 文件）
export ES_PATH_CONF="$CONFIG_DIR"
su -s /bin/bash -c "$ES_BIN -d -p $PID_FILE" "$ES_USER"

echo "Elasticsearch starting... PID file: $PID_FILE"

# 简单健康检查等待
for i in $(seq 1 60); do
  if curl -fsS http://127.0.0.1:9200/_cluster/health >/dev/null; then
    echo "Elasticsearch is up."
    exit 0
  fi
  echo "Waiting for Elasticsearch to start... ($i/60)"
  sleep 5
done

echo "Elasticsearch did not become healthy in time." >&2
exit 1


