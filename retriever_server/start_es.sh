#!/bin/bash

set -euo pipefail

# 目录与用户设置（使用项目内目录，避免 /var 与 /etc）
BASE_DIR="$(cd "$(dirname "$0")" && pwd)"
DATA_DIR="$BASE_DIR/es_data"
LOG_DIR="$BASE_DIR/es_logs"
RUN_DIR="$BASE_DIR/es_run"
CONFIG_DIR="$BASE_DIR/es_config"
PID_FILE=$RUN_DIR/elasticsearch.pid
ES_BIN=/usr/share/elasticsearch/bin/elasticsearch
ES_USER=elasticsearch

# 准备目录与权限（需要 sudo；若容器无 sudo，请去掉 sudo 再执行）
mkdir -p "$DATA_DIR" "$LOG_DIR" "$RUN_DIR" "$CONFIG_DIR"
chown -R "$ES_USER":"$ES_USER" "$DATA_DIR" "$LOG_DIR" "$RUN_DIR" 2>/dev/null || true

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


