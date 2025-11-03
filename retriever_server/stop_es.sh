#!/bin/bash

set -euo pipefail

BASE_DIR="$(cd "$(dirname "$0")" && pwd)"
RUN_DIR="$BASE_DIR/es_run"
PID_FILE=$RUN_DIR/elasticsearch.pid
ES_VERSION="8.15.0"
DIST_DIR="$BASE_DIR/es_dist"
ES_HOME="$DIST_DIR/elasticsearch-$ES_VERSION"
ES_BIN="$ES_HOME/bin/elasticsearch"

# 优雅停止
if [ -f "$PID_FILE" ]; then
  PID=$(cat "$PID_FILE" 2>/dev/null || true)
  if [ -n "${PID:-}" ] && ps -p "$PID" >/dev/null 2>&1; then
    echo "Stopping Elasticsearch (PID $PID) ..."
    kill -TERM "$PID" || true
    for i in $(seq 1 30); do
      ps -p "$PID" >/dev/null 2>&1 || { echo "Stopped."; rm -f "$PID_FILE"; exit 0; }
      sleep 1
    done
    echo "Timeout. Force killing PID $PID ..."
    kill -KILL "$PID" || true
    rm -f "$PID_FILE"
    echo "Stopped (forced)."
    exit 0
  fi
fi

echo "PID 文件不存在或进程不在。尝试按进程名关闭。"
pkill -f "$ES_BIN" || true
echo "Stop signal sent."


