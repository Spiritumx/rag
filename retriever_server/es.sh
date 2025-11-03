#!/bin/bash

set -euo pipefail

# 配置：全部使用仓库内目录与本地发行包，不使用 /usr /etc /var
BASE_DIR="$(cd "$(dirname "$0")" && pwd)"
ES_VERSION="8.15.2"
DIST_DIR="$BASE_DIR/es_dist"
ES_HOME="$DIST_DIR/elasticsearch-$ES_VERSION"
ES_BIN="$ES_HOME/bin/elasticsearch"
CONFIG_DIR="$BASE_DIR/es_config"
DATA_DIR="$BASE_DIR/es_data"
LOG_DIR="$BASE_DIR/es_logs"
RUN_DIR="$BASE_DIR/es_run"
PID_FILE="$RUN_DIR/elasticsearch.pid"
TGZ="elasticsearch-$ES_VERSION-linux-x86_64.tar.gz"
URL="https://artifacts.elastic.co/downloads/elasticsearch/$TGZ"
ES_USER="elasticsearch"

usage() {
  cat <<EOF
用法: $(basename "$0") <install|start|stop|status>

子命令:
  install  下载并解压 ES，生成最小配置与 JVM 设置（不启动）
  start    启动 ES（如缺资源给出提示），并进行健康检查
  stop     优雅停止 ES（超时强制关闭）
  status   显示运行状态与 PID
EOF
}

ensure_dirs() {
  mkdir -p "$DIST_DIR" "$CONFIG_DIR" "$DATA_DIR" "$LOG_DIR" "$RUN_DIR"
}

cmd_install() {
  ensure_dirs

  if [ ! -x "$ES_BIN" ]; then
    echo "Downloading Elasticsearch $ES_VERSION ..."
    curl -fSL "$URL" -o "$DIST_DIR/$TGZ"
    tar -xzf "$DIST_DIR/$TGZ" -C "$DIST_DIR"
  else
    echo "Elasticsearch binary exists: $ES_BIN"
  fi

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
    echo "Created $CONFIG_DIR/elasticsearch.yml"
  else
    echo "Config exists: $CONFIG_DIR/elasticsearch.yml"
  fi

  # 准备日志配置：优先复制发行包默认配置，如不可用则生成最小配置
  if [ ! -f "$CONFIG_DIR/log4j2.properties" ]; then
    if [ -f "$ES_HOME/config/log4j2.properties" ]; then
      cp "$ES_HOME/config/log4j2.properties" "$CONFIG_DIR/log4j2.properties"
      echo "Copied default log4j2.properties to $CONFIG_DIR"
    else
      cat > "$CONFIG_DIR/log4j2.properties" <<'EOF'
status = error
appender.console.type = Console
appender.console.name = console
appender.console.layout.type = PatternLayout
appender.console.layout.pattern = [%d{ISO8601}][%-5p][%-25c] %marker%.-10000m%n
rootLogger.level = info
rootLogger.appenderRef.console.ref = console
EOF
      echo "Created minimal log4j2.properties at $CONFIG_DIR"
    fi
  fi

  mkdir -p "$CONFIG_DIR/jvm.options.d"
  if [ ! -f "$CONFIG_DIR/jvm.options.d/heap.options" ]; then
    cat > "$CONFIG_DIR/jvm.options.d/heap.options" <<'EOF'
-Xms512m
-Xmx512m
EOF
    echo "Created $CONFIG_DIR/jvm.options.d/heap.options"
  fi

  # 非 root 环境 chown 可能失败，忽略即可
  chown -R "$ES_USER":"$ES_USER" "$DATA_DIR" "$LOG_DIR" "$RUN_DIR" 2>/dev/null || true
  echo "Install finished."
}

cmd_start() {
  ensure_dirs

  if [ ! -x "$ES_BIN" ]; then
    echo "Elasticsearch binary not found. Please run: $(basename "$0") install" >&2
    exit 1
  fi
  if [ ! -f "$CONFIG_DIR/elasticsearch.yml" ]; then
    echo "Config not found. Please run: $(basename "$0") install" >&2
    exit 1
  fi
  if [ ! -f "$CONFIG_DIR/log4j2.properties" ]; then
    if [ -f "$ES_HOME/config/log4j2.properties" ]; then
      cp "$ES_HOME/config/log4j2.properties" "$CONFIG_DIR/log4j2.properties"
    else
      cat > "$CONFIG_DIR/log4j2.properties" <<'EOF'
status = error
appender.console.type = Console
appender.console.name = console
appender.console.layout.type = PatternLayout
appender.console.layout.pattern = [%d{ISO8601}][%-5p][%-25c] %marker%.-10000m%n
rootLogger.level = info
rootLogger.appenderRef.console.ref = console
EOF
    fi
  fi

  if [ -f "$PID_FILE" ] && ps -p "$(cat "$PID_FILE" 2>/dev/null)" >/dev/null 2>&1; then
    echo "Elasticsearch already running with PID $(cat "$PID_FILE")."
    exit 0
  fi

  export ES_PATH_CONF="$CONFIG_DIR"
  # 以 elasticsearch 用户启动；若用户不存在或 su 不可用，提示并以当前用户启动
  if id "$ES_USER" >/dev/null 2>&1; then
    if command -v su >/dev/null 2>&1; then
      su -s /bin/bash -c "$ES_BIN -d -p '$PID_FILE'" "$ES_USER" || {
        echo "su 启动失败，改为当前用户直接启动" >&2
        "$ES_BIN" -d -p "$PID_FILE"
      }
    else
      echo "su 不可用，使用当前用户启动" >&2
      "$ES_BIN" -d -p "$PID_FILE"
    fi
  else
    echo "用户 '$ES_USER' 不存在，使用当前用户启动" >&2
    "$ES_BIN" -d -p "$PID_FILE"
  fi

  echo "Elasticsearch starting... PID file: $PID_FILE"
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
}

cmd_stop() {
  ensure_dirs
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
  # 兜底：按本地发行包可执行路径匹配
  if [ -x "$ES_BIN" ]; then
    echo "PID 文件不存在或进程不在。尝试按进程名关闭。"
    pkill -f "$ES_BIN" || true
    echo "Stop signal sent."
  else
    echo "未找到运行中的 Elasticsearch。"
  fi
}

cmd_status() {
  if [ -f "$PID_FILE" ] && ps -p "$(cat "$PID_FILE" 2>/dev/null)" >/dev/null 2>&1; then
    echo "running (PID $(cat "$PID_FILE"))"
  else
    echo "not running"
  fi
}

case "${1:-}" in
  install) shift; cmd_install "$@" ;;
  start)   shift; cmd_start   "$@" ;;
  stop)    shift; cmd_stop    "$@" ;;
  status)  shift; cmd_status  "$@" ;;
  -h|--help|help|*) usage; exit 1 ;;
esac


