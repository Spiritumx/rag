#!/bin/bash
# git clone https://github.com/Spiritum-coder/graduateRAG.git
# token ***REMOVED***



sudo apt update && sudo apt install p7zip-full unzip -y

cd download

chmod +x extract_datasets.sh
./extract_datasets.sh

cd ..

conda env update --file pixi_env.yaml

# 安装并启动 Elasticsearch（无 Docker）
# 1) 准备依赖与目录（使用项目内 es_dist，不使用系统目录）
apt update && apt install -y curl ca-certificates tar

# 3) 配置为单节点、禁用安全、监听 0.0.0.0（使用项目内目录，避免 /etc 与 /var）
ES_BASE_DIR="$(pwd)/retriever_server"
ES_CONFIG_DIR="$ES_BASE_DIR/es_config"
ES_DATA_DIR="$ES_BASE_DIR/es_data"
ES_LOG_DIR="$ES_BASE_DIR/es_logs"
ES_RUN_DIR="$ES_BASE_DIR/es_run"
ES_DIST_DIR="$ES_BASE_DIR/es_dist"
ES_VERSION="8.15.0"
ES_HOME="$ES_DIST_DIR/elasticsearch-$ES_VERSION"
ES_TGZ="elasticsearch-$ES_VERSION-linux-x86_64.tar.gz"
ES_URL="https://artifacts.elastic.co/downloads/elasticsearch/$ES_TGZ"

mkdir -p "$ES_CONFIG_DIR" "$ES_DATA_DIR" "$ES_LOG_DIR" "$ES_RUN_DIR" "$ES_DIST_DIR"

tee "$ES_CONFIG_DIR/elasticsearch.yml" >/dev/null <<'EOF'
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

# 4) 设置 JVM 内存参数（与 compose 保持一致 512m）
mkdir -p "$ES_CONFIG_DIR/jvm.options.d"
tee "$ES_CONFIG_DIR/jvm.options.d/heap.options" >/dev/null <<'EOF'
-Xms512m
-Xmx512m
EOF

# 4.1) 下载并解压 Elasticsearch 至项目目录（若不存在）
if [ ! -x "$ES_HOME/bin/elasticsearch" ]; then
  echo "Downloading Elasticsearch $ES_VERSION ..."
  curl -fSL "$ES_URL" -o "$ES_DIST_DIR/$ES_TGZ"
  tar -xzf "$ES_DIST_DIR/$ES_TGZ" -C "$ES_DIST_DIR"
fi

# 5) 在无 systemd 环境下直接以后台进程启动 Elasticsearch
#    准备权限与运行目录（若非 root，chown 可能失败，忽略即可）
chown -R elasticsearch:elasticsearch "$ES_DATA_DIR" "$ES_LOG_DIR" "$ES_RUN_DIR" 2>/dev/null || true

#    后台启动（写入 PID 文件）
export ES_PATH_CONF="$ES_CONFIG_DIR"
ES_BIN="$ES_HOME/bin/elasticsearch"
su -s /bin/bash -c "$ES_BIN -d -p '$ES_RUN_DIR/elasticsearch.pid'" elasticsearch

# 6) 健康检查等待（最多 ~5 分钟）
set +e
for i in $(seq 1 60); do
  if curl -fsS http://127.0.0.1:9200/_cluster/health >/dev/null; then
    echo "Elasticsearch is up."
    break
  fi
  echo "Waiting for Elasticsearch to start... ($i/60)"
  sleep 5
done
set -e

