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
# 1) 添加 Elastic APT 源与密钥
apt install -y curl gnupg apt-transport-https ca-certificates
curl -fsSL https://artifacts.elastic.co/GPG-KEY-elasticsearch | gpg --dearmor -o /usr/share/keyrings/elastic.gpg
echo "deb [signed-by=/usr/share/keyrings/elastic.gpg] https://artifacts.elastic.co/packages/8.x/apt stable main" | tee /etc/apt/sources.list.d/elastic-8.x.list >/dev/null
apt update

# 2) 安装 Elasticsearch（使用 8.x 稳定版）
apt install -y elasticsearch

# 3) 配置为单节点、禁用安全、监听 0.0.0.0（使用项目内目录，避免 /etc 与 /var）
ES_BASE_DIR="$(pwd)/retriever_server"
ES_CONFIG_DIR="$ES_BASE_DIR/es_config"
ES_DATA_DIR="$ES_BASE_DIR/es_data"
ES_LOG_DIR="$ES_BASE_DIR/es_logs"
ES_RUN_DIR="$ES_BASE_DIR/es_run"

mkdir -p "$ES_CONFIG_DIR" "$ES_DATA_DIR" "$ES_LOG_DIR" "$ES_RUN_DIR"

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
mkdir -p /etc/elasticsearch/jvm.options.d
tee /etc/elasticsearch/jvm.options.d/heap.options >/dev/null <<'EOF'
-Xms512m
-Xmx512m
EOF

# 5) 在无 systemd 环境下直接以后台进程启动 Elasticsearch
#    准备权限与运行目录（若非 root，chown 可能失败，忽略即可）
chown -R elasticsearch:elasticsearch "$ES_DATA_DIR" "$ES_LOG_DIR" "$ES_RUN_DIR" 2>/dev/null || true

#    后台启动（写入 PID 文件）
export ES_PATH_CONF="$ES_CONFIG_DIR"
su -s /bin/bash -c "/usr/share/elasticsearch/bin/elasticsearch -d -p '$ES_RUN_DIR/elasticsearch.pid'" elasticsearch

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

