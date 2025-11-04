# 本地模型使用指南

本指南说明如何下载模型到本地并使用，避免每次都从 HuggingFace 下载。

## 🚀 快速开始

### 步骤 1：下载模型到本地

```bash
# 下载所有模型（推荐）
python retriever_server/download_model.py --all

# 或者只下载需要的模型
python retriever_server/download_model.py --download-dense
python retriever_server/download_model.py --download-splade
```

**下载到指定目录（推荐 autodl-tmp）：**
```bash
python retriever_server/download_model.py --all --model-dir /root/autodl-tmp/models
```

### 步骤 2：使用本地模型构建索引

下载完成后，脚本会自动显示使用命令。例如：

```bash
# 使用本地模型构建索引
python retriever_server/build_index.py wiki \
  --use-dense \
  --dense-model-path ./models/dense/all-MiniLM-L6-v2 \
  --use-splade \
  --splade-model-path ./models/splade/splade-cocondenser-ensembledistil
```

## 📂 目录结构

下载后的模型目录结构：

```
models/
├── dense/
│   └── all-MiniLM-L6-v2/          # Dense embedding model
│       ├── config.json
│       ├── pytorch_model.bin
│       ├── tokenizer_config.json
│       └── ...
└── splade/
    └── splade-cocondenser-ensembledistil/  # SPLADE model
        ├── config.json
        ├── pytorch_model.bin
        ├── tokenizer_config.json
        └── ...
```

## 🔧 详细用法

### 下载模型的选项

```bash
# 1. 下载所有默认模型
python retriever_server/download_model.py --all

# 2. 只下载 Dense 模型
python retriever_server/download_model.py --download-dense

# 3. 只下载 SPLADE 模型
python retriever_server/download_model.py --download-splade

# 4. 下载特定的模型
python retriever_server/download_model.py \
  --download-dense \
  --dense-model sentence-transformers/all-mpnet-base-v2

# 5. 下载到指定目录
python retriever_server/download_model.py \
  --all \
  --model-dir /root/autodl-tmp/models
```

### 使用本地模型构建索引

```bash
# 完整命令示例
python retriever_server/build_index.py hotpotqa \
  --use-dense \
  --dense-model-path /root/autodl-tmp/models/dense/all-MiniLM-L6-v2 \
  --use-splade \
  --splade-model-path /root/autodl-tmp/models/splade/splade-cocondenser-ensembledistil

# 只使用本地 Dense 模型
python retriever_server/build_index.py wiki \
  --use-dense \
  --dense-model-path ./models/dense/all-MiniLM-L6-v2

# 只使用本地 SPLADE 模型
python retriever_server/build_index.py wiki \
  --use-splade \
  --splade-model-path ./models/splade/splade-cocondenser-ensembledistil
```

## 💡 推荐的工作流程

### 方案 A：autodl-tmp 目录（推荐）

适合 AutoDL 等云平台，数据持久化存储：

```bash
# 1. 下载模型到 autodl-tmp
cd /root/graduateRAG
python retriever_server/download_model.py --all --model-dir /root/autodl-tmp/models

# 2. 构建索引（使用本地模型）
python retriever_server/build_index.py wiki \
  --use-dense \
  --dense-model-path /root/autodl-tmp/models/dense/all-MiniLM-L6-v2 \
  --use-splade \
  --splade-model-path /root/autodl-tmp/models/splade/splade-cocondenser-ensembledistil
```

### 方案 B：项目目录

适合本地开发：

```bash
# 1. 下载模型到项目目录
python retriever_server/download_model.py --all

# 2. 构建索引
python retriever_server/build_index.py wiki \
  --use-dense \
  --dense-model-path ./models/dense/all-MiniLM-L6-v2 \
  --use-splade \
  --splade-model-path ./models/splade/splade-cocondenser-ensembledistil
```

## 📦 可用模型列表

### Dense Embedding 模型

| 模型名 | 维度 | 大小 | 特点 |
|-------|------|------|------|
| `sentence-transformers/all-MiniLM-L6-v2` | 384 | ~90MB | **默认**，快速 |
| `sentence-transformers/all-mpnet-base-v2` | 768 | ~420MB | 高质量 |
| `sentence-transformers/multi-qa-mpnet-base-dot-v1` | 768 | ~420MB | QA优化 |
| `sentence-transformers/paraphrase-MiniLM-L3-v2` | 384 | ~60MB | 最小，最快 |

### SPLADE 模型

| 模型名 | 大小 | 特点 |
|-------|------|------|
| `naver/splade-cocondenser-ensembledistil` | ~430MB | **默认**，推荐 |
| `naver/splade-cocondenser-selfdistil` | ~430MB | 备选 |

## ⚠️ 注意事项

### 1. 模型下载时间
- Dense 模型：约 1-3 分钟（取决于网络）
- SPLADE 模型：约 2-5 分钟
- **首次下载请耐心等待**

### 2. 磁盘空间
- Dense 模型：~100-500MB
- SPLADE 模型：~500MB
- **建议预留至少 2GB 空间**

### 3. 网络要求
- 需要能访问 HuggingFace (huggingface.co)
- 如果网络不好，可能需要多次尝试
- 可以使用镜像站点（设置环境变量）

### 4. 测试模型
下载完成后，脚本会自动测试模型是否可用：
```
✓ Successfully saved to: ./models/dense/all-MiniLM-L6-v2
  Model dimension: 384
```

## 🔍 故障排除

### 问题 1：下载失败

**错误信息：**
```
✗ Failed to download sentence-transformers/all-MiniLM-L6-v2: ...
```

**解决方案：**
1. 检查网络连接
2. 尝试使用 HuggingFace 镜像：
   ```bash
   export HF_ENDPOINT=https://hf-mirror.com
   python retriever_server/download_model.py --all
   ```
3. 手动下载后放到对应目录

### 问题 2：模型加载失败

**错误信息：**
```
Failed to load dense model: Could not import module 'BertModel'
```

**解决方案：**
```bash
# 重新安装依赖
pip install --upgrade transformers sentence-transformers torch
```

### 问题 3：磁盘空间不足

**解决方案：**
```bash
# 只下载必要的模型
python retriever_server/download_model.py --download-splade

# 或者下载到更大的磁盘
python retriever_server/download_model.py --all --model-dir /mnt/large_disk/models
```

## 📝 环境变量配置

### 使用 HuggingFace 镜像（国内加速）

```bash
# 临时设置
export HF_ENDPOINT=https://hf-mirror.com
python retriever_server/download_model.py --all

# 永久设置（添加到 ~/.bashrc）
echo 'export HF_ENDPOINT=https://hf-mirror.com' >> ~/.bashrc
source ~/.bashrc
```

### 设置缓存目录

```bash
# 设置 HuggingFace 缓存目录
export HF_HOME=/root/autodl-tmp/huggingface_cache
export TRANSFORMERS_CACHE=/root/autodl-tmp/transformers_cache
```

## 🎯 完整示例

### 示例 1：首次使用（从零开始）

```bash
# 1. 确保依赖已安装
pip install torch transformers sentence-transformers

# 2. 下载模型到 autodl-tmp
cd /root/graduateRAG
python retriever_server/download_model.py --all --model-dir /root/autodl-tmp/models

# 3. 启动 Elasticsearch
cd retriever_server
./es.sh install
./es.sh start

# 4. 使用本地模型构建索引
cd /root/graduateRAG
python retriever_server/build_index.py wiki \
  --use-dense \
  --dense-model-path /root/autodl-tmp/models/dense/all-MiniLM-L6-v2 \
  --use-splade \
  --splade-model-path /root/autodl-tmp/models/splade/splade-cocondenser-ensembledistil \
  --force
```

### 示例 2：只使用 Dense 模型（快速测试）

```bash
# 1. 下载 Dense 模型
python retriever_server/download_model.py --download-dense

# 2. 构建索引
python retriever_server/build_index.py wiki \
  --use-dense \
  --dense-model-path ./models/dense/all-MiniLM-L6-v2
```

### 示例 3：使用不同的模型

```bash
# 1. 下载高质量模型
python retriever_server/download_model.py \
  --download-dense \
  --dense-model sentence-transformers/all-mpnet-base-v2

# 2. 使用该模型构建索引
python retriever_server/build_index.py wiki \
  --use-dense \
  --dense-model-path ./models/dense/all-mpnet-base-v2
```

## 🆚 本地模型 vs 在线下载

| 特性 | 本地模型 | 在线下载 |
|-----|---------|---------|
| 首次使用 | 需要预下载 | 直接使用 |
| 后续使用 | 立即可用 | 每次都要检查/下载 |
| 网络要求 | 仅下载时需要 | 每次使用都需要 |
| 磁盘占用 | 需要预留空间 | 使用缓存 |
| 速度 | 快 | 较慢 |
| **推荐场景** | **生产环境、离线环境** | **快速测试** |

## 💾 清理和管理

### 查看已下载的模型

```bash
# 查看模型目录
ls -lh models/dense/
ls -lh models/splade/

# 查看模型大小
du -sh models/
```

### 删除不需要的模型

```bash
# 删除特定模型
rm -rf models/dense/all-MiniLM-L6-v2

# 删除所有模型
rm -rf models/
```

### 更新模型

```bash
# 删除旧模型并重新下载
rm -rf models/dense/all-MiniLM-L6-v2
python retriever_server/download_model.py --download-dense
```

---

**提示**：下载完成后，记得保存脚本输出的使用命令，方便后续使用！
