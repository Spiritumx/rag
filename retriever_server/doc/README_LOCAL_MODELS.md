# 本地模型快速使用指南

## 🎯 一键下载和使用

### 方法 1：使用快捷脚本（推荐）

```bash
# 给脚本添加执行权限（首次使用）
chmod +x retriever_server/setup_local_models.sh

# 下载所有模型到默认目录 (./models)
./retriever_server/setup_local_models.sh --all

# 或下载到 autodl-tmp 目录（推荐云平台）
./retriever_server/setup_local_models.sh --all --model-dir /root/autodl-tmp/models

# 只下载 dense 模型
./retriever_server/setup_local_models.sh --dense

# 只下载 SPLADE 模型
./retriever_server/setup_local_models.sh --splade
```

### 方法 2：直接使用 Python 脚本

```bash
# 下载所有模型
python retriever_server/download_model.py --all

# 下载到指定目录
python retriever_server/download_model.py --all --model-dir /root/autodl-tmp/models

# 只下载 dense 模型
python retriever_server/download_model.py --download-dense

# 只下载 SPLADE 模型
python retriever_server/download_model.py --download-splade
```

## 📦 使用本地模型构建索引

下载完成后，使用本地模型构建索引：

```bash
# 使用本地模型（默认路径）
python retriever_server/build_index.py wiki \
  --use-dense \
  --dense-model-path ./models/dense/all-MiniLM-L6-v2 \
  --use-splade \
  --splade-model-path ./models/splade/splade-cocondenser-ensembledistil

# 使用 autodl-tmp 目录的模型
python retriever_server/build_index.py wiki \
  --use-dense \
  --dense-model-path /root/autodl-tmp/models/dense/all-MiniLM-L6-v2 \
  --use-splade \
  --splade-model-path /root/autodl-tmp/models/splade/splade-cocondenser-ensembledistil
```

## 🔧 常见问题解决

### 1. 依赖安装问题

如果遇到 `sentence-transformers` 或 `transformers` 相关错误：

```bash
# 重新安装正确的版本
pip uninstall -y sentence-transformers torch transformers
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install transformers==4.36.0
pip install sentence-transformers==2.2.2
```

### 2. 网络下载慢

使用 HuggingFace 镜像加速：

```bash
export HF_ENDPOINT=https://hf-mirror.com
python retriever_server/download_model.py --all
```

### 3. 磁盘空间不足

只下载必要的模型：

```bash
# 只使用 SPLADE（效果好，空间小）
python retriever_server/download_model.py --download-splade

# 使用时
python retriever_server/build_index.py wiki \
  --use-splade \
  --splade-model-path ./models/splade/splade-cocondenser-ensembledistil
```

## 📊 模型对比

| 模型 | 大小 | 效果 | 速度 | 推荐场景 |
|------|------|------|------|----------|
| BM25（无需下载） | 0 | 基准 | 最快 | 快速测试 |
| Dense (all-MiniLM-L6-v2) | ~90MB | +10-15% | 快 | 通用推荐 |
| SPLADE | ~430MB | +15-20% | 快 | 高质量检索 |
| Dense + SPLADE | ~520MB | +20-25% | 中等 | 最佳效果 |

## 🎓 完整示例

### 情景 1：首次使用（推荐流程）

```bash
# 1. 下载模型到持久化存储
python retriever_server/download_model.py --all --model-dir /root/autodl-tmp/models

# 2. 启动 Elasticsearch
cd retriever_server
./es.sh install
./es.sh start
cd ..

# 3. 使用本地模型构建索引
python retriever_server/build_index.py wiki \
  --use-dense \
  --dense-model-path /root/autodl-tmp/models/dense/all-MiniLM-L6-v2 \
  --use-splade \
  --splade-model-path /root/autodl-tmp/models/splade/splade-cocondenser-ensembledistil \
  --force
```

### 情景 2：快速测试（只用 SPLADE）

```bash
# 1. 下载 SPLADE 模型
python retriever_server/download_model.py --download-splade

# 2. 构建索引
python retriever_server/build_index.py wiki \
  --use-splade \
  --splade-model-path ./models/splade/splade-cocondenser-ensembledistil
```

### 情景 3：CPU 环境优化

```bash
# CPU 环境推荐只用 SPLADE（避免 dense 模型太慢）
python retriever_server/download_model.py --download-splade

python retriever_server/build_index.py hotpotqa \
  --use-splade \
  --splade-model-path ./models/splade/splade-cocondenser-ensembledistil
```

## 📁 相关文档

- **详细指南**: `LOCAL_MODELS_GUIDE.md` - 包含所有详细信息和故障排除
- **多索引文档**: `README_MULTI_INDEX.md` - BM25/HNSW/SPLADE 技术细节
- **快速开始**: `QUICKSTART_MULTI_INDEX.md` - 快速上手指南

## ✅ 检查清单

下载和使用本地模型前，确保：

- [ ] 已安装 Python 依赖：`torch`, `transformers`, `sentence-transformers`
- [ ] 有足够的磁盘空间（至少 2GB）
- [ ] 能访问 HuggingFace（或配置了镜像）
- [ ] Elasticsearch 已启动

## 💡 小贴士

1. **首次下载**：第一次下载需要时间，请耐心等待
2. **模型复用**：下载后可以多次使用，无需重复下载
3. **路径记录**：下载完成后，脚本会显示模型路径，建议保存
4. **GPU 加速**：如有 GPU，索引速度会快 10-50 倍
5. **分步测试**：可以先只下载一个模型测试，确认可用后再下载其他

## 🚀 推荐配置

### 生产环境（高质量检索）

```bash
# 下载所有模型到持久化目录
python retriever_server/download_model.py --all --model-dir /root/autodl-tmp/models

# 使用所有索引类型
python retriever_server/build_index.py <dataset> \
  --use-dense \
  --dense-model-path /root/autodl-tmp/models/dense/all-MiniLM-L6-v2 \
  --use-splade \
  --splade-model-path /root/autodl-tmp/models/splade/splade-cocondenser-ensembledistil
```

### 开发环境（快速迭代）

```bash
# 只下载 SPLADE
python retriever_server/download_model.py --download-splade

# 快速构建索引
python retriever_server/build_index.py <dataset> \
  --use-splade \
  --splade-model-path ./models/splade/splade-cocondenser-ensembledistil
```

### 测试环境（最快）

```bash
# 不下载模型，只用 BM25
python retriever_server/build_index.py <dataset>
```

---

**有问题？** 查看 `LOCAL_MODELS_GUIDE.md` 获取更详细的帮助！

