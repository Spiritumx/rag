# 📚 超简化使用指南

本指南提供**最简单**的使用方式。模型会自动下载到 `retriever_server/models` 并自动加载。

## 🚀 快速开始（两步搞定）

### 步骤 1：下载模型

```bash
# 下载所有模型到 retriever_server/models
cd retriever_server
python download_model.py --all
```

输出示例：
```
✓ Successfully saved to: /path/to/retriever_server/models/dense/all-MiniLM-L6-v2
✓ Successfully saved to: /path/to/retriever_server/models/splade/splade-cocondenser-ensembledistil

✅ MODELS READY TO USE
Usage:
  python retriever_server/build_index.py <dataset> --use-dense --use-splade

Example:
  python retriever_server/build_index.py wiki --use-dense --use-splade
```

### 步骤 2：构建索引（自动使用本地模型）

```bash
# build_index.py 会自动从 retriever_server/models 加载模型
cd ..
python retriever_server/build_index.py wiki --use-dense --use-splade
```

**就这么简单！** 🎉

## 📂 目录结构

```
graduateRAG/
└── retriever_server/
    ├── models/                    # ← 模型存储在这里
    │   ├── dense/
    │   │   └── all-MiniLM-L6-v2/
    │   └── splade/
    │       └── splade-cocondenser-ensembledistil/
    ├── download_model.py          # ← 下载模型工具
    └── build_index.py             # ← 构建索引（自动加载本地模型）
```

## 🎯 完整示例

### 示例 1：使用所有索引（BM25 + Dense + SPLADE）

```bash
# 1. 下载模型
cd retriever_server
python download_model.py --all

# 2. 启动 Elasticsearch
./es.sh install
./es.sh start

# 3. 构建索引（自动使用本地模型）
cd ..
python retriever_server/build_index.py hotpotqa --use-dense --use-splade --force
```

### 示例 2：只使用 Dense 模型

```bash
# 1. 只下载 Dense 模型
cd retriever_server
python download_model.py --download-dense

# 2. 构建索引
cd ..
python retriever_server/build_index.py wiki --use-dense
```

### 示例 3：只使用 SPLADE 模型（推荐）

```bash
# 1. 只下载 SPLADE 模型
cd retriever_server
python download_model.py --download-splade

# 2. 构建索引
cd ..
python retriever_server/build_index.py wiki --use-splade
```

### 示例 4：只使用 BM25（无需下载模型）

```bash
# 直接构建，无需下载任何模型
python retriever_server/build_index.py wiki
```

## ✨ 自动检测功能

`build_index.py` 会自动检测并使用本地模型：

```bash
# 当你运行这个命令时：
python retriever_server/build_index.py wiki --use-dense --use-splade

# 它会自动：
📁 Found local dense model: /path/to/retriever_server/models/dense/all-MiniLM-L6-v2
📁 Found local SPLADE model: /path/to/retriever_server/models/splade/splade-cocondenser-ensembledistil
Loading dense embedding model: /path/to/retriever_server/models/dense/all-MiniLM-L6-v2
  ✓ Using local model
✓ Dense model loaded successfully (dimension: 384)
Loading SPLADE model: /path/to/retriever_server/models/splade/splade-cocondenser-ensembledistil
  ✓ Using local model
✓ SPLADE model loaded successfully
```

**不需要指定任何路径！** 🎊

## 💡 常见命令速查

```bash
# 下载所有模型
cd retriever_server && python download_model.py --all && cd ..

# 只用 BM25
python retriever_server/build_index.py <dataset>

# BM25 + Dense
python retriever_server/build_index.py <dataset> --use-dense

# BM25 + SPLADE（推荐）
python retriever_server/build_index.py <dataset> --use-splade

# BM25 + Dense + SPLADE（最佳效果）
python retriever_server/build_index.py <dataset> --use-dense --use-splade

# 强制重建
python retriever_server/build_index.py <dataset> --use-dense --use-splade --force
```

## 🔧 首次安装依赖

```bash
# 安装必要的 Python 包
pip install torch transformers sentence-transformers elasticsearch tqdm dill base58 beautifulsoup4
```

## ⚙️ Elasticsearch 管理

```bash
cd retriever_server

# 安装
./es.sh install

# 启动
./es.sh start

# 检查状态
./es.sh status

# 停止
./es.sh stop
```

## 📊 不同配置的效果对比

| 配置 | 下载时间 | 索引时间 | 效果 | 推荐场景 |
|------|---------|---------|------|----------|
| BM25 | 0分钟 | 快 | 基准 | 快速测试 |
| BM25 + SPLADE | 3-5分钟 | 中等 | +15-20% | **推荐使用** |
| BM25 + Dense | 1-2分钟 | 中等 | +10-15% | 语义搜索 |
| 三者结合 | 5-7分钟 | 慢 | +20-25% | 最佳效果 |

## ❓ 常见问题

### Q: 模型下载失败？
**A:** 使用 HuggingFace 镜像：
```bash
export HF_ENDPOINT=https://hf-mirror.com
cd retriever_server
python download_model.py --all
```

### Q: 找不到模型？
**A:** 确保在正确的目录：
```bash
# 检查模型是否存在
ls -la retriever_server/models/dense/
ls -la retriever_server/models/splade/

# 如果不存在，重新下载
cd retriever_server
python download_model.py --all
```

### Q: 想使用不同的模型？
**A:** 下载其他模型：
```bash
cd retriever_server
python download_model.py --download-dense --dense-model sentence-transformers/all-mpnet-base-v2
```

### Q: 如何清理模型？
**A:** 直接删除目录：
```bash
rm -rf retriever_server/models/
```

## 🎓 学习路径

1. **新手**：先用 BM25，了解基本流程
   ```bash
   python retriever_server/build_index.py wiki
   ```

2. **进阶**：添加 SPLADE，提升效果
   ```bash
   cd retriever_server && python download_model.py --download-splade && cd ..
   python retriever_server/build_index.py wiki --use-splade
   ```

3. **高级**：使用所有索引，获得最佳效果
   ```bash
   cd retriever_server && python download_model.py --all && cd ..
   python retriever_server/build_index.py wiki --use-dense --use-splade
   ```

## 📞 获取帮助

```bash
# 查看 download_model.py 帮助
python retriever_server/download_model.py --help

# 查看 build_index.py 帮助
python retriever_server/build_index.py --help
```

---

**记住：下载一次，永久使用！** 🚀

所有模型都会保存在 `retriever_server/models/`，之后每次使用都会自动加载，不需要重复下载！
