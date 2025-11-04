# 批量构建所有索引指南

本指南说明如何一次性构建所有数据集的索引。

## 🚀 快速开始

### 方法 1：使用 Shell 脚本（推荐 Linux/Mac）

```bash
# 给脚本添加执行权限
chmod +x retriever_server/build_all_indexes.sh

# 构建所有数据集（BM25 + Dense + SPLADE）
./retriever_server/build_all_indexes.sh --all

# 或者只用 BM25（最快）
./retriever_server/build_all_indexes.sh --bm25-only
```

### 方法 2：使用 Python 脚本（跨平台）

```bash
# 构建所有数据集（BM25 + Dense + SPLADE）
python retriever_server/build_all_indexes.py --all

# 或者只用 BM25（最快）
python retriever_server/build_all_indexes.py --bm25-only
```

## 📋 支持的数据集

脚本会自动构建以下 5 个数据集的索引：

1. **hotpotqa** - HotpotQA 多跳问答数据集
2. **iirc** - IIRC 间接问答数据集
3. **2wikimultihopqa** - 2WikiMultihopQA 数据集
4. **musique** - MuSiQue 多跳推理数据集
5. **wiki** - Wikipedia 段落数据集

## 🎯 使用选项

### Shell 脚本选项

```bash
# 只用 BM25（最快，适合快速测试）
./retriever_server/build_all_indexes.sh --bm25-only

# BM25 + Dense embeddings
./retriever_server/build_all_indexes.sh --with-dense

# BM25 + SPLADE（推荐）
./retriever_server/build_all_indexes.sh --with-splade

# BM25 + Dense + SPLADE（最佳效果，最慢）
./retriever_server/build_all_indexes.sh --all
```

### Python 脚本选项

```bash
# 基础选项
python retriever_server/build_all_indexes.py --bm25-only    # 只用 BM25
python retriever_server/build_all_indexes.py --with-dense   # 添加 Dense
python retriever_server/build_all_indexes.py --with-splade  # 添加 SPLADE
python retriever_server/build_all_indexes.py --all          # 全部功能

# 构建特定数据集
python retriever_server/build_all_indexes.py --datasets wiki hotpotqa --all

# 出错后继续（不推荐）
python retriever_server/build_all_indexes.py --all --continue-on-error

# 跳过 ES 检查
python retriever_server/build_all_indexes.py --all --skip-es-check
```

## 📊 预计时间

| 配置 | 单个数据集 | 全部 5 个数据集 |
|------|-----------|----------------|
| BM25 only | 5-10 分钟 | 25-50 分钟 |
| BM25 + Dense | 15-30 分钟 | 1-2.5 小时 |
| BM25 + SPLADE | 20-40 分钟 | 1.5-3 小时 |
| BM25 + Dense + SPLADE | 30-60 分钟 | 2.5-5 小时 |

*注：实际时间取决于 CPU/GPU 性能和数据集大小*

## ✨ 功能特性

### 1. 自动检查 Elasticsearch
- 自动检测 ES 是否运行
- 如果未运行，自动启动

### 2. 进度显示
```
[1/5] Building index: hotpotqa
[2/5] Building index: iirc
...
```

### 3. 错误处理
- 详细的错误信息
- 失败后询问是否继续
- 最终的成功/失败统计

### 4. 时间统计
- 每个数据集的构建时间
- 总体构建时间
- 格式化的时间显示（小时/分钟/秒）

## 🔧 使用前准备

### 1. 下载模型（如果使用 Dense 或 SPLADE）

```bash
# 下载所有模型
cd retriever_server
python download_model.py --all
cd ..
```

### 2. 确保 Elasticsearch 运行

```bash
cd retriever_server
./es.sh status    # 检查状态
./es.sh start     # 如果未运行，启动它
cd ..
```

### 3. 确保有足够的磁盘空间

- BM25: 每个数据集约 500MB-2GB
- +Dense: 额外 1-3GB
- +SPLADE: 额外 500MB-1GB

## 📝 输出示例

```
========================================
Build All Dataset Indexes
========================================

Mode: BM25 + Dense + SPLADE
Datasets: hotpotqa, iirc, 2wikimultihopqa, musique, wiki
Total: 5 dataset(s)

Checking Elasticsearch...
Elasticsearch is running

========================================
Starting Index Building
========================================

========================================
[1/5] Building index: hotpotqa
========================================

Using device: cuda
📁 Found local dense model: /path/to/models/dense/all-MiniLM-L6-v2
📁 Found local SPLADE model: /path/to/models/splade/splade-v3
...
✓ Successfully built index: hotpotqa
  Time taken: 15m 23s

...

========================================
Build Summary
========================================

Total time: 1h 23m 45s

Successful: 5
  ✓ hotpotqa (15m 23s)
  ✓ iirc (18m 42s)
  ✓ 2wikimultihopqa (12m 15s)
  ✓ musique (20m 31s)
  ✓ wiki (16m 54s)

Failed: 0

✓ All 5 dataset(s) built successfully!
```

## 🚨 常见问题

### 问题 1：某个数据集失败

**解决方案**：
1. 查看错误信息
2. 单独重建该数据集：
   ```bash
   python retriever_server/build_index.py <dataset> --use-dense --use-splade --force
   ```

### 问题 2：内存不足

**解决方案**：
1. 增加 ES 堆内存：编辑 `es_config/jvm.options.d/heap.options`
2. 一次只构建几个数据集：
   ```bash
   python retriever_server/build_all_indexes.py --datasets wiki hotpotqa --all
   ```

### 问题 3：中途中断

**解决方案**：
- 重新运行脚本，已成功的数据集会被 `--force` 覆盖重建
- 或者手动构建失败的数据集

### 问题 4：速度太慢

**解决方案**：
1. 使用 GPU（速度提升 10-50 倍）
2. 只用 BM25：`--bm25-only`
3. 只用 SPLADE（不用 Dense）：`--with-splade`

## 💡 最佳实践

### 推荐工作流程

```bash
# 1. 下载模型（一次性）
cd retriever_server && python download_model.py --all && cd ..

# 2. 启动 ES
cd retriever_server && ./es.sh start && cd ..

# 3. 先用 BM25 快速测试
python retriever_server/build_all_indexes.py --bm25-only

# 4. 确认无误后，构建完整索引
python retriever_server/build_all_indexes.py --all
```

### 生产环境

```bash
# 使用 Python 脚本 + 自动继续（配合日志）
python retriever_server/build_all_indexes.py --all --continue-on-error 2>&1 | tee build_log.txt
```

### 开发环境

```bash
# 只构建你需要的数据集
python retriever_server/build_all_indexes.py --datasets wiki hotpotqa --with-splade
```

## 📖 相关文档

- `SIMPLE_GUIDE.md` - 单个索引构建指南
- `QUICKREF.md` - 快速参考
- `BUILD_ALL_GUIDE.md` - 本文档

## 🎓 脚本对比

| 特性 | Shell 脚本 | Python 脚本 |
|------|-----------|-------------|
| 跨平台 | ❌ Linux/Mac | ✅ 全平台 |
| 彩色输出 | ✅ | ✅ |
| 选择数据集 | ❌ | ✅ |
| 错误处理 | ✅ | ✅ |
| 进度显示 | ✅ | ✅ |
| 推荐使用 | Linux/Mac | Windows/全平台 |

---

**提示**：首次运行建议使用 `--bm25-only` 快速测试，确保所有数据集都能正常构建后，再使用完整功能！

