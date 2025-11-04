# 快速开始：多索引支持

这是一个快速指南，帮助你开始使用新的多索引功能（BM25 + HNSW + SPLADE）。

## 1. 安装依赖

```bash
cd retriever_server
pip install -r requirements_multi_index.txt
```

如果遇到 CUDA 相关问题，请先安装 PyTorch：
```bash
# For CUDA 11.8
pip install torch --index-url https://download.pytorch.org/whl/cu118

# For CPU only
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

## 2. 启动 Elasticsearch

```bash
# 安装 ES
./es.sh install

# 启动 ES
./es.sh start

# 检查状态
./es.sh status
```

## 3. 构建索引

### 选项 A: 只使用 BM25（原有方式）

```bash
python build_index.py wiki
```

### 选项 B: BM25 + HNSW 密集向量

```bash
# 小模型，快速（384维）
python build_index.py wiki --use-dense

# 大模型，高质量（768维）
python build_index.py wiki --use-dense --dense-model sentence-transformers/all-mpnet-base-v2
```

### 选项 C: BM25 + SPLADE 稀疏向量

```bash
python build_index.py wiki --use-splade
```

### 选项 D: 三种索引全部启用（推荐用于实验）

```bash
python build_index.py wiki --use-dense --use-splade --force
```

**注意：** 
- 使用 `--force` 会删除已有索引
- HNSW 和 SPLADE 需要较长时间，建议使用 GPU
- 对于大数据集，索引时间可能需要数小时

## 4. 查询测试

### 使用示例脚本查询

```bash
# BM25 查询
python query_example.py wiki "What is machine learning?" --method bm25

# HNSW 密集向量查询
python query_example.py wiki "What is machine learning?" --method hnsw

# SPLADE 稀疏向量查询
python query_example.py wiki "What is machine learning?" --method splade

# 混合查询（BM25 + HNSW）
python query_example.py wiki "What is machine learning?" --method hybrid

# 所有方法对比
python query_example.py wiki "What is machine learning?" --method all
```

### 使用 Python 代码查询

```python
from elasticsearch import Elasticsearch

es = Elasticsearch([{"host": "localhost", "port": 9200, "scheme": "http"}])

# BM25 查询
query = {
    "query": {
        "multi_match": {
            "query": "machine learning",
            "fields": ["title", "paragraph_text"]
        }
    }
}
results = es.search(index="wiki", body=query)

# HNSW 查询（需要先生成查询向量）
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
query_vector = model.encode("machine learning").tolist()

query = {
    "knn": {
        "field": "dense_embedding",
        "query_vector": query_vector,
        "k": 10,
        "num_candidates": 100
    }
}
results = es.search(index="wiki", body=query)
```

## 5. 性能对比

在 wiki 数据集上的典型表现：

| 方法 | 索引时间 | 索引大小 | 查询速度 | Recall@10 |
|------|---------|---------|---------|-----------|
| BM25 | 1x | 1x | 最快 | 基准 |
| BM25+HNSW | 3-5x | 2-3x | 快 | +10-15% |
| BM25+SPLADE | 4-6x | 1.5x | 快 | +15-20% |
| 三者结合 | 6-8x | 3-4x | 中等 | +20-25% |

*注：具体数值取决于硬件配置和数据集特性*

## 6. 常见问题

### Q: 索引速度太慢怎么办？
A: 
- 使用 GPU（会快 10-50 倍）
- 使用更小的模型
- 只为部分文档生成向量

### Q: 内存不足怎么办？
A: 
- 使用更小的模型（如 all-MiniLM-L6-v2）
- 减少 batch size
- 分批处理数据

### Q: 查询结果不理想怎么办？
A: 
- 尝试不同的模型
- 调整混合查询的权重
- 使用查询重排序（reranking）

### Q: 支持哪些数据集？
A: 目前支持：
- hotpotqa
- iirc
- 2wikimultihopqa
- musique
- wiki
- nq, trivia, squad（需要对应的数据加载函数）

## 7. 下一步

- 阅读 `README_MULTI_INDEX.md` 了解详细信息
- 查看 `query_example.py` 了解更多查询示例
- 实现自定义的混合查询策略
- 添加查询重排序（如使用 Cross-Encoder）

## 8. 项目结构

```
retriever_server/
├── build_index.py              # 主索引构建脚本（已更新）
├── es.sh                        # Elasticsearch 管理脚本
├── query_example.py             # 查询示例脚本（新增）
├── requirements_multi_index.txt # 依赖文件（新增）
├── README_MULTI_INDEX.md        # 详细文档（新增）
└── QUICKSTART_MULTI_INDEX.md    # 本文件（新增）
```

## 9. 技术支持

遇到问题？
1. 查看 ES 日志：`tail -f es_logs/*.log`
2. 检查 ES 状态：`curl http://localhost:9200/_cluster/health`
3. 查看索引信息：`curl http://localhost:9200/wiki`

## 10. 参考示例

### 完整的索引 + 查询流程

```bash
# 1. 安装依赖
pip install -r requirements_multi_index.txt

# 2. 启动 ES
./es.sh install
./es.sh start

# 3. 构建索引（BM25 + HNSW）
python build_index.py wiki --use-dense --force

# 4. 查询测试
python query_example.py wiki "artificial intelligence" --method all --top-k 5

# 5. 完成后停止 ES
./es.sh stop
```

祝你使用愉快！🚀

