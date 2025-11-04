# Multi-Index Support: BM25 + HNSW + SPLADE

本文档说明如何使用 `build_index.py` 构建支持多种检索方式的 Elasticsearch 索引。

## 支持的索引类型

1. **BM25** (默认): 基于词频的稀疏检索，传统但有效
2. **HNSW**: 基于密集向量的近似最近邻搜索，适合语义检索
3. **SPLADE**: 稀疏学习表示，结合词法匹配和神经网络扩展

## 依赖安装

```bash
pip install torch sentence-transformers transformers elasticsearch
```

## 使用方法

### 1. 只使用 BM25（默认方式）

```bash
python build_index.py wiki
```

### 2. BM25 + HNSW 密集向量索引

```bash
# 使用默认模型 (all-MiniLM-L6-v2, 384维)
python build_index.py wiki --use-dense

# 使用指定模型
python build_index.py wiki --use-dense --dense-model sentence-transformers/all-mpnet-base-v2
```

**推荐的密集向量模型：**
- `sentence-transformers/all-MiniLM-L6-v2` (384维, 快速)
- `sentence-transformers/all-mpnet-base-v2` (768维, 高质量)
- `sentence-transformers/multi-qa-mpnet-base-dot-v1` (768维, 专为QA优化)

### 3. BM25 + SPLADE 稀疏向量索引

```bash
# 使用默认 SPLADE 模型
python build_index.py wiki --use-splade

# 使用指定模型
python build_index.py wiki --use-splade --splade-model naver/splade-cocondenser-ensembledistil
```

**SPLADE 模型推荐：**
- `naver/splade-cocondenser-ensembledistil` (推荐)
- `naver/splade-v2`
- `naver/splade_v2_max`

### 4. 三种索引同时使用 (BM25 + HNSW + SPLADE)

```bash
python build_index.py wiki --use-dense --use-splade
```

### 5. 强制重建索引

```bash
python build_index.py wiki --force --use-dense --use-splade
```

## 索引结构

### BM25 字段
- `title`: text (english analyzer)
- `paragraph_text`: text (english analyzer)
- `url`: text
- `paragraph_index`: integer
- `is_abstract`: boolean

### HNSW 字段
- `dense_embedding`: dense_vector
  - 维度: 取决于模型 (384/768/1024)
  - 相似度: cosine
  - HNSW 参数: m=16, ef_construction=100

### SPLADE 字段
- `splade_vector`: rank_features
  - 存储 token:score 对
  - 保留 top-200 个最高权重的 token

## 查询示例

### 1. BM25 查询

```python
query = {
    "query": {
        "multi_match": {
            "query": "What is machine learning?",
            "fields": ["title", "paragraph_text"]
        }
    }
}
```

### 2. HNSW 向量查询

```python
# 先生成查询向量
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
query_vector = model.encode("What is machine learning?").tolist()

# 执行向量查询
query = {
    "query": {
        "script_score": {
            "query": {"match_all": {}},
            "script": {
                "source": "cosineSimilarity(params.query_vector, 'dense_embedding') + 1.0",
                "params": {"query_vector": query_vector}
            }
        }
    }
}
```

或者使用 ES 8.0+ 的 kNN 查询：

```python
query = {
    "knn": {
        "field": "dense_embedding",
        "query_vector": query_vector,
        "k": 10,
        "num_candidates": 100
    }
}
```

### 3. SPLADE 查询

```python
# 先生成查询的 SPLADE 向量
from transformers import AutoModelForMaskedLM, AutoTokenizer
import torch

tokenizer = AutoTokenizer.from_pretrained("naver/splade-cocondenser-ensembledistil")
model = AutoModelForMaskedLM.from_pretrained("naver/splade-cocondenser-ensembledistil")

# 生成 SPLADE 向量
inputs = tokenizer("What is machine learning?", return_tensors='pt')
with torch.no_grad():
    outputs = model(**inputs)
    vec = torch.log(1 + torch.relu(outputs.logits))
    vec = torch.max(vec, dim=1).values.squeeze()

# 转换为 ES 查询
splade_dict = {}
for idx, score in enumerate(vec.cpu().numpy()):
    if score > 0:
        token = tokenizer.convert_ids_to_tokens([idx])[0]
        splade_dict[token] = float(score)

query = {
    "query": {
        "rank_feature": {
            "field": "splade_vector",
            "boost": splade_dict
        }
    }
}
```

### 4. 混合查询 (Hybrid Search)

```python
# BM25 + HNSW 混合
query = {
    "query": {
        "script_score": {
            "query": {
                "multi_match": {
                    "query": "What is machine learning?",
                    "fields": ["title", "paragraph_text"]
                }
            },
            "script": {
                "source": "_score + cosineSimilarity(params.query_vector, 'dense_embedding') + 1.0",
                "params": {"query_vector": query_vector}
            }
        }
    }
}
```

## 性能考虑

1. **索引时间**
   - BM25: 最快
   - SPLADE: 中等 (需要模型推理)
   - HNSW: 中等到慢 (取决于向量维度)

2. **索引大小**
   - BM25: 基准
   - SPLADE: +20-30% (稀疏向量)
   - HNSW: +50-200% (取决于维度)

3. **查询速度**
   - BM25: 极快
   - SPLADE: 快 (倒排索引)
   - HNSW: 快 (近似搜索)

4. **内存占用**
   - BM25: 基准
   - SPLADE: +少量
   - HNSW: +大量 (所有向量需要在内存中)

## 最佳实践

1. **小数据集 (<10万文档)**
   - 使用 BM25 + HNSW
   - 可以获得更好的语义理解

2. **中等数据集 (10万-100万文档)**
   - 使用 BM25 + SPLADE
   - 平衡性能和效果

3. **大数据集 (>100万文档)**
   - 主要使用 BM25
   - 可选择性地为重要部分添加向量索引

4. **最佳检索效果**
   - 使用三种索引 + 混合查询
   - 在应用层进行结果融合 (如 Reciprocal Rank Fusion)

## 注意事项

1. SPLADE 和 HNSW 需要 GPU 才能在合理时间内完成索引
2. 确保 Elasticsearch 版本 >= 8.0 以获得最佳的向量搜索支持
3. 对于大数据集，考虑批处理和分布式索引
4. 定期监控 ES 集群的内存和磁盘使用情况

## 故障排除

### 问题：内存不足
**解决方案：**
- 减小 batch_size
- 使用更小的模型
- 增加系统内存或使用更小的数据集

### 问题：索引速度太慢
**解决方案：**
- 使用 GPU 加速
- 减少 HNSW 的 ef_construction 参数
- 考虑只为部分文档生成向量

### 问题：查询结果不理想
**解决方案：**
- 尝试不同的模型
- 调整混合查询的权重
- 使用查询扩展或重排序

## 参考资源

- [Elasticsearch Dense Vector](https://www.elastic.co/guide/en/elasticsearch/reference/current/dense-vector.html)
- [Sentence Transformers](https://www.sbert.net/)
- [SPLADE Paper](https://arxiv.org/abs/2107.05720)

