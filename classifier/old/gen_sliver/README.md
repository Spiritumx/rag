# 银标签数据生成

该目录包含用于生成银标签（silver label）数据的脚本，用于训练检索策略分类器。

## 核心概念

### 检索策略标签（Z0-M4）

| 代码 | 策略名称 | 检索方式 | 检索次数 |
|------|----------|----------|----------|
| Z0 | Zero-Retrieval | 无检索 | 0 次 |
| S1 | Single Retriever - BM25 | BM25 文本检索 | 1 次 |
| S2 | Single Retriever - HNSW | HNSW 密集向量检索 | 1 次 |
| S3 | Single Retriever - SPLADE | SPLADE 稀疏检索 | 1 次 |
| S4 | Hybrid + Rerank | BM25 + HNSW + SPLADE → Cross-Encoder Rerank | 1 次 |
| M1 | Multi-Round BM25 | BM25 文本检索 | ≥2 次 |
| M2 | Multi-Round HNSW | HNSW 密集向量检索 | ≥2 次 |
| M3 | Multi-Round SPLADE | SPLADE 稀疏检索 | ≥2 次 |
| M4 | Multi-Round Hybrid + Rerank | BM25 + HNSW + SPLADE → Cross-Encoder Rerank | ≥2 次 |

## 文件说明

- `preprocess_utils.py`: 核心工具函数
  - `label_complexity()`: 为单个数据集生成标签
  - `select_best_strategy_per_question()`: 为每个问题选择最优策略
  
- `preprocess_silver_train_gpt.py`: 主脚本，生成银标签数据

## 使用方法

### 前提条件

1. 已运行预测脚本，生成了不同检索模式的预测结果
2. 预测结果位于 `predictions/dev_500/` 目录下
3. 文件夹命名格式：`ircot_qa_gpt_<dataset>____prompt_set_1___bm25_retrieval_count__3___distractor_count__1___retrieval_mode__<mode>`

### 运行脚本

```bash
# 自动选择每个问题的最优策略
python classifier/gen_sliver/preprocess_silver_train_gpt.py gpt
```

### 输出

脚本会生成：
- `classifier/data/musique_hotpot_wiki2_nq_tqa_sqd/gpt/silver/label_best.json`

输出文件包含每个问题的最优策略标签，格式如下：

```json
[
  {
    "id": "question_id",
    "question": "问题文本",
    "answer": "S1",
    "answer_description": "zero",
    "dataset_name": "nq",
    "retrieval_mode": "bm25",
    "total_answer": ["答案1", "答案2"]
  },
  ...
]
```

## 策略选择逻辑

脚本会为每个问题选择最优的检索策略：

1. **优先选择能正确回答的策略**
   - 比较所有检索模式（bm25, hnsw, splade, hybrid）的结果
   - 找出能正确回答问题的策略

2. **如果多个策略都正确，选择最简单的**
   - 优先级：Z0（无检索）> S1-S4（单次检索）> M1-M4（多轮检索）
   - 原因：更简单的策略计算成本更低

3. **如果都不正确，选择 Z0**
   - 默认使用最简单的策略

## 示例

假设一个问题在不同检索模式下的表现：

| 检索模式 | 是否正确 | 标签 |
|---------|---------|------|
| bm25    | ✓       | S1   |
| hnsw    | ✓       | S2   |
| splade  | ✗       | M3   |
| hybrid  | ✓       | S4   |

最终选择：**S1**（BM25单次检索）
- 原因：S1、S2、S4 都正确，但 S1 优先级最高（BM25 是最简单的单次检索）

## 统计信息

运行脚本后会显示策略分布统计：

```
Strategy distribution:
  Z0:  150 (10.00%)
  S1:  450 (30.00%)
  S2:  300 (20.00%)
  S3:  200 (13.33%)
  S4:  250 (16.67%)
  M1:   50 ( 3.33%)
  M2:   50 ( 3.33%)
  M3:   25 ( 1.67%)
  M4:   25 ( 1.67%)
```

这有助于了解数据集中各种策略的分布情况。

## 故障排除

### 错误：找不到检索模式

```
Warning: No retrieval modes found with '___retrieval_mode__' suffix.
```

**解决方案**：
- 确保预测文件夹名称包含 `___retrieval_mode__<mode>` 后缀
- 检查 `predictions/dev_500/` 目录是否存在

### 错误：找不到预测文件

```
Warning: Could not load data for mode bm25: [Errno 2] No such file or directory
```

**解决方案**：
- 确保已运行预测脚本生成结果
- 检查文件路径是否正确
- 确认文件名格式：`zero_single_multi_classification__<dataset>_to_<dataset>__dev_500_subsampled.json`

## 相关脚本

- `classifier/generate_prediction/run_retrieval_dev_all.sh`: 生成所有检索模式的预测结果
- `classifier/generate_prediction/run.py`: 预测脚本入口


