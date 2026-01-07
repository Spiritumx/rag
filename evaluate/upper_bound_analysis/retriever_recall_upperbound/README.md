# Retriever Recall Upper Bound Test (检索器物理召回上限测试)

## 目的

测试检索器能否把包含正确答案的文档召回，用于诊断瓶颈在粗排、精排还是 Top-K 截断。

## 核心思路

在 RAG 系统中，检索失败可能发生在三个阶段：
1. **粗排失败**（BM25/HNSW）：候选集中就没有正确文档
2. **精排失败**（Reranker）：候选集有正确文档，但 Reranker 排序错误
3. **Top-K 截断失败**：正确文档在召回集中，但排名 > K

本测试通过计算 **Recall@K**（正确答案在前 K 个文档中出现的比例），定位具体问题所在。

## 测试流程

```
检索结果 (predictions_contexts.json) + Gold Answers
                ↓
      检查答案是否在 Top-K 文档中
                ↓
    计算 Recall@5, Recall@20, Recall@100
                ↓
            诊断瓶颈
```

## 结果解读

| 指标 | 诊断 | 优化方向 |
|------|------|----------|
| **Recall@100 < 50%** | ✗ 粗排能力差 | 语义门控召回、混合检索、增加候选数 |
| **Recall@100 > 80%<br>Recall@5 < 40%** | △ 粗排OK，精排差 | 更好的 Reranker、ToT、增加给 LLM 的文档数 |
| **Recall@5 > 60%** | ✓ 检索优秀 | 去优化 Reader (LLM/Prompt) |

## 使用方法

### 1. 准备检索结果

首先需要运行检索流程，生成 `predictions_contexts.json`：

```bash
# 运行 Stage 2 生成检索结果
cd evaluate
python stage2_generate.py --datasets musique --save-contexts
```

这会生成：`evaluate/outputs/stage2_predictions/musique/predictions_contexts.json`

### 2. 运行召回率测试

```bash
cd upper_bound_analysis/retriever_recall_upperbound

# 基本用法
python test_retrieval_recall.py \
  --dataset musique \
  --predictions-file ../../outputs/stage2_predictions/musique/predictions_contexts.json

# 自定义 K 值
python test_retrieval_recall.py \
  --dataset musique \
  --predictions-file ../../outputs/stage2_predictions/musique/predictions_contexts.json \
  --k-values 5 10 20 50 100

# 使用严格匹配（默认是模糊匹配）
python test_retrieval_recall.py \
  --dataset musique \
  --predictions-file ../../outputs/stage2_predictions/musique/predictions_contexts.json \
  --strict-match
```

### 3. 分析结果

```bash
# 生成诊断报告
python analyze_recall.py --dataset musique

# 比较不同检索策略
python analyze_recall.py --dataset musique --compare-strategies bm25 dense hybrid
```

### 4. 查看输出文件

测试完成后，会在 `outputs/{dataset}/` 下生成：

```
outputs/musique/
├── recall_metrics.json        # 召回率指标
├── recall_details.jsonl       # 详细结果（每个问题）
├── context_overlap.json       # 与 Gold Contexts 的重叠率
└── diagnosis_report.md        # 诊断报告
```

## 匹配模式

### Fuzzy Match（默认，推荐）
模糊匹配，容忍标点、大小写差异，适合实际场景。

**匹配规则：**
1. 直接包含（忽略大小写）
2. 移除标点后包含
3. 分词后所有关键词都出现

**示例：**
```python
Gold Answer: "New York City"
✓ 匹配: "... in New York City ..."
✓ 匹配: "... in new york city ..."
✓ 匹配: "... New York, City of ..."
✓ 匹配: "... City of New York ..."
```

### Strict Match
严格匹配，要求完全一致（忽略大小写）。

**示例：**
```python
Gold Answer: "New York City"
✓ 匹配: "... New York City ..."
✗ 不匹配: "... City of New York ..."
```

## 输出指标说明

### Recall@K
前 K 个文档中包含正确答案的问题比例。

**计算公式：**
```
Recall@K = (包含答案的问题数) / (总问题数)
```

**解读：**
- `Recall@5 = 0.65` → 65% 的问题，答案出现在前 5 个文档中
- `Recall@100 = 0.82` → 82% 的问题，答案出现在前 100 个文档中

### Context Overlap
检索到的文档与 Gold Contexts 的重叠率。

**指标：**
- **Perfect Matches**: 检索结果完全包含所有 Gold Contexts
- **Partial Matches**: 检索到部分 Gold Contexts
- **No Matches**: 没有检索到任何 Gold Context
- **Average Overlap**: 平均重叠比例

## 典型使用场景

### 场景 1: 诊断检索瓶颈

**问题：** RAG 系统 EM 只有 30%，不知道是检索问题还是 LLM 问题。

**步骤：**
```bash
# 1. 测试检索召回率
python test_retrieval_recall.py \
  --dataset musique \
  --predictions-file ../../outputs/stage2_predictions/musique/predictions_contexts.json

# 假设输出：
# Recall@100: 0.45 (45%)
# Recall@5:   0.22 (22%)
```

**诊断：** Recall@100 < 50% → 粗排就失败了，检索是主要瓶颈

**下一步：**
```bash
# 优化检索召回策略
# 1. 实现语义门控召回（创新点三）
# 2. 使用混合检索（BM25 + Dense）
# 3. 增加候选数量（top_k: 100 -> 200）
```

---

### 场景 2: 评估 Reranker 效果

**问题：** 想知道 Reranker 是提升了还是降低了召回率。

**步骤：**
```bash
# 1. 测试 Rerank 前（假设保存了候选集）
python test_retrieval_recall.py \
  --dataset musique \
  --predictions-file ../../outputs/stage2_predictions/musique/candidates_before_rerank.json \
  --k-values 100

# 输出: Recall@100 = 0.85

# 2. 测试 Rerank 后
python test_retrieval_recall.py \
  --dataset musique \
  --predictions-file ../../outputs/stage2_predictions/musique/predictions_contexts.json \
  --k-values 5 20

# 输出:
# Recall@20 = 0.68
# Recall@5  = 0.38
```

**分析：**
- Recall@100 = 85% → 粗排很好
- Recall@5 = 38% → Reranker 把大量正确文档排到后面去了

**决策：** Reranker 表现不佳，需要更换或微调

---

### 场景 3: 对比不同检索策略

**问题：** 想知道 BM25、Dense 和 Hybrid 哪个召回率最高。

**步骤：**
```bash
# 分别运行三种策略，保存到不同目录
# (假设已经运行并保存了结果)

# 比较召回率
python analyze_recall.py \
  --dataset musique \
  --compare-strategies bm25 dense hybrid
```

**输出示例：**
```
Strategy             Recall@5   Recall@20  Recall@100
----------------------------------------------------------------------
bm25                   0.4520     0.6832     0.7545
dense                  0.5120     0.7145     0.8234
hybrid                 0.6340     0.7889     0.8756

✓ Best Strategy: hybrid (Recall@5 = 0.6340)
```

**结论：** 混合检索效果最好，应该采用

---

### 场景 4: 支撑论文创新点

**场景：** 需要实验数据证明"语义门控召回"的必要性。

**实验设计：**
```bash
# Baseline: 当前检索策略
python test_retrieval_recall.py \
  --dataset musique \
  --predictions-file ../../outputs/baseline/predictions_contexts.json

# 输出: Recall@100 = 0.48 (< 50%)

# 在论文中写：
"We evaluated the retrieval upper bound and found that Recall@100
is only 48%, indicating that the coarse retrieval stage (BM25/HNSW)
fails to recall the correct documents. This strongly motivates our
proposed semantic gating mechanism (Innovation 3)."
```

---

## 与 Reader Upper Bound 的配合使用

推荐的诊断流程：

```mermaid
graph TD
    A[RAG 系统 EM 低] --> B[运行 Reader Upper Bound]
    B --> C{Reader EM?}
    C -->|>70%| D[瓶颈在检索]
    C -->|<70%| E[瓶颈在 LLM/Prompt]

    D --> F[运行 Retriever Recall Upper Bound]
    F --> G{Recall@100?}
    G -->|<50%| H[优化粗排: 语义门控、混合检索]
    G -->|>80%| I{Recall@5?}
    I -->|<40%| J[优化精排: Reranker、ToT]
    I -->|>60%| K[检索很好，回去优化 Reader]
```

## 常见问题

### Q1: Recall@K 和 EM 的关系是什么？
A:
- **Recall@K**: 检索器能否召回正确文档
- **EM**: LLM 能否从文档中提取正确答案
- 关系: `EM <= Recall@5`（LLM 只看前 5 个文档）

### Q2: 为什么要测试多个 K 值？
A:
- **Recall@100**: 诊断粗排能力
- **Recall@20**: 诊断精排中间结果
- **Recall@5**: 诊断给 LLM 的最终文档质量

### Q3: Fuzzy Match 和 Strict Match 哪个准确？
A:
- **Fuzzy Match（推荐）**: 更接近实际场景，容忍格式差异
- **Strict Match**: 过于严格，会低估召回率

### Q4: Context Overlap 和 Recall@K 有什么区别？
A:
- **Recall@K**: 检查答案字符串是否出现
- **Context Overlap**: 检查是否召回了 Gold Contexts（标注的支撑文档）
- Context Overlap 更严格，Recall@K 更实用

### Q5: 如果 Recall@100 很高，但 EM 还是很低怎么办？
A: 两种可能：
1. **Recall@5 低**: 精排有问题，正确文档被排到后面了
2. **Recall@5 高**: 检索没问题，LLM 能力不足，去优化 Reader

## 实验建议

### 推荐的测试顺序

```bash
# 1. 快速测试当前检索策略
python test_retrieval_recall.py \
  --dataset musique \
  --predictions-file YOUR_PREDICTIONS.json

# 2. 查看诊断报告
python analyze_recall.py --dataset musique

# 3. 根据诊断结果优化检索

# 4. 重新测试验证改进
python test_retrieval_recall.py \
  --dataset musique \
  --predictions-file YOUR_NEW_PREDICTIONS.json

# 5. 对比改进效果
python analyze_recall.py --dataset musique --compare-strategies baseline improved
```

### 典型 Recall@K 参考值

| 数据集 | 难度 | 期望 Recall@100 | 期望 Recall@5 |
|--------|------|----------------|---------------|
| SQuAD | 简单 | > 90% | > 75% |
| NQ | 简单 | > 85% | > 70% |
| HotpotQA | 中等 | > 80% | > 60% |
| MuSiQue | 困难 | > 70% | > 50% |
| 2WikiMultiHopQA | 困难 | > 65% | > 45% |

## 文件说明

```
retriever_recall_upperbound/
├── test_retrieval_recall.py    # 主测试脚本
├── analyze_recall.py            # 结果分析脚本
├── README.md                    # 本文档
└── outputs/                     # 输出目录
    └── {dataset}/
        ├── recall_metrics.json
        ├── recall_details.jsonl
        ├── context_overlap.json
        └── diagnosis_report.md
```

## 后续优化方向

根据测试结果采取不同的优化措施：

### 如果 Recall@100 < 50%（粗排问题）
1. **语义门控召回**（创新点三）
2. **混合检索**（BM25 + Dense + Keyword）
3. **增加候选数量**（100 -> 200-500）
4. **改进 Query Rewrite**

### 如果 Recall@100 > 80% 但 Recall@5 < 40%（精排问题）
1. **更好的 Reranker**（Cross-Encoder、ColBERT）
2. **ToT 检索**（创新点二）
3. **增加给 LLM 的文档数**（5 -> 10-20）
4. **微调 Reranker**

### 如果 Recall@5 > 60%（检索很好）
1. **不需要优化检索**
2. **去优化 Reader**（运行 Reader Upper Bound Test）

## License

MIT License
