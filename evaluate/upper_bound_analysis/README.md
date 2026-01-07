# Upper Bound Analysis Framework (上限分析诊断框架)

一套完整的 RAG 系统性能诊断工具，帮助你精准定位瓶颈、指导优化方向。

## 🚀 一键运行（最简单的方式）⭐

```bash
cd evaluate/upper_bound_analysis
python run_all_tests.py
```

**8-12 分钟后，你会得到：**
- ✅ Reader Upper Bound 测试结果（LLM 阅读理解能力）
- ✅ Retriever Recall Upper Bound 测试结果（检索召回能力）
- ✅ Agent Reasoning Check 测试结果（逻辑推理能力）
- ✅ 完整的诊断报告和优化建议

**详细文档：**
- 📖 一键脚本完整指南：[`RUN_ALL_TESTS_GUIDE.md`](RUN_ALL_TESTS_GUIDE.md)
- 📋 快速参考：[`QUICK_REFERENCE.md`](QUICK_REFERENCE.md)

**常用命令：**
```bash
# 快速验证（3分钟）
python run_all_tests.py --max-samples 20

# 标准诊断（8分钟）⭐ 推荐
python run_all_tests.py --max-samples 100

# 单数据集测试
python run_all_tests.py --datasets musique --max-samples 100

# 跳过 Retriever（如果还没运行 Stage 2）
python run_all_tests.py --skip-retriever
```

---

## 核心思想

RAG 系统由两部分组成：**Retriever（检索器）** 和 **Reader（阅读器/LLM）**。性能不佳可能来自两个地方：

```
RAG = Retriever + Reader

如果 EM 低 → 谁的锅？
  - Retriever 没召回正确文档？ → 测 Retriever Recall Upper Bound
  - Reader 拿到文档也答不对？ → 测 Reader Upper Bound
```

本框架提供**三个诊断工具**，全方位分析 RAG 系统性能：

## 框架结构

```
upper_bound_analysis/
├── README.md                           # 本文档（框架总览）
├── run_all_tests.py                    # ⭐一键运行所有测试（推荐使用）
├── config.yaml                         # ⭐统一配置文件（三个模块共用）
├── config_helper.py                    # 配置加载工具
│
├── 📚 文档
│   ├── RUN_ALL_TESTS_GUIDE.md         # ⭐一键脚本详细指南
│   ├── QUICK_REFERENCE.md             # ⭐快速参考
│   ├── CONFIG_USAGE.md                # 配置使用指南
│   ├── CONFIG_FIX_GUIDE.md            # 配置修复指南
│   ├── FILES_SUMMARY.md               # 文件清单
│   └── PARALLEL_UPGRADE_SUMMARY.md    # 并行处理说明
│
├── 1️⃣ reader_upper_bound/             # Reader 上限测试
│   ├── test_reader_upperbound.py      # 测试 LLM 在给定正确文档时的表现
│   ├── analyze_results.py             # 分析错误模式，生成诊断报告
│   ├── README.md                      # 详细文档
│   └── outputs/                       # 输出目录
│
├── 2️⃣ retriever_recall_upperbound/    # Retriever 召回上限测试
│   ├── test_retrieval_recall.py       # 测试检索器能否召回正确文档
│   ├── analyze_recall.py              # 分析召回失败原因，生成诊断报告
│   ├── README.md                      # 详细文档
│   └── outputs/                       # 输出目录
│
├── 3️⃣ agent_reasoning_check/          # Agent 推理能力测试⭐
│   ├── llm_backend.py                 # LLM Backend 封装
│   ├── test_reasoning_ability.py      # 测试 LLM 逻辑规划能力
│   ├── analyze_performance_breakdown.py # 性能分解分析（论文用）
│   ├── README.md                      # 详细文档
│   └── outputs/                       # 输出目录
│
└── outputs/
    └── all_tests_summary.json         # 所有测试的汇总结果
```

## ⚙️ 统一配置

三个模块使用**统一配置文件**：`upper_bound_analysis/config.yaml`

**核心配置：**
```yaml
data:
  max_samples: 100  # 默认100个样本（5-10分钟）
  datasets:
    - musique
    - 2wikimultihopqa

llm:
  server_host: "localhost"
  server_port: 8000

retriever:
  host: "localhost"
  port: 8001
```

**详细说明：** 查看 `CONFIG_USAGE.md`

## 三个工具的对比

| 特性 | Reader Upper Bound | Retriever Recall Upper Bound | Agent Reasoning Check⭐ |
|------|-------------------|------------------------------|----------------------|
| **测试对象** | LLM（阅读理解能力） | Retriever（检索召回能力） | LLM（逻辑规划能力） |
| **输入** | Gold Paragraphs | 检索结果 + Gold Answers | 检索结果（可换 LLM） |
| **测试内容** | 给定正确文档，LLM 能否答对 | 检索到的文档是否包含答案 | 换强 LLM 后性能是否提升 |
| **核心指标** | EM / F1 | Recall@K (K=5, 20, 100) | EM Δ（提升幅度） |
| **诊断目标** | LLM/Prompt 是否足够强 | 检索策略是否有效 | LLM 逻辑能力是否充足 |
| **优化方向** | Prompt、微调、换模型 | 混合检索、Reranker、ToT | ToT、换模型、优化 Prompt |
| **论文价值** | 确定理论上限 | 量化检索损失 | **生成性能分解图⭐** |

## 诊断流程

### 推荐的诊断顺序

```mermaid
graph TD
    A[RAG 系统 EM 低] --> B{先测哪个？}
    B -->|建议| C[Reader Upper Bound]

    C --> D{Reader EM?}
    D -->|≥ 70%| E[瓶颈在检索]
    D -->|< 70%| F[瓶颈在 LLM/Prompt]

    E --> G[运行 Retriever Recall Upper Bound]
    G --> H{Recall@100?}
    H -->|< 50%| I[优化粗排: 语义门控、混合检索]
    H -->|> 80%| J{Recall@5?}
    J -->|< 40%| K[优化精排: Reranker、ToT]
    J -->|> 60%| L[检索很好，回去优化 Reader]

    F --> M[优化 Prompt/微调/换模型]
    M --> N[重新测试]
```

### 为什么先测 Reader？

1. **速度快**：不需要跑检索，直接用 Gold Paragraphs
2. **快速排除**：如果 Reader EM < 30%，说明模型太弱，优化检索没意义
3. **指导检索优化**：只有知道 Reader 能力上限，才能评估检索需要达到什么水平

### 典型诊断场景

**场景 1: LLM 太弱**
```
RAG EM = 25%
  ↓
Reader Upper Bound: EM = 28%
  ↓
诊断: LLM 能力太弱，即使给正确文档也答不对
决策: 必须换模型或大幅优化 Prompt，优化检索毫无意义
```

**场景 2: 检索太差**
```
RAG EM = 30%
  ↓
Reader Upper Bound: EM = 75%
  ↓
诊断: LLM 能力很好，瓶颈在检索
  ↓
Retriever Recall Upper Bound: Recall@100 = 45%
  ↓
诊断: 粗排就失败了
决策: 实现语义门控召回、混合检索
```

**场景 3: Reranker 有问题**
```
RAG EM = 35%
  ↓
Reader Upper Bound: EM = 70%
  ↓
Retriever Recall Upper Bound:
  Recall@100 = 85%
  Recall@5   = 38%
  ↓
诊断: 粗排很好，但精排把正确文档排到后面了
决策: 更换 Reranker 或实现 ToT
```

**场景 4: 都还行，但都不够好**
```
RAG EM = 40%
  ↓
Reader Upper Bound: EM = 62%
  ↓
Retriever Recall Upper Bound:
  Recall@100 = 72%
  Recall@5   = 48%
  ↓
诊断: Reader 和 Retriever 都有提升空间
决策: 先优化 Prompt（成本低），再优化检索
```

## 快速开始

### 1. 测试 Reader Upper Bound

```bash
# 进入 reader_upper_bound 目录
cd upper_bound_analysis/reader_upper_bound

# 快速测试（100 个样本）
python run_quick_test.py --dataset musique --max-samples 100

# 查看结果（会告诉你瓶颈在哪）
# 假设输出: EM = 0.68 (68%)
# → 建议: 同时优化 Prompt 和检索
```

### 2. 测试 Retriever Recall Upper Bound

```bash
# 进入 retriever_recall_upperbound 目录
cd ../retriever_recall_upperbound

# 运行测试（需要先有检索结果）
python test_retrieval_recall.py \
  --dataset musique \
  --predictions-file ../../outputs/stage2_predictions/musique/predictions_contexts.json

# 查看结果（会告诉你召回率）
# 假设输出:
# Recall@100 = 0.75 (75%)
# Recall@5   = 0.45 (45%)
# → 建议: 粗排还行，精排需要优化
```

### 3. 测试 Agent Reasoning Ability⭐

```bash
# 进入 agent_reasoning_check 目录
cd ../agent_reasoning_check

# 测试基线 LLM
python test_reasoning_ability.py \
  --dataset hotpotqa \
  --backend local_llama \
  --max-samples 50

# 测试强 LLM (GPT-4o)
export OPENAI_API_KEY="your-key"
python test_reasoning_ability.py \
  --dataset hotpotqa \
  --backend gpt4 \
  --max-samples 50

# 查看结果
# 假设输出:
# Llama: EM = 0.32 (32%)
# GPT-4: EM = 0.64 (64%)
# → EM 暴涨 +32%，说明 LLM 逻辑能力不足，需要 ToT
```

### 4. 生成性能分解报告（论文用）⭐

```bash
# 整合三个测试结果
python analyze_performance_breakdown.py \
  --dataset hotpotqa \
  --baseline-backend local_llama \
  --strong-backend gpt4

# 生成:
# 1. performance_breakdown.png - 瀑布图（论文用）
# 2. backend_comparison.png - Backend 对比图
# 3. performance_breakdown_report.md - 完整报告
```

### 5. 综合诊断

根据三个测试的结果，做出优化决策：

| Reader EM | Recall@100 | Recall@5 | GPT-4 提升 | 诊断 | 优化优先级 |
|-----------|-----------|----------|------------|------|-----------|
| < 30% | - | - | - | LLM 太弱 | 1. 换模型 2. 优化 Prompt |
| 70%+ | < 50% | - | < +10% | 粗排太差 | 1. 语义门控 2. 混合检索 |
| 70%+ | 80%+ | < 40% | < +10% | 精排太差 | 1. Reranker 2. ToT |
| 70%+ | 80%+ | 60%+ | < +10% | 检索很好 | 检索不是瓶颈，优化其他 |
| 50-70% | 60-80% | 40-60% | **+30%** | **推理是瓶颈** | **1. ToT 2. 换模型⭐** |

## 详细使用指南

### Reader Upper Bound

**完整文档**: `reader_upper_bound/README.md`
**使用示例**: `reader_upper_bound/USAGE_EXAMPLES.md`

**核心命令**:
```bash
# 测试单个数据集
python test_reader_upperbound.py --datasets musique

# 测试多种 Prompt 风格
python test_reader_upperbound.py --datasets musique --prompt-style cot

# 快速测试
python run_quick_test.py --max-samples 100

# 比较 Prompt 效果
python run_quick_test.py --compare-prompts
```

**输出指标**:
- **EM (Exact Match)**: 完全匹配率
- **F1 Score**: F1 分数
- **Accuracy**: 准确率

**诊断阈值**:
- EM ≥ 70%: 检索是瓶颈
- 50% ≤ EM < 70%: 优化 Prompt + 检索
- 30% ≤ EM < 50%: 优化 LLM
- EM < 30%: 必须换模型

### Retriever Recall Upper Bound

**完整文档**: `retriever_recall_upperbound/README.md`

**核心命令**:
```bash
# 测试召回率
python test_retrieval_recall.py \
  --dataset musique \
  --predictions-file YOUR_PREDICTIONS.json

# 分析失败案例
python analyze_recall.py --dataset musique

# 比较不同检索策略
python analyze_recall.py --dataset musique --compare-strategies bm25 dense hybrid
```

**输出指标**:
- **Recall@5**: 前 5 个文档的召回率
- **Recall@20**: 前 20 个文档的召回率
- **Recall@100**: 前 100 个文档的召回率
- **Context Overlap**: 与 Gold Contexts 的重叠率

**诊断阈值**:
- Recall@100 < 50%: 粗排太差
- Recall@100 > 80%, Recall@5 < 40%: 精排太差
- Recall@5 > 60%: 检索很好

## 实验案例

### 案例 1: 发现 LLM 太弱

**背景**: RAG 系统在 MuSiQue 上 EM = 22%

**实验**:
```bash
# 测试 Reader Upper Bound
cd reader_upper_bound
python test_reader_upperbound.py --datasets musique

# 结果: EM = 0.29 (29%)
```

**诊断**: Reader EM 只有 29%，说明即使给了正确文档，LLM 也答不对

**决策**: 换用 Qwen-2.5-14B（更强的模型）

**结果**: 换模型后 Reader EM 提升到 68%，RAG EM 提升到 42%

---

### 案例 2: 发现检索粗排太差

**背景**: RAG 系统在 MuSiQue 上 EM = 28%

**实验**:
```bash
# 测试 Reader Upper Bound
cd reader_upper_bound
python test_reader_upperbound.py --datasets musique
# 结果: EM = 0.72 (72%)

# LLM 能力很好，测试检索
cd ../retriever_recall_upperbound
python test_retrieval_recall.py \
  --dataset musique \
  --predictions-file ../../outputs/stage2_predictions/musique/predictions_contexts.json

# 结果:
# Recall@100 = 0.48 (48%)
# Recall@5   = 0.25 (25%)
```

**诊断**: Recall@100 < 50%，粗排阶段就漏掉了大量正确文档

**决策**: 实现语义门控召回 + 混合检索（BM25 + Dense）

**结果**: 优化后 Recall@100 提升到 78%，RAG EM 提升到 45%

---

### 案例 3: 发现 Reranker 有问题

**背景**: 使用了混合检索，但 EM 还是不高

**实验**:
```bash
# 测试检索召回率
python test_retrieval_recall.py \
  --dataset musique \
  --predictions-file ../../outputs/stage2_predictions/musique/predictions_contexts.json

# 结果:
# Recall@100 = 0.85 (85%)  ← 粗排很好
# Recall@20  = 0.72 (72%)
# Recall@5   = 0.38 (38%)  ← 精排后大幅下降
```

**诊断**: 粗排召回率高，但精排后召回率大幅下降，Reranker 有问题

**决策**: 换用 Cross-Encoder Reranker，并在数据集上微调

**结果**: 优化后 Recall@5 提升到 62%，RAG EM 提升到 51%

---

## 论文写作支持

这两个测试工具可以为你的论文提供强有力的实验支撑。

### 证明创新点的必要性

**创新点二: ToT (Tree of Thought) 检索**

实验数据:
```bash
# Baseline 检索
Recall@100 = 82%
Recall@5   = 38%

# 结论：粗排很好，但精排把正确文档排到后面了
# → 证明需要 ToT 让 LLM 看更多文档
```

论文写法:
> "We evaluated the retrieval upper bound and found that while Recall@100
> reaches 82%, Recall@5 is only 38%, indicating that the reranker fails to
> prioritize correct documents. This motivates our ToT-based retrieval
> approach (Innovation 2), which allows the LLM to explore multiple
> retrieval paths instead of relying on a fixed top-5 ranking."

---

**创新点三: 语义门控召回**

实验数据:
```bash
# Baseline BM25
Recall@100 = 45%

# 结论：粗排就失败了
# → 证明需要语义门控动态选择检索策略
```

论文写法:
> "Our analysis reveals that Recall@100 is only 45%, suggesting that
> BM25 alone fails to retrieve the correct documents in the candidate
> set. This strongly motivates our semantic gating mechanism (Innovation 3),
> which dynamically selects retrieval strategies based on question semantics."

---

### 消融实验

对比不同组件的贡献:

| Method | Reader EM | Recall@100 | Recall@5 | RAG EM |
|--------|-----------|-----------|----------|--------|
| Baseline | 68% | 48% | 25% | 28% |
| + Better Prompt (CoT) | 75% | 48% | 25% | 32% |
| + Hybrid Retrieval | 75% | 78% | 42% | 45% |
| + Improved Reranker | 75% | 78% | 62% | 51% |
| + ToT | 75% | 78% | 68% | 56% |

这样的表格可以清晰展示每个改进的贡献。

---

## 常见问题

### Q1: 必须先测 Reader 再测 Retriever 吗？
A: 建议先测 Reader，因为：
- 速度快（不需要跑检索）
- 如果 Reader EM < 30%，说明 LLM 太弱，优化检索没意义

### Q2: 两个测试都需要运行吗？
A: 取决于第一个测试的结果：
- 如果 Reader EM < 30%：不需要测 Retriever，直接换模型
- 如果 Reader EM ≥ 50%：建议测 Retriever，精准定位检索问题

### Q3: 测试需要多长时间？
A:
- **Reader Upper Bound**（100 样本）：5-10 分钟
- **Retriever Recall Upper Bound**：1-2 分钟（只是检查字符串匹配）

### Q4: 可以用自己的数据集吗？
A: 可以，只需要数据格式符合要求：
- 包含 `gold_answers`
- 包含 `contexts`（带 `is_supporting` 标记）

### Q5: 如果两个上限都很高，但 RAG EM 还是低怎么办？
A: 可能的原因：
1. 检查 Prompt 是否在实际运行时和测试时一致
2. 检查 LLM 是否在实际运行时温度参数过高
3. 检查是否有其他流程问题（如文档截断、格式错误等）

---

## 后续扩展

这个框架可以继续扩展：

1. **Query Rewrite Quality Test**: 测试问题重写质量
2. **Multi-hop Reasoning Test**: 测试多跳推理能力
3. **Adversarial Test**: 测试对抗样本的鲁棒性
4. **Efficiency Test**: 测试检索和生成的效率

欢迎贡献！

---

## 总结

Upper Bound Analysis Framework 帮助你：

✅ **精准诊断**：定位瓶颈在 Retriever 还是 Reader
✅ **指导优化**：基于数据决策优化方向
✅ **节省时间**：不在错误方向上浪费精力
✅ **支撑论文**：提供实验数据证明创新点必要性

记住核心原则：
> **如果 Reader 拿到正确答案都答不对，那优化 Retriever 毫无意义！**
> **如果 Retriever 召回率很高，那瓶颈在 Reader！**

Happy Diagnosing! 🚀
