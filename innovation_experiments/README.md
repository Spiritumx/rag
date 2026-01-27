# Innovation Experiments

本目录包含对基线 RAG 系统的三项创新改进，采用 A/B 测试架构进行对比评估。

---

## 系统架构总览

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           RAG + Innovations 系统架构                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   用户问题                                                                   │
│       │                                                                     │
│       ▼                                                                     │
│   ┌─────────────────┐                                                       │
│   │   Stage 1       │  分类器 (Qwen 2.5-3B)                                 │
│   │   路由分类       │  预测: Z / S-Sparse / S-Dense / S-Hybrid / M         │
│   └────────┬────────┘                                                       │
│            │                                                                │
│            ▼                                                                │
│   ┌─────────────────────────────────────────────────────────────────┐      │
│   │                      Stage 2: 生成答案                           │      │
│   │  ┌─────────────────────────────────────────────────────────┐   │      │
│   │  │  [创新1] 自适应检索 (端口 8002)                           │   │      │
│   │  │  • 分析查询特征 (实体密度 + 语义抽象度)                    │   │      │
│   │  │  • Softmax 动态计算 BM25/SPLADE/Dense 权重                │   │      │
│   │  └─────────────────────────────────────────────────────────┘   │      │
│   │                           │                                     │      │
│   │                           ▼                                     │      │
│   │  ┌─────────────────────────────────────────────────────────┐   │      │
│   │  │  执行检索策略 → 生成初始答案                              │   │      │
│   │  └─────────────────────────────────────────────────────────┘   │      │
│   │                           │                                     │      │
│   │                           ▼                                     │      │
│   │  ┌─────────────────────────────────────────────────────────┐   │      │
│   │  │  [创新2] 级联动态路由                                     │   │      │
│   │  │  • 置信度验证 (Cross-Encoder)                            │   │      │
│   │  │  • confidence < 0.6 → 级联到 M-ToT                       │   │      │
│   │  └─────────────────────────────────────────────────────────┘   │      │
│   │                           │                                     │      │
│   │              ┌────────────┴────────────┐                       │      │
│   │              ▼                          ▼                       │      │
│   │     高置信度: 保留答案          低置信度: 级联                   │      │
│   │                                         │                       │      │
│   │                                         ▼                       │      │
│   │  ┌─────────────────────────────────────────────────────────┐   │      │
│   │  │  [创新3] MI-RA-ToT 束搜索推理                             │   │      │
│   │  │  • 并行探索多条推理路径 (beam_width=3)                    │   │      │
│   │  │  • 互信息评分: α×相关性 + β×新颖性                        │   │      │
│   │  │  • 剪枝保留最优路径                                       │   │      │
│   │  └─────────────────────────────────────────────────────────┘   │      │
│   └─────────────────────────────────────────────────────────────────┘      │
│                           │                                                 │
│                           ▼                                                 │
│   ┌─────────────────┐                                                       │
│   │   Stage 3       │  评估: EM / F1 / Recall                              │
│   │   评估指标       │  级联分析 / 权重分析                                  │
│   └─────────────────┘                                                       │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 三大创新概览

| 创新点 | 核心问题 | 解决方案 | 技术手段 |
|-------|---------|---------|---------|
| 1. 自适应检索 | 固定权重无法适应不同查询类型 | 根据查询特征动态调整权重 | Temperature-scaled Softmax |
| 2. 级联动态路由 | 分类器误判导致答案质量下降 | 置信度后验验证 + 自动回退 | Cross-Encoder 评分 |
| 3. MI-RA-ToT | 贪心推理容易陷入局部最优 | 束搜索并行探索多条路径 | 互信息评分 + 剪枝 |

---

## 创新点 1: 自适应检索 (Adaptive Retrieval)

### 问题背景

基线系统使用**固定权重**进行混合检索：
```python
# 基线: 固定权重
weights = {"bm25": 1.0, "splade": 1.0, "dense": 1.0}
```

**问题**: 不同类型的查询需要不同的检索策略：
- **实体密集查询** (如 "Who directed Titanic?"): 需要精确匹配 → BM25/SPLADE 更好
- **语义抽象查询** (如 "What is the significance of..."): 需要语义理解 → Dense 更好

### 解决方案

根据查询的**语言学特征**动态调整权重，使用 **Temperature-scaled Softmax** 实现连续权重分配。

### 完整流程

```
┌─────────────────────────────────────────────────────────────────────┐
│                    自适应检索完整流程                                 │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  输入查询: "Who is the director of the movie Titanic?"              │
│       │                                                             │
│       ▼                                                             │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  Step 1: 实体检测 (NER)                                      │   │
│  │  ├─ 使用 spaCy (en_core_web_sm) 或启发式方法                 │   │
│  │  ├─ 检测实体: ["Titanic"]                                    │   │
│  │  └─ 输出: entities = ["Titanic"], entity_count = 1           │   │
│  └─────────────────────────────────────────────────────────────┘   │
│       │                                                             │
│       ▼                                                             │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  Step 2: 计算词汇特异性 (Lexical Specificity)                │   │
│  │  ├─ entity_ratio = 实体token数 / 总token数                   │
│  │  │   = 1 / 9 = 0.11                                          │
│  │  ├─ length_bonus = min(平均实体长度 / 3, 1.0)                │   │
│  │  │   = min(1/3, 1.0) = 0.33                                  │
│  │  └─ lexical_score = 0.7 × 0.11 + 0.3 × 0.33 = 0.18           │   │
│  └─────────────────────────────────────────────────────────────┘   │
│       │                                                             │
│       ▼                                                             │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  Step 3: 计算语义抽象度 (Semantic Abstractness)              │   │
│  │  ├─ 抽象关键词: {how, why, relationship, significance...}   │   │
│  │  ├─ 具体关键词: {who, what, when, where, name...}           │   │
│  │  ├─ 查询包含 "who" → 具体问题                                │   │
│  │  └─ semantic_score = 0.2 (偏具体)                            │   │
│  └─────────────────────────────────────────────────────────────┘   │
│       │                                                             │
│       ▼                                                             │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  Step 4: Temperature-scaled Softmax 权重计算                 │   │
│  │  ├─ α = 2.0 (缩放因子)                                       │   │
│  │  ├─ logit_sparse = α × lexical_score = 2.0 × 0.18 = 0.36    │   │
│  │  ├─ logit_dense  = α × semantic_score = 2.0 × 0.2 = 0.40    │   │
│  │  ├─ logit_splade = (0.36 + 0.40) / 2 = 0.38                 │   │
│  │  │                                                           │   │
│  │  │  Softmax 归一化:                                          │   │
│  │  │  exp([0.36, 0.38, 0.40]) = [1.433, 1.462, 1.492]         │   │
│  │  │  sum = 4.387                                              │   │
│  │  │  weights = [0.327, 0.333, 0.340]                          │   │
│  │  │                                                           │   │
│  │  └─ 输出: BM25=0.327, SPLADE=0.333, Dense=0.340              │   │
│  └─────────────────────────────────────────────────────────────┘   │
│       │                                                             │
│       ▼                                                             │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  Step 5: 应用权重执行混合检索                                 │   │
│  │  ├─ BM25 检索 → 候选集 A (权重 0.327)                        │   │
│  │  ├─ SPLADE 检索 → 候选集 B (权重 0.333)                      │   │
│  │  ├─ Dense 检索 → 候选集 C (权重 0.340)                       │   │
│  │  └─ 加权融合 + Reranker → 最终检索结果                       │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 数学公式

**词汇特异性计算:**
$$\text{lexical\_score} = 0.7 \times \frac{\text{entity\_tokens}}{\text{total\_tokens}} + 0.3 \times \min\left(\frac{\text{avg\_entity\_length}}{3}, 1\right)$$

**语义抽象度计算:**
$$\text{semantic\_score} = \frac{\text{abstract\_keyword\_count}}{\text{abstract\_count} + \text{concrete\_count}}$$

**Softmax 权重:**
$$w_i = \frac{\exp(\alpha \cdot s_i / \tau)}{\sum_j \exp(\alpha \cdot s_j / \tau)}$$

其中 $\alpha = 2.0$ (缩放因子), $\tau = 1.0$ (温度参数)

### 核心代码实现

```python
class QueryAnalyzer:
    def get_dynamic_weights(self, query: str, temperature: float = 1.0) -> Dict:
        # 1. 分析查询特征
        analysis = self.analyze(query)
        lex_score = analysis['lexical_score']   # 实体密度
        sem_score = analysis['semantic_score']  # 语义抽象度

        # 2. 构建 logits
        alpha = 2.0  # 缩放因子
        logit_sparse = alpha * lex_score   # BM25
        logit_dense  = alpha * sem_score   # Dense
        logit_splade = (logit_sparse + logit_dense) / 2  # SPLADE

        # 3. Softmax 归一化
        logits = np.array([logit_sparse, logit_splade, logit_dense])
        exp_logits = np.exp(logits / temperature - np.max(logits / temperature))
        weights = exp_logits / np.sum(exp_logits)

        return {
            "bm25": weights[0],
            "splade": weights[1],
            "dense": weights[2]
        }
```

### 设计优势

| 特性 | 基线 (固定阈值) | 创新 (Softmax) |
|-----|---------------|----------------|
| 权重变化 | 离散跳变 | 连续平滑 |
| 阈值解释 | 需要解释"为什么是0.3" | 无需硬编码阈值 |
| 数学性质 | 不可微 | 可微分，便于理论分析 |
| 归一化 | 需手动确保和为1 | 自动归一化 |

### 示例效果

| 查询类型 | 示例 | lexical | semantic | BM25 | SPLADE | Dense |
|---------|------|---------|----------|------|--------|-------|
| 实体密集 | "Albert Einstein born year" | 0.65 | 0.10 | **0.52** | 0.32 | 0.16 |
| 语义抽象 | "What is the significance of quantum mechanics?" | 0.05 | 0.85 | 0.14 | 0.28 | **0.58** |
| 平衡型 | "When did World War II end?" | 0.35 | 0.30 | 0.36 | 0.34 | 0.30 |

---

## 创新点 2: 级联动态路由 (Cascading Dynamic Routing)

### 问题背景

分类器可能**误判**问题复杂度：
- 简单问题被误判为复杂 → 浪费计算资源
- 复杂问题被误判为简单 → **答案质量下降** (更严重)

例如：
```
问题: "Who directed the film that won Best Picture in 1998?"
分类器预测: S-Hybrid (单跳)
实际需要: M (多跳 - 先找1998年最佳影片，再找导演)
```

### 解决方案

**后验验证**: 执行初始策略后，检查答案的**置信度**，低置信度时**自动升级**到更强策略。

### 完整流程

```
┌─────────────────────────────────────────────────────────────────────┐
│                    级联动态路由完整流程                               │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  输入: 问题 + 分类器预测 (如 S-Hybrid)                               │
│       │                                                             │
│       ▼                                                             │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  Step 1: 执行初始策略                                        │   │
│  │  ├─ 调用 S-Hybrid 检索配置                                   │   │
│  │  ├─ 检索相关文档: [doc1, doc2, doc3, ...]                   │   │
│  │  └─ LLM 生成答案: "James Cameron"                           │   │
│  └─────────────────────────────────────────────────────────────┘   │
│       │                                                             │
│       │  输出: answer="James Cameron", contexts=[doc1, doc2, ...]  │
│       ▼                                                             │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  Step 2: 置信度验证 (ConfidenceVerifier)                     │   │
│  │  │                                                           │   │
│  │  │  构建 QA 对:                                              │   │
│  │  │  qa_text = "Question: Who directed Titanic?               │   │
│  │  │             Answer: James Cameron"                        │   │
│  │  │                                                           │   │
│  │  │  对每个上下文评分 (Cross-Encoder):                        │   │
│  │  │  ├─ score(qa_text, doc1) = 0.85  ← 文档支持答案          │   │
│  │  │  ├─ score(qa_text, doc2) = 0.72                          │   │
│  │  │  └─ score(qa_text, doc3) = 0.45                          │   │
│  │  │                                                           │   │
│  │  │  置信度 = max(scores) = 0.85                              │   │
│  │  └─────────────────────────────────────────────────────────  │   │
│  └─────────────────────────────────────────────────────────────┘   │
│       │                                                             │
│       │  confidence = 0.85                                          │
│       ▼                                                             │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  Step 3: 级联决策                                            │   │
│  │  │                                                           │   │
│  │  │  if strategy.startswith("M"):                            │   │
│  │  │      return False  # M 策略已经最强，不级联               │   │
│  │  │                                                           │   │
│  │  │  if confidence < threshold (0.6):                        │   │
│  │  │      return True   # 低置信度，需要级联                   │   │
│  │  │  else:                                                   │   │
│  │  │      return False  # 高置信度，保留原答案                 │   │
│  │  │                                                           │   │
│  │  │  本例: 0.85 >= 0.6 → 不级联                               │   │
│  │  └─────────────────────────────────────────────────────────  │   │
│  └─────────────────────────────────────────────────────────────┘   │
│       │                                                             │
│       ├─────────────────────┬───────────────────────────────────   │
│       ▼                     ▼                                       │
│  ┌──────────────┐    ┌──────────────────────────────────────────┐  │
│  │ 高置信度路径  │    │ 低置信度路径 (级联)                       │  │
│  │              │    │                                          │  │
│  │ 保留原答案    │    │  Step 4: 调用 MI-RA-ToT                  │  │
│  │ "James       │    │  ├─ 束搜索多条推理路径                    │  │
│  │  Cameron"    │    │  ├─ 互信息评分选择最优                    │  │
│  │              │    │  └─ 生成新答案                            │  │
│  └──────────────┘    └──────────────────────────────────────────┘  │
│       │                     │                                       │
│       ▼                     ▼                                       │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  Step 5: 记录路由决策 (RoutingLogger)                        │   │
│  │  │                                                           │   │
│  │  │  log_entry = {                                           │   │
│  │  │      "question_id": "q001",                              │   │
│  │  │      "initial_action": "S-Hybrid",                       │   │
│  │  │      "confidence": 0.85,                                 │   │
│  │  │      "final_action": "S-Hybrid",  # 或 "M-ToT"           │   │
│  │  │      "cascaded": False,           # 或 True              │   │
│  │  │      "dataset": "hotpotqa"                               │   │
│  │  │  }                                                       │   │
│  │  └─────────────────────────────────────────────────────────  │   │
│  └─────────────────────────────────────────────────────────────┘   │
│       │                                                             │
│       ▼                                                             │
│  输出: 最终答案 + 路由日志                                           │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 置信度计算详解

```
┌────────────────────────────────────────────────────────────────┐
│                   置信度验证原理                                │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  问题: "What is the capital of France?"                        │
│  答案: "Paris"                                                 │
│  上下文: [doc1, doc2, doc3]                                    │
│                                                                │
│  ┌──────────────────────────────────────────────────────────┐ │
│  │ Cross-Encoder 评分原理:                                   │ │
│  │                                                          │ │
│  │ 输入: (QA对, 文档)                                        │ │
│  │       ↓                                                  │ │
│  │ "Question: What is the capital of France?                │ │
│  │  Answer: Paris"                                          │ │
│  │       +                                                  │ │
│  │ "Paris is the capital and most populous city of France." │ │
│  │       ↓                                                  │ │
│  │ Cross-Encoder (BERT-based) → score = 0.92               │ │
│  │                                                          │ │
│  │ 解释: 文档明确支持 "Paris 是 France 的首都"               │ │
│  │       → 高置信度 → 答案可靠                              │ │
│  └──────────────────────────────────────────────────────────┘ │
│                                                                │
│  ┌──────────────────────────────────────────────────────────┐ │
│  │ 低置信度情况:                                             │ │
│  │                                                          │ │
│  │ 问题: "Who won the 1998 Best Picture Oscar?"             │ │
│  │ 答案: "Titanic" (分类器预测 S-Hybrid，实际需要多跳)       │ │
│  │ 上下文: [关于电影奖项的泛泛描述，未提及1998年]            │ │
│  │       ↓                                                  │ │
│  │ Cross-Encoder → score = 0.35                            │ │
│  │                                                          │ │
│  │ 解释: 文档未明确支持答案 → 低置信度 → 需要级联           │ │
│  └──────────────────────────────────────────────────────────┘ │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

### 核心代码实现

```python
class ConfidenceVerifier:
    def __init__(self, model_name="cross-encoder/ms-marco-MiniLM-L-6-v2",
                 threshold=0.6, max_contexts=3):
        self.model = CrossEncoder(model_name)
        self.threshold = threshold

    def verify(self, question: str, answer: str, contexts: List[Dict]) -> float:
        """计算答案置信度"""
        # 1. 构建 QA 对
        qa_text = f"Question: {question}\nAnswer: {answer}"

        # 2. 对每个上下文评分
        scores = []
        for ctx in contexts[:self.max_contexts]:
            score = self.model.predict([(qa_text, ctx['paragraph_text'])])[0]
            scores.append(score)

        # 3. 取最高分作为置信度
        return max(scores) if scores else 0.0

    def should_cascade(self, confidence: float, strategy: str) -> bool:
        """决定是否级联"""
        if strategy.startswith("M"):
            return False  # M 策略不级联
        return confidence < self.threshold
```

### 级联决策统计示例

| 初始策略 | 总问题数 | 级联数 | 级联率 | 级联后 EM 提升 |
|---------|---------|--------|--------|---------------|
| S-Sparse | 120 | 28 | 23.3% | +0.08 |
| S-Dense | 95 | 15 | 15.8% | +0.05 |
| S-Hybrid | 185 | 32 | 17.3% | +0.06 |
| M | 100 | 0 | 0% | - |

---

## 创新点 3: MI-RA-ToT (互信息树思维推理)

### 问题背景

基线多跳推理使用**贪心搜索**:
```
问题 → 查询1 → 检索 → 查询2 → 检索 → ... → 答案
                 ↓
            只探索一条路径，可能陷入局部最优
```

**问题**:
- 第一步搜索方向错误 → 后续全部偏离
- 无法纠错，无法回退

### 解决方案

**束搜索 (Beam Search)**: 每层并行探索多条路径，用**互信息评分**剪枝保留最优。

### 与基线对比

```
┌─────────────────────────────────────────────────────────────────────┐
│                    基线 vs MI-RA-ToT 对比                            │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  【基线: 贪心搜索 (深度优先)】                                        │
│                                                                     │
│       问题                                                          │
│         │                                                           │
│         ▼                                                           │
│       查询1 ────────────────────────┐                               │
│         │                          │ 只有一条路径                    │
│         ▼                          │ 错了无法回头                    │
│       查询2 ────────────────────────┤                               │
│         │                          │                               │
│         ▼                          │                               │
│       查询3 ────────────────────────┤                               │
│         │                          │                               │
│         ▼                          ▼                               │
│       答案 (可能错误)                                                │
│                                                                     │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  【创新: MI-RA-ToT (广度优先束搜索)】                                 │
│                                                                     │
│                         问题                                        │
│                           │                                         │
│                           ▼                                         │
│               ┌───────────┼───────────┐                            │
│               ▼           ▼           ▼                            │
│            查询1a      查询1b      查询1c     ← Beam Width=3        │
│            (0.8)       (0.6)       (0.5)       MI评分              │
│               │           │           │                            │
│               ▼           ▼           ▼                            │
│         ┌─────┼─────┬─────┼─────┬─────┼─────┐                      │
│         ▼     ▼     ▼     ▼     ▼     ▼     ▼                      │
│       2a   2b   2c   2d   2e   2f   2g   2h   2i  共9个候选        │
│                                                                     │
│         剪枝: 保留 Top-3 (按累积 MI 分数)                            │
│               ▼                                                     │
│            [2a, 2d, 2g]                                            │
│               │                                                     │
│               ▼                                                     │
│         继续扩展...                                                  │
│               │                                                     │
│               ▼                                                     │
│         选择最优路径 → 生成答案                                      │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 完整流程

```
┌─────────────────────────────────────────────────────────────────────┐
│                    MI-RA-ToT 完整流程                                │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  输入: "Who is the director of the film that won Best Picture       │
│         at the 70th Academy Awards?"                                │
│       │                                                             │
│       ▼                                                             │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  Step 1: 初始化根节点                                        │   │
│  │  ├─ 执行初始检索: query = 原问题                             │   │
│  │  ├─ 检索结果: [doc1, doc2, ..., doc8]                       │   │
│  │  └─ 创建根节点: TreeNode(query=问题, contexts=[...], score=0) │   │
│  └─────────────────────────────────────────────────────────────┘   │
│       │                                                             │
│       │  current_beam = [root]                                      │
│       ▼                                                             │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  Step 2: 束搜索迭代 (Depth 1 ~ max_depth)                    │   │
│  │  │                                                           │   │
│  │  │  for depth in range(1, max_depth + 1):                   │   │
│  │  │      candidates = []                                     │   │
│  │  │                                                           │   │
│  │  │      for node in current_beam:                           │   │
│  │  │          │                                                │   │
│  │  │          ▼                                                │   │
│  │  │      ┌────────────────────────────────────────────────┐  │   │
│  │  │      │  Step 2a: 生成 k 个候选查询                     │  │   │
│  │  │      │  ├─ LLM 生成多样化查询 (temperature=0.7)        │  │   │
│  │  │      │  │                                              │  │   │
│  │  │      │  │  Prompt:                                     │  │   │
│  │  │      │  │  "Generate 3 different search queries        │  │   │
│  │  │      │  │   to find: {问题}                            │  │   │
│  │  │      │  │   Current context: {已有文档}                │  │   │
│  │  │      │  │   Past steps: {历史查询}"                    │  │   │
│  │  │      │  │                                              │  │   │
│  │  │      │  │  输出:                                       │  │   │
│  │  │      │  │  Query 1: "70th Academy Awards Best Picture" │  │   │
│  │  │      │  │  Query 2: "1998 Oscar winner film"           │  │   │
│  │  │      │  │  Query 3: "Best Picture 1998 director"       │  │   │
│  │  │      │  │                                              │  │   │
│  │  │      │  └─ 去重 + 清洗 → candidate_queries             │  │   │
│  │  │      └────────────────────────────────────────────────┘  │   │
│  │  │          │                                                │   │
│  │  │          ▼                                                │   │
│  │  │      ┌────────────────────────────────────────────────┐  │   │
│  │  │      │  Step 2b: 对每个候选查询执行检索               │  │   │
│  │  │      │  ├─ query = "70th Academy Awards Best Picture" │  │   │
│  │  │      │  └─ new_docs = retrieve(query, max_hits=5)     │  │   │
│  │  │      └────────────────────────────────────────────────┘  │   │
│  │  │          │                                                │   │
│  │  │          ▼                                                │   │
│  │  │      ┌────────────────────────────────────────────────┐  │   │
│  │  │      │  Step 2c: 计算互信息增益 (MI Gain)              │  │   │
│  │  │      │  │                                              │  │   │
│  │  │      │  │  existing_docs = node.get_all_contexts()    │  │   │
│  │  │      │  │                                              │  │   │
│  │  │      │  │  for doc in new_docs:                       │  │   │
│  │  │      │  │      relevance = CrossEncoder(问题, doc)    │  │   │
│  │  │      │  │      novelty = 1 - max_jaccard(doc, existing)│  │   │
│  │  │      │  │      mi_score = α×relevance + β×novelty     │  │   │
│  │  │      │  │                = 0.7×0.8 + 0.3×0.6 = 0.74   │  │   │
│  │  │      │  │                                              │  │   │
│  │  │      │  └─ mi_gain = avg(mi_scores) = 0.74            │  │   │
│  │  │      └────────────────────────────────────────────────┘  │   │
│  │  │          │                                                │   │
│  │  │          ▼                                                │   │
│  │  │      ┌────────────────────────────────────────────────┐  │   │
│  │  │      │  Step 2d: 创建子节点                            │  │   │
│  │  │      │  │                                              │  │   │
│  │  │      │  │  child = TreeNode(                          │  │   │
│  │  │      │  │      query = "70th Academy Awards...",      │  │   │
│  │  │      │  │      contexts = new_docs,                   │  │   │
│  │  │      │  │      score = parent.score + mi_gain,        │  │   │
│  │  │      │  │      parent = node,                         │  │   │
│  │  │      │  │      depth = depth                          │  │   │
│  │  │      │  │  )                                          │  │   │
│  │  │      │  │                                              │  │   │
│  │  │      │  └─ candidates.append(child)                   │  │   │
│  │  │      └────────────────────────────────────────────────┘  │   │
│  │  │                                                           │   │
│  │  │      # 所有节点扩展完毕后                                 │   │
│  │  │          │                                                │   │
│  │  │          ▼                                                │   │
│  │  │      ┌────────────────────────────────────────────────┐  │   │
│  │  │      │  Step 2e: 剪枝 - 保留 Top-B                     │  │   │
│  │  │      │  │                                              │  │   │
│  │  │      │  │  candidates 按 score 排序:                  │  │   │
│  │  │      │  │  [node_a(2.1), node_b(1.8), node_c(1.5),   │  │   │
│  │  │      │  │   node_d(1.2), node_e(0.9), ...]            │  │   │
│  │  │      │  │                                              │  │   │
│  │  │      │  │  保留 Top-3:                                 │  │   │
│  │  │      │  │  current_beam = [node_a, node_b, node_c]    │  │   │
│  │  │      │  │                                              │  │   │
│  │  │      │  └─ 其余节点被剪枝                              │  │   │
│  │  │      └────────────────────────────────────────────────┘  │   │
│  │  │                                                           │   │
│  │  └─── 循环直到达到 max_depth ───────────────────────────────  │   │
│  └─────────────────────────────────────────────────────────────┘   │
│       │                                                             │
│       ▼                                                             │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  Step 3: 选择最优路径                                        │   │
│  │  │                                                           │   │
│  │  │  best_node = current_beam[0]  # 得分最高的节点            │   │
│  │  │  best_path = best_node.get_path_to_root()                │   │
│  │  │  all_contexts = best_node.get_all_contexts()             │   │
│  │  │                                                           │   │
│  │  │  最优路径:                                                 │   │
│  │  │  root → "70th Academy Awards" → "Titanic 1997" → ...     │   │
│  │  │                                                           │   │
│  │  │  收集的文档: 15 篇 (去重后)                                │   │
│  │  └─────────────────────────────────────────────────────────  │   │
│  └─────────────────────────────────────────────────────────────┘   │
│       │                                                             │
│       ▼                                                             │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  Step 4: 生成最终答案                                        │   │
│  │  │                                                           │   │
│  │  │  Prompt:                                                  │   │
│  │  │  "Based on the investigation path and documents,          │   │
│  │  │   answer the question concisely.                          │   │
│  │  │                                                           │   │
│  │  │   Question: Who is the director of the film that won...  │   │
│  │  │   Path: [70th Academy Awards → Titanic → James Cameron]  │   │
│  │  │   Documents: [doc1, doc2, ...]                            │   │
│  │  │                                                           │   │
│  │  │   Answer:"                                                │   │
│  │  │                                                           │   │
│  │  │  LLM 输出: "James Cameron"                                │   │
│  │  └─────────────────────────────────────────────────────────  │   │
│  └─────────────────────────────────────────────────────────────┘   │
│       │                                                             │
│       ▼                                                             │
│  输出: {                                                             │
│      "answer": "James Cameron",                                     │
│      "chain": "[ToT] Question: ... \n [ToT] Depth 1: ...",         │
│      "contexts": [doc1, doc2, ...]                                  │
│  }                                                                  │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 互信息评分详解

```
┌─────────────────────────────────────────────────────────────────────┐
│                    互信息 (MI) 评分原理                              │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  MI_gain = α × relevance + β × novelty                             │
│         = 0.7 × relevance + 0.3 × novelty                          │
│                                                                     │
│  ┌───────────────────────────────────────────────────────────────┐ │
│  │  Relevance (相关性): 新文档对问题的直接贡献                    │ │
│  │  │                                                            │ │
│  │  │  计算方式: Cross-Encoder(问题, 文档)                       │ │
│  │  │                                                            │ │
│  │  │  例:                                                       │ │
│  │  │  问题: "Who directed Titanic?"                             │ │
│  │  │  文档: "Titanic was directed by James Cameron in 1997"     │ │
│  │  │  → relevance = 0.92 (高度相关)                             │ │
│  │  │                                                            │ │
│  │  │  文档: "The Titanic sank in 1912"                          │ │
│  │  │  → relevance = 0.35 (部分相关)                             │ │
│  └───────────────────────────────────────────────────────────────┘ │
│                                                                     │
│  ┌───────────────────────────────────────────────────────────────┐ │
│  │  Novelty (新颖性): 新文档与已有文档的差异度                    │ │
│  │  │                                                            │ │
│  │  │  计算方式: 1 - max(Jaccard相似度)                          │ │
│  │  │                                                            │ │
│  │  │  例:                                                       │ │
│  │  │  已有文档: "James Cameron directed Titanic"                │ │
│  │  │  新文档1: "James Cameron also directed Avatar"             │ │
│  │  │  → Jaccard = 2/7 = 0.29, novelty = 0.71 (较新颖)          │ │
│  │  │                                                            │ │
│  │  │  新文档2: "James Cameron is the director of Titanic film" │ │
│  │  │  → Jaccard = 5/8 = 0.63, novelty = 0.37 (冗余)            │ │
│  └───────────────────────────────────────────────────────────────┘ │
│                                                                     │
│  ┌───────────────────────────────────────────────────────────────┐ │
│  │  为什么需要两者结合?                                           │ │
│  │  │                                                            │ │
│  │  │  只看相关性: 可能检索到重复信息                            │ │
│  │  │      → 浪费检索次数，信息增益低                            │ │
│  │  │                                                            │ │
│  │  │  只看新颖性: 可能检索到不相关但新颖的信息                  │ │
│  │  │      → 偏离主题，无法回答问题                              │ │
│  │  │                                                            │ │
│  │  │  两者结合: 既相关又新颖 = 最大信息增益                     │ │
│  └───────────────────────────────────────────────────────────────┘ │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 核心代码实现

```python
@dataclass
class TreeNode:
    thought: str              # 推理文本
    query: str               # 搜索查询
    contexts: List[Dict]     # 检索到的文档
    score: float             # 累积 MI 得分
    parent: Optional['TreeNode'] = None
    depth: int = 0

    def get_all_contexts(self) -> List[Dict]:
        """获取路径上所有文档 (去重)"""
        path = self.get_path_to_root()
        all_contexts, seen = [], set()
        for node in path:
            for ctx in node.contexts:
                key = f"{ctx['title']}_{ctx['paragraph_text'][:20]}"
                if key not in seen:
                    all_contexts.append(ctx)
                    seen.add(key)
        return all_contexts


class MutualInformationScorer:
    def calculate_mi_gain(self, question, new_docs, existing_docs) -> float:
        total = 0.0
        for doc in new_docs:
            rel = self.calculate_relevance(question, doc)  # Cross-Encoder
            nov = self.calculate_novelty(doc, existing_docs)  # 1 - Jaccard
            total += self.mi_alpha * rel + self.mi_beta * nov
        return total / len(new_docs) if new_docs else 0.0


class BeamSearchToT:
    def search(self, question: str) -> Dict:
        # 初始化
        root = TreeNode(query=question, contexts=self.retrieve(question), score=0)
        current_beam = [root]

        # 束搜索
        for depth in range(1, self.max_depth + 1):
            candidates = []
            for node in current_beam:
                for query in self.generate_candidates(question, node, k=3):
                    new_docs = self.retrieve(query)
                    mi_gain = self.mi_scorer.calculate_mi_gain(
                        question, new_docs, node.get_all_contexts()
                    )
                    child = TreeNode(query=query, contexts=new_docs,
                                    score=node.score + mi_gain, parent=node)
                    candidates.append(child)

            # 剪枝
            current_beam = sorted(candidates, key=lambda n: n.score, reverse=True)[:self.beam_width]

        # 生成答案
        return self.generate_answer(question, current_beam[0])
```

### 参数配置

| 参数 | 默认值 | 说明 |
|-----|-------|------|
| beam_width | 3 | 每层保留的路径数 |
| max_depth | 4 | 最大搜索深度 |
| candidates_per_node | 3 | 每个节点生成的候选查询数 |
| mi_alpha | 0.7 | 相关性权重 |
| mi_beta | 0.3 | 新颖性权重 |

---

## 消融实验

### 实验设计

| 实验 | 配置 | 目的 | 预期结果 |
|-----|------|------|---------|
| Model A | 完整系统 | 基准 | 最高性能 |
| Model B | w/o MI-RA-ToT | 验证创新3 | 多跳数据集 (HotpotQA, MuSiQue) 下降明显 |
| Model C | w/o Cascade | 验证创新2 | 整体 Recall 下降 |
| Model D | w/o Adaptive | 验证创新1 | SQuAD/2Wiki 效果变差 |

### 运行命令

```bash
# 运行所有消融实验
python innovation_experiments/run_ablation_experiments.py

# 运行特定实验
python innovation_experiments/run_ablation_experiments.py --experiments A B

# 仅对比结果
python innovation_experiments/run_ablation_experiments.py --compare-only
```

---

## 快速开始

### 1. 环境准备
```bash
pip install spacy sentence-transformers numpy
python -m spacy download en_core_web_sm
```

### 2. 启动服务
```bash
# Terminal 1: V2 检索器 (端口 8002)
python innovation_experiments/retriever_server_v2/serve_v2.py

# Terminal 2: LLM 服务器 (端口 8000)
python llm_server/serve_llama.py
```

### 3. 运行实验
```bash
# 运行 V2 流程
python innovation_experiments/evaluate_v2/run_pipeline_v2.py --stages 2 3

# 分析结果
python innovation_experiments/run_all_analysis.py
```

---

## 预期效果

| 创新点 | EM 增益 | F1 增益 | 主要受益数据集 |
|-------|---------|---------|--------------|
| 自适应检索 | +0.02~0.05 | +0.03~0.06 | SQuAD, 2Wiki |
| 级联路由 | +0.03~0.08 | +0.04~0.10 | 全部 |
| MI-RA-ToT | +0.05~0.12 | +0.06~0.15 | HotpotQA, MuSiQue |
| **综合** | **+0.10~0.25** | **+0.13~0.31** | - |
