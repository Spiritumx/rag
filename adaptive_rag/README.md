# Adaptive-RAG 复现

基于论文 "Adaptive-RAG: Learning to Adapt Retrieval-Augmented Large Language Models through Question Complexity" 的复现实现。

## 核心思路

**结果导向的数据标注**：通过实际执行不同的 RAG 策略，根据答案正确性自动确定最优策略标签。

### 标注逻辑

```
对每个问题:
1. 执行 Zero Retrieval (直接 LLM 回答)
   - 如果答案正确 → 标签 = "Z"
2. 否则执行 Single Retrieval (BM25 检索 + LLM)
   - 如果答案正确 → 标签 = "S"
3. 否则 → 标签 = "M" (多跳检索)
```

## 目录结构

```
adaptive_rag/
├── __init__.py
├── README.md
├── config.yaml                           # 主配置
├── data/
│   ├── __init__.py
│   ├── sample_questions.py               # 从6个数据集采样1000条
│   ├── generate_labels.py                # 执行策略并自动标注
│   └── training_data/                    # 生成的训练数据
├── train/
│   ├── __init__.py
│   └── finetune.py                       # LoRA 微调
└── evaluate/
    ├── __init__.py
    ├── config.yaml                       # 评估配置
    ├── stage1_classify.py                # 分类
    ├── stage2_generate.py                # 生成答案
    ├── stage3_evaluate.py                # 计算指标
    └── run_pipeline.py                   # 一键评估
```

## 使用流程

### 前置条件

1. 启动 LLM 服务 (端口 8000)
2. 启动检索服务 (端口 8001)
3. 确保 `processed_data/` 目录下有各数据集的数据文件

### Step 1: 数据采样

从 6 个数据集中均匀采样 1000 条问题用于训练：

```bash
python -m adaptive_rag.data.sample_questions

# 自定义采样数量
python -m adaptive_rag.data.sample_questions --total 500
```

输出：`adaptive_rag/data/training_data/sampled_questions.jsonl`

### Step 2: 结果导向标注

执行 Z/S 策略并根据结果自动标注：

```bash
python -m adaptive_rag.data.generate_labels

# 并行加速
python -m adaptive_rag.data.generate_labels --workers 8

# 从检查点恢复
python -m adaptive_rag.data.generate_labels --resume
```

输出：`adaptive_rag/data/training_data/adaptive_rag_training.jsonl`

**注意**：此步骤需要 LLM 和检索服务运行中。

### Step 3: 训练分类器

使用标注数据微调 Qwen2.5-3B：

```bash
python -m adaptive_rag.train.finetune
```

输出：`/root/autodl-tmp/output/Qwen2.5-3B-Adaptive-RAG-Router/lora_model`

### Step 4: 评估

运行完整评估流程：

```bash
# 运行全部阶段
python -m adaptive_rag.evaluate.run_pipeline

# 只运行特定阶段
python -m adaptive_rag.evaluate.run_pipeline --stages 1 2 3

# 只评估特定数据集
python -m adaptive_rag.evaluate.run_pipeline --datasets squad hotpotqa

# 单独运行各阶段
python -m adaptive_rag.evaluate.stage1_classify
python -m adaptive_rag.evaluate.stage2_generate
python -m adaptive_rag.evaluate.stage3_evaluate
```

## 配置说明

### 主配置 (`config.yaml`)

```yaml
# LLM 服务
llm:
  host: "localhost"
  port: 8000

# 检索服务
retriever:
  host: "localhost"
  port: 8001

# 数据集
datasets:
  - squad
  - hotpotqa
  - trivia
  - nq
  - musique
  - 2wikimultihopqa

# Corpus 映射
corpus_mapping:
  hotpotqa: hotpotqa
  musique: musique
  2wikimultihopqa: 2wikimultihopqa
  squad: wiki
  trivia: wiki
  nq: wiki
```

### 评估配置 (`evaluate/config.yaml`)

```yaml
# 分类器配置 (指向训练好的 LoRA)
classifier:
  base_model_path: "/root/autodl-tmp/model/Qwen2.5-3B-Instruct"
  lora_adapter_path: "/root/autodl-tmp/output/Qwen2.5-3B-Adaptive-RAG-Router/lora_model"
```

## 策略说明

| 标签 | 策略 | 复杂度 | 描述 |
|------|------|--------|------|
| Z | Zero Retrieval | L0 | 直接 LLM 回答，无需检索 |
| S | Single BM25 | L1 | 单次 BM25 稀疏检索 |
| M | Multi-hop | L2 | 多跳推理 (IRCoT) |

## 预期结果

### 标签分布 (参考)

- Z (不检索): ~20-30%
- S (单次 BM25 检索): ~40-50%
- M (多次检索): ~20-30%

### 性能对比

训练完成后，对比 Adaptive-RAG 分类器与原始分类器的性能：

1. 在测试集上的分类准确率
2. 最终 QA 任务的 EM/F1/ACC 指标

## 依赖文件

| 文件 | 用途 |
|------|------|
| `metrics/squad_answer_em_f1.py` | 答案正确性判断 |
| `evaluate/M_core.py` | 多跳推理实现 |
| `evaluate/utils/*` | 配置/数据加载工具 |

## 常见问题

### 服务连接失败

确保 LLM 服务和检索服务已启动：

```bash
# 检查 LLM 服务
curl "http://localhost:8000/generate?prompt=test&max_length=10"

# 检查检索服务
curl -X POST "http://localhost:8001/retrieve/" \
  -H "Content-Type: application/json" \
  -d '{"query_text": "test", "corpus_name": "wiki", "max_hits_count": 1}'
```

### 标注速度慢

- 增加并行 workers: `--workers 8`
- 使用检查点恢复: `--resume`

### 内存不足

- 减少批处理大小 (修改 `config.yaml` 中的 `batch_size`)
- 减少并行线程数 (修改 `parallel_workers`)
