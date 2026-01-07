# 上限测试快速参考

## 🚀 一键运行（最简单）

```bash
cd evaluate/upper_bound_analysis
python run_all_tests.py
```

**就这么简单！** 8-12 分钟后你会得到完整的诊断报告。

---

## 📋 常用命令

### 快速验证（3分钟）
```bash
python run_all_tests.py --max-samples 20
```

### 标准诊断（8分钟）⭐ 推荐
```bash
python run_all_tests.py --max-samples 100
```

### 单数据集测试
```bash
python run_all_tests.py --datasets musique --max-samples 100
```

### 跳过某些测试
```bash
# 只运行 Reader 测试
python run_all_tests.py --skip-retriever --skip-agent

# 跳过 Retriever（如果还没运行 Stage 2）
python run_all_tests.py --skip-retriever
```

### 高性能模式（12线程）
```bash
python run_all_tests.py --parallel-threads 12
```

---

## 📂 结果在哪里？

运行完成后，查看这些文件：

### 1️⃣ 诊断报告（最重要）⭐
```
reader_upper_bound/outputs/{dataset}/diagnosis_report_standard.md
retriever_recall_upperbound/outputs/{dataset}/diagnosis_report.md
```

**这些报告会告诉你：**
- 系统瓶颈在哪里（Reader？Retriever？Agent？）
- 具体的优化建议
- 是否需要换模型、优化 Prompt 或改进检索策略

### 2️⃣ 评估指标
```
reader_upper_bound/outputs/{dataset}/metrics_standard.json
retriever_recall_upperbound/outputs/{dataset}/recall_metrics.json
agent_reasoning_check/outputs/{dataset}/metrics_local_llama.json
```

**关键指标：**
- Reader EM (Exact Match)
- Recall@5, Recall@20, Recall@100
- Agent EM

### 3️⃣ 测试摘要
```
outputs/all_tests_summary.json
```

**包含：**
- 所有测试的成功/失败状态
- 总耗时
- 配置信息

---

## 🎯 如何解读结果？

### 决策树

```
1. 查看 Reader EM
   ├─ EM < 30%  → LLM 太弱，必须换模型
   ├─ EM 30-70% → LLM 中等，可以优化 Prompt
   └─ EM > 70%  → LLM 很好，瓶颈在检索 ↓

2. 查看 Recall@100（如果 Reader EM > 70%）
   ├─ < 50%  → 粗排太差，需要混合检索/增加 top_k
   └─ > 80%  → 粗排 OK ↓

3. 查看 Recall@5（如果 Recall@100 > 80%）
   ├─ < 40% → 精排太差，需要更好的 Reranker
   └─ > 60% → 检索很好，瓶颈在 LLM ↓

4. 对比 Agent EM（baseline vs GPT-4）
   ├─ GPT-4 提升 > 30% → 需要 ToT 增强推理
   └─ GPT-4 提升 < 10% → 检索是主要问题
```

---

## ⚠️ 常见问题

### Retriever 测试被跳过？

**原因：** 没有检索结果文件

**解决：**
1. 先运行 Stage 2：
   ```bash
   cd evaluate
   python stage2_generate.py --dataset musique
   ```

2. 或者跳过：
   ```bash
   python run_all_tests.py --skip-retriever
   ```

### 服务连接失败？

**检查 LLM 服务：**
```bash
curl http://localhost:8000/generate?prompt=test&max_length=10
```

**检查 Retriever 服务：**
```bash
curl -X POST http://localhost:8001/search \
  -H "Content-Type: application/json" \
  -d '{"question": "test", "top_k": 5}'
```

---

## 📚 详细文档

- **一键脚本详细说明：** `RUN_ALL_TESTS_GUIDE.md`
- **框架总览：** `README.md`
- **配置说明：** `CONFIG_USAGE.md`
- **并行处理：** `PARALLEL_UPGRADE_SUMMARY.md`

---

## 🎓 典型工作流程

### 第一次使用

```bash
# 1. 快速验证（3分钟）
python run_all_tests.py --max-samples 20

# 2. 如果成功，运行标准测试（8分钟）
python run_all_tests.py --max-samples 100

# 3. 查看诊断报告
cat reader_upper_bound/outputs/musique/diagnosis_report_standard.md
cat retriever_recall_upperbound/outputs/musique/diagnosis_report.md
```

### 优化后验证

```bash
# 1. 运行测试
python run_all_tests.py --max-samples 100

# 2. 对比优化前后的指标
diff \
  reader_upper_bound/outputs/musique/metrics_standard.json \
  reader_upper_bound/outputs/musique/metrics_standard.json.backup
```

### 论文实验

```bash
# 1. 修改 config.yaml: max_samples: null

# 2. 运行完整测试（1-2小时）
python run_all_tests.py

# 3. 生成性能分解图（论文用）
cd agent_reasoning_check
python analyze_performance_breakdown.py --dataset musique
```

---

**需要帮助？** 查看详细文档或运行 `python run_all_tests.py --help`
