# 快速开始指南

## 前置要求

1. **已训练的分类器模型**
   - 位置：`/root/autodl-tmp/output/Qwen2.5-3B-RAG-Router/lora_model`

2. **Llama 3-8B模型**
   - 位置：`/root/autodl-tmp/model/Meta-Llama-3-8B-Instruct`

3. **测试数据集**
   - 位置：`processed_data/{dataset}/test_subsampled.jsonl`
   - 6个数据集，每个500个样本

## 步骤1：安装依赖

```bash
pip install unsloth transformers peft torch pyyaml tqdm requests fastapi uvicorn
```

## 步骤2：启动服务

### 终端1：启动Llama服务器（自动批处理）

```bash
# 推荐：使用自动批处理服务器（最佳性能）
bash start_llm_autobatch.sh

# 或使用基础批处理服务器
bash start_llm_server.sh
```

**自动批处理的优势：**
- 自动收集并发请求并批量处理
- GPU利用率提升至95-100%
- 相比单请求处理可获得3-5x加速

### 终端2：确保检索服务运行

检索服务应该已经在运行（`localhost:8001`）。如果没有，请启动它。

## 步骤3：验证配置

检查 `evaluate/config.yaml` 中的路径是否正确：

```yaml
classifier:
  base_model_path: "/root/autodl-tmp/model/Qwen2.5-3B-Instruct"
  lora_adapter_path: "/root/autodl-tmp/output/Qwen2.5-3B-RAG-Router/lora_model"

llm:
  model_path: "/root/autodl-tmp/model/Meta-Llama-3-8B-Instruct"
  server_host: "localhost"
  server_port: 8000

retriever:
  host: "localhost"
  port: 8001
```

## 步骤4：运行流程

### 选项A：运行完整流程（推荐用于首次运行）

```bash
cd /root/graduateRAG  # 或你的项目根目录
python evaluate/run_pipeline.py
```

这将：
1. 分类所有3000个问题（约15分钟）
2. 根据分类生成答案（约4-8小时）
3. 计算指标（约2分钟）

### 选项B：分阶段运行（用于调试）

```bash
# 仅阶段1：分类
python evaluate/run_pipeline.py --stages 1

# 检查分类结果
cat evaluate/outputs/stage1_classifications/squad_classifications.jsonl | head

# 仅阶段2：生成
python evaluate/run_pipeline.py --stages 2

# 检查预测结果
cat evaluate/outputs/stage2_predictions/squad_predictions.json | jq . | head

# 仅阶段3：评估
python evaluate/run_pipeline.py --stages 3

# 查看结果
cat evaluate/outputs/stage3_metrics/overall_metrics.json | jq .
cat evaluate/outputs/stage3_metrics/detailed_report.txt
```

### 选项C：测试运行（单个数据集）

```bash
# 仅处理squad数据集进行测试
python evaluate/run_pipeline.py --datasets squad
```

## 步骤5：查看结果

### 分类分布

```bash
# 查看squad的分类分布
python -c "
import json
with open('evaluate/outputs/stage1_classifications/squad_classifications.jsonl') as f:
    actions = [json.loads(line)['predicted_action'] for line in f]
from collections import Counter
print(Counter(actions))
"
```

### 评估指标

```bash
# 查看总体指标
cat evaluate/outputs/stage3_metrics/overall_metrics.json | jq '.overall'

# 查看详细报告
cat evaluate/outputs/stage3_metrics/detailed_report.txt
```

## 常见问题

### Q: 如何跳过服务健康检查？

```bash
python evaluate/run_pipeline.py --no-service-check
```

### Q: 如何只处理特定数据集？

```bash
python evaluate/run_pipeline.py --datasets squad hotpotqa
```

### Q: 流程中断了，如何继续？

只需重新运行相同的命令。系统会自动跳过已处理的项目。

### Q: 如何查看日志？

```bash
tail -f evaluate/pipeline.log
```

### Q: 如何更改Llama服务器端口？

编辑 `evaluate/config.yaml`：

```yaml
llm:
  server_host: "localhost"
  server_port: 8001  # 更改端口
```

## 预期输出示例

### 阶段1输出示例

```
==============================================================
STAGE 1: CLASSIFICATION
==============================================================
Loading classifier model...
✓ Classifier loaded successfully!

==============================================================
Processing dataset: squad
==============================================================
Loaded 500 examples from squad
Total questions: 500
Already processed: 0
Remaining: 500
Classifying squad: 100%|████████| 500/500 [08:45<00:00,  1.05s/it]

✓ Classification complete for squad
  Output saved to: evaluate/outputs/stage1_classifications/squad_classifications.jsonl

Action Distribution:
  Z           :   45 ( 9.0%)
  S-Sparse    :  280 (56.0%)
  S-Dense     :   95 (19.0%)
  S-Hybrid    :   60 (12.0%)
  M           :   20 ( 4.0%)
```

### 阶段2输出示例

```
==============================================================
STAGE 2: GENERATION
==============================================================

==============================================================
Generating for dataset: squad
==============================================================
Loaded 500 examples from squad
Question distribution by action:
  Z           :   45 questions
  S-Sparse    :  280 questions
  S-Dense     :   95 questions
  S-Hybrid    :   60 questions
  M           :   20 questions

Processing action: Z
  Processing 45/45 questions
    Running: python -m commaqa.inference.configurable_inference ...
  ✓ Completed 45 predictions for action Z

Processing action: S-Sparse
  Processing 280/280 questions
    Running: python -m commaqa.inference.configurable_inference ...
  ✓ Completed 280 predictions for action S-Sparse
...
```

### 阶段3输出示例

```
==============================================================
STAGE 3: EVALUATION
==============================================================

==============================================================
Evaluating dataset: squad
==============================================================
Loaded 500 examples from squad

✓ Metrics saved to evaluate/outputs/stage3_metrics/overall_metrics.json
✓ Detailed report saved to evaluate/outputs/stage3_metrics/detailed_report.txt

================================================================================
EVALUATION RESULTS
================================================================================

squad:
  Overall: EM=0.4560, F1=0.6230, Count=500
  By Action:
    Z           : EM=0.3011, F1=0.4450, Count=45
    S-Sparse    : EM=0.5121, F1=0.6870, Count=280
    S-Dense     : EM=0.4979, F1=0.6710, Count=95
    S-Hybrid    : EM=0.5333, F1=0.7120, Count=60
    M           : EM=0.4231, F1=0.5890, Count=20

================================================================================
OVERALL ACROSS ALL DATASETS
================================================================================
EM: 0.4120
F1: 0.5780
Total Questions: 3000
```

## 下一步

1. **分析结果**：查看哪些动作类型表现最好/最差
2. **错误分析**：检查失败的预测
3. **调优**：调整检索数量、提示模板等
4. **消融研究**：比较分类器路由与oracle路由

## 获取帮助

如有问题，请查看：
- `evaluate/README.md` - 完整文档
- `evaluate/pipeline.log` - 运行日志
- GitHub Issues - 报告问题
