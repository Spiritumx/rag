# RAG + Classifier Evaluation Pipeline

三阶段解耦执行的RAG系统评测流程，结合分类器进行路由决策。

## 概述

该系统实现了一个完整的评测流程：

1. **阶段1：分类** - 使用训练好的Qwen 2.5-3B LoRA分类器对测试集进行分类
2. **阶段2：生成** - 根据分类结果路由到不同的RAG流程并生成答案
3. **阶段3：评估** - 计算EM和F1指标，按数据集和动作类型汇总

## 目录结构

```
evaluate/
├── config.yaml                          # 主配置文件
├── run_pipeline.py                      # 主流程调度器
├── stage1_classify.py                   # 阶段1：分类
├── stage2_generate.py                   # 阶段2：生成
├── stage3_evaluate.py                   # 阶段3：评估
├── utils/                               # 工具模块
│   ├── classifier_loader.py             # 分类器加载
│   ├── config_loader.py                 # 配置加载
│   ├── service_checker.py               # 服务健康检查
│   ├── data_loader.py                   # 数据集加载
│   ├── llama_generator.py               # Llama服务器管理
│   ├── llama_fastapi_server.py          # Llama FastAPI服务器（备选）
│   └── result_manager.py                # 结果管理
├── configs/                             # 配置文件
│   ├── action_to_config_mapping.py      # 动作到配置的映射
│   └── llama_configs/                   # Llama专用JSONNET配置
│       ├── zero_retrieval.jsonnet       # Z动作（零检索）
│       ├── single_bm25.jsonnet          # S-Sparse动作
│       ├── single_dense.jsonnet         # S-Dense动作
│       ├── single_hybrid.jsonnet        # S-Hybrid动作
│       └── multi_hop.jsonnet            # M动作（多跳）
└── outputs/                             # 输出结果
    ├── stage1_classifications/          # 分类结果
    ├── stage2_predictions/              # 预测结果
    └── stage3_metrics/                  # 评估指标
```

## 安装依赖

```bash
pip install unsloth transformers peft torch
pip install pyyaml tqdm requests
pip install vllm  # 推荐用于Llama服务
# 或者
pip install fastapi uvicorn  # 备选方案
```

## 配置

编辑 `evaluate/config.yaml`：

```yaml
# 分类器设置
classifier:
  base_model_path: "/root/autodl-tmp/model/Qwen2.5-3B-Instruct"
  lora_adapter_path: "/root/autodl-tmp/output/Qwen2.5-3B-RAG-Router/lora_model"

# LLM设置（Llama 3-8B）
llm:
  model_path: "/root/autodl-tmp/model/Meta-Llama-3-8B-Instruct"
  server_host: "localhost"
  server_port: 8000

# 检索服务设置
retriever:
  host: "localhost"
  port: 8001

# 要处理的数据集
datasets:
  - squad
  - hotpotqa
  - trivia
  - nq
  - musique
  - 2wikimultihopqa
```

## 使用方法

### 1. 启动必要的服务

#### 启动Llama服务器（选项A - vLLM，推荐）

```bash
python -m vllm.entrypoints.api_server \
    --model /root/autodl-tmp/model/Meta-Llama-3-8B-Instruct \
    --host localhost \
    --port 8000 \
    --dtype auto
```

#### 启动Llama服务器（选项B - FastAPI）

```bash
uvicorn evaluate.utils.llama_fastapi_server:app \
    --host localhost \
    --port 8000
```

#### 启动检索服务

确保检索服务运行在 `localhost:8001`。

### 2. 运行完整流程

```bash
# 运行所有三个阶段
python evaluate/run_pipeline.py

# 或指定阶段
python evaluate/run_pipeline.py --stages 1 2 3
```

### 3. 分阶段运行

```bash
# 仅运行分类阶段
python evaluate/run_pipeline.py --stages 1

# 仅运行生成阶段
python evaluate/run_pipeline.py --stages 2

# 仅运行评估阶段
python evaluate/run_pipeline.py --stages 3
```

### 4. 处理特定数据集

```bash
# 仅处理squad和hotpotqa
python evaluate/run_pipeline.py --datasets squad hotpotqa
```

### 5. 跳过服务检查

```bash
python evaluate/run_pipeline.py --no-service-check
```

## 分阶段运行

每个阶段都可以独立运行：

### 阶段1：分类

```bash
python evaluate/stage1_classify.py --config evaluate/config.yaml
```

输出：`evaluate/outputs/stage1_classifications/{dataset}_classifications.jsonl`

格式：
```json
{
  "question_id": "single_nq_dev_2389",
  "question_text": "who sings i wont let the sun go down on me",
  "predicted_action": "S-Sparse",
  "full_response": "...",
  "dataset": "nq"
}
```

### 阶段2：生成

```bash
python evaluate/stage2_generate.py --config evaluate/config.yaml
```

输出：`evaluate/outputs/stage2_predictions/{dataset}_predictions.json`

格式：
```json
{
  "single_nq_dev_2389": "Elton John",
  "single_squad_dev_4465": "Mona Lisa Smile",
  ...
}
```

### 阶段3：评估

```bash
python evaluate/stage3_evaluate.py --config evaluate/config.yaml
```

输出：
- `evaluate/outputs/stage3_metrics/overall_metrics.json` - JSON格式的指标
- `evaluate/outputs/stage3_metrics/detailed_report.txt` - 详细报告

## 动作类型映射

分类器预测5种动作类型：

- **Z**: 零检索（直接LLM生成）
- **S-Sparse**: 单跳BM25检索
- **S-Dense**: 单跳密集向量检索（HNSW）
- **S-Hybrid**: 单跳混合检索
- **M**: 多跳推理（IRCoT）

## 断点续传

所有阶段都支持断点续传：

- 如果中断，重新运行相同命令会跳过已处理的项目
- 每10个问题自动保存检查点
- 可以安全地中断和重启流程

## 输出文件

### 分类结果

位置：`evaluate/outputs/stage1_classifications/`

每个数据集一个JSONL文件，包含每个问题的分类结果。

### 预测结果

位置：`evaluate/outputs/stage2_predictions/`

每个数据集一个JSON文件，映射question_id到预测答案。

### 评估指标

位置：`evaluate/outputs/stage3_metrics/`

- `overall_metrics.json`: 所有指标的JSON格式
- `detailed_report.txt`: 人类可读的详细报告

示例指标输出：
```json
{
  "squad": {
    "overall": {"em": 0.456, "f1": 0.623, "count": 500},
    "by_action": {
      "Z": {"em": 0.301, "f1": 0.445, "count": 123},
      "S-Sparse": {"em": 0.512, "f1": 0.687, "count": 245},
      ...
    }
  },
  "overall": {
    "em": 0.412,
    "f1": 0.578,
    "count": 3000
  }
}
```

## 预计运行时间

- **阶段1（分类）**: 约10-15分钟（3000个问题）
- **阶段2（生成）**: 约4-8小时（取决于动作分布）
  - Z（零检索）: ~2秒/问题
  - S-*（单跳）: ~5-10秒/问题
  - M（多跳）: ~15-30秒/问题
- **阶段3（评估）**: 约1-2分钟

**总计**: 约4-8小时完整流程

## 故障排除

### 问题：分类器加载失败

```
ModuleNotFoundError: No module named 'unsloth'
```

解决：
```bash
pip install unsloth
```

### 问题：LLM服务器未运行

```
RuntimeError: Cannot connect to LLM server
```

解决：确保Llama服务器运行在配置的主机和端口上。

### 问题：检索服务未运行

```
RuntimeError: Cannot connect to retriever
```

解决：确保检索服务运行在配置的主机和端口上。

### 问题：缺少配置文件

```
FileNotFoundError: Config file not found
```

解决：确保 `evaluate/configs/llama_configs/` 中存在所有5个JSONNET配置。

## 日志

日志输出到：
- 控制台（标准输出）
- `evaluate/pipeline.log`（配置中可修改）

## 环境变量

流程使用以下环境变量：

```bash
RETRIEVER_HOST=localhost
RETRIEVER_PORT=8001
LLM_SERVER_HOST=localhost
LLM_SERVER_PORT=8000
```

这些会从配置文件自动设置，无需手动导出。

## 许可证

与主项目相同。
