# RAG 路由分类训练数据生成项目

## 📋 项目概述

本项目使用 OpenAI 的 **Structured Outputs** 功能，基于 `processed_data` 中的数据生成 RAG 路由分类器的训练数据。

## 🎯 核心目标

为每个查询自动标注：
- **复杂度级别** (L0/L1/L2)
- **索引策略** (None/Lexical/Semantic/Hybrid)
- **执行动作** (Z/S-Sparse/S-Dense/S-Hybrid/M)
- **推理过程** (详细的分析说明)

## 📁 文件结构

```
classifier/data/
├── generate_training_data.py    # 主生成脚本 ⭐
├── test_single_query.py          # 单查询测试脚本
├── load_and_use_config.py        # 配置文件加载脚本
├── run_generation.sh             # Linux/Mac 启动脚本
├── run_generation.bat            # Windows 启动脚本
├── requirements.txt              # Python 依赖
├── config.example.json           # 配置文件示例
├── README.md                     # 详细使用文档
└── PROJECT_OVERVIEW.md           # 本文件
```

## 🚀 快速开始 (3 步)

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 配置 config.json
cp config.example.json config.json
# 编辑 config.json，设置你的 API key

# 3. 直接运行
python generate_training_data.py
```

## 🔧 主要功能

### 1. generate_training_data.py
- ✅ 异步并发处理
- ✅ OpenAI Structured Outputs (parse method)
- ✅ 完整的 Pydantic Schema
- ✅ 自动重试和错误处理
- ✅ 进度条显示
- ✅ 标签分布统计
- ✅ 支持命令行参数

**使用方法：**
```bash
# 1. 配置 config.json
cp config.example.json config.json
nano config.json  # 编辑配置

# 2. 直接运行
python generate_training_data.py
```

**配置示例：**
```json
{
  "api_settings": {
    "api_key": "sk-...",
    "model": "gpt-4o-mini",
    "max_concurrent": 10
  },
  "data_settings": {
    "datasets": null  // null = 处理全部数据集
  }
}
```

### 2. test_single_query.py
测试单个或多个查询的分类效果

**使用示例：**
```bash
# 测试单个查询
python test_single_query.py "Who is the president of France?"

# 运行内置测试集
python test_single_query.py
```

### 3. 交互式启动脚本
提供友好的交互式界面，无需记忆命令行参数

```bash
./run_generation.sh        # Linux/Mac
run_generation.bat          # Windows
```

### 4. 配置文件支持
```bash
# 复制配置模板
cp config.example.json config.json

# 编辑配置
nano config.json

# 使用配置运行
python load_and_use_config.py
```

## 📊 数据流程

```
processed_data/
  ├── nq/dev_500_subsampled.jsonl
  ├── squad/dev_500_subsampled.jsonl
  └── ...
       ↓
  [OpenAI API + Structured Outputs]
       ↓
training_data/
  ├── nq_training.jsonl
  ├── squad_training.jsonl
  ├── combined_training_data.jsonl
  └── ...
```

## 🏷️ 分类体系

### Complexity Label
- **L0**: 代码生成、逻辑推理、翻译 → 不需要检索
- **L1**: 单实体查询 → 单跳检索
- **L2**: 比较、多实体、嵌套逻辑 → 多跳检索

### Index Strategy
- **None**: L0 专用
- **Lexical**: 精确实体、专有名词 (BM25)
- **Semantic**: 描述性、抽象概念 (Vector)
- **Hybrid**: 复杂查询 (Both)

### Action (最终路由)
```
L0        → Z (Zero Retrieval)
L1+Lexical → S-Sparse (BM25)
L1+Semantic → S-Dense (Vector)
L1+Hybrid  → S-Hybrid (Hybrid)
L2        → M (Multi-hop)
```

## 📈 输出示例

```json
{
  "dataset": "nq",
  "question_id": "single_nq_dev_2389",
  "question_text": "who sings i wont let the sun go down on me",
  "reasoning": "This query asks for a specific artist associated with a specific song title. Both 'I Won't Let the Sun Go Down on Me' and the artist name are precise entities that can be matched with keywords. Lexical search is optimal.",
  "complexity_label": "L1",
  "index_strategy": "Lexical",
  "action": "S-Sparse",
  "answers": [{"spans": ["Elton John"]}]
}
```

## ⚙️ 高级配置

### 配置文件结构
```json
{
  "api_settings": {
    "api_key": "your-key-or-blank-for-env",
    "base_url": null,
    "model": "gpt-4o-mini",
    "max_concurrent": 10
  },
  "data_settings": {
    "data_dir": "../../processed_data",
    "output_dir": "./training_data",
    "datasets": null
  }
}
```

### 环境变量(可选)
```bash
# API key 也可以通过环境变量设置
export OPENAI_API_KEY="sk-..."  # Linux/Mac
set OPENAI_API_KEY=sk-...       # Windows
```

## 🐛 常见问题

### Q: config.json not found 错误怎么办？
A: 复制示例配置文件: `cp config.example.json config.json`，然后编辑设置

### Q: 如何处理 API 速率限制？
A: 在 config.json 中降低 `max_concurrent` 值（从 10 降到 5）

### Q: No API key provided 错误？
A: 在 config.json 中设置 `api_key`，或设置环境变量 `OPENAI_API_KEY`

### Q: 如何使用代理或自定义端点？
A: 在 config.json 的 `api_settings` 中设置 `base_url`

### Q: 如何只处理部分数据集？
A: 在 config.json 中设置 `"datasets": ["nq", "squad"]`

### Q: 生成的标签准确吗？
A: 建议先用 `test_single_query.py` 测试样本，确认符合预期

### Q: 处理全部数据需要多久？
A: 约 3000 条数据，通常 10-30 分钟（取决于并发数和网络速度）

## 💰 成本估算

- **gpt-4o-mini**: ~$0.15 / 1M tokens
- **预估**: 3000 条查询 × 平均 500 tokens ≈ $0.0008-0.001 每条
- **总计**: 全部数据集约 $2.4-3.0

## 📝 开发建议

1. **先小规模测试**: `--datasets nq --max-concurrent 5`
2. **检查输出质量**: 查看 `training_data/*.jsonl`
3. **调整提示词**: 根据实际效果优化 `SYSTEM_PROMPT`
4. **批量处理**: 确认无误后处理全部数据

## 🔗 相关资源

- [OpenAI Structured Outputs 文档](https://platform.openai.com/docs/guides/structured-outputs)
- [Pydantic 文档](https://docs.pydantic.dev/)
- [详细使用说明](./README.md)

## 📞 支持

如有问题，请检查：
1. API Key 是否正确设置
2. 依赖是否完整安装
3. 数据路径是否正确
4. 网络连接是否正常

---

**版本**: 1.0
**最后更新**: 2025-12-08
**作者**: AI Generated Script
