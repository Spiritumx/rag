# RAG 路由分类训练数据生成

## 概述

这个脚本使用 OpenAI 的 Structured Outputs 功能,基于 `processed_data` 中的数据生成 RAG 路由分类器的训练数据。

**⭐ 特点**:
- ✅ 基于 config.json 配置文件运行,无需复杂的命令行参数
- ✅ 完善的错误处理和友好的错误提示
- ✅ 自动检测和验证配置
- ✅ 支持速率限制处理和智能重试

## 快速开始 (3步)

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 配置 config.json
cp config.example.json config.json
# 编辑 config.json，设置你的 API key

# 3. 运行
python generate_training_data.py
```

## 依赖安装

```bash
pip install openai pydantic tqdm
```

或使用 requirements.txt:

```bash
pip install -r requirements.txt
```

## 配置说明

### 创建配置文件

```bash
# 复制示例配置
cp config.example.json config.json

# 编辑配置文件
nano config.json  # Linux/Mac
notepad config.json  # Windows
```

### 配置文件格式

```json
{
  "api_settings": {
    "api_key": "your-api-key-or-leave-blank-for-env-var",
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

### 配置项说明

#### api_settings

- `api_key`: OpenAI API密钥
  - 可以直接填写,例如 `"sk-..."`
  - 或留空/填`""`使用环境变量 `OPENAI_API_KEY`
- `base_url`: API端点地址 (可选)
  - 默认 `null` 使用官方端点
  - 可设置自定义端点,如使用代理
- `model`: 使用的模型
  - `"gpt-4o-mini"` - 成本低,速度快 (推荐)
  - `"gpt-4o"` - 效果更好,成本更高
- `max_concurrent`: 最大并发请求数
  - 默认 `10`
  - 建议范围: 5-20
  - 过高可能触发速率限制

#### data_settings

- `data_dir`: 输入数据目录 (相对于脚本的路径)
  - 默认: `"../../processed_data"`
- `output_dir`: 输出目录
  - 默认: `"./training_data"`
- `datasets`: 要处理的数据集列表
  - `null` - 处理所有6个数据集 (推荐)
  - `["nq"]` - 只处理 NQ 数据集
  - `["nq", "squad", "hotpotqa"]` - 处理指定的多个数据集
  - 可选数据集: `2wikimultihopqa`, `hotpotqa`, `musique`, `nq`, `squad`, `trivia`

## 使用方法

### 基本用法

```bash
# 确保 config.json 已配置
python generate_training_data.py
```

### 使用环境变量设置 API Key

```bash
# Linux/Mac
export OPENAI_API_KEY="sk-..."
python generate_training_data.py

# Windows CMD
set OPENAI_API_KEY=sk-...
python generate_training_data.py

# Windows PowerShell
$env:OPENAI_API_KEY="sk-..."
python generate_training_data.py
```

### 配置示例

**示例 1: 处理所有数据集**
```json
{
  "api_settings": {
    "api_key": "",
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

**示例 2: 只处理特定数据集**
```json
{
  "data_settings": {
    "datasets": ["nq", "squad"]
  }
}
```

**示例 3: 使用自定义端点**
```json
{
  "api_settings": {
    "base_url": "https://your-proxy.com/v1",
    "max_concurrent": 5
  }
}

## 输出格式

脚本会生成以下文件:

1. **单个数据集文件**: `training_data/{dataset_name}_training.jsonl`
2. **合并文件**: `training_data/combined_training_data.jsonl`

每条训练数据的格式:

```json
{
  "dataset": "nq",
  "question_id": "single_nq_dev_2389",
  "question_text": "who sings i wont let the sun go down on me",
  "reasoning": "详细的推理过程...",
  "complexity_label": "L1",
  "index_strategy": "Lexical",
  "action": "S-Sparse",
  "answers": [...]
}
```

## 标签说明

### complexity_label (复杂度标签)
- `L0`: 非事实性/逻辑/代码类问题
- `L1`: 单跳事实性问题
- `L2`: 多跳/比较/复杂问题

### index_strategy (索引策略)
- `None`: 不需要索引
- `Lexical`: 词法索引(精确实体)
- `Semantic`: 语义索引(模糊概念)
- `Hybrid`: 混合索引(复杂需求)

### action (执行动作)
- `Z`: 零检索(直接生成)
- `S-Sparse`: 稀疏检索(BM25)
- `S-Dense`: 密集检索(向量)
- `S-Hybrid`: 混合检索
- `M`: 多跳检索

## 错误处理

脚本包含完善的错误处理机制:

### 自动处理的错误

1. **配置文件错误**
   - config.json 不存在 → 友好提示如何创建
   - JSON 格式错误 → 显示具体的解析错误
   - 缺少 API key → 提示两种配置方法

2. **数据文件错误**
   - 数据目录不存在 → 显示完整路径并退出
   - JSONL 文件格式错误 → 跳过错误行,继续处理
   - 空数据文件 → 警告并跳过

3. **API 调用错误**
   - 速率限制(429) → 自动使用更长的等待时间重试
   - 网络错误 → 指数退避重试(最多3次)
   - 其他错误 → 显示详细错误信息

### 常见错误示例

**错误: config.json not found**
```
❌ ERROR: config.json not found!

Please create config.json based on config.example.json:
  1. Copy the example: cp config.example.json config.json
  2. Edit config.json and set your API key and other settings
```

**错误: No API key**
```
❌ ERROR: No API key provided!

Please provide an API key using one of these methods:
  1. Set 'api_key' in config.json
  2. Set OPENAI_API_KEY environment variable

Example:
  export OPENAI_API_KEY="sk-..."  # Linux/Mac
  set OPENAI_API_KEY=sk-...       # Windows
```

**错误: Data directory不存在**
```
❌ ERROR: Data directory does not exist: D:\code\graduate\graduateRAG\processed_data
   Please check the 'data_dir' setting in config.json
```

**警告: API 速率限制**
```
⚠️  WARNING: Rate limited (attempt 1/3)
   Waiting 5s before retry...
```
- 这是正常的,脚本会自动处理
- 如果频繁出现,降低 config.json 中的 `max_concurrent` 值

## System Prompt 说明

脚本已内置详细的 RAG 路由分析提示词，包含：

### 分类规则
- **L0 (No Retrieval)**: 非事实性查询（代码生成、逻辑推理、翻译等）
- **L1 (Single Hop)**: 单跳事实查询（查询单个实体或简单定义）
- **L2 (Multi Hop)**: 多跳复杂查询（比较、多实体关系、嵌套逻辑、时间约束）

### 索引策略
- **None**: 仅用于 L0
- **Lexical**: 用于包含唯一标识符、全名、特定代码的查询
- **Semantic**: 用于描述性、抽象性查询
- **Hybrid**: 用于 L2 或模糊的 L1 查询

### 映射逻辑
- L0 → Z (零检索)
- L1 + Lexical → S-Sparse (BM25)
- L1 + Semantic → S-Dense (向量检索)
- L1 + Hybrid → S-Hybrid (混合检索)
- L2 → M (多跳检索)

提示词包含 5 个详细示例，涵盖各种场景。如需修改，请编辑脚本中的 `SYSTEM_PROMPT` 变量。

## 性能优化

### 并发设置

在 config.json 中调整 `max_concurrent` 来平衡速度和稳定性:

```json
{
  "api_settings": {
    "max_concurrent": 10  // 调整此值
  }
}
```

- **低并发 (5-10)**: 更稳定,适合免费额度或慢速网络
- **中并发 (10-15)**: 推荐设置,平衡速度和稳定性
- **高并发 (15-20)**: 更快但可能触发速率限制

### 模型选择

- **gpt-4o-mini**: 成本低(~$0.15/1M tokens),速度快,推荐用于大规模数据
- **gpt-4o**: 质量更高但成本更高(~$5/1M tokens),可用于关键数据

### 成本估算

处理全部 3000 条数据(6个数据集 × 500条):
- **gpt-4o-mini**: 约 $2-3
- **gpt-4o**: 约 $20-30

## 注意事项

1. **API 配额**: 确保有足够的 OpenAI API 配额
2. **处理时间**: 处理全部数据集(6个 × 500条 = 3000条)可能需要10-30分钟
3. **自动重试**: 脚本会自动处理 API 错误和重试,无需手动干预
4. **建议测试**: 先用小数据集测试,确认无误后再处理全部数据

### 测试建议

```json
{
  "data_settings": {
    "datasets": ["nq"]  // 先只处理一个数据集测试
  }
}
```

处理成功后再改为 `null` 处理全部数据集。

## 故障排除

### 脚本运行中断怎么办?

脚本会为每个数据集单独保存结果,所以即使中断也不会丢失已处理的数据。
可以在 config.json 中移除已处理的数据集,然后重新运行:

```json
{
  "data_settings": {
    "datasets": ["squad", "trivia"]  // 只处理未完成的数据集
  }
}
```

### API 密钥无效

```
OpenAI API error: Invalid API key
```

- 检查 API key 是否正确
- 确认 API key 有效且未过期
- 检查 API key 是否有足够的配额

### 网络连接问题

如果遇到网络超时:
- 降低 `max_concurrent` 值
- 使用 `base_url` 配置代理
- 检查防火墙设置
