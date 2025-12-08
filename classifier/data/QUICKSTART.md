# 快速开始指南

## 3 步开始使用

### 1. 安装依赖

```bash
pip install openai pydantic tqdm
```

或使用 requirements.txt:
```bash
pip install -r requirements.txt
```

### 2. 配置 config.json

```bash
# 复制示例配置
cp config.example.json config.json

# 编辑配置文件
nano config.json  # Linux/Mac
notepad config.json  # Windows
```

最小配置示例:
```json
{
  "api_settings": {
    "api_key": "sk-your-api-key-here",
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

**提示**: `"datasets": null` 表示处理所有6个数据集

### 3. 运行脚本

```bash
python generate_training_data.py
```

## 输出

脚本运行后会在 `./training_data` 目录生成:

- `nq_training.jsonl` - NQ数据集的训练数据
- `squad_training.jsonl` - SQuAD数据集的训练数据
- ... (其他数据集)
- `combined_training_data.jsonl` - 所有数据集合并的文件

## 测试单个查询

在运行完整数据生成之前,可以先测试:

```bash
python test_single_query.py "Who is the president of France?"
```

或运行内置测试:
```bash
python test_single_query.py
```

## 常见问题

### ❌ ERROR: config.json not found
```bash
cp config.example.json config.json
# 然后编辑 config.json
```

### ❌ ERROR: No API key provided
在 config.json 中设置 `api_key`:
```json
{
  "api_settings": {
    "api_key": "sk-your-actual-key"
  }
}
```

或使用环境变量:
```bash
export OPENAI_API_KEY="sk-..."  # Linux/Mac
set OPENAI_API_KEY=sk-...       # Windows
```

### ⚠️ WARNING: Rate limited
这是正常的,脚本会自动重试。如果频繁出现,在 config.json 中降低 `max_concurrent`:
```json
{
  "api_settings": {
    "max_concurrent": 5
  }
}
```

## 进阶使用

### 只处理特定数据集

```json
{
  "data_settings": {
    "datasets": ["nq", "squad"]
  }
}
```

### 使用不同的模型

```json
{
  "api_settings": {
    "model": "gpt-4o"
  }
}
```

### 使用代理

```json
{
  "api_settings": {
    "base_url": "https://your-proxy.com/v1"
  }
}
```

## 下一步

- 查看 [README.md](./README.md) 了解详细配置选项
- 查看 [PROJECT_OVERVIEW.md](./PROJECT_OVERVIEW.md) 了解项目架构
- 检查生成的数据: `cat training_data/nq_training.jsonl | head -n 1 | jq .`

## 需要帮助?

- 检查 [README.md](./README.md) 的"故障排除"部分
- 查看 [PROJECT_OVERVIEW.md](./PROJECT_OVERVIEW.md) 的"常见问题"部分
