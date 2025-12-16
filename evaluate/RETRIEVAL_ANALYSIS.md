# 检索文档分析功能

## 概述

新增功能：保存并分析检索到的文档，用于诊断为什么部分预测输出的是Wikipedia标题而不是正确答案。

## 修改内容

### 1. 保存检索文档

**修改的文件**:
- `commaqa/inference/model_search.py` - 修改返回格式，包含检索上下文
- `commaqa/inference/configurable_inference.py` - 保存检索上下文到JSON文件

**新增输出文件**:
```
evaluate/outputs/stage2_predictions/
  ├── squad_predictions.json          # 预测答案（原有）
  ├── squad_predictions_chains.txt     # 推理链（原有）
  └── squad_predictions_contexts.json  # 检索文档（新增）✨
```

**contexts.json 格式**:
```json
{
  "question_id_1": {
    "titles": ["Wikipedia Title 1", "Wikipedia Title 2", ...],
    "paras": ["Paragraph text 1", "Paragraph text 2", ...]
  },
  "question_id_2": {
    ...
  }
}
```

### 2. 检索质量分析脚本

**新建文件**: `evaluate/analyze_retrieval.py`

## 使用方法

### 步骤1：生成预测（包含检索文档）

```bash
# 运行Stage 2生成（会自动保存contexts）
python evaluate/stage2_generate.py --datasets squad

# 或使用测试脚本
python evaluate/test_fixes.py
```

生成后会看到新文件：
```
✓ Writing retrieved contexts in evaluate/outputs/stage2_predictions/squad_predictions_contexts.json
```

### 步骤2：分析检索质量

```bash
# 分析单个数据集
python evaluate/analyze_retrieval.py --datasets squad --samples 20

# 分析所有数据集
python evaluate/analyze_retrieval.py

# 显示更多样本
python evaluate/analyze_retrieval.py --datasets squad --samples 50
```

## 分析报告内容

### 1. 整体统计

```
RETRIEVAL STATISTICS
================================================================================

Retrieved documents per question:
  Average: 8.5
  Min: 0
  Max: 15
  Empty retrievals: 3 (0.6%)
```

### 2. 样本分析

对于每个样本，显示：

```
Sample 1: single_squad_dev_4465
--------------------------------------------------------------------------------
Question: What was the name of popular movie film in New Haven...
Prediction: 'sa Smile'

Retrieved 8 documents:
  1. Mona Lisa Smile
  2. Yale University
  3. New Haven, Connecticut
  4. Julia Roberts
  5. Film production

  ⚠️  Prediction seems to match retrieved title: 'Mona Lisa Smile'

  Relevance (rough estimate):
    6/8 titles have word overlap with question
```

### 3. 问题诊断

脚本会自动识别：

✅ **检索相关性**
- 检索到的文档标题与问题的词汇重叠
- 空检索（没有检索到文档）

⚠️ **潜在问题**
- 预测匹配检索标题（答案提取错误）
- 检索文档与问题无关（检索质量差）
- 预测包含"Wikipedia Title"字样（明显错误）

## 常见问题诊断

### 问题1：预测是Wikipedia标题

**示例**:
```
Question: When did Menzies resign?
Prediction: 'Wikipedia Title: James J. Hill'
```

**可能原因**:
1. **检索错误** - 检索到了完全不相关的文档
   - 检查: `analyze_retrieval.py`显示检索的标题
   - 修复: 调整检索参数（retrieval_count, query preprocessing）

2. **答案提取错误** - LLM从文档标题而不是内容中提取答案
   - 检查: 查看`_chains.txt`推理链
   - 修复: 改进prompt，明确要求从段落内容而不是标题中提取答案

### 问题2：答案截断

**示例**:
```
Question: What is the CEO's name?
Prediction: 'melt'  (应该是 "Jeff Immelt")
Retrieved: ["General Electric", "Jeff Immelt", ...]
```

**诊断**:
- 检索是正确的（包含"Jeff Immelt"）
- 但答案被截断了
- 这是LLM服务器的prompt移除问题（已修复）

### 问题3：答案完全错误但检索正确

**示例**:
```
Question: Who was the CEO?
Prediction: 'General Electric'  (应该是 "Jeff Immelt")
Retrieved: ["General Electric", "Jeff Immelt", ...]
```

**可能原因**:
1. **LLM理解错误** - 没有正确理解问题
   - 修复: 改进prompt模板
   - 或使用更大/更好的LLM

2. **上下文过长** - LLM被太多文档干扰
   - 修复: 减少`global_max_num_paras`
   - 或改进re-ranking

## 典型工作流程

### 发现问题
```bash
# 1. 运行评估
python evaluate/test_fixes.py

# 2. 发现F1分数低
# EM=0.002, F1=0.097
```

### 诊断原因
```bash
# 3. 检查预测样本
python evaluate/diagnose_predictions.py --datasets squad --samples 20

# 4. 发现很多Wikipedia标题
# Prediction: 'Wikipedia Title: The Great Wave...'

# 5. 分析检索质量
python evaluate/analyze_retrieval.py --datasets squad --samples 20
```

### 确定问题类型

**如果分析显示**:
```
⚠️  Found 50 predictions that look like Wikipedia titles
⚠️  Warning: No retrieved titles seem relevant to question!
```
→ **检索质量问题** - 需要调整检索器

**如果分析显示**:
```
✓ Retrieved 8 documents
✓ 7/8 titles have word overlap with question
⚠️ Prediction seems to match retrieved title: 'General Electric'
```
→ **答案提取问题** - 需要改进prompt或答案提取逻辑

**如果分析显示**:
```
✓ Retrieved documents are relevant
✓ Prediction is truncated: 'melt' (should be 'Jeff Immelt')
```
→ **答案截断问题** - LLM服务器bug（已修复）

## 改进建议

### 1. 检索质量差

**调整配置** (`evaluate/configs/llama_configs/*.jsonnet`):
```jsonnet
{
  "retrieval_count": 10,  // 增加检索数量
  "global_max_num_paras": 10,  // 但限制使用的段落数
}
```

**或改进查询**:
- 添加查询扩展
- 使用更好的查询改写

### 2. 答案提取错误

**改进prompt** (在prompt文件中):
```
Answer the following question based on the PARAGRAPH CONTENT,
not the Wikipedia titles. Give a short, direct answer.

Q: ...
A:
```

### 3. 检索相关但答案错误

**考虑**:
- 使用更强的LLM（Llama-70B, GPT-4等）
- 添加re-ranking步骤
- 改进few-shot examples

## 文件清单

### 修改的文件
1. ✏️ `commaqa/inference/model_search.py` - 返回检索上下文
2. ✏️ `commaqa/inference/configurable_inference.py` - 保存上下文

### 新增的文件
3. ✨ `evaluate/analyze_retrieval.py` - 检索分析脚本
4. ✨ `evaluate/RETRIEVAL_ANALYSIS.md` - 本文档

### 生成的文件
5. 📄 `evaluate/outputs/stage2_predictions/*_contexts.json` - 检索文档

## 下一步

1. ✅ 修改代码保存检索文档
2. ⏳ 重新运行生成获取contexts
3. ⏳ 分析检索质量
4. ⏳ 根据分析结果决定优化方向

---

**提示**: 始终先运行`analyze_retrieval.py`再决定如何优化。盲目调参可能适得其反！
