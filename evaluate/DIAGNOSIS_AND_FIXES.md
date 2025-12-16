# 评估结果诊断与修复方案

## 问题总结

评估结果显示EM和F1分数极低（EM ≈ 0, F1 = 0.05-0.19），通过诊断发现三个主要问题：

### 1. **答案截断问题**（最严重）⭐

**现象**：大量预测答案的开头被截断了1-10个字符

**示例**：
```
预测: "lanetSolar"           → 正确: "PlanetSolar"
预测: "eberroth"             → 正确: "Peter Ueberroth"
预测: "lliam IV"             → 正确: "William IV"
预测: "ter United F.C."      → 正确: "Manchester United F.C."
预测: "irls and eight boys"  → 正确: "nine girls and 10 boys"
预测: "atel"                 → 正确: "Pawan Chamling"
```

**可能原因**：

a) **Prompt移除逻辑问题** (`commaqa/models/llm_client_generator.py:241-242`)
   ```python
   # remove the prompt
   if text.startswith(prompt):
       text = text[len(prompt) :]
   ```
   - 如果LLM服务器返回的文本tokenization与prompt不完全匹配
   - 可能导致多移除或少移除几个字符

b) **LLM服务器的生成问题**
   - `keep_prompt=False`设置可能在服务器端没有正确工作
   - 服务器在移除prompt时可能有bug

c) **答案从context中间位置开始复制**
   - LLM可能从检索到的文档中间某个位置开始复制答案
   - 导致答案开头缺失

### 2. **错误的Wikipedia标题**

**示例**：
```
问题: "When did Menzies resign from Parliament?"
预测: "Wikipedia Title: James J. Hill"  ← 完全不相关

问题: "What is the chemical symbol for Gold?"
预测: "ipedia Title: Shōnen Ace"
```

**原因**：
- 检索质量差，返回了不相关的文档
- BM25检索参数可能需要调整
- 或者问题本身需要多跳推理但被误分类为单跳

### 3. **过长的解释性答案**

**示例**：
```
问题: "How much was Ammon Bundy's son ordered to pay?"
预测: "s no text to this effect in this snippet from Wikipedia.
      There is nothing mentioning Ammon Bundy's son paying grazing
      fees anywhere here. However, it does mention Ammon's father,
      Cliven Bundy paying $1.4 million."
```

**原因**：
- LLM生成了解释而不是直接答案
- Prompt可能需要更强调"简洁回答"
- 或者需要更好的答案抽取正则表达式

## 修复方案

### 方案1：修复Prompt移除逻辑（推荐）⭐

**文件**: `commaqa/models/llm_client_generator.py`

**问题位置**: 行 241-242

**当前代码**:
```python
# remove the prompt
if text.startswith(prompt):
    text = text[len(prompt) :]
```

**修复建议**:
```python
# remove the prompt more carefully
if text.startswith(prompt):
    text = text[len(prompt):].lstrip()  # 添加lstrip()移除前导空格
else:
    # If text doesn't start with prompt, try to remove common prefix
    # This handles tokenization boundary issues
    common_len = 0
    min_len = min(len(prompt), len(text))
    for i in range(min_len):
        if prompt[i] == text[i]:
            common_len = i + 1
        else:
            break

    # Only remove if we have significant overlap (>50% of prompt)
    if common_len > len(prompt) * 0.5:
        text = text[common_len:].lstrip()
        print(f"Warning: Partial prompt match. Removed {common_len}/{len(prompt)} chars")
```

### 方案2：修改LLM服务器调用参数

**文件**: `evaluate/configs/llama_configs/single_bm25.jsonnet` 和 `multi_hop.jsonnet`

**修改**:
```jsonnet
"answer_main_question": {
    "name": "llmqa",
    ...
    "max_length": 50,  // 从200降到50，强制简洁回答
    "prompt_file": "prompts/squad/gold_with_2_distractors_context_direct_qa_flan_t5.txt",
    "question_prefix": "Answer the following question with a short, direct answer (1-5 words).\n",  // 更明确的指令
    ...
}
```

### 方案3：改进答案抽取正则表达式

**文件**: `evaluate/configs/llama_configs/multi_hop.jsonnet`

**当前配置** (行84):
```jsonnet
"regex": ".* answer is:? (.*)\\.?",
```

**问题**:
- 这个正则太宽泛，容易失败
- 失败时`match_all_on_failure: true`会返回整个文本

**修复建议**:
```jsonnet
// 更鲁棒的正则，支持多种答案格式
"regex": "(?:.*answer is:?\\s*|.*A:\\s*|^)([^.\\n]+).*",
```

### 方案4：检查并修复LLM服务器

**检查步骤**:

1. **验证服务器是否正确处理keep_prompt参数**:
   ```bash
   curl "http://localhost:8000/generate?prompt=Test&keep_prompt=false&max_length=20"
   ```
   查看返回的`generated_texts`是否包含"Test"

2. **检查服务器日志**，看是否有tokenization警告

3. **如果服务器有问题**，可能需要：
   - 更新LLM服务器代码
   - 或在客户端更鲁棒地处理返回结果

### 方案5：优化检索配置

**文件**: `evaluate/configs/llama_configs/single_bm25.jsonnet`

**当前配置**:
```jsonnet
"retrieval_count": 5,
"global_max_num_paras": 15,
```

**优化建议**:
```jsonnet
"retrieval_count": 10,  // 增加检索数量
"global_max_num_paras": 10,  // 但减少实际使用的段落数
```

这样可以：
- 检索更多候选文档
- 但只使用前10个最相关的
- 提高检索召回率，同时控制context长度

## 实施优先级

### 立即实施（高优先级）

1. **方案1**: 修复prompt移除逻辑
   - 影响: 解决大部分截断问题
   - 难度: 低
   - 风险: 低

2. **方案2**: 修改max_length和question_prefix
   - 影响: 强制LLM生成简洁答案
   - 难度: 低
   - 风险: 低

### 后续实施（中优先级）

3. **方案4**: 检查LLM服务器
   - 影响: 彻底解决prompt移除问题
   - 难度: 中
   - 风险: 中

4. **方案5**: 优化检索配置
   - 影响: 提高检索质量
   - 难度: 低
   - 风险: 低

### 可选实施（低优先级）

5. **方案3**: 改进答案抽取正则
   - 影响: 处理特殊答案格式
   - 难度: 中
   - 风险: 中

## 测试计划

实施修复后，执行以下测试：

### 1. 小规模测试
```bash
# 只测试squad的前50个样本
python evaluate/run_pipeline.py --datasets squad --stages 2 3

# 运行诊断
python evaluate/diagnose_predictions.py --datasets squad --samples 20
```

**预期改善**:
- 截断答案数量应该显著减少（从60%+ 降到 <10%）
- F1分数应该提升到 0.3-0.5 范围

### 2. 完整测试

如果小规模测试成功，运行完整评估：
```bash
python evaluate/run_pipeline.py --stages 2 3
```

**预期结果**:
- EM: 0.15 - 0.30（从0.002提升）
- F1: 0.35 - 0.55（从0.05-0.19提升）

## 调试技巧

### 1. 查看原始LLM输出

在`commaqa/models/llm_client_generator.py`的`generate_text_sequence`方法中添加日志：

```python
# 在第246行之后添加
print(f"DEBUG: Original LLM output: {generated_texts[0][:200]}")
print(f"DEBUG: After processing: {modified_texts[0][:200]}")
```

### 2. 保存推理链

修改`commaqa/inference/configurable_inference.py`，保存详细的推理链：

```bash
# 检查生成的chains文件
cat evaluate/outputs/stage2_predictions/squad_predictions_chains.txt | head -50
```

### 3. 手动测试单个问题

```python
# test_single_question.py
from commaqa.models.llm_client_generator import LLMClientGenerator

gen = LLMClientGenerator(
    model_name="Meta-Llama-3-8B-Instruct",
    max_length=50,
    temperature=0.1
)

prompt = """
Wikipedia Title: Test
Test paragraph here.

Q: Answer the following question.
What is 2+2?
A:"""

result = gen.generate_text_sequence(prompt)
print("Result:", result[0][0])
```

## 预期改善

实施方案1和2后：

| 指标 | 当前 | 预期 | 改善 |
|------|------|------|------|
| EM (squad) | 0.002 | 0.20-0.30 | +100x |
| F1 (squad) | 0.096 | 0.40-0.55 | +4-5x |
| EM (hotpotqa) | 0.002 | 0.15-0.25 | +75x |
| F1 (hotpotqa) | 0.111 | 0.35-0.50 | +3-4x |
| 截断答案比例 | 60%+ | <10% | -50% |

**注意**: 这些是保守估计。实际改善可能更大或更小，取决于问题的真实原因。

## 下一步

1. 实施方案1（修复prompt移除逻辑）
2. 实施方案2（修改LLM参数）
3. 运行小规模测试
4. 分析结果，决定是否需要进一步调整
5. 如果仍有问题，实施方案4（检查LLM服务器）

## 联系支持

如果问题持续存在：
1. 检查LLM服务器日志
2. 使用调试技巧收集更多信息
3. 考虑切换到不同的LLM或检索策略
