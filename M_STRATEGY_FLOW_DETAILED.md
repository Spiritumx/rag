# M策略（IRCoT）完整执行流程详解

## 概览

M策略使用IRCoT（Iterative Retrieval Chain-of-Thought）方法，通过迭代式检索和推理来回答多跳问题。

**核心思想：**
每次生成一句话 → 用这句话检索文档 → 基于新文档继续生成 → 循环直到找到答案或达到最大迭代次数

---

## 配置文件（multi_hop.jsonnet）

```jsonnet
{
    "start_state": "step_by_step_hybrid_retriever",
    "end_state": "[EOQ]",
    "models": {
        "step_by_step_hybrid_retriever": {...},      // 检索模块
        "step_by_step_cot_reasoning_gen": {...},     // 生成模块
        "step_by_step_exit_controller": {...},       // 退出控制模块
        "generate_main_question": {...},             // 最终问答阶段
        "answer_main_question": {...},               // 最终答案生成
        "extract_answer": {...}                      // 答案提取
    }
}
```

**状态机流转：**
```
step_by_step_hybrid_retriever
  ↓
step_by_step_cot_reasoning_gen
  ↓
step_by_step_exit_controller
  ↓ (循环)
step_by_step_hybrid_retriever (如果未找到答案)
  ↓ (如果找到答案)
generate_main_question
  ↓
answer_main_question
  ↓
extract_answer
  → [EOQ]
```

---

## 详细执行流程

### 阶段1：初始化

**输入：**
```json
{
    "question_id": "2hop__123456",
    "question_text": "When was Neville A. Stanton's employer founded?",
    "question": "When was Neville A. Stanton's employer founded?"
}
```

**初始状态（SearchState）：**
```python
state.data = {
    "question": "When was Neville A. Stanton's employer founded?",
    "titles": [],                    # 累积的文档标题
    "paras": [],                     # 累积的文档段落
    "generated_sentences": [],       # 已生成的推理句子
    "metadata": {...}
}
state.next = "step_by_step_hybrid_retriever"
```

---

### 阶段2：第一次迭代（Iteration 1）

#### Step 2.1: 混合检索（RetrieveAndResetParagraphsParticipant）

**代码位置：** `ircot.py:324-587`

**关键逻辑（ircot.py:421-427）：**
```python
elif self.query_source == "question_or_last_generated_sentence":
    question = state.data["question"]
    generated_sentences = state.data.get("generated_sentences", [])
    # 过滤掉推理句（以 So/Thus/Therefore 开头的句子）
    generated_sentences = remove_reasoning_sentences(generated_sentences)
    # 取最后一个非推理句
    last_generated_sentence_str = generated_sentences[-1].strip() if generated_sentences else ""
    # 如果有生成句，用生成句；否则用原始问题
    input_query = last_generated_sentence_str if last_generated_sentence_str else question
```

**第一次迭代时：**
- `generated_sentences` 为空
- `input_query = question = "When was Neville A. Stanton's employer founded?"`

**检索参数构建（ircot.py:509-520）：**
```python
params = {
    "retrieval_method": "retrieve_from_elasticsearch",
    "query_text": "When was Neville A. Stanton's employer founded?",  # BM25会过滤wh-words
    "max_hits_count": 10,           # 最终返回10个文档
    "max_buffer_count": 20,         # 每路检索先取20个
    "corpus_name": "wiki",
    "document_type": "title_paragraph_text",
    "retrieval_backend": "hybrid",
    "hybrid_weights": {"bm25": 1.0, "dense": 1.0, "splade": 1.0}  # 如果有配置
}
```

**发送请求：**
```python
url = "http://localhost:9200/retrieve/"
result = safe_post_request(url, params)
```

**检索服务器端处理（elasticsearch_retriever.py:412-476）：**

1. **BM25检索：**
   ```python
   # 移除wh-words后的查询
   backend_query = "Neville A. Stanton employer founded"
   bm25_hits = self._retrieve_bm25(
       query_text=backend_query,
       max_buffer_count=20  # 检索前20个
   )
   # 返回类似：
   # [
   #   {"title": "Neville A. Stanton", "paragraph_text": "...", "score": 8.5},
   #   {"title": "University of Southampton", "paragraph_text": "...", "score": 6.2},
   #   ...
   # ]
   ```

2. **Dense检索（HNSW）：**
   ```python
   # 使用原始查询（不移除wh-words）
   query_embedding = dense_model.encode("When was Neville A. Stanton's employer founded?")
   hnsw_hits = self._retrieve_hnsw(
       query_embedding=query_embedding,
       max_buffer_count=20
   )
   # 返回基于语义相似度的文档
   ```

3. **SPLADE检索：**
   ```python
   # 学习型稀疏表示
   splade_vector = splade_model.encode("When was Neville A. Stanton's employer founded?")
   splade_hits = self._retrieve_splade(
       splade_vector=splade_vector,
       max_buffer_count=20
   )
   ```

4. **分数融合：**
   ```python
   combined = {}
   # 合并三路结果
   for hit in bm25_hits:
       combined[hit["doc_id"]] = {"source": hit, "score": 1.0 * hit["score"]}

   for hit in hnsw_hits:
       if hit["doc_id"] in combined:
           combined[hit["doc_id"]]["score"] += 1.0 * hit["score"]
       else:
           combined[hit["doc_id"]] = {"source": hit, "score": 1.0 * hit["score"]}

   for hit in splade_hits:
       if hit["doc_id"] in combined:
           combined[hit["doc_id"]]["score"] += 1.0 * hit["score"]
       else:
           combined[hit["doc_id"]] = {"source": hit, "score": 1.0 * hit["score"]}

   # 按分数排序
   all_candidates = sorted(combined.values(), key=lambda x: x["score"], reverse=True)
   ```

5. **Reranker精排：**
   ```python
   # 对所有候选文档（可能40-60个）进行rerank
   reranked = self._apply_reranker(
       query_text="When was Neville A. Stanton's employer founded?",
       candidates=all_candidates
   )

   # 取前10个
   top_hits = reranked[:10]
   ```

**检索结果示例：**
```python
retrieval = [
    {
        "title": "Neville A. Stanton",
        "paragraph_text": "Neville A. Stanton is a British Professor of Human Factors...",
        "score": 0.95,
        "corpus_name": "wiki"
    },
    {
        "title": "University of Southampton",
        "paragraph_text": "The University of Southampton is a public research university...",
        "score": 0.82,
        "corpus_name": "wiki"
    },
    ... (共10个文档)
]
```

**更新状态（ircot.py:543-576）：**
```python
selected_titles = []
selected_paras = []

for retrieval_item in retrieval:
    title = retrieval_item["title"]
    para = retrieval_item["paragraph_text"]

    # 跳过过长段落（>600词）
    if len(para.split(" ")) > 600:
        continue

    # 去重并限制总文档数
    if title not in selected_titles and len(selected_paras) < 15:
        selected_titles.append(title)
        selected_paras.append(para)

# 累积到状态中
if self.cumulate_titles:  # M策略中为True
    state.data["titles"].extend(selected_titles)
    state.data["paras"].extend(selected_paras)
else:
    state.data["titles"] = selected_titles
    state.data["paras"] = selected_paras

state.next = "step_by_step_cot_reasoning_gen"
```

**第一次检索后的状态：**
```python
state.data = {
    "question": "When was Neville A. Stanton's employer founded?",
    "titles": ["Neville A. Stanton", "University of Southampton", ...],  # 10个标题
    "paras": ["...", "...", ...],                                         # 10个段落
    "generated_sentences": []
}
```

---

#### Step 2.2: CoT生成（StepByStepCOTGenParticipant）

**代码位置：** `ircot.py:788-973`

**构建上下文（ircot.py:891-904）：**
```python
question = state.data["question"]
titles = state.data["titles"]  # ["Neville A. Stanton", "University of Southampton", ...]
paras = state.data["paras"]    # ["...", "...", ...]

# 构建上下文文本
context = "\n\n".join([
    f"Title: {title}\n{para[:max_para_num_words words]}"  # 每段最多350词
    for title, para in zip(titles, paras)
])

# 结果类似：
# Title: Neville A. Stanton
# Neville A. Stanton is a British Professor of Human Factors...
#
# Title: University of Southampton
# The University of Southampton is a public research university...
# ...
```

**构建提示词（ircot.py:905-916）：**
```python
generation_so_far = " ".join(state.data["generated_sentences"])  # 第一次为空

if self.question_prefix:  # "Answer the following question by reasoning step-by-step.\n"
    question = self.question_prefix + question

if self.add_context:  # True
    test_example_str = context + "\n\n" + f"Q: {question}" + "\n" + f"A: {generation_so_far}"
else:
    test_example_str = f"Q: {question}" + "\n" + f"A: {generation_so_far}"

# 拼接few-shot prompt
prompt = "\n\n\n".join([self.prompt, test_example_str]).strip()
```

**完整Prompt示例（第一次迭代）：**
```
[Few-shot examples from prompts/hotpotqa/gold_with_2_distractors_context_cot_qa_codex.txt]
...


Title: Neville A. Stanton
Neville A. Stanton is a British Professor of Human Factors and Ergonomics at the University of Southampton...

Title: University of Southampton
The University of Southampton is a public research university in Southampton, England...

Title: ...
...

Q: Answer the following question by reasoning step-by-step.
When was Neville A. Stanton's employer founded?
A:
```

**发送到LLM服务器（ircot.py:918）：**
```python
output_text_scores = self.generator.generate_text_sequence(prompt)
```

**LLM服务器处理（serve_llama_autobatch.py）：**

1. **应用System Prompt覆盖：**
   ```python
   # 检测到 "reasoning step-by-step" 触发 is_ircot_cot = True
   system_prompt = """
   You are a multi-step reasoning agent...
   DECISION LOGIC:
   1. IF the documents contain the FINAL ANSWER → Write: 'So the answer is: [answer]'
   2. IF the documents contain a KEY FACT... → State that fact clearly
   ...
   """
   ```

2. **应用Llama-3聊天模板：**
   ```python
   formatted_prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>

{user_content}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
   ```

3. **生成：**
   ```python
   response = model.generate(formatted_prompt, max_tokens=200)
   # 输出: "Neville A. Stanton works at University of Southampton."
   ```

**句子提取（ircot.py:922-926）：**
```python
new_generation = "Neville A. Stanton works at University of Southampton."

# 使用spaCy分句，只取第一句
new_sents = list(spacy_object(new_generation).sents)
if new_sents:
    new_generation = new_sents[0].text  # "Neville A. Stanton works at University of Southampton."
    state.data["generated_sentences"].append(new_generation)
```

**检查终止条件（ircot.py:928-933）：**
```python
# 正则: "(?:.*?(?:answer is:?|A:)\s*|^)([^.\n]+?)(?:\.|\n|$)"
if self.answer_extractor_regex.match(new_generation):
    # 这句话不匹配（没有 "answer is" 或 "So"）
    exit_generation = False
```

**更新状态：**
```python
state.data = {
    "question": "When was Neville A. Stanton's employer founded?",
    "titles": ["Neville A. Stanton", "University of Southampton", ...],
    "paras": ["...", "...", ...],
    "generated_sentences": ["Neville A. Stanton works at University of Southampton."]
}
state.next = "step_by_step_exit_controller"
```

---

#### Step 2.3: 退出控制（StepByStepExitControllerParticipant）

**代码位置：** `ircot.py:975-1073`

**检查退出条件（ircot.py:1029-1047）：**
```python
generated_sentences = ["Neville A. Stanton works at University of Southampton."]
exit_generation = False

# 条件1: 最后一句为空
if generated_sentences and not generated_sentences[-1]:
    exit_generation = True  # False

# 条件2: 达到最大句数（20句）
if len(generated_sentences) >= self.max_num_sentences:  # 1 < 20
    exit_generation = False

# 条件3: 匹配答案正则
# regex: "(?:.*?(?:answer is:?|A:)\s*|^)([^.\n]+?)(?:\.|\n|$)"
if generated_sentences and self.answer_extractor_regex.match(generated_sentences[-1]):
    # "Neville A. Stanton works at University of Southampton." 不匹配
    exit_generation = False
```

**决定下一步（ircot.py:1049-1072）：**
```python
if exit_generation:
    # 终止，进入最终答案阶段
    state.next = self.terminal_state_next_model  # "generate_main_question"
else:
    # 继续检索
    state.next = self.next_model  # "step_by_step_hybrid_retriever"
```

**当前决定：** 继续检索（因为未找到答案）

---

### 阶段3：第二次迭代（Iteration 2）

#### Step 3.1: 混合检索（第二轮）

**查询提取（ircot.py:421-427）：**
```python
generated_sentences = ["Neville A. Stanton works at University of Southampton."]
generated_sentences = remove_reasoning_sentences(generated_sentences)
# 过滤后仍为: ["Neville A. Stanton works at University of Southampton."]
last_generated_sentence_str = "Neville A. Stanton works at University of Southampton."
input_query = last_generated_sentence_str  # 使用生成的句子！
```

**关键点：生成的句子变成检索查询！**

**检索参数：**
```python
params = {
    "query_text": "Neville A. Stanton works at University of Southampton.",  # 新查询
    "max_hits_count": 10,
    "max_buffer_count": 20,
    ...
}
```

**三路检索：**
```python
# BM25: 移除wh-words后
# query = "Neville A. Stanton works University Southampton"
# 关键词: Neville, Stanton, University, Southampton（实体丰富！）

# Dense: 原始查询
# "Neville A. Stanton works at University of Southampton."

# SPLADE: 学习型稀疏表示
```

**检索结果（预期）：**
```python
retrieval = [
    {
        "title": "University of Southampton",
        "paragraph_text": "...founded in 1862...",  # 包含答案！
        "score": 0.98
    },
    {
        "title": "Southampton",
        "paragraph_text": "...",
        "score": 0.85
    },
    ...
]
```

**更新状态（累积文档）：**
```python
state.data["titles"].extend(new_titles)      # 累积去重
state.data["paras"].extend(new_paras)
# 现在可能有15-20个文档（去重后）
```

---

#### Step 3.2: CoT生成（第二轮）

**上下文构建：**
```python
context = "\n\n".join([
    f"Title: {title}\n{para}"
    for title, para in zip(
        ["Neville A. Stanton", "University of Southampton", "Southampton", ...],
        ["...", "...founded in 1862...", "...", ...]
    )
])
```

**提示词：**
```
[Few-shot examples]
...

Title: Neville A. Stanton
...

Title: University of Southampton
The University of Southampton...was founded in 1862...

Title: Southampton
...

Q: Answer the following question by reasoning step-by-step.
When was Neville A. Stanton's employer founded?
A: Neville A. Stanton works at University of Southampton.
```

**注意：** `A:` 后面现在有第一次生成的句子！

**LLM生成：**
```python
# 模型看到：
# - 上下文中有 "University of Southampton...founded in 1862"
# - 已知 "Neville A. Stanton works at University of Southampton"
# - 问题问 "When was...employer founded?"

# 生成:
response = "University of Southampton was founded in 1862."
```

**更新状态：**
```python
state.data["generated_sentences"].append("University of Southampton was founded in 1862.")
# 现在: ["Neville A. Stanton works at University of Southampton.",
#        "University of Southampton was founded in 1862."]
```

---

#### Step 3.3: 退出控制（第二轮）

**检查：**
```python
# 仍然不匹配答案正则（没有 "So the answer is"）
exit_generation = False
state.next = "step_by_step_hybrid_retriever"  # 继续第三次迭代
```

---

### 阶段4：第三次迭代（Iteration 3）

#### Step 4.1: 混合检索（第三轮）

**查询提取：**
```python
generated_sentences = [
    "Neville A. Stanton works at University of Southampton.",
    "University of Southampton was founded in 1862."
]
generated_sentences = remove_reasoning_sentences(generated_sentences)
last_generated_sentence_str = "University of Southampton was founded in 1862."
input_query = "University of Southampton was founded in 1862."
```

**检索结果：**
```python
# 可能检索到关于Southampton历史、1862年等相关文档
# 这些文档对回答问题用处不大
```

---

#### Step 4.2: CoT生成（第三轮）

**提示词：**
```
...
Q: Answer the following question by reasoning step-by-step.
When was Neville A. Stanton's employer founded?
A: Neville A. Stanton works at University of Southampton. University of Southampton was founded in 1862.
```

**LLM生成（找到答案）：**
```python
# 模型发现：
# - 问题: "When was...employer founded?"
# - 已知: Stanton works at Southampton, Southampton founded in 1862
# - 结论: employer = Southampton, founded = 1862

response = "So the answer is: 1862"
```

**答案提取（ircot.py:928-932）：**
```python
if self.answer_extractor_regex.match("So the answer is: 1862"):
    # 匹配！
    return_answer = "1862"
    exit_generation = True
```

---

#### Step 4.3: 退出控制（第三轮）

**检查：**
```python
if generated_sentences and self.answer_extractor_regex.match(generated_sentences[-1]):
    # "So the answer is: 1862" 匹配！
    return_answer = "1862"
    exit_generation = True

if exit_generation:
    state.next = "generate_main_question"  # 进入最终答案阶段
```

---

### 阶段5：最终答案生成

#### Step 5.1: 复制问题（CopyQuestionParticipant）

```python
# 简单地将原始问题复制到状态中
state.data["current_question"] = state.data["question"]
state.next = "answer_main_question"
```

---

#### Step 5.2: 直接问答（LLMQAParticipantModel）

**提示词构建：**
```python
context = "\n\n".join([
    f"Title: {title}\n{para}"
    for title, para in zip(state.data["titles"], state.data["paras"])
])

prompt = f"""
[Few-shot examples from prompts/hotpotqa/gold_with_2_distractors_context_direct_qa_codex.txt]

{context}

Q: Answer the following question.
When was Neville A. Stanton's employer founded?
A:
"""
```

**LLM服务器处理：**
```python
# 检测到 "Answer the following question" → is_direct_qa = True
system_prompt = """
You are a precise QA assistant.
Rules:
1. Read the provided context and answer the question directly.
2. Output ONLY the answer - be concise and specific.
...
"""
```

**生成：**
```python
response = "1862"
state.data["last_answer"] = "1862"
state.next = "extract_answer"
```

---

#### Step 5.3: 答案提取（AnswerExtractor）

**代码位置：** `ircot.py:155-182`

```python
class AnswerExtractor(ParticipantModel):
    def query(self, state, debug=False):
        answer_text = state.data["last_answer"]  # "1862"

        # 应用正则提取
        match = self.regex.match(answer_text)
        if match:
            extracted = match.group(1)
        else:
            extracted = answer_text

        # 移除末尾句号
        if self.remove_last_fullstop and extracted.endswith("."):
            extracted = extracted[:-1]

        new_state = state.copy()
        new_state.data["answer"] = extracted  # "1862"
        new_state.next = self.end_state  # "[EOQ]"
        return new_state
```

---

### 阶段6：返回结果

**最终状态：**
```python
final_state.data = {
    "question": "When was Neville A. Stanton's employer founded?",
    "titles": ["Neville A. Stanton", "University of Southampton", ...],
    "paras": ["...", "...", ...],
    "generated_sentences": [
        "Neville A. Stanton works at University of Southampton.",
        "University of Southampton was founded in 1862.",
        "So the answer is: 1862"
    ],
    "answer": "1862"
}
```

**输出格式（model_search.py:176-214）：**

1. **预测答案：**
   ```python
   prediction = final_state.data["answer"]  # "1862"
   ```

2. **推理链（Chain）：**
   ```python
   chain = f"""
{qid}
Q: {question}
A: {sentence1} A: {sentence2} A: {sentence3} S: {score}
"""

   # 实际输出:
   """
   2hop__123456
   Q: When was Neville A. Stanton's employer founded?
   A: Neville A. Stanton works at University of Southampton. A: University of Southampton was founded in 1862. A: So the answer is: 1862 S: 0.0
   """
   ```

3. **上下文（Context）：**
   ```python
   context = {
       "qid": "2hop__123456",
       "titles": ["Neville A. Stanton", "University of Southampton", ...],
       "paras": ["...", "...", ...]
   }
   ```

4. **保存到文件：**
   ```python
   # predictions.json
   {"2hop__123456": "1862"}

   # chains.txt
   2hop__123456
   Q: When was Neville A. Stanton's employer founded?
   A: Neville A. Stanton works at University of Southampton. A: University of Southampton was founded in 1862. A: So the answer is: 1862 S: 0.0

   # contexts.json
   {"2hop__123456": {"titles": [...], "paras": [...]}}
   ```

---

## 关键流程总结

### 迭代循环

```
[开始]
  ↓
[检索] input_query = last_generated_sentence（或question）
  ↓ 混合检索: BM25 + Dense + SPLADE → Reranker → Top 10
  ↓ 累积到 state.data["titles"] 和 state.data["paras"]
  ↓
[生成] 基于累积的所有文档 + 已生成句子
  ↓ 构建prompt: context + Q + A: <已生成句子>
  ↓ LLM生成一句话（用spaCy分句，取第一句）
  ↓ 添加到 state.data["generated_sentences"]
  ↓
[退出控制] 检查是否找到答案或达到最大迭代次数
  ↓
[判断]
  找到答案（匹配 "So the answer is"） → 进入最终答案阶段
  未找到答案 → 返回[检索]，使用新生成的句子作为查询
  达到最大迭代次数 → 强制退出
```

---

## 每次迭代使用的信息

### 检索阶段

**用于检索的查询（query_text）：**
- **第1次迭代：** 原始问题（`state.data["question"]`）
- **第2次迭代：** 第1次生成的句子（如 "Neville A. Stanton works at University of Southampton."）
- **第3次迭代：** 第2次生成的句子（如 "University of Southampton was founded in 1862."）
- **第N次迭代：** 第N-1次生成的句子

**发送到Reranker的问题：**
- 始终是原始问题（`query_text` 参数）
- 用于计算query-document相关性

**检索结果处理：**
- 每次检索到的10个文档**累积**到 `state.data["titles"]` 和 `state.data["paras"]`
- 自动去重（相同title只保留一个）
- 限制总文档数（`global_max_num_paras = 15`）

---

### 生成阶段

**上下文（Context）：**
- **所有累积的文档**（第1次：10个，第2次：15-20个去重后）
- 格式：
  ```
  Title: doc1_title
  doc1_paragraph_text

  Title: doc2_title
  doc2_paragraph_text
  ...
  ```

**问题（Question）：**
- 原始问题（每次都相同）
- 加前缀 "Answer the following question by reasoning step-by-step.\n"

**已生成内容（A:）：**
- **所有之前生成的句子**（用空格连接）
- 例如第3次迭代时：
  ```
  A: Neville A. Stanton works at University of Southampton. University of Southampton was founded in 1862.
  ```

**完整Prompt结构：**
```
[Few-shot Examples]

Title: ...
...

Title: ...
...

Q: Answer the following question by reasoning step-by-step.
<original question>
A: <all previously generated sentences>
```

---

## 终止条件

### 正常终止（找到答案）

**触发条件：**
```python
# 生成的句子匹配正则表达式
answer_extractor_regex = "(?:.*?(?:answer is:?|A:)\s*|^)([^.\n]+?)(?:\.|\n|$)"

# 匹配示例:
# ✓ "So the answer is: 1862"
# ✓ "Thus the answer is 1862."
# ✓ "Therefore, the answer is Paris"
```

**处理流程：**
```
检测到答案 → exit_generation = True
  → state.next = "generate_main_question"
  → 进入最终答案生成阶段
  → 使用Direct QA模式生成精确答案
  → 提取答案 → 结束
```

---

### 异常终止（达到最大迭代次数）

**触发条件：**
```python
if len(state.data["generated_sentences"]) >= max_num_sentences:  # 20
    exit_generation = True
```

**处理流程：**
```
达到20句 → 强制退出
  → 使用最后一句（或所有句子拼接）作为答案
  → 可能输出 "I don't know" 或不完整答案
```

---

### 生成失败终止

**触发条件：**
```python
# spaCy分句失败（生成内容为空）
if not new_sents:
    exit_generation = True
```

---

## 实体稀疏问题的影响

### 场景：生成了实体稀疏的句子

**第1次迭代：**
```python
# 假设LLM生成:
generated_sentence = "We need to find more information about his employer."
```

**第2次迭代检索：**
```python
input_query = "We need to find more information about his employer."

# BM25查询（移除wh-words后）:
# "need find more information his employer"
# 关键词: need, find, more, information, employer（全是通用词！）

# 检索结果:
# - BM25: 可能检索到大量包含 "information" "employer" 的无关文档
# - Dense: 语义模糊，难以定位具体事实
# - SPLADE: 也依赖关键词，效果不佳

# Reranker: 即使精排，候选池质量差
retrieval = [
    {"title": "Employment", "paragraph": "...", "score": 0.6},   # 无关
    {"title": "Information", "paragraph": "...", "score": 0.55}, # 无关
    ...
]
```

**第3次迭代生成：**
```python
# 上下文中都是无关文档
context = """
Title: Employment
...general information about employment...

Title: Information
...about information systems...
"""

# LLM可能生成:
generated_sentence = "The information is not clear."
# 或者:
generated_sentence = "I cannot determine the answer."
```

**结果：**
- 陷入恶性循环
- 多次迭代后仍无法找到答案
- 最终达到20次上限 → 输出 "I don't know"

---

### 对比：实体丰富的句子

**第1次迭代：**
```python
generated_sentence = "Neville A. Stanton works at University of Southampton."
```

**第2次迭代检索：**
```python
input_query = "Neville A. Stanton works at University of Southampton."

# BM25: "Neville A. Stanton works University Southampton"
# 关键词: Neville, Stanton, University, Southampton（实体丰富！）

# 检索结果:
retrieval = [
    {"title": "University of Southampton", "paragraph": "...founded in 1862...", "score": 0.98},
    {"title": "Neville A. Stanton", "paragraph": "...", "score": 0.92},
    ...
]
# 精准检索到相关文档！
```

**第3次迭代生成：**
```python
# 上下文中有正确信息
context = """
Title: University of Southampton
...was founded in 1862...
"""

# LLM生成:
generated_sentence = "University of Southampton was founded in 1862."
# 或直接:
generated_sentence = "So the answer is: 1862"
```

**结果：**
- 2-3次迭代即可找到答案
- 检索准确，生成高质量

---

## 混合检索的优势与局限

### 优势

1. **互补性：**
   - BM25擅长精确关键词匹配
   - Dense擅长语义理解
   - SPLADE结合两者优点

2. **鲁棒性：**
   - 某一路失败时，其他路可补救

### 局限

1. **BM25仍占权重：**
   ```python
   final_score = 1.0 × bm25 + 1.0 × dense + 1.0 × splade
   # 即使dense和splade检索到相关文档，BM25可能拉低总分
   ```

2. **实体稀疏影响所有三路：**
   - BM25: 通用词噪音大
   - SPLADE: 学习型稀疏表示，仍依赖关键词
   - Dense: 语义模糊的元陈述难以匹配具体事实

3. **累积误差：**
   - 第一次检索偏离 → 生成偏离 → 第二次更偏离 → 恶性循环
