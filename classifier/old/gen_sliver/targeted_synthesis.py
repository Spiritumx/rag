import json
import os
import sys
import time
import random
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from openai import OpenAI

# 添加项目根目录到 sys.path
sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))
from classifier.gen_sliver.preprocess_utils import load_json, save_json

# --- 配置区域 ---
DOMAINS = [
    "医疗药物与诊断 (Medical Diagnosis & Pharmacology)",
    "IT技术支持与报错代码 (IT Support & Error Codes)",
    "法律条款与合规性 (Legal Compliance & Statutes)",
    "金融市场与宏观经济 (Financial Markets & Economics)",
    "流行文化与历史事件 (Pop Culture & History)"
]

# 定义不同策略的生成 Prompt
PROMPTS = {
    "S1": """
# Role
你是一个数据合成专家。我们需要为 RAG 系统生成"简单关键词匹配"的测试查询。
当前关注领域：{domain}

# Task
请生成 20 条适合 **S1 (BM25 稀疏检索)** 策略的简单查询。

# Context & Constraints
1. **策略定义**：S1 使用 BM25 进行简单的关键词匹配，适用于直接、明确的查询。
2. **适用场景**：查询意图清晰，包含明确的关键词，可以直接通过词频匹配找到相关文档。
3. **噪声风险**：查询简单直接，使用向量检索反而可能引入不必要的语义扩展。

# Real Examples (Learn from these)

## S1 策略的真实样本：
{target_examples}

## 其他策略的样本 (注意区分)：
{other_examples}

请注意 S1 样本的特点：查询直接、关键词明确、不需要语义理解或多跳推理。

# Output Format
JSON List:
[
  {{
    "query": "...",
    "strategy": "S1",
    "noise_analysis": "简单关键词匹配即可，向量检索会增加噪声...",
    "reasoning": "查询直接明确，BM25 足够..."
  }}
]
只输出 JSON 对象，不要包含 markdown 格式。
""",
    "S2": """
# Role
你是一个数据合成专家。我们需要为 RAG 系统生成"密集向量检索"的测试查询。
当前关注领域：{domain}

# Task
请生成 20 条适合 **S2 (Dense 向量检索)** 策略的查询。

# Context & Constraints
1. **策略定义**：S2 使用密集向量进行语义匹配，适用于需要理解查询语义但不需要多跳推理的场景。
2. **适用场景**：查询包含同义词、paraphrase，或需要语义理解，但单次检索即可找到答案。
3. **噪声风险**：纯关键词匹配会因为词汇不匹配而失败，需要向量捕捉语义。

# Real Examples (Learn from these)

## S2 策略的真实样本：
{target_examples}

## 其他策略的样本 (注意区分)：
{other_examples}

请注意 S2 样本的特点：需要语义理解但不需要多跳推理，单次向量检索即可。

# Output Format
JSON List:
[
  {{
    "query": "...",
    "strategy": "S2",
    "noise_analysis": "关键词不匹配，需要语义理解...",
    "reasoning": "查询需要理解同义词或paraphrase..."
  }}
]
只输出 JSON 对象，不要包含 markdown 格式。
""",
    "S3": """
# Role
你是一个数据合成专家。我们需要为 RAG 系统生成"高精准度匹配"的测试查询。
当前关注领域：{domain}

# Task
请生成 20 条必须使用 **S3 (SPLADE 稀疏检索)** 策略的复杂查询。

# Context & Constraints
1. **策略定义**：S3 侧重于关键词的精确匹配，而非语义相关性。
2. **适用场景**：查询中包含极易混淆的**特定实体**（如具体型号、错误代码、化学式、罕见人名）。
3. **噪声风险**：如果使用向量检索（Dense Retrieval），模型会被语义相似的文档误导（例如用户问 "iPhone 14 Plus"，向量检索可能找回 "iPhone 14 Pro" 的评测，这就是噪声）。

# Real Examples (Learn from these)

## S3 策略的真实样本：
{target_examples}

## 其他策略的样本 (注意区分)：
{other_examples}

请仔细观察 S3 样本的特点：包含精确实体、型号、代码等需要精确匹配的内容，与其他策略的区别在于不能依赖语义理解。

# Output Format
JSON List:
[
  {{
    "query": "...",
    "strategy": "S3",
    "noise_analysis": "需要精确匹配实体 X，防止被 Y 混淆...",
    "reasoning": "由于查询包含精确的非语义符号..."
  }}
]
只输出 JSON 对象，不要包含 markdown 格式。
""",
    "S4": """
# Role
数据合成专家。
当前关注领域：{domain}

# Task
请生成 20 条必须使用 **S4 (混合检索+重排)** 的高难度查询。

# Context & Constraints
1. **策略定义**：S4 结合了关键词匹配和语义检索，并必须经过 Cross-Encoder 重排序。
2. **适用场景**：
   - **多视角冲突**：问题涉及争议性话题，不同文档可能有冲突观点。
   - **实体比较**：问"A和B的区别"，需要同时找回A和B的文档。
   - **长尾/模糊查询**：查询意图不明确，单一检索方式召回率低。
3. **噪声风险**：单路检索（如只用 BM25）会漏掉重要信息；只用向量检索会引入大量不相关的"幻觉"文档。必须通过 Rerank 机制过滤掉 90% 的噪声。

# Real Examples (Learn from these)

## S4 策略的真实样本：
{target_examples}

## 其他策略的样本 (注意区分)：
{other_examples}

请注意 S4 样本的特点：需要综合多个角度、比较实体，单一检索策略不足以应对。

# Output Format
JSON List:
[
  {{
    "query": "...",
    "strategy": "S4",
    "noise_analysis": "单一检索无法覆盖 A 和 B 两个方面，且容易引入...",
    "reasoning": "问题涉及跨文档对比，需要高召回率配合重排序..."
  }}
]
只输出 JSON 对象，不要包含 markdown 格式。
""",
    "M1": """
# Role
你是一个数据合成专家。我们需要为 RAG 系统生成"简单语义理解"的测试查询。
当前关注领域：{domain}

# Task
请生成 20 条适合 **M1 (BM25 + LLM)** 策略的查询。

# Context & Constraints
1. **策略定义**：M1 使用 BM25 检索，然后通过 LLM 进行简单的理解和回答。
2. **适用场景**：查询明确，关键词匹配即可找到文档，但需要 LLM 总结或提取答案。
3. **噪声风险**：关键词检索足够，不需要向量检索，但需要 LLM 理解文档内容。

# Real Examples (Learn from these)

## M1 策略的真实样本：
{target_examples}

## 其他策略的样本 (注意区分)：
{other_examples}

请注意 M1 样本的特点：查询直接，BM25 可以找到文档，但需要 LLM 理解和回答。

# Output Format
JSON List:
[
  {{
    "query": "...",
    "strategy": "M1",
    "noise_analysis": "BM25 足够找到文档，需要 LLM 理解...",
    "reasoning": "查询明确，关键词匹配后需要 LLM 总结..."
  }}
]
只输出 JSON 对象，不要包含 markdown 格式。
""",
    "M2": """
# Role
数据合成专家。
当前关注领域：{domain}

# Task
请生成 20 条适合 **M2 (密集向量检索 + 多跳推理)** 的查询。

# Context & Constraints
1. **策略定义**：问题不能通过关键词直接搜到（Keyword Mismatch），必须依靠向量（Embedding）理解隐含意图。
2. **适用场景**：描述性问题、隐喻性问题、或者不知道具体实体名称的搜索。
3. **噪声风险**：关键词检索（S1/S3）完全失效（找不到文档）；需要向量检索捕捉"意思"，但需要 LLM 进行多步推断。

# Real Examples (Learn from these)

## M2 策略的真实样本：
{target_examples}

## 其他策略的样本 (注意区分)：
{other_examples}

请注意 M2 样本的特点：查询通常是描述性的、隐喻性的，或者不包含精确关键词，需要依赖语义理解和多步推理。

# Output Format
JSON List:
[
  {{
    "query": "...",
    "strategy": "M2",
    "noise_analysis": "关键词无法匹配...",
    "reasoning": "问题描述比较隐晦，需要语义理解..."
  }}
]
只输出 JSON 对象，不要包含 markdown 格式。
""",
    "M4": """
# Role
数据合成专家。
当前关注领域：{domain}

# Task
请生成 20 条必须使用 **M4 (多跳混合检索)** 的高难度查询。

# Context & Constraints
1. **策略定义**：M4 需要多轮检索和推理，结合关键词和语义检索，通过 LLM 进行复杂推理。
2. **适用场景**：
   - **多跳推理**：答案需要综合多个文档的信息。
   - **复杂关系**：需要理解实体之间的复杂关系。
   - **推理链**：需要 LLM 建立推理链条。
3. **噪声风险**：单次检索无法获得所有必要信息，需要多轮检索和深度推理。

# Real Examples (Learn from these)

## M4 策略的真实样本：
{target_examples}

## 其他策略的样本 (注意区分)：
{other_examples}

请注意 M4 样本的特点：需要多跳推理、综合多个文档、建立复杂推理链。

# Output Format
JSON List:
[
  {{
    "query": "...",
    "strategy": "M4",
    "noise_analysis": "需要多轮检索，单次无法获得完整信息...",
    "reasoning": "问题需要多跳推理，综合多个文档..."
  }}
]
只输出 JSON 对象，不要包含 markdown 格式。
""",
    "Z0": """
# Role
你是一个数据合成专家。我们需要为 RAG 系统生成"简单直接"的测试查询。
当前关注领域：{domain}

# Task
请生成 20 条适合 **Z0 (无需检索)** 策略的查询。

# Context & Constraints
1. **策略定义**：Z0 表示问题非常简单，不需要检索外部知识库，LLM 凭借自身知识即可回答。
2. **适用场景**：常识性问题、通用知识问题、或者 LLM 已经知道答案的问题。
3. **噪声风险**：检索反而会引入不必要的噪声，直接用 LLM 回答即可。

# Real Examples (Learn from these)

## Z0 策略的真实样本：
{target_examples}

## 其他策略的样本 (注意区分)：
{other_examples}

请注意 Z0 样本的特点：问题简单直接，LLM 自身知识足够，不需要检索。

# Output Format
JSON List:
[
  {{
    "query": "...",
    "strategy": "Z0",
    "noise_analysis": "无需检索，LLM 自身知识足够...",
    "reasoning": "问题为常识或通用知识，直接回答即可..."
  }}
]
只输出 JSON 对象，不要包含 markdown 格式。
"""
}

def sample_examples(existing_data, target_strategy, num_target=3, num_other=2):
    """从现有数据中采样示例

    Args:
        existing_data: 现有数据集
        target_strategy: 目标策略 (如 'S3', 'M2', 'S4', 'M4')
        num_target: 目标策略的样本数量
        num_other: 其他策略的样本数量

    Returns:
        dict: {"target": [...], "other": [...]}
    """
    if not existing_data:
        return {"target": [], "other": []}

    # 采样目标策略的样本
    target_items = [item for item in existing_data
                   if item.get('answer') == target_strategy]
    sampled_target = random.sample(target_items, min(num_target, len(target_items)))

    # 采样其他策略
    other_items = [item for item in existing_data
                  if item.get('answer') != target_strategy]
    sampled_other = random.sample(other_items, min(num_other, len(other_items)))

    return {"target": sampled_target, "other": sampled_other}

def format_examples(examples):
    """将样本格式化为 prompt 字符串"""
    if not examples:
        return "暂无示例"

    formatted = []
    for i, ex in enumerate(examples, 1):
        formatted.append(f"{i}. Query: \"{ex.get('question', '')}\"")
        formatted.append(f"   Strategy: {ex.get('answer', '')}")
        if 'complexity_analysis' in ex:
            reasoning = ex['complexity_analysis'].get('reasoning', '')
            if reasoning:
                formatted.append(f"   Reasoning: {reasoning[:150]}...")

    return "\n".join(formatted)

def generate_batch(client, model, prompt_key, domain, example_samples):
    """调用 LLM 生成一批数据，带重试机制

    Args:
        example_samples: dict with "target" and "other" keys containing sample items
    """
    prompt_template = PROMPTS[prompt_key]

    # 格式化示例
    target_examples_str = format_examples(example_samples.get("target", []))
    other_examples_str = format_examples(example_samples.get("other", []))

    prompt = prompt_template.format(
        domain=domain,
        target_examples=target_examples_str,
        other_examples=other_examples_str
    )

    max_retries = 3
    base_delay = 2

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant designed to output JSON list."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.8, # 稍微增加温度以提高多样性
                response_format={"type": "json_object"}
            )
            content = response.choices[0].message.content

            # 简单的清洗
            if "```json" in content:
                content = content.replace("```json", "").replace("```", "")

            data = json.loads(content)

            # 兼容有时候模型返回 {"items": [...]} 或者直接 [...]
            if isinstance(data, dict):
                for k, v in data.items():
                    if isinstance(v, list):
                        return v
                return [] # 没找到 list
            elif isinstance(data, list):
                return data
            else:
                return []

        except Exception as e:
            print(f"Error generating batch for {prompt_key} in {domain} (Attempt {attempt+1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                time.sleep(base_delay * (attempt + 1)) # 指数退避
            else:
                return []

def transform_to_schema(raw_item, index, strategy_prefix):
    """将生成的简化数据转换为项目标准格式"""
    
    # 提取生成的 strategy，如果没有则使用前缀
    strategy = raw_item.get('strategy', strategy_prefix).upper()
    
    # 构造符合 label_augmented.json 的结构
    new_item = {
        "id": f"syn_{strategy.lower()}_{int(time.time())}_{index}",
        "question": raw_item.get('query'),
        "answer": strategy, # 这里的 answer 在你的逻辑里就是 strategy label
        "recommended_strategy": strategy,
        "complexity_analysis": {
            "reasoning": raw_item.get('reasoning', 'Generated by targeted synthesis.'),
            "noise_risk": raw_item.get('noise_analysis', 'High noise risk identified.')
        },
        "search_intent": f"Targeted synthesis for {strategy}",
        "rationale": raw_item.get('reasoning', 'Generated rationale.')
    }
    return new_item

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, default="classifier/data/label_augmented.json")
    parser.add_argument("--output_file", type=str, default="classifier/data/label_balanced.json")
    parser.add_argument("--model", type=str, default="gpt-4o")
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--min_samples", type=int, default=150, help="Minimum samples per category")
    parser.add_argument("--batch_size", type=int, default=20, help="Number of samples per batch")
    args = parser.parse_args()

    # 路径处理
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
    if not os.path.isabs(args.input_file):
        args.input_file = os.path.join(project_root, args.input_file)
    if not os.path.isabs(args.output_file):
        args.output_file = os.path.join(project_root, args.output_file)

    # 初始化 OpenAI
    api_key = os.getenv("OPENAI_API_KEY") or "sk-BZQdNZeSwyih3TKpD95fDd83A90e4556A95f7eB7D489C36b"
    base_url = os.getenv("OPENAI_BASE_URL") or "https://api.gpt.ge/v1/"
    client = OpenAI(api_key=api_key, base_url=base_url)

    print(f"Loading existing data from {args.input_file}...")
    try:
        existing_data = load_json(args.input_file)
    except Exception:
        print("Input file not found, starting fresh.")
        existing_data = []

    print(f"Current count: {len(existing_data)}")

    # 统计现有数据的分布
    from collections import Counter
    label_counts = Counter([item.get('answer') for item in existing_data])

    print("\nCurrent Label Distribution:")
    print(f"{'Label':<10} | {'Count':<8} | {'Need':<8}")
    print("-" * 35)

    # 动态生成任务计划
    tasks = []
    all_strategies = ['S1', 'S2', 'S3', 'S4', 'M1', 'M2', 'M4', 'Z0']

    for strategy in all_strategies:
        current_count = label_counts.get(strategy, 0)
        need_count = max(0, args.min_samples - current_count)
        print(f"{strategy:<10} | {current_count:<8} | {need_count:<8}")

        if need_count > 0:
            # 计算需要生成的批次数 (每批 20 条)
            num_batches = (need_count + args.batch_size - 1) // args.batch_size

            # 为每个批次分配一个领域
            for i in range(num_batches):
                domain = DOMAINS[i % len(DOMAINS)]
                tasks.append((strategy, domain))

    print("-" * 35)
    print(f"\nTotal tasks to generate: {len(tasks)} batches")

    if not tasks:
        print("All categories already have sufficient samples!")
        return

    new_items = []
    print(f"Starting generation with {len(tasks)} tasks using {args.workers} workers...")

    # 为每个任务预先采样示例
    task_with_examples = []
    for task_type, domain in tasks:
        examples = sample_examples(existing_data, task_type, num_target=3, num_other=2)
        task_with_examples.append((task_type, domain, examples))

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(generate_batch, client, args.model, task[0], task[1], task[2]): task for task in task_with_examples}

        for future in tqdm(as_completed(futures), total=len(tasks)):
            task_type, domain, _ = futures[future]  # 忽略 examples，因为已经用过了
            batch_results = future.result()

            if batch_results:
                for idx, item in enumerate(batch_results):
                    processed_item = transform_to_schema(item, len(new_items) + idx, task_type)
                    if processed_item['question'] and processed_item['answer']:
                        new_items.append(processed_item)

    print(f"Generated {len(new_items)} new synthetic samples.")

    # 合并数据
    final_data = existing_data + new_items
    
    # 简单的 M3 -> M4 清洗 (可选，根据你的建议)
    # 如果 M3 数量极少，将其归类为 M4，避免模型学不到
    m3_count = sum(1 for item in final_data if item['answer'] == 'M3')
    if m3_count < 10:
        print(f"Warning: M3 count is very low ({m3_count}). Remapping M3 to M4...")
        for item in final_data:
            if item['answer'] == 'M3':
                item['answer'] = 'M4'
                item['recommended_strategy'] = 'M4'

    # 打印最终分布
    print("\nNew Label Distribution:")
    from collections import Counter
    counts = Counter([item['answer'] for item in final_data])
    total = len(final_data)
    print(f"{'Label':<10} | {'Count':<8} | {'Percentage':<10}")
    print("-" * 35)
    for label in sorted(counts.keys()):
        print(f"{label:<10} | {counts[label]:<8} | {counts[label]/total:.2%}")
    print("-" * 35)

    save_json(args.output_file, final_data)
    print(f"Saved balanced dataset to {args.output_file}")

if __name__ == "__main__":
    main()
