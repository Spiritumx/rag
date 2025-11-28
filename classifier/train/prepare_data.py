import json
import os
import sys
import random
import numpy as np
from sklearn.model_selection import train_test_split
from datasets import Dataset, DatasetDict

# 路径配置
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
INPUT_FILE = os.path.join(PROJECT_ROOT, "classifier", "data", "label_balanced.json")
# OUTPUT_DIR = os.path.join(PROJECT_ROOT, "classifier", "train", "data")
OUTPUT_DIR = "/root/autodl-tmp/data"

def format_instruction(item):
    """
    将原始数据转换为 Chat 格式 (Qwen2.5 Instruct 偏好这种格式)
    """
    question = item['question']
    label = item['answer'] # e.g., S3
    
    # 提取 reasoning，如果不存在则使用默认文本
    analysis = item.get('complexity_analysis', {})
    reasoning = analysis.get('reasoning', item.get('rationale', ''))
    noise_risk = analysis.get('noise_risk', '')
    
    # 构建 System Prompt
    system_prompt = """You are an expert in Information Retrieval optimization. Your task is to analyze the user's query and recommend the most efficient retrieval strategy to minimize noise and latency.

Available Strategies:
- Z0: No Retrieval (LLM Internal Knowledge). Use for logic/common sense/self-contained questions.
- S1: BM25 (Keyword Match). Use for simple questions with unique keywords.
- S2: Dense Retrieval (Vector). Use for semantic questions where keywords might mismatch.
- S3: Sparse Retrieval (SPLADE). Use for precise entity/ID matching to avoid semantic drift.
- S4: Hybrid + Rerank. Use for complex entities requiring high recall and precision.
- M1-M4: Multi-hop versions of above. Use when reasoning requires combining information from multiple documents.
"""

    # 构建 User Input
    user_content = f"Question: \"{question}\"\n\nAnalyze the query complexity and determine the optimal strategy."

    # 构建 Assistant Output (包含思维链 CoT)
    assistant_content = f"""### Analysis:
{reasoning}

### Risk Assessment:
{noise_risk}

### Recommendation:
The optimal strategy is [[{label}]]."""

    return {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content}
        ],
        "label": label, # 保留 label 用于分层
        "is_synthetic": str(item.get('id', '')).startswith('syn_') # 标记是否为合成数据
    }

def main():
    print(f"Loading data from {INPUT_FILE}...")
    
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)
    
    print(f"Total samples: {len(raw_data)}")
    
    # 1. 格式化数据并打标
    formatted_data = [format_instruction(item) for item in raw_data]
    
    # 2. 分离真实数据和合成数据
    real_data = [d for d in formatted_data if not d['is_synthetic']]
    syn_data = [d for d in formatted_data if d['is_synthetic']]
    
    print(f"Real Data: {len(real_data)}")
    print(f"Synthetic Data: {len(syn_data)}")

    # 3. 优先从真实数据中划分测试集 (Test Set)
    # 我们希望测试集尽可能反映真实分布，但同时覆盖所有类别
    # 策略：从真实数据中分层抽取 15% 作为测试集
    
    real_labels = [d['label'] for d in real_data]
    
    # 检查真实数据中是否有某些类别样本太少，导致无法分层划分
    from collections import Counter
    label_counts = Counter(real_labels)
    print("Real Data Label Distribution:", label_counts)
    
    # 过滤掉只有1个样本的类别，防止 stratify 报错
    valid_indices = [i for i, label in enumerate(real_labels) if label_counts[label] > 1]
    valid_real_data = [real_data[i] for i in valid_indices]
    valid_real_labels = [real_labels[i] for i in valid_indices]
    
    # 剩下的那些只有1个样本的数据，直接放入训练集，因为放测试集也没法统计
    leftover_data = [real_data[i] for i in range(len(real_data)) if i not in valid_indices]

    try:
        train_real, test_real = train_test_split(
            valid_real_data,
            test_size=0.15, # 15% 的真实数据作为测试集
            random_state=42,
            shuffle=True,
            stratify=valid_real_labels
        )
    except ValueError as e:
        print(f"Warning: Stratified split failed ({e}). Fallback to random split.")
        train_real, test_real = train_test_split(
            valid_real_data,
            test_size=0.15,
            random_state=42,
            shuffle=True
        )

    # 4. 组装最终数据集
    # 训练集 = 剩余的真实数据 + 所有的合成数据 + 之前剩下的孤儿数据
    final_train = train_real + syn_data + leftover_data
    # 测试集 = 抽出来的真实数据
    final_test = test_real
    
    # 打乱训练集
    random.seed(42)
    random.shuffle(final_train)

    print("-" * 30)
    print(f"Final Train Set: {len(final_train)} (Real: {len(train_real) + len(leftover_data)}, Syn: {len(syn_data)})")
    print(f"Final Test Set:  {len(final_test)} (All Real)")
    print("-" * 30)

    # 移除辅助字段
    def clean_dataset(data_list):
        return [{k: v for k, v in item.items() if k in ['messages']} for item in data_list]

    dataset = DatasetDict({
        "train": Dataset.from_list(clean_dataset(final_train)),
        "test": Dataset.from_list(clean_dataset(final_test))
    })
    
    # 保存
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    dataset.save_to_disk(OUTPUT_DIR)
    print(f"Dataset saved to {OUTPUT_DIR}")
    
    # 创建 jsonl 供检查
    with open(os.path.join(OUTPUT_DIR, "test_set_preview.jsonl"), 'w', encoding='utf-8') as f:
        for item in final_test[:20]:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    main()
