import os
import sys

# ================= [关键配置：强制离线] =================
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
# ======================================================

import torch
import re
import json
from datasets import load_dataset
from unsloth import FastLanguageModel
from tqdm import tqdm
from collections import Counter

# --- 路径配置 (请确保与训练时一致) ---
LORA_MODEL_DIR = "/root/autodl-tmp/output/Qwen2.5-3B-RAG-Router/lora_model"
BASE_MODEL_PATH = "/root/autodl-tmp/model/Qwen2.5-3B-Instruct"
DATA_FILE = "/root/graduateRAG/classifier/data/training_data/final_finetuning_dataset.jsonl"
MAX_SEQ_LENGTH = 2048

SYSTEM_PROMPT = """You are an expert RAG router. Analyze the user query complexity and determine the optimal retrieval strategy.
Output the analysis in the following format:
Analysis: <reasoning process>
Complexity: <L0/L1/L2>
Index: <None/Lexical/Semantic/Hybrid>
Action: <Z/S-Sparse/S-Dense/S-Hybrid/M>"""


def parse_action_with_hybrid_rule(text):
    """
    新规则：从 Complexity + Index 字段推导最终 Action。
    - L0 → Z
    - L1 + Lexical → S-Sparse
    - L1 + Semantic → S-Dense
    - L1 + Hybrid，或 L1 但 Index 不明确 → S-Hybrid（不确定时兜底）
    - L2 → M
    若 Complexity 无法解析，回退到直接读取 Action 字段。
    """
    # 1. 解析 Complexity
    comp_match = re.search(r"Complexity:\s*(L[012])", text)
    complexity = comp_match.group(1) if comp_match else None

    # 2. 解析 Index
    idx_match = re.search(r"Index:\s*(None|Lexical|Semantic|Hybrid)", text, re.IGNORECASE)
    index = idx_match.group(1).capitalize() if idx_match else None

    # 3. 从 Complexity + Index 推导 Action
    if complexity == "L0":
        return "Z"
    elif complexity == "L2":
        return "M"
    elif complexity == "L1":
        if index == "Lexical":
            return "S-Sparse"
        elif index == "Semantic":
            return "S-Dense"
        else:
            # Index=Hybrid 或无法解析 → 不确定，使用混合检索
            return "S-Hybrid"

    # 4. Complexity 无法解析时，回退到读取 Action 字段
    action_match = re.search(r"Action:\s*([A-Za-z0-9\-]+)", text)
    if action_match:
        action = action_match.group(1).strip().rstrip('.')
        if action in ["Z", "S-Sparse", "S-Dense", "S-Hybrid", "M"]:
            return action

    # 5. 最后兜底：扫描末行
    lines = text.strip().split('\n')
    last_line = lines[-1] if lines else ""
    for valid in ["Z", "S-Sparse", "S-Dense", "S-Hybrid", "M"]:
        if valid in last_line:
            return valid

    return "Unknown"


def map_to_coarse(label):
    if label in ["S-Sparse", "S-Dense", "S-Hybrid"]:
        return "S"
    return label


def main():
    print(f"🚀 Loading LoRA from {LORA_MODEL_DIR}...")

    try:
        print(f"   Base Model: {BASE_MODEL_PATH}")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=BASE_MODEL_PATH,
            max_seq_length=MAX_SEQ_LENGTH,
            dtype=None,
            load_in_4bit=False,
            local_files_only=True,
        )
        print("   Attaching LoRA adapters...")
        model.load_adapter(LORA_MODEL_DIR)
    except Exception as e:
        print(f"\n❌ 模型加载失败: {e}")
        return

    FastLanguageModel.for_inference(model)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # 加载测试集，使用 80:20 分割
    print("\n📂 Loading test dataset (80:20 split)...")
    full_dataset = load_dataset("json", data_files=DATA_FILE, split="train")
    dataset_split = full_dataset.train_test_split(test_size=0.2, seed=42)
    test_dataset = dataset_split["test"]
    print(f"📊 Evaluating on {len(test_dataset)} samples...")

    y_true_fine = []
    y_pred_fine = []

    for item in tqdm(test_dataset, desc="Inference"):
        query = item["question_text"]
        true_action = item["action"]

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": query}
        ]

        inputs = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
        ).to("cuda")

        outputs = model.generate(
            inputs,
            max_new_tokens=256,
            temperature=0.01,
            use_cache=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

        response = tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)
        pred_action = parse_action_with_hybrid_rule(response)

        y_true_fine.append(true_action)
        y_pred_fine.append(pred_action)

    y_true_coarse = [map_to_coarse(y) for y in y_true_fine]
    y_pred_coarse = [map_to_coarse(y) for y in y_pred_fine]

    def compute_metrics(y_true, y_pred, labels):
        rows = {}
        for l in labels:
            tp = sum(1 for t, p in zip(y_true, y_pred) if t == l and p == l)
            fp = sum(1 for t, p in zip(y_true, y_pred) if t != l and p == l)
            fn = sum(1 for t, p in zip(y_true, y_pred) if t == l and p != l)
            support = sum(1 for t in y_true if t == l)
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
            rows[l] = dict(prec=prec, rec=rec, f1=f1, support=support)
        acc = sum(1 for t, p in zip(y_true, y_pred) if t == p) / len(y_true)
        macro_p = sum(rows[l]['prec'] for l in labels) / len(labels)
        macro_r = sum(rows[l]['rec']  for l in labels) / len(labels)
        macro_f = sum(rows[l]['f1']   for l in labels) / len(labels)
        return rows, acc, (macro_p, macro_r, macro_f)

    def print_report(rows, acc, macro, labels):
        print(f"\n{'类别':<12} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>8}")
        print("-" * 55)
        for l in labels:
            r = rows[l]
            print(f"{l:<12} {r['prec']:10.4f} {r['rec']:10.4f} {r['f1']:10.4f} {r['support']:8d}")
        print("-" * 55)
        print(f"{'宏平均':<12} {macro[0]:10.4f} {macro[1]:10.4f} {macro[2]:10.4f}")
        print(f"\n总体准确率: {acc:.4f} ({acc:.2%})")

    def print_confusion(y_true, y_pred, labels):
        header = "True \\ Pred"
        print(f"\n{header:<12}", end="")
        for l in labels:
            print(f" {l:<10}", end="")
        print()
        for tl in labels:
            print(f"{tl:<12}", end="")
            for pl in labels:
                count = sum(1 for t, p in zip(y_true, y_pred) if t == tl and p == pl)
                print(f" {count:<10}", end="")
            print()

    # ===== 粗粒度报告 =====
    labels_coarse = ["Z", "S", "M"]
    rows_c, acc_c, macro_c = compute_metrics(y_true_coarse, y_pred_coarse, labels_coarse)
    print("\n" + "#" * 60)
    print("Coarse-grained Evaluation (Z vs S vs M)")
    print("#" * 60)
    print_report(rows_c, acc_c, macro_c, labels_coarse)
    print_confusion(y_true_coarse, y_pred_coarse, labels_coarse)

    # ===== 细粒度报告 =====
    labels_fine = ["Z", "S-Sparse", "S-Dense", "S-Hybrid", "M"]
    rows_f, acc_f, macro_f = compute_metrics(y_true_fine, y_pred_fine, labels_fine)
    print("\n" + "=" * 60)
    print("Fine-grained Evaluation (5-class)")
    print("=" * 60)
    print_report(rows_f, acc_f, macro_f, labels_fine)
    print_confusion(y_true_fine, y_pred_fine, labels_fine)

    unknown_count = y_pred_fine.count("Unknown")
    if unknown_count > 0:
        print(f"\nWarning: {unknown_count} samples could not be parsed (Unknown).")

    # 保存结果
    output_log = os.path.join(os.path.dirname(DATA_FILE), "evaluation_results_hybrid_rule.json")
    with open(output_log, 'w') as f:
        json.dump({
            "coarse_accuracy": coarse_acc,
            "fine_accuracy": fine_acc,
            "y_true_fine": y_true_fine,
            "y_pred_fine": y_pred_fine
        }, f)
    print(f"\n💾 Results saved to {output_log}")


if __name__ == "__main__":
    main()
