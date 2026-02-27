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
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

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

    # ===== 报告输出 =====

    print("\n" + "#" * 60)
    print("🚀 Coarse-grained Evaluation (Z vs S vs M)")
    print("#" * 60)
    coarse_acc = accuracy_score(y_true_coarse, y_pred_coarse)
    print(f"🏆 Overall Accuracy: {coarse_acc:.4f} ({coarse_acc:.2%})")
    labels_coarse = ["Z", "S", "M"]
    print(classification_report(y_true_coarse, y_pred_coarse, labels=labels_coarse, digits=4))

    print("\n[Confusion Matrix (Coarse)]")
    print(f"{'True \\ Pred':<12} {'Z':<8} {'S':<8} {'M':<8}")
    cm = confusion_matrix(y_true_coarse, y_pred_coarse, labels=labels_coarse)
    for idx, label in enumerate(labels_coarse):
        print(f"{label:<12} {cm[idx][0]:<8} {cm[idx][1]:<8} {cm[idx][2]:<8}")

    print("\n" + "=" * 60)
    print("🔬 Fine-grained Evaluation (5-class)")
    print("=" * 60)
    labels_fine = ["Z", "S-Sparse", "S-Dense", "S-Hybrid", "M"]
    fine_acc = accuracy_score(y_true_fine, y_pred_fine)
    print(f"🏆 Overall Accuracy: {fine_acc:.4f} ({fine_acc:.2%})")
    print(classification_report(y_true_fine, y_pred_fine, labels=labels_fine, digits=4))

    unknown_count = y_pred_fine.count("Unknown")
    if unknown_count > 0:
        print(f"\n⚠️ Warning: {unknown_count} samples could not be parsed (Unknown).")

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
