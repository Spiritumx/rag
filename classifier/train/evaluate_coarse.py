import os
import sys

# ================= [关键配置：强制离线] =================
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
# ======================================================

import torch
import re
import json
import numpy as np
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

def parse_action(text):
    """从模型输出中提取 Action (增强版)"""
    # 1. 尝试标准匹配
    match = re.search(r"Action:\s*([A-Za-z0-9\-]+)", text)
    if match:
        action = match.group(1).strip().rstrip('.')
        valid_actions = ["Z", "S-Sparse", "S-Dense", "S-Hybrid", "M"]
        if action in valid_actions:
            return action
            
    # 2. 兜底匹配：在最后一行查找
    lines = text.strip().split('\n')
    if not lines: return "Unknown"
    last_line = lines[-1]
    for valid in ["Z", "S-Sparse", "S-Dense", "S-Hybrid", "M"]:
        if valid in last_line:
            return valid
    return "Unknown"

def map_to_coarse(label):
    """
    【核心逻辑】将细粒度标签映射为粗粒度 (Z / S / M)
    """
    if label in ["S-Sparse", "S-Dense", "S-Hybrid"]:
        return "S"
    return label

def main():
    print(f"🚀 Loading LoRA from {LORA_MODEL_DIR}...")
    
    # 1. 安全加载模型 (离线模式)
    try:
        print(f"   Base Model: {BASE_MODEL_PATH}")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name = BASE_MODEL_PATH,
            max_seq_length = MAX_SEQ_LENGTH,
            dtype = None,
            load_in_4bit = False,
            local_files_only = True,
        )
        print("   Attaching LoRA adapters...")
        model.load_adapter(LORA_MODEL_DIR)
    except Exception as e:
        print(f"\n❌ 模型加载失败: {e}")
        return

    FastLanguageModel.for_inference(model)
    
    # Tokenizer 修正
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # 2. 加载测试集
    print("\n📂 Loading test dataset...")
    full_dataset = load_dataset("json", data_files=DATA_FILE, split="train")
    # 必须使用与训练时完全相同的 seed=42 来切分，否则测试集会混入训练数据
    dataset_split = full_dataset.train_test_split(test_size=0.1, seed=42)
    test_dataset = dataset_split["test"]
    
    print(f"📊 Evaluating on {len(test_dataset)} samples...")

    y_true_fine = [] # 细粒度真值
    y_pred_fine = [] # 细粒度预测
    
    # 3. 批量推理
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
        pred_action = parse_action(response)
        
        y_true_fine.append(true_action)
        y_pred_fine.append(pred_action)

    # 4. 数据映射 (Fine -> Coarse)
    y_true_coarse = [map_to_coarse(y) for y in y_true_fine]
    y_pred_coarse = [map_to_coarse(y) for y in y_pred_fine]

    # ================= 报告输出 =================

    # 报告 1: 粗粒度 (Z / S / M) - 论文主结果
    print("\n" + "#"*60)
    print("🚀 Coarse-grained Evaluation (Z vs S vs M)")
    print("#"*60)
    
    coarse_acc = accuracy_score(y_true_coarse, y_pred_coarse)
    print(f"🏆 Overall Accuracy: {coarse_acc:.2%}")
    print("-" * 60)
    
    labels_coarse = ["Z", "S", "M"]
    print(classification_report(y_true_coarse, y_pred_coarse, labels=labels_coarse, digits=4))
    
    print("\n[Confusion Matrix]")
    print(f"{'True \\ Pred':<12} {'Z':<8} {'S':<8} {'M':<8}")
    cm = confusion_matrix(y_true_coarse, y_pred_coarse, labels=labels_coarse)
    for idx, label in enumerate(labels_coarse):
        print(f"{label:<12} {cm[idx][0]:<8} {cm[idx][1]:<8} {cm[idx][2]:<8}")

    # 报告 2: 细粒度 (Index Strategy) - 用于分析局限性
    print("\n" + "="*60)
    print("🔬 Fine-grained Evaluation (Index Strategy Analysis)")
    print("="*60)
    
    # 过滤掉 Unknown 以防报错
    labels_fine = sorted(list(set([l for l in y_true_fine + y_pred_fine if l != "Unknown"])))
    print(classification_report(y_true_fine, y_pred_fine, labels=labels_fine, digits=4))

    # 保存结果到文件（方便画图）
    output_log = os.path.join(os.path.dirname(DATA_FILE), "evaluation_results.json")
    with open(output_log, 'w') as f:
        json.dump({
            "coarse_accuracy": coarse_acc,
            "y_true_fine": y_true_fine,
            "y_pred_fine": y_pred_fine
        }, f)
    print(f"\n💾 Results saved to {output_log}")

if __name__ == "__main__":
    main()