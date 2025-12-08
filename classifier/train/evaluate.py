import os
import sys

# ================= [关键修复 1：暴力离线] =================
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
# ========================================================

import torch
import re
import json
from datasets import load_dataset
from unsloth import FastLanguageModel
from tqdm import tqdm
from sklearn.metrics import classification_report

# --- 配置 ---
# LoRA 权重路径
LORA_MODEL_DIR = "/root/autodl-tmp/output/Qwen2.5-3B-RAG-Router/lora_model"

# 本地基座模型路径 (必须存在！)
# 如果你的 LoRA config 记的是远程路径，这里需要强制指定本地路径以实现离线加载
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
    """
    从模型输出中提取 Action，增强鲁棒性
    支持: "Action: M", "Action: **M**", "Action: M."
    """
    # 1. 尝试标准匹配
    match = re.search(r"Action:\s*([A-Za-z0-9\-]+)", text)
    if match:
        action = match.group(1).strip()
        # 简单清洗，去掉可能的标点
        action = action.rstrip('.')
        
        # 验证是否是合法标签
        valid_actions = ["Z", "S-Sparse", "S-Dense", "S-Hybrid", "M"]
        if action in valid_actions:
            return action
            
    # 2. 如果没找到，尝试在最后一行找
    lines = text.strip().split('\n')
    last_line = lines[-1]
    for valid in ["Z", "S-Sparse", "S-Dense", "S-Hybrid", "M"]:
        if valid in last_line:
            return valid
            
    return "Unknown"

def main():
    print(f"🚀 Loading LoRA from {LORA_MODEL_DIR}...")
    
    # ================= [关键修复 2：加载逻辑] =================
    # Unsloth 加载 LoRA 时，会自动去读取 adapter_config.json 里的 base_model。
    # 如果那是远程路径且断网，会报错。
    # 这里我们不用 from_pretrained 直接加载 LoRA，而是先加载本地 Base，再加载 LoRA。
    
    try:
        # 1. 加载本地基座模型
        print(f"   Base Model: {BASE_MODEL_PATH}")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name = BASE_MODEL_PATH, # 强制用本地基座
            max_seq_length = MAX_SEQ_LENGTH,
            dtype = None,
            load_in_4bit = False,
            local_files_only = True,
        )
        
        # 2. 加载 LoRA 适配器
        print("   Attaching LoRA adapters...")
        model.load_adapter(LORA_MODEL_DIR)
        
    except Exception as e:
        print(f"\n❌ 模型加载失败: {e}")
        print("尝试直接加载 LoRA 目录（如果 config 正常的话）...")
        # 备选方案：直接加载 LoRA (要求 adapter_config.json 里的路径能被解析)
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name = LORA_MODEL_DIR,
            max_seq_length = MAX_SEQ_LENGTH,
            dtype = None,
            load_in_4bit = False,
        )

    # 3. 推理加速设置
    FastLanguageModel.for_inference(model)
    
    # 4. Tokenizer 修正 (防止无限生成或报错)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # 推理时 padding 在左边是最佳实践（虽然 batch=1 没影响，但为了规范）
    tokenizer.padding_side = "left" 

    # 2. 加载测试集
    print("\n📂 Loading test dataset...")
    full_dataset = load_dataset("json", data_files=DATA_FILE, split="train")
    dataset_split = full_dataset.train_test_split(test_size=0.1, seed=42)
    test_dataset = dataset_split["test"]
    
    print(f"📊 Evaluating on {len(test_dataset)} samples...")

    y_true = []
    y_pred = []
    
    # 计数器
    correct = 0
    total = 0

    # 3. 批量推理
    for item in tqdm(test_dataset, desc="Inference"):
        query = item["question_text"]
        true_action = item["action"]

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": query}
        ]
        
        # apply_chat_template 返回的是 tensor
        inputs = tokenizer.apply_chat_template(
            messages, 
            tokenize=True, 
            add_generation_prompt=True, 
            return_tensors="pt"
        ).to("cuda")

        # 生成
        outputs = model.generate(
            inputs, 
            max_new_tokens=256, 
            temperature=0.01, # 低温，保证确定性
            use_cache=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
        
        # 解码 (只取生成部分)
        response = tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)
        
        # 解析
        pred_action = parse_action(response)
        
        y_true.append(true_action)
        y_pred.append(pred_action)
        
        if pred_action == true_action:
            correct += 1
        total += 1

    # 4. 输出报告
    print("\n" + "="*60)
    print(f"🔥 Final Accuracy: {correct/total:.2%}")
    print("="*60)
    print("Classification Report:")
    
    # 获取所有出现的标签，确保 Unknown 也能显示
    all_labels = sorted(list(set(y_true + y_pred)))
    
    # 为了报告好看，过滤掉 Unknown (如果它不在 y_true 里)
    target_names = [l for l in all_labels if l != "Unknown"]
    
    print(classification_report(y_true, y_pred, labels=target_names, digits=4))
    
    # 如果有解析失败的，单独打印
    unknown_count = y_pred.count("Unknown")
    if unknown_count > 0:
        print(f"\n⚠️ Warning: {unknown_count} samples could not be parsed (labeled as Unknown).")

if __name__ == "__main__":
    main()