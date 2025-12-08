import os
import sys
import re
import torch
import gc
from unsloth import FastLanguageModel
from datasets import load_from_disk
from tqdm import tqdm
from transformers import AutoTokenizer
from collections import defaultdict

# --- 配置 ---
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["UNSLOTH_DISABLE_STATISTICS"] = "1"

# 路径配置
LOCAL_MODEL_PATH = "/root/autodl-tmp/model/Qwen2.5-3B-Instruct"
BASE_MODEL_NAME = LOCAL_MODEL_PATH if os.path.exists(LOCAL_MODEL_PATH) else "Qwen/Qwen2.5-3B-Instruct"
LORA_PATH = "/root/autodl-tmp/output/lora_model"
DATA_DIR = "/root/autodl-tmp/data"

MAX_SEQ_LENGTH = 2048
DTYPE = torch.bfloat16
LOAD_IN_4BIT = False # 显存够的话建议 False，精度更高

def extract_label(text):
    """标签提取逻辑 (验证过有效)"""
    if not text: return "UNKNOWN"
    text_upper = text.upper()
    # 优先匹配 [[S1]] 格式
    match = re.search(r'\[\[([ZSM]\d)\]\]', text_upper)
    if match: return match.group(1)
    # 兜底匹配单词边界
    matches = re.findall(r'\b([ZSM]\d)\b', text_upper)
    if matches: return matches[-1]
    return "UNKNOWN"

def get_coarse_label(label):
    if not label or label == "UNKNOWN": return "UNKNOWN"
    return label[0]

def main():
    print("=" * 60)
    print("🚀 全量评估脚本 (LoRA Mode)")
    print("=" * 60)

    # 1. 强制加载官方 Tokenizer (修复乱码的关键)
    print(f"Loading Base Tokenizer from: {BASE_MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME, trust_remote_code=True)
    tokenizer.padding_side = "left" # 生成任务必须 Left Padding
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # 2. 加载 LoRA 模型
    print(f"Loading LoRA Model from: {LORA_PATH}")
    model, _ = FastLanguageModel.from_pretrained(
        model_name = LORA_PATH,
        max_seq_length = MAX_SEQ_LENGTH,
        dtype = DTYPE,
        load_in_4bit = LOAD_IN_4BIT,
    )
    FastLanguageModel.for_inference(model)
    print("✓ 模型加载完成")

    # 3. 加载测试数据
    dataset = load_from_disk(DATA_DIR)
    test_data = dataset["test"]
    print(f"✓ 测试集大小: {len(test_data)}")

    # 4. 开始评估
    stats = {
        "fine": {"correct": 0, "total": 0},
        "coarse": {"correct": 0, "total": 0},
        "unknown": 0
    }
    
    predictions_log = []
    
    print("\n开始推理...")
    pbar = tqdm(test_data)
    
    for item in pbar:
        # 准备输入
        input_msgs = item["messages"][:-1]
        gt_text = item["messages"][-1]["content"]
        gt_fine = extract_label(gt_text)
        gt_coarse = get_coarse_label(gt_fine)

        # 构建 Prompt
        inputs = tokenizer.apply_chat_template(
            input_msgs,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True
        ).to("cuda")

        # 生成
        with torch.no_grad():
            outputs = model.generate(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_new_tokens=256, # 稍微缩短一点，提高速度
                do_sample=False,    # 贪婪解码，最稳定
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.05
            )

        # 解码
        pred_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True)
        pred_fine = extract_label(pred_text)
        pred_coarse = get_coarse_label(pred_fine)

        # 统计
        if pred_fine == "UNKNOWN":
            stats["unknown"] += 1
        
        is_fine_correct = (gt_fine == pred_fine)
        if is_fine_correct: stats["fine"]["correct"] += 1
        stats["fine"]["total"] += 1

        is_coarse_correct = (gt_coarse == pred_coarse) and (gt_coarse != "UNKNOWN")
        if is_coarse_correct: stats["coarse"]["correct"] += 1
        stats["coarse"]["total"] += 1

        # 实时显示准确率
        pbar.set_postfix({
            "Fine": f"{stats['fine']['correct']/stats['fine']['total']:.1%}",
            "Coarse": f"{stats['coarse']['correct']/stats['coarse']['total']:.1%}"
        })

        predictions_log.append({
            "gt": gt_fine, "pred": pred_fine, "correct": is_fine_correct,
            "output": pred_text
        })

    # 5. 最终报告
    total = len(test_data)
    print("\n" + "="*60)
    print(f"最终结果 (Sample: {total})")
    print(f"细粒度准确率 (Fine Acc):   {stats['fine']['correct']/total:.2%}")
    print(f"粗粒度准确率 (Coarse Acc): {stats['coarse']['correct']/total:.2%}")
    print(f"UNKNOWN 占比:            {stats['unknown']/total:.2%}")
    print("="*60)
    
    # 保存日志
    save_log(predictions_log, LORA_PATH)

def save_log(logs, base_dir):
    # 保存到 output 根目录
    output_dir = os.path.dirname(base_dir) 
    log_path = os.path.join(output_dir, "eval_final_lora.txt")
    
    with open(log_path, "w", encoding="utf-8") as f:
        # 混淆矩阵
        from collections import defaultdict
        matrix = defaultdict(lambda: defaultdict(int))
        for l in logs: matrix[l['gt']][l['pred']] += 1
        
        f.write("=== Confusion Matrix ===\n")
        for gt in sorted(matrix.keys()):
            for pred in sorted(matrix[gt].keys()):
                f.write(f"{gt} -> {pred}: {matrix[gt][pred]}\n")
        f.write("\n=== Errors ===\n")
        for l in [x for x in logs if not x['correct']]:
            f.write(f"GT: {l['gt']} | Pred: {l['pred']}\nOutput: {l['output'][:200]}...\n---\n")
            
    print(f"日志已保存至: {log_path}")

if __name__ == "__main__":
    main()