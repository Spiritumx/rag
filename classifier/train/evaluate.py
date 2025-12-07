import os
import sys
import re
import torch
import gc
from unsloth import FastLanguageModel
from datasets import load_from_disk
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

# --- 配置 ---
# 设置 HF 镜像源
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
# 禁用 Unsloth 统计检查 (加快启动)
os.environ["UNSLOTH_DISABLE_STATISTICS"] = "1"

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
LOCAL_MODEL_PATH = "/root/autodl-tmp/model/Qwen2.5-3B-Instruct"
# 如果本地没有，则使用 HF Hub ID
BASE_MODEL_NAME = LOCAL_MODEL_PATH if os.path.exists(LOCAL_MODEL_PATH) else "Qwen/Qwen2.5-3B-Instruct"

LORA_PATH = "/root/autodl-tmp/output/lora_model"
MERGED_MODEL_PATH = "/root/autodl-tmp/output/merged_model"
DATA_DIR = "/root/autodl-tmp/data"

MAX_SEQ_LENGTH = 2048
DTYPE = torch.bfloat16
LOAD_IN_4BIT = False

def get_best_available_model():
    """自动判断使用合并模型还是 LoRA"""
    if os.path.exists(MERGED_MODEL_PATH) and os.listdir(MERGED_MODEL_PATH):
        print(f"✓ 检测到合并模型: {MERGED_MODEL_PATH}")
        return MERGED_MODEL_PATH, True
    elif os.path.exists(LORA_PATH) and os.listdir(LORA_PATH):
        print(f"✓ 检测到 LoRA 适配器: {LORA_PATH}")
        return LORA_PATH, False
    else:
        raise FileNotFoundError(f"未找到有效模型！请检查:\n1. {MERGED_MODEL_PATH}\n2. {LORA_PATH}")

def extract_label(text):
    """
    强化版标签提取逻辑 (v4)
    """
    if not text:
        return "UNKNOWN"

    text = text.strip()
    text_upper = text.upper()
    
    # 0. 预处理：移除 Markdown 代码块标记（如果有）
    text_upper = text_upper.replace("```JSON", "").replace("```", "")

    # 1. 优先匹配标准输出格式 [[LABEL]]
    match = re.search(r'\[\[([ZSM]\d)\]\]', text_upper)
    if match: return match.group(1)

    # 2. 匹配 key-value 格式
    match = re.search(r'(?:LABEL|CATEGORY|CLASS|ANSWER|RESULT)\s*[:：-]\s*([ZSM]\d)', text_upper)
    if match: return match.group(1)

    # 3. 匹配单独的括号格式 [LABEL] 或 (LABEL)
    match = re.search(r'[\[\(]([ZSM]\d)[\]\)]', text_upper)
    if match: return match.group(1)

    # 4. 匹配最后出现的标签（通常 CoT 的结论在最后）
    # 使用 \b 确保匹配单词边界，避免匹配到类似 "CLASS1" 中的 S1
    all_matches = re.findall(r'\b([ZSM]\d)\b', text_upper)
    if all_matches:
        return all_matches[-1]

    # 5. 关键词推断兜底
    text_lower = text.lower()
    if 'no retrieval' in text_lower or 'zero retrieval' in text_lower: return 'Z0'
    if 'single-hop' in text_lower or 'single hop' in text_lower: return 'S1'
    if 'multi-hop' in text_lower or 'multi hop' in text_lower: return 'M1'

    return "UNKNOWN"

def load_model_and_tokenizer():
    """加载模型和 Tokenizer 的统一入口"""
    model_path, is_merged = get_best_available_model()
    
    # 清理显存
    gc.collect()
    torch.cuda.empty_cache()

    if is_merged:
        print("🚀 使用 Transformers 加载合并后的全量模型...")
        # 优先从合并目录加载 Tokenizer
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        except:
            print(f"⚠️ 无法从 {model_path} 加载 Tokenizer，回退到基础模型 {BASE_MODEL_NAME}")
            tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME, trust_remote_code=True)

        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=DTYPE,
            device_map="cuda",
            trust_remote_code=True
        )
    else:
        print("🚀 使用 Unsloth 加载 LoRA 模型...")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name = model_path, # 这里传入 LoRA 路径
            max_seq_length = MAX_SEQ_LENGTH,
            dtype = DTYPE,
            load_in_4bit = LOAD_IN_4BIT,
        )
        FastLanguageModel.for_inference(model)

    # === 关键修复: 设置 Tokenizer 属性 ===
    # 1. 设置 Padding Side 为 Left (生成任务必须)
    tokenizer.padding_side = "left"
    
    # 2. 修复 Pad Token (Qwen 默认没有 pad_token)
    if tokenizer.pad_token is None:
        if tokenizer.eos_token:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        else:
            # 如果连 eos 都没有（极少见），手动添加
            tokenizer.add_special_tokens({'pad_token': '<|pad|>'})
            model.resize_token_embeddings(len(tokenizer))

    print(f"✓ 模型加载完成 | Pad Token ID: {tokenizer.pad_token_id} | Padding Side: {tokenizer.padding_side}")
    return model, tokenizer, model_path

def evaluate():
    print("=" * 60)
    print("模型评估脚本 (v4 - Fixed & Optimized)")
    print("=" * 60)

    # 1. 加载资源
    model, tokenizer, current_model_path = load_model_and_tokenizer()
    
    # 2. 加载数据
    print(f"正在从 {DATA_DIR} 加载测试集...")
    dataset = load_from_disk(DATA_DIR)
    test_data = dataset["test"]
    print(f"✓ 测试集大小: {len(test_data)}")

    # 3. 准备统计
    correct = 0
    total = 0
    predictions_log = []
    unknown_count = 0

    print("\n开始推理评估...")
    # 使用 tqdm 显示实时准确率
    pbar = tqdm(test_data, desc="Evaluating")
    
    for item in pbar:
        # 提取输入和标签
        input_msgs = item["messages"][:-1]
        ground_truth_msg = item["messages"][-1]["content"]
        gt_label = extract_label(ground_truth_msg)

        # 构建 Prompt
        # apply_chat_template 返回的是 list[int]
        inputs = tokenizer.apply_chat_template(
            input_msgs,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True
        )

        # 移至 GPU
        input_ids = inputs["input_ids"].to("cuda")
        attention_mask = inputs["attention_mask"].to("cuda")

        # 生成
        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=512,  # 足够容纳分析过程+标签
                temperature=0.01,    # 近似贪婪搜索，比 0.0 更兼容某些库
                top_p=0.9,
                do_sample=False,     # 关闭采样，结果确定性
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.05 # 轻微惩罚重复
            )

        # 解码 (只保留新生成的部分)
        generated_ids = outputs[0][input_ids.shape[-1]:]
        prediction_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

        # 提取标签
        pred_label = extract_label(prediction_text)

        # 统计
        if pred_label == "UNKNOWN":
            unknown_count += 1
        
        is_correct = (gt_label == pred_label)
        if is_correct:
            correct += 1
        total += 1

        # 更新进度条后缀信息
        current_acc = correct / total
        pbar.set_postfix({"Acc": f"{current_acc:.2%}", "UNK": unknown_count})

        # 记录详细日志
        predictions_log.append({
            "gt": gt_label,
            "pred": pred_label,
            "correct": is_correct,
            "output": prediction_text,
            "gt_text": ground_truth_msg
        })

    # 4. 最终统计
    accuracy = correct / total if total > 0 else 0
    unknown_rate = unknown_count / total if total > 0 else 0

    print("\n" + "=" * 60)
    print(f"最终评估结果 (Model: {os.path.basename(current_model_path)})")
    print("=" * 60)
    print(f"Samples:    {total}")
    print(f"Accuracy:   {accuracy:.2%} ({correct}/{total})")
    print(f"UNKNOWN:    {unknown_rate:.2%} ({unknown_count})")
    print("=" * 60)

    # 5. 错误分析与日志保存
    save_log_analysis(current_model_path, predictions_log, accuracy, unknown_count)

def save_log_analysis(model_path, logs, accuracy, unknown_count):
    """保存详细的评估报告"""
    log_dir = os.path.dirname(model_path) if os.path.isfile(model_path) else model_path
    # 如果是 LoRA，保存到上一级或同级 output 目录
    if "checkpoint" in log_dir or "lora" in log_dir:
         # 尝试保存到 output 根目录，方便查找
         log_dir = os.path.dirname(log_dir)
         
    log_file = os.path.join(log_dir, "eval_result_final.txt")
    
    print(f"正在生成详细报告: {log_file}")
    
    # 计算混淆矩阵
    from collections import defaultdict
    confusion = defaultdict(lambda: defaultdict(int))
    label_stats = defaultdict(lambda: {"total": 0, "correct": 0})
    
    for log in logs:
        gt = log['gt']
        pred = log['pred']
        confusion[gt][pred] += 1
        label_stats[gt]["total"] += 1
        if log['correct']:
            label_stats[gt]["correct"] += 1

    with open(log_file, 'w', encoding='utf-8') as f:
        f.write(f"=== Evaluation Report ===\n")
        f.write(f"Model: {model_path}\n")
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write(f"Unknown Count: {unknown_count}\n\n")

        f.write("=== Per-Class Accuracy ===\n")
        for label in sorted(label_stats.keys()):
            stats = label_stats[label]
            acc = stats["correct"] / stats["total"] if stats["total"] > 0 else 0
            f.write(f"{label}: {acc:.2%} ({stats['correct']}/{stats['total']})\n")
        f.write("\n")

        f.write("=== Confusion Matrix (Errors Only) ===\n")
        error_pairs = []
        for gt in confusion:
            for pred in confusion[gt]:
                if gt != pred:
                    error_pairs.append((gt, pred, confusion[gt][pred]))
        
        error_pairs.sort(key=lambda x: x[2], reverse=True)
        for gt, pred, count in error_pairs:
            f.write(f"GT [{gt}] -> Pred [{pred}]: {count}\n")
        f.write("\n")

        f.write("=== Error Cases (Sample 50) ===\n")
        errors = [l for l in logs if not l['correct']]
        for i, log in enumerate(errors[:50]):
            f.write(f"[{i+1}] GT: {log['gt']} | Pred: {log['pred']}\n")
            f.write(f"Model Output: {log['output'][:300]}...\n") # 只截取前300字符
            f.write("-" * 50 + "\n")
            
    print("完成！")

if __name__ == "__main__":
    evaluate()