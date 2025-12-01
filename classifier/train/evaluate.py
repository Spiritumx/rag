import os
# 设置 HF 镜像源
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
# 禁用 Unsloth 统计检查
os.environ["UNSLOTH_DISABLE_STATISTICS"] = "1"

import torch
import re
from unsloth import FastLanguageModel
from datasets import load_from_disk
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

# --- 配置 ---
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
LOCAL_MODEL_PATH = "/root/autodl-tmp/model/Qwen2.5-3B-Instruct"
BASE_MODEL_NAME = LOCAL_MODEL_PATH if os.path.exists(LOCAL_MODEL_PATH) else "Qwen/Qwen2.5-3B-Instruct"

LORA_PATH = "/root/autodl-tmp/output/lora_model"
MERGED_MODEL_PATH = "/root/autodl-tmp/output/merged_model"
DATA_DIR = "/root/autodl-tmp/data"

MAX_SEQ_LENGTH = 2048
DTYPE = torch.bfloat16
LOAD_IN_4BIT = False

def get_model_path():
    if os.path.exists(MERGED_MODEL_PATH):
        print(f"✓ 使用合并后的模型: {MERGED_MODEL_PATH}")
        return MERGED_MODEL_PATH, True
    elif os.path.exists(LORA_PATH):
        print(f"✓ 使用 LoRA 适配器: {LORA_PATH}")
        return LORA_PATH, False
    else:
        raise FileNotFoundError(f"未找到模型！请检查 {LORA_PATH} 或 {MERGED_MODEL_PATH}")

def extract_label(text):
    """
    强化版标签提取逻辑：
    1. 优先匹配明确的 Key-Value 格式 (例如 "Complexity: M1")
    2. 其次匹配双括号 [[M1]]
    3. 最后尝试在文本末尾寻找标签 (防止匹配到分析过程中的干扰项)
    """
    if not text:
        return "EMPTY"
        
    text = text.strip()
    
    # 策略 1: 寻找明确的分类标识词 (Case Insensitive)
    # 匹配: "Classification: M1", "Label: S2", "Complexity: Z0"
    match = re.search(r'(?:Classification|Label|Complexity|Category|Type)\s*[:：]?\s*([ZSM]\d+)', text, re.IGNORECASE)
    if match:
        return match.group(1).upper()

    # 策略 2: 匹配 [[M1]] 格式
    match = re.search(r'\[\[([ZSM]\d+)\]\]', text)
    if match:
        return match.group(1).upper()

    # 策略 3: 兜底策略 - 寻找文本中出现的 *最后一个* 合法标签
    # 使用 findall 找到所有标签，取最后一个。这能有效解决 CoT 分析中提到其他标签的问题。
    # 例如: "Unlike M1, this is actually S2." -> 应该提取 S2
    all_matches = re.findall(r'\b([ZSM]\d+)\b', text)
    if all_matches:
        return all_matches[-1].upper()

    return "UNKNOWN"

def evaluate():
    print("=" * 60)
    print("模型评估脚本 (Optimized Version)")
    print("=" * 60)

    # 1. 加载模型
    model_path, is_merged = get_model_path()
    
    if is_merged:
        print("使用 transformers 直接加载合并后的模型...")
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
        # 设置 pad_token，避免和 eos_token 相同
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=DTYPE,
            device_map="cuda"
        )
        model.eval()
    else:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name = model_path,
            max_seq_length = MAX_SEQ_LENGTH,
            dtype = DTYPE,
            load_in_4bit = LOAD_IN_4BIT,
        )
        # 确保 LoRA 模型也设置 pad_token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        FastLanguageModel.for_inference(model)

    print("✓ 模型加载完成\n")

    # 2. 加载测试集
    dataset = load_from_disk(DATA_DIR)
    test_data = dataset["test"]
    print(f"✓ 测试集加载完成，共 {len(test_data)} 个样本\n")

    correct = 0
    total = 0
    predictions_log = []

    # 3. 批量推理
    print("开始评估...\n")
    for item in tqdm(test_data, desc="评估进度"):
        input_msgs = item["messages"][:-1]
        ground_truth_msg = item["messages"][-1]["content"]

        # 处理 GT: 有些 GT 可能也很长，同样用提取函数处理
        gt_label = extract_label(ground_truth_msg)

        # 获取完整的 tokenizer 输出，包括 attention_mask
        inputs = tokenizer.apply_chat_template(
            input_msgs,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True  # 返回字典，包含 input_ids 和 attention_mask
        )

        # 将所有输入移到 GPU
        input_ids = inputs["input_ids"].to("cuda")
        attention_mask = inputs.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to("cuda")

        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,  # 传递 attention_mask
                # --- 关键修改：增加生成长度 ---
                max_new_tokens=512,  # 从 128 改为 512，防止分析被截断
                temperature=0.1,
                do_sample=False,     # 建议分类任务关闭采样，使用贪婪搜索确保确定性
                pad_token_id=tokenizer.eos_token_id
            )

        generated_ids = outputs[0][input_ids.shape[-1]:]
        prediction = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

        # 提取预测标签
        pred_label = extract_label(prediction)

        is_correct = (gt_label == pred_label)
        if is_correct:
            correct += 1
        total += 1

        predictions_log.append({
            "gt_label": gt_label,
            "pred_label": pred_label,
            "is_correct": is_correct,
            "prediction": prediction
        })

    # 4. 计算结果
    accuracy = correct / total if total > 0 else 0

    print("\n" + "=" * 60)
    print("评估结果")
    print("=" * 60)
    print(f"总样本数:    {total}")
    print(f"正确预测:    {correct}")
    print(f"错误预测:    {total - correct}")
    print(f"准确率:      {accuracy:.2%} ({accuracy:.4f})")
    print("=" * 60)

    # 5. 保存详细日志 (用于 Debug)
    log_file = os.path.join(os.path.dirname(model_path), "evaluation_log_v2.txt")
    with open(log_file, 'w', encoding='utf-8') as f:
        f.write(f"Model: {model_path}\nAccuracy: {accuracy:.2%}\n\n")
        
        # 先把错误的写在前面，方便查看
        f.write("=== 错误案例 (Mistakes) ===\n")
        for log in predictions_log:
            if not log["is_correct"]:
                f.write(f"GT: {log['gt_label']} | Pred: {log['pred_label']}\n")
                f.write(f"Raw Output: {log['prediction'][:200]}...\n") # 只记录前200字
                f.write("-" * 30 + "\n")
                
        f.write("\n=== 正确案例 (Correct) ===\n")
        for log in predictions_log:
            if log["is_correct"]:
                f.write(f"GT: {log['gt_label']} | Pred: {log['pred_label']}\n")

    print(f"详细日志已保存到: {log_file}")

if __name__ == "__main__":
    evaluate()