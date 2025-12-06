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
    强化版标签提取逻辑（v3 - 防止 UNKNOWN）：
    1. 优先匹配明确的 Key-Value 格式 (例如 "Complexity: M1")
    2. 其次匹配双括号 [[M1]]
    3. 匹配所有可能的标签格式（大小写不敏感）
    4. 提取文本中出现的最后一个合法标签
    5. 如果仍然失败，提取第一个出现的标签
    6. 最终兜底：分析文本内容关键词推断标签
    """
    if not text:
        return "UNKNOWN"

    text = text.strip()
    original_text = text  # 保留原始文本用于关键词分析
    text_upper = text.upper()  # 统一转大写进行匹配

    # 策略 1: 寻找明确的分类标识词 (Case Insensitive)
    # 匹配: "Classification: M1", "Label: S2", "Complexity: Z0", "答案: M1", "分类: S2"
    match = re.search(
        r'(?:Classification|Label|Complexity|Category|Type|Answer|答案|分类|复杂度|类别)\s*[:：]?\s*([ZSM]\d+)',
        text,
        re.IGNORECASE
    )
    if match:
        return match.group(1).upper()

    # 策略 2: 匹配各种括号格式 [[M1]], [M1], (M1), 【M1】
    match = re.search(r'[\[\(【]([ZSM]\d+)[\]\)】]', text_upper)
    if match:
        return match.group(1)

    # 策略 3: 匹配引号格式 "M1", 'M1'
    match = re.search(r'["\']([ZSM]\d+)["\']', text_upper)
    if match:
        return match.group(1)

    # 策略 4: 寻找文本中出现的 *最后一个* 合法标签
    # 这能有效解决 CoT 分析中提到其他标签的问题
    # 例如: "Unlike M1, this is actually S2." -> 应该提取 S2
    all_matches = re.findall(r'\b([ZSM]\d)\b', text_upper)
    if all_matches:
        # 返回最后一个匹配的标签（最可能是最终答案）
        return all_matches[-1]

    # 策略 5: 尝试匹配更宽松的模式（允许标签前后有其他字符）
    # 例如: "...therefore: M1." 或 "answer is M1!"
    match = re.search(r'[^A-Z]([ZSM]\d)[^A-Z0-9]', text_upper)
    if match:
        return match.group(1)

    # 策略 6: 如果上述都失败，尝试从文本内容推断
    # 关键词匹配策略
    text_lower = original_text.lower()

    # Z0 (无检索) 关键词
    if any(kw in text_lower for kw in ['no retrieval', 'zero retrieval', '无需检索', '不需要检索', 'z0', 'z-0']):
        return 'Z0'

    # S (单跳) 关键词
    if any(kw in text_lower for kw in ['single', 'one-hop', '单跳', 's1', 's2', 's3', 's4']):
        # 尝试进一步细分
        if any(kw in text_lower for kw in ['s2', 'indirect', '间接']):
            return 'S2'
        if any(kw in text_lower for kw in ['s1', 'direct', '直接']):
            return 'S1'
        if any(kw in text_lower for kw in ['s3', 'comparison', '比较']):
            return 'S3'
        if any(kw in text_lower for kw in ['s4', 'aggregation', '聚合']):
            return 'S4'
        return 'S1'  # 默认单跳

    # M (多跳) 关键词
    if any(kw in text_lower for kw in ['multi', 'multiple', 'two-hop', 'three-hop', '多跳', 'm1', 'm2', 'm4']):
        # 尝试进一步细分
        if any(kw in text_lower for kw in ['m2', 'implicit', '隐式']):
            return 'M2'
        if any(kw in text_lower for kw in ['m4', 'comparison', '比较']):
            return 'M4'
        return 'M1'  # 默认多跳

    # 最终仍无法识别，返回 UNKNOWN
    print(f"⚠️  无法提取标签，原始文本: {original_text[:100]}...")
    return "UNKNOWN"

def evaluate():
    print("=" * 60)
    print("模型评估脚本 (Enhanced v3 - 防止 UNKNOWN)")
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
    unknown_count = 0  # 统计 UNKNOWN 数量

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

        # 统计 UNKNOWN
        if pred_label == "UNKNOWN":
            unknown_count += 1

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
    unknown_rate = unknown_count / total if total > 0 else 0

    print("\n" + "=" * 60)
    print("评估结果")
    print("=" * 60)
    print(f"总样本数:      {total}")
    print(f"正确预测:      {correct}")
    print(f"错误预测:      {total - correct}")
    print(f"准确率:        {accuracy:.2%} ({accuracy:.4f})")
    print(f"UNKNOWN 数量:  {unknown_count} ({unknown_rate:.2%})")
    print("=" * 60)

    # 5. 构建混淆矩阵分析
    from collections import defaultdict
    confusion = defaultdict(lambda: defaultdict(int))
    for log in predictions_log:
        confusion[log['gt_label']][log['pred_label']] += 1

    # 打印混淆矩阵（仅显示错误分类）
    print("\n混淆矩阵 (主要错误方向):")
    print("-" * 60)
    error_pairs = []
    for gt in confusion:
        for pred in confusion[gt]:
            if gt != pred and confusion[gt][pred] > 0:
                error_pairs.append((gt, pred, confusion[gt][pred]))

    # 按错误次数排序
    error_pairs.sort(key=lambda x: x[2], reverse=True)
    for gt, pred, count in error_pairs[:15]:  # 只显示前15个
        print(f"真实是 [{gt}] 但预测成了 -> [{pred}]: {count} 次")

    # 6. 保存详细日志 (用于 Debug)
    log_file = os.path.join(os.path.dirname(model_path), "evaluation_log_v3.txt")
    with open(log_file, 'w', encoding='utf-8') as f:
        f.write(f"Model: {model_path}\n")
        f.write(f"Accuracy: {accuracy:.2%}\n")
        f.write(f"UNKNOWN Count: {unknown_count} ({unknown_rate:.2%})\n\n")

        # 混淆矩阵
        f.write("=== 混淆矩阵 (所有错误方向) ===\n")
        for gt, pred, count in error_pairs:
            f.write(f"{gt} -> {pred}: {count} 次\n")
        f.write("\n")

        # UNKNOWN 案例（重点关注）
        f.write("=== UNKNOWN 预测案例 (需要重点关注) ===\n")
        unknown_cases = [log for log in predictions_log if log['pred_label'] == 'UNKNOWN']
        for i, log in enumerate(unknown_cases[:20]):  # 只显示前20个
            f.write(f"\n[案例 {i+1}] GT: {log['gt_label']}\n")
            f.write(f"模型输出: {log['prediction'][:300]}\n")
            f.write("-" * 50 + "\n")

        # 先把错误的写在前面，方便查看
        f.write("\n=== 其他错误案例 (非 UNKNOWN) ===\n")
        other_errors = [log for log in predictions_log if not log["is_correct"] and log['pred_label'] != 'UNKNOWN']
        for log in other_errors:
            f.write(f"GT: {log['gt_label']} | Pred: {log['pred_label']}\n")
            f.write(f"Raw Output: {log['prediction'][:200]}...\n")
            f.write("-" * 30 + "\n")

        f.write("\n=== 正确案例 (抽样) ===\n")
        correct_cases = [log for log in predictions_log if log["is_correct"]]
        for log in correct_cases[:10]:  # 只抽样显示10个
            f.write(f"GT: {log['gt_label']} | Pred: {log['pred_label']}\n")

    print(f"\n详细日志已保存到: {log_file}")

if __name__ == "__main__":
    evaluate()