import torch
import os
import re
from unsloth import FastLanguageModel
from datasets import load_from_disk
from tqdm import tqdm
from transformers import AutoTokenizer

# --- 配置 ---
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
LOCAL_MODEL_PATH = "/root/autodl-tmp/model/Qwen2.5-3B-Instruct"
BASE_MODEL_NAME = LOCAL_MODEL_PATH if os.path.exists(LOCAL_MODEL_PATH) else "Qwen/Qwen2.5-3B-Instruct"

LORA_PATH = "/root/autodl-tmp/output/lora_model"  # LoRA 适配器路径
MERGED_MODEL_PATH = "/root/autodl-tmp/output/merged_model"  # 合并后的模型路径
DATA_DIR = "/root/autodl-tmp/data"

MAX_SEQ_LENGTH = 2048
DTYPE = torch.bfloat16
LOAD_IN_4BIT = False

def get_model_path():
    """
    优先使用合并后的模型（如果存在），否则使用 LoRA 路径
    """
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
    从文本中提取标签，支持多种格式:
    - [[S3]]
    - S3
    - "strategy is S3"
    """
    # 优先匹配 [[label]] 格式
    match = re.search(r'\[\[([ZSM]\d+)\]\]', text)
    if match:
        return match.group(1)

    # 匹配独立的标签格式 (例如 "S3" 或 "M1")
    match = re.search(r'\b([ZSM]\d+)\b', text)
    if match:
        return match.group(1)

    # 如果都没匹配到，返回原文本的前10个字符用于调试
    return text.strip()[:10]

def evaluate():
    print("=" * 60)
    print("模型评估脚本")
    print("=" * 60)

    # 1. 加载模型
    model_path, is_merged = get_model_path()
    print(f"\n正在加载模型: {model_path}")
    print(f"模型类型: {'合并模型' if is_merged else 'LoRA 适配器'}")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = model_path,
        max_seq_length = MAX_SEQ_LENGTH,
        dtype = DTYPE,
        load_in_4bit = LOAD_IN_4BIT,
    )
    FastLanguageModel.for_inference(model)
    print("✓ 模型加载完成\n")

    # 2. 加载测试集
    print(f"正在加载测试集: {DATA_DIR}")
    if not os.path.exists(DATA_DIR):
        print(f"错误: 数据目录不存在: {DATA_DIR}")
        return

    dataset = load_from_disk(DATA_DIR)
    if "test" not in dataset:
        print("错误: 数据集中没有 test split！")
        return

    test_data = dataset["test"]
    print(f"✓ 测试集加载完成，共 {len(test_data)} 个样本\n")
    print("=" * 60)

    correct = 0
    total = 0
    predictions_log = []

    # 3. 批量推理
    print("开始评估...\n")
    for item in tqdm(test_data, desc="评估进度"):
        # messages 格式: [system, user, assistant]
        # 我们需要去掉最后的 assistant 消息（即正确答案），只保留 system + user
        input_msgs = item["messages"][:-1]  # 去掉最后的正确答案
        ground_truth_msg = item["messages"][-1]["content"]  # 获取正确答案

        # 从正确答案中提取标签
        gt_label = extract_label(ground_truth_msg)

        # 构造 Prompt
        inputs = tokenizer.apply_chat_template(
            input_msgs,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to("cuda")

        # 生成
        with torch.no_grad():
            outputs = model.generate(
                input_ids=inputs,
                max_new_tokens=128,  # 增加一些长度以容纳完整的思维链输出
                temperature=0.1,  # 低温度确保稳定性
                do_sample=False,  # 使用贪婪解码
                pad_token_id=tokenizer.eos_token_id
            )

        # 解码并提取由模型生成的部分
        generated_ids = outputs[0][inputs.shape[-1]:]
        prediction = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

        # 从预测中提取标签
        pred_label = extract_label(prediction)

        # 比对标签（严格匹配）
        is_correct = (gt_label == pred_label)
        if is_correct:
            correct += 1

        total += 1

        # 记录详细信息用于调试
        predictions_log.append({
            "gt_label": gt_label,
            "pred_label": pred_label,
            "is_correct": is_correct,
            "prediction": prediction[:150]  # 只保留前150个字符
        })

        # 打印前5个样本看看效果
        if total <= 5:
            print(f"\n{'=' * 60}")
            print(f"[样本 {total}]")
            print(f"  真实标签:   {gt_label}")
            print(f"  预测标签:   {pred_label}")
            print(f"  匹配结果:   {'✓ 正确' if is_correct else '✗ 错误'}")
            print(f"  完整预测:   {prediction[:100]}...")
            print('=' * 60)

    # 4. 计算并打印结果
    accuracy = correct / total if total > 0 else 0

    print("\n" + "=" * 60)
    print("评估完成！")
    print("=" * 60)
    print(f"总样本数:    {total}")
    print(f"正确预测:    {correct}")
    print(f"错误预测:    {total - correct}")
    print(f"准确率:      {accuracy:.2%} ({accuracy:.4f})")
    print("=" * 60)

    # 5. 分析错误案例
    error_cases = [log for log in predictions_log if not log["is_correct"]]
    if error_cases:
        print(f"\n错误案例分析 (前10个):")
        print("-" * 60)
        for i, case in enumerate(error_cases[:10], 1):
            print(f"{i}. GT: {case['gt_label']} | Pred: {case['pred_label']}")
            print(f"   预测文本: {case['prediction'][:80]}...")
            print()

    # 6. 保存详细日志
    log_file = os.path.join(os.path.dirname(model_path), "evaluation_log.txt")
    with open(log_file, 'w', encoding='utf-8') as f:
        f.write(f"Evaluation Results\n")
        f.write(f"=" * 60 + "\n")
        f.write(f"Model: {model_path}\n")
        f.write(f"Accuracy: {accuracy:.2%}\n")
        f.write(f"Correct: {correct}/{total}\n\n")
        f.write(f"Detailed Predictions:\n")
        f.write("-" * 60 + "\n")
        for i, log in enumerate(predictions_log, 1):
            status = "✓" if log["is_correct"] else "✗"
            f.write(f"{i}. {status} GT: {log['gt_label']} | Pred: {log['pred_label']}\n")
            f.write(f"   {log['prediction'][:100]}...\n\n")

    print(f"\n详细日志已保存到: {log_file}")

if __name__ == "__main__":
    evaluate()
