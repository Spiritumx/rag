import torch
import os
import sys

# 设置环境变量（必须在导入 unsloth 之前）
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from unsloth import FastLanguageModel
from datasets import load_from_disk, Dataset
from trl import SFTTrainer
from transformers import TrainingArguments
from collections import Counter
import re
from tqdm import tqdm

# --- 配置 ---
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
LOCAL_MODEL_PATH = "/root/autodl-tmp/model/Qwen2.5-3B-Instruct"
MODEL_NAME = LOCAL_MODEL_PATH if os.path.exists(LOCAL_MODEL_PATH) else "Qwen/Qwen2.5-3B-Instruct"

DATA_DIR = "/root/autodl-tmp/data"
OUTPUT_DIR = "/root/autodl-tmp/output"
MAX_SEQ_LENGTH = 2048
DTYPE = torch.bfloat16
LOAD_IN_4BIT = False

# 标签映射
LABEL_TO_ID = {
    "Z0": 0,
    "S1": 1,
    "S2": 2,
    "S3": 3,
    "S4": 4,
    "M1": 5,
    "M2": 6,
    "M4": 7
}


def extract_label_from_example(example):
    """从样本中提取标签"""
    messages = example['messages']
    for msg in messages:
        if msg['role'] == 'assistant':
            content = msg['content']
            match = re.search(r'\[\[(\w+)\]\]', content)
            if match:
                label = match.group(1)
                if label in LABEL_TO_ID:
                    return label
    return None


def oversample_dataset(dataset, target_samples_per_class=None, random_seed=42):
    """
    对数据集进行过采样，平衡各类别样本数

    Args:
        dataset: 原始数据集
        target_samples_per_class: 每个类别的目标样本数，None 则使用最大类别数
        random_seed: 随机种子

    Returns:
        平衡后的数据集
    """
    print("=" * 60)
    print("数据重采样（Oversampling）")
    print("=" * 60)

    # 1. 统计每个类别的样本
    label_to_indices = {label: [] for label in LABEL_TO_ID.keys()}

    for idx, example in enumerate(tqdm(dataset, desc="分析数据集")):
        label = extract_label_from_example(example)
        if label:
            label_to_indices[label].append(idx)

    # 2. 显示原始分布
    print("\n原始类别分布:")
    for label, indices in sorted(label_to_indices.items()):
        print(f"  {label}: {len(indices)} 样本")

    # 3. 确定目标样本数
    if target_samples_per_class is None:
        target_samples_per_class = max(len(indices) for indices in label_to_indices.values())

    print(f"\n目标样本数/类别: {target_samples_per_class}")

    # 4. 对每个类别进行过采样
    import random
    random.seed(random_seed)

    balanced_indices = []
    for label, indices in sorted(label_to_indices.items()):
        if len(indices) == 0:
            print(f"⚠️  警告: {label} 类别没有样本")
            continue

        # 如果样本数不足，进行重复采样
        if len(indices) < target_samples_per_class:
            # 重复采样直到达到目标数量
            sampled = random.choices(indices, k=target_samples_per_class)
        else:
            # 如果样本数足够，随机采样
            sampled = random.sample(indices, target_samples_per_class)

        balanced_indices.extend(sampled)

    # 5. 打乱顺序
    random.shuffle(balanced_indices)

    # 6. 创建新数据集
    balanced_dataset = dataset.select(balanced_indices)

    print(f"\n平衡后的数据集大小: {len(balanced_dataset)}")
    print("=" * 60)

    return balanced_dataset


def main():
    print("=" * 60)
    print("微调训练 (v2 - 使用数据重采样)")
    print("=" * 60)

    # 1. 加载模型
    print(f"Loading model from {MODEL_NAME}...")
    if os.path.exists(MODEL_NAME):
        print("Using local model and offline mode.")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = MODEL_NAME,
        max_seq_length = MAX_SEQ_LENGTH,
        dtype = DTYPE,
        load_in_4bit = LOAD_IN_4BIT,
    )

    # 手动加载 tokenizer
    print("Loading tokenizer manually...")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # 2. 配置 LoRA
    model = FastLanguageModel.get_peft_model(
        model,
        r = 16,
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj"],
        lora_alpha = 16,
        lora_dropout = 0,
        bias = "none",
        use_gradient_checkpointing = "unsloth",
        random_state = 3407,
        use_rslora = False,
        loftq_config = None,
    )

    # 3. 加载数据集
    print(f"Loading dataset from {DATA_DIR}...")
    dataset = load_from_disk(DATA_DIR)

    # 4. 对训练集进行过采样
    train_dataset = oversample_dataset(
        dataset["train"],
        target_samples_per_class=None,  # 使用最大类别数
        random_seed=42
    )

    # 测试集不需要重采样
    test_dataset = dataset["test"]

    # 5. 训练参数
    training_args = TrainingArguments(
        output_dir = OUTPUT_DIR,
        per_device_train_batch_size = 32,
        gradient_accumulation_steps = 1,
        warmup_steps = 10,
        num_train_epochs = 3,
        learning_rate = 2e-4,
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(),
        logging_steps = 10,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        save_strategy = "epoch",
        save_total_limit = 2,
    )

    # 6. 创建 Trainer（不使用自定义 compute_loss）
    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = train_dataset,
        eval_dataset = test_dataset,
        dataset_text_field = "text",
        packing = False,  # 禁用 packing
        args = training_args,
    )

    print("\nStarting training...")
    trainer_stats = trainer.train()
    print("Training completed!")

    # 7. 保存 LoRA 适配器
    print(f"\nSaving LoRA adapters to {OUTPUT_DIR}/lora_model...")
    model.save_pretrained(os.path.join(OUTPUT_DIR, "lora_model"))
    tokenizer.save_pretrained(os.path.join(OUTPUT_DIR, "lora_model"))

    print("\n" + "=" * 60)
    print("训练完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
