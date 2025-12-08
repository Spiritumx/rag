import torch
import os
import sys
import re
import random
from collections import Counter
from tqdm import tqdm

# 设置环境变量（必须在导入 unsloth 之前）
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from unsloth import FastLanguageModel
from datasets import load_from_disk
from trl import SFTTrainer
from transformers import TrainingArguments

# --- 配置 ---
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
# 你的模型路径
LOCAL_MODEL_PATH = "/root/autodl-tmp/model/Qwen2.5-3B-Instruct" 
MODEL_NAME = LOCAL_MODEL_PATH if os.path.exists(LOCAL_MODEL_PATH) else "Qwen/Qwen2.5-3B-Instruct"

DATA_DIR = "/root/autodl-tmp/data"
OUTPUT_DIR = "/root/autodl-tmp/output"
MAX_SEQ_LENGTH = 2048
DTYPE = torch.bfloat16  # A100/3090/4090 用 bfloat16，T4/V100 用 float16
LOAD_IN_4BIT = False    # 如果显存不够（<16GB），请设为 True

# 标签映射
LABEL_TO_ID = {
    "Z0": 0, "S1": 1, "S2": 2, "S3": 3,
    "S4": 4, "M1": 5, "M2": 6, "M4": 7
}

def extract_label_from_example(example):
    """从样本中提取标签，支持 [[S1]] 或 [[ S1 ]]"""
    messages = example['messages']
    for msg in messages:
        if msg['role'] == 'assistant':
            content = msg['content']
            # 优化正则：允许标签周围有空格
            match = re.search(r'\[\[\s*(\w+)\s*\]\]', content)
            if match:
                label = match.group(1)
                if label in LABEL_TO_ID:
                    return label
    return None

def oversample_dataset(dataset, target_samples_per_class=None, random_seed=42):
    """对数据集进行过采样平衡"""
    print("=" * 60)
    print("正在进行数据重采样 (Oversampling)...")
    
    # 1. 统计类别
    label_to_indices = {label: [] for label in LABEL_TO_ID.keys()}
    # 记录没找到标签的样本（可选，防止丢数据）
    unknown_indices = [] 

    for idx, example in enumerate(tqdm(dataset, desc="分析类别分布")):
        label = extract_label_from_example(example)
        if label:
            label_to_indices[label].append(idx)
        else:
            unknown_indices.append(idx)

    # 2. 打印分布
    print("\n原始分布:")
    current_counts = []
    for label, indices in sorted(label_to_indices.items()):
        count = len(indices)
        current_counts.append(count)
        print(f"  {label}: {count}")
    
    if unknown_indices:
        print(f"  Unknown/No Label: {len(unknown_indices)}")

    # 3. 确定目标数量
    if target_samples_per_class is None:
        if current_counts:
            target_samples_per_class = max(current_counts)
        else:
            target_samples_per_class = 0
            
    print(f"\n目标每类样本数: {target_samples_per_class}")

    # 4. 执行重采样
    random.seed(random_seed)
    balanced_indices = []
    
    for label, indices in sorted(label_to_indices.items()):
        if not indices:
            continue
            
        if len(indices) < target_samples_per_class:
            # 重复采样
            sampled = random.choices(indices, k=target_samples_per_class)
        else:
            # 随机下采样（虽然通常是取最大值，不需要下采样，但保留逻辑以防万一）
            sampled = random.sample(indices, target_samples_per_class)
            
        balanced_indices.extend(sampled)
    
    # 可选：把未识别标签的数据也加回去（防止浪费），或者为了纯净度选择丢弃
    # balanced_indices.extend(unknown_indices)

    # 5. 打乱
    random.shuffle(balanced_indices)
    
    # 6. 生成新数据集
    balanced_dataset = dataset.select(balanced_indices)
    print(f"重采样后总样本数: {len(balanced_dataset)}")
    print("=" * 60)
    return balanced_dataset

def main():
    print(f"Loading model from {MODEL_NAME}...")
    
    # 1. 加载模型和 Tokenizer (直接使用 Unsloth 返回的 tokenizer)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = MODEL_NAME,
        max_seq_length = MAX_SEQ_LENGTH,
        dtype = DTYPE,
        load_in_4bit = LOAD_IN_4BIT,
    )

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
    )

    # 3. 加载并处理数据
    print(f"Loading dataset from {DATA_DIR}...")
    dataset = load_from_disk(DATA_DIR)

    # --- 步骤 A: 数据平衡 ---
    train_dataset = oversample_dataset(dataset["train"])
    test_dataset = dataset["test"] # 测试集不需要平衡

    # --- 步骤 B: 格式化 (修复报错的关键步骤) ---
    print("正在格式化数据集 (Messages -> Text)...")
    
    def formatting_prompts_func(examples):
        convos = examples["messages"]
        texts = [tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=False) for convo in convos]
        return { "text": texts }

    # 使用 batched=True 加速处理
    train_dataset = train_dataset.map(formatting_prompts_func, batched=True)
    test_dataset = test_dataset.map(formatting_prompts_func, batched=True)

    # 4. 训练参数
    # 显存优化建议：Batch Size 32 容易 OOM，建议改为 8，累积步数改为 4
    # 有效 Batch Size = 8 * 4 = 32，效果一样，但显存占用只有 1/4
    training_args = TrainingArguments(
        output_dir = OUTPUT_DIR,
        per_device_train_batch_size = 8,   # 改小防止 OOM
        gradient_accumulation_steps = 4,   # 改大保持总 Batch Size
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
        report_to = "tensorboard", # 或者 "none"
    )

    # 5. Trainer
    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = train_dataset,
        eval_dataset = test_dataset,
        dataset_text_field = "text", # 指定处理好的文本列
        max_seq_length = MAX_SEQ_LENGTH,
        dataset_num_proc = 2,
        packing = False, # 数据平衡模式下建议关闭 packing
        args = training_args,
    )

    print("\nStarting training...")
    trainer_stats = trainer.train()
    print("Training completed!")

    # 6. 保存
    print(f"Saving to {OUTPUT_DIR}/lora_model...")
    model.save_pretrained(os.path.join(OUTPUT_DIR, "lora_model"))
    tokenizer.save_pretrained(os.path.join(OUTPUT_DIR, "lora_model"))

if __name__ == "__main__":
    main()