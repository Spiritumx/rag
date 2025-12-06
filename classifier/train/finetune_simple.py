import torch
import os
import sys

# 设置环境变量（必须在导入 unsloth 之前）
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["UNSLOTH_RETURN_LOGITS"] = "1"

from unsloth import FastLanguageModel
from datasets import load_from_disk
from trl import SFTTrainer
from transformers import TrainingArguments, AutoTokenizer
from collections import Counter

# === Monkey Patch: 强制启用 logits 返回 ===
try:
    import unsloth.models._utils as unsloth_utils
    flag_names = ['UNSLOTH_RETURN_LOGITS', 'return_logits', '_return_logits']
    for flag_name in flag_names:
        if hasattr(unsloth_utils, flag_name):
            setattr(unsloth_utils, flag_name, True)
            print(f"✓ Monkey patch: {flag_name} = True")
except Exception as e:
    print(f"⚠ Monkey patch warning: {e}")
# === End of Monkey Patch ===

# --- 配置 ---
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
LOCAL_MODEL_PATH = "/root/autodl-tmp/model/Qwen2.5-3B-Instruct"
MODEL_NAME = LOCAL_MODEL_PATH if os.path.exists(LOCAL_MODEL_PATH) else "Qwen/Qwen2.5-3B-Instruct"

DATA_DIR = "/root/autodl-tmp/data"
OUTPUT_DIR = "/root/autodl-tmp/output"

MAX_SEQ_LENGTH = 2048
DTYPE = torch.bfloat16
LOAD_IN_4BIT = False  # 4080 SUPER 显存足够，不需要量化

def main():
    print(f"Loading model from {MODEL_NAME}...")

    # 1. 加载基础模型
    model, _ = FastLanguageModel.from_pretrained(
        model_name = MODEL_NAME,
        max_seq_length = MAX_SEQ_LENGTH,
        dtype = DTYPE,
        load_in_4bit = LOAD_IN_4BIT,
        fix_tokenizer = False,
    )

    # 2. 手动加载 Tokenizer
    print("Loading tokenizer manually...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 3. 配置 LoRA
    model = FastLanguageModel.get_peft_model(
        model,
        r = 32,
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

    # 4. 加载数据
    print(f"Loading dataset from {DATA_DIR}...")
    if not os.path.exists(DATA_DIR):
        print("Dataset not found! Please run prepare_data.py first.")
        return

    dataset = load_from_disk(DATA_DIR)

    # 数据格式化函数
    def formatting_prompts_func(examples):
        convos = examples["messages"]
        texts = [tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=False) for convo in convos]
        return {"text": texts}

    dataset = dataset.map(formatting_prompts_func, batched=True)

    # 统计标签分布（用于参考）
    label_counts = Counter()
    for example in dataset["train"]:
        messages = example['messages']
        for msg in messages:
            if msg['role'] == 'assistant':
                import re
                match = re.search(r'\[\[(\w+)\]\]', msg['content'])
                if match:
                    label = match.group(1)
                    label_counts[label] += 1
                    break

    print(f"\n标签分布: {dict(label_counts)}")
    print(f"总样本数: {sum(label_counts.values())}\n")

    # 5. 训练参数
    training_args = TrainingArguments(
        output_dir = OUTPUT_DIR,
        per_device_train_batch_size = 8,
        gradient_accumulation_steps = 4,  # 有效 batch size = 32
        warmup_steps = 50,  # 增加 warmup
        num_train_epochs = 5,  # 增加到 5 个 epoch
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
        # eval_strategy = "epoch",  # 移除：某些版本不支持
    )

    # 6. 创建 Trainer（使用标准 SFTTrainer，不使用类别权重）
    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = dataset["train"],
        eval_dataset = dataset["test"],
        dataset_text_field = "text",
        packing = False,
        args = training_args,
    )

    print("Starting training...")
    print(f"训练轮数: {training_args.num_train_epochs}")
    print(f"Batch size: {training_args.per_device_train_batch_size}")
    print(f"梯度累积: {training_args.gradient_accumulation_steps}")
    print(f"有效 batch size: {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}\n")

    trainer_stats = trainer.train()
    print("Training completed!")

    # 7. 保存 LoRA 适配器
    print(f"Saving LoRA adapters to {OUTPUT_DIR}/lora_model...")
    model.save_pretrained(os.path.join(OUTPUT_DIR, "lora_model"))
    tokenizer.save_pretrained(os.path.join(OUTPUT_DIR, "lora_model"))

    print("\n训练完成！下一步：")
    print("1. 运行评估: python classifier/train/evaluate.py")
    print("2. 如果需要更好的推理性能，可以合并模型:")
    print("   python classifier/train/merge_model.py")

if __name__ == "__main__":
    main()
