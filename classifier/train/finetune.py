import torch
import os
import sys
from unsloth import FastLanguageModel
from datasets import load_from_disk
from trl import SFTTrainer
from transformers import TrainingArguments, DataCollatorForSeq2Seq, AutoConfig

# --- 配置 ---
# 尝试从项目根目录加载本地模型，如果失败则使用 HF Hub
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
LOCAL_MODEL_PATH = "/root/autodl-tmp/model/Qwen2.5-3B-Instruct"
MODEL_NAME = LOCAL_MODEL_PATH if os.path.exists(LOCAL_MODEL_PATH) else "Qwen/Qwen2.5-3B-Instruct"

DATA_DIR = "/root/autodl-tmp/data"
OUTPUT_DIR = "/root/autodl-tmp/output"

MAX_SEQ_LENGTH = 2048 # Qwen2.5 支持更长，但分类任务通常不长，2048足够
DTYPE = torch.bfloat16 # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
LOAD_IN_4BIT = False # 使用 4bit 量化加载以节省显存

def main():
    print(f"Loading model from {MODEL_NAME}...")
    
    extra_kwargs = {}
    if os.path.exists(MODEL_NAME):
        extra_kwargs["local_files_only"] = True
        os.environ["HF_HUB_OFFLINE"] = "1"
        print("Using local model and offline mode.")
        # Manually load config to avoid unsloth/transformers bug with local_files_only dict handling
        # config = AutoConfig.from_pretrained(MODEL_NAME)
        # extra_kwargs["config"] = config
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = MODEL_NAME,
        max_seq_length = MAX_SEQ_LENGTH,
        dtype = DTYPE,
        load_in_4bit = LOAD_IN_4BIT,
        **extra_kwargs,
    )

    # 配置 LoRA
    model = FastLanguageModel.get_peft_model(
        model,
        r = 32, # LoRA rank
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj",],
        lora_alpha = 16,
        lora_dropout = 0, # Supports any, but = 0 is optimized
        bias = "none",    # Supports any, but = "none" is optimized
        use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
        random_state = 3407,
        use_rslora = False,  # Rank stabilized LoRA
        loftq_config = None, # And LoftQ
    )

    # 加载数据
    print(f"Loading dataset from {DATA_DIR}...")
    if not os.path.exists(DATA_DIR):
        print("Dataset not found! Please run prepare_data.py first.")
        return

    dataset = load_from_disk(DATA_DIR)
    
    # 数据格式化函数 (Chat Template)
    def formatting_prompts_func(examples):
        convos = examples["messages"]
        texts = [tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=False) for convo in convos]
        return { "text": texts }

    dataset = dataset.map(formatting_prompts_func, batched=True)

    # 训练参数
    training_args = TrainingArguments(
        output_dir = OUTPUT_DIR,
        per_device_train_batch_size = 32, # 根据显存调整，5090可以尝试 8 或 16
        gradient_accumulation_steps = 1,
        warmup_steps = 10,
        max_steps = 0, # Set to 0 to use num_train_epochs
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

    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = dataset["train"],
        eval_dataset = dataset["test"],
        dataset_text_field = "text",
        max_seq_length = MAX_SEQ_LENGTH,
        dataset_num_proc = 2,
        packing = True, # Can make training 5x faster for short sequences.
        args = training_args,
    )

    print("Starting training...")
    trainer_stats = trainer.train()
    print("Training completed!")

    # 保存 LoRA 适配器
    print(f"Saving LoRA adapters to {OUTPUT_DIR}/lora_model...")
    model.save_pretrained(os.path.join(OUTPUT_DIR, "lora_model"))
    tokenizer.save_pretrained(os.path.join(OUTPUT_DIR, "lora_model"))
    
    # (可选) 保存合并后的模型
    # model.save_pretrained_merged("model_merged", tokenizer, save_method = "merged_16bit",)

if __name__ == "__main__":
    main()

