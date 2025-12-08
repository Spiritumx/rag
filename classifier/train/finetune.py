import torch
import os
import sys
import json
from datasets import load_dataset
# 关键修改：引入 SFTConfig
from unsloth import FastLanguageModel
from trl import SFTTrainer, SFTConfig

# --- 1. 路径配置 ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.abspath(os.path.join(CURRENT_DIR, "../data/training_data/final_finetuning_dataset.jsonl"))

# 模型路径
LOCAL_MODEL_PATH = "/root/autodl-tmp/model/Qwen2.5-3B-Instruct" 
MODEL_NAME = LOCAL_MODEL_PATH if os.path.exists(LOCAL_MODEL_PATH) else "Qwen/Qwen2.5-3B-Instruct"

# 输出路径
OUTPUT_DIR = "/root/autodl-tmp/output/Qwen2.5-3B-RAG-Router"

# --- 2. 训练超参数 ---
MAX_SEQ_LENGTH = 2048
DTYPE = None 
LOAD_IN_4BIT = False 

# --- 3. System Prompt ---
SYSTEM_PROMPT = """You are an expert RAG router. Analyze the user query complexity and determine the optimal retrieval strategy.
Output the analysis in the following format:
Analysis: <reasoning process>
Complexity: <L0/L1/L2>
Index: <None/Lexical/Semantic/Hybrid>
Action: <Z/S-Sparse/S-Dense/S-Hybrid/M>"""

def formatting_prompts_func(examples, tokenizer):
    """
    将 JSON 数据格式化为 Qwen 的 Chat 模板
    """
    convos = []
    texts = []
    
    for i in range(len(examples["question_text"])):
        question = examples["question_text"][i]
        
        # 构建 Assistant 的思维链回答
        assistant_content = (
            f"Analysis: {examples['reasoning'][i]}\n"
            f"Complexity: {examples['complexity_label'][i]}\n"
            f"Index: {examples['index_strategy'][i]}\n"
            f"Action: {examples['action'][i]}"
        )

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
            {"role": "assistant", "content": assistant_content}
        ]
        
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        texts.append(text)
        
    return { "text": texts }

def main():
    # --- 路径检查逻辑 ---
    if os.path.exists(LOCAL_MODEL_PATH):
        MODEL_NAME = LOCAL_MODEL_PATH
        print(f"✅ 检测到本地模型，路径: {MODEL_NAME}")
        extra_kwargs = {"local_files_only": True}
    else:
        MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
        print(f"⚠️ 未找到本地模型 {LOCAL_MODEL_PATH}，尝试从 HuggingFace 下载...")
        extra_kwargs = {}

    # 1. 加载模型
    print(f"🚀 Loading model from {MODEL_NAME}...")
    try:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name = MODEL_NAME,
            max_seq_length = MAX_SEQ_LENGTH,
            dtype = DTYPE,
            load_in_4bit = LOAD_IN_4BIT,
            **extra_kwargs 
        )
    except Exception as e:
        print("\n❌ 模型加载失败！请检查路径。")
        raise e

    # 2. 配置 LoRA
    model = FastLanguageModel.get_peft_model(
        model,
        r = 16,
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj"],
        lora_alpha = 16,
        lora_dropout = 0, # 设置为 0 以启用 Unsloth 加速
        bias = "none",
        use_gradient_checkpointing = "unsloth", 
        random_state = 3407,
    )

    # 3. 加载数据集
    print(f"📂 Loading dataset from {DATA_FILE}...")
    if not os.path.exists(DATA_FILE):
        raise FileNotFoundError(f"找不到数据文件: {DATA_FILE}")

    full_dataset = load_dataset("json", data_files=DATA_FILE, split="train")
    
    print("✂️ Splitting dataset...")
    dataset_split = full_dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = dataset_split["train"]
    test_dataset = dataset_split["test"]
    
    print(f"   Train samples: {len(train_dataset)}")
    print(f"   Test samples:  {len(test_dataset)}")

    # 4. 数据格式化
    print("🔄 Formatting prompts...")
    format_func = lambda x: formatting_prompts_func(x, tokenizer)
    train_dataset = train_dataset.map(format_func, batched=True)
    test_dataset = test_dataset.map(format_func, batched=True)

    # 5. 训练参数配置 (使用 SFTConfig 替代 TrainingArguments)
    # 修复点：TRL v0.12.0+ 要求所有 SFT 参数放入 SFTConfig
    training_args = SFTConfig(
        output_dir = OUTPUT_DIR,
        dataset_text_field = "text",       # 明确指定文本列名
        max_length = MAX_SEQ_LENGTH,   # 移入 Config
        dataset_num_proc = 4,              # 移入 Config
        packing = False,                   # 移入 Config
        
        per_device_train_batch_size = 4,
        gradient_accumulation_steps = 8,
        warmup_steps = 20,
        num_train_epochs = 2,
        learning_rate = 1e-4,
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(),
        logging_steps = 10,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "cosine",
        seed = 3407,
        
        save_strategy = "steps", 
        eval_strategy = "steps",           # 修复点：evaluation_strategy -> eval_strategy
        eval_steps = 50,
        save_steps = 50,
        load_best_model_at_end = True,
        metric_for_best_model = "loss",
        save_total_limit = 2,
        report_to = "tensorboard",
    )

    # 6. Trainer 初始化
    trainer = SFTTrainer(
        model = model,
        processing_class = tokenizer,      # 修复点：tokenizer -> processing_class
        train_dataset = train_dataset,
        eval_dataset = test_dataset,
        args = training_args,              # 所有的 max_seq_length 等参数都在这里面了
    )

    # 7. 开始训练
    print("\n🔥 Starting training...")
    trainer_stats = trainer.train()
    print("✅ Training completed!")

    # 8. 保存模型
    print(f"💾 Saving to {OUTPUT_DIR}/lora_model...")
    model.save_pretrained(os.path.join(OUTPUT_DIR, "lora_model"))
    tokenizer.save_pretrained(os.path.join(OUTPUT_DIR, "lora_model"))
    
    tokenizer.save_vocabulary(os.path.join(OUTPUT_DIR, "lora_model"))

if __name__ == "__main__":
    main()