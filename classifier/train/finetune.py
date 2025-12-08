import torch
import os
import sys
import json
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import FastLanguageModel

# --- 1. 路径配置 (根据你的描述修改) ---
# 获取当前脚本所在目录 (classifier/train)
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
# 数据文件路径 (classifier/data/training_data/final_finetuning_dataset.jsonl)
DATA_FILE = os.path.abspath(os.path.join(CURRENT_DIR, "../data/training_data/final_finetuning_dataset.jsonl"))

# 模型路径
LOCAL_MODEL_PATH = "/root/autodl-tmp/model/Qwen2.5-3B-Instruct" 
MODEL_NAME = LOCAL_MODEL_PATH if os.path.exists(LOCAL_MODEL_PATH) else "Qwen/Qwen2.5-3B-Instruct"

# 输出路径
OUTPUT_DIR = "/root/autodl-tmp/output/Qwen2.5-3B-RAG-Router"

# --- 2. 训练超参数 ---
MAX_SEQ_LENGTH = 2048
DTYPE = None # None = 自动检测 (float16 for T4, bfloat16 for Ampere+)
LOAD_IN_4BIT = False # 显存大于 16G 可设为 False，否则设为 True

# --- 3. System Prompt (精简版，用于微调) ---
# 我们不需要把生成数据时那么长的 prompt 放进去，只要通过微调让模型学会输出格式即可
SYSTEM_PROMPT = """You are an expert RAG router. Analyze the user query complexity and determine the optimal retrieval strategy.
Output the analysis in the following format:
Analysis: <reasoning process>
Complexity: <L0/L1/L2>
Index: <None/Lexical/Semantic/Hybrid>
Action: <Z/S-Sparse/S-Dense/S-Hybrid/M>"""

def formatting_prompts_func(examples, tokenizer):
    """
    将 JSON 数据格式化为 Qwen 的 Chat 模板
    Input fields: question_text, reasoning, complexity_label, index_strategy, action
    """
    convos = []
    texts = []
    
    # 批量处理
    for i in range(len(examples["question_text"])):
        question = examples["question_text"][i]
        
        # 构建 Assistant 的思维链回答
        # 这里的顺序至关重要：先推理(Reasoning)，再给结论(Action)
        assistant_content = (
            f"Analysis: {examples['reasoning'][i]}\n"
            f"Complexity: {examples['complexity_label'][i]}\n"
            f"Index: {examples['index_strategy'][i]}\n"
            f"Action: {examples['action'][i]}"
        )

        # 构建对话消息
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
            {"role": "assistant", "content": assistant_content}
        ]
        
        # 使用 Tokenizer 应用 Chat 模板
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        texts.append(text)
        
    return { "text": texts }
def main():
    # --- 路径检查逻辑优化 ---
    if os.path.exists(LOCAL_MODEL_PATH):
        MODEL_NAME = LOCAL_MODEL_PATH
        print(f"✅ 检测到本地模型，路径: {MODEL_NAME}")
        # 关键：如果本地存在，强制开启离线模式
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
            # 将 extra_kwargs 解包传入，强制 transformers 只看本地
            **extra_kwargs 
        )
    except Exception as e:
        print("\n❌ 模型加载失败！请检查以下几点：")
        print("1. 本地路径是否正确？")
        print(f"2. 目录 {MODEL_NAME} 下是否包含 config.json 和 model.safetensors 文件？")
        print("3. 报错信息如下：")
        raise e
    # 2. 配置 LoRA
    model = FastLanguageModel.get_peft_model(
        model,
        r = 16,
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj"],
        lora_alpha = 16,
        lora_dropout = 0.05, # 微调推荐稍微加一点 dropout 防止过拟合
        bias = "none",
        use_gradient_checkpointing = "unsloth", 
        random_state = 3407,
    )

    # 3. 加载数据集
    print(f"📂 Loading dataset from {DATA_FILE}...")
    if not os.path.exists(DATA_FILE):
        raise FileNotFoundError(f"找不到数据文件: {DATA_FILE}")

    # 直接加载 JSONL
    full_dataset = load_dataset("json", data_files=DATA_FILE, split="train")
    
    # 自动划分训练集和验证集 (90% Train, 10% Test)
    print("✂️ Splitting dataset...")
    dataset_split = full_dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = dataset_split["train"]
    test_dataset = dataset_split["test"]
    
    print(f"   Train samples: {len(train_dataset)}")
    print(f"   Test samples:  {len(test_dataset)}")

    # 4. 数据格式化
    print("🔄 Formatting prompts...")
    # 使用 functools.partial 或者 lambda 传递 tokenizer
    format_func = lambda x: formatting_prompts_func(x, tokenizer)
    
    train_dataset = train_dataset.map(format_func, batched=True)
    test_dataset = test_dataset.map(format_func, batched=True)

    # 5. 训练参数配置
    training_args = TrainingArguments(
        output_dir = OUTPUT_DIR,
        per_device_train_batch_size = 4,   # 显存优化: 4
        gradient_accumulation_steps = 8,   # 累积: 8 -> 等效 Batch Size 32
        warmup_steps = 20,
        num_train_epochs = 2,              # 建议跑 2-3 个 epoch 以充分学习 CoT
        learning_rate = 1e-4,              # Qwen 微调常用学习率
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(),
        logging_steps = 10,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "cosine",      # Cosine 通常比 linear 效果好
        seed = 3407,
        save_strategy = "epoch",
        evaluation_strategy = "steps",     # 边训练边验证
        eval_steps = 50,                   # 每 50 步验证一次
        save_total_limit = 2,
        report_to = "tensorboard",
    )

    # 6. Trainer 初始化
    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = train_dataset,
        eval_dataset = test_dataset,
        dataset_text_field = "text",
        max_seq_length = MAX_SEQ_LENGTH,
        dataset_num_proc = 4,
        packing = False, 
        args = training_args,
    )

    # 7. 开始训练
    print("\n🔥 Starting training...")
    trainer_stats = trainer.train()
    print("✅ Training completed!")

    # 8. 保存模型
    print(f"💾 Saving to {OUTPUT_DIR}/lora_model...")
    model.save_pretrained(os.path.join(OUTPUT_DIR, "lora_model"))
    tokenizer.save_pretrained(os.path.join(OUTPUT_DIR, "lora_model"))
    
    # 保存 tokenizer 的配置以防万一
    tokenizer.save_vocabulary(os.path.join(OUTPUT_DIR, "lora_model"))

if __name__ == "__main__":
    main()