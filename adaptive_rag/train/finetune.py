"""
Adaptive-RAG LoRA 训练脚本

基于 Qwen2.5-3B 微调路由分类器，使用 Adaptive-RAG 标注数据

Usage:
    python -m adaptive_rag.train.finetune
    python -m adaptive_rag.train.finetune --config adaptive_rag/config.yaml
"""

import torch
import os
import sys
import json
import argparse
from pathlib import Path

import yaml
from datasets import load_dataset
from unsloth import FastLanguageModel
from trl import SFTTrainer, SFTConfig

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def load_config(config_path: str = None) -> dict:
    """加载配置文件"""
    if config_path is None:
        config_path = PROJECT_ROOT / "adaptive_rag" / "config.yaml"

    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


# System Prompt for Adaptive-RAG Router
SYSTEM_PROMPT = """You are an expert RAG router. Analyze the user query complexity and determine the optimal retrieval strategy.
Output the analysis in the following format:
Analysis: <reasoning process>
Complexity: <L0/L1/L2>
Index: <None/BM25>
Action: <Z/S/M>"""


def formatting_prompts_func(examples, tokenizer):
    """
    将 JSON 数据格式化为 Qwen 的 Chat 模板
    """
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

        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )
        texts.append(text)

    return {"text": texts}


def main():
    parser = argparse.ArgumentParser(description="Adaptive-RAG LoRA 训练")
    parser.add_argument('--config', type=str, default=None,
                       help='配置文件路径')
    parser.add_argument('--data', type=str, default=None,
                       help='训练数据路径')

    args = parser.parse_args()

    # 加载配置
    config = load_config(args.config)
    train_config = config['training']

    # 数据路径
    if args.data:
        DATA_FILE = args.data
    else:
        DATA_FILE = str(PROJECT_ROOT / config['data']['training_data_path'])

    # 模型路径
    LOCAL_MODEL_PATH = train_config['base_model_path']
    OUTPUT_DIR = train_config['output_dir']

    # 训练超参数
    hyperparams = train_config['hyperparams']
    MAX_SEQ_LENGTH = hyperparams['max_seq_length']
    DTYPE = None
    LOAD_IN_4BIT = False

    # --- 路径检查 ---
    if os.path.exists(LOCAL_MODEL_PATH):
        MODEL_NAME = LOCAL_MODEL_PATH
        print(f"检测到本地模型，路径: {MODEL_NAME}")
        extra_kwargs = {"local_files_only": True}
    else:
        MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
        print(f"未找到本地模型 {LOCAL_MODEL_PATH}，尝试从 HuggingFace 下载...")
        extra_kwargs = {}

    # 1. 加载模型
    print(f"\n{'='*60}")
    print(f"Loading model from {MODEL_NAME}...")
    print(f"{'='*60}")

    try:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=MODEL_NAME,
            max_seq_length=MAX_SEQ_LENGTH,
            dtype=DTYPE,
            load_in_4bit=LOAD_IN_4BIT,
            **extra_kwargs
        )
    except Exception as e:
        print(f"\n模型加载失败！请检查路径。")
        raise e

    # 2. 配置 LoRA
    lora_config = train_config['lora']
    print(f"\nConfiguring LoRA...")
    print(f"  r = {lora_config['r']}")
    print(f"  alpha = {lora_config['alpha']}")
    print(f"  target_modules = {lora_config['target_modules']}")

    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_config['r'],
        target_modules=lora_config['target_modules'],
        lora_alpha=lora_config['alpha'],
        lora_dropout=lora_config['dropout'],
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )

    # 3. 加载数据集
    print(f"\n{'='*60}")
    print(f"Loading dataset from {DATA_FILE}...")
    print(f"{'='*60}")

    if not os.path.exists(DATA_FILE):
        raise FileNotFoundError(
            f"找不到训练数据文件: {DATA_FILE}\n"
            f"请先运行: python -m adaptive_rag.data.generate_labels"
        )

    full_dataset = load_dataset("json", data_files=DATA_FILE, split="train")

    print(f"Splitting dataset...")
    dataset_split = full_dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = dataset_split["train"]
    test_dataset = dataset_split["test"]

    print(f"  Train samples: {len(train_dataset)}")
    print(f"  Test samples:  {len(test_dataset)}")

    # 4. 数据格式化
    print(f"\nFormatting prompts...")
    format_func = lambda x: formatting_prompts_func(x, tokenizer)
    train_dataset = train_dataset.map(format_func, batched=True)
    test_dataset = test_dataset.map(format_func, batched=True)

    # 打印一个样本
    print(f"\nSample training text:")
    print("-" * 40)
    print(train_dataset[0]['text'][:500] + "...")
    print("-" * 40)

    # 5. 训练参数配置
    print(f"\n{'='*60}")
    print(f"Configuring training...")
    print(f"{'='*60}")
    print(f"  Output dir: {OUTPUT_DIR}")
    print(f"  Batch size: {hyperparams['batch_size']}")
    print(f"  Gradient accumulation: {hyperparams['gradient_accumulation_steps']}")
    print(f"  Learning rate: {hyperparams['learning_rate']}")
    print(f"  Epochs: {hyperparams['num_epochs']}")

    training_args = SFTConfig(
        output_dir=OUTPUT_DIR,
        dataset_text_field="text",
        max_length=MAX_SEQ_LENGTH,
        dataset_num_proc=4,
        packing=False,

        per_device_train_batch_size=hyperparams['batch_size'],
        gradient_accumulation_steps=hyperparams['gradient_accumulation_steps'],
        warmup_steps=hyperparams['warmup_steps'],
        num_train_epochs=hyperparams['num_epochs'],
        learning_rate=hyperparams['learning_rate'],
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=10,
        optim="adamw_8bit",
        weight_decay=hyperparams['weight_decay'],
        lr_scheduler_type=hyperparams['lr_scheduler'],
        seed=3407,

        save_strategy=train_config['save_strategy'],
        eval_strategy=train_config['eval_strategy'],
        eval_steps=train_config['eval_steps'],
        save_steps=train_config['save_steps'],
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        save_total_limit=train_config['save_total_limit'],
        report_to="tensorboard",
    )

    # 6. Trainer 初始化
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        args=training_args,
    )

    # 7. 开始训练
    print(f"\n{'='*60}")
    print(f"Starting training...")
    print(f"{'='*60}")

    trainer_stats = trainer.train()
    print(f"\nTraining completed!")

    # 8. 保存模型
    lora_save_path = os.path.join(OUTPUT_DIR, "lora_model")
    print(f"\n{'='*60}")
    print(f"Saving to {lora_save_path}...")
    print(f"{'='*60}")

    model.save_pretrained(lora_save_path)
    tokenizer.save_pretrained(lora_save_path)
    tokenizer.save_vocabulary(lora_save_path)

    print(f"\nDone!")
    print(f"LoRA adapter saved to: {lora_save_path}")


if __name__ == "__main__":
    main()
