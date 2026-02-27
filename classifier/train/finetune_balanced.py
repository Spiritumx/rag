import torch
import torch.nn as nn
import os
import random
import json
from collections import Counter
from datasets import Dataset
from unsloth import FastLanguageModel
from trl import SFTTrainer, SFTConfig

# --- 1. 路径配置 ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.abspath(os.path.join(CURRENT_DIR, "../data/training_data/final_finetuning_dataset.jsonl"))
LOCAL_MODEL_PATH = "/root/autodl-tmp/model/Qwen2.5-3B-Instruct"
OUTPUT_DIR = "/root/autodl-tmp/output/Qwen2.5-3B-RAG-Router-Balanced"

MAX_SEQ_LENGTH = 2048
DTYPE = None
LOAD_IN_4BIT = False
SEED = 42

# --- 2. 过采样目标（仅针对训练集）---
# S-Dense: 270 → 810 (≈3x)   S-Hybrid: 74 → 370 (≈5x)
OVERSAMPLE_TARGETS = {
    "S-Dense":  810,
    "S-Hybrid": 370,
}

# --- 3. 类别加权损失权重（与过采样配合使用）---
# 过采样后仍保留一定权重，强化少数类的 gradient 信号
CLASS_LOSS_WEIGHTS = {
    "Z":        1.0,
    "S-Sparse": 1.0,
    "S-Dense":  2.0,   # 过采样已 3x，再额外 2x 权重
    "S-Hybrid": 3.0,   # 过采样已 5x，再额外 3x 权重
    "M":        1.0,
}

SYSTEM_PROMPT = """You are an expert RAG router. Analyze the user query complexity and determine the optimal retrieval strategy.
Output the analysis in the following format:
Analysis: <reasoning process>
Complexity: <L0/L1/L2>
Index: <None/Lexical/Semantic/Hybrid>
Action: <Z/S-Sparse/S-Dense/S-Hybrid/M>"""


# ==================== 数据处理 ====================

def load_records(data_file):
    records = []
    with open(data_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def split_records(records, test_size=0.2, seed=42):
    """手动切分，保证与 evaluate_hybrid_rule.py (test_size=0.2, seed=42) 完全一致"""
    # HuggingFace train_test_split 内部也是 shuffle + slice，用相同逻辑复现
    from datasets import Dataset as HFDataset
    ds = HFDataset.from_list(records)
    split = ds.train_test_split(test_size=test_size, seed=seed)
    return split["train"].to_list(), split["test"].to_list()


def oversample(records, targets, seed=42):
    """对少数类过采样至目标数量（只增不减）"""
    rng = random.Random(seed)
    by_class = {}
    for r in records:
        by_class.setdefault(r["action"], []).append(r)

    result = list(records)
    for label, target in targets.items():
        pool = by_class.get(label, [])
        current = len(pool)
        if current >= target:
            continue
        need = target - current
        extra = [pool[i % current] for i in range(need)]
        # 打乱 extra 避免固定顺序
        rng.shuffle(extra)
        result.extend(extra)

    rng.shuffle(result)
    return result


def print_distribution(records, tag=""):
    cnt = Counter(r["action"] for r in records)
    total = len(records)
    print(f"\n{'='*40}\n{tag} 样本分布（共 {total} 条）")
    for k in ["Z", "S-Sparse", "S-Dense", "S-Hybrid", "M"]:
        n = cnt.get(k, 0)
        print(f"  {k:<12}: {n:5d}  ({n/total:.1%})")
    print('='*40)


def formatting_prompts_func(examples, tokenizer):
    texts = []
    for i in range(len(examples["question_text"])):
        assistant_content = (
            f"Analysis: {examples['reasoning'][i]}\n"
            f"Complexity: {examples['complexity_label'][i]}\n"
            f"Index: {examples['index_strategy'][i]}\n"
            f"Action: {examples['action'][i]}"
        )
        messages = [
            {"role": "system",    "content": SYSTEM_PROMPT},
            {"role": "user",      "content": examples["question_text"][i]},
            {"role": "assistant", "content": assistant_content},
        ]
        texts.append(tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        ))
    return {"text": texts}


# ==================== 加权损失 Trainer ====================

class WeightedSFTTrainer(SFTTrainer):
    """在 SFTTrainer 基础上支持 per-sample 损失加权。

    数据集中须包含 "sample_weight" 字段（float），
    训练时对每条样本的 token-level loss 乘以对应权重。
    """

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # 从 inputs 中弹出权重（避免传入 model.forward）
        sample_weights = inputs.pop("sample_weight", None)

        # 没有权重时走标准流程
        if sample_weights is None or return_outputs:
            return super().compute_loss(
                model, inputs, return_outputs=return_outputs, **kwargs
            )

        labels = inputs.get("labels")
        if labels is None:
            return super().compute_loss(
                model, inputs, return_outputs=return_outputs, **kwargs
            )

        outputs = model(**inputs)
        logits = outputs.logits  # [B, L, V]

        # token-level cross-entropy，不做 reduction
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        loss_fct = nn.CrossEntropyLoss(reduction="none", ignore_index=-100)
        per_token_loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        ).view(shift_labels.size())   # [B, L-1]

        # 每条样本取非 padding 位置的平均 loss
        mask = (shift_labels != -100).float()
        per_sample_loss = (per_token_loss * mask).sum(dim=-1) / mask.sum(dim=-1).clamp(min=1)

        # 乘以样本权重后取 batch 均值
        w = sample_weights.to(per_sample_loss.dtype).to(per_sample_loss.device)
        weighted_loss = (per_sample_loss * w).mean()
        return weighted_loss


# ==================== 主流程 ====================

def main():
    # 1. 加载模型
    if os.path.exists(LOCAL_MODEL_PATH):
        model_name = LOCAL_MODEL_PATH
        extra_kwargs = {"local_files_only": True}
    else:
        model_name = "Qwen/Qwen2.5-3B-Instruct"
        extra_kwargs = {}

    print(f"🚀 Loading model: {model_name}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=DTYPE,
        load_in_4bit=LOAD_IN_4BIT,
        **extra_kwargs,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                         "gate_proj", "up_proj", "down_proj"],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )

    # 2. 加载 & 切分原始数据（80:20，seed=42）
    print(f"\n📂 Loading dataset: {DATA_FILE}")
    records = load_records(DATA_FILE)
    train_records, test_records = split_records(records, test_size=0.2, seed=SEED)

    print_distribution(train_records, "原始训练集")
    print_distribution(test_records,  "测试集（固定不变）")

    # 3. 仅对训练集过采样
    train_records = oversample(train_records, OVERSAMPLE_TARGETS, seed=SEED)
    print_distribution(train_records, "过采样后训练集")

    # 4. 为每条样本添加 sample_weight 字段
    for r in train_records:
        r["sample_weight"] = CLASS_LOSS_WEIGHTS.get(r["action"], 1.0)
    for r in test_records:
        r["sample_weight"] = 1.0   # 评估不用权重，但字段须一致

    train_dataset = Dataset.from_list(train_records)
    test_dataset  = Dataset.from_list(test_records)

    # 5. 格式化文本
    print("\n🔄 Formatting prompts...")
    fmt = lambda x: formatting_prompts_func(x, tokenizer)
    train_dataset = train_dataset.map(fmt, batched=True)
    test_dataset  = test_dataset.map(fmt, batched=True)

    # 6. 训练配置
    training_args = SFTConfig(
        output_dir=OUTPUT_DIR,
        dataset_text_field="text",
        max_length=MAX_SEQ_LENGTH,
        dataset_num_proc=4,
        packing=False,

        per_device_train_batch_size=4,
        gradient_accumulation_steps=8,
        warmup_steps=20,
        num_train_epochs=3,          # 过采样后每轮覆盖更多少数类，适当增加 epoch
        learning_rate=8e-5,          # 稍降 lr，避免少数类重复样本过拟合
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=10,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        seed=3407,

        save_strategy="steps",
        eval_strategy="steps",
        eval_steps=50,
        save_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        save_total_limit=2,
        report_to="tensorboard",

        # 让 Trainer 保留 sample_weight 列（不自动移除未知列）
        remove_unused_columns=False,
    )

    trainer = WeightedSFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        args=training_args,
    )

    print("\n🔥 Starting balanced training...")
    trainer.train()
    print("✅ Training completed!")

    save_path = os.path.join(OUTPUT_DIR, "lora_model")
    print(f"💾 Saving to {save_path}...")
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    tokenizer.save_vocabulary(save_path)


if __name__ == "__main__":
    main()
