import torch
import os
import sys

# 设置环境变量（必须在导入 unsloth 之前）
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
# 注意：旧版本 Unsloth (2025.1.x) 默认返回 logits，不需要设置这个环境变量
os.environ["UNSLOTH_RETURN_LOGITS"] = "1"  # 仅 2024.11+ 版本需要

from unsloth import FastLanguageModel
from datasets import load_from_disk
from trl import SFTTrainer
from transformers import TrainingArguments, AutoTokenizer
import torch.nn.functional as F
from collections import Counter
import json

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

# 类别映射 (用于计算类别权重)
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

class BalancedSFTTrainer(SFTTrainer):
    """
    支持类别加权损失的 SFTTrainer
    通过为少数类赋予更高权重来缓解类别不平衡问题
    """
    def __init__(self, *args, class_weights=None, label_token_ids=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights
        self.label_token_ids = label_token_ids

        if self.class_weights is not None:
            # 将权重移到正确的设备
            self.class_weights = self.class_weights.to(self.model.device)
            print(f"Using class weights: {self.class_weights}")

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """
        重写损失计算，使用类别加权的交叉熵
        """
        # 先调用原始的 forward 得到 logits
        # 对于 Unsloth 2024.11+，需要明确设置 return_dict=True 和 output_hidden_states=False
        outputs = model(**inputs, return_dict=True, use_cache=False)

        # 检查是否有 logits 属性
        if not hasattr(outputs, 'logits') or outputs.logits is None:
            raise RuntimeError(
                "模型输出中没有 logits！请确保在脚本开头设置了: "
                "os.environ['UNSLOTH_RETURN_LOGITS'] = '1'"
            )

        logits = outputs.logits
        labels = inputs["labels"]

        # 如果没有设置类别权重，使用默认损失
        if self.class_weights is None or self.label_token_ids is None:
            loss = outputs.loss if hasattr(outputs, "loss") and outputs.loss is not None else None
            if loss is None:
                # 手动计算标准交叉熵
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                loss = F.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                    ignore_index=-100
                )
            return (loss, outputs) if return_outputs else loss

        # 使用类别加权损失
        # Shift logits and labels for next-token prediction
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        # Flatten the tokens
        flat_logits = shift_logits.view(-1, shift_logits.size(-1))
        flat_labels = shift_labels.view(-1)

        # 找出所有预测标签位置（即输出 [[Z0]], [[S1]] 等的位置）
        # 我们只对这些位置的 token 应用类别权重
        # 对于其他位置，使用标准交叉熵

        # 创建一个权重张量
        sample_weights = torch.ones_like(flat_labels, dtype=torch.float)

        # 检测每个位置是否是标签 token
        for label_name, label_id in LABEL_TO_ID.items():
            # 找到对应的 token ID
            if label_name in self.label_token_ids:
                token_id = self.label_token_ids[label_name]
                # 找到所有预测这个标签的位置
                mask = (flat_labels == token_id)
                # 应用类别权重
                sample_weights[mask] = self.class_weights[label_id]

        # 计算加权交叉熵
        loss = F.cross_entropy(
            flat_logits,
            flat_labels,
            ignore_index=-100,
            reduction='none'
        )

        # 应用权重
        weighted_loss = loss * sample_weights

        # 只对非 padding 的 token 求平均
        mask = (flat_labels != -100)
        final_loss = weighted_loss[mask].sum() / mask.sum()

        return (final_loss, outputs) if return_outputs else final_loss


def compute_class_weights(dataset, tokenizer, method='balanced'):
    """
    从数据集中计算类别权重

    Args:
        dataset: 训练数据集
        tokenizer: tokenizer 用于解析标签
        method: 'balanced' 使用 sklearn 风格的平衡权重, 'inverse' 使用逆频率

    Returns:
        class_weights: torch.Tensor, 每个类别的权重
        label_token_ids: dict, 标签名到 token ID 的映射
    """
    print("Computing class weights from training data...")

    # 1. 统计每个类别的样本数
    label_counts = Counter()
    label_token_ids = {}

    for example in dataset:
        messages = example['messages']
        # 找到 assistant 的回复
        for msg in messages:
            if msg['role'] == 'assistant':
                content = msg['content']
                # 提取标签 (格式: [[Z0]], [[S1]] 等)
                import re
                match = re.search(r'\[\[(\w+)\]\]', content)
                if match:
                    label = match.group(1)
                    if label in LABEL_TO_ID:
                        label_counts[label] += 1

                        # 记录标签对应的 token ID (只记录一次)
                        if label not in label_token_ids:
                            # 编码标签文本得到 token ID
                            token_ids = tokenizer.encode(label, add_special_tokens=False)
                            if len(token_ids) > 0:
                                label_token_ids[label] = token_ids[0]

    print(f"Label distribution in training set: {dict(label_counts)}")
    print(f"Label token IDs: {label_token_ids}")

    # 2. 计算权重
    total_samples = sum(label_counts.values())
    num_classes = len(LABEL_TO_ID)

    class_weights = torch.zeros(num_classes)

    if method == 'balanced':
        # sklearn 风格: n_samples / (n_classes * n_samples_per_class)
        for label, count in label_counts.items():
            if label in LABEL_TO_ID:
                class_id = LABEL_TO_ID[label]
                class_weights[class_id] = total_samples / (num_classes * count)
    else:  # inverse
        # 简单的逆频率
        for label, count in label_counts.items():
            if label in LABEL_TO_ID:
                class_id = LABEL_TO_ID[label]
                class_weights[class_id] = 1.0 / count
        # 归一化
        class_weights = class_weights / class_weights.sum() * num_classes

    print(f"Computed class weights: {class_weights}")

    return class_weights, label_token_ids

def main():
    print(f"Loading model from {MODEL_NAME}...")
    
    extra_kwargs = {}
    if os.path.exists(MODEL_NAME):
        print("Using local model and offline mode.")
        # Manually load config to avoid unsloth/transformers bug with local_files_only dict handling
    
    model, _ = FastLanguageModel.from_pretrained(
        model_name = MODEL_NAME,
        max_seq_length = MAX_SEQ_LENGTH,
        dtype = DTYPE,
        load_in_4bit = LOAD_IN_4BIT,
        fix_tokenizer = False,
        **extra_kwargs,
    )
     # 2. 手动加载 Tokenizer (使用官方 Transformers 方式，这就不会报错了)
    print("Loading tokenizer manually...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.padding_side = "right" # 训练时通常用 right padding (Unsloth 默认也是 right)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

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

    # 计算类别权重 (基于训练集)
    class_weights, label_token_ids = compute_class_weights(
        dataset["train"],
        tokenizer,
        method='balanced'  # 可选: 'balanced' 或 'inverse'
    )

    # 训练参数
    training_args = TrainingArguments(
        output_dir = OUTPUT_DIR,
        per_device_train_batch_size = 32, # 根据显存调整，5090可以尝试 8 或 16
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

    trainer = BalancedSFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = dataset["train"],
        eval_dataset = dataset["test"],
        dataset_text_field = "text",
        packing = True, # Can make training 5x faster for short sequences.
        args = training_args,
        class_weights = class_weights,  # 传入类别权重
        label_token_ids = label_token_ids,  # 传入标签 token ID 映射
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

