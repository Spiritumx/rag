#!/usr/bin/env python3
"""测试单个样本 - 查看模型实际输出"""
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["UNSLOTH_DISABLE_STATISTICS"] = "1"

import torch
from unsloth import FastLanguageModel
from datasets import load_from_disk
from transformers import AutoTokenizer
import json

LORA_PATH = "/root/autodl-tmp/output/lora_model"
DATA_DIR = "/root/autodl-tmp/data"
MAX_SEQ_LENGTH = 2048
DTYPE = torch.bfloat16

print("=" * 70)
print("加载模型和数据")
print("=" * 70)

# 加载模型
print("\n1. 加载 LoRA 模型...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = LORA_PATH,
    max_seq_length = MAX_SEQ_LENGTH,
    dtype = DTYPE,
    load_in_4bit = False,
)

# 确保 tokenizer 设置正确
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

print(f"   Tokenizer pad_token: {tokenizer.pad_token}")
print(f"   Tokenizer eos_token: {tokenizer.eos_token}")

FastLanguageModel.for_inference(model)
print("   ✓ 模型加载完成")

# 加载测试数据
print("\n2. 加载测试数据...")
dataset = load_from_disk(DATA_DIR)
test_data = dataset["test"]
print(f"   测试集样本数: {len(test_data)}")

# 测试第一个样本
print("\n" + "=" * 70)
print("测试样本 1")
print("=" * 70)

sample = test_data[0]
input_msgs = sample["messages"][:-1]
ground_truth = sample["messages"][-1]["content"]

print("\n输入消息:")
print(json.dumps(input_msgs, indent=2, ensure_ascii=False))

print("\n真实标签（Ground Truth）:")
print(ground_truth[:300])
print("...")

# 生成预测
print("\n" + "=" * 70)
print("生成预测")
print("=" * 70)

inputs = tokenizer.apply_chat_template(
    input_msgs,
    tokenize=True,
    add_generation_prompt=True,
    return_tensors="pt",
    return_dict=True
)

input_ids = inputs["input_ids"].to("cuda")
attention_mask = inputs.get("attention_mask")
if attention_mask is not None:
    attention_mask = attention_mask.to("cuda")

print(f"\n输入 token 数量: {input_ids.shape[-1]}")

# 测试不同的 max_new_tokens
for max_tokens in [256, 512]:
    print(f"\n{'=' * 70}")
    print(f"测试 max_new_tokens = {max_tokens}")
    print('=' * 70)

    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_tokens,
            temperature=0.0,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    generated_ids = outputs[0][input_ids.shape[-1]:]
    prediction = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    print(f"\n生成的 token 数量: {len(generated_ids)}")
    print(f"生成的文本长度: {len(prediction)} 字符")
    print(f"\n完整输出:")
    print("-" * 70)
    print(prediction)
    print("-" * 70)

    # 检查是否包含标签
    import re
    match = re.search(r'\[\[(\w+)\]\]', prediction)
    if match:
        print(f"\n✓ 找到标签: {match.group(1)}")
    else:
        print("\n✗ 未找到标签！")

    # 检查输出格式
    if "### Analysis:" in prediction or "### Risk Assessment:" in prediction:
        print("✓ 输出格式正确（包含分析结构）")
    else:
        print("✗ 输出格式异常（缺少分析结构）")

print("\n" + "=" * 70)
print("诊断建议")
print("=" * 70)

print("""
根据上面的输出，检查以下几点：

1. 如果输出包含正确的标签 [[XX]]，说明模型工作正常
2. 如果输出是乱码或不完整，可能的原因：
   - LoRA 模型加载有问题
   - tokenizer 设置不正确
   - 训练数据和评估数据格式不一致

3. 如果生成的 token 数量接近 max_new_tokens，说明模型一直在生成
   - 可能没有学会何时停止
   - 或者 eos_token 设置有问题

4. 下一步：
   - 如果这个测试正常，但评估脚本有问题，说明评估脚本需要修复
   - 如果这个测试也不正常，说明模型加载或推理有问题
""")
