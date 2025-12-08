#!/usr/bin/env python3
"""诊断评估速度 - 查看模型实际生成了多少 tokens"""
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["UNSLOTH_DISABLE_STATISTICS"] = "1"

import torch
import time
from unsloth import FastLanguageModel
from datasets import load_from_disk
from transformers import AutoTokenizer

LORA_PATH = "/root/autodl-tmp/output/lora_model"
DATA_DIR = "/root/autodl-tmp/data"
MAX_SEQ_LENGTH = 2048
DTYPE = torch.bfloat16

print("加载模型...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = LORA_PATH,
    max_seq_length = MAX_SEQ_LENGTH,
    dtype = DTYPE,
    load_in_4bit = False,
)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
FastLanguageModel.for_inference(model)

print("加载数据...")
dataset = load_from_disk(DATA_DIR)
test_data = dataset["test"]

print("\n" + "=" * 70)
print("测试前 5 个样本的生成速度和长度")
print("=" * 70)

for i in range(min(5, len(test_data))):
    item = test_data[i]
    input_msgs = item["messages"][:-1]

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

    # 测试生成时间
    start_time = time.time()
    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=512,
            temperature=0.0,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    end_time = time.time()

    generated_ids = outputs[0][input_ids.shape[-1]:]
    prediction = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    # 统计
    num_generated_tokens = len(generated_ids)
    elapsed_time = end_time - start_time
    tokens_per_sec = num_generated_tokens / elapsed_time if elapsed_time > 0 else 0

    print(f"\n样本 {i+1}:")
    print(f"  生成 tokens 数量: {num_generated_tokens}")
    print(f"  生成时间: {elapsed_time:.2f} 秒")
    print(f"  生成速度: {tokens_per_sec:.1f} tokens/秒")
    print(f"  生成内容长度: {len(prediction)} 字符")
    print(f"  生成内容预览:")
    print(f"    开头: {prediction[:150]}...")
    print(f"    结尾: ...{prediction[-100:]}")

print("\n" + "=" * 70)
print("诊断建议:")
print("=" * 70)
print("""
如果生成 tokens 数量接近 512，说明模型一直在生成直到达到 max_new_tokens。
可能的原因：
1. 模型没有学会生成 eos_token 来停止
2. 训练数据中的回复太长

如果生成速度 < 20 tokens/秒，说明生成本身很慢（可能是模型或硬件问题）。
正常速度应该在 30-50 tokens/秒左右。

如果每个样本耗时 > 10 秒，建议：
1. 减小 max_new_tokens 到 256
2. 检查模型是否正确加载
3. 检查 GPU 利用率
""")
