import os
import sys

# =========================================================
# 必须放在最前面！在导入 unsloth 之前！
# =========================================================
# 1. 强制使用镜像源 (防止连接官方 HF 超时)
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
# 2. 彻底禁用 Unsloth 的统计和升级检查 (解决 stats_check 报错的核心)
os.environ["UNSLOTH_DISABLE_STATISTICS"] = "1"
os.environ["WANDB_DISABLED"] = "true" # 可选：禁用 WandB 防止额外的网络请求
# =========================================================

import torch
import gc
# 现在再导入 unsloth
from unsloth import FastLanguageModel 
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import re

# --- 配置 ---
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
LOCAL_MODEL_PATH = "/root/autodl-tmp/model/Qwen2.5-3B-Instruct"
BASE_MODEL_NAME = LOCAL_MODEL_PATH if os.path.exists(LOCAL_MODEL_PATH) else "Qwen/Qwen2.5-3B-Instruct"

# ⚠️ 强制先测试 LoRA，跳过 Merged Model，以判断是否是合并过程搞坏了模型
USE_LORA_ONLY = True 
LORA_PATH = "/root/autodl-tmp/output/lora_model"
DATA_DIR = "/root/autodl-tmp/data"
MAX_SEQ_LENGTH = 2048
DTYPE = torch.bfloat16

def extract_label(text):
    if not text: return "UNKNOWN"
    match = re.search(r'\[\[([ZSM]\d)\]\]', text)
    if match: return match.group(1)
    match = re.search(r'\b([ZSM]\d)\b', text)
    if match: return match.group(1)
    return "UNKNOWN"

def debug_inference():
    print("=" * 60)
    print("🔍 深度诊断模式 (Debug Mode)")
    print("=" * 60)

    # 1. 强制加载 Base Tokenizer (确保特殊符号正确)
    print(f"1. 正在加载官方 Tokenizer: {BASE_MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME, trust_remote_code=True)
    tokenizer.padding_side = "left"
    
    # 修复 Qwen 可能缺失 pad_token 的问题
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    print(f"   PAD ID: {tokenizer.pad_token_id} | EOS ID: {tokenizer.eos_token_id}")

    # 2. 加载模型 (优先使用 LoRA 检查训练是否成功)
    print(f"2. 正在加载模型 (Mode: {'LoRA Adapter' if USE_LORA_ONLY else 'Merged Model'})...")
    
    # 清理显存
    gc.collect()
    torch.cuda.empty_cache()

    if USE_LORA_ONLY:
        # 使用 Unsloth 加载 LoRA，这是最稳的方式
        model, _ = FastLanguageModel.from_pretrained(
            model_name = LORA_PATH, 
            max_seq_length = MAX_SEQ_LENGTH,
            dtype = DTYPE,
            load_in_4bit = False,
        )
        FastLanguageModel.for_inference(model)
    else:
        # 加载合并后的模型
        model = AutoModelForCausalLM.from_pretrained(
            "/root/autodl-tmp/output/merged_model",
            torch_dtype=DTYPE,
            device_map="cuda",
            trust_remote_code=True
        )

    print("✓ 模型加载完成")

    # 3. 加载数据
    dataset = load_from_disk(DATA_DIR)
    test_data = dataset["test"]
    
    # 4. 取一个样本进行详细解剖
    print("\n" + "="*30 + " PROMPT 检查 " + "="*30)
    sample = test_data[0] # 取第一个样本
    input_msgs = sample["messages"][:-1]
    
    # 查看 Apply Chat Template 后的真实样子
    prompt_str = tokenizer.apply_chat_template(input_msgs, tokenize=False, add_generation_prompt=True)
    print("【发送给模型的真实 Prompt (前300字符)】:")
    print(prompt_str[:300])
    print("...\n【Prompt 结尾 (检查 header)】:")
    print(prompt_str[-100:])
    
    if "<|im_start|>" not in prompt_str:
        print("\n❌ 严重警告: Prompt 中未发现 <|im_start|> 标签！Chat Template 未生效！")
    else:
        print("\n✅ Chat Template 格式看起来正常。")

    print("\n" + "="*30 + " 生成测试 " + "="*30)
    
    inputs = tokenizer.apply_chat_template(
        input_msgs, 
        tokenize=True, 
        add_generation_prompt=True, 
        return_tensors="pt", 
        return_dict=True
    ).to("cuda")

    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_new_tokens=128,
            temperature=0.1,    # 低温
            do_sample=False,    # 贪婪解码
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.1 # 加强一点惩罚
        )
    
    decoded = tokenizer.decode(outputs[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True)
    print("\n【模型输出结果】:")
    print(decoded)
    print("=" * 60)

    # 5. 快速验证前 10 个样本的准确率
    print("\n正在快速验证前 10 个样本...")
    correct = 0
    for i in range(10):
        item = test_data[i]
        gt = extract_label(item["messages"][-1]["content"])
        
        inputs = tokenizer.apply_chat_template(
            item["messages"][:-1], tokenize=True, add_generation_prompt=True, return_tensors="pt", return_dict=True
        ).to("cuda")
        
        with torch.no_grad():
            outputs = model.generate(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_new_tokens=128,
                do_sample=False,
                eos_token_id=tokenizer.eos_token_id
            )
        pred_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True)
        pred = extract_label(pred_text)
        
        print(f"[{i+1}] GT: {gt} | Pred: {pred} | Output: {pred_text[:50].replace(chr(10), ' ')}...")
        if gt == pred: correct += 1
        
    print(f"\nTop-10 Accuracy: {correct/10:.0%}")

if __name__ == "__main__":
    debug_inference()