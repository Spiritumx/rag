import torch
import os
import sys
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from unsloth import FastLanguageModel
from transformers import AutoTokenizer

# --- 配置 ---
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
LOCAL_MODEL_PATH = "/root/autodl-tmp/model/Qwen2.5-3B-Instruct"
BASE_MODEL_NAME = LOCAL_MODEL_PATH if os.path.exists(LOCAL_MODEL_PATH) else "Qwen/Qwen2.5-3B-Instruct"

LORA_PATH = "/root/autodl-tmp/output/lora_model"  # LoRA 适配器路径
OUTPUT_DIR = "/root/autodl-tmp/output"
MERGED_MODEL_DIR = os.path.join(OUTPUT_DIR, "merged_model")  # 合并后的模型保存路径

MAX_SEQ_LENGTH = 2048
DTYPE = torch.bfloat16
LOAD_IN_4BIT = False

def main():
    print("=" * 50)
    print("LoRA 模型合并脚本")
    print("=" * 50)

    # 检查 LoRA 路径是否存在
    if not os.path.exists(LORA_PATH):
        print(f"错误: LoRA 路径不存在: {LORA_PATH}")
        print("请先运行 finetune.py 进行微调！")
        return

    print(f"基础模型: {BASE_MODEL_NAME}")
    print(f"LoRA 适配器: {LORA_PATH}")
    print(f"输出路径: {MERGED_MODEL_DIR}")
    print()

    # 1. 使用 unsloth 加载模型和 LoRA 适配器
    print("正在加载基础模型和 LoRA 适配器...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = LORA_PATH,  # 直接加载 LoRA 路径，unsloth 会自动处理
        max_seq_length = MAX_SEQ_LENGTH,
        dtype = DTYPE,
        load_in_4bit = LOAD_IN_4BIT,
    )

    print("✓ 模型加载完成")
    print()

    # 2. 合并并保存模型
    print(f"正在合并 LoRA 权重并保存到 {MERGED_MODEL_DIR}...")
    os.makedirs(MERGED_MODEL_DIR, exist_ok=True)

    # 使用 unsloth 的 save_pretrained_merged 方法
    model.save_pretrained_merged(
        MERGED_MODEL_DIR,
        tokenizer,
        save_method="merged_16bit",  # 使用 float16 保存，平衡大小和精度
    )

    print("✓ 合并后的模型保存完成")
    print()
    print("=" * 50)
    print(f"合并完成！模型已保存到: {MERGED_MODEL_DIR}")
    print("=" * 50)
    print()
    print("下一步: 运行 evaluate.py 评估模型性能")

if __name__ == "__main__":
    main()
