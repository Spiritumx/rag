import torch
import os
import re
from datasets import load_dataset
from unsloth import FastLanguageModel
from tqdm import tqdm
from sklearn.metrics import classification_report

# --- 配置 ---
LORA_MODEL_DIR = "/root/autodl-tmp/output/Qwen2.5-3B-RAG-Router/lora_model"
DATA_FILE = "/root/graduateRAG/classifier/data/training_data/final_finetuning_dataset.jsonl"
MAX_SEQ_LENGTH = 2048

SYSTEM_PROMPT = """You are an expert RAG router. Analyze the user query complexity and determine the optimal retrieval strategy.
Output the analysis in the following format:
Analysis: <reasoning process>
Complexity: <L0/L1/L2>
Index: <None/Lexical/Semantic/Hybrid>
Action: <Z/S-Sparse/S-Dense/S-Hybrid/M>"""

def parse_action(text):
    """从模型输出中提取 Action"""
    match = re.search(r"Action:\s*(Z|S-Sparse|S-Dense|S-Hybrid|M)", text)
    if match:
        return match.group(1)
    return "Unknown"

def main():
    # 1. 加载模型
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = LORA_MODEL_DIR,
        max_seq_length = MAX_SEQ_LENGTH,
        dtype = None,
        load_in_4bit = False,
    )
    FastLanguageModel.for_inference(model)

    # 2. 加载测试集 (与训练时一样的 split 种子)
    print("Loading test dataset...")
    full_dataset = load_dataset("json", data_files=DATA_FILE, split="train")
    dataset_split = full_dataset.train_test_split(test_size=0.1, seed=42)
    test_dataset = dataset_split["test"]
    
    # 如果想跑得快一点，可以只测前 100 条
    # test_dataset = test_dataset.select(range(100)) 

    print(f"Evaluating on {len(test_dataset)} samples...")

    y_true = []
    y_pred = []

    # 3. 批量推理
    for item in tqdm(test_dataset):
        query = item["question_text"]
        true_action = item["action"]

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": query}
        ]
        inputs = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt").to("cuda")

        outputs = model.generate(inputs, max_new_tokens=256, temperature=0.01)
        response = tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)
        
        pred_action = parse_action(response)
        
        y_true.append(true_action)
        y_pred.append(pred_action)

    # 4. 输出报告
    print("\n" + "="*50)
    print("Classification Report")
    print("="*50)
    # 处理可能的 Unknown 标签
    labels = sorted(list(set(y_true + y_pred)))
    print(classification_report(y_true, y_pred, labels=labels, digits=4))

if __name__ == "__main__":
    main()