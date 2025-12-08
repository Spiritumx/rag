"""
L0 数据生成与合并脚本
功能：
1. 使用 GPT-4o 生成非检索类 (L0) 数据（代码、逻辑、闲聊等）
2. 读取现有的 RAG 训练数据
3. 合并两者，并进行随机打乱 (Shuffle)
4. 输出最终的微调数据集和分布统计
"""

import asyncio
import json
import os
import random
from pathlib import Path
from typing import List, Literal, Dict, Any
from collections import Counter
from pydantic import BaseModel, Field
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm

# ==========================================
# 配置区域
# ==========================================
# 目标 L0 生成数量 (约占总数据的 1/5 - 1/6)
TARGET_L0_COUNT = 600
BATCH_SIZE = 10

# 路径配置
DATA_DIR = Path("training_data")
EXISTING_DATA_FILE = DATA_DIR / "combined_training_data.jsonl" # 之前的 NQ/HotpotQA 数据
L0_OUTPUT_FILE = DATA_DIR / "l0_synthetic.jsonl"               #生成的 L0 中间文件
FINAL_OUTPUT_FILE = DATA_DIR / "final_finetuning_dataset.jsonl" # 最终合并文件

# 定义 L0 的子类别，确保多样性
CATEGORIES = [
    "Coding & Debugging (Python, JS, SQL, no external docs needed)",
    "Logic Puzzles & Math (Reasoning based)",
    "Creative Writing (Poems, Emails, Stories, Re-writing)",
    "General Chitchat & Roleplay (Greeting, Advice)",
    "Text Processing (Translation, Summarization of provided text)",
    "Common Sense (General knowledge that models definitely know, e.g., color of sky)"
]

# ==========================================
# 数据结构与 Prompt
# ==========================================

class L0Sample(BaseModel):
    question_text: str = Field(..., description="The user query")
    reasoning: str = Field(..., description="Why this is L0. E.g., 'This is a logic puzzle...', 'This is a coding request...'")
    complexity_label: Literal["L0"] = "L0"
    index_strategy: Literal["None"] = "None"
    action: Literal["Z"] = "Z"

class L0Batch(BaseModel):
    samples: List[L0Sample]

SYSTEM_PROMPT = """You are a data generator for RAG router training. 
Your task is to generate diverse User Queries that strictly belong to the **L0 (No Retrieval)** category.

### DEFINITION OF L0
- **Self-contained**: The answer can be derived purely from the LLM's internal weights (logic, coding, language skills) or the query itself.
- **No External Facts**: Do NOT generate queries that ask for real-world entities, events, people, or locations (e.g., No "Who is...", No "Where is...").
- **Categories**: Logic, Math, Coding, Writing, Translation, Chitchat.

### REQUIREMENT
Generate distinct, realistic, and diverse queries based on the requested category.
Provide the 'reasoning' for why it is L0.
"""

# ==========================================
# 核心函数
# ==========================================

async def generate_batch(client: AsyncOpenAI, category: str) -> List[dict]:
    """生成一批 L0 数据"""
    try:
        response = await client.beta.chat.completions.parse(
            model="gpt-4o-2024-08-06",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Generate {BATCH_SIZE} diverse L0 examples for the category: {category}."}
            ],
            response_format=L0Batch,
        )
        
        results = []
        for sample in response.choices[0].message.parsed.samples:
            results.append({
                "dataset": "synthetic_l0",
                "question_id": f"l0_{os.urandom(4).hex()}",
                "question_text": sample.question_text,
                "reasoning": sample.reasoning,
                "complexity_label": "L0",
                "index_strategy": "None",
                "action": "Z",
                "answers": []
            })
        return results
    except Exception as e:
        print(f"⚠️ Error generating batch: {e}")
        return []

def load_jsonl(file_path: Path) -> List[Dict[str, Any]]:
    """读取 JSONL 文件"""
    data = []
    if not file_path.exists():
        print(f"⚠️ Warning: File not found: {file_path}")
        return []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return data

def save_jsonl(data: List[Dict[str, Any]], file_path: Path):
    """保存 JSONL 文件"""
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

def merge_and_shuffle():
    """合并现有数据和生成的 L0 数据"""
    print("\n" + "="*60)
    print("🔄 Merging and Shuffling Datasets...")
    print("="*60)

    # 1. 读取数据
    existing_data = load_jsonl(EXISTING_DATA_FILE)
    l0_data = load_jsonl(L0_OUTPUT_FILE)
    
    print(f"📄 Existing RAG Data: {len(existing_data)} items")
    print(f"📄 Generated L0 Data: {len(l0_data)} items")
    
    if not existing_data and not l0_data:
        print("❌ Error: No data to merge!")
        return

    # 2. 合并
    combined_data = existing_data + l0_data
    
    # 3. 打乱 (Shuffle) -这对训练至关重要
    random.shuffle(combined_data)
    
    # 4. 保存
    save_jsonl(combined_data, FINAL_OUTPUT_FILE)
    
    # 5. 统计分布
    print("\n" + "="*60)
    print("📊 Final Dataset Statistics")
    print("="*60)
    
    complexity_counts = Counter(item['complexity_label'] for item in combined_data)
    action_counts = Counter(item['action'] for item in combined_data)
    
    print(f"Total Samples: {len(combined_data)}")
    print(f"Output File:   {FINAL_OUTPUT_FILE}")
    print("\n[Complexity Distribution]")
    for k, v in complexity_counts.items():
        print(f"  {k}: {v}")
        
    print("\n[Action Distribution]")
    for k, v in action_counts.items():
        print(f"  {k}: {v}")
        
    print("\n✅ Ready for Fine-tuning!")

# ==========================================
# 主流程
# ==========================================

async def main():
    print("="*60)
    print("🚀 L0 Data Generation & Merging Tool")
    print("="*60)

    # 1. 检查 API Key
    api_key = os.getenv("OPENAI_API_KEY")
    # 也可以从 config.json 读取，这里为了脚本独立性直接读取环境变量或手动填入
    # if not api_key: api_key = "YOUR_KEY_HERE"
    
    if not api_key:
        print("❌ Error: Please set OPENAI_API_KEY environment variable.")
        return

    client = AsyncOpenAI(api_key=api_key)
    
    # 2. 生成 L0 数据
    if L0_OUTPUT_FILE.exists():
        print(f"ℹ️  {L0_OUTPUT_FILE} already exists.")
        choice = input("Overwrite? (y/n): ").strip().lower()
        if choice != 'y':
            print("Skipping generation, proceeding to merge...")
            merge_and_shuffle()
            return

    # 确保目录存在
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    total_batches = TARGET_L0_COUNT // BATCH_SIZE
    tasks = []
    
    print(f"\nGenerating {TARGET_L0_COUNT} samples across {len(CATEGORIES)} categories...")
    
    for i in range(total_batches):
        category = CATEGORIES[i % len(CATEGORIES)]
        tasks.append(generate_batch(client, category))
    
    all_l0_data = []
    for f in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="GPT-4o Generating"):
        batch_data = await f
        all_l0_data.extend(batch_data)
        
    # 保存中间结果
    save_jsonl(all_l0_data, L0_OUTPUT_FILE)
    print(f"✅ Saved intermediate L0 data to {L0_OUTPUT_FILE}")
    
    # 3. 执行合并
    merge_and_shuffle()

if __name__ == "__main__":
    asyncio.run(main())