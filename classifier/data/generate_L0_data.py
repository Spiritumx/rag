"""
L0 数据生成与合并脚本
功能：
1. 异步并发生成 L0 (非检索类) 数据
2. 读取现有的 RAG 训练数据
3. 合并两者，进行随机打乱 (Shuffle)
4. 输出最终微调数据集
"""

import asyncio
import json
import os
import random
from pathlib import Path
from typing import List, Dict, Any, Literal
from collections import Counter
from pydantic import BaseModel, Field
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm

# ==========================================
# 常量定义
# ==========================================

# L0 的子类别，确保多样性
L0_CATEGORIES = [
    "Coding & Debugging (Python, JS, SQL, no external docs needed)",
    "Logic Puzzles & Math (Reasoning based)",
    "Creative Writing (Poems, Emails, Stories, Re-writing)",
    "General Chitchat & Roleplay (Greeting, Advice)",
    "Text Processing (Translation, Summarization of provided text)",
    "Common Sense (General knowledge that models definitely know, e.g., color of sky)"
]

# ==========================================
# Pydantic Schema
# ==========================================

class L0Sample(BaseModel):
    question_text: str = Field(..., description="The user query")
    reasoning: str = Field(..., description="Why this is L0. E.g., 'This is a logic puzzle...', 'This is a coding request...'")
    # 强制固定字段，确保与训练数据格式一致
    complexity_label: Literal["L0"] = "L0"
    index_strategy: Literal["None"] = "None"
    action: Literal["Z"] = "Z"

class L0Batch(BaseModel):
    samples: List[L0Sample]

# ==========================================
# System Prompt
# ==========================================

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
# 核心管道类
# ==========================================

class L0GeneratorPipeline:
    def __init__(
        self,
        api_key: str = None,
        base_url: str = None,
        model: str = "gpt-4o-2024-08-06",
        max_concurrent: int = 5,
        target_count: int = 600,
        batch_size: int = 10,
        data_dir: str = "./training_data"
    ):
        """
        初始化生成管道
        """
        self.client = AsyncOpenAI(
            api_key=api_key or os.getenv("OPENAI_API_KEY"),
            base_url=base_url
        )
        self.model = model
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.target_count = target_count
        self.batch_size = batch_size
        self.data_dir = Path(data_dir)
        
        # 文件路径配置
        self.existing_data_file = self.data_dir / "combined_training_data.jsonl"
        self.l0_output_file = self.data_dir / "l0_synthetic.jsonl"
        self.final_output_file = self.data_dir / "final_finetuning_dataset.jsonl"

        # 创建目录
        try:
            self.data_dir.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            print(f"❌ ERROR: Cannot create output directory: {self.data_dir}")
            raise

    async def generate_batch(self, category: str, retry: int = 3) -> List[Dict[str, Any]]:
        """生成一批 L0 数据"""
        async with self.semaphore:
            for attempt in range(retry):
                try:
                    response = await self.client.beta.chat.completions.parse(
                        model=self.model,
                        messages=[
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": f"Generate {self.batch_size} diverse L0 examples for the category: {category}."}
                        ],
                        response_format=L0Batch,
                    )
                    
                    # 转换为标准字典格式
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
                            "answers": [] # L0 通常无标准答案
                        })
                    return results

                except Exception as e:
                    if attempt == retry - 1:
                        print(f"⚠️  Error generating batch for {category}: {e}")
                        return []
                    await asyncio.sleep(2 ** attempt)

    async def run_generation(self):
        """执行全量生成"""
        if self.l0_output_file.exists():
            print(f"ℹ️  L0 file already exists: {self.l0_output_file}")
            choice = input("Overwrite? (y/n): ").strip().lower()
            if choice != 'y':
                print("Skipping generation...")
                return

        total_batches = self.target_count // self.batch_size
        tasks = []

        print(f"\n🚀 Starting generation of {self.target_count} L0 samples...")
        print(f"   Model: {self.model}")
        print(f"   Categories: {len(L0_CATEGORIES)}")

        # 分配任务
        for i in range(total_batches):
            category = L0_CATEGORIES[i % len(L0_CATEGORIES)]
            tasks.append(self.generate_batch(category))

        # 并发执行
        all_l0_data = []
        for f in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Generating Batches"):
            batch_results = await f
            all_l0_data.extend(batch_results)

        # 保存中间结果
        print(f"💾 Saving {len(all_l0_data)} samples to {self.l0_output_file}")
        with open(self.l0_output_file, 'w', encoding='utf-8') as f:
            for item in all_l0_data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')

    def merge_datasets(self):
        """合并、打乱并保存最终数据集"""
        print("\n" + "="*60)
        print("🔄 Merging and Shuffling Datasets")
        print("="*60)

        existing_data = []
        l0_data = []

        # 1. 读取现有数据
        if self.existing_data_file.exists():
            with open(self.existing_data_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        existing_data.append(json.loads(line))
            print(f"📄 Loaded Existing Data: {len(existing_data)} items")
        else:
            print(f"⚠️  Warning: Existing data file not found: {self.existing_data_file}")

        # 2. 读取 L0 数据
        if self.l0_output_file.exists():
            with open(self.l0_output_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        l0_data.append(json.loads(line))
            print(f"📄 Loaded L0 Synthetic Data: {len(l0_data)} items")
        else:
            print(f"❌ Error: L0 data file not found!")
            return

        if not existing_data and not l0_data:
            print("❌ No data to merge. Exiting.")
            return

        # 3. 合并
        combined_data = existing_data + l0_data

        # 4. 打乱 (Shuffle) - 关键步骤
        print("🎲 Shuffling data...")
        random.shuffle(combined_data)

        # 5. 保存
        with open(self.final_output_file, 'w', encoding='utf-8') as f:
            for item in combined_data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')

        # 6. 统计
        self._print_stats(combined_data)

    def _print_stats(self, data: List[Dict[str, Any]]):
        print("\n" + "="*60)
        print("📊 Final Dataset Statistics")
        print("="*60)
        
        complexity_counts = Counter(item.get('complexity_label', 'unknown') for item in data)
        action_counts = Counter(item.get('action', 'unknown') for item in data)
        
        print(f"Total Samples: {len(data)}")
        print(f"Output File:   {self.final_output_file}")
        print("\n[Complexity Distribution]")
        for k, v in complexity_counts.items():
            print(f"  {k:<5}: {v}")
            
        print("\n[Action Distribution]")
        for k, v in action_counts.items():
            print(f"  {k:<10}: {v}")

# ==========================================
# 主函数
# ==========================================

async def main():
    print("=" * 60)
    print("🛠️  L0 Data Generator & Merger")
    print("=" * 60)
    print()

    # 1. 加载 Config
    config_file = Path("config.json")
    if not config_file.exists():
        print("❌ ERROR: config.json not found!")
        return

    print("📄 Loading configuration...")
    with open(config_file, 'r', encoding='utf-8') as f:
        config = json.load(f)

    api_settings = config.get("api_settings", {})
    data_settings = config.get("data_settings", {})

    # 2. 检查 API Key
    api_key = api_settings.get("api_key") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ ERROR: No API key provided in config or environment variables.")
        return

    # 3. 获取生成配置 (支持在 config.json 中添加 l0_target_count)
    target_count = data_settings.get("l0_target_count", 600)
    output_dir = data_settings.get("output_dir", "./training_data")

    # 4. 初始化 Pipeline
    pipeline = L0GeneratorPipeline(
        api_key=api_key,
        base_url=api_settings.get("base_url"),
        model=api_settings.get("model", "gpt-4o-2024-08-06"), # 建议 L0 生成使用强模型
        max_concurrent=api_settings.get("max_concurrent", 5),
        target_count=target_count,
        batch_size=10,
        data_dir=output_dir
    )

    # 5. 运行生成
    await pipeline.run_generation()

    # 6. 运行合并
    pipeline.merge_datasets()

    print("\n✅ All tasks completed successfully!")

if __name__ == "__main__":
    asyncio.run(main())