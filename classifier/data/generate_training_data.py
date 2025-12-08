"""
数据增强脚本 - 使用 OpenAI Structured Outputs 生成 RAG 路由分类训练数据
"""

import asyncio
import json
import os
from pathlib import Path
from typing import List, Dict, Any, Literal
from pydantic import BaseModel, Field
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm


# Pydantic Schema Definition
class RagRoutingAnalysis(BaseModel):
    reasoning: str = Field(
        ...,
        description="Step-by-step analysis of the query's linguistic structure, entity specificity, and logical complexity. Do NOT answer the question."
    )
    complexity_label: Literal["L0", "L1", "L2"] = Field(
        ...,
        description="L0: Non-factual/Logic/Code. L1: Single-hop factual. L2: Multi-hop/Comparison/Complex."
    )
    index_strategy: Literal["None", "Lexical", "Semantic", "Hybrid"] = Field(
        ...,
        description="The optimal indexing method based on linguistic features. 'Lexical' for precise entities, 'Semantic' for vague concepts, 'Hybrid' for complex needs."
    )
    action: Literal["Z", "S-Sparse", "S-Dense", "S-Hybrid", "M"] = Field(
        ...,
        description="The final RAG execution code. Z=No Ret, S-Sparse=BM25, S-Dense=Vector, S-Hybrid=Both, M=Multi-hop."
    )


# System Prompt
SYSTEM_PROMPT = """You are an expert RAG (Retrieval-Augmented Generation) Router Optimization Engine.
Your task is to analyze user queries to determine the optimal retrieval complexity and indexing strategy.

### OBJECTIVE
Analyze the **linguistic structure** and **logic** of the query to predict the necessary RAG pipeline configuration.
- **Do NOT** judge based on whether you personally know the answer. Even if you know the answer, if the question asks for a real-world fact, it requires retrieval.
- **Do NOT** answer the question.

### CLASSIFICATION RULES

#### Step 1: Determine Complexity (L0, L1, L2)
- **L0 (No Retrieval)**:
  - Non-factual queries, logic puzzles, code generation, translation, chitchat, or creative writing.
  - Queries containing NO specific real-world entities.
- **L1 (Single Hop)**:
  - Factual queries about a single entity or simple definition.
  - Requires looking up one specific document or fact.
- **L2 (Multi Hop)**:
  - Queries involving **Comparison** (e.g., "vs", "younger than", "difference").
  - **Multi-entity relations** (e.g., "A and B", "both").
  - **Nested Logic** (e.g., "The director of the movie [X]...", "The capital of the country where...").
  - **Temporal Constraints** (e.g., "first", "last", "before").

#### Step 2: Determine Index Strategy (For L1 & L2)
- **None**: For L0 only.
- **Lexical (Sparse)**:
  - Use when the query contains **Unique Identifiers**, **Full Names** (e.g., "Elon Musk"), **Specific Codes** (e.g., "Error 503"), or **Quoted Text**.
  - Keyword matching (BM25) is sufficient and precise.
- **Semantic (Dense)**:
  - Use when the query is **Descriptive**, **Abstract**, or uses **Synonyms** (e.g., "the angry green hero", "how to fix slow wifi").
  - Keywords might miss; vector search is needed to capture intent.
- **Hybrid**:
  - Use for **L2 (Complex)** queries (to ensure high recall).
  - Use for **L1** queries that are ambiguous (contain both specific entities and vague descriptions).

### MAPPING LOGIC (Action)
- If L0 -> Action: "Z"
- If L1 + Lexical -> Action: "S-Sparse"
- If L1 + Semantic -> Action: "S-Dense"
- If L1 + Hybrid -> Action: "S-Hybrid"
- If L2 -> Action: "M" (Always Multi-hop)

### EXAMPLES

Input: "Write a python script to reverse a list."
Output: {
  "reasoning": "The user is asking for code generation. This relies on internal logic and syntax knowledge, not external real-world facts.",
  "complexity_label": "L0",
  "index_strategy": "None",
  "action": "Z"
}

Input: "What is the atomic number of Gold?"
Output: {
  "reasoning": "The query asks for a specific attribute of a distinct entity 'Gold'. 'Atomic number' and 'Gold' are precise keywords. Lexical search is optimal.",
  "complexity_label": "L1",
  "index_strategy": "Lexical",
  "action": "S-Sparse"
}

Input: "How can I improve my sleep quality naturally?"
Output: {
  "reasoning": "The query is informational but uses general terms ('improve', 'sleep quality', 'naturally') rather than unique entities. Semantic search is required to match the intent.",
  "complexity_label": "L1",
  "index_strategy": "Semantic",
  "action": "S-Dense"
}

Input: "Who is older, Brad Pitt or Tom Cruise?"
Output: {
  "reasoning": "The query explicitly compares two distinct entities ('Brad Pitt', 'Tom Cruise') using the comparative adjective 'older'. This requires retrieving birth dates for both and comparing them.",
  "complexity_label": "L2",
  "index_strategy": "Hybrid",
  "action": "M"
}"""


class DataAugmentationPipeline:
    def __init__(
        self,
        api_key: str = None,
        base_url: str = None,
        model: str = "gpt-4o-mini",
        max_concurrent: int = 10,
        output_dir: str = "./training_data"
    ):
        """
        初始化数据增强管道

        Args:
            api_key: OpenAI API key (如果为 None,将从环境变量读取)
            base_url: API base URL (可选,用于自定义端点)
            model: 使用的模型名称
            max_concurrent: 最大并发请求数
            output_dir: 输出目录
        """
        self.client = AsyncOpenAI(
            api_key=api_key or os.getenv("OPENAI_API_KEY"),
            base_url=base_url
        )
        self.model = model
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.output_dir = Path(output_dir)

        # 创建输出目录，并提供友好的错误提示
        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            print(f"📁 Output directory: {self.output_dir.absolute()}")
        except OSError as e:
            print(f"❌ ERROR: Cannot create output directory: {self.output_dir}")
            print(f"   {e}")
            raise

    async def analyze_query(self, question: str, retry: int = 3) -> RagRoutingAnalysis:
        """
        使用 OpenAI Structured Outputs 分析单个查询

        Args:
            question: 查询文本
            retry: 重试次数

        Returns:
            RagRoutingAnalysis 对象
        """
        async with self.semaphore:
            for attempt in range(retry):
                try:
                    response = await self.client.beta.chat.completions.parse(
                        model=self.model,
                        messages=[
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": f"Analyze this query: {question}"}
                        ],
                        response_format=RagRoutingAnalysis,
                    )

                    return response.choices[0].message.parsed

                except Exception as e:
                    error_msg = str(e)

                    # 检查是否是速率限制错误
                    if "rate_limit" in error_msg.lower() or "429" in error_msg:
                        print(f"⚠️  WARNING: Rate limited (attempt {attempt + 1}/{retry})")
                        if attempt < retry - 1:
                            wait_time = 2 ** attempt * 5  # 速率限制时等待更长时间
                            print(f"   Waiting {wait_time}s before retry...")
                            await asyncio.sleep(wait_time)
                            continue

                    # 其他错误
                    if attempt == retry - 1:
                        print(f"\n❌ ERROR: Failed to analyze query after {retry} attempts")
                        print(f"   Query: '{question[:100]}...'")
                        print(f"   Error: {e}")
                        raise

                    # 正常重试
                    wait_time = 2 ** attempt
                    print(f"⚠️  WARNING: API call failed (attempt {attempt + 1}/{retry}), retrying in {wait_time}s...")
                    await asyncio.sleep(wait_time)

    async def process_single_item(self, item: Dict[str, Any], dataset_name: str) -> Dict[str, Any]:
        """
        处理单个数据项

        Args:
            item: 原始数据项
            dataset_name: 数据集名称（从文件路径提取）

        Returns:
            增强后的数据项
        """
        question = item.get("question_text", "")
        if not question:
            raise ValueError("Missing question_text in item")

        analysis = await self.analyze_query(question)

        # 构建训练数据格式
        # 使用 get() 方法安全访问字段，如果数据中没有 dataset 字段则使用传入的 dataset_name
        training_item = {
            "dataset": item.get("dataset", dataset_name),
            "question_id": item.get("question_id", "unknown"),
            "question_text": question,
            "reasoning": analysis.reasoning,
            "complexity_label": analysis.complexity_label,
            "index_strategy": analysis.index_strategy,
            "action": analysis.action,
            # 保留原始答案供参考
            "answers": item.get("answers_objects", []),
        }

        return training_item

    async def process_file(self, file_path: Path, dataset_name: str) -> List[Dict[str, Any]]:
        """
        处理单个 JSONL 文件

        Args:
            file_path: 输入文件路径
            dataset_name: 数据集名称（从文件路径提取）

        Returns:
            处理后的数据列表
        """
        # 读取数据，支持错误处理
        items = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:  # 跳过空行
                    continue
                try:
                    items.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"⚠️  WARNING: Skipping malformed JSON at {file_path.name}:{line_num}")
                    print(f"   Error: {e}")
                    continue

        if not items:
            print(f"❌ ERROR: No valid items found in {file_path}")
            return []

        print(f"\n📊 Processing {file_path.name}: {len(items)} items")

        # 并发处理所有数据项，传递 dataset_name
        tasks = [self.process_single_item(item, dataset_name) for item in items]
        results = []

        # 使用 tqdm 显示进度
        for coro in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc=f"Processing {file_path.name}"):
            try:
                result = await coro
                results.append(result)
            except Exception as e:
                # 尝试获取失败项的信息（如果可用）
                print(f"\n❌ ERROR: Failed to process an item")
                print(f"   Error: {e}")
                continue

        return results

    async def process_all_datasets(
        self,
        data_dir: str = "../../processed_data",
        datasets: List[str] = None
    ):
        """
        处理所有数据集

        Args:
            data_dir: 数据目录路径
            datasets: 要处理的数据集列表(如果为 None,处理所有数据集)
        """
        if datasets is None:
            datasets = ["2wikimultihopqa", "hotpotqa", "musique", "nq", "squad", "trivia"]

        # 验证数据目录是否存在
        data_path = Path(data_dir)
        if not data_path.exists():
            print(f"❌ ERROR: Data directory does not exist: {data_path.absolute()}")
            print(f"   Please check the 'data_dir' setting in config.json")
            return

        print(f"📂 Data directory: {data_path.absolute()}")
        print()

        all_training_data = []

        for dataset_name in datasets:
            dataset_path = data_path / dataset_name / "dev_500_subsampled.jsonl"

            if not dataset_path.exists():
                print(f"⚠️  WARNING: {dataset_path} does not exist, skipping...")
                continue

            try:
                results = await self.process_file(dataset_path, dataset_name)
                all_training_data.extend(results)

                # 保存单个数据集的结果
                dataset_output_file = self.output_dir / f"{dataset_name}_training.jsonl"
                with open(dataset_output_file, 'w', encoding='utf-8') as f:
                    for item in results:
                        f.write(json.dumps(item, ensure_ascii=False) + '\n')

                print(f"✅ Saved {len(results)} items to {dataset_output_file}")

            except Exception as e:
                print(f"❌ ERROR: Failed to process dataset {dataset_name}")
                print(f"   {e}")
                continue

        # 保存合并的训练数据
        combined_output_file = self.output_dir / "combined_training_data.jsonl"
        with open(combined_output_file, 'w', encoding='utf-8') as f:
            for item in all_training_data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')

        print()
        print("=" * 60)
        print("📊 Summary")
        print("=" * 60)
        print(f"Total items processed: {len(all_training_data)}")
        print(f"Combined training data: {combined_output_file}")
        print()

        # 统计标签分布
        self._print_label_distribution(all_training_data)

    def _print_label_distribution(self, data: List[Dict[str, Any]]):
        """打印标签分布统计"""
        from collections import Counter

        complexity_dist = Counter(item["complexity_label"] for item in data)
        strategy_dist = Counter(item["index_strategy"] for item in data)
        action_dist = Counter(item["action"] for item in data)

        print("=" * 60)
        print("🏷️  Label Distribution")
        print("=" * 60)
        print(f"Complexity Labels: {dict(complexity_dist)}")
        print(f"Index Strategies:  {dict(strategy_dist)}")
        print(f"Actions:           {dict(action_dist)}")


async def main():
    """主函数 - 从 config.json 加载配置并运行数据生成"""
    print("=" * 60)
    print("RAG 路由分类训练数据生成工具")
    print("=" * 60)
    print()

    # 1. 检查 config.json 是否存在
    config_file = Path("config.json")
    if not config_file.exists():
        print("❌ ERROR: config.json not found!")
        print()
        print("Please create config.json based on config.example.json:")
        print("  1. Copy the example: cp config.example.json config.json")
        print("  2. Edit config.json and set your API key and other settings")
        print()
        return

    # 2. 读取并验证 config.json
    print("📄 Loading configuration from config.json...")
    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            config = json.load(f)
    except json.JSONDecodeError as e:
        print(f"❌ ERROR: config.json is not valid JSON!")
        print(f"   {e}")
        print()
        print("Please check your config.json file for syntax errors.")
        return
    except Exception as e:
        print(f"❌ ERROR: Failed to read config.json: {e}")
        return

    # 3. 提取配置
    api_settings = config.get("api_settings", {})
    data_settings = config.get("data_settings", {})

    # 4. 验证必需字段
    api_key = api_settings.get("api_key") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ ERROR: No API key provided!")
        print()
        print("Please provide an API key using one of these methods:")
        print("  1. Set 'api_key' in config.json")
        print("  2. Set OPENAI_API_KEY environment variable")
        print()
        print("Example:")
        print('  export OPENAI_API_KEY="sk-..."  # Linux/Mac')
        print('  set OPENAI_API_KEY=sk-...       # Windows')
        print()
        return

    # 显示配置信息
    print("✅ Configuration loaded successfully!")
    print()
    print("Settings:")
    print(f"  Model: {api_settings.get('model', 'gpt-4o-mini')}")
    print(f"  Max Concurrent: {api_settings.get('max_concurrent', 10)}")
    print(f"  Data Directory: {data_settings.get('data_dir', '../../processed_data')}")
    print(f"  Output Directory: {data_settings.get('output_dir', './training_data')}")
    if data_settings.get('datasets'):
        print(f"  Datasets: {', '.join(data_settings.get('datasets'))}")
    else:
        print(f"  Datasets: All (2wikimultihopqa, hotpotqa, musique, nq, squad, trivia)")
    print()

    # 5. 创建 pipeline 并运行
    try:
        pipeline = DataAugmentationPipeline(
            api_key=api_key,
            base_url=api_settings.get("base_url"),
            model=api_settings.get("model", "gpt-4o-mini"),
            max_concurrent=api_settings.get("max_concurrent", 10),
            output_dir=data_settings.get("output_dir", "./training_data")
        )

        await pipeline.process_all_datasets(
            data_dir=data_settings.get("data_dir", "../../processed_data"),
            datasets=data_settings.get("datasets")
        )

        print()
        print("=" * 60)
        print("✅ Data generation completed successfully!")
        print("=" * 60)

    except Exception as e:
        print()
        print("=" * 60)
        print(f"❌ ERROR: Data generation failed!")
        print(f"   {e}")
        print("=" * 60)
        raise


if __name__ == "__main__":
    asyncio.run(main())
