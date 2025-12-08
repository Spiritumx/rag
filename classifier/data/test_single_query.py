"""
测试脚本 - 用于测试单个查询的分类效果
"""

import asyncio
import os
from generate_training_data import DataAugmentationPipeline, RagRoutingAnalysis


async def test_single_query(question: str):
    """
    测试单个查询

    Args:
        question: 要测试的查询文本
    """
    pipeline = DataAugmentationPipeline(
        api_key=os.getenv("OPENAI_API_KEY"),
        model="gpt-4o-mini",
        max_concurrent=1
    )

    print(f"Query: {question}\n")
    print("Analyzing...")

    try:
        analysis = await pipeline.analyze_query(question)

        print("\n=== Analysis Result ===")
        print(f"Reasoning: {analysis.reasoning}\n")
        print(f"Complexity Label: {analysis.complexity_label}")
        print(f"Index Strategy: {analysis.index_strategy}")
        print(f"Action: {analysis.action}")

    except Exception as e:
        print(f"Error: {e}")


async def test_multiple_queries():
    """测试多个示例查询"""
    test_queries = [
        # L1 - 单跳事实性
        "Who is the president of the United States?",
        "What is the capital of France?",

        # L2 - 多跳复杂
        "Which movie starring Tom Hanks won the Oscar for Best Picture?",
        "What is the population of the capital of Japan?",

        # L0 - 非事实性/逻辑
        "How do I sort a list in Python?",
        "Explain the concept of recursion",
        "What is 2 + 2?",
    ]

    pipeline = DataAugmentationPipeline(
        api_key=os.getenv("OPENAI_API_KEY"),
        model="gpt-4o-mini",
        max_concurrent=3
    )

    print("=== Testing Multiple Queries ===\n")

    for i, question in enumerate(test_queries, 1):
        print(f"\n{'='*60}")
        print(f"Query {i}: {question}")
        print('='*60)

        try:
            analysis = await pipeline.analyze_query(question)

            print(f"\nReasoning: {analysis.reasoning}")
            print(f"\nComplexity: {analysis.complexity_label}")
            print(f"Strategy: {analysis.index_strategy}")
            print(f"Action: {analysis.action}")

        except Exception as e:
            print(f"Error: {e}")

        await asyncio.sleep(0.5)  # 避免速率限制


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        # 从命令行参数获取查询
        query = " ".join(sys.argv[1:])
        asyncio.run(test_single_query(query))
    else:
        # 运行多个测试查询
        asyncio.run(test_multiple_queries())
