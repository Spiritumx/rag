"""
验证生成的训练数据质量
- 读取已生成的训练数据
- 随机抽样验证分类质量
- 生成质量报告
"""

import asyncio
import json
import os
import random
from pathlib import Path
from collections import Counter
from typing import List, Dict, Any
from generate_training_data import DataAugmentationPipeline


def load_config():
    """加载 config.json 配置"""
    config_file = Path("config.json")

    if not config_file.exists():
        print("❌ ERROR: config.json not found")
        print("Please copy config.example.json to config.json")
        return None

    with open(config_file, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_training_data(file_path: Path) -> List[Dict[str, Any]]:
    """加载训练数据文件"""
    if not file_path.exists():
        return []

    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


async def validate_sample(pipeline: DataAugmentationPipeline, item: Dict[str, Any]) -> Dict[str, Any]:
    """
    验证单个样本 - 重新分类并与原标签对比

    Returns:
        包含对比结果的字典
    """
    question = item["question_text"]

    # 重新分类
    new_analysis = await pipeline.analyze_query(question)

    # 对比结果
    return {
        "question": question[:100],  # 截断显示
        "dataset": item.get("dataset", "unknown"),
        "original": {
            "complexity": item["complexity_label"],
            "strategy": item["index_strategy"],
            "action": item["action"],
            "reasoning": item["reasoning"][:150]  # 截断显示
        },
        "new": {
            "complexity": new_analysis.complexity_label,
            "strategy": new_analysis.index_strategy,
            "action": new_analysis.action,
            "reasoning": new_analysis.reasoning[:150]
        },
        "match": {
            "complexity": item["complexity_label"] == new_analysis.complexity_label,
            "strategy": item["index_strategy"] == new_analysis.index_strategy,
            "action": item["action"] == new_analysis.action
        }
    }


def print_validation_result(result: Dict[str, Any], index: int):
    """打印单个验证结果"""
    print(f"\n{'='*80}")
    print(f"Sample {index} - Dataset: {result['dataset']}")
    print(f"{'='*80}")
    print(f"Question: {result['question']}...")
    print()

    # 对比表格
    print(f"{'Metric':<20} {'Original':<20} {'New':<20} {'Match':<10}")
    print(f"{'-'*70}")

    complexity_match = "✅ Yes" if result['match']['complexity'] else "❌ No"
    strategy_match = "✅ Yes" if result['match']['strategy'] else "❌ No"
    action_match = "✅ Yes" if result['match']['action'] else "❌ No"

    print(f"{'Complexity':<20} {result['original']['complexity']:<20} {result['new']['complexity']:<20} {complexity_match:<10}")
    print(f"{'Strategy':<20} {result['original']['strategy']:<20} {result['new']['strategy']:<20} {strategy_match:<10}")
    print(f"{'Action':<20} {result['original']['action']:<20} {result['new']['action']:<20} {action_match:<10}")

    # 显示推理对比（如果不匹配）
    if not all(result['match'].values()):
        print(f"\n{'Original Reasoning':<20}: {result['original']['reasoning']}...")
        print(f"{'New Reasoning':<20}: {result['new']['reasoning']}...")


def print_summary_report(results: List[Dict[str, Any]]):
    """打印汇总报告"""
    print("\n" + "="*80)
    print("📊 VALIDATION SUMMARY REPORT")
    print("="*80)

    total = len(results)

    # 计算一致性
    complexity_matches = sum(1 for r in results if r['match']['complexity'])
    strategy_matches = sum(1 for r in results if r['match']['strategy'])
    action_matches = sum(1 for r in results if r['match']['action'])
    all_match = sum(1 for r in results if all(r['match'].values()))

    print(f"\nTotal samples validated: {total}")
    print(f"\nConsistency Rates:")
    print(f"  Complexity Label:  {complexity_matches}/{total} ({complexity_matches/total*100:.1f}%)")
    print(f"  Index Strategy:    {strategy_matches}/{total} ({strategy_matches/total*100:.1f}%)")
    print(f"  Action:            {action_matches}/{total} ({action_matches/total*100:.1f}%)")
    print(f"  All Match:         {all_match}/{total} ({all_match/total*100:.1f}%)")

    # 不匹配的样本统计
    mismatches = [r for r in results if not all(r['match'].values())]
    if mismatches:
        print(f"\n⚠️  Mismatched samples: {len(mismatches)}")
        print("\nMismatch breakdown:")

        complexity_mismatches = [r for r in results if not r['match']['complexity']]
        strategy_mismatches = [r for r in results if not r['match']['strategy']]
        action_mismatches = [r for r in results if not r['match']['action']]

        if complexity_mismatches:
            print(f"  Complexity: {len(complexity_mismatches)} samples")
            transitions = Counter(
                f"{r['original']['complexity']} → {r['new']['complexity']}"
                for r in complexity_mismatches
            )
            for transition, count in transitions.most_common(3):
                print(f"    {transition}: {count}")

        if strategy_mismatches:
            print(f"  Strategy: {len(strategy_mismatches)} samples")
            transitions = Counter(
                f"{r['original']['strategy']} → {r['new']['strategy']}"
                for r in strategy_mismatches
            )
            for transition, count in transitions.most_common(3):
                print(f"    {transition}: {count}")

        if action_mismatches:
            print(f"  Action: {len(action_mismatches)} samples")
            transitions = Counter(
                f"{r['original']['action']} → {r['new']['action']}"
                for r in action_mismatches
            )
            for transition, count in transitions.most_common(3):
                print(f"    {transition}: {count}")

    # 评估建议
    print("\n" + "="*80)
    print("💡 RECOMMENDATIONS")
    print("="*80)

    overall_consistency = all_match / total * 100

    if overall_consistency >= 90:
        print("✅ Excellent! Data quality is very high (≥90% consistency)")
        print("   The generated training data is reliable for model training.")
    elif overall_consistency >= 75:
        print("⚠️  Good. Data quality is acceptable (75-90% consistency)")
        print("   Most labels are consistent, minor variations are expected.")
        print("   Consider reviewing mismatched samples if critical.")
    elif overall_consistency >= 60:
        print("⚠️  Fair. Data quality has some issues (60-75% consistency)")
        print("   Recommendations:")
        print("   1. Review the System Prompt for clarity")
        print("   2. Check if mismatches follow a pattern")
        print("   3. Consider using gpt-4o for better consistency")
    else:
        print("❌ Poor. Data quality needs improvement (<60% consistency)")
        print("   Critical actions needed:")
        print("   1. Review and refine the System Prompt")
        print("   2. Use gpt-4o instead of gpt-4o-mini")
        print("   3. Add more examples to the prompt")
        print("   4. Consider regenerating the training data")


async def main():
    """主函数"""
    print("="*80)
    print("🔍 Training Data Quality Validation Tool")
    print("="*80)
    print()

    # 加载配置
    config = load_config()
    if not config:
        return

    api_settings = config.get("api_settings", {})
    data_settings = config.get("data_settings", {})

    # 检查 API key
    api_key = api_settings.get("api_key") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ ERROR: No API key provided!")
        print("Please set 'api_key' in config.json or OPENAI_API_KEY env var")
        return

    # 获取输出目录
    output_dir = Path(data_settings.get("output_dir", "./training_data"))

    if not output_dir.exists():
        print(f"❌ ERROR: Training data directory not found: {output_dir}")
        print("Please run generate_training_data.py first to generate data")
        return

    # 查找所有生成的训练数据文件
    training_files = list(output_dir.glob("*_training.jsonl"))

    if not training_files:
        print(f"❌ ERROR: No training data files found in {output_dir}")
        print("Please run generate_training_data.py first")
        return

    print(f"📁 Found {len(training_files)} training data files")
    print()

    # 让用户选择验证模式
    print("Select validation mode:")
    print("  1. Quick validation (10 random samples)")
    print("  2. Standard validation (50 random samples)")
    print("  3. Thorough validation (100 random samples)")
    print("  4. Custom number of samples")
    print("  5. Validate specific dataset")
    print()

    choice = input("Enter choice (1-5, default=1): ").strip() or "1"

    # 加载数据
    all_data = []
    datasets_to_validate = None

    if choice == "5":
        print("\nAvailable datasets:")
        for i, f in enumerate(training_files, 1):
            dataset_name = f.stem.replace("_training", "")
            print(f"  {i}. {dataset_name}")

        dataset_choice = input("\nEnter dataset number: ").strip()
        try:
            selected_file = training_files[int(dataset_choice) - 1]
            all_data = load_training_data(selected_file)
            print(f"\n✅ Loaded {len(all_data)} samples from {selected_file.name}")
        except (ValueError, IndexError):
            print("❌ Invalid choice")
            return
    else:
        # 加载所有数据
        for f in training_files:
            data = load_training_data(f)
            all_data.extend(data)
        print(f"✅ Loaded {len(all_data)} total samples from all datasets")

    if not all_data:
        print("❌ No data to validate")
        return

    # 确定验证样本数
    if choice == "1":
        num_samples = min(10, len(all_data))
    elif choice == "2":
        num_samples = min(50, len(all_data))
    elif choice == "3":
        num_samples = min(100, len(all_data))
    elif choice == "4":
        num_input = input(f"Enter number of samples (max {len(all_data)}): ").strip()
        try:
            num_samples = min(int(num_input), len(all_data))
        except ValueError:
            print("❌ Invalid number")
            return
    else:
        num_samples = min(50, len(all_data))

    # 随机抽样
    samples = random.sample(all_data, num_samples)

    print(f"\n🎲 Randomly selected {num_samples} samples for validation")
    print(f"⏳ This may take a few minutes...")
    print()

    # 创建 pipeline
    pipeline = DataAugmentationPipeline(
        api_key=api_key,
        base_url=api_settings.get("base_url"),
        model=api_settings.get("model", "gpt-4o-mini"),
        max_concurrent=min(api_settings.get("max_concurrent", 5), 5),  # 限制并发
        output_dir=output_dir
    )

    # 验证样本
    results = []
    for i, sample in enumerate(samples, 1):
        print(f"Validating sample {i}/{num_samples}...", end='\r')
        try:
            result = await validate_sample(pipeline, sample)
            results.append(result)
        except Exception as e:
            print(f"\n⚠️  Error validating sample {i}: {e}")
            continue

    print()  # 换行

    # 显示详细结果（前5个和所有不匹配的）
    print("\n" + "="*80)
    print("📋 DETAILED VALIDATION RESULTS")
    print("="*80)

    # 显示前5个样本
    print("\n--- First 5 samples ---")
    for i, result in enumerate(results[:5], 1):
        print_validation_result(result, i)

    # 显示所有不匹配的样本
    mismatches = [r for r in results if not all(r['match'].values())]
    if mismatches and len(mismatches) > 0:
        print(f"\n--- All {len(mismatches)} mismatched samples ---")
        for i, result in enumerate(mismatches, 1):
            print_validation_result(result, i)

    # 打印汇总报告
    print_summary_report(results)

    # 保存验证报告
    report_file = output_dir / "validation_report.json"
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump({
            "total_samples": len(results),
            "consistency": {
                "complexity": sum(1 for r in results if r['match']['complexity']) / len(results),
                "strategy": sum(1 for r in results if r['match']['strategy']) / len(results),
                "action": sum(1 for r in results if r['match']['action']) / len(results),
                "overall": sum(1 for r in results if all(r['match'].values())) / len(results)
            },
            "results": results
        }, f, indent=2, ensure_ascii=False)

    print(f"\n💾 Validation report saved to: {report_file}")
    print()


if __name__ == "__main__":
    asyncio.run(main())
