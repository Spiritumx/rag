"""
分析生成数据的标签分布 - 验证数据集类型弱标签的效果
"""
import json
from pathlib import Path
from collections import Counter

def analyze_distribution():
    """分析不同数据集的标签分布"""

    training_dir = Path("./training_data")
    combined_file = training_dir / "combined_training_data.jsonl"

    if not combined_file.exists():
        print("❌ ERROR: combined_training_data.jsonl not found!")
        print("   Please run generate_training_data.py first")
        return

    # 读取所有数据
    data = []
    with open(combined_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))

    print("=" * 70)
    print("数据集标签分布分析")
    print("=" * 70)
    print()

    # 数据集分类
    single_hop_datasets = ['nq', 'squad', 'trivia']
    multi_hop_datasets = ['2wikimultihopqa', 'hotpotqa', 'musique']

    # 统计各数据集的标签分布
    print("📊 各数据集的复杂度标签分布:")
    print("-" * 70)

    for dataset_type, datasets in [
        ("单跳数据集", single_hop_datasets),
        ("多跳数据集", multi_hop_datasets)
    ]:
        print(f"\n【{dataset_type}】")

        for dataset_name in datasets:
            dataset_data = [item for item in data if item.get('dataset') == dataset_name]

            if not dataset_data:
                print(f"  {dataset_name:20} - No data")
                continue

            complexity_counter = Counter(item['complexity_label'] for item in dataset_data)
            total = len(dataset_data)

            # 计算百分比
            l0_pct = complexity_counter.get('L0', 0) / total * 100
            l1_pct = complexity_counter.get('L1', 0) / total * 100
            l2_pct = complexity_counter.get('L2', 0) / total * 100

            print(f"  {dataset_name:20} (n={total:3d}): "
                  f"L0={l0_pct:5.1f}% | L1={l1_pct:5.1f}% | L2={l2_pct:5.1f}%")

    # 汇总统计
    print("\n" + "=" * 70)
    print("📈 汇总统计:")
    print("-" * 70)

    # 单跳数据集汇总
    single_hop_data = [item for item in data if item.get('dataset') in single_hop_datasets]
    if single_hop_data:
        complexity_counter = Counter(item['complexity_label'] for item in single_hop_data)
        total = len(single_hop_data)
        print(f"\n单跳数据集总计 (n={total}):")
        for label in ['L0', 'L1', 'L2']:
            count = complexity_counter.get(label, 0)
            pct = count / total * 100
            print(f"  {label}: {count:4d} ({pct:5.1f}%)")

    # 多跳数据集汇总
    multi_hop_data = [item for item in data if item.get('dataset') in multi_hop_datasets]
    if multi_hop_data:
        complexity_counter = Counter(item['complexity_label'] for item in multi_hop_data)
        total = len(multi_hop_data)
        print(f"\n多跳数据集总计 (n={total}):")
        for label in ['L0', 'L1', 'L2']:
            count = complexity_counter.get(label, 0)
            pct = count / total * 100
            print(f"  {label}: {count:4d} ({pct:5.1f}%)")

    # Action 分布
    print("\n" + "=" * 70)
    print("🎯 Action 标签分布:")
    print("-" * 70)

    action_counter = Counter(item['action'] for item in data)
    total = len(data)
    for action, count in action_counter.most_common():
        pct = count / total * 100
        print(f"  {action:15} {count:4d} ({pct:5.1f}%)")

    # 索引策略分布
    print("\n" + "=" * 70)
    print("🔍 Index Strategy 分布:")
    print("-" * 70)

    strategy_counter = Counter(item['index_strategy'] for item in data)
    for strategy, count in strategy_counter.most_common():
        pct = count / total * 100
        print(f"  {strategy:15} {count:4d} ({pct:5.1f}%)")

    # 分析结论
    print("\n" + "=" * 70)
    print("💡 分析结论:")
    print("-" * 70)

    if single_hop_data:
        l1_ratio = complexity_counter.get('L1', 0) / len(single_hop_data)
        if l1_ratio > 0.7:
            print("  ✅ 单跳数据集主要被标记为 L1，弱标签效果良好")
        elif l1_ratio > 0.5:
            print("  ⚠️  单跳数据集 L1 占比适中，可能存在一些复杂问题")
        else:
            print("  ❌ 单跳数据集 L1 占比较低，请检查数据质量或弱标签设置")

    if multi_hop_data:
        multi_hop_counter = Counter(item['complexity_label'] for item in multi_hop_data)
        l2_ratio = multi_hop_counter.get('L2', 0) / len(multi_hop_data)
        if l2_ratio > 0.7:
            print("  ✅ 多跳数据集主要被标记为 L2，弱标签效果良好")
        elif l2_ratio > 0.5:
            print("  ⚠️  多跳数据集 L2 占比适中，符合预期")
        else:
            print("  ⚠️  多跳数据集 L2 占比较低，可能数据集本身包含简单问题")

    print("\n" + "=" * 70)

if __name__ == "__main__":
    analyze_distribution()
