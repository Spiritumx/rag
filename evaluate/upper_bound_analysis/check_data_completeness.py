#!/usr/bin/env python3
"""
检查 predictions_contexts.json 数据完整性

诊断为什么某些数据集会出现 qid 缺失的问题
"""

import json
import os
import sys


def check_data_completeness(dataset_name, data_dir="processed_data", predictions_dir="evaluate/outputs/stage2_predictions"):
    """
    检查数据完整性

    Args:
        dataset_name: 数据集名称
        data_dir: 测试数据目录
        predictions_dir: 预测结果目录
    """
    print("="*70)
    print(f"数据完整性检查: {dataset_name}")
    print("="*70)

    # 1. 加载测试数据
    test_file = os.path.join(data_dir, dataset_name, "test_subsampled.jsonl")
    if not os.path.exists(test_file):
        print(f"✗ 测试文件不存在: {test_file}")
        return False

    test_qids = []
    with open(test_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                test_qids.append(item['question_id'])

    print(f"✓ 测试数据: {len(test_qids)} 个问题")

    # 2. 加载 predictions
    predictions_file = os.path.join(predictions_dir, f"{dataset_name}_predictions.json")
    if os.path.exists(predictions_file):
        with open(predictions_file, 'r', encoding='utf-8') as f:
            predictions = json.load(f)
        print(f"✓ Predictions: {len(predictions)} 个问题")
    else:
        print(f"⚠ Predictions 文件不存在: {predictions_file}")
        predictions = {}

    # 3. 加载 contexts
    contexts_file = os.path.join(predictions_dir, f"{dataset_name}_predictions_contexts.json")
    if not os.path.exists(contexts_file):
        print(f"✗ Contexts 文件不存在: {contexts_file}")
        return False

    with open(contexts_file, 'r', encoding='utf-8') as f:
        contexts = json.load(f)
    print(f"✓ Contexts: {len(contexts)} 个问题")

    # 4. 比对缺失的 qid
    test_qids_set = set(test_qids)
    predictions_qids = set(predictions.keys()) if predictions else set()
    contexts_qids = set(contexts.keys())

    print(f"\n{'='*70}")
    print("数据对比：")
    print("="*70)

    # 测试数据 vs Predictions
    if predictions:
        missing_predictions = test_qids_set - predictions_qids
        extra_predictions = predictions_qids - test_qids_set

        if missing_predictions:
            print(f"✗ Predictions 中缺失 {len(missing_predictions)} 个问题")
            print(f"  缺失的 QID (前5个): {list(missing_predictions)[:5]}")
        else:
            print(f"✓ Predictions 完整")

        if extra_predictions:
            print(f"⚠ Predictions 中有 {len(extra_predictions)} 个多余问题")

    # 测试数据 vs Contexts
    missing_contexts = test_qids_set - contexts_qids
    extra_contexts = contexts_qids - test_qids_set

    if missing_contexts:
        print(f"\n✗ Contexts 中缺失 {len(missing_contexts)} 个问题")
        print(f"  缺失比例: {len(missing_contexts)/len(test_qids)*100:.2f}%")
        print(f"  缺失的 QID (前10个):")
        for qid in list(missing_contexts)[:10]:
            print(f"    - {qid}")

        # 检查这些问题在 predictions 中是否存在
        if predictions:
            missing_in_predictions = missing_contexts & predictions_qids
            if missing_in_predictions:
                print(f"\n  其中 {len(missing_in_predictions)} 个在 predictions 中存在")
                print(f"  -> 说明这些问题生成了答案但没有保存 contexts")
    else:
        print(f"\n✓ Contexts 完整")

    if extra_contexts:
        print(f"\n⚠ Contexts 中有 {len(extra_contexts)} 个多余问题")

    # 5. 检查 contexts 格式
    print(f"\n{'='*70}")
    print("Contexts 格式检查：")
    print("="*70)

    format_issues = 0
    empty_contexts = 0
    sample_checked = 0

    for qid in list(contexts_qids)[:20]:  # 检查前20个
        ctx = contexts[qid]
        sample_checked += 1

        if isinstance(ctx, dict):
            if 'titles' in ctx or 'paras' in ctx:
                format_issues += 1
        elif isinstance(ctx, list):
            if len(ctx) == 0:
                empty_contexts += 1
        else:
            format_issues += 1

    print(f"检查样本: {sample_checked} 个")
    if format_issues > 0:
        print(f"⚠ 发现 {format_issues} 个旧格式（字典）")
        print(f"  -> 测试脚本会自动转换")
    else:
        print(f"✓ 格式正确（列表）")

    if empty_contexts > 0:
        print(f"⚠ 发现 {empty_contexts} 个空列表")

    # 6. 总结
    print(f"\n{'='*70}")
    print("总结：")
    print("="*70)

    if not missing_contexts and not missing_predictions:
        print("✓ 数据完整，可以正常运行测试")
        return True
    else:
        print("✗ 数据不完整，建议操作：")
        if missing_contexts:
            print(f"\n1. 重新运行 Stage2 生成完整数据：")
            print(f"   python evaluate/stage2_generate.py --datasets {dataset_name}")
        if missing_predictions:
            print(f"\n2. 或者只测试已有的 {len(contexts_qids)} 个问题")
            print(f"   （修改 test_subsampled.jsonl 只保留已有的 qid）")
        return False


def main():
    import argparse

    parser = argparse.ArgumentParser(description="检查数据完整性")
    parser.add_argument('--dataset', required=True, help='数据集名称')
    parser.add_argument('--data-dir', default='processed_data', help='测试数据目录')
    parser.add_argument('--predictions-dir',
                       default='evaluate/outputs/stage2_predictions',
                       help='预测结果目录')

    args = parser.parse_args()

    success = check_data_completeness(
        args.dataset,
        args.data_dir,
        args.predictions_dir
    )

    if not success:
        sys.exit(1)


if __name__ == '__main__':
    main()
