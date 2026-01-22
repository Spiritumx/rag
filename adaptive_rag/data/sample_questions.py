"""
Adaptive-RAG 数据采样脚本

从 6 个数据集的 dev_500_subsampled.jsonl 中采样，共 1000 条
每个数据集采样约 167 条

Usage:
    python -m adaptive_rag.data.sample_questions
    python -m adaptive_rag.data.sample_questions --total 500
"""

import os
import sys
import json
import random
import argparse
from pathlib import Path

import yaml

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def load_config(config_path: str = None) -> dict:
    """加载配置文件"""
    if config_path is None:
        config_path = PROJECT_ROOT / "adaptive_rag" / "config.yaml"

    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def load_jsonl(file_path: str) -> list:
    """加载 JSONL 文件"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def save_jsonl(data: list, file_path: str):
    """保存 JSONL 文件"""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')


def sample_from_datasets(
    config: dict,
    total_samples: int = 1000,
    seed: int = 42
) -> list:
    """
    从多个数据集中均匀采样

    Args:
        config: 配置字典
        total_samples: 总采样数
        seed: 随机种子

    Returns:
        采样后的数据列表
    """
    random.seed(seed)

    datasets = config['datasets']
    base_path = PROJECT_ROOT / config['data']['base_path']
    dev_file = config['data']['dev_file']

    # 计算每个数据集的采样数
    samples_per_dataset = total_samples // len(datasets)
    remainder = total_samples % len(datasets)

    all_samples = []
    stats = {}

    print(f"\n{'='*60}")
    print(f"Adaptive-RAG 数据采样")
    print(f"{'='*60}")
    print(f"总采样数: {total_samples}")
    print(f"数据集数量: {len(datasets)}")
    print(f"每个数据集基础采样数: {samples_per_dataset}")
    print(f"{'='*60}\n")

    for i, dataset_name in enumerate(datasets):
        # 处理余数，前几个数据集多采样 1 条
        n_samples = samples_per_dataset + (1 if i < remainder else 0)

        # 构建文件路径
        file_path = base_path / dataset_name / dev_file

        if not file_path.exists():
            print(f"  [Warning] 文件不存在: {file_path}")
            stats[dataset_name] = {'available': 0, 'sampled': 0}
            continue

        # 加载数据
        data = load_jsonl(str(file_path))

        # 采样
        if len(data) <= n_samples:
            sampled = data
        else:
            sampled = random.sample(data, n_samples)

        # 添加数据集标记
        for item in sampled:
            item['source_dataset'] = dataset_name

        all_samples.extend(sampled)
        stats[dataset_name] = {
            'available': len(data),
            'sampled': len(sampled)
        }

        print(f"  {dataset_name:20s}: {len(data):4d} 可用 -> {len(sampled):4d} 采样")

    # 打乱顺序
    random.shuffle(all_samples)

    print(f"\n{'='*60}")
    print(f"采样完成: 共 {len(all_samples)} 条")
    print(f"{'='*60}")

    return all_samples, stats


def main():
    parser = argparse.ArgumentParser(description="Adaptive-RAG 数据采样")
    parser.add_argument('--config', type=str, default=None,
                       help='配置文件路径')
    parser.add_argument('--total', type=int, default=1000,
                       help='总采样数 (默认: 1000)')
    parser.add_argument('--seed', type=int, default=42,
                       help='随机种子 (默认: 42)')
    parser.add_argument('--output', type=str, default=None,
                       help='输出文件路径')

    args = parser.parse_args()

    # 加载配置
    config = load_config(args.config)

    # 采样
    samples, stats = sample_from_datasets(
        config=config,
        total_samples=args.total,
        seed=args.seed
    )

    # 确定输出路径
    if args.output:
        output_path = args.output
    else:
        output_path = PROJECT_ROOT / config['data']['sampled_questions_path']

    # 保存结果
    save_jsonl(samples, str(output_path))
    print(f"\n输出文件: {output_path}")

    # 保存统计信息
    stats_path = str(output_path).replace('.jsonl', '_stats.json')
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump({
            'total_samples': len(samples),
            'seed': args.seed,
            'by_dataset': stats
        }, f, indent=2, ensure_ascii=False)
    print(f"统计信息: {stats_path}")


if __name__ == '__main__':
    main()
