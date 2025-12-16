#!/usr/bin/env python3
"""
分析检索质量：对比检索到的上下文与数据集中的标准上下文
"""

import os
import sys
import json
import argparse
from collections import defaultdict
from typing import List, Dict, Set

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluate.utils.config_loader import ConfigLoader
from evaluate.utils.data_loader import DataLoader
from evaluate.utils.result_manager import ResultManager


class RetrievalQualityAnalyzer:
    """分析检索上下文与标准上下文的质量对比"""

    def __init__(self, config):
        self.config = config
        self.data_loader = DataLoader(config)
        self.result_manager = ResultManager(config)

    def normalize_title(self, title: str) -> str:
        """标准化标题以便比较"""
        return title.strip().lower()

    def extract_gold_titles(self, test_item: dict) -> Set[str]:
        """提取数据集中的标准上下文标题"""
        gold_titles = set()

        # 检查 contexts 字段
        if 'contexts' in test_item:
            for ctx in test_item['contexts']:
                if isinstance(ctx, dict) and 'title' in ctx:
                    gold_titles.add(self.normalize_title(ctx['title']))
                elif isinstance(ctx, dict) and 'paragraph' in ctx:
                    # 有些数据集可能有不同的结构
                    title = ctx.get('title', '')
                    if title:
                        gold_titles.add(self.normalize_title(title))

        # 检查 supporting_facts 字段（HotpotQA 等数据集）
        if 'supporting_facts' in test_item:
            for fact in test_item['supporting_facts']:
                if isinstance(fact, list) and len(fact) > 0:
                    gold_titles.add(self.normalize_title(fact[0]))
                elif isinstance(fact, dict) and 'title' in fact:
                    gold_titles.add(self.normalize_title(fact['title']))

        return gold_titles

    def extract_retrieved_titles(self, retrieved_contexts: List[dict]) -> Set[str]:
        """提取检索到的上下文标题"""
        retrieved_titles = set()

        for ctx in retrieved_contexts:
            if isinstance(ctx, dict):
                # 尝试不同的标题字段
                title = ctx.get('title') or ctx.get('wikipedia_title') or ctx.get('doc_title', '')
                if title:
                    retrieved_titles.add(self.normalize_title(title))

        return retrieved_titles

    def calculate_retrieval_metrics(self, gold_titles: Set[str], retrieved_titles: Set[str]) -> dict:
        """计算检索指标"""
        if not gold_titles:
            # 如果没有标准答案，无法计算
            return {
                'precision': None,
                'recall': None,
                'f1': None,
                'hit': None
            }

        # 计算交集
        hits = gold_titles & retrieved_titles

        # Precision: 检索到的上下文中有多少是相关的
        precision = len(hits) / len(retrieved_titles) if retrieved_titles else 0

        # Recall: 标准上下文中有多少被检索到了
        recall = len(hits) / len(gold_titles) if gold_titles else 0

        # F1
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        # Hit@K: 是否至少检索到一个标准上下文
        hit = 1 if len(hits) > 0 else 0

        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'hit': hit,
            'num_gold': len(gold_titles),
            'num_retrieved': len(retrieved_titles),
            'num_hits': len(hits)
        }

    def analyze_dataset(self, dataset_name: str, sample_size: int = None) -> dict:
        """分析一个数据集的检索质量"""
        print(f"\n{'='*80}")
        print(f"分析数据集: {dataset_name}")
        print(f"{'='*80}")

        # 加载数据
        test_data = self.data_loader.load_test_data(dataset_name)
        test_data_map = {item['question_id']: item for item in test_data}

        # 加载检索到的上下文
        contexts_file = self.result_manager.get_stage2_output_path(dataset_name).replace('.json', '_contexts.json')

        if not os.path.exists(contexts_file):
            print(f"  警告: 未找到检索上下文文件: {contexts_file}")
            print(f"  请先运行 Stage 2 生成")
            return None

        with open(contexts_file, 'r', encoding='utf-8') as f:
            retrieved_contexts_map = json.load(f)

        # 统计数据
        stats = {
            'total': 0,
            'with_gold': 0,
            'with_retrieval': 0,
            'total_precision': 0.0,
            'total_recall': 0.0,
            'total_f1': 0.0,
            'total_hits': 0,
            'perfect_retrieval': 0,  # Recall = 1.0
            'no_hits': 0,  # Hit = 0
            'samples': []
        }

        # 分析每个问题
        question_ids = list(test_data_map.keys())
        if sample_size:
            import random
            question_ids = random.sample(question_ids, min(sample_size, len(question_ids)))

        for qid in question_ids:
            if qid not in test_data_map:
                continue

            test_item = test_data_map[qid]
            gold_titles = self.extract_gold_titles(test_item)

            # 检查是否有检索结果
            if qid not in retrieved_contexts_map:
                continue

            retrieved_contexts = retrieved_contexts_map[qid]
            retrieved_titles = self.extract_retrieved_titles(retrieved_contexts)

            # 计算指标
            metrics = self.calculate_retrieval_metrics(gold_titles, retrieved_titles)

            if metrics['precision'] is not None:
                stats['total'] += 1
                stats['with_gold'] += 1
                stats['with_retrieval'] += 1

                stats['total_precision'] += metrics['precision']
                stats['total_recall'] += metrics['recall']
                stats['total_f1'] += metrics['f1']
                stats['total_hits'] += metrics['hit']

                if metrics['recall'] == 1.0:
                    stats['perfect_retrieval'] += 1

                if metrics['hit'] == 0:
                    stats['no_hits'] += 1

                # 保存样本
                stats['samples'].append({
                    'qid': qid,
                    'question': test_item.get('question_text', test_item.get('question', '')),
                    'gold_titles': list(gold_titles),
                    'retrieved_titles': list(retrieved_titles),
                    'metrics': metrics
                })

        # 计算平均值
        if stats['total'] > 0:
            stats['avg_precision'] = stats['total_precision'] / stats['total']
            stats['avg_recall'] = stats['total_recall'] / stats['total']
            stats['avg_f1'] = stats['total_f1'] / stats['total']
            stats['hit_rate'] = stats['total_hits'] / stats['total']
            stats['perfect_rate'] = stats['perfect_retrieval'] / stats['total']
            stats['no_hit_rate'] = stats['no_hits'] / stats['total']
        else:
            stats['avg_precision'] = 0
            stats['avg_recall'] = 0
            stats['avg_f1'] = 0
            stats['hit_rate'] = 0
            stats['perfect_rate'] = 0
            stats['no_hit_rate'] = 0

        return stats

    def print_summary(self, dataset_name: str, stats: dict):
        """打印统计摘要"""
        print(f"\n检索质量分析报告 - {dataset_name}")
        print("=" * 80)
        print(f"\n总样本数: {stats['total']}")
        print(f"  - 有标准上下文的样本: {stats['with_gold']}")
        print(f"  - 有检索结果的样本: {stats['with_retrieval']}")

        print(f"\n平均检索指标:")
        print(f"  Precision (精确率): {stats['avg_precision']:.4f}")
        print(f"    含义: 检索到的上下文中有多少是标准上下文")
        print(f"  Recall (召回率):    {stats['avg_recall']:.4f}")
        print(f"    含义: 标准上下文中有多少被成功检索到")
        print(f"  F1 Score:          {stats['avg_f1']:.4f}")
        print(f"    含义: Precision 和 Recall 的调和平均")
        print(f"  Hit@K (命中率):    {stats['hit_rate']:.4f}")
        print(f"    含义: 至少检索到一个标准上下文的比例")

        print(f"\n检索质量分布:")
        print(f"  完美检索 (Recall=1.0): {stats['perfect_retrieval']} ({stats['perfect_rate']:.2%})")
        print(f"  完全失败 (Hit=0):      {stats['no_hits']} ({stats['no_hit_rate']:.2%})")

    def print_samples(self, stats: dict, num_samples: int = 5):
        """打印样本案例"""
        print(f"\n样本案例分析:")
        print("=" * 80)

        samples = stats['samples']

        # 按 F1 分数排序
        samples_sorted = sorted(samples, key=lambda x: x['metrics']['f1'])

        # 打印最差的样本
        print(f"\n【检索质量最差的 {num_samples} 个样本】")
        print("-" * 80)
        for i, sample in enumerate(samples_sorted[:num_samples], 1):
            metrics = sample['metrics']
            print(f"\n样本 {i}:")
            print(f"  问题: {sample['question'][:100]}...")
            print(f"  标准上下文 ({len(sample['gold_titles'])}): {', '.join(sample['gold_titles'][:3])}...")
            print(f"  检索上下文 ({len(sample['retrieved_titles'])}): {', '.join(sample['retrieved_titles'][:3])}...")
            print(f"  指标: Precision={metrics['precision']:.2f}, Recall={metrics['recall']:.2f}, "
                  f"F1={metrics['f1']:.2f}, Hit={metrics['hit']}")

        # 打印最好的样本
        print(f"\n【检索质量最好的 {num_samples} 个样本】")
        print("-" * 80)
        for i, sample in enumerate(samples_sorted[-num_samples:][::-1], 1):
            metrics = sample['metrics']
            print(f"\n样本 {i}:")
            print(f"  问题: {sample['question'][:100]}...")
            print(f"  标准上下文 ({len(sample['gold_titles'])}): {', '.join(sample['gold_titles'][:3])}...")
            print(f"  检索上下文 ({len(sample['retrieved_titles'])}): {', '.join(sample['retrieved_titles'][:3])}...")
            print(f"  指标: Precision={metrics['precision']:.2f}, Recall={metrics['recall']:.2f}, "
                  f"F1={metrics['f1']:.2f}, Hit={metrics['hit']}")

    def save_report(self, dataset_name: str, stats: dict):
        """保存详细报告"""
        output_dir = self.config['outputs'].get('analysis_dir', 'evaluate/outputs/analysis')
        os.makedirs(output_dir, exist_ok=True)

        output_file = os.path.join(output_dir, f'{dataset_name}_retrieval_quality.json')

        # 准备输出数据
        report = {
            'dataset': dataset_name,
            'summary': {
                'total': stats['total'],
                'avg_precision': stats['avg_precision'],
                'avg_recall': stats['avg_recall'],
                'avg_f1': stats['avg_f1'],
                'hit_rate': stats['hit_rate'],
                'perfect_rate': stats['perfect_rate'],
                'no_hit_rate': stats['no_hit_rate']
            },
            'samples': stats['samples']
        }

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        print(f"\n详细报告已保存到: {output_file}")

    def run(self, datasets: List[str] = None, sample_size: int = None, num_display_samples: int = 5):
        """运行分析"""
        if datasets is None:
            datasets = self.config['datasets']

        print("\n" + "="*80)
        print("检索质量分析")
        print("="*80)

        for dataset_name in datasets:
            stats = self.analyze_dataset(dataset_name, sample_size)

            if stats:
                self.print_summary(dataset_name, stats)
                self.print_samples(stats, num_display_samples)
                self.save_report(dataset_name, stats)

        print("\n" + "="*80)
        print("分析完成")
        print("="*80)


def main():
    """主入口"""
    parser = argparse.ArgumentParser(
        description="分析检索质量：对比检索上下文与标准上下文"
    )
    parser.add_argument(
        '--config',
        default='evaluate/config.yaml',
        help='配置文件路径'
    )
    parser.add_argument(
        '--datasets',
        nargs='+',
        help='要分析的数据集（默认: 配置文件中的所有数据集）'
    )
    parser.add_argument(
        '--sample-size',
        type=int,
        default=None,
        help='随机采样的样本数量（默认: 分析所有样本）'
    )
    parser.add_argument(
        '--num-samples',
        type=int,
        default=5,
        help='显示的样本案例数量（默认: 5）'
    )

    args = parser.parse_args()

    # 加载配置
    config = ConfigLoader.load_config(args.config)

    # 运行分析
    analyzer = RetrievalQualityAnalyzer(config)
    analyzer.run(
        datasets=args.datasets,
        sample_size=args.sample_size,
        num_display_samples=args.num_samples
    )


if __name__ == '__main__':
    main()
