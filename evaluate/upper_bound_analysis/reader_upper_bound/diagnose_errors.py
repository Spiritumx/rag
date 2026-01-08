"""
Error Diagnosis Script for Reader Upper Bound Test
诊断阅读理解上限测试中的错误案例

功能：
1. 分析错误案例的模式
2. 检查答案格式问题
3. 识别常见错误类型
4. 生成详细诊断报告
"""

import json
import os
import re
from collections import defaultdict, Counter
from typing import List, Dict, Any
import difflib


class ErrorDiagnoser:
    """错误诊断器"""

    def __init__(self, output_dir: str = "evaluate/reader_upper_bound/outputs"):
        self.output_dir = output_dir

    def load_error_cases(self, dataset_name: str, prompt_style: str = "standard") -> List[Dict]:
        """加载错误案例"""
        error_file = os.path.join(
            self.output_dir,
            dataset_name,
            f"error_cases_{prompt_style}.jsonl"
        )

        if not os.path.exists(error_file):
            print(f"Warning: Error file not found: {error_file}")
            return []

        errors = []
        with open(error_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    errors.append(json.loads(line))

        print(f"✓ Loaded {len(errors)} error cases from {dataset_name}")
        return errors

    def load_all_results(self, dataset_name: str, prompt_style: str = "standard") -> List[Dict]:
        """加载所有结果（包括正确和错误的）"""
        results_file = os.path.join(
            self.output_dir,
            dataset_name,
            f"results_{prompt_style}.jsonl"
        )

        if not os.path.exists(results_file):
            print(f"Warning: Results file not found: {results_file}")
            return []

        results = []
        with open(results_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    results.append(json.loads(line))

        return results

    def normalize_text(self, text: str) -> str:
        """标准化文本用于比较"""
        if not text:
            return ""

        # 转小写
        text = text.lower()

        # 移除标点符号
        text = re.sub(r'[^\w\s]', ' ', text)

        # 移除多余空格
        text = ' '.join(text.split())

        return text

    def check_partial_match(self, predicted: str, gold_answers: List[str]) -> Dict[str, Any]:
        """检查部分匹配情况"""
        pred_norm = self.normalize_text(predicted)

        results = {
            'has_partial_match': False,
            'best_similarity': 0.0,
            'best_match': None,
            'contains_answer': False,
            'answer_contained_in_pred': None
        }

        for gold in gold_answers:
            gold_norm = self.normalize_text(gold)

            # 计算相似度
            similarity = difflib.SequenceMatcher(None, pred_norm, gold_norm).ratio()
            if similarity > results['best_similarity']:
                results['best_similarity'] = similarity
                results['best_match'] = gold

            # 检查包含关系
            if gold_norm in pred_norm:
                results['contains_answer'] = True
                results['answer_contained_in_pred'] = gold

            if pred_norm in gold_norm:
                results['has_partial_match'] = True

        return results

    def categorize_error(self, error_case: Dict) -> str:
        """分类错误类型"""
        predicted = error_case['predicted_answer']
        gold_answers = error_case['gold_answers']

        # 检查是否为空答案
        if not predicted or predicted.strip() == "":
            return "empty_prediction"

        # 检查是否为错误信息
        if predicted.startswith("Error"):
            return "llm_error"

        # 检查是否为 "I don't know" 类型
        if "don't know" in predicted.lower() or "cannot" in predicted.lower():
            return "refusal"

        # 检查部分匹配
        partial_info = self.check_partial_match(predicted, gold_answers)

        if partial_info['contains_answer']:
            return "answer_embedded"  # 答案被包含在更长的文本中

        if partial_info['best_similarity'] > 0.7:
            return "high_similarity"  # 高相似度但不完全匹配

        if partial_info['best_similarity'] > 0.4:
            return "moderate_similarity"  # 中等相似度

        # 检查答案长度差异
        pred_len = len(predicted.split())
        avg_gold_len = sum(len(g.split()) for g in gold_answers) / len(gold_answers)

        if pred_len > avg_gold_len * 2:
            return "too_verbose"  # 答案过长

        if pred_len < avg_gold_len * 0.5 and pred_len > 0:
            return "too_short"  # 答案过短

        return "completely_wrong"  # 完全错误

    def analyze_dataset(self, dataset_name: str, prompt_style: str = "standard") -> Dict[str, Any]:
        """分析单个数据集的错误"""
        print(f"\n{'='*70}")
        print(f"Analyzing {dataset_name}")
        print(f"{'='*70}")

        # 加载错误案例
        error_cases = self.load_error_cases(dataset_name, prompt_style)
        all_results = self.load_all_results(dataset_name, prompt_style)

        if not error_cases:
            print(f"No errors found for {dataset_name}")
            return {}

        # 统计错误类型
        error_types = Counter()
        error_examples = defaultdict(list)

        # 答案长度统计
        pred_lengths = []
        gold_lengths = []

        # 相似度统计
        similarities = []

        for error in error_cases:
            # 分类错误
            error_type = self.categorize_error(error)
            error_types[error_type] += 1

            # 保存示例（每种类型最多3个）
            if len(error_examples[error_type]) < 3:
                error_examples[error_type].append({
                    'question': error['question'],
                    'predicted': error['predicted_answer'],
                    'gold': error['gold_answers'],
                    'context_preview': error.get('gold_context_preview', '')[:200]
                })

            # 统计长度
            pred_lengths.append(len(error['predicted_answer'].split()))
            avg_gold_len = sum(len(g.split()) for g in error['gold_answers']) / len(error['gold_answers'])
            gold_lengths.append(avg_gold_len)

            # 统计相似度
            partial_info = self.check_partial_match(
                error['predicted_answer'],
                error['gold_answers']
            )
            similarities.append(partial_info['best_similarity'])

        # 计算统计信息
        total_samples = len(all_results)
        total_errors = len(error_cases)
        error_rate = total_errors / total_samples if total_samples > 0 else 0

        avg_pred_len = sum(pred_lengths) / len(pred_lengths) if pred_lengths else 0
        avg_gold_len = sum(gold_lengths) / len(gold_lengths) if gold_lengths else 0
        avg_similarity = sum(similarities) / len(similarities) if similarities else 0

        # 生成报告
        report = {
            'dataset': dataset_name,
            'total_samples': total_samples,
            'total_errors': total_errors,
            'error_rate': error_rate,
            'error_type_distribution': dict(error_types),
            'error_examples': dict(error_examples),
            'statistics': {
                'avg_predicted_length': avg_pred_len,
                'avg_gold_length': avg_gold_len,
                'avg_similarity': avg_similarity,
                'length_ratio': avg_pred_len / avg_gold_len if avg_gold_len > 0 else 0
            }
        }

        # 打印报告
        self._print_report(report)

        # 保存报告
        self._save_report(dataset_name, report, prompt_style)

        return report

    def _print_report(self, report: Dict):
        """打印诊断报告"""
        print(f"\n{'='*70}")
        print("ERROR ANALYSIS REPORT")
        print(f"{'='*70}")

        print(f"\nDataset: {report['dataset']}")
        print(f"Total Samples: {report['total_samples']}")
        print(f"Total Errors: {report['total_errors']}")
        print(f"Error Rate: {report['error_rate']*100:.2f}%")

        # 错误类型分布
        print(f"\n{'='*70}")
        print("ERROR TYPE DISTRIBUTION")
        print(f"{'='*70}")

        error_types = report['error_type_distribution']
        sorted_types = sorted(error_types.items(), key=lambda x: x[1], reverse=True)

        for error_type, count in sorted_types:
            percentage = count / report['total_errors'] * 100
            print(f"  {error_type:25s}: {count:3d} ({percentage:5.2f}%)")

        # 统计信息
        print(f"\n{'='*70}")
        print("STATISTICS")
        print(f"{'='*70}")

        stats = report['statistics']
        print(f"  Average Predicted Length: {stats['avg_predicted_length']:.2f} words")
        print(f"  Average Gold Length:      {stats['avg_gold_length']:.2f} words")
        print(f"  Length Ratio (Pred/Gold): {stats['length_ratio']:.2f}")
        print(f"  Average Similarity:       {stats['avg_similarity']:.2%}")

        # 错误示例
        print(f"\n{'='*70}")
        print("ERROR EXAMPLES (Top 3 per type)")
        print(f"{'='*70}")

        for error_type in sorted_types[:5]:  # 显示前5种错误类型
            type_name = error_type[0]
            examples = report['error_examples'].get(type_name, [])

            if examples:
                print(f"\n[{type_name}]")
                for i, example in enumerate(examples, 1):
                    print(f"\n  Example {i}:")
                    print(f"    Question:  {example['question'][:80]}...")
                    print(f"    Predicted: {example['predicted']}")
                    print(f"    Gold:      {example['gold']}")

    def _save_report(self, dataset_name: str, report: Dict, prompt_style: str):
        """保存诊断报告"""
        dataset_output_dir = os.path.join(self.output_dir, dataset_name)
        os.makedirs(dataset_output_dir, exist_ok=True)

        # 保存 JSON 报告
        report_file = os.path.join(dataset_output_dir, f"diagnosis_{prompt_style}.json")
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        print(f"\n✓ Diagnosis report saved to {report_file}")

    def compare_datasets(self, datasets: List[str], prompt_style: str = "standard"):
        """比较多个数据集的错误模式"""
        print(f"\n{'='*70}")
        print("CROSS-DATASET COMPARISON")
        print(f"{'='*70}")

        all_reports = []
        for dataset in datasets:
            report = self.analyze_dataset(dataset, prompt_style)
            if report:
                all_reports.append(report)

        if len(all_reports) < 2:
            print("Need at least 2 datasets for comparison")
            return

        # 比较错误类型分布
        print(f"\n{'='*70}")
        print("ERROR TYPE COMPARISON")
        print(f"{'='*70}")

        # 收集所有错误类型
        all_error_types = set()
        for report in all_reports:
            all_error_types.update(report['error_type_distribution'].keys())

        # 打印表头
        header = f"{'Error Type':25s}"
        for report in all_reports:
            header += f" | {report['dataset'][:15]:>15s}"
        print(header)
        print("-" * len(header))

        # 打印每种错误类型的分布
        for error_type in sorted(all_error_types):
            row = f"{error_type:25s}"
            for report in all_reports:
                count = report['error_type_distribution'].get(error_type, 0)
                total = report['total_errors']
                percentage = count / total * 100 if total > 0 else 0
                row += f" | {percentage:6.2f}% ({count:2d})"
            print(row)

        # 比较统计信息
        print(f"\n{'='*70}")
        print("STATISTICS COMPARISON")
        print(f"{'='*70}")

        print(f"{'Metric':25s}", end="")
        for report in all_reports:
            print(f" | {report['dataset'][:15]:>15s}", end="")
        print()
        print("-" * (25 + 19 * len(all_reports)))

        metrics = [
            ('avg_predicted_length', 'Avg Pred Length'),
            ('avg_gold_length', 'Avg Gold Length'),
            ('length_ratio', 'Length Ratio'),
            ('avg_similarity', 'Avg Similarity')
        ]

        for metric_key, metric_name in metrics:
            print(f"{metric_name:25s}", end="")
            for report in all_reports:
                value = report['statistics'].get(metric_key, 0)
                if metric_key == 'avg_similarity':
                    print(f" | {value:14.2%} ", end="")
                else:
                    print(f" | {value:15.2f}", end="")
            print()


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description="Diagnose Reader Upper Bound Test Errors")
    parser.add_argument('--datasets', nargs='+',
                       default=['musique', 'trivia'],
                       help='Datasets to diagnose')
    parser.add_argument('--output-dir',
                       default='evaluate/upper_bound_analysis/reader_upper_bound/outputs',
                       help='Output directory')
    parser.add_argument('--prompt-style', default='standard',
                       choices=['standard', 'cot', 'structured'],
                       help='Prompt style used in the test')
    parser.add_argument('--compare', action='store_true',
                       help='Compare error patterns across datasets')

    args = parser.parse_args()

    print("\n" + "="*70)
    print("ERROR DIAGNOSIS FOR READER UPPER BOUND TEST")
    print("="*70)
    print(f"Datasets: {', '.join(args.datasets)}")
    print(f"Prompt Style: {args.prompt_style}")
    print("="*70)

    # 创建诊断器
    diagnoser = ErrorDiagnoser(args.output_dir)

    if args.compare:
        # 比较模式
        diagnoser.compare_datasets(args.datasets, args.prompt_style)
    else:
        # 单独分析每个数据集
        for dataset in args.datasets:
            diagnoser.analyze_dataset(dataset, args.prompt_style)

    print("\n" + "="*70)
    print("✓ DIAGNOSIS COMPLETE")
    print("="*70)


if __name__ == '__main__':
    main()
