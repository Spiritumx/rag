"""
分析检索召回率结果

功能：
1. 比较不同检索策略的召回率
2. 分析召回失败的案例
3. 生成详细的诊断报告
4. 可视化召回率曲线
"""

import json
import os
from typing import List, Dict, Any
from collections import defaultdict, Counter


class RecallAnalyzer:
    """召回率分析器"""

    def __init__(self, output_dir: str = "evaluate/upper_bound_analysis/retriever_recall_upperbound/outputs"):
        """
        初始化分析器

        Args:
            output_dir: 输出目录
        """
        self.output_dir = output_dir

    def load_recall_metrics(self, dataset_name: str) -> Dict[str, Any]:
        """
        加载召回率指标

        Args:
            dataset_name: 数据集名称

        Returns:
            召回率指标
        """
        metrics_file = os.path.join(self.output_dir, dataset_name, "recall_metrics.json")

        if not os.path.exists(metrics_file):
            raise FileNotFoundError(f"Metrics file not found: {metrics_file}")

        with open(metrics_file, 'r', encoding='utf-8') as f:
            metrics = json.load(f)

        return metrics

    def load_recall_details(self, dataset_name: str) -> List[Dict]:
        """
        加载详细结果

        Args:
            dataset_name: 数据集名称

        Returns:
            详细结果列表
        """
        details_file = os.path.join(self.output_dir, dataset_name, "recall_details.jsonl")

        if not os.path.exists(details_file):
            raise FileNotFoundError(f"Details file not found: {details_file}")

        details = []
        with open(details_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    details.append(json.loads(line))

        return details

    def load_context_overlap(self, dataset_name: str) -> Dict[str, Any]:
        """
        加载文档重叠率统计

        Args:
            dataset_name: 数据集名称

        Returns:
            重叠率统计
        """
        overlap_file = os.path.join(self.output_dir, dataset_name, "context_overlap.json")

        if not os.path.exists(overlap_file):
            print(f"Warning: Overlap file not found: {overlap_file}")
            return {}

        with open(overlap_file, 'r', encoding='utf-8') as f:
            overlap = json.load(f)

        return overlap

    def analyze_failure_cases(self, details: List[Dict], k: int = 5) -> Dict[str, Any]:
        """
        分析召回失败的案例

        Args:
            details: 详细结果
            k: Top-K 值

        Returns:
            失败案例分析
        """
        print(f"\n{'='*70}")
        print(f"Analyzing Failure Cases @ {k}")
        print(f"{'='*70}")

        failure_cases = []
        for item in details:
            if not item.get(f'found_at_{k}', False):
                failure_cases.append(item)

        total_failures = len(failure_cases)
        total_cases = len(details)

        print(f"\nTotal Failures @ {k}: {total_failures} / {total_cases} ({total_failures/total_cases*100:.2f}%)")

        if total_failures == 0:
            print("No failures to analyze!")
            return {}

        # 分析问题类型分布
        question_types = self._analyze_question_types(failure_cases)

        print(f"\nFailure Distribution by Question Type:")
        for qtype, count in question_types.items():
            pct = count / total_failures * 100
            print(f"  {qtype:20s}: {count:4d} ({pct:5.2f}%)")

        # 分析答案长度分布
        answer_lengths = [len(item['gold_answers'][0]) for item in failure_cases if item['gold_answers']]
        avg_answer_length = sum(answer_lengths) / len(answer_lengths) if answer_lengths else 0

        print(f"\nAverage Answer Length in Failures: {avg_answer_length:.2f} characters")

        return {
            'total_failures': total_failures,
            'failure_rate': total_failures / total_cases,
            'question_types': question_types,
            'avg_answer_length': avg_answer_length,
            'sample_failures': failure_cases[:10]  # 前10个失败案例
        }

    def _analyze_question_types(self, cases: List[Dict]) -> Dict[str, int]:
        """分析问题类型分布"""
        question_types = defaultdict(int)

        for case in cases:
            question = case['question'].lower()

            if question.startswith('what'):
                question_types['what'] += 1
            elif question.startswith('who'):
                question_types['who'] += 1
            elif question.startswith('when'):
                question_types['when'] += 1
            elif question.startswith('where'):
                question_types['where'] += 1
            elif question.startswith('why'):
                question_types['why'] += 1
            elif question.startswith('how'):
                question_types['how'] += 1
            elif question.startswith('which'):
                question_types['which'] += 1
            else:
                question_types['other'] += 1

        return dict(question_types)

    def analyze_recall_degradation(self, details: List[Dict], k_values: List[int] = [5, 20, 100]) -> Dict[str, Any]:
        """
        分析召回率衰减

        Args:
            details: 详细结果
            k_values: K值列表

        Returns:
            衰减分析结果
        """
        print(f"\n{'='*70}")
        print("Analyzing Recall Degradation")
        print(f"{'='*70}")

        degradation_cases = []

        for item in details:
            # 找出在哪个 K 值首次找到答案
            first_found_at = None
            for k in sorted(k_values):
                if item.get(f'found_at_{k}', False):
                    first_found_at = k
                    break

            if first_found_at is not None:
                degradation_cases.append({
                    'question_id': item['question_id'],
                    'question': item['question'],
                    'first_found_at': first_found_at
                })

        # 统计分布
        distribution = Counter(case['first_found_at'] for case in degradation_cases)

        print(f"\nFirst Found Distribution:")
        for k in sorted(k_values):
            count = distribution.get(k, 0)
            pct = count / len(details) * 100 if details else 0
            print(f"  First found @ {k:3d}: {count:4d} ({pct:5.2f}%)")

        # 计算损失
        losses = {}
        for i, k in enumerate(sorted(k_values)):
            if i > 0:
                prev_k = sorted(k_values)[i-1]
                # 在 prev_k 没找到但在 k 找到的案例
                lost = distribution.get(k, 0)
                losses[f'{prev_k}->{k}'] = lost

        print(f"\nRecall Losses Between K Values:")
        for transition, lost in losses.items():
            print(f"  Lost in {transition}: {lost} cases")

        return {
            'distribution': dict(distribution),
            'losses': losses,
            'degradation_cases': degradation_cases
        }

    def generate_diagnosis_report(
        self,
        dataset_name: str,
        output_file: str = None
    ):
        """
        生成诊断报告

        Args:
            dataset_name: 数据集名称
            output_file: 输出文件路径
        """
        print(f"\n{'='*70}")
        print(f"Generating Diagnosis Report for {dataset_name}")
        print(f"{'='*70}")

        # 加载数据
        metrics = self.load_recall_metrics(dataset_name)
        details = self.load_recall_details(dataset_name)
        overlap = self.load_context_overlap(dataset_name)

        # 分析失败案例
        k_values = metrics['config']['k_values']
        failure_analysis = {}
        for k in k_values:
            failure_analysis[k] = self.analyze_failure_cases(details, k)

        # 分析召回率衰减
        degradation_analysis = self.analyze_recall_degradation(details, k_values)

        # 生成建议
        recommendations = self._generate_recommendations(metrics, overlap, failure_analysis)

        # 构建报告
        report = self._build_report(
            dataset_name=dataset_name,
            metrics=metrics,
            overlap=overlap,
            failure_analysis=failure_analysis,
            degradation_analysis=degradation_analysis,
            recommendations=recommendations
        )

        # 打印报告
        print("\n" + report)

        # 保存报告
        if output_file is None:
            output_file = os.path.join(self.output_dir, dataset_name, "diagnosis_report.md")

        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(report)

        print(f"\n✓ Diagnosis report saved to {output_file}")

    def _generate_recommendations(
        self,
        metrics: Dict,
        overlap: Dict,
        failure_analysis: Dict
    ) -> List[str]:
        """生成优化建议"""
        recommendations = []
        recall_scores = metrics['recall_scores']

        recall_100 = recall_scores.get('recall@100', 0)
        recall_20 = recall_scores.get('recall@20', 0)
        recall_5 = recall_scores.get('recall@5', 0)

        # 基于召回率的建议
        if recall_100 < 0.50:
            recommendations.append("✗ 粗排能力严重不足 (Recall@100 < 50%)")
            recommendations.append("→ 实现语义门控召回（创新点三）")
            recommendations.append("→ 增加检索候选数量（top_k: 100 -> 200-500）")
            recommendations.append("→ 改进 Query Rewrite")
            recommendations.append("→ 使用混合检索（BM25 + Dense）")

        elif recall_100 >= 0.80 and recall_5 < 0.40:
            recommendations.append("△ 粗排能力良好，但精排能力不足")
            recommendations.append("→ 更换更好的 Reranker（Cross-Encoder）")
            recommendations.append("→ 实现 ToT 检索（创新点二）")
            recommendations.append("→ 增加给 LLM 的文档数量（top_k: 5 -> 10-20）")
            recommendations.append("→ 在数据集上微调 Reranker")

        elif recall_100 >= 0.80 and recall_5 >= 0.60:
            recommendations.append("✓ 检索能力优秀")
            recommendations.append("→ 检索不是瓶颈，去优化 Reader（LLM/Prompt）")
            recommendations.append("→ 运行 Reader Upper Bound Test")

        # 基于文档重叠率的建议
        if overlap:
            avg_overlap = overlap.get('average_overlap', 0)
            if avg_overlap < 0.50:
                recommendations.append("→ Gold Context 重叠率低，检索策略偏离目标")
                recommendations.append("→ 检查 Query Rewrite 质量")

        # 基于失败案例的建议
        if failure_analysis:
            failure_5 = failure_analysis.get(5, {})
            if failure_5:
                question_types = failure_5.get('question_types', {})
                if question_types:
                    max_type = max(question_types, key=question_types.get)
                    recommendations.append(f"→ '{max_type}' 类型问题召回率最低，需要针对性优化")

        return recommendations

    def _build_report(
        self,
        dataset_name: str,
        metrics: Dict,
        overlap: Dict,
        failure_analysis: Dict,
        degradation_analysis: Dict,
        recommendations: List[str]
    ) -> str:
        """构建诊断报告"""
        lines = []

        # 标题
        lines.append("="*70)
        lines.append(f"RETRIEVER RECALL UPPER BOUND - DIAGNOSIS REPORT")
        lines.append("="*70)
        lines.append(f"\nDataset: {dataset_name}")
        lines.append(f"\n{'='*70}")

        # 召回率指标
        lines.append("\n## RECALL METRICS")
        for metric_name, score in metrics['recall_scores'].items():
            lines.append(f"\n- **{metric_name}**: {score:.4f} ({score*100:.2f}%)")

        # 文档重叠率
        if overlap:
            lines.append(f"\n{'='*70}")
            lines.append("\n## CONTEXT OVERLAP (vs Gold Contexts)")
            lines.append(f"\n- **Average Overlap**: {overlap.get('average_overlap', 0):.4f} ({overlap.get('average_overlap', 0)*100:.2f}%)")

            overlap_stats = overlap.get('overlap_stats', {})
            if overlap_stats:
                total = overlap_stats.get('total', 1)
                lines.append(f"\n- **Perfect Matches**: {overlap_stats.get('perfect_matches', 0)} ({overlap_stats.get('perfect_matches', 0)/total*100:.2f}%)")
                lines.append(f"- **Partial Matches**: {overlap_stats.get('partial_matches', 0)} ({overlap_stats.get('partial_matches', 0)/total*100:.2f}%)")
                lines.append(f"- **No Matches**: {overlap_stats.get('no_matches', 0)} ({overlap_stats.get('no_matches', 0)/total*100:.2f}%)")

        # 召回率衰减分析
        lines.append(f"\n{'='*70}")
        lines.append("\n## RECALL DEGRADATION ANALYSIS")

        distribution = degradation_analysis.get('distribution', {})
        lines.append("\n### First Found Distribution")
        for k, count in sorted(distribution.items()):
            lines.append(f"  - First found @ {k}: {count}")

        losses = degradation_analysis.get('losses', {})
        if losses:
            lines.append("\n### Recall Losses Between K Values")
            for transition, lost in losses.items():
                lines.append(f"  - Lost in {transition}: {lost} cases")

        # 失败案例分析
        lines.append(f"\n{'='*70}")
        lines.append("\n## FAILURE CASE ANALYSIS")

        for k, analysis in sorted(failure_analysis.items()):
            if analysis:
                lines.append(f"\n### Failures @ {k}")
                lines.append(f"\n- **Total Failures**: {analysis['total_failures']} ({analysis['failure_rate']*100:.2f}%)")

                question_types = analysis.get('question_types', {})
                if question_types:
                    lines.append("\n- **Question Type Distribution**:")
                    for qtype, count in question_types.items():
                        pct = count / analysis['total_failures'] * 100
                        lines.append(f"    - {qtype}: {count} ({pct:.2f}%)")

        # 优化建议
        lines.append(f"\n{'='*70}")
        lines.append("\n## RECOMMENDATIONS")
        for rec in recommendations:
            lines.append(f"\n{rec}")

        # 失败案例样本
        if failure_analysis.get(5, {}).get('sample_failures'):
            lines.append(f"\n{'='*70}")
            lines.append("\n## SAMPLE FAILURE CASES @ 5 (First 10)")

            for i, failure in enumerate(failure_analysis[5]['sample_failures'][:10], 1):
                lines.append(f"\n### Failure #{i}")
                lines.append(f"- **Question**: {failure['question']}")
                lines.append(f"- **Gold Answers**: {', '.join(failure['gold_answers'])}")
                lines.append(f"- **Num Retrieved**: {failure['num_retrieved']}")

        lines.append(f"\n{'='*70}")
        lines.append("\n## END OF REPORT")
        lines.append("="*70)

        return "\n".join(lines)

    def compare_strategies(
        self,
        dataset_name: str,
        strategy_names: List[str]
    ):
        """
        比较不同检索策略

        Args:
            dataset_name: 数据集名称
            strategy_names: 策略名称列表（对应不同的输出目录）
        """
        print(f"\n{'='*70}")
        print(f"Comparing Retrieval Strategies for {dataset_name}")
        print(f"{'='*70}")

        results = {}

        for strategy in strategy_names:
            strategy_dir = os.path.join(self.output_dir.replace(dataset_name, ''), strategy, dataset_name)
            metrics_file = os.path.join(strategy_dir, "recall_metrics.json")

            if not os.path.exists(metrics_file):
                print(f"Warning: Metrics file not found for strategy '{strategy}'")
                continue

            with open(metrics_file, 'r', encoding='utf-8') as f:
                metrics = json.load(f)
                results[strategy] = metrics['recall_scores']

        # 打印比较表格
        if results:
            k_values = sorted(set(k.replace('recall@', '') for strategy_scores in results.values() for k in strategy_scores.keys()))

            print(f"\n{'Strategy':<20}", end='')
            for k in k_values:
                print(f"{'Recall@'+k:>12}", end='')
            print()
            print("-" * 70)

            for strategy, scores in results.items():
                print(f"{strategy:<20}", end='')
                for k in k_values:
                    score = scores.get(f'recall@{k}', 0)
                    print(f"{score:>12.4f}", end='')
                print()

            # 找出最佳策略
            best_strategy = max(results.keys(), key=lambda s: results[s].get('recall@5', 0))
            print(f"\n✓ Best Strategy: {best_strategy} (Recall@5 = {results[best_strategy].get('recall@5', 0):.4f})")

        print(f"\n{'='*70}")


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description="Analyze Retrieval Recall Results")
    parser.add_argument('--dataset', required=True,
                       help='Dataset name to analyze')
    parser.add_argument('--output-dir',
                       default='evaluate/upper_bound_analysis/retriever_recall_upperbound/outputs',
                       help='Output directory')
    parser.add_argument('--compare-strategies', nargs='+',
                       help='Compare multiple retrieval strategies')

    args = parser.parse_args()

    analyzer = RecallAnalyzer(args.output_dir)

    if args.compare_strategies:
        # 比较不同检索策略
        analyzer.compare_strategies(args.dataset, args.compare_strategies)
    else:
        # 生成诊断报告
        analyzer.generate_diagnosis_report(args.dataset)


if __name__ == '__main__':
    main()
