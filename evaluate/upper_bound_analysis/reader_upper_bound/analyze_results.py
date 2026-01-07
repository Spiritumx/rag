"""
分析 Reader Upper Bound 测试结果

功能：
1. 分析错误案例，找出模型的弱点
2. 生成详细的诊断报告
3. 提供针对性的优化建议
"""

import json
import os
from typing import List, Dict, Any
from collections import defaultdict, Counter
import re


class ResultAnalyzer:
    """结果分析器"""

    def __init__(self, output_dir: str = "evaluate/reader_upper_bound/outputs"):
        """
        初始化分析器

        Args:
            output_dir: 输出目录
        """
        self.output_dir = output_dir

    def load_results(self, dataset_name: str, prompt_style: str = "standard") -> List[Dict]:
        """
        加载结果数据

        Args:
            dataset_name: 数据集名称
            prompt_style: Prompt 风格

        Returns:
            结果列表
        """
        results_file = os.path.join(self.output_dir, dataset_name, f"results_{prompt_style}.jsonl")

        if not os.path.exists(results_file):
            raise FileNotFoundError(f"Results file not found: {results_file}")

        results = []
        with open(results_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    results.append(json.loads(line))

        return results

    def load_error_cases(self, dataset_name: str, prompt_style: str = "standard") -> List[Dict]:
        """
        加载错误案例

        Args:
            dataset_name: 数据集名称
            prompt_style: Prompt 风格

        Returns:
            错误案例列表
        """
        errors_file = os.path.join(self.output_dir, dataset_name, f"error_cases_{prompt_style}.jsonl")

        if not os.path.exists(errors_file):
            print(f"Warning: Error cases file not found: {errors_file}")
            return []

        errors = []
        with open(errors_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    errors.append(json.loads(line))

        return errors

    def analyze_error_patterns(self, error_cases: List[Dict]) -> Dict[str, Any]:
        """
        分析错误模式

        Args:
            error_cases: 错误案例列表

        Returns:
            错误模式分析结果
        """
        print(f"\n{'='*70}")
        print("ERROR PATTERN ANALYSIS")
        print(f"{'='*70}")

        total_errors = len(error_cases)
        print(f"\nTotal Error Cases: {total_errors}")

        if total_errors == 0:
            print("No errors to analyze!")
            return {}

        # 分析错误类型
        error_types = {
            'empty_answer': 0,       # 空答案或 "I don't know"
            'wrong_extraction': 0,   # 提取错误
            'format_error': 0,       # 格式错误
            'reasoning_error': 0     # 推理错误
        }

        answer_patterns = Counter()

        for error in error_cases:
            pred = error['predicted_answer'].lower().strip()

            # 分类错误类型
            if not pred or pred in ["i don't know", "error", "unknown", "none", "no answer"]:
                error_types['empty_answer'] += 1
                answer_patterns['empty/unknown'] += 1
            elif pred.startswith("error"):
                error_types['format_error'] += 1
                answer_patterns['format_error'] += 1
            else:
                error_types['wrong_extraction'] += 1
                # 记录常见错误答案模式
                if len(pred) > 50:
                    answer_patterns['too_long'] += 1
                else:
                    answer_patterns[pred[:30]] += 1

        # 打印错误类型分布
        print(f"\n{'='*70}")
        print("Error Type Distribution:")
        print(f"{'='*70}")
        for error_type, count in error_types.items():
            pct = count / total_errors * 100
            print(f"  {error_type:20s}: {count:4d} ({pct:5.2f}%)")

        # 打印最常见的错误答案
        print(f"\n{'='*70}")
        print("Most Common Error Answers:")
        print(f"{'='*70}")
        for answer, count in answer_patterns.most_common(10):
            pct = count / total_errors * 100
            print(f"  '{answer}': {count} ({pct:.2f}%)")

        # 分析问题类型
        question_types = self._analyze_question_types(error_cases)

        print(f"\n{'='*70}")
        print("Error Distribution by Question Type:")
        print(f"{'='*70}")
        for qtype, count in question_types.items():
            pct = count / total_errors * 100
            print(f"  {qtype:20s}: {count:4d} ({pct:5.2f}%)")

        return {
            'total_errors': total_errors,
            'error_types': error_types,
            'answer_patterns': dict(answer_patterns.most_common(20)),
            'question_types': question_types
        }

    def _analyze_question_types(self, error_cases: List[Dict]) -> Dict[str, int]:
        """
        分析问题类型分布

        Args:
            error_cases: 错误案例列表

        Returns:
            问题类型计数
        """
        question_types = defaultdict(int)

        for error in error_cases:
            question = error['question'].lower()

            # 简单的问题类型分类
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

    def generate_diagnosis_report(
        self,
        dataset_name: str,
        prompt_style: str = "standard",
        output_file: str = None
    ):
        """
        生成诊断报告

        Args:
            dataset_name: 数据集名称
            prompt_style: Prompt 风格
            output_file: 输出文件路径（可选）
        """
        print(f"\n{'='*70}")
        print(f"Generating Diagnosis Report for {dataset_name}")
        print(f"{'='*70}")

        # 加载数据
        results = self.load_results(dataset_name, prompt_style)
        error_cases = self.load_error_cases(dataset_name, prompt_style)

        # 加载指标
        metrics_file = os.path.join(self.output_dir, dataset_name, f"metrics_{prompt_style}.json")
        with open(metrics_file, 'r', encoding='utf-8') as f:
            metrics = json.load(f)

        # 分析错误模式
        error_analysis = self.analyze_error_patterns(error_cases)

        # 生成建议
        recommendations = self._generate_recommendations(metrics, error_analysis)

        # 准备报告内容
        report = self._build_report(
            dataset_name=dataset_name,
            prompt_style=prompt_style,
            metrics=metrics,
            error_analysis=error_analysis,
            recommendations=recommendations,
            sample_errors=error_cases[:10]  # 前10个错误案例
        )

        # 打印报告
        print("\n" + report)

        # 保存报告
        if output_file is None:
            output_file = os.path.join(self.output_dir, dataset_name, f"diagnosis_report_{prompt_style}.md")

        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(report)

        print(f"\n✓ Diagnosis report saved to {output_file}")

    def _generate_recommendations(self, metrics: Dict, error_analysis: Dict) -> List[str]:
        """
        生成优化建议

        Args:
            metrics: 评估指标
            error_analysis: 错误分析结果

        Returns:
            建议列表
        """
        recommendations = []
        em = metrics['em']

        # 基于 EM 的建议
        if em >= 0.70:
            recommendations.append("✓ 模型阅读理解能力优秀，瓶颈在检索")
            recommendations.append("→ 优先优化检索策略（门控召回、ToT、IRCoT 改进）")
            recommendations.append("→ 提升检索召回率（增加 top_k、混合检索、Reranker）")

        elif em >= 0.50:
            recommendations.append("○ 模型具备一定能力，但仍有提升空间")
            recommendations.append("→ 先尝试优化 Prompt（CoT、Few-shot）")
            recommendations.append("→ 再优化检索策略")

        elif em >= 0.30:
            recommendations.append("△ 模型能力较弱，主要问题在 LLM/Prompt")
            recommendations.append("→ 优先优化 Prompt（CoT、结构化）")
            recommendations.append("→ 考虑微调或换更强模型")

        else:
            recommendations.append("✗ 模型能力严重不足")
            recommendations.append("→ 必须换更强的模型（Qwen-2.5-14B、Llama-3-70B）")
            recommendations.append("→ 大幅优化 Prompt 或进行微调")

        # 基于错误类型的建议
        if error_analysis:
            error_types = error_analysis.get('error_types', {})

            if error_types.get('empty_answer', 0) > len(error_analysis.get('total_errors', 1)) * 0.3:
                recommendations.append("→ 大量空答案：需要优化 Prompt 引导模型输出")

            if error_types.get('wrong_extraction', 0) > len(error_analysis.get('total_errors', 1)) * 0.5:
                recommendations.append("→ 提取错误较多：使用 CoT 帮助模型定位答案")

        return recommendations

    def _build_report(
        self,
        dataset_name: str,
        prompt_style: str,
        metrics: Dict,
        error_analysis: Dict,
        recommendations: List[str],
        sample_errors: List[Dict]
    ) -> str:
        """
        构建诊断报告

        Args:
            dataset_name: 数据集名称
            prompt_style: Prompt 风格
            metrics: 评估指标
            error_analysis: 错误分析结果
            recommendations: 建议列表
            sample_errors: 错误案例样本

        Returns:
            报告文本
        """
        lines = []

        # 标题
        lines.append("="*70)
        lines.append(f"READER UPPER BOUND - DIAGNOSIS REPORT")
        lines.append("="*70)
        lines.append(f"\nDataset: {dataset_name}")
        lines.append(f"Prompt Style: {prompt_style}")
        lines.append(f"\n{'='*70}")

        # 指标
        lines.append("\n## METRICS")
        lines.append(f"\n- **EM (Exact Match)**: {metrics['em']:.4f} ({metrics['em']*100:.2f}%)")
        lines.append(f"- **F1 Score**: {metrics['f1']:.4f} ({metrics['f1']*100:.2f}%)")
        lines.append(f"- **Accuracy**: {metrics['acc']:.4f} ({metrics['acc']*100:.2f}%)")
        lines.append(f"- **Recall**: {metrics['recall']:.4f} ({metrics['recall']*100:.2f}%)")
        lines.append(f"- **Total Samples**: {metrics['count']}")

        # 错误分析
        if error_analysis:
            lines.append(f"\n{'='*70}")
            lines.append("\n## ERROR ANALYSIS")
            lines.append(f"\n- **Total Errors**: {error_analysis['total_errors']}")

            lines.append("\n### Error Type Distribution")
            for error_type, count in error_analysis.get('error_types', {}).items():
                pct = count / error_analysis['total_errors'] * 100
                lines.append(f"  - {error_type}: {count} ({pct:.2f}%)")

            lines.append("\n### Question Type Distribution (Errors)")
            for qtype, count in error_analysis.get('question_types', {}).items():
                pct = count / error_analysis['total_errors'] * 100
                lines.append(f"  - {qtype}: {count} ({pct:.2f}%)")

        # 建议
        lines.append(f"\n{'='*70}")
        lines.append("\n## RECOMMENDATIONS")
        for rec in recommendations:
            lines.append(f"\n{rec}")

        # 错误案例样本
        if sample_errors:
            lines.append(f"\n{'='*70}")
            lines.append("\n## SAMPLE ERROR CASES (First 10)")

            for i, error in enumerate(sample_errors, 1):
                lines.append(f"\n### Error #{i}")
                lines.append(f"- **Question**: {error['question']}")
                lines.append(f"- **Predicted**: {error['predicted_answer']}")
                lines.append(f"- **Gold Answers**: {', '.join(error['gold_answers'])}")
                lines.append(f"- **Num Gold Contexts**: {error['num_gold_contexts']}")
                if 'gold_context_preview' in error:
                    lines.append(f"- **Context Preview**:")
                    lines.append(f"  ```")
                    lines.append(f"  {error['gold_context_preview']}")
                    lines.append(f"  ```")

        lines.append(f"\n{'='*70}")
        lines.append("\n## END OF REPORT")
        lines.append("="*70)

        return "\n".join(lines)

    def compare_prompt_styles(self, dataset_name: str, prompt_styles: List[str]):
        """
        比较不同 Prompt 风格的效果

        Args:
            dataset_name: 数据集名称
            prompt_styles: Prompt 风格列表
        """
        print(f"\n{'='*70}")
        print(f"Comparing Prompt Styles for {dataset_name}")
        print(f"{'='*70}")

        results = {}

        for style in prompt_styles:
            metrics_file = os.path.join(self.output_dir, dataset_name, f"metrics_{style}.json")

            if not os.path.exists(metrics_file):
                print(f"Warning: Metrics file not found for style '{style}'")
                continue

            with open(metrics_file, 'r', encoding='utf-8') as f:
                metrics = json.load(f)
                results[style] = metrics

        # 打印比较表格
        print(f"\n{'Prompt Style':<20} {'EM':>10} {'F1':>10} {'Acc':>10}")
        print("-" * 70)

        for style, metrics in results.items():
            print(f"{style:<20} {metrics['em']:>10.4f} {metrics['f1']:>10.4f} {metrics['acc']:>10.4f}")

        # 找出最佳 Prompt
        if results:
            best_style = max(results.keys(), key=lambda s: results[s]['em'])
            print(f"\n✓ Best Prompt Style: {best_style} (EM = {results[best_style]['em']:.4f})")

        print(f"\n{'='*70}")


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description="Analyze Reader Upper Bound Test Results")
    parser.add_argument('--dataset', required=True,
                       help='Dataset name to analyze')
    parser.add_argument('--prompt-style', default='standard',
                       help='Prompt style to analyze')
    parser.add_argument('--output-dir', default='evaluate/reader_upper_bound/outputs',
                       help='Output directory')
    parser.add_argument('--compare-prompts', action='store_true',
                       help='Compare different prompt styles')

    args = parser.parse_args()

    analyzer = ResultAnalyzer(args.output_dir)

    if args.compare_prompts:
        # 比较不同 Prompt 风格
        prompt_styles = ['standard', 'cot', 'structured']
        analyzer.compare_prompt_styles(args.dataset, prompt_styles)
    else:
        # 生成诊断报告
        analyzer.generate_diagnosis_report(args.dataset, args.prompt_style)


if __name__ == '__main__':
    main()
