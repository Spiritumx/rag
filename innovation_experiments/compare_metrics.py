"""
A/B Comparison: Baseline vs V2 (with Innovations)

Compares evaluation metrics between baseline RAG system and V2 with three innovations:
1. Adaptive Retrieval (dynamic hybrid weights)
2. Cascading Dynamic Routing (confidence-based fallback)
3. MI-RA-ToT (beam search reasoning)

Outputs:
- Side-by-side comparison table (markdown)
- Per-dataset improvements
- Overall performance gains
- Statistical significance (optional)
"""

import json
import os
import sys
from typing import Dict, List, Tuple
from collections import defaultdict

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


class MetricsComparator:
    """Compare baseline and V2 metrics for A/B testing."""

    def __init__(self):
        self.baseline_path = 'evaluate/outputs/stage3_metrics/overall_metrics.json'
        self.v2_path = 'innovation_experiments/evaluate_v2/outputs_v2/stage3_metrics_v2/overall_metrics_v2.json'

        self.baseline_metrics = None
        self.v2_metrics = None

    def load_metrics(self) -> bool:
        """
        Load both baseline and V2 metrics.

        Returns:
            True if both loaded successfully, False otherwise
        """
        # Load baseline
        if not os.path.exists(self.baseline_path):
            print(f"ERROR: Baseline metrics not found at {self.baseline_path}")
            print("Please run baseline pipeline first: python evaluate/run_pipeline.py")
            return False

        with open(self.baseline_path, 'r', encoding='utf-8') as f:
            self.baseline_metrics = json.load(f)

        # Load V2
        if not os.path.exists(self.v2_path):
            print(f"ERROR: V2 metrics not found at {self.v2_path}")
            print("Please run V2 pipeline first: python innovation_experiments/evaluate_v2/run_pipeline_v2.py")
            return False

        with open(self.v2_path, 'r', encoding='utf-8') as f:
            self.v2_metrics = json.load(f)

        print("✓ Loaded baseline and V2 metrics successfully")
        return True

    def calculate_improvement(self, baseline_val: float, v2_val: float) -> Tuple[float, float]:
        """
        Calculate absolute and relative improvement.

        Args:
            baseline_val: Baseline metric value
            v2_val: V2 metric value

        Returns:
            Tuple of (absolute_delta, relative_improvement_percent)
        """
        absolute_delta = v2_val - baseline_val

        if baseline_val > 0:
            relative_improvement = (absolute_delta / baseline_val) * 100
        else:
            relative_improvement = 0.0

        return absolute_delta, relative_improvement

    def compare_overall(self) -> Dict:
        """
        Compare overall metrics across all datasets.

        Returns:
            Dictionary with comparison results
        """
        baseline_overall = self.baseline_metrics.get('overall', {})
        v2_overall = self.v2_metrics.get('overall', {})

        results = {}
        for metric in ['em', 'f1', 'acc', 'recall']:
            baseline_val = baseline_overall.get(metric, 0)
            v2_val = v2_overall.get(metric, 0)

            abs_delta, rel_improvement = self.calculate_improvement(baseline_val, v2_val)

            results[metric] = {
                'baseline': baseline_val,
                'v2': v2_val,
                'absolute_delta': abs_delta,
                'relative_improvement': rel_improvement
            }

        results['count'] = {
            'baseline': baseline_overall.get('count', 0),
            'v2': v2_overall.get('count', 0)
        }

        return results

    def compare_by_dataset(self) -> Dict:
        """
        Compare metrics for each dataset individually.

        Returns:
            Dictionary mapping dataset to comparison results
        """
        dataset_comparisons = {}

        # Get common datasets
        baseline_datasets = set(self.baseline_metrics.keys()) - {'overall'}
        v2_datasets = set(self.v2_metrics.keys()) - {'overall'}
        common_datasets = baseline_datasets & v2_datasets

        for dataset in sorted(common_datasets):
            baseline_data = self.baseline_metrics[dataset].get('overall', {})
            v2_data = self.v2_metrics[dataset].get('overall', {})

            dataset_results = {}
            for metric in ['em', 'f1', 'acc', 'recall']:
                baseline_val = baseline_data.get(metric, 0)
                v2_val = v2_data.get(metric, 0)

                abs_delta, rel_improvement = self.calculate_improvement(baseline_val, v2_val)

                dataset_results[metric] = {
                    'baseline': baseline_val,
                    'v2': v2_val,
                    'absolute_delta': abs_delta,
                    'relative_improvement': rel_improvement
                }

            dataset_results['count'] = {
                'baseline': baseline_data.get('count', 0),
                'v2': v2_data.get('count', 0)
            }

            dataset_comparisons[dataset] = dataset_results

        return dataset_comparisons

    def compare_by_action(self) -> Dict:
        """
        Compare performance by action (Z, S-Sparse, S-Dense, S-Hybrid, M).

        Returns:
            Dictionary mapping action to comparison results
        """
        action_comparisons = defaultdict(lambda: defaultdict(dict))

        # Get common datasets
        baseline_datasets = set(self.baseline_metrics.keys()) - {'overall'}
        v2_datasets = set(self.v2_metrics.keys()) - {'overall'}
        common_datasets = baseline_datasets & v2_datasets

        # Aggregate by action across all datasets
        action_aggregates = {
            'baseline': defaultdict(lambda: {'em_sum': 0, 'f1_sum': 0, 'count': 0}),
            'v2': defaultdict(lambda: {'em_sum': 0, 'f1_sum': 0, 'count': 0})
        }

        for dataset in common_datasets:
            baseline_by_action = self.baseline_metrics[dataset].get('by_action', {})
            v2_by_action = self.v2_metrics[dataset].get('by_action', {})

            # Aggregate baseline
            for action, metrics in baseline_by_action.items():
                action_aggregates['baseline'][action]['em_sum'] += metrics['em'] * metrics['count']
                action_aggregates['baseline'][action]['f1_sum'] += metrics['f1'] * metrics['count']
                action_aggregates['baseline'][action]['count'] += metrics['count']

            # Aggregate V2
            for action, metrics in v2_by_action.items():
                action_aggregates['v2'][action]['em_sum'] += metrics['em'] * metrics['count']
                action_aggregates['v2'][action]['f1_sum'] += metrics['f1'] * metrics['count']
                action_aggregates['v2'][action]['count'] += metrics['count']

        # Calculate averages and improvements
        all_actions = set(action_aggregates['baseline'].keys()) | set(action_aggregates['v2'].keys())

        for action in all_actions:
            baseline_data = action_aggregates['baseline'][action]
            v2_data = action_aggregates['v2'][action]

            baseline_em = baseline_data['em_sum'] / baseline_data['count'] if baseline_data['count'] > 0 else 0
            baseline_f1 = baseline_data['f1_sum'] / baseline_data['count'] if baseline_data['count'] > 0 else 0

            v2_em = v2_data['em_sum'] / v2_data['count'] if v2_data['count'] > 0 else 0
            v2_f1 = v2_data['f1_sum'] / v2_data['count'] if v2_data['count'] > 0 else 0

            em_delta, em_rel = self.calculate_improvement(baseline_em, v2_em)
            f1_delta, f1_rel = self.calculate_improvement(baseline_f1, v2_f1)

            action_comparisons[action] = {
                'em': {
                    'baseline': round(baseline_em, 4),
                    'v2': round(v2_em, 4),
                    'absolute_delta': round(em_delta, 4),
                    'relative_improvement': round(em_rel, 2)
                },
                'f1': {
                    'baseline': round(baseline_f1, 4),
                    'v2': round(v2_f1, 4),
                    'absolute_delta': round(f1_delta, 4),
                    'relative_improvement': round(f1_rel, 2)
                },
                'count': {
                    'baseline': baseline_data['count'],
                    'v2': v2_data['count']
                }
            }

        return action_comparisons

    def generate_markdown_table(self) -> str:
        """
        Generate markdown table comparing baseline vs V2.

        Returns:
            Markdown formatted string
        """
        md = []
        md.append("# A/B Comparison: Baseline vs V2 (with Innovations)\n")
        md.append("## Overall Performance Across All Datasets\n")

        overall = self.compare_overall()

        # Overall table
        md.append("| Metric | Baseline | V2 (Innovations) | Δ Absolute | Δ Relative | Status |")
        md.append("|--------|----------|------------------|------------|------------|--------|")

        for metric in ['em', 'f1', 'acc', 'recall']:
            data = overall[metric]
            baseline_val = data['baseline']
            v2_val = data['v2']
            abs_delta = data['absolute_delta']
            rel_improvement = data['relative_improvement']

            status = "✅ Improved" if abs_delta > 0 else "❌ Degraded" if abs_delta < 0 else "➖ No Change"

            md.append(f"| {metric.upper():6s} | {baseline_val:.4f}   | {v2_val:.4f}         | "
                     f"{abs_delta:+.4f}     | {rel_improvement:+.2f}%      | {status} |")

        md.append(f"| **Count** | {overall['count']['baseline']} | {overall['count']['v2']} | - | - | - |\n")

        # Per-dataset comparison
        md.append("## Performance by Dataset\n")
        md.append("| Dataset | Metric | Baseline | V2 | Δ Absolute | Δ Relative |")
        md.append("|---------|--------|----------|-------|------------|------------|")

        dataset_comparisons = self.compare_by_dataset()
        for dataset in sorted(dataset_comparisons.keys()):
            data = dataset_comparisons[dataset]

            for i, metric in enumerate(['em', 'f1']):
                metric_data = data[metric]
                dataset_name = dataset if i == 0 else ""

                md.append(f"| {dataset_name:15s} | {metric.upper():2s} | {metric_data['baseline']:.4f}   | "
                         f"{metric_data['v2']:.4f} | {metric_data['absolute_delta']:+.4f}     | "
                         f"{metric_data['relative_improvement']:+.2f}%      |")

        md.append("")

        # By-action comparison
        md.append("## Performance by Action (Strategy)\n")
        md.append("| Action | Metric | Baseline | V2 | Δ Absolute | Δ Relative | Count (B/V2) |")
        md.append("|--------|--------|----------|-------|------------|------------|--------------|")

        action_comparisons = self.compare_by_action()
        for action in ['Z', 'S-Sparse', 'S-Dense', 'S-Hybrid', 'M', 'Unknown']:
            if action not in action_comparisons:
                continue

            data = action_comparisons[action]

            for i, metric in enumerate(['em', 'f1']):
                metric_data = data[metric]
                action_name = action if i == 0 else ""
                count_str = f"{data['count']['baseline']}/{data['count']['v2']}" if i == 0 else ""

                md.append(f"| {action_name:10s} | {metric.upper():2s} | {metric_data['baseline']:.4f}   | "
                         f"{metric_data['v2']:.4f} | {metric_data['absolute_delta']:+.4f}     | "
                         f"{metric_data['relative_improvement']:+.2f}%      | {count_str:12s} |")

        md.append("")

        # Key insights
        md.append("## Key Insights\n")

        em_improvement = overall['em']['absolute_delta']
        f1_improvement = overall['f1']['absolute_delta']

        md.append(f"1. **Overall EM Improvement**: {em_improvement:+.4f} ({overall['em']['relative_improvement']:+.2f}%)")
        md.append(f"2. **Overall F1 Improvement**: {f1_improvement:+.4f} ({overall['f1']['relative_improvement']:+.2f}%)")

        if em_improvement > 0:
            md.append(f"3. ✅ **V2 innovations are EFFECTIVE** - improved performance across metrics")
        else:
            md.append(f"3. ❌ **V2 needs tuning** - performance degraded, review hyperparameters")

        # Find best performing action
        best_action = None
        best_improvement = -999
        for action, data in action_comparisons.items():
            if data['em']['absolute_delta'] > best_improvement:
                best_improvement = data['em']['absolute_delta']
                best_action = action

        if best_action:
            md.append(f"4. **Best Action Improvement**: {best_action} (+{best_improvement:.4f} EM)")

        md.append("\n---\n")
        md.append("**Innovations Tested:**")
        md.append("1. Adaptive Retrieval (dynamic hybrid weights based on query analysis)")
        md.append("2. Cascading Dynamic Routing (confidence-based fallback to stronger strategies)")
        md.append("3. MI-RA-ToT (beam search multi-hop reasoning with mutual information scoring)")

        return "\n".join(md)

    def save_comparison(self, output_dir='innovation_experiments/analysis_results'):
        """
        Save comparison results to files.

        Args:
            output_dir: Directory to save results
        """
        os.makedirs(output_dir, exist_ok=True)

        # Save markdown report
        md_content = self.generate_markdown_table()
        md_path = os.path.join(output_dir, 'baseline_vs_v2_comparison.md')
        with open(md_path, 'w', encoding='utf-8') as f:
            f.write(md_content)
        print(f"✓ Markdown report saved to {md_path}")

        # Save JSON data
        comparison_data = {
            'overall': self.compare_overall(),
            'by_dataset': self.compare_by_dataset(),
            'by_action': self.compare_by_action()
        }

        json_path = os.path.join(output_dir, 'baseline_vs_v2_comparison.json')
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(comparison_data, f, indent=2, ensure_ascii=False)
        print(f"✓ JSON data saved to {json_path}")

        return md_path, json_path

    def print_summary(self):
        """Print comparison summary to console."""
        overall = self.compare_overall()

        print("\n" + "="*80)
        print("A/B COMPARISON SUMMARY: Baseline vs V2")
        print("="*80)

        print("\nOverall Performance:")
        print(f"  EM:     Baseline={overall['em']['baseline']:.4f}, "
              f"V2={overall['em']['v2']:.4f}, "
              f"Δ={overall['em']['absolute_delta']:+.4f} ({overall['em']['relative_improvement']:+.2f}%)")
        print(f"  F1:     Baseline={overall['f1']['baseline']:.4f}, "
              f"V2={overall['f1']['v2']:.4f}, "
              f"Δ={overall['f1']['absolute_delta']:+.4f} ({overall['f1']['relative_improvement']:+.2f}%)")

        print("\nBy Dataset:")
        dataset_comparisons = self.compare_by_dataset()
        for dataset in sorted(dataset_comparisons.keys()):
            data = dataset_comparisons[dataset]
            em_delta = data['em']['absolute_delta']
            f1_delta = data['f1']['absolute_delta']
            print(f"  {dataset:15s}: EM Δ{em_delta:+.4f}, F1 Δ{f1_delta:+.4f}")

        print("\n" + "="*80)


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Compare Baseline vs V2 (with Innovations) metrics"
    )
    parser.add_argument('--output-dir', default='innovation_experiments/analysis_results',
                       help='Directory to save comparison results')
    parser.add_argument('--print-only', action='store_true',
                       help='Only print to console, do not save files')

    args = parser.parse_args()

    # Create comparator
    comparator = MetricsComparator()

    # Load metrics
    if not comparator.load_metrics():
        sys.exit(1)

    # Print summary
    comparator.print_summary()

    # Save results
    if not args.print_only:
        md_path, json_path = comparator.save_comparison(args.output_dir)
        print(f"\n✓ Full comparison report available at: {md_path}")
        print(f"✓ JSON data available at: {json_path}")
    else:
        print("\n(--print-only mode, results not saved)")


if __name__ == '__main__':
    main()
