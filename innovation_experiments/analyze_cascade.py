"""
Cascade Routing Analysis (Innovation 2)

Deep dive into cascading dynamic routing decisions:
- Cascade trigger rate by initial action
- Confidence distribution analysis
- Direct vs cascaded accuracy comparison
- False positive cascade detection
- Confidence threshold optimization

Reads cascade logs from: innovation_experiments/evaluate_v2/outputs_v2/cascade_analysis/
"""

import os
import sys
import csv
import json
from collections import defaultdict
from typing import Dict, List, Tuple
import statistics

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


class CascadeAnalyzer:
    """Analyze cascade routing decisions for Innovation 2."""

    def __init__(self, cascade_dir='innovation_experiments/evaluate_v2/outputs_v2/cascade_analysis'):
        self.cascade_dir = cascade_dir
        self.cascade_data = {}  # dataset -> list of routing decisions

    def load_cascade_logs(self, datasets=None) -> bool:
        """
        Load cascade logs for specified datasets.

        Args:
            datasets: List of dataset names (None = all found)

        Returns:
            True if at least one log loaded successfully
        """
        if not os.path.exists(self.cascade_dir):
            print(f"ERROR: Cascade directory not found: {self.cascade_dir}")
            print("Please run V2 pipeline first (Stage 2 & 3)")
            return False

        # Find all cascade log files
        log_files = [f for f in os.listdir(self.cascade_dir) if f.endswith('_cascade_log.csv')]

        if not log_files:
            print(f"ERROR: No cascade logs found in {self.cascade_dir}")
            return False

        # Filter by datasets if specified
        if datasets:
            log_files = [f for f in log_files if any(d in f for d in datasets)]

        loaded_count = 0
        for log_file in log_files:
            dataset_name = log_file.replace('_cascade_log.csv', '')
            log_path = os.path.join(self.cascade_dir, log_file)

            try:
                decisions = []
                with open(log_path, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        decisions.append({
                            'question_id': row['question_id'],
                            'initial_action': row['initial_action'],
                            'confidence': float(row['confidence']),
                            'final_action': row['final_action'],
                            'cascaded': row['cascaded'].lower() == 'true',
                            'question_text': row.get('question_text', ''),
                        })

                self.cascade_data[dataset_name] = decisions
                loaded_count += 1
                print(f"✓ Loaded {len(decisions)} cascade decisions from {dataset_name}")

            except Exception as e:
                print(f"Warning: Failed to load {log_file}: {e}")

        print(f"\n✓ Loaded {loaded_count} cascade logs successfully")
        return loaded_count > 0

    def analyze_cascade_rate(self) -> Dict:
        """
        Analyze cascade trigger rate overall and by action.

        Returns:
            Dictionary with cascade rate statistics
        """
        results = {
            'overall': {'total': 0, 'cascaded': 0, 'rate': 0.0},
            'by_action': defaultdict(lambda: {'total': 0, 'cascaded': 0, 'rate': 0.0}),
            'by_dataset': {}
        }

        for dataset_name, decisions in self.cascade_data.items():
            dataset_total = len(decisions)
            dataset_cascaded = sum(1 for d in decisions if d['cascaded'])

            results['by_dataset'][dataset_name] = {
                'total': dataset_total,
                'cascaded': dataset_cascaded,
                'rate': dataset_cascaded / dataset_total if dataset_total > 0 else 0.0
            }

            # Aggregate overall
            results['overall']['total'] += dataset_total
            results['overall']['cascaded'] += dataset_cascaded

            # By action
            for decision in decisions:
                action = decision['initial_action']
                results['by_action'][action]['total'] += 1
                if decision['cascaded']:
                    results['by_action'][action]['cascaded'] += 1

        # Calculate overall rate
        if results['overall']['total'] > 0:
            results['overall']['rate'] = results['overall']['cascaded'] / results['overall']['total']

        # Calculate action rates
        for action, stats in results['by_action'].items():
            if stats['total'] > 0:
                stats['rate'] = stats['cascaded'] / stats['total']

        return results

    def analyze_confidence_distribution(self) -> Dict:
        """
        Analyze confidence score distribution for direct vs cascaded.

        Returns:
            Dictionary with confidence statistics
        """
        direct_confidences = []
        cascaded_confidences = []

        for decisions in self.cascade_data.values():
            for decision in decisions:
                if decision['cascaded']:
                    cascaded_confidences.append(decision['confidence'])
                else:
                    direct_confidences.append(decision['confidence'])

        def calc_stats(values):
            if not values:
                return {'count': 0, 'mean': 0, 'median': 0, 'std': 0, 'min': 0, 'max': 0}

            return {
                'count': len(values),
                'mean': statistics.mean(values),
                'median': statistics.median(values),
                'std': statistics.stdev(values) if len(values) > 1 else 0,
                'min': min(values),
                'max': max(values)
            }

        # Calculate percentiles for direct confidence
        direct_percentiles = {}
        if direct_confidences:
            sorted_direct = sorted(direct_confidences)
            for p in [10, 25, 50, 75, 90]:
                idx = int(len(sorted_direct) * p / 100)
                direct_percentiles[f'p{p}'] = sorted_direct[idx]

        # Calculate percentiles for cascaded confidence
        cascaded_percentiles = {}
        if cascaded_confidences:
            sorted_cascaded = sorted(cascaded_confidences)
            for p in [10, 25, 50, 75, 90]:
                idx = int(len(sorted_cascaded) * p / 100)
                cascaded_percentiles[f'p{p}'] = sorted_cascaded[idx]

        return {
            'direct': {
                'stats': calc_stats(direct_confidences),
                'percentiles': direct_percentiles
            },
            'cascaded': {
                'stats': calc_stats(cascaded_confidences),
                'percentiles': cascaded_percentiles
            }
        }

    def analyze_threshold_sensitivity(self, threshold_range=(0.4, 0.8, 0.05)) -> List[Dict]:
        """
        Analyze how different confidence thresholds affect cascade rate.

        Args:
            threshold_range: Tuple of (start, end, step)

        Returns:
            List of dictionaries with threshold analysis
        """
        start, end, step = threshold_range
        thresholds = []
        current = start

        while current <= end:
            thresholds.append(current)
            current += step

        results = []

        for threshold in thresholds:
            total = 0
            would_cascade = 0

            for decisions in self.cascade_data.values():
                for decision in decisions:
                    total += 1
                    # Simulate cascade decision with this threshold
                    if decision['initial_action'] != 'M' and decision['confidence'] < threshold:
                        would_cascade += 1

            cascade_rate = would_cascade / total if total > 0 else 0

            results.append({
                'threshold': round(threshold, 2),
                'cascade_count': would_cascade,
                'cascade_rate': round(cascade_rate, 4),
                'direct_count': total - would_cascade
            })

        return results

    def analyze_by_action_detailed(self) -> Dict:
        """
        Detailed analysis of cascade behavior per action.

        Returns:
            Dictionary with per-action cascade details
        """
        action_analysis = defaultdict(lambda: {
            'total': 0,
            'cascaded': 0,
            'confidences': [],
            'cascaded_confidences': [],
            'direct_confidences': []
        })

        for decisions in self.cascade_data.values():
            for decision in decisions:
                action = decision['initial_action']
                confidence = decision['confidence']

                action_analysis[action]['total'] += 1
                action_analysis[action]['confidences'].append(confidence)

                if decision['cascaded']:
                    action_analysis[action]['cascaded'] += 1
                    action_analysis[action]['cascaded_confidences'].append(confidence)
                else:
                    action_analysis[action]['direct_confidences'].append(confidence)

        # Calculate statistics for each action
        results = {}
        for action, data in action_analysis.items():
            cascade_rate = data['cascaded'] / data['total'] if data['total'] > 0 else 0

            results[action] = {
                'total': data['total'],
                'cascaded': data['cascaded'],
                'cascade_rate': round(cascade_rate, 4),
                'confidence_mean': round(statistics.mean(data['confidences']), 4) if data['confidences'] else 0,
                'cascaded_confidence_mean': round(statistics.mean(data['cascaded_confidences']), 4) if data['cascaded_confidences'] else 0,
                'direct_confidence_mean': round(statistics.mean(data['direct_confidences']), 4) if data['direct_confidences'] else 0,
            }

        return results

    def find_low_confidence_examples(self, n=10) -> List[Dict]:
        """
        Find examples with lowest confidence that triggered cascade.

        Args:
            n: Number of examples to return

        Returns:
            List of example dictionaries
        """
        all_cascaded = []

        for dataset_name, decisions in self.cascade_data.items():
            for decision in decisions:
                if decision['cascaded']:
                    all_cascaded.append({
                        'dataset': dataset_name,
                        'question_id': decision['question_id'],
                        'question_text': decision['question_text'],
                        'initial_action': decision['initial_action'],
                        'confidence': decision['confidence'],
                        'final_action': decision['final_action']
                    })

        # Sort by confidence (lowest first)
        all_cascaded.sort(key=lambda x: x['confidence'])

        return all_cascaded[:n]

    def generate_report(self) -> str:
        """
        Generate comprehensive cascade analysis report.

        Returns:
            Markdown formatted report
        """
        md = []
        md.append("# Cascade Routing Analysis Report (Innovation 2)\n")
        md.append("## Overview\n")

        # Cascade rate analysis
        cascade_rate_analysis = self.analyze_cascade_rate()

        md.append("### Overall Cascade Statistics\n")
        overall = cascade_rate_analysis['overall']
        md.append(f"- **Total Questions**: {overall['total']}")
        md.append(f"- **Cascaded Questions**: {overall['cascaded']}")
        md.append(f"- **Cascade Rate**: {overall['rate']:.2%}\n")

        # By dataset
        md.append("### Cascade Rate by Dataset\n")
        md.append("| Dataset | Total | Cascaded | Cascade Rate |")
        md.append("|---------|-------|----------|--------------|")
        for dataset, stats in sorted(cascade_rate_analysis['by_dataset'].items()):
            md.append(f"| {dataset:15s} | {stats['total']:5d} | {stats['cascaded']:8d} | {stats['rate']:12.2%} |")
        md.append("")

        # By action
        md.append("### Cascade Rate by Initial Action\n")
        md.append("| Action | Total | Cascaded | Cascade Rate | Interpretation |")
        md.append("|--------|-------|----------|--------------|----------------|")

        action_analysis = self.analyze_by_action_detailed()
        for action in ['Z', 'S-Sparse', 'S-Dense', 'S-Hybrid', 'M']:
            if action not in cascade_rate_analysis['by_action']:
                continue

            stats = cascade_rate_analysis['by_action'][action]
            interpretation = ""
            if stats['rate'] > 0.3:
                interpretation = "High uncertainty"
            elif stats['rate'] > 0.1:
                interpretation = "Moderate uncertainty"
            else:
                interpretation = "High confidence"

            md.append(f"| {action:10s} | {stats['total']:5d} | {stats['cascaded']:8d} | "
                     f"{stats['rate']:12.2%} | {interpretation:14s} |")
        md.append("")

        # Confidence distribution
        md.append("## Confidence Distribution Analysis\n")
        conf_dist = self.analyze_confidence_distribution()

        md.append("### Direct (High Confidence) Predictions\n")
        direct_stats = conf_dist['direct']['stats']
        md.append(f"- **Count**: {direct_stats['count']}")
        md.append(f"- **Mean Confidence**: {direct_stats['mean']:.4f}")
        md.append(f"- **Median**: {direct_stats['median']:.4f}")
        md.append(f"- **Std Dev**: {direct_stats['std']:.4f}")
        md.append(f"- **Range**: [{direct_stats['min']:.4f}, {direct_stats['max']:.4f}]\n")

        if conf_dist['direct']['percentiles']:
            md.append("**Percentiles:**")
            for p, val in sorted(conf_dist['direct']['percentiles'].items()):
                md.append(f"  - {p}: {val:.4f}")
        md.append("")

        md.append("### Cascaded (Low Confidence) Predictions\n")
        cascaded_stats = conf_dist['cascaded']['stats']
        md.append(f"- **Count**: {cascaded_stats['count']}")
        md.append(f"- **Mean Confidence**: {cascaded_stats['mean']:.4f}")
        md.append(f"- **Median**: {cascaded_stats['median']:.4f}")
        md.append(f"- **Std Dev**: {cascaded_stats['std']:.4f}")
        md.append(f"- **Range**: [{cascaded_stats['min']:.4f}, {cascaded_stats['max']:.4f}]\n")

        if conf_dist['cascaded']['percentiles']:
            md.append("**Percentiles:**")
            for p, val in sorted(conf_dist['cascaded']['percentiles'].items()):
                md.append(f"  - {p}: {val:.4f}")
        md.append("")

        # Confidence gap
        mean_gap = direct_stats['mean'] - cascaded_stats['mean']
        md.append(f"**Confidence Gap**: {mean_gap:.4f} (Direct - Cascaded)\n")

        if mean_gap > 0.2:
            md.append("✅ **Good separation** - Clear distinction between high and low confidence cases\n")
        else:
            md.append("⚠️ **Weak separation** - Consider adjusting confidence threshold\n")

        # Threshold sensitivity
        md.append("## Threshold Sensitivity Analysis\n")
        md.append("| Threshold | Cascade Count | Cascade Rate | Direct Count |")
        md.append("|-----------|---------------|--------------|--------------|")

        threshold_analysis = self.analyze_threshold_sensitivity()
        for result in threshold_analysis:
            md.append(f"| {result['threshold']:.2f}      | {result['cascade_count']:13d} | "
                     f"{result['cascade_rate']:12.2%} | {result['direct_count']:12d} |")
        md.append("")

        # By action detailed
        md.append("## Detailed Analysis by Action\n")
        md.append("| Action | Total | Cascade Rate | Avg Conf (All) | Avg Conf (Cascaded) | Avg Conf (Direct) |")
        md.append("|--------|-------|--------------|----------------|---------------------|-------------------|")

        for action in ['Z', 'S-Sparse', 'S-Dense', 'S-Hybrid', 'M']:
            if action not in action_analysis:
                continue

            data = action_analysis[action]
            md.append(f"| {action:10s} | {data['total']:5d} | {data['cascade_rate']:12.2%} | "
                     f"{data['confidence_mean']:14.4f} | {data['cascaded_confidence_mean']:19.4f} | "
                     f"{data['direct_confidence_mean']:17.4f} |")
        md.append("")

        # Low confidence examples
        md.append("## Examples: Lowest Confidence Cascaded Questions\n")
        low_conf_examples = self.find_low_confidence_examples(n=5)

        for i, example in enumerate(low_conf_examples, 1):
            md.append(f"### Example {i}")
            md.append(f"- **Dataset**: {example['dataset']}")
            md.append(f"- **Question ID**: {example['question_id']}")
            md.append(f"- **Initial Action**: {example['initial_action']}")
            md.append(f"- **Confidence**: {example['confidence']:.4f}")
            md.append(f"- **Final Action**: {example['final_action']}")
            if example['question_text']:
                md.append(f"- **Question**: {example['question_text'][:150]}...")
            md.append("")

        # Key insights
        md.append("## Key Insights\n")

        cascade_rate_overall = cascade_rate_analysis['overall']['rate']
        md.append(f"1. **Overall cascade rate**: {cascade_rate_overall:.2%} of predictions triggered fallback")

        # Find action with highest cascade rate
        max_cascade_action = max(cascade_rate_analysis['by_action'].items(),
                                key=lambda x: x[1]['rate'])
        md.append(f"2. **Highest uncertainty action**: {max_cascade_action[0]} ({max_cascade_action[1]['rate']:.2%} cascade rate)")

        # Find action with lowest cascade rate
        min_cascade_action = min(cascade_rate_analysis['by_action'].items(),
                                key=lambda x: x[1]['rate'])
        md.append(f"3. **Most confident action**: {min_cascade_action[0]} ({min_cascade_action[1]['rate']:.2%} cascade rate)")

        md.append(f"4. **Confidence separation**: {mean_gap:.4f} mean difference between direct and cascaded")

        if cascade_rate_overall < 0.15:
            md.append(f"5. ⚠️ **Low cascade rate** - Consider lowering confidence threshold to trigger more fallbacks")
        elif cascade_rate_overall > 0.40:
            md.append(f"5. ⚠️ **High cascade rate** - Consider raising confidence threshold to reduce false positives")
        else:
            md.append(f"5. ✅ **Reasonable cascade rate** - Current threshold seems well-calibrated")

        md.append("\n---")
        md.append("\n**Innovation 2: Cascading Dynamic Routing**")
        md.append("- Confidence-based fallback mechanism")
        md.append("- Low-confidence predictions escalate to MI-RA-ToT")
        md.append("- Improves robustness for uncertain cases")

        return "\n".join(md)

    def save_analysis(self, output_dir='innovation_experiments/analysis_results'):
        """
        Save cascade analysis to files.

        Args:
            output_dir: Directory to save results
        """
        os.makedirs(output_dir, exist_ok=True)

        # Generate and save report
        report = self.generate_report()
        report_path = os.path.join(output_dir, 'cascade_analysis_report.md')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"✓ Cascade analysis report saved to {report_path}")

        # Save JSON data
        analysis_data = {
            'cascade_rate': self.analyze_cascade_rate(),
            'confidence_distribution': self.analyze_confidence_distribution(),
            'threshold_sensitivity': self.analyze_threshold_sensitivity(),
            'by_action_detailed': self.analyze_by_action_detailed(),
            'low_confidence_examples': self.find_low_confidence_examples(n=10)
        }

        json_path = os.path.join(output_dir, 'cascade_analysis_data.json')
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(analysis_data, f, indent=2, ensure_ascii=False)
        print(f"✓ Cascade analysis data saved to {json_path}")

        return report_path, json_path


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Analyze cascade routing decisions (Innovation 2)"
    )
    parser.add_argument('--datasets', nargs='+',
                       help='Datasets to analyze (default: all)')
    parser.add_argument('--cascade-dir',
                       default='innovation_experiments/evaluate_v2/outputs_v2/cascade_analysis',
                       help='Directory containing cascade logs')
    parser.add_argument('--output-dir',
                       default='innovation_experiments/analysis_results',
                       help='Directory to save analysis results')

    args = parser.parse_args()

    # Create analyzer
    analyzer = CascadeAnalyzer(cascade_dir=args.cascade_dir)

    # Load cascade logs
    if not analyzer.load_cascade_logs(datasets=args.datasets):
        sys.exit(1)

    # Perform analysis
    print("\n" + "="*80)
    print("ANALYZING CASCADE ROUTING DECISIONS")
    print("="*80)

    cascade_rate = analyzer.analyze_cascade_rate()
    print(f"\nOverall Cascade Rate: {cascade_rate['overall']['rate']:.2%} "
          f"({cascade_rate['overall']['cascaded']}/{cascade_rate['overall']['total']})")

    print("\nBy Action:")
    for action in ['Z', 'S-Sparse', 'S-Dense', 'S-Hybrid', 'M']:
        if action in cascade_rate['by_action']:
            stats = cascade_rate['by_action'][action]
            print(f"  {action:12s}: {stats['rate']:6.2%} ({stats['cascaded']}/{stats['total']})")

    # Save analysis
    report_path, json_path = analyzer.save_analysis(args.output_dir)

    print(f"\n✓ Full cascade analysis report available at: {report_path}")
    print(f"✓ JSON data available at: {json_path}")


if __name__ == '__main__':
    main()
