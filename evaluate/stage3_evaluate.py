"""
Stage 3: Evaluation
Calculate metrics from generated predictions.
"""

import os
import sys
import json
from collections import defaultdict

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluate.utils.config_loader import ConfigLoader
from evaluate.utils.data_loader import DataLoader
from evaluate.utils.result_manager import ResultManager

# Import metrics
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from metrics.squad_answer_em_f1 import SquadAnswerEmF1Metric


class Stage3Evaluator:
    """Stage 3: Calculate metrics from predictions."""

    def __init__(self, config):
        self.config = config
        self.data_loader = DataLoader(config)
        self.result_manager = ResultManager(config)

    def run(self, datasets=None):
        """
        Evaluate all datasets and aggregate results.

        Args:
            datasets: List of dataset names to process (None = all)
        """
        if datasets is None:
            datasets = self.config['datasets']

        print("\n" + "="*60)
        print("STAGE 3: EVALUATION")
        print("="*60)

        all_metrics = {}

        for dataset_name in datasets:
            print(f"\n{'='*60}")
            print(f"Evaluating dataset: {dataset_name}")
            print(f"{'='*60}")

            metrics = self.evaluate_dataset(dataset_name)
            if metrics:
                all_metrics[dataset_name] = metrics

        if not all_metrics:
            print("\nNo metrics calculated. Please check that Stage 2 has been run.")
            return

        # Aggregate overall metrics
        overall = self.aggregate_metrics(all_metrics)
        all_metrics['overall'] = overall

        # Save metrics
        self.save_metrics(all_metrics)

        # Print report
        self.print_report(all_metrics)

        print("\n" + "="*60)
        print("✓ STAGE 3 COMPLETE")
        print("="*60)

    def evaluate_dataset(self, dataset_name: str) -> dict:
        """
        Calculate EM and F1 for a dataset.

        Args:
            dataset_name: Name of dataset to evaluate

        Returns:
            Dictionary of metrics
        """
        # Load predictions
        predictions = self.result_manager.load_stage2_results(dataset_name)

        if not predictions:
            print(f"  Warning: No predictions found for {dataset_name}")
            print(f"  Please run Stage 2 first for this dataset")
            return None

        # Load ground truth
        test_data = self.data_loader.load_test_data(dataset_name)
        test_data_map = {item['question_id']: item for item in test_data}

        # Load classifications to get action distribution
        classifications = self.result_manager.load_stage1_results(dataset_name)

        # Initialize metrics
        overall_metric = SquadAnswerEmF1Metric()
        action_metrics = defaultdict(SquadAnswerEmF1Metric)

        # Calculate metrics
        for qid, pred_answer in predictions.items():
            if qid not in test_data_map:
                print(f"  Warning: {qid} not in test data")
                continue

            ground_truth = test_data_map[qid]

            # Extract gold answers
            gold_answers = self.extract_gold_answers(ground_truth)

            if not gold_answers:
                print(f"  Warning: No gold answers for {qid}")
                continue

            # Update overall metric
            overall_metric(pred_answer, gold_answers)

            # Update action-specific metric
            if qid in classifications:
                action = classifications[qid]['predicted_action']
                action_metrics[action](pred_answer, gold_answers)

        # Get results
        results = {
            'overall': overall_metric.get_metric(reset=False),
            'by_action': {
                action: metric.get_metric(reset=False)
                for action, metric in action_metrics.items()
                if metric._count > 0
            },
            'action_distribution': {
                action: len([qid for qid, c in classifications.items()
                           if c['predicted_action'] == action])
                for action in set(c['predicted_action'] for c in classifications.values())
            }
        }

        return results

    def extract_gold_answers(self, ground_truth: dict) -> list:
        """
        Extract gold answers from ground truth data.

        Args:
            ground_truth: Ground truth item

        Returns:
            List of gold answer strings
        """
        gold_answers = []

        if 'answers_objects' in ground_truth:
            for ans_obj in ground_truth['answers_objects']:
                formatted = self.format_answer(ans_obj)
                if formatted:
                    gold_answers.append(formatted)

        # Fallback: check for 'answers' field
        if not gold_answers and 'answers' in ground_truth:
            if isinstance(ground_truth['answers'], list):
                gold_answers = [str(ans) for ans in ground_truth['answers']]
            else:
                gold_answers = [str(ground_truth['answers'])]

        return gold_answers

    def format_answer(self, answer_obj: dict) -> str:
        """
        Format answer from DROP-style format.

        Args:
            answer_obj: Answer object with number/spans/date fields

        Returns:
            Formatted answer string
        """
        if isinstance(answer_obj, str):
            return answer_obj

        # Handle number
        if answer_obj.get('number'):
            return str(answer_obj['number'])

        # Handle spans
        if answer_obj.get('spans'):
            spans = answer_obj['spans']
            if isinstance(spans, list) and len(spans) > 0:
                return str(spans[0])
            elif isinstance(spans, str):
                return spans

        # Handle date
        date = answer_obj.get('date', {})
        if date:
            parts = []
            if date.get('day'):
                parts.append(str(date['day']))
            if date.get('month'):
                parts.append(str(date['month']))
            if date.get('year'):
                parts.append(str(date['year']))
            if parts:
                return '-'.join(parts)

        return None

    def aggregate_metrics(self, all_metrics: dict) -> dict:
        """
        Aggregate metrics across all datasets.

        Args:
            all_metrics: Dictionary mapping dataset to metrics

        Returns:
            Aggregated overall metrics
        """
        total_em = 0
        total_f1 = 0
        total_acc = 0
        total_recall = 0
        total_count = 0

        for dataset_name, metrics in all_metrics.items():
            if 'overall' in metrics:
                overall = metrics['overall']
                total_em += overall['em'] * overall['count']
                total_f1 += overall['f1'] * overall['count']
                total_acc += overall.get('acc', 0) * overall['count']
                total_recall += overall.get('recall', 0) * overall['count']
                total_count += overall['count']

        return {
            'em': round(total_em / total_count, 4) if total_count > 0 else 0,
            'f1': round(total_f1 / total_count, 4) if total_count > 0 else 0,
            'acc': round(total_acc / total_count, 4) if total_count > 0 else 0,
            'recall': round(total_recall / total_count, 4) if total_count > 0 else 0,
            'count': total_count
        }

    def save_metrics(self, all_metrics: dict):
        """Save metrics to JSON file."""
        output_path = self.result_manager.get_stage3_output_path()

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(all_metrics, f, indent=2, ensure_ascii=False)

        print(f"\n✓ Metrics saved to {output_path}")

        # Also save detailed report
        report_path = os.path.join(
            self.config['outputs']['stage3_dir'],
            'detailed_report.txt'
        )
        self.save_detailed_report(all_metrics, report_path)
        print(f"✓ Detailed report saved to {report_path}")

    def save_detailed_report(self, all_metrics: dict, output_path: str):
        """Save detailed text report."""
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("DETAILED EVALUATION REPORT\n")
            f.write("="*80 + "\n\n")

            for dataset_name in sorted(all_metrics.keys()):
                if dataset_name == 'overall':
                    continue

                metrics = all_metrics[dataset_name]

                f.write(f"\nDataset: {dataset_name}\n")
                f.write("-" * 60 + "\n")

                # Overall metrics for this dataset
                if 'overall' in metrics:
                    overall = metrics['overall']
                    f.write(f"Overall Performance:\n")
                    f.write(f"  EM:     {overall['em']:.4f}\n")
                    f.write(f"  F1:     {overall['f1']:.4f}\n")
                    f.write(f"  ACC:    {overall.get('acc', 0):.4f}\n")
                    f.write(f"  Recall: {overall.get('recall', 0):.4f}\n")
                    f.write(f"  Count:  {overall['count']}\n\n")

                # Action distribution
                if 'action_distribution' in metrics:
                    f.write(f"Action Distribution:\n")
                    for action, count in sorted(metrics['action_distribution'].items()):
                        f.write(f"  {action:12s}: {count:4d}\n")
                    f.write("\n")

                # By action metrics
                if 'by_action' in metrics:
                    f.write(f"Performance by Action:\n")
                    for action in ['Z', 'S-Sparse', 'S-Dense', 'S-Hybrid', 'M', 'Unknown']:
                        if action in metrics['by_action']:
                            action_metrics = metrics['by_action'][action]
                            f.write(f"  {action:12s}: EM={action_metrics['em']:.4f}, "
                                  f"F1={action_metrics['f1']:.4f}, "
                                  f"ACC={action_metrics.get('acc', 0):.4f}, "
                                  f"Recall={action_metrics.get('recall', 0):.4f}, "
                                  f"Count={action_metrics['count']}\n")
                    f.write("\n")

            # Overall across all datasets
            f.write("\n" + "="*80 + "\n")
            f.write("OVERALL ACROSS ALL DATASETS\n")
            f.write("="*80 + "\n")
            if 'overall' in all_metrics:
                overall = all_metrics['overall']
                f.write(f"EM:              {overall['em']:.4f}\n")
                f.write(f"F1:              {overall['f1']:.4f}\n")
                f.write(f"ACC:             {overall.get('acc', 0):.4f}\n")
                f.write(f"Recall:          {overall.get('recall', 0):.4f}\n")
                f.write(f"Total Questions: {overall['count']}\n")

    def print_report(self, all_metrics: dict):
        """Print evaluation report to console."""
        print("\n" + "="*80)
        print("EVALUATION RESULTS")
        print("="*80)

        for dataset_name in sorted(all_metrics.keys()):
            if dataset_name == 'overall':
                continue

            metrics = all_metrics[dataset_name]

            print(f"\n{dataset_name}:")
            if 'overall' in metrics:
                overall = metrics['overall']
                print(f"  Overall: EM={overall['em']:.4f}, F1={overall['f1']:.4f}, "
                     f"ACC={overall.get('acc', 0):.4f}, Recall={overall.get('recall', 0):.4f}, "
                     f"Count={overall['count']}")

            if 'by_action' in metrics:
                print(f"  By Action:")
                for action in ['Z', 'S-Sparse', 'S-Dense', 'S-Hybrid', 'M', 'Unknown']:
                    if action in metrics['by_action']:
                        action_metrics = metrics['by_action'][action]
                        print(f"    {action:12s}: EM={action_metrics['em']:.4f}, "
                             f"F1={action_metrics['f1']:.4f}, "
                             f"ACC={action_metrics.get('acc', 0):.4f}, "
                             f"Recall={action_metrics.get('recall', 0):.4f}, "
                             f"Count={action_metrics['count']}")

        print(f"\n{'='*80}")
        print(f"OVERALL ACROSS ALL DATASETS")
        print(f"{'='*80}")
        if 'overall' in all_metrics:
            overall = all_metrics['overall']
            print(f"EM:              {overall['em']:.4f}")
            print(f"F1:              {overall['f1']:.4f}")
            print(f"ACC:             {overall.get('acc', 0):.4f}")
            print(f"Recall:          {overall.get('recall', 0):.4f}")
            print(f"Total Questions: {overall['count']}")


def main():
    """Main entry point for Stage 3."""
    import argparse

    parser = argparse.ArgumentParser(description="Stage 3: Evaluation")
    parser.add_argument('--config', default='evaluate/config.yaml',
                       help='Path to config file')
    parser.add_argument('--datasets', nargs='+',
                       help='Datasets to evaluate (default: all in config)')

    args = parser.parse_args()

    # Load config
    config = ConfigLoader.load_config(args.config)

    # Run evaluation
    evaluator = Stage3Evaluator(config)
    evaluator.run(datasets=args.datasets)


if __name__ == '__main__':
    main()
