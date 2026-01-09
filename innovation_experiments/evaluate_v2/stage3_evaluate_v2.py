"""
Stage 3: Evaluation with Cascade Analysis (V2)

Extends baseline evaluation with:
- Standard EM/F1 metrics
- Cascade routing analysis
- Confidence distribution analysis
- Direct vs cascaded performance comparison
"""

import os
import sys
import json
import csv
from collections import defaultdict
from typing import Dict, List, Optional

# Add paths for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from evaluate.utils.config_loader import ConfigLoader
from evaluate.utils.data_loader import DataLoader
from evaluate_v2.utils.result_manager_v2 import ResultManagerV2
from metrics.squad_answer_em_f1 import SquadAnswerEmF1Metric


class Stage3EvaluatorV2:
    """
    Stage 3 Evaluator with Innovation Analysis (V2).

    Calculates:
    1. Standard metrics (EM, F1, ACC, Recall) - same as baseline
    2. Cascade-specific metrics (cascade rate, improvement rate, confidence analysis)
    3. Action-specific performance with cascade breakdown
    """

    def __init__(self, config):
        self.config = config
        self.data_loader = DataLoader(config)
        self.result_manager = ResultManagerV2(config)

    def run(self, datasets=None):
        """
        Evaluate all datasets with cascade analysis.

        Args:
            datasets: List of dataset names to process (None = all)
        """
        if datasets is None:
            datasets = self.config['datasets']

        print("\n" + "="*80)
        print("STAGE 3: EVALUATION (V2 with Cascade Analysis)")
        print("="*80)

        all_metrics = {}
        all_cascade_analysis = {}

        for dataset_name in datasets:
            print(f"\n{'='*80}")
            print(f"Evaluating dataset: {dataset_name}")
            print(f"{'='*80}")

            # Standard metrics
            metrics = self.evaluate_dataset(dataset_name)
            if metrics:
                all_metrics[dataset_name] = metrics

            # Cascade analysis
            cascade_analysis = self.analyze_cascade_performance(dataset_name)
            if cascade_analysis:
                all_cascade_analysis[dataset_name] = cascade_analysis

        if not all_metrics:
            print("\nNo metrics calculated. Please check that Stage 2 has been run.")
            return

        # Aggregate overall metrics
        overall = self.aggregate_metrics(all_metrics)
        all_metrics['overall'] = overall

        # Aggregate cascade metrics
        overall_cascade = self.aggregate_cascade_metrics(all_cascade_analysis)
        all_cascade_analysis['overall'] = overall_cascade

        # Save results
        self.save_metrics(all_metrics)
        self.save_cascade_analysis(all_cascade_analysis)

        # Print reports
        self.print_report(all_metrics)
        self.print_cascade_report(all_cascade_analysis)

        print("\n" + "="*80)
        print("✓ STAGE 3 (V2) COMPLETE")
        print("="*80)
        print(f"\nResults saved to:")
        print(f"  Metrics: {self.config['outputs']['stage3_dir']}")
        print(f"  Cascade Analysis: {self.config['outputs']['cascade_dir']}")

    def evaluate_dataset(self, dataset_name: str) -> Optional[Dict]:
        """
        Calculate EM and F1 for a dataset (same as baseline).

        Args:
            dataset_name: Name of dataset to evaluate

        Returns:
            Dictionary of metrics
        """
        # Load predictions (V2 paths)
        predictions = self.result_manager.load_stage2_results(dataset_name)

        if not predictions:
            print(f"  Warning: No predictions found for {dataset_name}")
            print(f"  Please run Stage 2 (V2) first for this dataset")
            return None

        # Load ground truth
        test_data = self.data_loader.load_test_data(dataset_name)
        test_data_map = {item['question_id']: item for item in test_data}

        # Load classifications (SHARED with baseline)
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

    def analyze_cascade_performance(self, dataset_name: str) -> Optional[Dict]:
        """
        Analyze cascade routing decisions and performance.

        Returns:
            Dictionary with cascade analysis metrics
        """
        # Load cascade log
        cascade_log_path = self.result_manager.get_cascade_log_path(dataset_name)

        if not os.path.exists(cascade_log_path):
            print(f"  Warning: No cascade log found for {dataset_name}")
            print(f"  Expected at: {cascade_log_path}")
            return None

        # Load routing decisions
        routing_decisions = {}
        with open(cascade_log_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                qid = row['question_id']
                routing_decisions[qid] = {
                    'initial_action': row['initial_action'],
                    'confidence': float(row['confidence']),
                    'final_action': row['final_action'],
                    'cascaded': row['cascaded'].lower() == 'true',
                    'question_text': row.get('question_text', ''),
                }

        if not routing_decisions:
            print(f"  Warning: Empty cascade log for {dataset_name}")
            return None

        # Load predictions and ground truth
        predictions = self.result_manager.load_stage2_results(dataset_name)
        test_data = self.data_loader.load_test_data(dataset_name)
        test_data_map = {item['question_id']: item for item in test_data}

        # Calculate metrics
        total_questions = len(routing_decisions)
        cascaded_questions = [qid for qid, d in routing_decisions.items() if d['cascaded']]
        direct_questions = [qid for qid, d in routing_decisions.items() if not d['cascaded']]

        # EM metrics for direct vs cascaded
        direct_em_metric = SquadAnswerEmF1Metric()
        cascaded_em_metric = SquadAnswerEmF1Metric()

        # Confidence distributions
        direct_confidences = []
        cascaded_confidences = []

        # By initial action analysis
        by_initial_action = defaultdict(lambda: {
            'total': 0,
            'cascaded': 0,
            'direct_correct': 0,
            'direct_total': 0,
            'cascaded_correct': 0,
            'cascaded_total': 0,
        })

        # Improvement tracking
        improvement_cases = []  # Cases where cascade improved result
        degradation_cases = []  # Cases where cascade hurt result
        no_change_cases = []    # Cases where cascade didn't change result

        for qid, decision in routing_decisions.items():
            if qid not in predictions or qid not in test_data_map:
                continue

            pred_answer = predictions[qid]
            gold_answers = self.extract_gold_answers(test_data_map[qid])

            if not gold_answers:
                continue

            initial_action = decision['initial_action']
            confidence = decision['confidence']
            cascaded = decision['cascaded']

            by_initial_action[initial_action]['total'] += 1

            if cascaded:
                # Cascaded case
                cascaded_confidences.append(confidence)
                cascaded_em_metric(pred_answer, gold_answers)
                by_initial_action[initial_action]['cascaded'] += 1
                by_initial_action[initial_action]['cascaded_total'] += 1

                # Check if correct
                em_result = cascaded_em_metric.get_metric(reset=False)
                if em_result['em'] > 0:  # Last prediction was correct
                    by_initial_action[initial_action]['cascaded_correct'] += 1

                # For improvement analysis, we'd need original prediction
                # Since we only have final prediction, we track cascade success rate
                # This is a simplification - ideally we'd store both predictions

            else:
                # Direct case
                direct_confidences.append(confidence)
                direct_em_metric(pred_answer, gold_answers)
                by_initial_action[initial_action]['direct_total'] += 1

                # Check if correct
                em_result = direct_em_metric.get_metric(reset=False)
                if em_result['em'] > 0:  # Last prediction was correct
                    by_initial_action[initial_action]['direct_correct'] += 1

        # Calculate aggregate metrics
        cascade_rate = len(cascaded_questions) / total_questions if total_questions > 0 else 0

        direct_metrics = direct_em_metric.get_metric(reset=False)
        cascaded_metrics = cascaded_em_metric.get_metric(reset=False)

        # Confidence statistics
        def calc_stats(values):
            if not values:
                return {'mean': 0, 'min': 0, 'max': 0, 'std': 0}
            import statistics
            return {
                'mean': statistics.mean(values),
                'min': min(values),
                'max': max(values),
                'std': statistics.stdev(values) if len(values) > 1 else 0
            }

        analysis = {
            'total_questions': total_questions,
            'cascade_rate': round(cascade_rate, 4),
            'direct_count': len(direct_questions),
            'cascaded_count': len(cascaded_questions),

            'direct_performance': {
                'em': round(direct_metrics['em'], 4),
                'f1': round(direct_metrics['f1'], 4),
                'count': direct_metrics['count']
            },

            'cascaded_performance': {
                'em': round(cascaded_metrics['em'], 4),
                'f1': round(cascaded_metrics['f1'], 4),
                'count': cascaded_metrics['count']
            },

            'confidence_stats': {
                'direct': calc_stats(direct_confidences),
                'cascaded': calc_stats(cascaded_confidences)
            },

            'by_initial_action': {
                action: {
                    'total': stats['total'],
                    'cascade_rate': round(stats['cascaded'] / stats['total'], 4) if stats['total'] > 0 else 0,
                    'direct_em': round(stats['direct_correct'] / stats['direct_total'], 4) if stats['direct_total'] > 0 else 0,
                    'cascaded_em': round(stats['cascaded_correct'] / stats['cascaded_total'], 4) if stats['cascaded_total'] > 0 else 0,
                    'direct_count': stats['direct_total'],
                    'cascaded_count': stats['cascaded_total'],
                }
                for action, stats in by_initial_action.items()
            }
        }

        print(f"  Cascade Analysis:")
        print(f"    Total questions: {total_questions}")
        print(f"    Cascade rate: {cascade_rate:.2%}")
        print(f"    Direct EM: {direct_metrics['em']:.4f} (n={direct_metrics['count']})")
        print(f"    Cascaded EM: {cascaded_metrics['em']:.4f} (n={cascaded_metrics['count']})")

        return analysis

    def extract_gold_answers(self, ground_truth: dict) -> List[str]:
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

    def format_answer(self, answer_obj: dict) -> Optional[str]:
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

    def aggregate_metrics(self, all_metrics: Dict) -> Dict:
        """
        Aggregate metrics across all datasets (same as baseline).

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

    def aggregate_cascade_metrics(self, all_cascade_analysis: Dict) -> Dict:
        """
        Aggregate cascade metrics across all datasets.

        Args:
            all_cascade_analysis: Dictionary mapping dataset to cascade analysis

        Returns:
            Aggregated cascade metrics
        """
        if not all_cascade_analysis:
            return {}

        total_questions = 0
        total_cascaded = 0
        total_direct = 0

        total_direct_em = 0
        total_direct_f1 = 0
        total_cascaded_em = 0
        total_cascaded_f1 = 0

        all_direct_confidences = []
        all_cascaded_confidences = []

        for dataset_name, analysis in all_cascade_analysis.items():
            total_questions += analysis['total_questions']
            total_cascaded += analysis['cascaded_count']
            total_direct += analysis['direct_count']

            # Weighted metrics
            direct_perf = analysis['direct_performance']
            cascaded_perf = analysis['cascaded_performance']

            total_direct_em += direct_perf['em'] * direct_perf['count']
            total_direct_f1 += direct_perf['f1'] * direct_perf['count']
            total_cascaded_em += cascaded_perf['em'] * cascaded_perf['count']
            total_cascaded_f1 += cascaded_perf['f1'] * cascaded_perf['count']

        overall = {
            'total_questions': total_questions,
            'cascade_rate': round(total_cascaded / total_questions, 4) if total_questions > 0 else 0,
            'direct_count': total_direct,
            'cascaded_count': total_cascaded,

            'direct_performance': {
                'em': round(total_direct_em / total_direct, 4) if total_direct > 0 else 0,
                'f1': round(total_direct_f1 / total_direct, 4) if total_direct > 0 else 0,
                'count': total_direct
            },

            'cascaded_performance': {
                'em': round(total_cascaded_em / total_cascaded, 4) if total_cascaded > 0 else 0,
                'f1': round(total_cascaded_f1 / total_cascaded, 4) if total_cascaded > 0 else 0,
                'count': total_cascaded
            }
        }

        return overall

    def save_metrics(self, all_metrics: Dict):
        """Save standard metrics to JSON file."""
        output_path = self.result_manager.get_stage3_output_path()

        os.makedirs(os.path.dirname(output_path), exist_ok=True)

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

    def save_cascade_analysis(self, all_cascade_analysis: Dict):
        """Save cascade analysis to JSON file."""
        cascade_dir = self.config['outputs']['cascade_dir']
        os.makedirs(cascade_dir, exist_ok=True)

        output_path = os.path.join(cascade_dir, 'cascade_analysis.json')

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(all_cascade_analysis, f, indent=2, ensure_ascii=False)

        print(f"✓ Cascade analysis saved to {output_path}")

        # Save detailed cascade report
        report_path = os.path.join(cascade_dir, 'cascade_report.txt')
        self.save_cascade_detailed_report(all_cascade_analysis, report_path)
        print(f"✓ Cascade report saved to {report_path}")

    def save_detailed_report(self, all_metrics: Dict, output_path: str):
        """Save detailed text report (same as baseline)."""
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("DETAILED EVALUATION REPORT (V2 with Innovations)\n")
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

    def save_cascade_detailed_report(self, all_cascade_analysis: Dict, output_path: str):
        """Save detailed cascade analysis report."""
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("CASCADE ROUTING ANALYSIS REPORT (Innovation 2)\n")
            f.write("="*80 + "\n\n")

            for dataset_name in sorted(all_cascade_analysis.keys()):
                if dataset_name == 'overall':
                    continue

                analysis = all_cascade_analysis[dataset_name]

                f.write(f"\nDataset: {dataset_name}\n")
                f.write("-" * 60 + "\n")

                f.write(f"Total Questions:  {analysis['total_questions']}\n")
                f.write(f"Cascade Rate:     {analysis['cascade_rate']:.2%}\n")
                f.write(f"Direct Count:     {analysis['direct_count']}\n")
                f.write(f"Cascaded Count:   {analysis['cascaded_count']}\n\n")

                # Performance comparison
                f.write("Performance Comparison:\n")
                f.write(f"  Direct:    EM={analysis['direct_performance']['em']:.4f}, "
                       f"F1={analysis['direct_performance']['f1']:.4f}, "
                       f"n={analysis['direct_performance']['count']}\n")
                f.write(f"  Cascaded:  EM={analysis['cascaded_performance']['em']:.4f}, "
                       f"F1={analysis['cascaded_performance']['f1']:.4f}, "
                       f"n={analysis['cascaded_performance']['count']}\n\n")

                # Confidence statistics
                f.write("Confidence Statistics:\n")
                direct_conf = analysis['confidence_stats']['direct']
                cascaded_conf = analysis['confidence_stats']['cascaded']
                f.write(f"  Direct:    mean={direct_conf['mean']:.4f}, "
                       f"std={direct_conf['std']:.4f}, "
                       f"range=[{direct_conf['min']:.4f}, {direct_conf['max']:.4f}]\n")
                f.write(f"  Cascaded:  mean={cascaded_conf['mean']:.4f}, "
                       f"std={cascaded_conf['std']:.4f}, "
                       f"range=[{cascaded_conf['min']:.4f}, {cascaded_conf['max']:.4f}]\n\n")

                # By initial action
                if 'by_initial_action' in analysis:
                    f.write("Performance by Initial Action:\n")
                    for action in sorted(analysis['by_initial_action'].keys()):
                        action_stats = analysis['by_initial_action'][action]
                        f.write(f"  {action:12s}: cascade_rate={action_stats['cascade_rate']:.2%}, "
                               f"direct_em={action_stats['direct_em']:.4f} (n={action_stats['direct_count']}), "
                               f"cascaded_em={action_stats['cascaded_em']:.4f} (n={action_stats['cascaded_count']})\n")
                    f.write("\n")

            # Overall
            f.write("\n" + "="*80 + "\n")
            f.write("OVERALL CASCADE ANALYSIS\n")
            f.write("="*80 + "\n")
            if 'overall' in all_cascade_analysis:
                overall = all_cascade_analysis['overall']
                f.write(f"Total Questions:      {overall['total_questions']}\n")
                f.write(f"Overall Cascade Rate: {overall['cascade_rate']:.2%}\n\n")

                f.write(f"Direct Performance:\n")
                f.write(f"  EM:    {overall['direct_performance']['em']:.4f}\n")
                f.write(f"  F1:    {overall['direct_performance']['f1']:.4f}\n")
                f.write(f"  Count: {overall['direct_performance']['count']}\n\n")

                f.write(f"Cascaded Performance:\n")
                f.write(f"  EM:    {overall['cascaded_performance']['em']:.4f}\n")
                f.write(f"  F1:    {overall['cascaded_performance']['f1']:.4f}\n")
                f.write(f"  Count: {overall['cascaded_performance']['count']}\n\n")

                # Key insight
                direct_em = overall['direct_performance']['em']
                cascaded_em = overall['cascaded_performance']['em']
                improvement = cascaded_em - direct_em
                f.write(f"Cascade Improvement: {improvement:+.4f} EM points\n")
                if improvement > 0:
                    f.write("  ⇒ Cascade mechanism is EFFECTIVE (improved low-confidence predictions)\n")
                elif improvement < 0:
                    f.write("  ⇒ Cascade mechanism needs tuning (degraded performance)\n")
                else:
                    f.write("  ⇒ Cascade mechanism has neutral effect\n")

    def print_report(self, all_metrics: Dict):
        """Print standard evaluation report to console."""
        print("\n" + "="*80)
        print("EVALUATION RESULTS (V2)")
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

    def print_cascade_report(self, all_cascade_analysis: Dict):
        """Print cascade analysis report to console."""
        if not all_cascade_analysis:
            return

        print("\n" + "="*80)
        print("CASCADE ROUTING ANALYSIS")
        print("="*80)

        for dataset_name in sorted(all_cascade_analysis.keys()):
            if dataset_name == 'overall':
                continue

            analysis = all_cascade_analysis[dataset_name]

            print(f"\n{dataset_name}:")
            print(f"  Cascade Rate: {analysis['cascade_rate']:.2%} "
                  f"({analysis['cascaded_count']}/{analysis['total_questions']})")
            print(f"  Direct:    EM={analysis['direct_performance']['em']:.4f}, "
                  f"F1={analysis['direct_performance']['f1']:.4f}, "
                  f"n={analysis['direct_performance']['count']}")
            print(f"  Cascaded:  EM={analysis['cascaded_performance']['em']:.4f}, "
                  f"F1={analysis['cascaded_performance']['f1']:.4f}, "
                  f"n={analysis['cascaded_performance']['count']}")

        print(f"\n{'='*80}")
        print(f"OVERALL CASCADE PERFORMANCE")
        print(f"{'='*80}")
        if 'overall' in all_cascade_analysis:
            overall = all_cascade_analysis['overall']
            print(f"Cascade Rate:     {overall['cascade_rate']:.2%}")
            print(f"Direct EM:        {overall['direct_performance']['em']:.4f}")
            print(f"Cascaded EM:      {overall['cascaded_performance']['em']:.4f}")
            improvement = overall['cascaded_performance']['em'] - overall['direct_performance']['em']
            print(f"Improvement:      {improvement:+.4f}")


def main():
    """Main entry point for Stage 3 (V2)."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Stage 3: Evaluation with Cascade Analysis (V2)"
    )
    parser.add_argument('--config',
                       default='innovation_experiments/evaluate_v2/config_v2.yaml',
                       help='Path to config file (default: config_v2.yaml)')
    parser.add_argument('--datasets', nargs='+',
                       help='Datasets to evaluate (default: all in config)')

    args = parser.parse_args()

    # Load config
    config = ConfigLoader.load_config(args.config)

    # Run evaluation
    evaluator = Stage3EvaluatorV2(config)
    evaluator.run(datasets=args.datasets)


if __name__ == '__main__':
    main()
