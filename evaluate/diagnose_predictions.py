"""
Diagnostic script to identify issues with predictions and evaluation.
This script helps debug low EM/F1 scores by analyzing:
1. Prediction file existence and format
2. Answer format issues
3. Sample comparisons between predictions and ground truth
4. Common error patterns
"""

import os
import sys
import json
from collections import defaultdict, Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluate.utils.config_loader import ConfigLoader
from evaluate.utils.data_loader import DataLoader
from evaluate.utils.result_manager import ResultManager
from metrics.squad_answer_em_f1 import normalize_answer, compute_exact, compute_f1


class PredictionDiagnostics:
    """Diagnose prediction and evaluation issues."""

    def __init__(self, config):
        self.config = config
        self.data_loader = DataLoader(config)
        self.result_manager = ResultManager(config)

    def diagnose_dataset(self, dataset_name: str, num_samples: int = 10):
        """
        Diagnose issues for a specific dataset.

        Args:
            dataset_name: Name of dataset
            num_samples: Number of sample comparisons to show
        """
        print(f"\n{'='*80}")
        print(f"DIAGNOSING: {dataset_name}")
        print(f"{'='*80}")

        # Check file existence
        stage1_path = self.result_manager.get_stage1_output_path(dataset_name)
        stage2_path = self.result_manager.get_stage2_output_path(dataset_name)

        print(f"\nFile Checks:")
        print(f"  Stage 1 classifications: {stage1_path}")
        print(f"    Exists: {os.path.exists(stage1_path)}")
        if os.path.exists(stage1_path):
            classifications = self.result_manager.load_stage1_results(dataset_name)
            print(f"    Count: {len(classifications)}")

        print(f"  Stage 2 predictions: {stage2_path}")
        print(f"    Exists: {os.path.exists(stage2_path)}")
        if not os.path.exists(stage2_path):
            print(f"\n❌ ERROR: Prediction file does not exist!")
            print(f"    Please run Stage 2 generation first.")
            return

        # Load predictions
        predictions = self.result_manager.load_stage2_results(dataset_name)
        print(f"    Count: {len(predictions)}")

        if not predictions:
            print(f"\n❌ ERROR: No predictions found!")
            return

        # Load ground truth
        test_data = self.data_loader.load_test_data(dataset_name)
        test_data_map = {item['question_id']: item for item in test_data}
        print(f"  Ground truth count: {len(test_data)}")

        # Analyze prediction formats
        print(f"\n{'='*80}")
        print(f"PREDICTION FORMAT ANALYSIS")
        print(f"{'='*80}")

        pred_types = Counter()
        empty_preds = 0
        list_preds = 0
        dict_preds = 0

        for qid, pred in predictions.items():
            pred_types[type(pred).__name__] += 1
            if pred == "" or pred == []:
                empty_preds += 1
            if isinstance(pred, list):
                list_preds += 1
            if isinstance(pred, dict):
                dict_preds += 1

        print(f"\nPrediction types:")
        for ptype, count in pred_types.most_common():
            print(f"  {ptype:20s}: {count:5d} ({count/len(predictions)*100:.1f}%)")

        print(f"\nIssues:")
        print(f"  Empty predictions: {empty_preds:5d} ({empty_preds/len(predictions)*100:.1f}%)")
        print(f"  List predictions:  {list_preds:5d} ({list_preds/len(predictions)*100:.1f}%)")
        print(f"  Dict predictions:  {dict_preds:5d} ({dict_preds/len(predictions)*100:.1f}%)")

        if empty_preds > len(predictions) * 0.5:
            print(f"\n⚠️  WARNING: More than 50% predictions are empty!")
            print(f"    This suggests inference is failing frequently.")
            print(f"    Check:")
            print(f"      - LLM server is running and accessible")
            print(f"      - Retriever service is running and accessible")
            print(f"      - No timeout errors in generation logs")

        # Extract gold answers format
        print(f"\n{'='*80}")
        print(f"GROUND TRUTH FORMAT ANALYSIS")
        print(f"{'='*80}")

        gold_formats = defaultdict(int)
        for item in test_data[:100]:  # Sample first 100
            if 'answers_objects' in item:
                gold_formats['answers_objects'] += 1
            if 'answers' in item:
                gold_formats['answers'] += 1

        print(f"\nGround truth fields (sampled from first 100):")
        for field, count in gold_formats.items():
            print(f"  {field:20s}: {count:5d}")

        # Sample comparisons
        print(f"\n{'='*80}")
        print(f"SAMPLE COMPARISONS")
        print(f"{'='*80}")

        samples_shown = 0
        exact_matches = 0
        high_f1 = 0

        for qid, pred in list(predictions.items())[:num_samples]:
            if qid not in test_data_map:
                continue

            samples_shown += 1
            gt = test_data_map[qid]

            # Extract gold answers
            gold_answers = self.extract_gold_answers(gt)

            # Convert pred to string if needed
            pred_str = pred
            if isinstance(pred, list):
                pred_str = pred[0] if pred else ""
            elif not isinstance(pred, str):
                pred_str = str(pred)

            # Compute metrics
            if gold_answers:
                em_scores = [compute_exact(pred_str, gold) for gold in gold_answers]
                f1_scores = [compute_f1(pred_str, gold) for gold in gold_answers]
                em = max(em_scores)
                f1 = max(f1_scores)
            else:
                em, f1 = 0, 0

            if em == 1:
                exact_matches += 1
            if f1 > 0.5:
                high_f1 += 1

            print(f"\nSample {samples_shown}:")
            print(f"  QID: {qid}")
            print(f"  Question: {gt.get('question_text', '')[:100]}...")
            print(f"  Prediction: '{pred_str}'")
            print(f"  Normalized: '{normalize_answer(pred_str)}'")
            print(f"  Gold answers: {gold_answers}")
            print(f"  Gold normalized: {[normalize_answer(g) for g in gold_answers]}")
            print(f"  EM: {em}, F1: {f1:.3f}")

        if samples_shown > 0:
            print(f"\nSample statistics ({samples_shown} samples):")
            print(f"  Exact matches: {exact_matches}/{samples_shown} ({exact_matches/samples_shown*100:.1f}%)")
            print(f"  High F1 (>0.5): {high_f1}/{samples_shown} ({high_f1/samples_shown*100:.1f}%)")

        # Common error patterns
        print(f"\n{'='*80}")
        print(f"COMMON ERROR PATTERNS")
        print(f"{'='*80}")

        print(f"\n1. Empty predictions:")
        empty_count = 0
        for qid, pred in list(predictions.items())[:100]:
            pred_str = pred
            if isinstance(pred, list):
                pred_str = pred[0] if pred else ""
            if pred_str == "":
                empty_count += 1
        print(f"   {empty_count}/100 samples are empty")

        print(f"\n2. Format mismatches:")
        print(f"   - Predictions are lists: {list_preds > 0}")
        print(f"   - Predictions are dicts: {dict_preds > 0}")

        print(f"\n3. Length analysis:")
        lengths = []
        for pred in list(predictions.values())[:100]:
            pred_str = pred
            if isinstance(pred, list):
                pred_str = pred[0] if pred else ""
            elif not isinstance(pred, str):
                pred_str = str(pred)
            lengths.append(len(pred_str))

        if lengths:
            avg_len = sum(lengths) / len(lengths)
            print(f"   Average prediction length: {avg_len:.1f} characters")
            print(f"   Min: {min(lengths)}, Max: {max(lengths)}")

    def extract_gold_answers(self, ground_truth: dict) -> list:
        """Extract gold answers (same logic as stage3_evaluate.py)."""
        gold_answers = []

        if 'answers_objects' in ground_truth:
            for ans_obj in ground_truth['answers_objects']:
                formatted = self.format_answer(ans_obj)
                if formatted:
                    gold_answers.append(formatted)

        if not gold_answers and 'answers' in ground_truth:
            if isinstance(ground_truth['answers'], list):
                gold_answers = [str(ans) for ans in ground_truth['answers']]
            else:
                gold_answers = [str(ground_truth['answers'])]

        return gold_answers

    def format_answer(self, answer_obj: dict) -> str:
        """Format answer from DROP-style format."""
        if isinstance(answer_obj, str):
            return answer_obj

        if answer_obj.get('number'):
            return str(answer_obj['number'])

        if answer_obj.get('spans'):
            spans = answer_obj['spans']
            if isinstance(spans, list) and len(spans) > 0:
                return str(spans[0])
            elif isinstance(spans, str):
                return spans

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

    def run(self, datasets=None, num_samples=10):
        """Run diagnostics on all datasets."""
        if datasets is None:
            datasets = self.config['datasets']

        print(f"\n{'='*80}")
        print(f"PREDICTION DIAGNOSTICS")
        print(f"{'='*80}")
        print(f"Analyzing {len(datasets)} datasets")
        print(f"Showing {num_samples} samples per dataset")

        for dataset_name in datasets:
            try:
                self.diagnose_dataset(dataset_name, num_samples)
            except Exception as e:
                print(f"\n❌ Error diagnosing {dataset_name}: {e}")
                import traceback
                traceback.print_exc()

        print(f"\n{'='*80}")
        print(f"DIAGNOSTICS COMPLETE")
        print(f"{'='*80}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Diagnose prediction issues")
    parser.add_argument('--config', default='evaluate/config.yaml',
                       help='Path to config file')
    parser.add_argument('--datasets', nargs='+',
                       help='Datasets to diagnose (default: all)')
    parser.add_argument('--samples', type=int, default=10,
                       help='Number of samples to show per dataset')

    args = parser.parse_args()

    config = ConfigLoader.load_config(args.config)
    diagnostics = PredictionDiagnostics(config)
    diagnostics.run(datasets=args.datasets, num_samples=args.samples)


if __name__ == '__main__':
    main()
