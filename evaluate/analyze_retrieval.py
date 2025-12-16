#!/usr/bin/env python3
"""
Analyze retrieved documents to diagnose retrieval quality issues.
Helps identify if wrong Wikipedia titles in predictions are due to retrieval errors.
"""

import json
import sys
import os
from collections import defaultdict, Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluate.utils.config_loader import ConfigLoader
from evaluate.utils.data_loader import DataLoader


class RetrievalAnalyzer:
    """Analyze retrieval quality and identify issues."""

    def __init__(self, config):
        self.config = config
        self.data_loader = DataLoader(config)

    def analyze_dataset(self, dataset_name: str, num_samples: int = 20):
        """
        Analyze retrieval quality for a dataset.

        Args:
            dataset_name: Name of dataset
            num_samples: Number of samples to analyze in detail
        """
        print(f"\n{'='*80}")
        print(f"ANALYZING RETRIEVAL: {dataset_name}")
        print(f"{'='*80}")

        # Load contexts
        contexts_file = f"evaluate/outputs/stage2_predictions/{dataset_name}_predictions_contexts.json"

        if not os.path.exists(contexts_file):
            print(f"\n❌ ERROR: Contexts file not found: {contexts_file}")
            print(f"    Please run Stage 2 generation first to create this file.")
            return

        with open(contexts_file, 'r', encoding='utf-8') as f:
            contexts = json.load(f)

        # Load predictions
        predictions_file = f"evaluate/outputs/stage2_predictions/{dataset_name}_predictions.json"
        with open(predictions_file, 'r', encoding='utf-8') as f:
            predictions = json.load(f)

        # Load test data
        test_data = self.data_loader.load_test_data(dataset_name)
        test_data_map = {item['question_id']: item for item in test_data}

        print(f"\nDataset Statistics:")
        print(f"  Total questions: {len(test_data)}")
        print(f"  Questions with contexts: {len(contexts)}")
        print(f"  Questions with predictions: {len(predictions)}")

        # Analyze retrieval statistics
        self.analyze_retrieval_stats(contexts)

        # Analyze samples
        self.analyze_samples(
            contexts, predictions, test_data_map,
            dataset_name, num_samples
        )

    def analyze_retrieval_stats(self, contexts: dict):
        """Analyze overall retrieval statistics."""
        print(f"\n{'='*80}")
        print(f"RETRIEVAL STATISTICS")
        print(f"{'='*80}")

        num_docs_list = []
        empty_retrievals = 0

        for qid, context in contexts.items():
            num_docs = len(context.get('titles', []))
            num_docs_list.append(num_docs)
            if num_docs == 0:
                empty_retrievals += 1

        if num_docs_list:
            avg_docs = sum(num_docs_list) / len(num_docs_list)
            print(f"\nRetrieved documents per question:")
            print(f"  Average: {avg_docs:.2f}")
            print(f"  Min: {min(num_docs_list)}")
            print(f"  Max: {max(num_docs_list)}")
            print(f"  Empty retrievals: {empty_retrievals} ({empty_retrievals/len(num_docs_list)*100:.1f}%)")

    def analyze_samples(self, contexts: dict, predictions: dict,
                       test_data_map: dict, dataset_name: str,
                       num_samples: int):
        """Analyze sample questions in detail."""
        print(f"\n{'='*80}")
        print(f"SAMPLE ANALYSIS (showing {num_samples} examples)")
        print(f"{'='*80}")

        # Find questions where prediction looks like a Wikipedia title
        title_pattern_preds = []
        for qid, pred in predictions.items():
            pred_str = str(pred).lower()
            if ('wikipedia title' in pred_str or
                'the great' in pred_str or
                pred_str.startswith('ipedia') or
                pred_str.startswith('ikipedia')):
                title_pattern_preds.append(qid)

        print(f"\n⚠️  Found {len(title_pattern_preds)} predictions that look like Wikipedia titles")
        print(f"    (These are likely retrieval or answer extraction errors)")

        # Analyze first N samples
        sample_count = 0
        for qid in list(contexts.keys())[:num_samples]:
            if qid not in test_data_map:
                continue

            sample_count += 1
            question_data = test_data_map[qid]
            context = contexts[qid]
            prediction = predictions.get(qid, "")

            print(f"\n{'-'*80}")
            print(f"Sample {sample_count}: {qid}")
            print(f"{'-'*80}")

            # Question
            question = question_data.get('question_text', '')
            print(f"Question: {question[:100]}...")

            # Prediction
            pred_str = str(prediction)
            print(f"Prediction: '{pred_str[:100]}{'...' if len(pred_str) > 100 else ''}'")

            # Retrieved titles
            titles = context.get('titles', [])
            print(f"\nRetrieved {len(titles)} documents:")

            for i, title in enumerate(titles[:5], 1):  # Show first 5
                print(f"  {i}. {title}")

            if len(titles) > 5:
                print(f"  ... and {len(titles) - 5} more")

            # Check if prediction matches any retrieved title
            pred_lower = pred_str.lower()
            title_match = False
            for title in titles:
                if title.lower() in pred_lower or pred_lower in title.lower():
                    title_match = True
                    print(f"\n  ⚠️  Prediction seems to match retrieved title: '{title}'")
                    break

            # Check if retrieved docs are relevant
            question_lower = question.lower()
            relevant_count = 0
            for title in titles:
                # Simple relevance heuristic: check if title words appear in question
                title_words = set(title.lower().split())
                question_words = set(question_lower.split())
                overlap = len(title_words & question_words)
                if overlap > 0:
                    relevant_count += 1

            print(f"\n  Relevance (rough estimate):")
            print(f"    {relevant_count}/{len(titles)} titles have word overlap with question")
            if relevant_count == 0 and len(titles) > 0:
                print(f"    ⚠️  Warning: No retrieved titles seem relevant to the question!")

        print(f"\n{'='*80}")

    def run(self, datasets=None, num_samples=20):
        """Run analysis on all datasets."""
        if datasets is None:
            datasets = self.config['datasets']

        print(f"\n{'='*80}")
        print(f"RETRIEVAL QUALITY ANALYSIS")
        print(f"{'='*80}")

        for dataset_name in datasets:
            try:
                self.analyze_dataset(dataset_name, num_samples)
            except Exception as e:
                print(f"\n❌ Error analyzing {dataset_name}: {e}")
                import traceback
                traceback.print_exc()

        print(f"\n{'='*80}")
        print(f"ANALYSIS COMPLETE")
        print(f"{'='*80}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Analyze retrieval quality")
    parser.add_argument('--config', default='evaluate/config.yaml',
                       help='Path to config file')
    parser.add_argument('--datasets', nargs='+',
                       help='Datasets to analyze (default: all)')
    parser.add_argument('--samples', type=int, default=20,
                       help='Number of samples to show per dataset')

    args = parser.parse_args()

    config = ConfigLoader.load_config(args.config)
    analyzer = RetrievalAnalyzer(config)
    analyzer.run(datasets=args.datasets, num_samples=args.samples)


if __name__ == '__main__':
    main()
