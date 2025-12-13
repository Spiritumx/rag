"""
Stage 1: Classification
Classify all test questions using the trained Qwen 2.5-3B LoRA classifier.
"""

import os
import sys
from tqdm import tqdm
from collections import Counter

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluate.utils.config_loader import ConfigLoader
from evaluate.utils.classifier_loader import ClassifierLoader
from evaluate.utils.data_loader import DataLoader
from evaluate.utils.result_manager import ResultManager


class Stage1Classifier:
    """Stage 1: Run classifier on all test questions and save results."""

    def __init__(self, config):
        self.config = config
        self.classifier_loader = ClassifierLoader(config)
        self.data_loader = DataLoader(config)
        self.result_manager = ResultManager(config)

    def run(self, datasets=None):
        """
        Run classification for all datasets.

        Args:
            datasets: List of dataset names to process (None = all)
        """
        if datasets is None:
            datasets = self.config['datasets']

        # Load classifier model once
        print("\n" + "="*60)
        print("STAGE 1: CLASSIFICATION")
        print("="*60)
        print("Loading classifier model...")
        self.classifier_loader.load_model()
        print("✓ Classifier loaded successfully!\n")

        # Process each dataset
        for dataset_name in datasets:
            print(f"\n{'='*60}")
            print(f"Processing dataset: {dataset_name}")
            print(f"{'='*60}")

            self.classify_dataset(dataset_name)

        print("\n" + "="*60)
        print("✓ STAGE 1 COMPLETE")
        print("="*60)

    def classify_dataset(self, dataset_name: str):
        """
        Classify all questions in a dataset.

        Args:
            dataset_name: Name of the dataset to process
        """
        # Load test data
        test_data = self.data_loader.load_test_data(dataset_name)

        # Check for existing results (resume capability)
        output_path = self.result_manager.get_stage1_output_path(dataset_name)
        existing_results = self.result_manager.load_existing_results(output_path)
        processed_qids = set(existing_results.keys())

        print(f"Total questions: {len(test_data)}")
        print(f"Already processed: {len(processed_qids)}")
        print(f"Remaining: {len(test_data) - len(processed_qids)}")

        # Process each question
        results = existing_results.copy()

        for item in tqdm(test_data, desc=f"Classifying {dataset_name}"):
            qid = item['question_id']

            # Skip if already processed
            if qid in processed_qids:
                continue

            question_text = item['question_text']

            # Run classifier
            try:
                classification = self.classifier_loader.classify_question(question_text)

                # Save result
                results[qid] = {
                    'question_id': qid,
                    'question_text': question_text,
                    'predicted_action': classification['action'],
                    'full_response': classification['full_response'],
                    'dataset': dataset_name
                }

                # Save checkpoint every 10 questions
                if len(results) % 10 == 0:
                    self.result_manager.save_stage1_results(dataset_name, results)

            except Exception as e:
                print(f"\nError classifying {qid}: {e}")
                # Save Unknown classification for failed questions
                results[qid] = {
                    'question_id': qid,
                    'question_text': question_text,
                    'predicted_action': 'Unknown',
                    'full_response': f"Error: {str(e)}",
                    'dataset': dataset_name
                }

        # Final save
        self.result_manager.save_stage1_results(dataset_name, results)
        print(f"\n✓ Classification complete for {dataset_name}")
        print(f"  Output saved to: {output_path}")

        # Print action distribution
        self.print_action_distribution(results)

    def print_action_distribution(self, results: dict):
        """Print distribution of predicted actions."""
        actions = [r['predicted_action'] for r in results.values()]
        dist = Counter(actions)

        print("\nAction Distribution:")
        total = len(actions)
        for action in ['Z', 'S-Sparse', 'S-Dense', 'S-Hybrid', 'M', 'Unknown']:
            if action in dist:
                count = dist[action]
                percentage = (count / total) * 100
                print(f"  {action:12s}: {count:4d} ({percentage:5.1f}%)")


def main():
    """Main entry point for Stage 1."""
    import argparse

    parser = argparse.ArgumentParser(description="Stage 1: Classification")
    parser.add_argument('--config', default='evaluate/config.yaml',
                       help='Path to config file')
    parser.add_argument('--datasets', nargs='+',
                       help='Datasets to process (default: all in config)')

    args = parser.parse_args()

    # Load config
    config = ConfigLoader.load_config(args.config)

    # Run classification
    classifier = Stage1Classifier(config)
    classifier.run(datasets=args.datasets)


if __name__ == '__main__':
    main()
