"""
Adaptive-RAG Stage 1: Classification
Classify all test questions using the Adaptive-RAG trained classifier.

Usage:
    python -m adaptive_rag.evaluate.stage1_classify
    python -m adaptive_rag.evaluate.stage1_classify --datasets squad hotpotqa
"""

import os
import sys
import time
import threading
import queue
import argparse
from pathlib import Path
from tqdm import tqdm
from collections import Counter

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Reuse existing utils
from evaluate.utils.config_loader import ConfigLoader
from evaluate.utils.data_loader import DataLoader
from evaluate.utils.result_manager import ResultManager

# Use Adaptive-RAG specific classifier loader (Z/S/M only)
from adaptive_rag.evaluate.adaptive_classifier_loader import AdaptiveClassifierLoader


class AdaptiveStage1Classifier:
    """Stage 1: Run Adaptive-RAG classifier on all test questions."""

    def __init__(self, config):
        self.config = config
        self.classifier_loader = AdaptiveClassifierLoader(config)
        self.data_loader = DataLoader(config)
        self.result_manager = ResultManager(config)

    def run(self, datasets=None):
        """Run classification for all datasets."""
        if datasets is None:
            datasets = self.config['datasets']

        print("\n" + "="*60)
        print("ADAPTIVE-RAG STAGE 1: CLASSIFICATION")
        print("="*60)
        print("Loading classifier model...")
        self.classifier_loader.load_model()
        print("Classifier loaded successfully!\n")

        for dataset_name in datasets:
            print(f"\n{'='*60}")
            print(f"Processing dataset: {dataset_name}")
            print(f"{'='*60}")

            self.classify_dataset(dataset_name)

        print("\n" + "="*60)
        print("STAGE 1 COMPLETE")
        print("="*60)

    def classify_dataset(self, dataset_name: str):
        """Classify all questions in a dataset using batch inference."""
        # Load test data
        test_data = self.data_loader.load_test_data(dataset_name)

        # Check for existing results
        output_path = self.result_manager.get_stage1_output_path(dataset_name)
        existing_results = self.result_manager.load_existing_results(output_path)
        processed_qids = set(existing_results.keys())

        print(f"Total questions: {len(test_data)}")
        print(f"Already processed: {len(processed_qids)}")
        print(f"Remaining: {len(test_data) - len(processed_qids)}")

        # Filter unprocessed data
        unprocessed_data = [
            item for item in test_data
            if item['question_id'] not in processed_qids
        ]

        if not unprocessed_data:
            print("All questions already processed!")
            self.print_action_distribution(existing_results)
            return

        results = existing_results.copy()
        batch_size = self.config['execution'].get('batch_size', 16)
        checkpoint_frequency = self.config['execution'].get('checkpoint_frequency', 5)

        # Async saver
        save_queue = queue.Queue()

        def async_saver():
            while True:
                item = save_queue.get()
                if item is None:
                    break
                dataset, results_snapshot = item
                self.result_manager.save_stage1_results(dataset, results_snapshot)
                save_queue.task_done()

        save_thread = threading.Thread(target=async_saver, daemon=True)
        save_thread.start()

        # Batch processing
        total_batches = (len(unprocessed_data) + batch_size - 1) // batch_size
        start_time = time.time()
        processed_count = 0

        with tqdm(total=len(unprocessed_data),
                  desc=f"Classifying {dataset_name}",
                  unit="q") as pbar:

            for batch_idx in range(0, len(unprocessed_data), batch_size):
                batch_data = unprocessed_data[batch_idx:batch_idx + batch_size]
                batch_questions = [item['question_text'] for item in batch_data]

                try:
                    batch_results = self.classifier_loader.classify_batch(batch_questions)

                    for item, classification in zip(batch_data, batch_results):
                        # Action is already normalized to Z/S/M by AdaptiveClassifierLoader
                        action = classification['action']

                        results[item['question_id']] = {
                            'question_id': item['question_id'],
                            'question_text': item['question_text'],
                            'predicted_action': action,
                            'full_response': classification['full_response'],
                            'dataset': dataset_name
                        }

                    processed_count += len(batch_data)

                    if (batch_idx // batch_size) % checkpoint_frequency == 0:
                        save_queue.put((dataset_name, results.copy()))

                    pbar.update(len(batch_data))

                    elapsed = time.time() - start_time
                    speed = processed_count / elapsed if elapsed > 0 else 0
                    remaining_time = (len(unprocessed_data) - processed_count) / speed if speed > 0 else 0

                    pbar.set_postfix({
                        'speed': f'{speed:.2f} q/s',
                        'eta': f'{remaining_time:.0f}s',
                        'batch': f'{batch_idx//batch_size + 1}/{total_batches}'
                    })

                except Exception as e:
                    print(f"\nError processing batch {batch_idx//batch_size}: {e}")
                    for item in batch_data:
                        try:
                            classification = self.classifier_loader.classify_question(
                                item['question_text']
                            )
                            # Action is already normalized to Z/S/M
                            action = classification['action']

                            results[item['question_id']] = {
                                'question_id': item['question_id'],
                                'question_text': item['question_text'],
                                'predicted_action': action,
                                'full_response': classification['full_response'],
                                'dataset': dataset_name
                            }
                        except Exception as e2:
                            results[item['question_id']] = {
                                'question_id': item['question_id'],
                                'question_text': item['question_text'],
                                'predicted_action': 'M',  # Default to M on error
                                'full_response': f"Error: {str(e2)}",
                                'dataset': dataset_name
                            }

                        processed_count += 1
                        pbar.update(1)

        # Final save
        save_queue.put((dataset_name, results.copy()))
        save_queue.put(None)
        if save_thread:
            save_thread.join()

        total_time = time.time() - start_time
        print(f"\nClassification complete for {dataset_name}")
        print(f"  Total time: {total_time:.2f}s")
        print(f"  Average speed: {processed_count / total_time:.2f} q/s")
        print(f"  Output saved to: {output_path}")

        self.print_action_distribution(results)

    def print_action_distribution(self, results: dict):
        """Print distribution of predicted actions."""
        actions = [r['predicted_action'] for r in results.values()]
        dist = Counter(actions)

        print("\nAction Distribution:")
        total = len(actions)
        for action in ['Z', 'S', 'M', 'Unknown']:
            if action in dist:
                count = dist[action]
                percentage = (count / total) * 100
                print(f"  {action:12s}: {count:4d} ({percentage:5.1f}%)")


def main():
    parser = argparse.ArgumentParser(description="Adaptive-RAG Stage 1: Classification")
    parser.add_argument('--config', default='adaptive_rag/evaluate/config.yaml',
                       help='Path to config file')
    parser.add_argument('--datasets', nargs='+',
                       help='Datasets to process (default: all in config)')

    args = parser.parse_args()

    config = ConfigLoader.load_config(args.config)
    classifier = AdaptiveStage1Classifier(config)
    classifier.run(datasets=args.datasets)


if __name__ == '__main__':
    main()
