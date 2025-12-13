"""
Stage 1: Classification
Classify all test questions using the trained Qwen 2.5-3B LoRA classifier.
"""

import os
import sys
import time
import threading
import queue
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
        Classify all questions in a dataset using batch inference.

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

        # Filter unprocessed data
        unprocessed_data = [
            item for item in test_data
            if item['question_id'] not in processed_qids
        ]

        if not unprocessed_data:
            print("✓ All questions already processed!")
            self.print_action_distribution(existing_results)
            return

        results = existing_results.copy()
        batch_size = self.config['execution'].get('batch_size', 16)
        checkpoint_frequency = self.config['execution'].get('checkpoint_frequency', 5)

        # Create async saver thread
        save_queue = queue.Queue()

        def async_saver():
            """Background saver thread"""
            while True:
                item = save_queue.get()
                if item is None:  # Termination signal
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

                # Extract question texts and IDs
                batch_questions = [item['question_text'] for item in batch_data]
                batch_qids = [item['question_id'] for item in batch_data]

                try:
                    # Batch inference (critical optimization)
                    batch_results = self.classifier_loader.classify_batch(batch_questions)

                    # Save results
                    for item, classification in zip(batch_data, batch_results):
                        results[item['question_id']] = {
                            'question_id': item['question_id'],
                            'question_text': item['question_text'],
                            'predicted_action': classification['action'],
                            'full_response': classification['full_response'],
                            'dataset': dataset_name
                        }

                    processed_count += len(batch_data)

                    # Async save checkpoint (every N batches)
                    if (batch_idx // batch_size) % checkpoint_frequency == 0:
                        save_queue.put((dataset_name, results.copy()))

                    # Update progress bar
                    pbar.update(len(batch_data))

                    # Calculate and display speed
                    elapsed = time.time() - start_time
                    speed = processed_count / elapsed if elapsed > 0 else 0
                    remaining_time = (len(unprocessed_data) - processed_count) / speed if speed > 0 else 0

                    pbar.set_postfix({
                        'speed': f'{speed:.2f} q/s',
                        'eta': f'{remaining_time:.0f}s',
                        'batch': f'{batch_idx//batch_size + 1}/{total_batches}'
                    })

                except Exception as e:
                    print(f"\n❌ Error processing batch {batch_idx//batch_size}: {e}")
                    # Fallback: process failed batch one by one
                    print(f"   Falling back to single-question processing for this batch...")
                    for item in batch_data:
                        try:
                            classification = self.classifier_loader.classify_question(
                                item['question_text']
                            )
                            results[item['question_id']] = {
                                'question_id': item['question_id'],
                                'question_text': item['question_text'],
                                'predicted_action': classification['action'],
                                'full_response': classification['full_response'],
                                'dataset': dataset_name
                            }
                        except Exception as e2:
                            print(f"   Error on {item['question_id']}: {e2}")
                            results[item['question_id']] = {
                                'question_id': item['question_id'],
                                'question_text': item['question_text'],
                                'predicted_action': 'Unknown',
                                'full_response': f"Error: {str(e2)}",
                                'dataset': dataset_name
                            }

                        processed_count += 1
                        pbar.update(1)

        # Final save
        save_queue.put((dataset_name, results.copy()))
        save_queue.put(None)  # Termination signal
        if save_thread:
            save_thread.join()  # Wait for save to complete

        total_time = time.time() - start_time
        print(f"\n✓ Classification complete for {dataset_name}")
        print(f"  Total time: {total_time:.2f}s")
        print(f"  Average speed: {processed_count / total_time:.2f} q/s")
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
