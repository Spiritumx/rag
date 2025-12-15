"""
Stage 2: Generation
Route questions to appropriate RAG pipelines based on classification and generate answers.
"""

import os
import sys
import json
import subprocess
import tempfile
from collections import defaultdict
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluate.utils.config_loader import ConfigLoader
from evaluate.utils.data_loader import DataLoader
from evaluate.utils.result_manager import ResultManager
from evaluate.configs.action_to_config_mapping import ActionConfigMapper


class Stage2Generator:
    """Stage 2: Generate answers using appropriate RAG pipeline per question."""

    def __init__(self, config):
        self.config = config
        self.data_loader = DataLoader(config)
        self.result_manager = ResultManager(config)
        self.config_mapper = ActionConfigMapper(config)

    def run(self, datasets=None):
        """
        Run generation for all datasets.

        Args:
            datasets: List of dataset names to process (None = all)
        """
        if datasets is None:
            datasets = self.config['datasets']

        print("\n" + "="*60)
        print("STAGE 2: GENERATION")
        print("="*60)

        # Check that all required configs exist
        if not self.config_mapper.check_configs_exist():
            raise RuntimeError(
                "Missing required config files. "
                "Please ensure all Llama configs are in evaluate/configs/llama_configs/"
            )

        # Process each dataset
        for dataset_name in datasets:
            print(f"\n{'='*60}")
            print(f"Generating for dataset: {dataset_name}")
            print(f"{'='*60}")

            self.generate_for_dataset(dataset_name)

        print("\n" + "="*60)
        print("✓ STAGE 2 COMPLETE")
        print("="*60)

    def generate_for_dataset(self, dataset_name: str):
        """
        Generate answers for all questions in a dataset.
        Routes each question to appropriate config based on classification.

        Args:
            dataset_name: Name of the dataset to process
        """
        # Load classifications
        classifications = self.result_manager.load_stage1_results(dataset_name)

        if not classifications:
            print(f"Warning: No classifications found for {dataset_name}")
            print(f"  Please run Stage 1 first for this dataset")
            return

        # Load test data
        test_data = self.data_loader.load_test_data(dataset_name)
        test_data_map = {item['question_id']: item for item in test_data}

        # Check existing predictions (resume capability)
        output_path = self.result_manager.get_stage2_output_path(dataset_name)
        existing_preds = self.result_manager.load_existing_results(output_path)

        # Group questions by action
        action_groups = defaultdict(list)
        for qid, classification in classifications.items():
            action = classification['predicted_action']
            if action == "Unknown":
                print(f"  Skipping {qid} with Unknown action")
                continue
            action_groups[action].append(qid)

        print(f"Question distribution by action:")
        for action in ['Z', 'S-Sparse', 'S-Dense', 'S-Hybrid', 'M']:
            if action in action_groups:
                count = len(action_groups[action])
                print(f"  {action:12s}: {count:4d} questions")

        # Process each action group
        all_predictions = existing_preds.copy()

        for action, qids in action_groups.items():
            print(f"\nProcessing action: {action}")

            # Filter to unprocessed questions
            unprocessed_qids = [qid for qid in qids if qid not in existing_preds]

            if not unprocessed_qids:
                print(f"  All {len(qids)} questions already processed, skipping")
                continue

            print(f"  Processing {len(unprocessed_qids)}/{len(qids)} questions")

            try:
                # Get config for this action
                config_path = self.config_mapper.get_config_path(action, dataset_name)

                # Create temporary input file for this action group
                temp_input_file = self.create_temp_input_file(
                    dataset_name, action, unprocessed_qids, test_data_map
                )

                # Run generation using configurable_inference
                predictions = self.run_inference(
                    config_path=config_path,
                    input_file=temp_input_file,
                    dataset_name=dataset_name,
                    action=action
                )

                # Merge predictions
                all_predictions.update(predictions)

                # Save checkpoint
                self.result_manager.save_stage2_results(dataset_name, all_predictions)

                # Cleanup temp file
                if os.path.exists(temp_input_file):
                    os.remove(temp_input_file)

                print(f"  ✓ Completed {len(predictions)} predictions for action {action}")

            except Exception as e:
                print(f"  Error processing action {action}: {e}")
                import traceback
                traceback.print_exc()

        print(f"\n✓ Generation complete for {dataset_name}")
        print(f"  Total predictions: {len(all_predictions)}/{len(test_data)}")
        print(f"  Output saved to: {output_path}")

    def create_temp_input_file(self, dataset_name: str, action: str, qids: list, test_data_map: dict) -> str:
        """
        Create temporary JSONL file with subset of questions for this action.

        Args:
            dataset_name: Name of dataset
            action: Action label
            qids: List of question IDs to include
            test_data_map: Map from question ID to test data

        Returns:
            Path to temporary file
        """
        temp_file = tempfile.NamedTemporaryFile(
            mode='w',
            suffix='.jsonl',
            delete=False,
            prefix=f'{dataset_name}_{action}_',
            encoding='utf-8'
        )

        for qid in qids:
            if qid in test_data_map:
                item = test_data_map[qid]
                # Convert to format expected by dataset_readers.py
                inference_item = {
                    'question_id': item['question_id'],
                    'question_text': item['question_text'],
                    'answers_objects': item.get('answers_objects', []),
                    'contexts': item.get('contexts', []),
                }

                temp_file.write(json.dumps(inference_item, ensure_ascii=False) + '\n')

        temp_file.close()
        return temp_file.name

    def run_inference(self, config_path: str, input_file: str, dataset_name: str, action: str) -> dict:
        """
        Run configurable_inference using subprocess.

        Args:
            config_path: Path to JSONNET config
            input_file: Path to input JSONL file
            dataset_name: Name of dataset
            action: Action label

        Returns:
            Dictionary of predictions {qid: answer}
        """
        # Create temp output file
        temp_output = tempfile.NamedTemporaryFile(
            mode='w',
            suffix='.json',
            delete=False,
            prefix=f'{dataset_name}_{action}_pred_',
            encoding='utf-8'
        )
        temp_output.close()

        # Set environment variables
        env = os.environ.copy()

        # Note: Removed forced offline mode to allow model loading from cache/hub
        # If you want offline mode, ensure all models are cached first

        # Set tokenizer model (use Llama tokenizer instead of flan-t5)
        tokenizer_path = self.config['llm'].get('model_path', '/root/autodl-tmp/model/Meta-Llama-3-8B-Instruct')
        env['TOKENIZER_MODEL_NAME'] = tokenizer_path

        # Service endpoints (add http:// prefix if not present)
        retriever_host = self.config['retriever']['host']
        if not retriever_host.startswith('http'):
            retriever_host = f'http://{retriever_host}'
        env['RETRIEVER_HOST'] = retriever_host
        env['RETRIEVER_PORT'] = str(self.config['retriever']['port'])

        llm_host = self.config['llm']['server_host']
        if not llm_host.startswith('http'):
            llm_host = f'http://{llm_host}'
        env['LLM_SERVER_HOST'] = llm_host
        env['LLM_SERVER_PORT'] = str(self.config['llm']['server_port'])

        # Build command
        cmd = [
            'python', '-m', 'commaqa.inference.configurable_inference',
            '--config', config_path,
            '--input', input_file,
            '--output', temp_output.name,
        ]

        # Add parallel processing if configured
        parallel_threads = self.config.get('execution', {}).get('parallel_threads', 1)
        if parallel_threads > 1:
            cmd.extend(['--threads', str(parallel_threads)])

        print(f"    Running: {' '.join(cmd[:4])} ... (threads: {parallel_threads})")

        try:
            # Run inference
            # Calculate dynamic timeout based on number of questions and parallelism
            # M action: ~5 min/question, others: ~10 sec/question
            questions_count = len(unprocessed_qids)
            if action == 'M':
                base_time_per_question = 300  # 5 min per question
            else:
                base_time_per_question = 20   # 20 sec per question

            # Adjust for parallelism (with overhead factor)
            parallel_efficiency = 0.7  # Assume 70% efficiency due to overhead
            estimated_time = (questions_count * base_time_per_question) / (parallel_threads * parallel_efficiency)
            timeout = max(int(estimated_time * 1.5), 600)  # 1.5x buffer, at least 10 minutes

            speedup = parallel_threads * parallel_efficiency
            print(f"    Estimated time: {estimated_time//60:.1f} min (timeout: {timeout//60:.1f} min, speedup: {speedup:.1f}x)")

            result = subprocess.run(
                cmd,
                env=env,
                capture_output=True,
                text=True,
                timeout=timeout
            )

            if result.returncode != 0:
                print(f"    Error: Inference failed with return code {result.returncode}")
                print(f"    STDERR (full output):")
                print(result.stderr)
                print(f"    STDOUT:")
                print(result.stdout)
                return {}

            # Load predictions
            if os.path.exists(temp_output.name):
                with open(temp_output.name, 'r', encoding='utf-8') as f:
                    predictions = json.load(f)
            else:
                print(f"    Warning: Output file not created")
                predictions = {}

            # Cleanup
            if os.path.exists(temp_output.name):
                os.remove(temp_output.name)

            return predictions

        except subprocess.TimeoutExpired:
            print(f"    Error: Inference timed out after 1 hour")
            return {}
        except Exception as e:
            print(f"    Error running inference: {e}")
            return {}


def main():
    """Main entry point for Stage 2."""
    import argparse

    parser = argparse.ArgumentParser(description="Stage 2: Generation")
    parser.add_argument('--config', default='evaluate/config.yaml',
                       help='Path to config file')
    parser.add_argument('--datasets', nargs='+',
                       help='Datasets to process (default: all in config)')

    args = parser.parse_args()

    # Load config
    config = ConfigLoader.load_config(args.config)

    # Run generation
    generator = Stage2Generator(config)
    generator.run(datasets=args.datasets)


if __name__ == '__main__':
    main()
