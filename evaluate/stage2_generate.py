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
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluate.utils.config_loader import ConfigLoader
from evaluate.utils.data_loader import DataLoader
from evaluate.utils.result_manager import ResultManager
from evaluate.configs.action_to_config_mapping import ActionConfigMapper
from evaluate.M_core import execute_real_multihop  # M策略专用多跳函数


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
        all_chains = {}
        all_contexts = {}

        for action, qids in action_groups.items():
            print(f"\n{'='*60}")
            print(f"Processing action: {action} ({len(qids)} questions)")
            print(f"{'='*60}")

            # Filter to unprocessed questions
            unprocessed_qids = [qid for qid in qids if qid not in existing_preds]

            if not unprocessed_qids:
                print(f"  All {len(qids)} questions already processed, skipping")
                continue

            print(f"  Processing {len(unprocessed_qids)}/{len(qids)} questions")

            try:
                # 🔥 M策略：使用新的简洁多跳实现
                if action == 'M':
                    # 准备配置
                    retriever_config = {
                        'host': self.config['retriever']['host'],
                        'port': self.config['retriever']['port']
                    }
                    llm_config = {
                        'host': self.config['llm']['server_host'],
                        'port': self.config['llm']['server_port']
                    }

                    # 获取并行线程数
                    parallel_threads = self.config.get('execution', {}).get('parallel_threads', 1)
                    print(f"  Using {parallel_threads} parallel threads for multi-hop reasoning")

                    # 定义单个问题的处理函数
                    def process_single_question(qid):
                        question_text = test_data_map[qid]['question_text']
                        try:
                            result = execute_real_multihop(
                                query=question_text,
                                retriever_config=retriever_config,
                                llm_config=llm_config,
                                dataset_name=dataset_name
                            )
                            return qid, result, None
                        except Exception as e:
                            import traceback
                            traceback.print_exc()
                            return qid, None, str(e)

                    # 并行处理问题
                    predictions = {}
                    chains = {}
                    contexts = {}

                    with ThreadPoolExecutor(max_workers=parallel_threads) as executor:
                        # 提交所有任务
                        future_to_qid = {
                            executor.submit(process_single_question, qid): qid
                            for qid in unprocessed_qids
                        }

                        # 使用tqdm显示进度
                        with tqdm(total=len(unprocessed_qids), desc="  Multi-hop reasoning") as pbar:
                            for future in as_completed(future_to_qid):
                                qid, result, error = future.result()

                                if error:
                                    print(f"\n    Error processing {qid}: {error}")
                                    predictions[qid] = "I don't know"
                                    chains[qid] = f"Error: {error}"
                                    contexts[qid] = []
                                else:
                                    predictions[qid] = result['answer']
                                    chains[qid] = result['chain']
                                    contexts[qid] = result['contexts']

                                pbar.update(1)

                    # 构造结果
                    result = {
                        'predictions': predictions,
                        'chains': chains,
                        'contexts': contexts
                    }

                else:
                    # 其他策略：使用原有的subprocess方式
                    # Get config for this action
                    config_path = self.config_mapper.get_config_path(action, dataset_name)

                    # Create temporary input file for this action group
                    temp_input_file = self.create_temp_input_file(
                        dataset_name, action, unprocessed_qids, test_data_map
                    )

                    # Run generation using configurable_inference
                    result = self.run_inference(
                        config_path=config_path,
                        input_file=temp_input_file,
                        dataset_name=dataset_name,
                        action=action,
                        num_questions=len(unprocessed_qids)
                    )

                    # Cleanup temp file
                    if os.path.exists(temp_input_file):
                        os.remove(temp_input_file)

                # Merge predictions, chains, and contexts
                all_predictions.update(result['predictions'])
                all_chains.update(result['chains'])
                all_contexts.update(result['contexts'])

                # Save checkpoint
                self.result_manager.save_stage2_results(dataset_name, all_predictions)

                print(f"  ✓ Completed {len(result['predictions'])} predictions for action {action}")

            except Exception as e:
                print(f"  Error processing action {action}: {e}")
                import traceback
                traceback.print_exc()

        # Save chains and contexts
        if all_chains:
            chains_path = output_path.replace('.json', '_chains.txt')
            with open(chains_path, 'w', encoding='utf-8') as f:
                for qid in sorted(all_chains.keys()):
                    f.write(all_chains[qid] + '\n')
            print(f"  Chains saved to: {chains_path}")

        if all_contexts:
            contexts_path = output_path.replace('.json', '_contexts.json')
            with open(contexts_path, 'w', encoding='utf-8') as f:
                json.dump(all_contexts, f, indent=2, ensure_ascii=False)
            print(f"  Contexts saved to: {contexts_path}")

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

    def run_inference(self, config_path: str, input_file: str, dataset_name: str, action: str, num_questions: int) -> dict:
        """
        Run configurable_inference using subprocess.

        Args:
            config_path: Path to JSONNET config
            input_file: Path to input JSONL file
            dataset_name: Name of dataset
            action: Action label
            num_questions: Number of questions to process

        Returns:
            Dictionary with:
                'predictions': {qid: answer}
                'chains': {qid: chain}
                'contexts': {qid: context}
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

        # Set corpus name based on dataset (for retrieval)
        dataset_to_corpus = {
            'hotpotqa': 'hotpotqa',
            'musique': 'musique',
            '2wikimultihopqa': '2wikimultihopqa',
            'iirc': 'iirc',
            'wiki': 'wiki'
        }
        corpus_name = dataset_to_corpus.get(dataset_name.lower(), 'wiki')
        env['CORPUS_NAME'] = corpus_name

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

        # Build command (use current Python interpreter to ensure pixi env)
        cmd = [
            sys.executable, '-m', 'commaqa.inference.configurable_inference',
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
            questions_count = num_questions
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
                return {'predictions': {}, 'chains': {}, 'contexts': {}}

            # Load predictions
            predictions = {}
            chains = {}
            contexts = {}

            if os.path.exists(temp_output.name):
                with open(temp_output.name, 'r', encoding='utf-8') as f:
                    predictions = json.load(f)
            else:
                print(f"    Warning: Output file not created")

            # Load chains
            chains_file = temp_output.name.replace('.json', '_chains.txt')
            if os.path.exists(chains_file):
                with open(chains_file, 'r', encoding='utf-8') as f:
                    # Parse chains file - each chain starts with qid
                    current_chain = []
                    current_qid = None
                    for line in f:
                        line = line.rstrip('\n')
                        # Each chain starts with a blank line followed by qid
                        if line == '' and current_chain:
                            if current_qid:
                                chains[current_qid] = '\n'.join(current_chain)
                            current_chain = []
                            current_qid = None
                        elif current_qid is None and line:
                            # First non-empty line is the qid
                            current_qid = line
                            current_chain = [line]
                        elif line:
                            current_chain.append(line)
                    # Don't forget the last chain
                    if current_qid and current_chain:
                        chains[current_qid] = '\n'.join(current_chain)

            # Load contexts
            contexts_file = temp_output.name.replace('.json', '_contexts.json')
            if os.path.exists(contexts_file):
                with open(contexts_file, 'r', encoding='utf-8') as f:
                    contexts = json.load(f)

            # Cleanup temp files
            for file_path in [temp_output.name, chains_file, contexts_file]:
                if os.path.exists(file_path):
                    os.remove(file_path)

            return {
                'predictions': predictions,
                'chains': chains,
                'contexts': contexts
            }

        except subprocess.TimeoutExpired:
            print(f"    Error: Inference timed out after 1 hour")
            return {'predictions': {}, 'chains': {}, 'contexts': {}}
        except Exception as e:
            print(f"    Error running inference: {e}")
            return {'predictions': {}, 'chains': {}, 'contexts': {}}


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
