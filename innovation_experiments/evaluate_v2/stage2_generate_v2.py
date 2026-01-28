"""
Stage 2: Generation with Innovations (V2)

Integrates three innovations:
1. Adaptive Retrieval (Port 8002 - already in retriever_server_v2)
2. Cascading Dynamic Routing (posterior verification + fallback)
3. MI-RA-ToT (beam search multi-hop reasoning)
"""

import os
import sys
import json
import subprocess
import tempfile
from collections import defaultdict
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add paths for imports
base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, base_dir)  # For 'from evaluate.utils...'
sys.path.insert(0, os.path.join(base_dir, 'innovation_experiments'))  # For 'from evaluate_v2...'

# Import baseline utilities (shared via symlinks)
from evaluate.utils.config_loader import ConfigLoader
from evaluate.utils.data_loader import DataLoader
from evaluate.configs.action_to_config_mapping import ActionConfigMapper

# Import V2-specific utilities
from evaluate_v2.utils.result_manager_v2 import ResultManagerV2
from evaluate_v2.utils.confidence_verifier import ConfidenceVerifier
from evaluate_v2.utils.routing_logger import RoutingLogger

# Import Innovation 3: MI-RA-ToT
from evaluate_v2.M_core_tot import execute_tot_multihop

# Import baseline M_core for ablation (Model B: w/o ToT)
from evaluate.M_core import execute_real_multihop as execute_linear_multihop


class Stage2GeneratorV2:
    """
    Stage 2 Generator with Innovations.

    Differences from baseline:
    1. Uses port 8002 retriever (adaptive weights)
    2. Posterior verification for non-M strategies
    3. Cascading to MI-RA-ToT on low confidence
    4. Routing decision logging
    """

    def __init__(self, config):
        self.config = config
        self.data_loader = DataLoader(config)
        self.result_manager = ResultManagerV2(config)
        self.config_mapper = ActionConfigMapper(config)

        # INNOVATION 2: Initialize confidence verifier
        innovations_config = config.get('innovations', {})
        cascade_config = innovations_config.get('cascading_routing', {})

        if cascade_config.get('enabled', True):
            self.verifier = ConfidenceVerifier(
                model_name=cascade_config.get('verifier_model', 'cross-encoder/ms-marco-MiniLM-L-6-v2'),
                device=cascade_config.get('verifier_device', 'cuda'),
                threshold=cascade_config.get('confidence_threshold', 0.6),
                max_contexts=cascade_config.get('max_contexts_for_verification', 3),
            )
            self.cascade_enabled = True
            print(f"[V2] Cascading enabled with threshold {self.verifier.threshold}")
        else:
            self.verifier = None
            self.cascade_enabled = False
            print("[V2] Cascading disabled")

        # Initialize routing logger
        self.routing_logger = RoutingLogger()

        # Get ToT hyperparameters from config
        tot_config = innovations_config.get('tot_reasoning', {})
        self.tot_enabled = tot_config.get('enabled', True)
        self.tot_beam_width = tot_config.get('beam_width', 3)
        self.tot_max_depth = tot_config.get('max_depth', 4)
        self.tot_mi_alpha = tot_config.get('mi_alpha', 0.7)
        self.tot_mi_beta = tot_config.get('mi_beta', 0.3)

        print(f"[V2] ToT Reasoning: {'Enabled (beam search)' if self.tot_enabled else 'Disabled (linear M_core)'}")

        # INNOVATION 3: Initialize ToT reranker for MI scoring
        tot_reranker_model = tot_config.get('reranker_model')
        tot_reranker_device = tot_config.get('reranker_device', 'cuda')

        if tot_reranker_model:
            try:
                from sentence_transformers import CrossEncoder
                self.tot_reranker = CrossEncoder(tot_reranker_model, device=tot_reranker_device)
                print(f"[V2] ToT reranker loaded: {tot_reranker_model}")
            except Exception as e:
                print(f"[V2] Failed to load ToT reranker: {e}")
                self.tot_reranker = None
        else:
            self.tot_reranker = None
            print("[V2] ToT reranker not configured, using simple scoring")

        print(f"[V2] ToT parameters: beam_width={self.tot_beam_width}, max_depth={self.tot_max_depth}")

    def run(self, datasets=None):
        """
        Run generation for all datasets.

        Args:
            datasets: List of dataset names to process (None = all)
        """
        if datasets is None:
            datasets = self.config['datasets']

        print("\n" + "="*60)
        print("STAGE 2: GENERATION (V2 with Innovations)")
        print("="*60)
        print("[V2] Adaptive Retrieval: Port 8002")
        print("[V2] Cascading Routing: Enabled" if self.cascade_enabled else "[V2] Cascading Routing: Disabled")
        print("[V2] MI-RA-ToT: Beam Search")
        print("="*60)

        # Check that all required configs exist
        if not self.config_mapper.check_configs_exist():
            raise RuntimeError(
                "Missing required config files. "
                "Please ensure all configs are in evaluate_v2/configs/llama_configs_v2/"
            )

        # Process each dataset
        for dataset_name in datasets:
            print(f"\n{'='*60}")
            print(f"Generating for dataset: {dataset_name}")
            print(f"{'='*60}")

            self.generate_for_dataset(dataset_name)

        print("\n" + "="*60)
        print("✓ STAGE 2 (V2) COMPLETE")
        print("="*60)

    def generate_for_dataset(self, dataset_name: str):
        """
        Generate answers for all questions in a dataset with cascading.

        Args:
            dataset_name: Name of the dataset to process
        """
        # Load classifications (SHARED with baseline)
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
                # INNOVATION 3: M strategy → MI-RA-ToT
                if action == 'M':
                    result = self.process_m_strategy_tot(
                        unprocessed_qids, test_data_map, dataset_name
                    )

                # INNOVATION 2: Non-M strategies → With cascading
                else:
                    result = self.process_non_m_strategy_with_cascade(
                        action, unprocessed_qids, test_data_map, dataset_name
                    )

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
                    f.write(f"QID: {qid}\n")
                    f.write(all_chains[qid] + '\n')
                    f.write('\n')
            print(f"  Chains saved to: {chains_path}")

        if all_contexts:
            contexts_path = output_path.replace('.json', '_contexts.json')
            with open(contexts_path, 'w', encoding='utf-8') as f:
                json.dump(all_contexts, f, indent=2, ensure_ascii=False)
            print(f"  Contexts saved to: {contexts_path}")

        # INNOVATION 2: Save cascade routing log
        cascade_log_path = self.result_manager.get_cascade_log_path(dataset_name)
        self.routing_logger.save(cascade_log_path)
        print(f"  Cascade log saved to: {cascade_log_path}")

        # Print cascade statistics
        self.routing_logger.print_summary()

        print(f"\n✓ Generation complete for {dataset_name}")
        print(f"  Total predictions: {len(all_predictions)}/{len(test_data)}")
        print(f"  Output saved to: {output_path}")

    def process_m_strategy_tot(
        self, qids: list, test_data_map: dict, dataset_name: str
    ) -> dict:
        """
        Process M strategy questions using MI-RA-ToT or linear M_core.

        If tot_enabled=True: Use MI-RA-ToT (beam search)
        If tot_enabled=False: Use baseline linear M_core (for ablation Model B)

        Args:
            qids: List of question IDs
            test_data_map: Map from question ID to test data
            dataset_name: Dataset name

        Returns:
            Dict with 'predictions', 'chains', 'contexts'
        """
        # Prepare config
        retriever_config = {
            'host': self.config['retriever']['host'],
            'port': self.config['retriever']['port']  # Port 8002 for adaptive retrieval
        }
        llm_config = {
            'host': self.config['llm']['server_host'],
            'port': self.config['llm']['server_port']
        }

        # Get parallel threads
        parallel_threads = self.config.get('execution', {}).get('parallel_threads', 1)

        if self.tot_enabled:
            print(f"  [M-ToT] Using MI-RA-ToT beam search with {parallel_threads} threads")
            strategy_label = 'M-ToT'
        else:
            print(f"  [M-Linear] Using baseline linear M_core with {parallel_threads} threads")
            strategy_label = 'M-Linear'

        # Define single question processing function
        def process_single_question(qid):
            question_text = test_data_map[qid]['question_text']
            try:
                if self.tot_enabled:
                    # INNOVATION 3: Use MI-RA-ToT (beam search)
                    result = execute_tot_multihop(
                        query=question_text,
                        retriever_config=retriever_config,
                        llm_config=llm_config,
                        dataset_name=dataset_name,
                        beam_width=self.tot_beam_width,
                        max_depth=self.tot_max_depth,
                        mi_alpha=self.tot_mi_alpha,
                        mi_beta=self.tot_mi_beta,
                        reranker_model=self.tot_reranker,
                    )
                else:
                    # ABLATION: Use baseline linear M_core
                    result = execute_linear_multihop(
                        query=question_text,
                        retriever_config=retriever_config,
                        llm_config=llm_config,
                        dataset_name=dataset_name,
                    )

                # Log routing decision (M strategy never cascades)
                self.routing_logger.log_decision(
                    question_id=qid,
                    initial_action='M',
                    confidence=1.0,  # M strategies are not verified
                    final_action=strategy_label,
                    cascaded=False,
                    question_text=question_text,
                    dataset=dataset_name,
                )

                return qid, result, None
            except Exception as e:
                import traceback
                traceback.print_exc()
                return qid, None, str(e)

        # Parallel processing
        predictions = {}
        chains = {}
        contexts = {}

        with ThreadPoolExecutor(max_workers=parallel_threads) as executor:
            future_to_qid = {
                executor.submit(process_single_question, qid): qid
                for qid in qids
            }

            progress_desc = "  MI-RA-ToT reasoning" if self.tot_enabled else "  Linear M reasoning"
            with tqdm(total=len(qids), desc=progress_desc) as pbar:
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

        return {
            'predictions': predictions,
            'chains': chains,
            'contexts': contexts
        }

    def process_non_m_strategy_with_cascade(
        self, action: str, qids: list, test_data_map: dict, dataset_name: str
    ) -> dict:
        """
        Process non-M strategy with cascading (Innovation 2).

        Steps:
        1. Execute initial strategy (Z, S-Sparse, S-Dense, S-Hybrid)
        2. Verify answer confidence
        3. If confidence < threshold → Cascade to MI-RA-ToT
        4. Log routing decision

        Args:
            action: Action label (Z, S-Sparse, S-Dense, S-Hybrid)
            qids: List of question IDs
            test_data_map: Map from question ID to test data
            dataset_name: Dataset name

        Returns:
            Dict with 'predictions', 'chains', 'contexts'
        """
        # Step 1: Execute initial strategy using baseline method
        print(f"  [Cascade] Step 1: Execute initial strategy ({action})")

        # Get config for this action
        config_path = self.config_mapper.get_config_path(action, dataset_name)

        # Create temporary input file
        temp_input_file = self.create_temp_input_file(
            dataset_name, action, qids, test_data_map
        )

        # Run generation using configurable_inference
        initial_result = self.run_inference(
            config_path=config_path,
            input_file=temp_input_file,
            dataset_name=dataset_name,
            action=action,
            num_questions=len(qids)
        )

        # Cleanup temp file
        if os.path.exists(temp_input_file):
            os.remove(temp_input_file)

        # Step 2: Posterior verification (if enabled)
        if not self.cascade_enabled:
            # No cascading, just log direct routing
            for qid in qids:
                question_text = test_data_map[qid]['question_text']
                self.routing_logger.log_decision(
                    question_id=qid,
                    initial_action=action,
                    confidence=1.0,  # No verification
                    final_action=action,
                    cascaded=False,
                    question_text=question_text,
                    dataset=dataset_name,
                )
            return initial_result

        print(f"  [Cascade] Step 2: Posterior verification")

        # Prepare final results
        final_predictions = {}
        final_chains = {}
        final_contexts = {}

        cascade_count = 0
        retriever_config = {
            'host': self.config['retriever']['host'],
            'port': self.config['retriever']['port']
        }
        llm_config = {
            'host': self.config['llm']['server_host'],
            'port': self.config['llm']['server_port']
        }

        # Verify each prediction
        for qid in qids:
            question_text = test_data_map[qid]['question_text']
            initial_answer = initial_result['predictions'].get(qid, "I don't know")
            initial_contexts = initial_result['contexts'].get(qid, [])

            # Verify confidence
            confidence = self.verifier.verify(
                question=question_text,
                answer=initial_answer,
                contexts=initial_contexts
            )

            # Check if should cascade
            should_cascade = self.verifier.should_cascade(confidence, strategy=action)

            if should_cascade:
                cascade_target = 'M-ToT' if self.tot_enabled else 'M-Linear'
                print(f"    [Cascade] {qid}: confidence={confidence:.3f} → Cascading to {cascade_target}")
                cascade_count += 1

                try:
                    # Cascade to M strategy (ToT or Linear based on config)
                    if self.tot_enabled:
                        # Use MI-RA-ToT (beam search)
                        cascade_result = execute_tot_multihop(
                            query=question_text,
                            retriever_config=retriever_config,
                            llm_config=llm_config,
                            dataset_name=dataset_name,
                            beam_width=self.tot_beam_width,
                            max_depth=self.tot_max_depth,
                            mi_alpha=self.tot_mi_alpha,
                            mi_beta=self.tot_mi_beta,
                            reranker_model=self.tot_reranker,
                        )
                    else:
                        # Use baseline linear M_core (for ablation)
                        cascade_result = execute_linear_multihop(
                            query=question_text,
                            retriever_config=retriever_config,
                            llm_config=llm_config,
                            dataset_name=dataset_name,
                        )

                    final_predictions[qid] = cascade_result['answer']
                    final_chains[qid] = f"[CASCADED from {action}]\n" + cascade_result['chain']
                    final_contexts[qid] = cascade_result['contexts']

                    # Log cascade decision
                    self.routing_logger.log_decision(
                        question_id=qid,
                        initial_action=action,
                        confidence=confidence,
                        final_action=cascade_target,
                        cascaded=True,
                        question_text=question_text,
                        dataset=dataset_name,
                    )

                except Exception as e:
                    print(f"    [Cascade] Error in ToT for {qid}: {e}")
                    # Fallback to initial answer
                    final_predictions[qid] = initial_answer
                    final_chains[qid] = initial_result['chains'].get(qid, "")
                    final_contexts[qid] = initial_contexts

                    self.routing_logger.log_decision(
                        question_id=qid,
                        initial_action=action,
                        confidence=confidence,
                        final_action=action,  # Failed cascade, kept original
                        cascaded=False,
                        question_text=question_text,
                        dataset=dataset_name,
                    )

            else:
                # High confidence, keep initial answer
                final_predictions[qid] = initial_answer
                final_chains[qid] = initial_result['chains'].get(qid, "")
                final_contexts[qid] = initial_contexts

                # Log direct routing
                self.routing_logger.log_decision(
                    question_id=qid,
                    initial_action=action,
                    confidence=confidence,
                    final_action=action,
                    cascaded=False,
                    question_text=question_text,
                    dataset=dataset_name,
                )

        print(f"  [Cascade] Step 3: Results - Cascaded {cascade_count}/{len(qids)} questions ({cascade_count/len(qids)*100:.1f}%)")

        return {
            'predictions': final_predictions,
            'chains': final_chains,
            'contexts': final_contexts
        }

    def create_temp_input_file(self, dataset_name: str, action: str, qids: list, test_data_map: dict) -> str:
        """Create temporary JSONL file for subprocess inference."""
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

        Returns:
            Dict with 'predictions', 'chains', 'contexts'
        """
        # Create output directory
        output_dir = tempfile.mkdtemp(prefix=f'{dataset_name}_{action}_output_')
        output_file = os.path.join(output_dir, 'predictions.json')

        try:
            # Build command
            cmd = [
                'python', '-m', 'commaqa.inference.prompt_based_inference.configurable_inference',
                '--config', config_path,
                '--input_file', input_file,
                '--output_file', output_file,
            ]

            # Prepare environment with retriever config (CRITICAL for ablation experiments)
            env = os.environ.copy()
            env['RETRIEVER_HOST'] = str(self.config['retriever']['host'])
            env['RETRIEVER_PORT'] = str(self.config['retriever']['port'])
            env['DATASET_NAME'] = dataset_name

            # Map dataset to corpus name
            dataset_to_corpus = {
                'hotpotqa': 'hotpotqa', 'musique': 'musique',
                '2wikimultihopqa': '2wikimultihopqa', 'iirc': 'iirc',
                'squad': 'wiki', 'trivia': 'wiki', 'nq': 'wiki'
            }
            env['CORPUS_NAME'] = dataset_to_corpus.get(dataset_name.lower(), 'wiki')

            # Run inference
            print(f"    Running inference with config: {config_path}")
            print(f"    Retriever: {env['RETRIEVER_HOST']}:{env['RETRIEVER_PORT']}, Corpus: {env['CORPUS_NAME']}")
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=3600,  # 1 hour timeout
                env=env  # Pass environment variables to subprocess
            )

            if result.returncode != 0:
                print(f"    Warning: Inference returned code {result.returncode}")
                print(f"    stderr: {result.stderr[:500]}")

            # Load predictions
            if os.path.exists(output_file):
                with open(output_file, 'r', encoding='utf-8') as f:
                    predictions_data = json.load(f)

                # Convert to expected format
                predictions = {}
                chains = {}
                contexts = {}

                for item in predictions_data:
                    qid = item['question_id']
                    predictions[qid] = item.get('predicted_answer', "I don't know")
                    chains[qid] = item.get('reasoning_chain', "")
                    contexts[qid] = item.get('contexts', [])

                return {
                    'predictions': predictions,
                    'chains': chains,
                    'contexts': contexts
                }
            else:
                print(f"    Error: Output file not created: {output_file}")
                return {'predictions': {}, 'chains': {}, 'contexts': {}}

        except subprocess.TimeoutExpired:
            print(f"    Error: Inference timed out after 1 hour")
            return {'predictions': {}, 'chains': {}, 'contexts': {}}
        except Exception as e:
            print(f"    Error running inference: {e}")
            import traceback
            traceback.print_exc()
            return {'predictions': {}, 'chains': {}, 'contexts': {}}
        finally:
            # Cleanup
            import shutil
            if os.path.exists(output_dir):
                shutil.rmtree(output_dir, ignore_errors=True)
