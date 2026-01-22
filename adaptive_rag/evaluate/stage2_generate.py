"""
Adaptive-RAG Stage 2: Generation
Route questions to appropriate RAG pipelines based on classification and generate answers.

Strategies:
- Z: Zero retrieval (direct LLM)
- S: Single hybrid retrieval + LLM
- M: Multi-hop reasoning (IRCoT)

Usage:
    python -m adaptive_rag.evaluate.stage2_generate
    python -m adaptive_rag.evaluate.stage2_generate --datasets squad hotpotqa
"""

import os
import sys
import json
import requests
import argparse
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from evaluate.utils.config_loader import ConfigLoader
from evaluate.utils.data_loader import DataLoader
from evaluate.utils.result_manager import ResultManager
from evaluate.M_core import execute_real_multihop


def extract_llm_text(response_json: dict) -> str:
    """从 LLM 响应中提取文本"""
    text = ""
    if 'generated_texts' in response_json:
        texts = response_json['generated_texts']
        text = texts[0] if isinstance(texts, list) and len(texts) > 0 else ""
    elif 'text' in response_json:
        text = response_json['text']
    elif 'generated_text' in response_json:
        text = response_json['generated_text']
    elif 'choices' in response_json:
        choices = response_json['choices']
        if isinstance(choices, list) and len(choices) > 0:
            text = choices[0].get('text', '') or choices[0].get('message', {}).get('content', '')
    return text.strip()


class AdaptiveStage2Generator:
    """Stage 2: Generate answers using Adaptive-RAG routing."""

    CORPUS_MAPPING = {
        'hotpotqa': 'hotpotqa',
        'musique': 'musique',
        '2wikimultihopqa': '2wikimultihopqa',
        'squad': 'wiki',
        'trivia': 'wiki',
        'nq': 'wiki'
    }

    def __init__(self, config):
        self.config = config
        self.data_loader = DataLoader(config)
        self.result_manager = ResultManager(config)

        self.llm_url = f"http://{config['llm']['server_host']}:{config['llm']['server_port']}/generate"
        self.retriever_url = f"http://{config['retriever']['host']}:{config['retriever']['port']}/retrieve/"

    def get_corpus_name(self, dataset_name: str) -> str:
        """获取数据集对应的语料库名称"""
        return self.CORPUS_MAPPING.get(dataset_name.lower(), 'wiki')

    def run(self, datasets=None):
        """Run generation for all datasets."""
        if datasets is None:
            datasets = self.config['datasets']

        print("\n" + "="*60)
        print("ADAPTIVE-RAG STAGE 2: GENERATION")
        print("="*60)

        for dataset_name in datasets:
            print(f"\n{'='*60}")
            print(f"Generating for dataset: {dataset_name}")
            print(f"{'='*60}")

            self.generate_for_dataset(dataset_name)

        print("\n" + "="*60)
        print("STAGE 2 COMPLETE")
        print("="*60)

    def generate_for_dataset(self, dataset_name: str):
        """Generate answers for all questions in a dataset."""
        # Load classifications
        classifications = self.result_manager.load_stage1_results(dataset_name)

        if not classifications:
            print(f"Warning: No classifications found for {dataset_name}")
            print(f"  Please run Stage 1 first for this dataset")
            return

        # Load test data
        test_data = self.data_loader.load_test_data(dataset_name)
        test_data_map = {item['question_id']: item for item in test_data}

        # Check existing predictions
        output_path = self.result_manager.get_stage2_output_path(dataset_name)
        existing_preds = self.result_manager.load_existing_results(output_path)

        # Group questions by action
        action_groups = defaultdict(list)
        for qid, classification in classifications.items():
            action = classification['predicted_action']
            if action == "Unknown":
                action = "M"  # Default to M for unknown
            action_groups[action].append(qid)

        print(f"Question distribution by action:")
        for action in ['Z', 'S', 'M']:
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

            unprocessed_qids = [qid for qid in qids if qid not in existing_preds]

            if not unprocessed_qids:
                print(f"  All {len(qids)} questions already processed, skipping")
                continue

            print(f"  Processing {len(unprocessed_qids)}/{len(qids)} questions")

            try:
                if action == 'Z':
                    result = self.process_zero_retrieval(
                        unprocessed_qids, test_data_map
                    )
                elif action == 'S':
                    result = self.process_single_retrieval(
                        unprocessed_qids, test_data_map, dataset_name
                    )
                else:  # M
                    result = self.process_multi_hop(
                        unprocessed_qids, test_data_map, dataset_name
                    )

                all_predictions.update(result['predictions'])
                all_chains.update(result.get('chains', {}))
                all_contexts.update(result.get('contexts', {}))

                # Save checkpoint
                self.result_manager.save_stage2_results(dataset_name, all_predictions)

                print(f"  Completed {len(result['predictions'])} predictions for action {action}")

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
                    f.write(all_chains[qid] + '\n\n')
            print(f"  Chains saved to: {chains_path}")

        if all_contexts:
            contexts_path = output_path.replace('.json', '_contexts.json')
            with open(contexts_path, 'w', encoding='utf-8') as f:
                json.dump(all_contexts, f, indent=2, ensure_ascii=False)
            print(f"  Contexts saved to: {contexts_path}")

        print(f"\nGeneration complete for {dataset_name}")
        print(f"  Total predictions: {len(all_predictions)}/{len(test_data)}")
        print(f"  Output saved to: {output_path}")

    def process_zero_retrieval(self, qids: list, test_data_map: dict) -> dict:
        """Process questions with Zero Retrieval (direct LLM)."""
        predictions = {}
        chains = {}
        contexts = {}

        parallel_threads = self.config.get('execution', {}).get('parallel_threads', 8)

        def process_single(qid):
            question_text = test_data_map[qid]['question_text']

            prompt = f"""Answer the following question directly and concisely. Give only the answer, no explanation.

Question: {question_text}

Answer:"""

            try:
                response = requests.get(
                    self.llm_url,
                    params={
                        'prompt': prompt,
                        'max_length': self.config['llm'].get('max_length', 100),
                        'temperature': 0.1
                    },
                    timeout=60
                )
                answer = extract_llm_text(response.json())
                return qid, answer, "Zero retrieval - direct LLM response", []

            except Exception as e:
                return qid, "I don't know", f"Error: {e}", []

        with ThreadPoolExecutor(max_workers=parallel_threads) as executor:
            futures = {executor.submit(process_single, qid): qid for qid in qids}

            with tqdm(total=len(qids), desc="  Zero retrieval") as pbar:
                for future in as_completed(futures):
                    qid, answer, chain, ctx = future.result()
                    predictions[qid] = answer
                    chains[qid] = chain
                    contexts[qid] = ctx
                    pbar.update(1)

        return {'predictions': predictions, 'chains': chains, 'contexts': contexts}

    def process_single_retrieval(self, qids: list, test_data_map: dict, dataset_name: str) -> dict:
        """Process questions with Single BM25 Retrieval."""
        predictions = {}
        chains = {}
        contexts = {}

        corpus_name = self.get_corpus_name(dataset_name)
        parallel_threads = self.config.get('execution', {}).get('parallel_threads', 8)

        def process_single(qid):
            question_text = test_data_map[qid]['question_text']

            # Step 1: BM25 Retrieve
            try:
                retrieval_response = requests.post(
                    self.retriever_url,
                    json={
                        "retrieval_method": "retrieve_from_elasticsearch",
                        "query_text": question_text,
                        "rerank_query_text": question_text,
                        "max_hits_count": 8,
                        "max_buffer_count": 40,
                        "corpus_name": corpus_name,
                        "document_type": "title_paragraph_text",
                        "retrieval_backend": "bm25"
                    },
                    timeout=30
                )
                hits = retrieval_response.json().get('retrieval', [])

            except Exception as e:
                return qid, "I don't know", f"Retrieval error: {e}", []

            if not hits:
                return qid, "I don't know", "No documents retrieved", []

            # Step 2: Build context
            context_parts = []
            for i, hit in enumerate(hits[:5]):
                title = hit.get('title', '')
                text = hit.get('paragraph_text', '')
                context_parts.append(f"[{i+1}] {title}: {text[:500]}")

            context = "\n\n".join(context_parts)

            # Step 3: Generate answer
            prompt = f"""Based on the following context, answer the question concisely. Give only the answer, no explanation.

Context:
{context}

Question: {question_text}

Answer:"""

            try:
                response = requests.get(
                    self.llm_url,
                    params={
                        'prompt': prompt,
                        'max_length': self.config['llm'].get('max_length', 100),
                        'temperature': 0.1
                    },
                    timeout=60
                )
                answer = extract_llm_text(response.json())

                chain = f"Single BM25 retrieval\nRetrieved {len(hits)} documents\nTop docs: {[h['title'] for h in hits[:3]]}"
                return qid, answer, chain, hits[:5]

            except Exception as e:
                return qid, "I don't know", f"LLM error: {e}", hits[:5]

        with ThreadPoolExecutor(max_workers=parallel_threads) as executor:
            futures = {executor.submit(process_single, qid): qid for qid in qids}

            with tqdm(total=len(qids), desc="  Single retrieval") as pbar:
                for future in as_completed(futures):
                    qid, answer, chain, ctx = future.result()
                    predictions[qid] = answer
                    chains[qid] = chain
                    contexts[qid] = ctx
                    pbar.update(1)

        return {'predictions': predictions, 'chains': chains, 'contexts': contexts}

    def process_multi_hop(self, qids: list, test_data_map: dict, dataset_name: str) -> dict:
        """Process questions with Multi-hop reasoning."""
        predictions = {}
        chains = {}
        contexts = {}

        retriever_config = {
            'host': self.config['retriever']['host'],
            'port': self.config['retriever']['port']
        }
        llm_config = {
            'host': self.config['llm']['server_host'],
            'port': self.config['llm']['server_port']
        }

        parallel_threads = self.config.get('execution', {}).get('parallel_threads', 8)
        print(f"  Using {parallel_threads} parallel threads for multi-hop reasoning")

        def process_single(qid):
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

        with ThreadPoolExecutor(max_workers=parallel_threads) as executor:
            futures = {executor.submit(process_single, qid): qid for qid in qids}

            with tqdm(total=len(qids), desc="  Multi-hop reasoning") as pbar:
                for future in as_completed(futures):
                    qid, result, error = future.result()

                    if error:
                        predictions[qid] = "I don't know"
                        chains[qid] = f"Error: {error}"
                        contexts[qid] = []
                    else:
                        predictions[qid] = result['answer']
                        chains[qid] = result['chain']
                        contexts[qid] = result['contexts']

                    pbar.update(1)

        return {'predictions': predictions, 'chains': chains, 'contexts': contexts}


def main():
    parser = argparse.ArgumentParser(description="Adaptive-RAG Stage 2: Generation")
    parser.add_argument('--config', default='adaptive_rag/evaluate/config.yaml',
                       help='Path to config file')
    parser.add_argument('--datasets', nargs='+',
                       help='Datasets to process (default: all in config)')

    args = parser.parse_args()

    config = ConfigLoader.load_config(args.config)
    generator = AdaptiveStage2Generator(config)
    generator.run(datasets=args.datasets)


if __name__ == '__main__':
    main()
