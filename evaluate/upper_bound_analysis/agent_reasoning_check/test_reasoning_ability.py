"""
Agent Reasoning Ability Test (Agent 逻辑规划能力测试)

目的：测试 LLM 的逻辑规划能力是否足以驾驭复杂的多跳推理
用途：验证是否需要 ToT 等增强推理策略

核心逻辑：
1. 保持检索器不变（本地 BM25/HNSW）
2. 替换 LLM "大脑"（Llama-3-8B → GPT-4o/DeepSeek-V3）
3. 运行完整的 RAG 流程
4. 对比性能差异

结果解读：
- 如果 GPT-4 性能暴涨（EM +30%）→ 本地 LLM 逻辑能力不足，需要 ToT
- 如果 GPT-4 性能依然低 → 检索器太差，需要优化检索策略
"""

import json
import os
import sys
from typing import List, Dict, Any
from tqdm import tqdm
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add project root directory to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, project_root)

# Add current directory to path for llm_backend
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from llm_backend import create_backend, LLMBackend
from metrics.squad_answer_em_f1 import SquadAnswerEmF1Metric


class ReasoningAbilityTester:
    """推理能力测试器"""

    def __init__(
        self,
        llm_backend: LLMBackend,
        retriever_host: str = "localhost",
        retriever_port: int = 8001,
        output_dir: str = "evaluate/upper_bound_analysis/agent_reasoning_check/outputs",
        parallel_threads: int = 6
    ):
        """
        初始化测试器

        Args:
            llm_backend: LLM Backend
            retriever_host: 检索服务 host
            retriever_port: 检索服务 port
            output_dir: 输出目录
            parallel_threads: 并行线程数
        """
        self.llm_backend = llm_backend
        self.retriever_url = f"http://{retriever_host}:{retriever_port}/retrieve/"
        self.output_dir = output_dir
        self.parallel_threads = parallel_threads
        self.metric = SquadAnswerEmF1Metric()
        self.current_corpus_name = 'wiki'  # 默认corpus，会在运行时更新

        os.makedirs(output_dir, exist_ok=True)

    def load_test_data(
        self,
        dataset_name: str,
        data_dir: str = "processed_data",
        max_samples: int = None
    ) -> List[Dict]:
        """加载测试数据"""
        test_file = os.path.join(data_dir, dataset_name, "test_subsampled.jsonl")

        if not os.path.exists(test_file):
            raise FileNotFoundError(f"Test file not found: {test_file}")

        data = []
        with open(test_file, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if max_samples and i >= max_samples:
                    break
                if line.strip():
                    data.append(json.loads(line))

        print(f"✓ Loaded {len(data)} examples from {dataset_name}")
        return data

    def retrieve_contexts(self, question: str, top_k: int = 5) -> List[Dict]:
        """
        调用检索服务（使用与stage2相同的完整格式）

        Args:
            question: 问题
            top_k: 返回文档数量

        Returns:
            检索到的文档列表
        """
        try:
            response = requests.post(
                self.retriever_url,
                json={
                    'retrieval_method': 'retrieve_from_elasticsearch',
                    'query_text': question,
                    'rerank_query_text': question,
                    'max_hits_count': top_k,
                    'max_buffer_count': top_k * 4,
                    'corpus_name': self.current_corpus_name,
                    'document_type': 'title_paragraph_text',
                    'retrieval_backend': 'hybrid'
                },
                timeout=30
            )

            if response.status_code != 200:
                print(f"Warning: Retriever failed (status {response.status_code})")
                if response.status_code == 400:
                    print(f"  Response: {response.text[:200]}")
                return []

            result = response.json()
            return result.get('retrieval', [])

        except Exception as e:
            print(f"Warning: Retriever error: {e}")
            return []

    def generate_answer(
        self,
        question: str,
        contexts: List[Dict],
        max_tokens: int = 300,
        temperature: float = 0.1
    ) -> str:
        """
        使用 LLM 生成答案

        Args:
            question: 问题
            contexts: 检索到的上下文
            max_tokens: 最大生成长度
            temperature: 温度参数

        Returns:
            生成的答案
        """
        # 构建 Prompt
        context_str = self._format_contexts(contexts)

        prompt = f"""Answer the question based on the provided documents. Be concise and accurate.

*** DOCUMENTS ***
{context_str}

*** QUESTION ***
{question}

*** INSTRUCTIONS ***
1. Read the documents carefully
2. Extract the answer directly from the documents
3. Provide a concise answer (1-5 words if possible)

Answer: """

        # 调用 LLM
        response = self.llm_backend.generate(prompt, max_tokens, temperature)

        # 提取答案
        import re
        ans_match = re.search(r'Answer:\s*(.*)', response, re.IGNORECASE)
        if ans_match:
            answer = ans_match.group(1).strip().split('\n')[0].strip()
            return answer

        # 如果没有 Answer: 标记，返回最后一行
        lines = [l.strip() for l in response.split('\n') if l.strip()]
        return lines[-1] if lines else "I don't know"

    def _format_contexts(self, contexts: List[Dict]) -> str:
        """格式化上下文"""
        context_parts = []
        for i, ctx in enumerate(contexts, 1):
            title = ctx.get('title', 'Unknown')
            text = ctx.get('paragraph_text', ctx.get('text', ''))
            context_parts.append(f"[{i}] {title}: {text}")

        return "\n\n".join(context_parts)

    def extract_gold_answers(self, item: Dict) -> List[str]:
        """提取标准答案"""
        answers = []

        if 'answers_objects' in item:
            for ans_obj in item['answers_objects']:
                if ans_obj.get('spans'):
                    spans = ans_obj['spans']
                    if isinstance(spans, list) and len(spans) > 0:
                        answers.append(str(spans[0]))
                    elif isinstance(spans, str):
                        answers.append(spans)
                elif ans_obj.get('number'):
                    answers.append(str(ans_obj['number']))

        answers = list(set(answers))
        return answers if answers else [""]

    def run_evaluation(
        self,
        dataset_name: str,
        data_dir: str = "processed_data",
        max_samples: int = None,
        top_k: int = 5
    ) -> Dict[str, Any]:
        """
        运行评估

        Args:
            dataset_name: 数据集名称
            data_dir: 数据目录
            max_samples: 最大样本数
            top_k: 检索文档数量

        Returns:
            评估结果
        """
        print(f"\n{'='*70}")
        print(f"Reasoning Ability Test: {dataset_name}")
        print(f"LLM Backend: {self.llm_backend.model_name}")
        print(f"Using {self.parallel_threads} parallel threads")
        print(f"{'='*70}")

        # 设置corpus_name（与stage2保持一致）
        dataset_to_corpus = {
            'hotpotqa': 'hotpotqa',
            'musique': 'musique',
            '2wikimultihopqa': '2wikimultihopqa',
            'iirc': 'iirc',
            'wiki': 'wiki'
        }
        self.current_corpus_name = dataset_to_corpus.get(dataset_name.lower(), 'wiki')
        print(f"Corpus: {self.current_corpus_name}")

        # 加载数据
        test_data = self.load_test_data(dataset_name, data_dir, max_samples)

        # 重置指标
        self.metric.reset()

        # 定义单个样本的处理函数
        def process_single_item(item):
            qid = item['question_id']
            question = item['question_text']

            try:
                gold_answers = self.extract_gold_answers(item)

                if not gold_answers:
                    return qid, None, None, f"No gold answers for {qid}"

                # 检索文档
                contexts = self.retrieve_contexts(question, top_k)

                # 生成答案
                predicted_answer = self.generate_answer(question, contexts)

                # 构建结果
                result_item = {
                    'question_id': qid,
                    'question': question,
                    'predicted_answer': predicted_answer,
                    'gold_answers': gold_answers,
                    'num_retrieved_contexts': len(contexts),
                    'is_correct': predicted_answer.lower().strip() in [ans.lower().strip() for ans in gold_answers]
                }

                return qid, predicted_answer, result_item, None

            except Exception as e:
                import traceback
                traceback.print_exc()
                return qid, None, None, str(e)

        # 并行处理所有测试数据
        results_detail = []
        predictions = {}

        with ThreadPoolExecutor(max_workers=self.parallel_threads) as executor:
            # 提交所有任务
            future_to_item = {
                executor.submit(process_single_item, item): item
                for item in test_data
            }

            # 使用tqdm显示进度
            with tqdm(total=len(test_data), desc=f"Testing {dataset_name}") as pbar:
                for future in as_completed(future_to_item):
                    qid, predicted_answer, result_item, error = future.result()

                    if error:
                        print(f"\nWarning: {error}")
                    elif result_item:
                        # 更新指标
                        self.metric(predicted_answer, result_item['gold_answers'])

                        # 保存结果
                        predictions[qid] = predicted_answer
                        results_detail.append(result_item)

                    pbar.update(1)

        # 获取指标
        metrics = self.metric.get_metric(reset=False)

        # 保存结果
        self._save_results(
            dataset_name=dataset_name,
            backend_name=self.llm_backend.model_name,
            metrics=metrics,
            predictions=predictions,
            results_detail=results_detail
        )

        # 打印结果
        self._print_results(dataset_name, metrics)

        return {
            'dataset': dataset_name,
            'backend': self.llm_backend.model_name,
            'metrics': metrics,
            'num_samples': len(test_data)
        }

    def _save_results(
        self,
        dataset_name: str,
        backend_name: str,
        metrics: Dict,
        predictions: Dict,
        results_detail: List[Dict]
    ):
        """保存结果"""
        # 创建数据集特定的输出目录
        dataset_output_dir = os.path.join(self.output_dir, dataset_name)
        os.makedirs(dataset_output_dir, exist_ok=True)

        # 生成文件名（基于 backend）
        backend_slug = backend_name.replace(' ', '_').replace('/', '_').lower()

        # 保存指标
        metrics_file = os.path.join(dataset_output_dir, f"metrics_{backend_slug}.json")
        with open(metrics_file, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
        print(f"\n✓ Metrics saved to {metrics_file}")

        # 保存预测
        predictions_file = os.path.join(dataset_output_dir, f"predictions_{backend_slug}.json")
        with open(predictions_file, 'w', encoding='utf-8') as f:
            json.dump(predictions, f, indent=2, ensure_ascii=False)
        print(f"✓ Predictions saved to {predictions_file}")

        # 保存详细结果
        details_file = os.path.join(dataset_output_dir, f"results_{backend_slug}.jsonl")
        with open(details_file, 'w', encoding='utf-8') as f:
            for result in results_detail:
                f.write(json.dumps(result, ensure_ascii=False) + '\n')
        print(f"✓ Detailed results saved to {details_file}")

    def _print_results(self, dataset_name: str, metrics: Dict):
        """打印结果"""
        print(f"\n{'='*70}")
        print(f"Results for {dataset_name} with {self.llm_backend.model_name}:")
        print(f"  EM (Exact Match):  {metrics['em']:.4f} ({metrics['em']*100:.2f}%)")
        print(f"  F1 Score:          {metrics['f1']:.4f} ({metrics['f1']*100:.2f}%)")
        print(f"  Accuracy:          {metrics['acc']:.4f} ({metrics['acc']*100:.2f}%)")
        print(f"  Recall:            {metrics['recall']:.4f} ({metrics['recall']*100:.2f}%)")
        print(f"  Total Samples:     {metrics['count']}")
        print(f"{'='*70}")


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description="Agent Reasoning Ability Test")
    parser.add_argument('--dataset', required=True,
                       help='Dataset name')
    parser.add_argument('--backend', required=True,
                       choices=['local_llama', 'gpt4', 'deepseek', 'custom'],
                       help='LLM backend to use')
    parser.add_argument('--data-dir', default='processed_data',
                       help='Data directory')
    parser.add_argument('--output-dir',
                       default='evaluate/upper_bound_analysis/agent_reasoning_check/outputs',
                       help='Output directory')
    parser.add_argument('--max-samples', type=int, default=None,
                       help='Maximum number of samples to test')
    parser.add_argument('--top-k', type=int, default=5,
                       help='Number of documents to retrieve')
    parser.add_argument('--parallel-threads', type=int, default=6,
                       help='Number of parallel threads (default: 6)')

    # Backend specific arguments
    parser.add_argument('--llm-host', default='localhost',
                       help='LLM service host (for local_llama)')
    parser.add_argument('--llm-port', default='8000',
                       help='LLM service port (for local_llama)')
    parser.add_argument('--retriever-host', default='localhost',
                       help='Retriever service host')
    parser.add_argument('--retriever-port', default='8001',
                       help='Retriever service port')
    parser.add_argument('--api-key', default=None,
                       help='API key (for gpt4/deepseek)')
    parser.add_argument('--model', default=None,
                       help='Model name (for gpt4/deepseek/custom)')
    parser.add_argument('--api-base', default=None,
                       help='API base URL (for custom backend)')

    args = parser.parse_args()

    print("\n" + "="*70)
    print("AGENT REASONING ABILITY TEST (Agent 逻辑规划能力测试)")
    print("="*70)
    print(f"Dataset: {args.dataset}")
    print(f"Backend: {args.backend}")
    print(f"Parallel Threads: {args.parallel_threads}")
    if args.max_samples:
        print(f"Max Samples: {args.max_samples}")
    print("="*70)

    # 创建 LLM Backend
    backend_kwargs = {}

    if args.backend == "local_llama":
        backend_kwargs = {
            'host': args.llm_host,
            'port': int(args.llm_port)
        }
    elif args.backend == "gpt4":
        backend_kwargs = {
            'model': args.model or 'gpt-4o',
            'api_key': args.api_key
        }
    elif args.backend == "deepseek":
        backend_kwargs = {
            'model': args.model or 'deepseek-chat',
            'api_key': args.api_key
        }
    elif args.backend == "custom":
        if not args.api_base:
            raise ValueError("--api-base is required for custom backend")
        backend_kwargs = {
            'model': args.model or 'unknown',
            'api_base': args.api_base,
            'api_key': args.api_key
        }

    llm_backend = create_backend(args.backend, **backend_kwargs)
    print(f"✓ LLM Backend created: {llm_backend.model_name}")

    # 创建测试器
    tester = ReasoningAbilityTester(
        llm_backend=llm_backend,
        retriever_host=args.retriever_host,
        retriever_port=int(args.retriever_port),
        output_dir=args.output_dir,
        parallel_threads=args.parallel_threads
    )

    try:
        # 运行评估
        result = tester.run_evaluation(
            dataset_name=args.dataset,
            data_dir=args.data_dir,
            max_samples=args.max_samples,
            top_k=args.top_k
        )

        print("\n" + "="*70)
        print("✓ REASONING ABILITY TEST COMPLETE")
        print("="*70)

    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
