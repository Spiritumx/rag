"""
Reader Upper Bound Test (阅读理解上限测试)

目的：测试 LLM 在"已知答案就在文中"的情况下能否答对
用途：诊断瓶颈在检索还是在 LLM/Prompt

核心逻辑：
1. 给 LLM 提供数据集自带的 Gold Paragraphs（专家标注的支撑文档）
2. 让 LLM 基于这些完美文档回答问题
3. 计算 EM/F1 指标
4. 根据结果给出优化建议

结果解读：
- EM > 60-70%：LLM 能力 OK，瓶颈在检索，需要优化检索策略
- EM < 40%：LLM 能力不足，需要优化 Prompt 或换更强模型
"""

import json
import os
import sys
import requests
from typing import List, Dict, Any, Tuple
from tqdm import tqdm
from collections import defaultdict
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add project root directory to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, project_root)
from metrics.squad_answer_em_f1 import SquadAnswerEmF1Metric


class ReaderUpperBoundTester:
    """阅读理解上限测试器"""

    def __init__(self, llm_url: str, output_dir: str = "evaluate/reader_upper_bound/outputs", parallel_threads: int = 6):
        """
        初始化测试器

        Args:
            llm_url: LLM 服务地址
            output_dir: 输出目录
            parallel_threads: 并行线程数
        """
        self.llm_url = llm_url
        self.output_dir = output_dir
        self.parallel_threads = parallel_threads
        self.metric = SquadAnswerEmF1Metric()

        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)

    def load_test_data(self, dataset_name: str, data_dir: str = "processed_data", max_samples: int = None) -> List[Dict]:
        """
        加载测试集数据

        Args:
            dataset_name: 数据集名称
            data_dir: 数据目录
            max_samples: 最大样本数（用于快速测试）

        Returns:
            测试数据列表
        """
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

    def extract_gold_contexts(self, item: Dict) -> Tuple[str, int]:
        """
        提取真实的支持段落作为上下文

        Args:
            item: 测试数据项

        Returns:
            (拼接后的上下文字符串, 支持段落数量)
        """
        contexts = item.get('contexts', [])

        # 提取所有支持段落
        supporting_contexts = [
            ctx for ctx in contexts
            if ctx.get('is_supporting', False)
        ]

        # 如果没有明确标记 supporting 的段落，使用所有段落
        if not supporting_contexts:
            supporting_contexts = contexts

        # 拼接上下文
        context_parts = []
        for i, ctx in enumerate(supporting_contexts, 1):
            title = ctx.get('title', 'Unknown')
            text = ctx.get('paragraph_text', '')
            context_parts.append(f"Document [{i}] - {title}:\n{text}")

        context_str = "\n\n".join(context_parts)
        return context_str, len(supporting_contexts)

    def extract_gold_answers(self, item: Dict) -> List[str]:
        """
        提取真实答案

        Args:
            item: 测试数据项

        Returns:
            答案列表
        """
        answers = []

        # 从 answers_objects 提取
        if 'answers_objects' in item:
            for ans_obj in item['answers_objects']:
                # 提取 spans
                if ans_obj.get('spans'):
                    spans = ans_obj['spans']
                    if isinstance(spans, list) and len(spans) > 0:
                        answers.append(str(spans[0]))
                    elif isinstance(spans, str):
                        answers.append(spans)

                # 提取 number
                elif ans_obj.get('number'):
                    answers.append(str(ans_obj['number']))

        # 去重
        answers = list(set(answers))

        return answers if answers else [""]

    def build_prompt(self, question: str, context: str, prompt_style: str = "standard") -> str:
        """
        构建 Prompt

        Args:
            question: 问题
            context: 上下文
            prompt_style: Prompt 风格 (standard, cot, structured)

        Returns:
            完整的 prompt
        """
        if prompt_style == "cot":
            # Chain-of-Thought Prompting
            prompt = f"""Read the following documents carefully and answer the question.

*** DOCUMENTS ***
{context}

*** QUESTION ***
{question}

*** INSTRUCTIONS ***
1. First, identify which document(s) contain the answer
2. Then, extract the relevant information
3. Finally, provide a concise answer (1-5 words if possible)

Think step by step:
Step 1 - Relevant documents:
Step 2 - Key information:
Step 3 - Final answer:

Answer: """

        elif prompt_style == "structured":
            # Structured Prompting
            prompt = f"""You are given documents that contain the answer to the question.

DOCUMENTS:
{context}

QUESTION:
{question}

TASK:
Extract the answer from the documents above. The answer should be:
- Accurate: directly from the documents
- Concise: 1-5 words when possible
- Specific: avoid vague responses

OUTPUT FORMAT:
Answer: <your concise answer>

Answer: """

        else:
            # Standard Prompting
            prompt = f"""Instructions:
1. You are a precise Question Answering machine.
2. Read the following documents carefully.
3. Answer the question using ONLY the information from the documents.
4. Output VERY SHORT answers (1-5 words). Do NOT write full sentences.
5. Do NOT say "The answer is...". Just output the entity or phrase.

Context:
{context}

Q: {question}
A: """

        return prompt

    def generate_answer(
        self,
        question: str,
        context: str,
        prompt_style: str = "standard",
        max_length: int = 300,
        temperature: float = 0.1
    ) -> str:
        """
        使用真实上下文生成答案

        Args:
            question: 问题
            context: 真实上下文
            prompt_style: Prompt 风格
            max_length: 最大生成长度
            temperature: 温度参数

        Returns:
            生成的答案
        """
        prompt = self.build_prompt(question, context, prompt_style)

        try:
            response = requests.get(
                self.llm_url,
                params={'prompt': prompt, 'max_length': max_length, 'temperature': temperature},
                timeout=40
            )

            if response.status_code != 200:
                return "Error: LLM service failed"

            llm_output = response.json()

            # 提取文本
            text = ""
            if 'generated_texts' in llm_output:
                texts = llm_output['generated_texts']
                text = texts[0] if isinstance(texts, list) and len(texts) > 0 else ""
            elif 'text' in llm_output:
                text = llm_output['text']
            elif 'generated_text' in llm_output:
                text = llm_output['generated_text']
            elif 'choices' in llm_output:
                choices = llm_output['choices']
                if isinstance(choices, list) and len(choices) > 0:
                    text = choices[0].get('text', '') or choices[0].get('message', {}).get('content', '')

            text = text.strip()

            # 解析 Answer:
            ans_match = re.search(r'Answer:\s*(.*)', text, re.IGNORECASE | re.DOTALL)
            if ans_match:
                answer = ans_match.group(1).strip().split('\n')[0].strip()
                return answer

            # 如果没有 Answer: 标记，返回最后一个非空行
            lines = [l.strip() for l in text.split('\n') if l.strip()]
            return lines[-1] if lines else "I don't know"

        except Exception as e:
            print(f"  Error generating answer: {e}")
            return "Error"

    def run_evaluation(
        self,
        dataset_name: str,
        data_dir: str = "processed_data",
        max_samples: int = None,
        prompt_style: str = "standard"
    ) -> Dict[str, Any]:
        """
        运行评估

        Args:
            dataset_name: 数据集名称
            data_dir: 数据目录
            max_samples: 最大样本数
            prompt_style: Prompt 风格

        Returns:
            评估结果
        """
        print(f"\n{'='*70}")
        print(f"Reader Upper Bound Test: {dataset_name}")
        print(f"Prompt Style: {prompt_style}")
        print(f"Using {self.parallel_threads} parallel threads")
        print(f"{'='*70}")

        # 加载数据
        test_data = self.load_test_data(dataset_name, data_dir, max_samples)

        # 重置指标
        self.metric.reset()

        # 定义单个样本的处理函数
        def process_single_item(item):
            qid = item['question_id']
            question = item['question_text']

            try:
                # 提取真实上下文和答案
                gold_context, num_contexts = self.extract_gold_contexts(item)
                gold_answers = self.extract_gold_answers(item)

                if not gold_answers:
                    return qid, None, None, f"No gold answers for {qid}"

                # 生成答案
                predicted_answer = self.generate_answer(
                    question=question,
                    context=gold_context,
                    prompt_style=prompt_style
                )

                # 计算是否正确
                is_correct = (predicted_answer.lower().strip() in [ans.lower().strip() for ans in gold_answers])

                # 构建结果
                result_item = {
                    'question_id': qid,
                    'question': question,
                    'predicted_answer': predicted_answer,
                    'gold_answers': gold_answers,
                    'num_gold_contexts': num_contexts,
                    'is_correct': is_correct
                }

                # 构建错误案例（如果错误）
                error_item = None
                if not is_correct:
                    error_item = {
                        **result_item,
                        'gold_context_preview': gold_context[:500] + "..." if len(gold_context) > 500 else gold_context
                    }

                return qid, result_item, error_item, None

            except Exception as e:
                import traceback
                traceback.print_exc()
                return qid, None, None, str(e)

        # 并行处理所有测试数据
        results_detail = []
        error_cases = []

        with ThreadPoolExecutor(max_workers=self.parallel_threads) as executor:
            # 提交所有任务
            future_to_item = {
                executor.submit(process_single_item, item): item
                for item in test_data
            }

            # 使用tqdm显示进度
            with tqdm(total=len(test_data), desc=f"Testing {dataset_name}") as pbar:
                for future in as_completed(future_to_item):
                    qid, result_item, error_item, error = future.result()

                    if error:
                        print(f"\n  Warning: {error}")
                    elif result_item:
                        # 更新指标
                        self.metric(result_item['predicted_answer'], result_item['gold_answers'])

                        # 保存结果
                        results_detail.append(result_item)

                        if error_item:
                            error_cases.append(error_item)

                    pbar.update(1)

        # 获取最终指标
        metrics = self.metric.get_metric(reset=False)

        # 保存结果
        self._save_results(dataset_name, metrics, results_detail, error_cases, prompt_style)

        # 打印结果
        self._print_results(dataset_name, metrics)

        # 给出诊断建议
        self._print_diagnosis(metrics)

        return {
            'dataset': dataset_name,
            'metrics': metrics,
            'num_samples': len(test_data),
            'error_cases': len(error_cases)
        }

    def _save_results(
        self,
        dataset_name: str,
        metrics: Dict,
        results_detail: List[Dict],
        error_cases: List[Dict],
        prompt_style: str
    ):
        """保存结果到文件"""

        # 创建数据集特定的输出目录
        dataset_output_dir = os.path.join(self.output_dir, dataset_name)
        os.makedirs(dataset_output_dir, exist_ok=True)

        # 保存指标
        metrics_file = os.path.join(dataset_output_dir, f"metrics_{prompt_style}.json")
        with open(metrics_file, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
        print(f"✓ Metrics saved to {metrics_file}")

        # 保存详细结果
        details_file = os.path.join(dataset_output_dir, f"results_{prompt_style}.jsonl")
        with open(details_file, 'w', encoding='utf-8') as f:
            for result in results_detail:
                f.write(json.dumps(result, ensure_ascii=False) + '\n')
        print(f"✓ Detailed results saved to {details_file}")

        # 保存错误案例
        if error_cases:
            errors_file = os.path.join(dataset_output_dir, f"error_cases_{prompt_style}.jsonl")
            with open(errors_file, 'w', encoding='utf-8') as f:
                for error in error_cases:
                    f.write(json.dumps(error, ensure_ascii=False) + '\n')
            print(f"✓ Error cases saved to {errors_file}")

    def _print_results(self, dataset_name: str, metrics: Dict):
        """打印评估结果"""
        print(f"\n{'='*70}")
        print(f"Results for {dataset_name}:")
        print(f"  EM (Exact Match):  {metrics['em']:.4f} ({metrics['em']*100:.2f}%)")
        print(f"  F1 Score:          {metrics['f1']:.4f} ({metrics['f1']*100:.2f}%)")
        print(f"  Accuracy:          {metrics['acc']:.4f} ({metrics['acc']*100:.2f}%)")
        print(f"  Recall:            {metrics['recall']:.4f} ({metrics['recall']*100:.2f}%)")
        print(f"  Total Samples:     {metrics['count']}")
        print(f"{'='*70}")

    def _print_diagnosis(self, metrics: Dict):
        """打印诊断建议"""
        em = metrics['em']

        print(f"\n{'='*70}")
        print("DIAGNOSIS & RECOMMENDATIONS")
        print(f"{'='*70}")

        if em >= 0.70:
            print("\n✓ LLM 阅读理解能力: EXCELLENT (EM >= 70%)")
            print("\n【结论】")
            print("  模型在拿到正确文档的情况下能很好地回答问题。")
            print("  这说明瓶颈在检索环节，而不是 LLM 本身。")
            print("\n【建议优化方向】")
            print("  1. 优先优化检索策略：")
            print("     - 实现门控召回 (Gated Retrieval)")
            print("     - 使用 ToT (Tree of Thought) 检索")
            print("     - 优化 IRCoT 多跳检索的 query rewrite")
            print("  2. 提升检索召回率：")
            print("     - 增加检索文档数量 (top_k)")
            print("     - 使用混合检索 (Hybrid: BM25 + Dense)")
            print("     - 添加 Reranker 重排序")
            print("  3. 不需要优化 LLM 或 Prompt")

        elif em >= 0.50:
            print("\n○ LLM 阅读理解能力: GOOD (50% <= EM < 70%)")
            print("\n【结论】")
            print("  模型具备一定的阅读理解能力，但仍有提升空间。")
            print("  瓶颈可能同时存在于检索和 LLM/Prompt。")
            print("\n【建议优化方向】")
            print("  1. 同时优化检索和 Prompt：")
            print("     - 先尝试优化 Prompt (成本低)")
            print("     - 再优化检索策略")
            print("  2. Prompt 优化方法：")
            print("     - 使用 Chain-of-Thought (CoT) Prompting")
            print("     - 明确指示 LLM 从文档中提取答案")
            print("     - 添加示例 (Few-shot Learning)")
            print("  3. 检索优化方法：")
            print("     - 提高检索精度和召回率")
            print("     - 使用重排序模型")

        elif em >= 0.30:
            print("\n△ LLM 阅读理解能力: MODERATE (30% <= EM < 50%)")
            print("\n【结论】")
            print("  即使给了正确文档，模型答题能力也比较弱。")
            print("  主要瓶颈在 LLM 能力或 Prompt 设计。")
            print("\n【建议优化方向】")
            print("  1. 优先优化 Prompt：")
            print("     - 使用 Chain-of-Thought (CoT)")
            print("     - 结构化 Prompt (Structured Prompting)")
            print("     - Few-shot Examples")
            print("  2. 考虑微调 LLM：")
            print("     - 在 SQuAD/HotpotQA 等数据集上微调")
            print("     - 使用 LoRA 等参数高效微调方法")
            print("  3. 考虑换更强的模型：")
            print("     - Qwen-2.5-7B/14B")
            print("     - Llama-3-8B/70B")
            print("  4. 此时优化检索的收益有限")

        else:
            print("\n✗ LLM 阅读理解能力: POOR (EM < 30%)")
            print("\n【结论】")
            print("  模型阅读理解能力严重不足。")
            print("  优化检索毫无意义，必须先解决 LLM 问题。")
            print("\n【建议优化方向】")
            print("  1. 必须换更强的模型：")
            print("     - 当前模型 (Llama-3-8B) 可能太弱")
            print("     - 推荐: Qwen-2.5-14B, Llama-3-70B, GPT-4")
            print("  2. 大幅度优化 Prompt：")
            print("     - 使用 CoT + Few-shot")
            print("     - 详细的 step-by-step 指令")
            print("  3. 微调 LLM (如果资源允许)：")
            print("     - 在 QA 数据集上全参数微调或 LoRA")
            print("  4. 完全不要在检索上浪费时间")

        print(f"{'='*70}\n")


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description="Reader Upper Bound Test")
    parser.add_argument('--datasets', nargs='+',
                       default=['musique', '2wikimultihopqa'],
                       help='Datasets to evaluate')
    parser.add_argument('--llm-host', default='localhost',
                       help='LLM service host')
    parser.add_argument('--llm-port', default='8000',
                       help='LLM service port')
    parser.add_argument('--data-dir', default='processed_data',
                       help='Data directory')
    parser.add_argument('--output-dir', default='evaluate/reader_upper_bound/outputs',
                       help='Output directory')
    parser.add_argument('--max-samples', type=int, default=None,
                       help='Maximum number of samples to test (for quick testing)')
    parser.add_argument('--prompt-style', default='standard',
                       choices=['standard', 'cot', 'structured'],
                       help='Prompt style to use')
    parser.add_argument('--parallel-threads', type=int, default=6,
                       help='Number of parallel threads (default: 6)')

    args = parser.parse_args()

    # LLM URL
    llm_url = f"http://{args.llm_host}:{args.llm_port}/generate"

    print("\n" + "="*70)
    print("READER UPPER BOUND TEST (阅读理解上限测试)")
    print("="*70)
    print(f"LLM Service: {llm_url}")
    print(f"Datasets: {', '.join(args.datasets)}")
    print(f"Prompt Style: {args.prompt_style}")
    print(f"Parallel Threads: {args.parallel_threads}")
    if args.max_samples:
        print(f"Max Samples: {args.max_samples} (Quick Test Mode)")
    print("="*70)

    # 创建测试器
    tester = ReaderUpperBoundTester(llm_url, args.output_dir, args.parallel_threads)

    # 运行评估
    all_results = []
    for dataset_name in args.datasets:
        try:
            result = tester.run_evaluation(
                dataset_name=dataset_name,
                data_dir=args.data_dir,
                max_samples=args.max_samples,
                prompt_style=args.prompt_style
            )
            all_results.append(result)
        except Exception as e:
            print(f"\n✗ Error evaluating {dataset_name}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # 汇总结果
    if all_results:
        print("\n" + "="*70)
        print("OVERALL SUMMARY")
        print("="*70)

        total_em = 0
        total_f1 = 0
        total_count = 0

        for result in all_results:
            metrics = result['metrics']
            dataset = result['dataset']

            print(f"\n{dataset}:")
            print(f"  EM:  {metrics['em']:.4f} ({metrics['em']*100:.2f}%)")
            print(f"  F1:  {metrics['f1']:.4f} ({metrics['f1']*100:.2f}%)")
            print(f"  Count: {metrics['count']}")

            total_em += metrics['em'] * metrics['count']
            total_f1 += metrics['f1'] * metrics['count']
            total_count += metrics['count']

        # 总体平均
        if total_count > 0:
            avg_em = total_em / total_count
            avg_f1 = total_f1 / total_count

            print(f"\n{'='*70}")
            print(f"OVERALL AVERAGE:")
            print(f"  EM:  {avg_em:.4f} ({avg_em*100:.2f}%)")
            print(f"  F1:  {avg_f1:.4f} ({avg_f1*100:.2f}%)")
            print(f"  Total Samples: {total_count}")
            print(f"{'='*70}")

            # 保存总体结果
            summary_file = os.path.join(args.output_dir, f"summary_{args.prompt_style}.json")
            summary = {
                'overall': {
                    'em': round(avg_em, 4),
                    'f1': round(avg_f1, 4),
                    'count': total_count
                },
                'by_dataset': {
                    r['dataset']: r['metrics']
                    for r in all_results
                },
                'config': {
                    'prompt_style': args.prompt_style,
                    'llm_url': llm_url
                }
            }

            with open(summary_file, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
            print(f"\n✓ Summary saved to {summary_file}")

    print("\n" + "="*70)
    print("✓ READER UPPER BOUND TEST COMPLETE")
    print("="*70)


if __name__ == '__main__':
    main()
