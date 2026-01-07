"""
Retriever Recall Upper Bound Test (检索器物理召回上限测试)

目的：测试检索器能否把包含正确答案的文档召回
用途：诊断瓶颈在粗排、精排还是 Top-K 截断

核心逻辑：
1. 加载检索结果（predictions_contexts.json）
2. 加载标准答案（Gold Answers）
3. 检查答案字符串是否出现在检索到的文档中
4. 计算 Recall@K (K=5, 20, 100)

结果解读：
- Recall@100 < 50%：粗排就漏了，需要优化召回策略（语义门控、混合检索）
- Recall@100 > 80% 但 Recall@5 < 40%：精排有问题，需要更好的 Reranker 或 ToT
- Recall@5 > 60%：检索没问题，瓶颈在 LLM
"""

import json
import os
import sys
from typing import List, Dict, Any, Tuple, Set
from collections import defaultdict
from tqdm import tqdm
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add project root directory to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, project_root)


class RetrievalRecallTester:
    """检索召回测试器"""

    def __init__(self, output_dir: str = "evaluate/upper_bound_analysis/retriever_recall_upperbound/outputs", parallel_threads: int = 6):
        """
        初始化测试器

        Args:
            output_dir: 输出目录
            parallel_threads: 并行线程数
        """
        self.output_dir = output_dir
        self.parallel_threads = parallel_threads
        os.makedirs(output_dir, exist_ok=True)

    def load_gold_data(self, dataset_name: str, data_dir: str = "processed_data") -> Dict[str, Dict]:
        """
        加载标准数据（包含 Gold Answers 和 Gold Contexts）

        Args:
            dataset_name: 数据集名称
            data_dir: 数据目录

        Returns:
            {question_id: {question, gold_answers, gold_contexts}}
        """
        test_file = os.path.join(data_dir, dataset_name, "test_subsampled.jsonl")

        if not os.path.exists(test_file):
            raise FileNotFoundError(f"Test file not found: {test_file}")

        gold_data = {}
        with open(test_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    qid = item['question_id']

                    # 提取 Gold Answers
                    gold_answers = self._extract_gold_answers(item)

                    # 提取 Gold Context IDs（用于后续对比）
                    gold_context_ids = set()
                    for ctx in item.get('contexts', []):
                        if ctx.get('is_supporting', False):
                            # 使用 title 作为唯一标识
                            gold_context_ids.add(ctx.get('title', ''))

                    gold_data[qid] = {
                        'question': item['question_text'],
                        'gold_answers': gold_answers,
                        'gold_context_ids': gold_context_ids,
                        'contexts': item.get('contexts', [])
                    }

        print(f"✓ Loaded {len(gold_data)} examples from {dataset_name}")
        return gold_data

    def _extract_gold_answers(self, item: Dict) -> List[str]:
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

        # 去重并转小写（用于匹配）
        answers = list(set(answers))
        return answers if answers else [""]

    def load_retrieval_results(self, predictions_file: str) -> Dict[str, List[Dict]]:
        """
        加载检索结果

        Args:
            predictions_file: 预测文件路径（predictions_contexts.json）

        Returns:
            {question_id: [retrieved_contexts]}
        """
        if not os.path.exists(predictions_file):
            raise FileNotFoundError(f"Predictions file not found: {predictions_file}")

        with open(predictions_file, 'r', encoding='utf-8') as f:
            predictions = json.load(f)

        print(f"✓ Loaded retrieval results from {predictions_file}")
        return predictions

    def check_answer_in_context(self, answer: str, context_text: str, fuzzy: bool = True) -> bool:
        """
        检查答案是否在文档中

        Args:
            answer: 标准答案
            context_text: 文档文本
            fuzzy: 是否使用模糊匹配

        Returns:
            是否包含答案
        """
        answer_lower = answer.lower().strip()
        context_lower = context_text.lower()

        if fuzzy:
            # 模糊匹配：允许部分匹配和词序变化
            # 1. 直接包含
            if answer_lower in context_lower:
                return True

            # 2. 移除标点后匹配
            answer_clean = re.sub(r'[^\w\s]', '', answer_lower)
            context_clean = re.sub(r'[^\w\s]', '', context_lower)
            if answer_clean in context_clean:
                return True

            # 3. 分词后检查所有关键词是否都出现
            answer_words = set(answer_clean.split())
            if len(answer_words) > 1:
                # 对于多词答案，检查是否所有词都出现
                context_words = set(context_clean.split())
                if answer_words.issubset(context_words):
                    return True

            return False
        else:
            # 严格匹配
            return answer_lower in context_lower

    def calculate_recall_at_k(
        self,
        gold_data: Dict[str, Dict],
        retrieval_results: Dict[str, List[Dict]],
        k_values: List[int] = [5, 20, 100],
        fuzzy_match: bool = True
    ) -> Dict[str, Any]:
        """
        计算 Recall@K

        Args:
            gold_data: 标准数据
            retrieval_results: 检索结果
            k_values: K值列表
            fuzzy_match: 是否使用模糊匹配

        Returns:
            召回率结果
        """
        print(f"\n{'='*70}")
        print(f"Calculating Recall@K (K={k_values})")
        print(f"Fuzzy Match: {fuzzy_match}")
        print(f"Using {self.parallel_threads} parallel threads")
        print(f"{'='*70}")

        # 定义单个问题的处理函数
        def process_single_question(qid, gold_info):
            gold_answers = gold_info['gold_answers']

            try:
                # 获取检索结果
                if qid not in retrieval_results:
                    return qid, None, f"No retrieval results for {qid}"

                retrieved_contexts = retrieval_results[qid]

                # 对于每个 K 值，检查前 K 个文档
                result_item = {
                    'question_id': qid,
                    'question': gold_info['question'],
                    'gold_answers': gold_answers,
                    'num_retrieved': len(retrieved_contexts)
                }

                k_results = {}
                for k in k_values:
                    top_k_contexts = retrieved_contexts[:k]

                    # 检查是否任何一个答案出现在前 K 个文档中
                    found = False
                    for context in top_k_contexts:
                        context_text = context.get('paragraph_text', '')

                        # 检查是否包含任何一个标准答案
                        for answer in gold_answers:
                            if self.check_answer_in_context(answer, context_text, fuzzy_match):
                                found = True
                                break

                        if found:
                            break

                    k_results[k] = found
                    result_item[f'found_at_{k}'] = found

                return qid, result_item, k_results, None

            except Exception as e:
                import traceback
                traceback.print_exc()
                return qid, None, None, str(e)

        # 并行处理所有问题
        recall_stats = {k: {'hits': 0, 'total': 0} for k in k_values}
        detailed_results = []

        with ThreadPoolExecutor(max_workers=self.parallel_threads) as executor:
            # 提交所有任务
            future_to_qid = {
                executor.submit(process_single_question, qid, gold_info): qid
                for qid, gold_info in gold_data.items()
            }

            # 使用tqdm显示进度
            with tqdm(total=len(gold_data), desc="Calculating Recall") as pbar:
                for future in as_completed(future_to_qid):
                    qid, result_item, k_results, error = future.result()

                    if error:
                        print(f"\nWarning: {error}")
                    elif result_item and k_results:
                        # 更新统计
                        for k in k_values:
                            if k_results[k]:
                                recall_stats[k]['hits'] += 1
                            recall_stats[k]['total'] += 1

                        detailed_results.append(result_item)

                    pbar.update(1)

        # 计算召回率
        recall_scores = {}
        for k in k_values:
            total = recall_stats[k]['total']
            hits = recall_stats[k]['hits']
            recall = hits / total if total > 0 else 0
            recall_scores[f'recall@{k}'] = recall

            print(f"Recall@{k:3d}: {recall:.4f} ({recall*100:.2f}%) - {hits}/{total}")

        return {
            'recall_scores': recall_scores,
            'recall_stats': recall_stats,
            'detailed_results': detailed_results,
            'config': {
                'k_values': k_values,
                'fuzzy_match': fuzzy_match
            }
        }

    def calculate_context_overlap(
        self,
        gold_data: Dict[str, Dict],
        retrieval_results: Dict[str, List[Dict]],
        k: int = 20
    ) -> Dict[str, Any]:
        """
        计算检索文档与 Gold Contexts 的重叠率

        Args:
            gold_data: 标准数据
            retrieval_results: 检索结果
            k: Top-K 值

        Returns:
            重叠率统计
        """
        print(f"\n{'='*70}")
        print(f"Calculating Context Overlap @ {k}")
        print(f"{'='*70}")

        overlap_stats = {
            'perfect_matches': 0,    # 检索到的文档完全包含所有 Gold Contexts
            'partial_matches': 0,    # 检索到部分 Gold Contexts
            'no_matches': 0,         # 没有检索到任何 Gold Context
            'total': 0
        }

        overlap_ratios = []

        for qid, gold_info in tqdm(gold_data.items(), desc="Calculating Overlap"):
            gold_context_ids = gold_info['gold_context_ids']

            if qid not in retrieval_results:
                continue

            retrieved_contexts = retrieval_results[qid][:k]
            retrieved_titles = set(ctx.get('title', '') for ctx in retrieved_contexts)

            # 计算重叠
            overlap = gold_context_ids & retrieved_titles
            overlap_ratio = len(overlap) / len(gold_context_ids) if gold_context_ids else 0

            overlap_ratios.append(overlap_ratio)

            # 分类
            if overlap_ratio == 1.0:
                overlap_stats['perfect_matches'] += 1
            elif overlap_ratio > 0:
                overlap_stats['partial_matches'] += 1
            else:
                overlap_stats['no_matches'] += 1

            overlap_stats['total'] += 1

        # 平均重叠率
        avg_overlap = sum(overlap_ratios) / len(overlap_ratios) if overlap_ratios else 0

        print(f"\nContext Overlap Statistics:")
        print(f"  Perfect Matches: {overlap_stats['perfect_matches']} ({overlap_stats['perfect_matches']/overlap_stats['total']*100:.2f}%)")
        print(f"  Partial Matches: {overlap_stats['partial_matches']} ({overlap_stats['partial_matches']/overlap_stats['total']*100:.2f}%)")
        print(f"  No Matches:      {overlap_stats['no_matches']} ({overlap_stats['no_matches']/overlap_stats['total']*100:.2f}%)")
        print(f"  Average Overlap: {avg_overlap:.4f} ({avg_overlap*100:.2f}%)")

        return {
            'overlap_stats': overlap_stats,
            'average_overlap': avg_overlap,
            'overlap_ratios': overlap_ratios
        }

    def run_evaluation(
        self,
        dataset_name: str,
        predictions_file: str,
        data_dir: str = "processed_data",
        k_values: List[int] = [5, 20, 100],
        fuzzy_match: bool = True
    ) -> Dict[str, Any]:
        """
        运行召回率评估

        Args:
            dataset_name: 数据集名称
            predictions_file: 检索结果文件
            data_dir: 数据目录
            k_values: K值列表
            fuzzy_match: 是否模糊匹配

        Returns:
            评估结果
        """
        print(f"\n{'='*70}")
        print(f"Retriever Recall Upper Bound Test: {dataset_name}")
        print(f"{'='*70}")

        # 加载数据
        gold_data = self.load_gold_data(dataset_name, data_dir)
        retrieval_results = self.load_retrieval_results(predictions_file)

        # 计算 Recall@K
        recall_results = self.calculate_recall_at_k(
            gold_data=gold_data,
            retrieval_results=retrieval_results,
            k_values=k_values,
            fuzzy_match=fuzzy_match
        )

        # 计算文档重叠率（基于 Gold Contexts）
        overlap_results = self.calculate_context_overlap(
            gold_data=gold_data,
            retrieval_results=retrieval_results,
            k=max(k_values)
        )

        # 保存结果
        self._save_results(
            dataset_name=dataset_name,
            recall_results=recall_results,
            overlap_results=overlap_results
        )

        # 打印诊断建议
        self._print_diagnosis(recall_results['recall_scores'])

        return {
            'dataset': dataset_name,
            'recall_results': recall_results,
            'overlap_results': overlap_results
        }

    def _save_results(
        self,
        dataset_name: str,
        recall_results: Dict,
        overlap_results: Dict
    ):
        """保存结果"""
        dataset_output_dir = os.path.join(self.output_dir, dataset_name)
        os.makedirs(dataset_output_dir, exist_ok=True)

        # 保存召回率指标
        recall_file = os.path.join(dataset_output_dir, "recall_metrics.json")
        with open(recall_file, 'w', encoding='utf-8') as f:
            json.dump({
                'recall_scores': recall_results['recall_scores'],
                'recall_stats': recall_results['recall_stats'],
                'config': recall_results['config']
            }, f, indent=2, ensure_ascii=False)
        print(f"\n✓ Recall metrics saved to {recall_file}")

        # 保存详细结果
        details_file = os.path.join(dataset_output_dir, "recall_details.jsonl")
        with open(details_file, 'w', encoding='utf-8') as f:
            for result in recall_results['detailed_results']:
                f.write(json.dumps(result, ensure_ascii=False) + '\n')
        print(f"✓ Detailed results saved to {details_file}")

        # 保存重叠率统计
        overlap_file = os.path.join(dataset_output_dir, "context_overlap.json")
        with open(overlap_file, 'w', encoding='utf-8') as f:
            json.dump({
                'overlap_stats': overlap_results['overlap_stats'],
                'average_overlap': overlap_results['average_overlap']
            }, f, indent=2, ensure_ascii=False)
        print(f"✓ Context overlap saved to {overlap_file}")

    def _print_diagnosis(self, recall_scores: Dict[str, float]):
        """打印诊断建议"""
        recall_100 = recall_scores.get('recall@100', 0)
        recall_20 = recall_scores.get('recall@20', 0)
        recall_5 = recall_scores.get('recall@5', 0)

        print(f"\n{'='*70}")
        print("DIAGNOSIS & RECOMMENDATIONS")
        print(f"{'='*70}")

        # 诊断逻辑
        if recall_100 < 0.50:
            print("\n✗ 粗排召回能力: POOR (Recall@100 < 50%)")
            print("\n【结论】")
            print("  粗排阶段（BM25/HNSW）就把大量正确答案漏掉了。")
            print("  这是最严重的问题，必须优先解决。")
            print("\n【建议优化方向】")
            print("  1. 优先实现语义门控召回（创新点三）：")
            print("     - 根据问题类型动态选择检索策略")
            print("     - 组合多个检索源（BM25 + Dense + Keyword）")
            print("  2. 增加召回文档数量（top_k）：")
            print("     - 将 top_k 从 100 增加到 200-500")
            print("  3. 改进 Query Rewrite：")
            print("     - 优化问题重写质量")
            print("     - 使用多样化的查询策略")
            print("  4. 混合检索：")
            print("     - BM25 + Dense Retrieval")
            print("     - 使用 Reciprocal Rank Fusion (RRF)")

        elif recall_100 >= 0.80 and recall_5 < 0.40:
            print("\n△ 粗排召回能力: GOOD，但精排能力: POOR")
            print(f"  Recall@100 = {recall_100:.2%} (粗排能召回)")
            print(f"  Recall@5   = {recall_5:.2%} (精排后丢失)")
            print("\n【结论】")
            print("  粗排能找到正确文档，但 Reranker 把它们排到后面去了。")
            print("  瓶颈在精排和 Top-K 截断。")
            print("\n【建议优化方向】")
            print("  1. 更换更好的 Reranker：")
            print("     - 使用 Cross-Encoder Reranker")
            print("     - 尝试 ColBERT、SPLADE 等高级模型")
            print("  2. 实现 ToT (Tree of Thought) 检索（创新点二）：")
            print("     - 让 LLM 看更多文档（增加阅读广度）")
            print("     - 探索多条检索路径")
            print("  3. 增加给 LLM 的文档数量：")
            print("     - 将 top_k 从 5 增加到 10-20")
            print("  4. 优化 Reranker 训练：")
            print("     - 在你的数据集上微调 Reranker")

        elif recall_100 >= 0.80 and recall_5 >= 0.60:
            print("\n✓ 检索能力: EXCELLENT")
            print(f"  Recall@100 = {recall_100:.2%}")
            print(f"  Recall@5   = {recall_5:.2%}")
            print("\n【结论】")
            print("  检索器表现很好，能够召回并排序正确文档。")
            print("  如果整体 EM 还是低，说明瓶颈在 LLM/Prompt。")
            print("\n【建议优化方向】")
            print("  1. 检索已经不是瓶颈，不需要优化")
            print("  2. 去优化 Reader（LLM/Prompt）：")
            print("     - 运行 Reader Upper Bound Test")
            print("     - 使用 CoT Prompting")
            print("     - 考虑微调 LLM")

        else:
            print("\n○ 检索能力: MODERATE")
            print(f"  Recall@100 = {recall_100:.2%}")
            print(f"  Recall@20  = {recall_20:.2%}")
            print(f"  Recall@5   = {recall_5:.2%}")
            print("\n【结论】")
            print("  检索能力中等，粗排和精排都有提升空间。")
            print("\n【建议优化方向】")
            print("  1. 如果 Recall@100 < 70%：")
            print("     - 优先优化粗排（混合检索、增加 top_k）")
            print("  2. 如果 Recall@100 > 70% 但 Recall@5 < 50%：")
            print("     - 优先优化精排（Reranker、增加给 LLM 的文档数）")
            print("  3. 同时优化粗排和精排：")
            print("     - 语义门控召回 + 更好的 Reranker")

        print(f"{'='*70}\n")


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description="Retriever Recall Upper Bound Test")
    parser.add_argument('--dataset', required=True,
                       help='Dataset name')
    parser.add_argument('--predictions-file', required=True,
                       help='Path to predictions_contexts.json')
    parser.add_argument('--data-dir', default='processed_data',
                       help='Data directory')
    parser.add_argument('--output-dir',
                       default='evaluate/upper_bound_analysis/retriever_recall_upperbound/outputs',
                       help='Output directory')
    parser.add_argument('--k-values', nargs='+', type=int, default=[5, 20, 100],
                       help='K values for Recall@K')
    parser.add_argument('--strict-match', action='store_true',
                       help='Use strict matching (default: fuzzy match)')
    parser.add_argument('--parallel-threads', type=int, default=6,
                       help='Number of parallel threads (default: 6)')

    args = parser.parse_args()

    print("\n" + "="*70)
    print("RETRIEVER RECALL UPPER BOUND TEST (检索器物理召回上限测试)")
    print("="*70)
    print(f"Dataset: {args.dataset}")
    print(f"Predictions: {args.predictions_file}")
    print(f"K values: {args.k_values}")
    print(f"Match mode: {'Strict' if args.strict_match else 'Fuzzy'}")
    print(f"Parallel Threads: {args.parallel_threads}")
    print("="*70)

    tester = RetrievalRecallTester(args.output_dir, args.parallel_threads)

    try:
        result = tester.run_evaluation(
            dataset_name=args.dataset,
            predictions_file=args.predictions_file,
            data_dir=args.data_dir,
            k_values=args.k_values,
            fuzzy_match=not args.strict_match
        )

        print("\n" + "="*70)
        print("✓ RETRIEVER RECALL TEST COMPLETE")
        print("="*70)

    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
