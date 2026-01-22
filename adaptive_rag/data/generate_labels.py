"""
Adaptive-RAG 结果导向标注脚本

核心标注逻辑：
1. 执行 Zero Retrieval (直接 LLM 回答)
   - 如果答案正确 → 标签 = "Z"
2. 否则执行 Single Retrieval (混合检索 + LLM)
   - 如果答案正确 → 标签 = "S"
3. 否则 → 标签 = "M"

Usage:
    python -m adaptive_rag.data.generate_labels
    python -m adaptive_rag.data.generate_labels --workers 4
"""

import os
import sys
import json
import time
import argparse
import requests
from pathlib import Path
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

import yaml

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from metrics.squad_answer_em_f1 import (
    compute_accuracy,
    metric_max_over_ground_truths
)


def load_config(config_path: str = None) -> dict:
    """加载配置文件"""
    if config_path is None:
        config_path = PROJECT_ROOT / "adaptive_rag" / "config.yaml"

    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def load_jsonl(file_path: str) -> list:
    """加载 JSONL 文件"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def save_jsonl(data: list, file_path: str):
    """保存 JSONL 文件"""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')


def extract_gold_answers(item: dict) -> List[str]:
    """从数据项中提取标准答案"""
    gold_answers = []

    if 'answers_objects' in item:
        for ans_obj in item['answers_objects']:
            formatted = format_answer(ans_obj)
            if formatted:
                gold_answers.append(formatted)

    # Fallback: check for 'answers' field
    if not gold_answers and 'answers' in item:
        if isinstance(item['answers'], list):
            gold_answers = [str(ans) for ans in item['answers']]
        else:
            gold_answers = [str(item['answers'])]

    return gold_answers


def format_answer(answer_obj: dict) -> Optional[str]:
    """格式化答案对象"""
    if isinstance(answer_obj, str):
        return answer_obj

    # Handle number
    if answer_obj.get('number'):
        return str(answer_obj['number'])

    # Handle spans
    if answer_obj.get('spans'):
        spans = answer_obj['spans']
        if isinstance(spans, list) and len(spans) > 0:
            return str(spans[0])
        elif isinstance(spans, str):
            return spans

    # Handle date
    date = answer_obj.get('date', {})
    if date:
        parts = []
        if date.get('day'):
            parts.append(str(date['day']))
        if date.get('month'):
            parts.append(str(date['month']))
        if date.get('year'):
            parts.append(str(date['year']))
        if parts:
            return '-'.join(parts)

    return None


def is_answer_correct(pred_answer: str, gold_answers: List[str]) -> bool:
    """
    判断预测答案是否正确
    使用 compute_accuracy 函数，支持双向包含判断
    """
    if not gold_answers:
        return False

    score = metric_max_over_ground_truths(
        compute_accuracy,
        pred_answer,
        gold_answers
    )
    return score >= 1


def extract_llm_text(response_json: Dict[str, Any]) -> str:
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


class AdaptiveLabeler:
    """Adaptive-RAG 标注器"""

    def __init__(self, config: dict):
        self.config = config
        self.llm_url = f"http://{config['llm']['host']}:{config['llm']['port']}/generate"
        self.retriever_url = f"http://{config['retriever']['host']}:{config['retriever']['port']}/retrieve/"
        self.corpus_mapping = config['corpus_mapping']
        self.llm_timeout = config['llm'].get('timeout', 60)
        self.retriever_timeout = config['retriever'].get('timeout', 30)

    def get_corpus_name(self, dataset_name: str) -> str:
        """获取数据集对应的语料库名称"""
        return self.corpus_mapping.get(dataset_name.lower(), 'wiki')

    def execute_zero_retrieval(self, question: str) -> str:
        """
        Zero Retrieval: 直接调用 LLM 回答问题，不使用检索

        Args:
            question: 问题文本

        Returns:
            LLM 生成的答案
        """
        prompt = f"""Answer the following question directly and concisely. Give only the answer, no explanation.

Question: {question}

Answer:"""

        try:
            response = requests.get(
                self.llm_url,
                params={
                    'prompt': prompt,
                    'max_length': self.config['llm'].get('max_length', 100),
                    'temperature': self.config['llm'].get('temperature', 0.1)
                },
                timeout=self.llm_timeout
            )
            response.raise_for_status()
            return extract_llm_text(response.json())

        except Exception as e:
            print(f"  [Zero] Error: {e}")
            return ""

    def execute_single_retrieval(self, question: str, dataset_name: str) -> str:
        """
        Single Retrieval: 使用 BM25 检索获取上下文，然后调用 LLM 回答

        Args:
            question: 问题文本
            dataset_name: 数据集名称

        Returns:
            LLM 生成的答案
        """
        corpus_name = self.get_corpus_name(dataset_name)

        # Step 1: BM25 检索
        try:
            retrieval_response = requests.post(
                self.retriever_url,
                json={
                    "retrieval_method": "retrieve_from_elasticsearch",
                    "query_text": question,
                    "rerank_query_text": question,
                    "max_hits_count": self.config['retriever'].get('max_hits', 8),
                    "max_buffer_count": self.config['retriever'].get('max_buffer', 40),
                    "corpus_name": corpus_name,
                    "document_type": "title_paragraph_text",
                    "retrieval_backend": "bm25"
                },
                timeout=self.retriever_timeout
            )
            retrieval_response.raise_for_status()
            hits = retrieval_response.json().get('retrieval', [])

        except Exception as e:
            print(f"  [Single] Retrieval error: {e}")
            return ""

        if not hits:
            return ""

        # Step 2: 构建上下文
        context_parts = []
        for i, hit in enumerate(hits[:5]):  # 使用前 5 个结果
            title = hit.get('title', '')
            text = hit.get('paragraph_text', '')
            context_parts.append(f"[{i+1}] {title}: {text[:500]}")

        context = "\n\n".join(context_parts)

        # Step 3: 调用 LLM
        prompt = f"""Based on the following context, answer the question concisely. Give only the answer, no explanation.

Context:
{context}

Question: {question}

Answer:"""

        try:
            response = requests.get(
                self.llm_url,
                params={
                    'prompt': prompt,
                    'max_length': self.config['llm'].get('max_length', 100),
                    'temperature': self.config['llm'].get('temperature', 0.1)
                },
                timeout=self.llm_timeout
            )
            response.raise_for_status()
            return extract_llm_text(response.json())

        except Exception as e:
            print(f"  [Single] LLM error: {e}")
            return ""

    def label_question(self, item: dict) -> dict:
        """
        对单个问题进行标注

        Args:
            item: 包含问题和答案的数据项

        Returns:
            包含标签和推理信息的结果字典
        """
        question_id = item.get('question_id', '')
        question_text = item.get('question_text', '')
        dataset_name = item.get('source_dataset', item.get('dataset', 'unknown'))

        gold_answers = extract_gold_answers(item)

        if not gold_answers:
            return {
                'question_id': question_id,
                'question_text': question_text,
                'dataset': dataset_name,
                'label': 'M',
                'reasoning': 'No gold answers available',
                'zero_answer': '',
                'single_answer': '',
                'gold_answers': []
            }

        result = {
            'question_id': question_id,
            'question_text': question_text,
            'dataset': dataset_name,
            'gold_answers': gold_answers,
            'zero_answer': '',
            'single_answer': '',
            'label': None,
            'reasoning': ''
        }

        # Step 1: Try Zero Retrieval
        zero_answer = self.execute_zero_retrieval(question_text)
        result['zero_answer'] = zero_answer

        if zero_answer and is_answer_correct(zero_answer, gold_answers):
            result['label'] = 'Z'
            result['reasoning'] = f'Zero retrieval correct: "{zero_answer}" matches gold'
            return result

        # Step 2: Try Single Retrieval
        single_answer = self.execute_single_retrieval(question_text, dataset_name)
        result['single_answer'] = single_answer

        if single_answer and is_answer_correct(single_answer, gold_answers):
            result['label'] = 'S'
            result['reasoning'] = f'Single retrieval correct: "{single_answer}" matches gold'
            return result

        # Step 3: Default to M (Multi-hop)
        result['label'] = 'M'
        result['reasoning'] = f'Both strategies failed. Zero: "{zero_answer}", Single: "{single_answer}"'
        return result


def process_item_wrapper(args):
    """并行处理的包装函数"""
    labeler, item = args
    try:
        return labeler.label_question(item)
    except Exception as e:
        return {
            'question_id': item.get('question_id', ''),
            'question_text': item.get('question_text', ''),
            'dataset': item.get('source_dataset', 'unknown'),
            'label': 'M',
            'reasoning': f'Error during labeling: {str(e)}',
            'zero_answer': '',
            'single_answer': '',
            'gold_answers': []
        }


def generate_training_data(labeled_data: List[dict]) -> List[dict]:
    """
    将标注数据转换为训练格式

    训练格式:
    {
        "question_text": "...",
        "reasoning": "...",
        "complexity_label": "L0/L1/L2",
        "index_strategy": "None/BM25",
        "action": "Z/S/M"
    }
    """
    training_data = []

    # 映射: label -> (complexity, index_strategy)
    label_mapping = {
        'Z': ('L0', 'None'),           # 简单问题，不需要检索
        'S': ('L1', 'BM25'),           # 中等问题，单次 BM25 检索
        'M': ('L2', 'BM25')            # 复杂问题，多跳检索
    }

    for item in labeled_data:
        label = item.get('label', 'M')
        complexity, index_strategy = label_mapping.get(label, ('L2', 'BM25'))

        # 生成推理文本
        reasoning = generate_reasoning_text(item)

        training_item = {
            'question_text': item['question_text'],
            'reasoning': reasoning,
            'complexity_label': complexity,
            'index_strategy': index_strategy,
            'action': label
        }
        training_data.append(training_item)

    return training_data


def generate_reasoning_text(item: dict) -> str:
    """
    生成推理文本，解释为什么选择这个标签

    Args:
        item: 标注后的数据项

    Returns:
        推理文本
    """
    label = item.get('label', 'M')
    question = item.get('question_text', '')

    if label == 'Z':
        return (
            f"This is a simple factual question that can be answered directly from "
            f"the model's knowledge without external retrieval. The question asks for "
            f"straightforward information that is commonly known."
        )
    elif label == 'S':
        return (
            f"This question requires specific information that needs retrieval from "
            f"external sources. A single BM25 retrieval is sufficient to find the "
            f"relevant context for answering this question."
        )
    else:  # M
        return (
            f"This is a complex question that likely requires multiple reasoning steps "
            f"or connecting information from multiple sources. Multi-hop retrieval is "
            f"needed to gather all necessary context for answering correctly."
        )


def load_checkpoint(checkpoint_path: str) -> dict:
    """加载检查点"""
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {'processed_ids': [], 'results': []}


def save_checkpoint(checkpoint_path: str, processed_ids: List[str], results: List[dict]):
    """保存检查点"""
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    with open(checkpoint_path, 'w', encoding='utf-8') as f:
        json.dump({
            'processed_ids': processed_ids,
            'results': results
        }, f, ensure_ascii=False)


def check_services(config: dict) -> bool:
    """检查服务是否可用"""
    print("\n检查服务状态...")

    # Check LLM service
    llm_url = f"http://{config['llm']['host']}:{config['llm']['port']}/generate"
    try:
        response = requests.get(
            llm_url,
            params={'prompt': 'test', 'max_length': 10},
            timeout=10
        )
        print(f"  LLM 服务 ({llm_url}): OK")
    except Exception as e:
        print(f"  LLM 服务 ({llm_url}): FAILED - {e}")
        return False

    # Check Retriever service
    retriever_url = f"http://{config['retriever']['host']}:{config['retriever']['port']}/retrieve/"
    try:
        response = requests.post(
            retriever_url,
            json={
                "retrieval_method": "retrieve_from_elasticsearch",
                "query_text": "test",
                "max_hits_count": 1,
                "corpus_name": "wiki",
                "document_type": "title_paragraph_text",
                "retrieval_backend": "hybrid"
            },
            timeout=10
        )
        print(f"  检索服务 ({retriever_url}): OK")
    except Exception as e:
        print(f"  检索服务 ({retriever_url}): FAILED - {e}")
        return False

    return True


def main():
    parser = argparse.ArgumentParser(description="Adaptive-RAG 结果导向标注")
    parser.add_argument('--config', type=str, default=None,
                       help='配置文件路径')
    parser.add_argument('--input', type=str, default=None,
                       help='输入文件路径 (采样后的问题)')
    parser.add_argument('--output', type=str, default=None,
                       help='输出文件路径 (训练数据)')
    parser.add_argument('--workers', type=int, default=1,
                       help='并行工作线程数 (默认: 1)')
    parser.add_argument('--resume', action='store_true',
                       help='从检查点恢复')
    parser.add_argument('--skip-service-check', action='store_true',
                       help='跳过服务检查')

    args = parser.parse_args()

    # 加载配置
    config = load_config(args.config)

    # 检查服务
    if not args.skip_service_check:
        if not check_services(config):
            print("\n服务不可用，请启动 LLM 和检索服务后重试")
            sys.exit(1)

    # 确定输入路径
    if args.input:
        input_path = args.input
    else:
        input_path = PROJECT_ROOT / config['data']['sampled_questions_path']

    # 确定输出路径
    if args.output:
        output_path = args.output
    else:
        output_path = PROJECT_ROOT / config['data']['training_data_path']

    # 检查点路径
    checkpoint_path = str(PROJECT_ROOT / config['labeling']['checkpoint_path'])

    # 加载输入数据
    if not os.path.exists(input_path):
        print(f"\n输入文件不存在: {input_path}")
        print("请先运行: python -m adaptive_rag.data.sample_questions")
        sys.exit(1)

    data = load_jsonl(str(input_path))
    print(f"\n加载 {len(data)} 条待标注数据")

    # 加载检查点
    if args.resume:
        checkpoint = load_checkpoint(checkpoint_path)
        processed_ids = set(checkpoint['processed_ids'])
        results = checkpoint['results']
        print(f"从检查点恢复: 已处理 {len(processed_ids)} 条")
    else:
        processed_ids = set()
        results = []

    # 过滤未处理的数据
    unprocessed_data = [
        item for item in data
        if item.get('question_id', '') not in processed_ids
    ]
    print(f"待处理: {len(unprocessed_data)} 条")

    if not unprocessed_data:
        print("所有数据已处理完成!")
    else:
        # 创建标注器
        labeler = AdaptiveLabeler(config)

        # 标注
        checkpoint_frequency = config['labeling'].get('checkpoint_frequency', 50)

        print(f"\n开始标注 (workers={args.workers})...")
        start_time = time.time()

        if args.workers <= 1:
            # 串行处理
            with tqdm(total=len(unprocessed_data), desc="Labeling") as pbar:
                for i, item in enumerate(unprocessed_data):
                    result = labeler.label_question(item)
                    results.append(result)
                    processed_ids.add(item.get('question_id', ''))
                    pbar.update(1)
                    pbar.set_postfix({
                        'label': result['label'],
                        'Z': sum(1 for r in results if r['label'] == 'Z'),
                        'S': sum(1 for r in results if r['label'] == 'S'),
                        'M': sum(1 for r in results if r['label'] == 'M')
                    })

                    # 保存检查点
                    if (i + 1) % checkpoint_frequency == 0:
                        save_checkpoint(checkpoint_path, list(processed_ids), results)
        else:
            # 并行处理
            with ThreadPoolExecutor(max_workers=args.workers) as executor:
                futures = {
                    executor.submit(process_item_wrapper, (labeler, item)): item
                    for item in unprocessed_data
                }

                with tqdm(total=len(unprocessed_data), desc="Labeling") as pbar:
                    for i, future in enumerate(as_completed(futures)):
                        result = future.result()
                        results.append(result)
                        processed_ids.add(result.get('question_id', ''))
                        pbar.update(1)
                        pbar.set_postfix({
                            'label': result['label'],
                            'Z': sum(1 for r in results if r['label'] == 'Z'),
                            'S': sum(1 for r in results if r['label'] == 'S'),
                            'M': sum(1 for r in results if r['label'] == 'M')
                        })

                        # 保存检查点
                        if (i + 1) % checkpoint_frequency == 0:
                            save_checkpoint(checkpoint_path, list(processed_ids), results)

        elapsed = time.time() - start_time
        print(f"\n标注完成! 耗时: {elapsed:.2f}s ({len(unprocessed_data)/elapsed:.2f} q/s)")

    # 统计标签分布
    print(f"\n{'='*60}")
    print("标签分布:")
    print(f"{'='*60}")

    label_counts = {'Z': 0, 'S': 0, 'M': 0}
    for r in results:
        label = r.get('label', 'M')
        label_counts[label] = label_counts.get(label, 0) + 1

    total = len(results)
    for label, count in label_counts.items():
        pct = count / total * 100 if total > 0 else 0
        print(f"  {label:12s}: {count:4d} ({pct:5.1f}%)")

    # 生成训练数据
    training_data = generate_training_data(results)

    # 保存训练数据
    save_jsonl(training_data, str(output_path))
    print(f"\n训练数据已保存: {output_path}")

    # 保存原始标注结果 (用于分析)
    raw_output_path = str(output_path).replace('.jsonl', '_raw.jsonl')
    save_jsonl(results, raw_output_path)
    print(f"原始标注结果: {raw_output_path}")

    # 清理检查点
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
        print(f"检查点已清理: {checkpoint_path}")


if __name__ == '__main__':
    main()
