"""
Oracle Evaluation Script
使用真实的支持段落作为上下文，测试模型性能上限
"""

import json
import os
import sys
import requests
from typing import List, Dict, Any
from tqdm import tqdm
from collections import defaultdict

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from metrics.squad_answer_em_f1 import SquadAnswerEmF1Metric


def load_test_data(dataset_name: str, data_dir: str = "processed_data") -> List[Dict]:
    """
    加载测试集数据

    Args:
        dataset_name: 数据集名称 (musique, 2wikimultihopqa)
        data_dir: 数据目录

    Returns:
        测试数据列表
    """
    test_file = os.path.join(data_dir, dataset_name, "test_subsampled.jsonl")

    if not os.path.exists(test_file):
        raise FileNotFoundError(f"Test file not found: {test_file}")

    data = []
    with open(test_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))

    print(f"✓ Loaded {len(data)} examples from {dataset_name}")
    return data


def extract_gold_contexts(item: Dict) -> str:
    """
    提取真实的支持段落作为上下文

    Args:
        item: 测试数据项

    Returns:
        拼接后的上下文字符串
    """
    contexts = item.get('contexts', [])

    # 提取所有支持段落
    supporting_contexts = [
        ctx for ctx in contexts
        if ctx.get('is_supporting', False)
    ]

    # 如果没有明确标记supporting的段落，使用所有段落（针对某些数据集）
    if not supporting_contexts:
        supporting_contexts = contexts

    # 拼接上下文
    context_parts = []
    for i, ctx in enumerate(supporting_contexts, 1):
        title = ctx.get('title', 'Unknown')
        text = ctx.get('paragraph_text', '')
        context_parts.append(f"[{i}] {title}: {text}")

    return "\n\n".join(context_parts)


def extract_gold_answers(item: Dict) -> List[str]:
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


def generate_answer_with_context(
    question: str,
    context: str,
    llm_url: str,
    max_length: int = 300,
    temperature: float = 0.1
) -> str:
    """
    使用真实上下文生成答案

    Args:
        question: 问题
        context: 真实上下文
        llm_url: LLM服务地址
        max_length: 最大生成长度
        temperature: 温度参数

    Returns:
        生成的答案
    """
    prompt = f"""Answer the question based on the provided context. Be concise and accurate.

*** CONTEXT ***
{context}

*** QUESTION ***
{question}

*** INSTRUCTIONS ***
1. Read the context carefully.
2. Extract the answer directly from the context.
3. Provide a concise answer (1-5 words if possible).

*** FORMAT ***
Answer: <your answer>
"""

    try:
        response = requests.get(
            llm_url,
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
        import re
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


def evaluate_oracle(
    dataset_name: str,
    llm_config: Dict[str, Any],
    data_dir: str = "processed_data",
    output_dir: str = "evaluate/outputs/oracle"
) -> Dict[str, Any]:
    """
    Oracle评估：使用真实上下文测试模型

    Args:
        dataset_name: 数据集名称
        llm_config: LLM配置
        data_dir: 数据目录
        output_dir: 输出目录

    Returns:
        评估结果
    """
    print(f"\n{'='*60}")
    print(f"Oracle Evaluation: {dataset_name}")
    print(f"{'='*60}")

    # 加载数据
    test_data = load_test_data(dataset_name, data_dir)

    # 初始化指标
    metric = SquadAnswerEmF1Metric()

    # LLM URL
    llm_url = f"http://{llm_config['host']}:{llm_config['port']}/generate"

    # 存储预测结果
    predictions = {}
    results_detail = []

    # 遍历测试数据
    for item in tqdm(test_data, desc=f"Evaluating {dataset_name}"):
        qid = item['question_id']
        question = item['question_text']

        # 提取真实上下文和答案
        gold_context = extract_gold_contexts(item)
        gold_answers = extract_gold_answers(item)

        if not gold_answers:
            print(f"  Warning: No gold answers for {qid}")
            continue

        # 使用真实上下文生成答案
        predicted_answer = generate_answer_with_context(
            question=question,
            context=gold_context,
            llm_url=llm_url
        )

        # 更新指标
        metric(predicted_answer, gold_answers)

        # 保存预测
        predictions[qid] = predicted_answer

        # 保存详细结果
        results_detail.append({
            'question_id': qid,
            'question': question,
            'predicted_answer': predicted_answer,
            'gold_answers': gold_answers,
            'num_supporting_contexts': len([c for c in item.get('contexts', []) if c.get('is_supporting', False)])
        })

    # 获取指标
    metrics = metric.get_metric(reset=False)

    # 打印结果
    print(f"\n{'='*60}")
    print(f"Results for {dataset_name}:")
    print(f"  EM:     {metrics['em']:.4f}")
    print(f"  F1:     {metrics['f1']:.4f}")
    print(f"  ACC:    {metrics['acc']:.4f}")
    print(f"  Recall: {metrics['recall']:.4f}")
    print(f"  Count:  {metrics['count']}")
    print(f"{'='*60}")

    # 保存结果
    os.makedirs(output_dir, exist_ok=True)

    # 保存指标
    metrics_file = os.path.join(output_dir, f"{dataset_name}_metrics.json")
    with open(metrics_file, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(f"✓ Metrics saved to {metrics_file}")

    # 保存预测
    predictions_file = os.path.join(output_dir, f"{dataset_name}_predictions.json")
    with open(predictions_file, 'w', encoding='utf-8') as f:
        json.dump(predictions, f, indent=2, ensure_ascii=False)
    print(f"✓ Predictions saved to {predictions_file}")

    # 保存详细结果
    details_file = os.path.join(output_dir, f"{dataset_name}_details.jsonl")
    with open(details_file, 'w', encoding='utf-8') as f:
        for result in results_detail:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')
    print(f"✓ Detailed results saved to {details_file}")

    return {
        'dataset': dataset_name,
        'metrics': metrics,
        'predictions': predictions
    }


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description="Oracle Evaluation with Gold Contexts")
    parser.add_argument('--datasets', nargs='+',
                       default=['musique', '2wikimultihopqa'],
                       help='Datasets to evaluate')
    parser.add_argument('--llm-host', default='localhost',
                       help='LLM service host')
    parser.add_argument('--llm-port', default='8000',
                       help='LLM service port')
    parser.add_argument('--data-dir', default='processed_data',
                       help='Data directory')
    parser.add_argument('--output-dir', default='evaluate/outputs/oracle',
                       help='Output directory')

    args = parser.parse_args()

    llm_config = {
        'host': args.llm_host,
        'port': args.llm_port
    }

    print("\n" + "="*60)
    print("ORACLE EVALUATION (Using Gold Contexts)")
    print("="*60)
    print(f"LLM Service: http://{llm_config['host']}:{llm_config['port']}")
    print(f"Datasets: {', '.join(args.datasets)}")
    print("="*60)

    all_results = []

    for dataset_name in args.datasets:
        try:
            result = evaluate_oracle(
                dataset_name=dataset_name,
                llm_config=llm_config,
                data_dir=args.data_dir,
                output_dir=args.output_dir
            )
            all_results.append(result)
        except Exception as e:
            print(f"\n✗ Error evaluating {dataset_name}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # 汇总结果
    if all_results:
        print("\n" + "="*60)
        print("OVERALL SUMMARY")
        print("="*60)

        total_em = 0
        total_f1 = 0
        total_acc = 0
        total_recall = 0
        total_count = 0

        for result in all_results:
            metrics = result['metrics']
            dataset = result['dataset']

            print(f"\n{dataset}:")
            print(f"  EM:  {metrics['em']:.4f}")
            print(f"  F1:  {metrics['f1']:.4f}")
            print(f"  ACC: {metrics['acc']:.4f}")

            total_em += metrics['em'] * metrics['count']
            total_f1 += metrics['f1'] * metrics['count']
            total_acc += metrics['acc'] * metrics['count']
            total_recall += metrics['recall'] * metrics['count']
            total_count += metrics['count']

        # 总体平均
        if total_count > 0:
            print(f"\n{'='*60}")
            print(f"OVERALL AVERAGE:")
            print(f"  EM:     {total_em / total_count:.4f}")
            print(f"  F1:     {total_f1 / total_count:.4f}")
            print(f"  ACC:    {total_acc / total_count:.4f}")
            print(f"  Recall: {total_recall / total_count:.4f}")
            print(f"  Total:  {total_count}")
            print(f"{'='*60}")

            # 保存总体结果
            summary_file = os.path.join(args.output_dir, "oracle_summary.json")
            summary = {
                'overall': {
                    'em': round(total_em / total_count, 4),
                    'f1': round(total_f1 / total_count, 4),
                    'acc': round(total_acc / total_count, 4),
                    'recall': round(total_recall / total_count, 4),
                    'count': total_count
                },
                'by_dataset': {
                    r['dataset']: r['metrics']
                    for r in all_results
                }
            }

            with open(summary_file, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
            print(f"\n✓ Summary saved to {summary_file}")

    print("\n" + "="*60)
    print("✓ ORACLE EVALUATION COMPLETE")
    print("="*60)


if __name__ == '__main__':
    main()
