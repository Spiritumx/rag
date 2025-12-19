"""
Debug M Strategy Errors
分析M策略生成错误的案例，展示每一跳的提示词和模型回复
"""

import json
import os
import sys
import re
import requests
from typing import List, Dict, Any, Set

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from metrics.squad_answer_em_f1 import normalize_answer, compute_exact


def _extract_llm_text(response_json: Dict[str, Any]) -> str:
    """从LLM响应中提取文本"""
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


def _remove_conversational_phrases(query: str) -> str:
    """清洗对话式废话"""
    original_query = query
    query = query.strip().strip('"\'[]').lower()

    stop_phrases = [
        "i need to find", "i need to identify", "we need to", "try to find",
        "search for", "looking for", "find out", "figure out",
        "tell me", "what is", "who is", "where is", "when was",
        "the name of", "information about", "details regarding",
        "associated with", "related to", "step 1", "next step",
        "first step", "bridge entity"
    ]

    for phrase in stop_phrases:
        if phrase in query:
            query = query.replace(phrase, " ")

    query = " ".join(query.split())

    if len(query) < 1:
        return original_query.strip().lower()

    return query


def _is_semantically_similar(new_query: str, history_queries: Set[str], threshold: float = 0.9) -> bool:
    """语义去重"""
    new_tokens = set(new_query.lower().split())
    if not new_tokens:
        return False

    for old_q in history_queries:
        old_tokens = set(old_q.lower().split())
        if not old_tokens:
            continue

        intersection = new_tokens.intersection(old_tokens)
        union = new_tokens.union(old_tokens)
        similarity = len(intersection) / len(union)

        if similarity >= threshold:
            return True
    return False


def execute_multihop_with_debug(
    query: str,
    retriever_config: dict,
    llm_config: dict,
    dataset_name: str
) -> dict:
    """
    执行多跳推理（调试版本）- 记录每一跳的详细信息

    Returns:
        {
            'answer': 最终答案,
            'hops': [
                {
                    'hop_num': 跳数,
                    'prompt': 发送的完整提示词,
                    'llm_response': 模型原始回复,
                    'thought': 解析的思考,
                    'action': 解析的动作,
                    'query': 搜索查询,
                    'hits': 检索到的文档数
                },
                ...
            ],
            'final_generation': {
                'prompt': 最终生成的提示词,
                'llm_response': 模型原始回复,
                'answer': 解析的答案
            }
        }
    """
    MAX_HOPS = 6
    MAX_RETRIES = 3
    MAX_CONTEXT_DOCS = 40

    dataset_to_corpus = {
        'hotpotqa': 'hotpotqa',
        'musique': 'musique',
        '2wikimultihopqa': '2wikimultihopqa',
        'iirc': 'iirc',
        'wiki': 'wiki'
    }
    corpus_name = dataset_to_corpus.get(dataset_name.lower(), 'wiki')

    retriever_url = f"http://{retriever_config['host']}:{retriever_config['port']}/retrieve/"
    llm_url = f"http://{llm_config['host']}:{llm_config['port']}/generate"

    all_retrieved_docs: Dict[str, Dict] = {}
    history_log: List[str] = []
    executed_queries: Set[str] = set()
    reasoning_chain: List[str] = []  # 与M_core.py一致的推理链记录

    debug_hops = []  # 记录每一跳的详细信息（用于调试日志）

    print(f"\n{'='*80}")
    print(f"Question: {query}")
    print(f"{'='*80}\n")

    # ==============================================================================
    # Step 0: 初始检索
    # ==============================================================================
    print("[Step 0] Initial Retrieval...")
    try:
        r_resp = requests.post(
            retriever_url,
            json={
                "retrieval_method": "retrieve_from_elasticsearch",
                "query_text": query,
                "rerank_query_text": query,
                "max_hits_count": 8,
                "max_buffer_count": 40,
                "corpus_name": corpus_name,
                "document_type": "title_paragraph_text",
                "retrieval_backend": "hybrid"
            }, timeout=30
        )
        hits = r_resp.json().get('retrieval', [])

        init_snippets = []
        for h in hits:
            key = f"{h.get('title')} {h.get('paragraph_text')[:20]}"
            if key not in all_retrieved_docs:
                all_retrieved_docs[key] = h
                init_snippets.append(f"Title: {h['title']}\nContent: {h['paragraph_text'][:200]}...")

        if init_snippets:
            current_context_snippet = "\n---\n".join(init_snippets[:3])
            titles = [h['title'] for h in hits[:3]]
            history_log.append(f"Initial Search '{query}' -> Found docs: {', '.join(titles)}")
            executed_queries.add(query.lower())
            executed_queries.add(_remove_conversational_phrases(query.lower()))  # 🔧 修复：与M_core.py一致
            print(f"  ✓ Found {len(hits)} documents")
            for i, title in enumerate(titles[:3], 1):
                print(f"    {i}. {title}")
        else:
            current_context_snippet = "Initial search returned no direct results."
            history_log.append(f"Initial Search '{query}' -> No direct results.")
            print(f"  ✗ No results")

        reasoning_chain.append(f"[Step 0] Initial Search Found {len(hits)} hits.")  # 🔧 修复：添加reasoning_chain记录

    except Exception as e:
        print(f"  ✗ Initial search failed: {e}")
        current_context_snippet = "Initial search failed."

    # ==============================================================================
    # Stage 1: 推理循环
    # ==============================================================================
    should_stop_reasoning = False

    for step in range(MAX_HOPS):
        if should_stop_reasoning:
            break

        print(f"\n{'-'*80}")
        print(f"[Hop {step+1}] Starting reasoning...")
        print(f"{'-'*80}")

        valid_next_query = None
        feedback_msg = ""
        thought = ""
        action_raw = ""
        llm_response_raw = ""

        for attempt in range(MAX_RETRIES):
            history_str = "\n".join([f"Hop {i+1}: {h}" for i, h in enumerate(history_log)]) if history_log else "None"

            base_prompt = f"""You are an expert research agent. Break down the question into KEYWORD SEARCHES.

*** RULES ***
1. Check "Current Info" first. If it helps, move to the next logical entity.
2. DO NOT repeat "Past Actions".
3. OUTPUT ONLY KEYWORDS (Entities, Names, Places).

*** EXAMPLE ***
Question: "Who is the director of the film Titanic?"
Hop 1:
Thought: Initial search found Titanic movie info. I need the director's name specifically.
Action: Search [Titanic film director]
Result: Found James Cameron.
Hop 2:
Thought: Now I need to find James Cameron's birth country.
Action: Search [James Cameron birth place]
Result: Found Kapuskasing, Ontario, Canada.
Hop 3:
Thought: I have the answer.
Action: Answer

*** YOUR TASK ***
Question: {query}

Past Actions:
{history_str}

Current Info:
{current_context_snippet}

Format:
Thought: <reasoning>
Action: <Search [Keywords] OR Answer>"""

            if feedback_msg:
                final_prompt = base_prompt + f"\n\n*** ERROR ***\n{feedback_msg}"
            else:
                final_prompt = base_prompt

            print(f"\n>>> Attempt {attempt+1}/{MAX_RETRIES}")
            print(f">>> Prompt Length: {len(final_prompt)} characters")

            try:
                resp = requests.get(
                    llm_url,
                    params={'prompt': final_prompt, 'max_length': 128, 'temperature': 0.1},
                    timeout=30
                )
                llm_output = _extract_llm_text(resp.json())
                llm_response_raw = llm_output

                print(f">>> LLM Response:")
                print(f"    {llm_output[:200]}...")

            except Exception as e:
                print(f"✗ LLM Error: {e}")
                should_stop_reasoning = True
                break

            # 解析
            thought_match = re.search(r'Thought:(.*?)(Action:|$)', llm_output, re.DOTALL | re.IGNORECASE)
            thought = thought_match.group(1).strip() if thought_match else ""

            action_match = re.search(r'Action:\s*(.*)', llm_output, re.IGNORECASE)
            if not action_match:
                fallback = re.search(r'Search\s*\[(.*?)\]', llm_output, re.IGNORECASE)
                if fallback:
                    action_raw = f"Search [{fallback.group(1)}]"
                else:
                    feedback_msg = "Format Error: Output 'Action: Search [Keywords]'."
                    print(f"✗ {feedback_msg}")
                    continue
            else:
                action_raw = action_match.group(1).strip().split('\n')[0]

            # Case A: Answer
            if "Answer" in action_raw and "Search" not in action_raw:
                print(f">>> Thought: {thought}")
                print(f">>> Action: Answer")

                # 🔧 修复：添加reasoning_chain记录（与M_core.py一致）
                reasoning_chain.append(f"[Hop {step+1}] Thought: {thought}")
                reasoning_chain.append(f"[Hop {step+1}] Action: Answer")

                debug_hops.append({
                    'hop_num': step + 1,
                    'prompt': final_prompt,
                    'llm_response': llm_response_raw,
                    'thought': thought,
                    'action': 'Answer',
                    'query': None,
                    'hits': 0
                })

                should_stop_reasoning = True
                valid_next_query = None
                break

            # Case B: Search
            search_match = re.search(r'Search\s*\[?(.*?)\]?$', action_raw, re.IGNORECASE)
            if search_match:
                raw_q = search_match.group(1)
            elif action_raw.lower().startswith("search"):
                raw_q = action_raw[6:].strip()
            else:
                feedback_msg = "Format Error."
                print(f"✗ {feedback_msg}")
                continue

            clean_q = _remove_conversational_phrases(raw_q)

            if len(clean_q) < 2:
                feedback_msg = "Query too short."
                print(f"✗ {feedback_msg}")
                continue

            # 语义去重
            if _is_semantically_similar(clean_q, executed_queries):
                print(f"✗ Loop Detected: '{clean_q}' similar to history.")
                feedback_msg = f"Loop Detected: You already searched '{clean_q}'. Try a different angle."
                continue

            print(f">>> Thought: {thought}")
            print(f">>> Action: Search [{clean_q}]")

            valid_next_query = clean_q
            break

        if should_stop_reasoning:
            break

        if not valid_next_query:
            print(f"✗ Failed to generate valid query after {MAX_RETRIES} attempts")

            # 🔧 修复：添加reasoning_chain记录（与M_core.py一致）
            reasoning_chain.append(f"[Hop {step+1}] Failed to generate valid query. Stopping.")

            debug_hops.append({
                'hop_num': step + 1,
                'prompt': final_prompt if 'final_prompt' in locals() else '',
                'llm_response': llm_response_raw,
                'thought': thought,
                'action': 'Failed',
                'query': None,
                'hits': 0
            })
            break

        # 🔧 修复：添加reasoning_chain记录（与M_core.py一致）
        reasoning_chain.append(f"[Hop {step+1}] Thought: {thought}")
        reasoning_chain.append(f"[Hop {step+1}] Action: Search [{valid_next_query}]")
        executed_queries.add(valid_next_query.lower())

        # 执行检索
        print(f"\n>>> Retrieving documents for: '{valid_next_query}'")
        try:
            r_resp = requests.post(
                retriever_url,
                json={
                    "retrieval_method": "retrieve_from_elasticsearch",
                    "query_text": valid_next_query,
                    "rerank_query_text": valid_next_query,
                    "max_hits_count": 5,
                    "max_buffer_count": 20,
                    "corpus_name": corpus_name,
                    "document_type": "title_paragraph_text",
                    "retrieval_backend": "hybrid"
                }, timeout=120
            )
            hits = r_resp.json().get('retrieval', [])
            print(f"  ✓ Found {len(hits)} documents")

        except Exception as e:
            print(f"  ✗ Retrieval Error: {e}")
            reasoning_chain.append(f"[Error] Retrieve: {e}")  # 🔧 修复：添加错误记录
            hits = []

        new_snippets = []
        for h in hits:
            key = f"{h.get('title')} {h.get('paragraph_text')[:20]}"
            if key not in all_retrieved_docs:
                all_retrieved_docs[key] = h
                new_snippets.append(f"Title: {h['title']}\nContent: {h['paragraph_text'][:200]}...")

        # 🔧 修复：添加reasoning_chain记录（与M_core.py一致）
        reasoning_chain.append(f"[Hop {step+1}] Found {len(hits)} hits.")

        if new_snippets:
            titles = [h['title'] for h in hits[:2]]
            history_log.append(f"Searched '{valid_next_query}' -> Found: {', '.join(titles)}")
            current_context_snippet = "\n---\n".join(new_snippets[:3])
            print(f"  Retrieved documents:")
            for i, title in enumerate(titles, 1):
                print(f"    {i}. {title}")
        else:
            history_log.append(f"Searched '{valid_next_query}' -> No results.")
            current_context_snippet = "No relevant documents found. Try different keywords."
            print(f"  ✗ No new documents")

        # 记录本跳信息
        debug_hops.append({
            'hop_num': step + 1,
            'prompt': final_prompt,
            'llm_response': llm_response_raw,
            'thought': thought,
            'action': f'Search [{valid_next_query}]',
            'query': valid_next_query,
            'hits': len(hits)
        })

    # ==============================================================================
    # Stage 2: 生成答案
    # ==============================================================================
    print(f"\n{'='*80}")
    print(f"[Final Generation] Generating answer...")
    print(f"{'='*80}\n")

    final_docs = list(all_retrieved_docs.values())

    if not final_docs:
        return {
            'answer': "I don't know",
            'hops': debug_hops,
            'final_generation': {
                'prompt': '',
                'llm_response': '',
                'answer': "I don't know"
            }
        }

    if len(final_docs) > MAX_CONTEXT_DOCS:
        selected_docs = final_docs[:15] + final_docs[-15:]
    else:
        selected_docs = final_docs

    context_str = "\n\n".join([f"[{i+1}] {d['title']}: {d['paragraph_text']}" for i, d in enumerate(selected_docs)])

    # 🔧 修复：使用与M_core.py一致的investigation_log构建方式
    clean_history = [line for line in reasoning_chain
                     if "Thought:" in line or "Action:" in line or "Found" in line]
    investigation_log = "\n".join(clean_history)

    final_prompt = f"""Task: Answer the complex question based on the Investigation Log and Retrieved Documents.

*** INVESTIGATION LOG ***
{investigation_log}

*** RETRIEVED DOCUMENTS ***
{context_str}

*** INSTRUCTIONS ***
1. Synthesize the information found in the Log and Documents.
2. Think step-by-step.
3. Provide a concise answer.

*** QUESTION ***
{query}

*** FORMAT ***
Thought: <analysis>
Answer: <concise answer>
"""

    print(f">>> Final Prompt Length: {len(final_prompt)} characters")
    print(f">>> Using {len(selected_docs)} documents")

    try:
        resp = requests.get(
            llm_url,
            params={'prompt': final_prompt, 'max_length': 300, 'temperature': 0.1},
            timeout=40
        )
        llm_output = _extract_llm_text(resp.json())

        print(f">>> LLM Response:")
        print(f"    {llm_output[:200]}...")

        # 🔧 修复：添加reasoning_chain记录（与M_core.py一致）
        reasoning_chain.append(f"[Final Generation] Raw: {llm_output[:100]}...")

        ans_match = re.search(r'Answer:\s*(.*)', llm_output, re.DOTALL | re.IGNORECASE)
        if ans_match:
            final_answer = ans_match.group(1).strip().split('\n')[0].strip()
        else:
            lines = [l for l in llm_output.split('\n') if l.strip()]
            final_answer = lines[-1] if lines else "I don't know"

        print(f"\n>>> Final Answer: {final_answer}")

    except Exception as e:
        print(f"✗ Final Generation Error: {e}")
        reasoning_chain.append(f"[Final Error] {e}")  # 🔧 修复：添加错误记录
        llm_output = f"Error: {e}"
        final_answer = "Error"

    return {
        'answer': final_answer,
        'hops': debug_hops,
        'reasoning_chain': reasoning_chain,  # 🔧 修复：添加reasoning_chain用于验证与M_core.py一致性
        'final_generation': {
            'prompt': final_prompt,
            'llm_response': llm_output,
            'answer': final_answer
        }
    }


def find_error_cases(dataset_name: str, num_cases: int = 2) -> List[Dict]:
    """
    找到M策略预测错误的案例

    Args:
        dataset_name: 数据集名称
        num_cases: 要找的错误案例数量

    Returns:
        错误案例列表
    """
    # 加载预测结果
    predictions_file = f"evaluate/outputs/stage2/{dataset_name}_predictions.json"
    if not os.path.exists(predictions_file):
        print(f"✗ Predictions file not found: {predictions_file}")
        return []

    with open(predictions_file, 'r', encoding='utf-8') as f:
        predictions = json.load(f)

    # 加载分类结果
    classifications_file = f"evaluate/outputs/stage1/{dataset_name}_classifications.json"
    if not os.path.exists(classifications_file):
        print(f"✗ Classifications file not found: {classifications_file}")
        return []

    with open(classifications_file, 'r', encoding='utf-8') as f:
        classifications = json.load(f)

    # 加载测试数据
    test_file = f"processed_data/{dataset_name}/test_subsampled.jsonl"
    if not os.path.exists(test_file):
        print(f"✗ Test file not found: {test_file}")
        return []

    test_data = []
    with open(test_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                test_data.append(json.loads(line))

    test_data_map = {item['question_id']: item for item in test_data}

    # 找到M策略的错误案例
    error_cases = []

    for qid, pred_answer in predictions.items():
        # 检查是否是M策略
        if qid not in classifications:
            continue

        if classifications[qid]['predicted_action'] != 'M':
            continue

        # 检查是否预测错误
        if qid not in test_data_map:
            continue

        item = test_data_map[qid]

        # 提取真实答案
        gold_answers = []
        if 'answers_objects' in item:
            for ans_obj in item['answers_objects']:
                if ans_obj.get('spans'):
                    spans = ans_obj['spans']
                    if isinstance(spans, list) and len(spans) > 0:
                        gold_answers.append(str(spans[0]))
                    elif isinstance(spans, str):
                        gold_answers.append(spans)

        if not gold_answers:
            continue

        # 检查是否EM错误
        em = max([compute_exact(pred_answer, gold) for gold in gold_answers])

        if em == 0:  # 预测错误
            error_cases.append({
                'question_id': qid,
                'question': item['question_text'],
                'predicted_answer': pred_answer,
                'gold_answers': gold_answers
            })

        if len(error_cases) >= num_cases:
            break

    return error_cases


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description="Debug M Strategy Errors")
    parser.add_argument('--dataset', default='musique',
                       help='Dataset name')
    parser.add_argument('--num-cases', type=int, default=2,
                       help='Number of error cases to analyze')
    parser.add_argument('--llm-host', default='localhost',
                       help='LLM service host')
    parser.add_argument('--llm-port', default='8000',
                       help='LLM service port')
    parser.add_argument('--retriever-host', default='localhost',
                       help='Retriever service host')
    parser.add_argument('--retriever-port', default='8001',
                       help='Retriever service port')
    parser.add_argument('--output-dir', default='evaluate/outputs/debug',
                       help='Output directory for debug logs')

    args = parser.parse_args()

    print("\n" + "="*80)
    print("M STRATEGY ERROR ANALYSIS")
    print("="*80)
    print(f"Dataset: {args.dataset}")
    print(f"Looking for {args.num_cases} error cases...")
    print("="*80)

    # 查找错误案例
    error_cases = find_error_cases(args.dataset, args.num_cases)

    if not error_cases:
        print("\n✗ No error cases found!")
        return

    print(f"\n✓ Found {len(error_cases)} error cases")

    # 配置
    retriever_config = {
        'host': args.retriever_host,
        'port': args.retriever_port
    }
    llm_config = {
        'host': args.llm_host,
        'port': args.llm_port
    }

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    # 分析每个错误案例
    for i, case in enumerate(error_cases, 1):
        print(f"\n{'#'*80}")
        print(f"ERROR CASE {i}/{len(error_cases)}")
        print(f"{'#'*80}")
        print(f"Question ID: {case['question_id']}")
        print(f"Question: {case['question']}")
        print(f"Predicted: {case['predicted_answer']}")
        print(f"Gold: {case['gold_answers']}")

        # 重新执行带调试信息
        result = execute_multihop_with_debug(
            query=case['question'],
            retriever_config=retriever_config,
            llm_config=llm_config,
            dataset_name=args.dataset
        )

        # 保存详细日志
        log_file = os.path.join(args.output_dir, f"error_case_{i}_{case['question_id']}.txt")

        with open(log_file, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write(f"ERROR CASE {i}\n")
            f.write("="*80 + "\n\n")

            f.write(f"Question ID: {case['question_id']}\n")
            f.write(f"Question: {case['question']}\n")
            f.write(f"Predicted Answer: {case['predicted_answer']}\n")
            f.write(f"Gold Answers: {', '.join(case['gold_answers'])}\n")
            f.write(f"Re-run Answer: {result['answer']}\n\n")

            f.write("="*80 + "\n")
            f.write("REASONING HOPS\n")
            f.write("="*80 + "\n\n")

            for hop in result['hops']:
                f.write(f"\n{'='*80}\n")
                f.write(f"HOP {hop['hop_num']}\n")
                f.write(f"{'='*80}\n\n")

                f.write(">>> PROMPT SENT TO LLM:\n")
                f.write("-"*80 + "\n")
                f.write(hop['prompt'] + "\n")
                f.write("-"*80 + "\n\n")

                f.write(">>> LLM RESPONSE:\n")
                f.write("-"*80 + "\n")
                f.write(hop['llm_response'] + "\n")
                f.write("-"*80 + "\n\n")

                f.write(f">>> PARSED:\n")
                f.write(f"Thought: {hop['thought']}\n")
                f.write(f"Action: {hop['action']}\n")
                if hop['query']:
                    f.write(f"Query: {hop['query']}\n")
                    f.write(f"Hits: {hop['hits']}\n")
                f.write("\n")

            f.write("="*80 + "\n")
            f.write("FINAL GENERATION\n")
            f.write("="*80 + "\n\n")

            f.write(">>> PROMPT SENT TO LLM:\n")
            f.write("-"*80 + "\n")
            f.write(result['final_generation']['prompt'] + "\n")
            f.write("-"*80 + "\n\n")

            f.write(">>> LLM RESPONSE:\n")
            f.write("-"*80 + "\n")
            f.write(result['final_generation']['llm_response'] + "\n")
            f.write("-"*80 + "\n\n")

            f.write(f">>> FINAL ANSWER:\n")
            f.write(f"{result['final_generation']['answer']}\n")

            # 🔧 添加：输出完整的reasoning_chain用于验证
            f.write("\n" + "="*80 + "\n")
            f.write("REASONING CHAIN (for verification with M_core.py)\n")
            f.write("="*80 + "\n\n")
            f.write("\n".join(result['reasoning_chain']))
            f.write("\n")

        print(f"\n✓ Detailed log saved to: {log_file}")

    print("\n" + "="*80)
    print("✓ ANALYSIS COMPLETE")
    print("="*80)
    print(f"Analyzed {len(error_cases)} error cases")
    print(f"Logs saved to: {args.output_dir}/")


if __name__ == '__main__':
    main()
