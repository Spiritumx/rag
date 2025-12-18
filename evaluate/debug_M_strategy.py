"""
调试M策略：输出完整的多跳检索流程

随机选择2-3个M策略问题，详细输出:
1. 问题本身
2. 每次检索的查询语句
3. 每次检索到的文档
4. Reranker使用的查询
5. 发送给LLM的完整prompt
6. LLM返回的完整响应
7. 最终答案和黄金答案对比
"""

import os
import sys
import json
import random
import requests
from datetime import datetime
from typing import List, Dict

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluate.utils.config_loader import ConfigLoader
from evaluate.utils.data_loader import DataLoader
from evaluate.utils.result_manager import ResultManager


class MStrategyDebugger:
    """详细调试M策略的执行流程"""

    def __init__(self, config):
        self.config = config
        self.data_loader = DataLoader(config)
        self.result_manager = ResultManager(config)

        # 服务配置
        self.retriever_url = f"http://{config['retriever']['host']}:{config['retriever']['port']}/retrieve/"
        self.llm_url = f"http://{config['llm']['server_host']}:{config['llm']['server_port']}/generate"

        # 输出目录
        self.output_dir = "evaluate/debug_outputs"
        os.makedirs(self.output_dir, exist_ok=True)

    def select_random_M_questions(self, dataset_name: str, num_samples: int = 3) -> List[Dict]:
        """随机选择M策略的问题"""
        # 加载分类结果
        classifications = self.result_manager.load_stage1_results(dataset_name)

        # 筛选M策略问题
        m_questions = [
            qid for qid, cls in classifications.items()
            if cls['predicted_action'] == 'M'
        ]

        print(f"Total M strategy questions: {len(m_questions)}")

        # 随机选择
        selected_qids = random.sample(m_questions, min(num_samples, len(m_questions)))

        # 加载完整问题数据
        test_data = self.data_loader.load_test_data(dataset_name)
        test_data_map = {item['question_id']: item for item in test_data}

        selected_questions = [
            test_data_map[qid] for qid in selected_qids if qid in test_data_map
        ]

        return selected_questions

    def call_llm(self, prompt: str, max_length: int = 50) -> Dict:
        """调用LLM并返回完整响应信息"""
        try:
            response = requests.get(
                self.llm_url,
                params={
                    'prompt': prompt,
                    'max_length': max_length,
                    'temperature': 0.1,
                    'do_sample': False
                },
                timeout=60
            )

            response_json = response.json()

            # 解析生成的文本
            if 'generated_texts' in response_json:
                texts = response_json['generated_texts']
                generated_text = texts[0].strip() if isinstance(texts, list) and len(texts) > 0 else ""
            elif 'text' in response_json:
                generated_text = response_json['text'].strip()
            elif 'generated_text' in response_json:
                generated_text = response_json['generated_text'].strip()
            else:
                generated_text = ""

            return {
                'generated_text': generated_text,
                'full_response': response_json,
                'status_code': response.status_code
            }
        except Exception as e:
            return {
                'generated_text': "",
                'error': str(e),
                'status_code': None
            }

    def call_retriever(self, query_text: str, rerank_query_text: str = None) -> Dict:
        """调用检索器并返回完整响应信息"""
        if rerank_query_text is None:
            rerank_query_text = query_text

        try:
            params = {
                "retrieval_method": "retrieve_from_elasticsearch",
                "query_text": query_text,
                "rerank_query_text": rerank_query_text,
                "max_hits_count": 10,
                "max_buffer_count": 60,
                "corpus_name": "wiki",
                "document_type": "title_paragraph_text",
                "retrieval_backend": "hybrid"
            }

            response = requests.post(
                self.retriever_url,
                json=params,
                timeout=30
            )

            response_json = response.json()
            hits = response_json.get('retrieval', [])

            return {
                'hits': hits,
                'query_text': query_text,
                'rerank_query_text': rerank_query_text,
                'num_hits': len(hits),
                'status_code': response.status_code,
                'params': params
            }
        except Exception as e:
            return {
                'hits': [],
                'error': str(e),
                'query_text': query_text,
                'rerank_query_text': rerank_query_text,
                'status_code': None
            }

    def execute_multihop_debug(self, question: str, question_id: str, golden_answers: List[str]) -> Dict:
        """执行多跳检索并记录所有中间过程"""
        print(f"\n{'='*80}")
        print(f"Processing Question: {question_id}")
        print(f"{'='*80}")

        debug_log = {
            'question_id': question_id,
            'question': question,
            'golden_answers': golden_answers,
            'hops': [],
            'final_generation': {},
            'timestamp': datetime.now().isoformat()
        }

        # 上下文管理
        context_text = ""
        seen_ids = set()
        final_docs = []
        MAX_HOPS = 4
        MAX_CONTEXT_DOCS = 40

        # 多跳检索
        for step in range(MAX_HOPS):
            print(f"\n--- Hop {step+1}/{MAX_HOPS} ---")

            hop_log = {
                'hop_number': step + 1,
                'planning': {},
                'retrieval': {},
                'context_update': {}
            }

            # ========== Step A: LLM规划 ==========
            planning_prompt = f"""Task: Decompose a complex question into a search query for the missing information.

Example 1:
Question: Who is the CEO of the company that acquired WhatsApp?
Known: WhatsApp was acquired by Facebook in 2014.
Next Search Query: CEO of Facebook

Example 2:
Question: Where was the author of "Harry Potter" born?
Known: "Harry Potter" was written by J.K. Rowling.
Next Search Query: birthplace of J.K. Rowling

Example 3:
Question: When was the employer of Neville A. Stanton founded?
Known: Neville A. Stanton works at University of Southampton.
Next Search Query: University of Southampton founded

Current Task:
Question: {question}
Known: {context_text if context_text else "None."}

Instructions:
- If you have enough information to answer, output "DONE".
- Otherwise, output a specific, short search query (2-5 words).
- Do NOT output full sentences.

Next Search Query:"""

            print(f"[Planning] Calling LLM...")
            llm_response = self.call_llm(planning_prompt, max_length=20)
            next_query = llm_response['generated_text']

            hop_log['planning'] = {
                'prompt': planning_prompt,
                'llm_response': llm_response,
                'next_query': next_query
            }

            print(f"[Planning] Next query: '{next_query}'")

            # 检查退出条件
            if "DONE" in next_query or len(next_query) < 2:
                print(f"[Planning] Decision: Stop searching (query={next_query})")
                hop_log['planning']['decision'] = 'stop'
                debug_log['hops'].append(hop_log)
                break

            hop_log['planning']['decision'] = 'continue'

            # ========== Step B: 执行检索 ==========
            print(f"[Retrieval] Calling retriever with query='{next_query}'...")
            retrieval_response = self.call_retriever(
                query_text=next_query,
                rerank_query_text=next_query  # 中间步骤使用相同query
            )

            hits = retrieval_response['hits']
            hop_log['retrieval'] = {
                'query_text': next_query,
                'rerank_query_text': next_query,
                'params': retrieval_response['params'],
                'num_hits': len(hits),
                'hits': [
                    {
                        'title': h.get('title', ''),
                        'text': h.get('paragraph_text', h.get('text', ''))[:300] + "...",
                        'id': h.get('id', ''),
                        'score': h.get('score', 0)
                    }
                    for h in hits[:10]
                ]
            }

            print(f"[Retrieval] Retrieved {len(hits)} documents")
            if hits:
                print(f"[Retrieval] Top-3 titles: {[h.get('title', '') for h in hits[:3]]}")

            # ========== Step C: 更新上下文 ==========
            new_info = []
            for h in hits:
                doc_id = h.get('id', h.get('title'))
                if doc_id not in seen_ids:
                    seen_ids.add(doc_id)
                    final_docs.append(h)
                    text = h.get('paragraph_text', h.get('text', ''))
                    snippet = f"Title: {h.get('title', '')}\nText: {text[:200]}..."
                    new_info.append(snippet)

            # 滚动更新
            if len(final_docs) > MAX_CONTEXT_DOCS:
                removed_count = len(final_docs) - MAX_CONTEXT_DOCS
                removed_docs = final_docs[:removed_count]
                final_docs = final_docs[removed_count:]

                for doc in removed_docs:
                    doc_id = doc.get('id', doc.get('title'))
                    if doc_id in seen_ids:
                        seen_ids.remove(doc_id)

                print(f"[Context] Rolling update: Removed {removed_count} oldest docs, kept {MAX_CONTEXT_DOCS}")

            hop_log['context_update'] = {
                'new_docs_added': len(new_info),
                'total_docs_accumulated': len(final_docs),
                'rolling_update_triggered': len(final_docs) == MAX_CONTEXT_DOCS
            }

            if not new_info:
                print(f"[Context] Warning: No new documents found")
                if step == 0:
                    print(f"[Context] Fallback: Using original question")
                    fallback_response = self.call_retriever(question, question)
                    final_docs.extend(fallback_response['hits'])
                    hop_log['context_update']['fallback_used'] = True
                break

            context_text += "\n\n" + "\n".join(new_info)
            print(f"[Context] Added {len(new_info)} new docs, total: {len(final_docs)}")

            debug_log['hops'].append(hop_log)

        # ========== Step D: 全局Rerank ==========
        print(f"\n--- Final Global Rerank ---")

        # 去重
        unique_docs = []
        seen_doc_ids = set()
        for doc in final_docs:
            doc_id = doc.get('id', doc.get('title'))
            if doc_id not in seen_doc_ids:
                seen_doc_ids.add(doc_id)
                unique_docs.append(doc)

        print(f"[Rerank] Unique documents: {len(unique_docs)}")

        # 全局Rerank（使用原问题）
        if len(unique_docs) > 10:
            print(f"[Rerank] Performing global rerank with original question")
            rerank_response = self.call_retriever(
                query_text=question,
                rerank_query_text=question
            )

            reranked_hits = rerank_response['hits']

            # 构建ID映射
            doc_map = {doc.get('id', doc.get('title')): doc for doc in unique_docs}

            # 重新排序
            final_docs_reranked = []
            used_ids = set()

            for hit in reranked_hits:
                hit_id = hit.get('id', hit.get('title'))
                if hit_id in doc_map and hit_id not in used_ids:
                    final_docs_reranked.append(doc_map[hit_id])
                    used_ids.add(hit_id)

            for doc in unique_docs:
                doc_id = doc.get('id', doc.get('title'))
                if doc_id not in used_ids:
                    final_docs_reranked.append(doc)
                    used_ids.add(doc_id)

            final_docs = final_docs_reranked[:10]

            debug_log['global_rerank'] = {
                'performed': True,
                'query': question,
                'input_docs': len(unique_docs),
                'output_docs': len(final_docs),
                'rerank_response': {
                    'num_hits': len(reranked_hits),
                    'top_3_titles': [h.get('title', '') for h in reranked_hits[:3]]
                }
            }
        else:
            final_docs = unique_docs[:10]
            debug_log['global_rerank'] = {
                'performed': False,
                'reason': 'less_than_10_docs',
                'output_docs': len(final_docs)
            }

        print(f"[Rerank] Final documents: {len(final_docs)}")
        if final_docs:
            print(f"[Rerank] Top-3 titles: {[d.get('title', '') for d in final_docs[:3]]}")

        # ========== Step E: 最终生成答案 ==========
        print(f"\n--- Final Answer Generation ---")

        if not final_docs:
            debug_log['final_generation'] = {
                'prompt': None,
                'llm_response': None,
                'answer': "I don't know",
                'golden_answers': golden_answers,
                'match': False
            }
            print(f"[Generation] No documents, returning 'I don't know'")
        else:
            # 构造文档上下文
            docs_text = "\n\n".join([
                f"Document {i+1}:\nTitle: {doc.get('title', 'N/A')}\nText: {doc.get('paragraph_text', doc.get('text', ''))[:500]}..."
                for i, doc in enumerate(final_docs)
            ])

            final_prompt = f"""Answer the following question based on the provided documents.

Documents:
{docs_text}

Question: {question}

Answer the question concisely and directly. Output ONLY the answer, do NOT include phrases like 'The answer is' or 'Based on the documents'.

Answer:"""

            print(f"[Generation] Calling LLM for final answer...")
            llm_response = self.call_llm(final_prompt, max_length=100)
            answer = llm_response['generated_text'] or "I don't know"

            # 检查答案是否匹配
            answer_lower = answer.lower()
            match = any(golden.lower() in answer_lower for golden in golden_answers)

            debug_log['final_generation'] = {
                'prompt': final_prompt,
                'llm_response': llm_response,
                'answer': answer,
                'golden_answers': golden_answers,
                'match': match,
                'used_documents': [
                    {
                        'title': doc.get('title', ''),
                        'text': doc.get('paragraph_text', doc.get('text', ''))[:300] + "..."
                    }
                    for doc in final_docs
                ]
            }

            print(f"[Generation] Answer: {answer}")
            print(f"[Generation] Golden: {golden_answers}")
            print(f"[Generation] Match: {match}")

        return debug_log

    def save_debug_log(self, debug_log: Dict, filename: str):
        """保存调试日志为JSON和易读的Markdown"""
        # 保存JSON
        json_path = os.path.join(self.output_dir, f"{filename}.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(debug_log, f, indent=2, ensure_ascii=False)

        print(f"Saved JSON to: {json_path}")

        # 保存Markdown
        md_path = os.path.join(self.output_dir, f"{filename}.md")
        with open(md_path, 'w', encoding='utf-8') as f:
            self._write_markdown(f, debug_log)

        print(f"Saved Markdown to: {md_path}")

    def _write_markdown(self, f, log: Dict):
        """将调试日志写成Markdown格式"""
        f.write(f"# M Strategy Debug Log: {log['question_id']}\n\n")
        f.write(f"**Timestamp:** {log['timestamp']}\n\n")

        f.write(f"## Question\n\n")
        f.write(f"{log['question']}\n\n")

        f.write(f"## Golden Answers\n\n")
        for ans in log['golden_answers']:
            f.write(f"- {ans}\n")
        f.write("\n")

        # 每一跳的详细信息
        for hop in log['hops']:
            f.write(f"## Hop {hop['hop_number']}\n\n")

            # 规划阶段
            f.write(f"### Planning\n\n")
            planning = hop['planning']
            f.write(f"**Decision:** {planning['decision']}\n\n")
            f.write(f"**Next Query:** `{planning['next_query']}`\n\n")

            f.write(f"<details>\n<summary>Full Planning Prompt</summary>\n\n")
            f.write(f"```\n{planning['prompt']}\n```\n")
            f.write(f"</details>\n\n")

            f.write(f"<details>\n<summary>LLM Response</summary>\n\n")
            f.write(f"```json\n{json.dumps(planning['llm_response'], indent=2, ensure_ascii=False)}\n```\n")
            f.write(f"</details>\n\n")

            if planning['decision'] == 'stop':
                continue

            # 检索阶段
            f.write(f"### Retrieval\n\n")
            retrieval = hop['retrieval']
            f.write(f"**Query Text:** `{retrieval['query_text']}`\n\n")
            f.write(f"**Rerank Query Text:** `{retrieval['rerank_query_text']}`\n\n")
            f.write(f"**Retrieved:** {retrieval['num_hits']} documents\n\n")

            if retrieval['hits']:
                f.write(f"**Top-5 Documents:**\n\n")
                for i, hit in enumerate(retrieval['hits'][:5], 1):
                    f.write(f"{i}. **{hit['title']}** (score: {hit.get('score', 'N/A')})\n")
                    f.write(f"   ```\n   {hit['text'][:200]}...\n   ```\n\n")

            f.write(f"<details>\n<summary>Retrieval Parameters</summary>\n\n")
            f.write(f"```json\n{json.dumps(retrieval['params'], indent=2, ensure_ascii=False)}\n```\n")
            f.write(f"</details>\n\n")

            # 上下文更新
            f.write(f"### Context Update\n\n")
            ctx = hop['context_update']
            f.write(f"- New documents added: {ctx['new_docs_added']}\n")
            f.write(f"- Total documents accumulated: {ctx['total_docs_accumulated']}\n")
            f.write(f"- Rolling update triggered: {ctx['rolling_update_triggered']}\n")
            if ctx.get('fallback_used'):
                f.write(f"- **Fallback used** (no new docs found)\n")
            f.write("\n")

        # 全局Rerank
        if 'global_rerank' in log:
            f.write(f"## Global Rerank\n\n")
            rerank = log['global_rerank']
            f.write(f"**Performed:** {rerank['performed']}\n\n")
            if rerank['performed']:
                f.write(f"**Query:** `{rerank['query']}`\n\n")
                f.write(f"**Input docs:** {rerank['input_docs']} → **Output docs:** {rerank['output_docs']}\n\n")
                f.write(f"**Top-3 after rerank:** {', '.join(rerank['rerank_response']['top_3_titles'])}\n\n")
            else:
                f.write(f"**Reason:** {rerank.get('reason', 'N/A')}\n\n")

        # 最终生成
        f.write(f"## Final Answer Generation\n\n")
        gen = log['final_generation']

        if gen['prompt']:
            f.write(f"### Used Documents\n\n")
            for i, doc in enumerate(gen.get('used_documents', []), 1):
                f.write(f"{i}. **{doc['title']}**\n")
                f.write(f"   ```\n   {doc['text'][:200]}...\n   ```\n\n")

            f.write(f"<details>\n<summary>Full Answer Generation Prompt</summary>\n\n")
            f.write(f"```\n{gen['prompt']}\n```\n")
            f.write(f"</details>\n\n")

            f.write(f"<details>\n<summary>LLM Response</summary>\n\n")
            f.write(f"```json\n{json.dumps(gen['llm_response'], indent=2, ensure_ascii=False)}\n```\n")
            f.write(f"</details>\n\n")

        f.write(f"### Result\n\n")
        f.write(f"**Generated Answer:** {gen['answer']}\n\n")
        f.write(f"**Golden Answers:** {', '.join(gen['golden_answers'])}\n\n")
        f.write(f"**Match:** {'✅ YES' if gen['match'] else '❌ NO'}\n\n")

    def run(self, dataset_name: str = 'musique', num_samples: int = 3):
        """运行调试流程"""
        print(f"{'='*80}")
        print(f"M Strategy Debugger")
        print(f"{'='*80}")
        print(f"Dataset: {dataset_name}")
        print(f"Samples to debug: {num_samples}")
        print(f"Output directory: {self.output_dir}")
        print(f"{'='*80}\n")

        # 选择问题
        questions = self.select_random_M_questions(dataset_name, num_samples)

        if not questions:
            print("No M strategy questions found!")
            return

        print(f"Selected {len(questions)} questions:")
        for q in questions:
            print(f"  - {q['question_id']}: {q['question_text'][:60]}...")
        print()

        # 逐个调试
        all_logs = []
        for i, q in enumerate(questions, 1):
            print(f"\n{'='*80}")
            print(f"Question {i}/{len(questions)}")
            print(f"{'='*80}")

            # 提取黄金答案
            golden_answers = []
            answers_objects = q.get('answers_objects', [])
            if answers_objects:
                # MuSiQue格式：answers_objects[0]['spans']
                for ans_obj in answers_objects:
                    if 'spans' in ans_obj:
                        golden_answers.extend(ans_obj['spans'])
                    elif 'text' in ans_obj:
                        golden_answers.append(ans_obj['text'])

            # 如果没有找到答案，尝试使用answer字段（某些数据集）
            if not golden_answers and 'answer' in q:
                golden_answers = [q['answer']]

            debug_log = self.execute_multihop_debug(
                question=q['question_text'],
                question_id=q['question_id'],
                golden_answers=golden_answers
            )

            # 保存单个问题的日志
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"debug_{dataset_name}_{q['question_id']}_{timestamp}"
            self.save_debug_log(debug_log, filename)

            all_logs.append(debug_log)

        # 保存汇总
        summary_filename = f"debug_{dataset_name}_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        summary_path = os.path.join(self.output_dir, f"{summary_filename}.json")
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(all_logs, f, indent=2, ensure_ascii=False)

        print(f"\n{'='*80}")
        print(f"Debug Complete!")
        print(f"{'='*80}")
        print(f"Total questions debugged: {len(all_logs)}")
        print(f"Output directory: {self.output_dir}")
        print(f"Summary saved to: {summary_path}")

        # 统计匹配率
        matches = sum(1 for log in all_logs if log['final_generation'].get('match', False))
        print(f"Matched answers: {matches}/{len(all_logs)}")
        print(f"{'='*80}\n")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Debug M Strategy")
    parser.add_argument('--config', default='evaluate/config.yaml',
                       help='Path to config file')
    parser.add_argument('--dataset', default='musique',
                       help='Dataset to debug')
    parser.add_argument('--num-samples', type=int, default=3,
                       help='Number of samples to debug')

    args = parser.parse_args()

    # 加载配置
    config = ConfigLoader.load_config(args.config)

    # 运行调试
    debugger = MStrategyDebugger(config)
    debugger.run(dataset_name=args.dataset, num_samples=args.num_samples)


if __name__ == '__main__':
    main()
