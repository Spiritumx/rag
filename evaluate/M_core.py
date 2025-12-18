import requests
from typing import List, Dict


def execute_real_multihop(
    query: str,
    retriever_config: dict,
    llm_config: dict,
    dataset_name: str = 'musique'
) -> dict:
    """
    真正的主动多跳检索 (Agentic Retrieval)
    逻辑：LLM 规划 -> 检索 -> 补充上下文 -> LLM 再规划 -> ... -> 回答

    Args:
        query: 用户问题
        retriever_config: {'host': 'localhost', 'port': 8001}
        llm_config: {'host': 'localhost', 'port': 8000}
        dataset_name: 数据集名称（用于corpus_name）

    Returns:
        {
            'answer': str,           # 最终答案
            'chain': str,            # 推理链（用于chains.txt）
            'contexts': List[dict]   # 检索的文档（用于contexts.json）
        
    """
    # 动态维护的上下文（随着检索不断增长）
    context_text = ""
    # 用于去重的 ID 集合
    seen_ids = set()
    # 最终用于生成答案的所有文档
    final_docs = []
    # 推理链记录（每一步的思考和检索）
    reasoning_steps = []
    # 桥梁实体标题（用于黄金三角策略）
    bridge_titles = []

    # 最大跳数 (MuSiQue 一般 2 跳，最多 3 跳)
    MAX_HOPS = 4

    # 构造服务URL
    retriever_url = f"http://{retriever_config['host']}:{retriever_config['port']}/retrieve/"
    llm_url = f"http://{llm_config['host']}:{llm_config['port']}/generate"

    for step in range(MAX_HOPS):

        # --- Step A: 让 LLM 决定下一步搜什么 ---
        # 这是一个 "Thought" 过程
        # 添加Few-shot Examples提高生成质量
        prompt_content = f"""Task: Decompose a complex question into a search query for the missing information.

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
Question: {query}
Known: {context_text if context_text else "None."}

Instructions:
- If you have enough information to answer, output "DONE".
- Otherwise, output a specific, short search query (2-5 words).
- Do NOT output full sentences.

Next Search Query:"""

        # 调用LLM API生成查询
        try:
            response = requests.get(
                llm_url,
                params={
                    'prompt': prompt_content,
                    'max_length': 20,
                    'temperature': 0.1,
                    'do_sample': False
                },
                timeout=60
            )

            # 解析响应
            response_json = response.json()

            # 尝试不同的key（支持多种LLM服务器格式）
            if 'generated_texts' in response_json:
                # Llama autobatch server format: {"generated_texts": [text]}
                texts = response_json['generated_texts']
                next_query = texts[0].strip() if isinstance(texts, list) and len(texts) > 0 else ""
            elif 'text' in response_json:
                # Simple format: {"text": "..."}
                next_query = response_json['text'].strip()
            elif 'generated_text' in response_json:
                # Single text format: {"generated_text": "..."}
                next_query = response_json['generated_text'].strip()
            elif 'response' in response_json:
                # Response format: {"response": "..."}
                next_query = response_json['response'].strip()
            elif 'choices' in response_json:
                # OpenAI format: {"choices": [{"text": "..."}]}
                choices = response_json['choices']
                if isinstance(choices, list) and len(choices) > 0:
                    next_query = choices[0].get('text', '').strip()
                else:
                    next_query = ""
            else:
                reasoning_steps.append(f"[Hop {step+1}] Error: LLM returned unexpected format: {list(response_json.keys())}")
                break

        except Exception as e:
            import traceback
            traceback.print_exc()
            reasoning_steps.append(f"[Hop {step+1}] Error calling LLM: {e}")
            break

        # 记录这一步的思考
        reasoning_steps.append(f"[Hop {step+1}] Thought: Need to search for '{next_query}'")

        # --- Step B: 检查是否结束 ---
        if "DONE" in next_query or len(next_query) < 2:
            reasoning_steps.append(f"[Hop {step+1}] Decision: Enough information collected, proceeding to answer.")
            break

        # --- Step C: 执行混合检索 ---
        # 修正：中间步骤的Rerank只用next_query（当前搜索意图）
        # 避免Reranker因为"CEO"压低"wife"文档的分数
        try:
            # 调用Retriever API
            retrieval_params = {
                "retrieval_method": "retrieve_from_elasticsearch",
                "query_text": next_query,        # Initial retrieval
                "rerank_query_text": next_query,  #  中间步骤：Rerank也用当前query
                "max_hits_count": 10,             #  返回Top10（混合检索：BM25+HNSW+SPLADE=60候选→Rerank→Top10）
                "max_buffer_count": 60,           #  Buffer保留60个候选（每种检索20个）
                "corpus_name": "wiki",
                "document_type": "title_paragraph_text",
                "retrieval_backend": "hybrid"
            }

            response = requests.post(
                retriever_url,
                json=retrieval_params,
                timeout=30
            )

            hits = response.json()['retrieval']
        except Exception as e:
            import traceback
            traceback.print_exc()
            reasoning_steps.append(f"[Hop {step+1}] Error calling retriever: {e}")
            hits = []

        # 记录检索结果
        reasoning_steps.append(f"[Hop {step+1}] Retrieval: Found {len(hits)} documents for query '{next_query}'")

        # --- Step D: 更新上下文 ---
        #  M策略上下文管理：最多保存40个文档，使用滚动更新
        MAX_CONTEXT_DOCS = 40

        new_info = []
        for h in hits:
            # 简单的去重逻辑
            doc_id = h.get('id', h.get('title'))
            if doc_id not in seen_ids:
                seen_ids.add(doc_id)
                final_docs.append(h)
                # 收集桥梁实体标题
                if h.get('title'):
                    bridge_titles.append(h['title'])
                # 提取摘要加入 Context (防止 Context 爆掉，只加前200字)
                text = h.get('paragraph_text', h.get('text', ''))
                snippet = f"Title: {h.get('title', '')}\nText: {text[:200]}..."
                new_info.append(snippet)

        #  滚动更新：如果超过40个文档，保留最新的40个（后续Hop通常更接近答案）
        if len(final_docs) > MAX_CONTEXT_DOCS:
            removed_count = len(final_docs) - MAX_CONTEXT_DOCS
            # 移除最旧的文档（保留最新的40个）
            removed_docs = final_docs[:removed_count]
            final_docs = final_docs[removed_count:]
            reasoning_steps.append(f"[Hop {step+1}] Rolling update: Removed {removed_count} oldest documents (kept latest {MAX_CONTEXT_DOCS}).")

            # 更新seen_ids（移除被删除文档的ID，允许后续重新检索）
            for doc in removed_docs:
                doc_id = doc.get('id', doc.get('title'))
                if doc_id in seen_ids:
                    seen_ids.remove(doc_id)

        if not new_info:
            reasoning_steps.append(f"[Hop {step+1}] Warning: No new information found.")
            # 如果没搜到新东西，强制用原问题搜一次兜底，然后结束
            if step == 0:
                reasoning_steps.append(f"[Hop {step+1}] Fallback: Using original question for retrieval.")
                try:
                    response = requests.post(
                        retriever_url,
                        json={
                            "retrieval_method": "retrieve_from_elasticsearch",
                            "query_text": query,
                            "max_hits_count": 10,             # 返回Top10
                            "max_buffer_count": 60,           # 与正常检索保持一致
                            "corpus_name": "wiki",
                            "document_type": "title_paragraph_text",
                            "retrieval_backend": "hybrid"
                        },
                        timeout=30
                    )
                    fallback_hits = response.json()['retrieval']
                    final_docs.extend(fallback_hits)
                except Exception as e:
                    reasoning_steps.append(f"[Hop {step+1}] Fallback retrieval failed: {e}")
            break

        # 将新发现的信息追加到 Context，供下一轮 LLM 参考
        context_text += "\n\n" + "\n".join(new_info)
        reasoning_steps.append(f"[Hop {step+1}] Context: Added {len(new_info)} new documents to context.")

    # --- Step E: 最终生成 ---
    # 此时 final_docs 包含了多轮检索积累的所有文档
    # 修正：先去重，然后用全局expanded_query重新rerank
    unique_docs = []
    seen_doc_ids = set()
    for doc in final_docs:
        doc_id = doc.get('id', doc.get('title'))
        if doc_id not in seen_doc_ids:
            seen_doc_ids.add(doc_id)
            unique_docs.append(doc)

    reasoning_steps.append(f"[Final] Collected {len(unique_docs)} unique documents from all hops.")

    # 全局Rerank：使用原问题（而非扩展查询）
    # 原因：最终生成时，我们要找"最能回答原问题"的文档，不需要桥梁实体干扰
    if len(unique_docs) > 10:
        reasoning_steps.append(f"[Final] Performing global rerank with original question: '{query[:100]}...'")

        try:
            # 用原问题重新检索并rerank
            # 这样会把最能回答原问题的文档顶到前面
            response = requests.post(
                retriever_url,
                json={
                    "retrieval_method": "retrieve_from_elasticsearch",
                    "query_text": query,              #  只用原问题
                    "rerank_query_text": query,       #  Rerank也只用原问题
                    "max_hits_count": 20,             # 全局Rerank返回Top20（因为unique_docs可能有40个）
                    "max_buffer_count": 60,           #  保持一致
                    "corpus_name": "wiki",
                    "document_type": "title_paragraph_text",
                    "retrieval_backend": "hybrid"
                },
                timeout=30
            )
            reranked_hits = response.json()['retrieval']

            # 构建ID到文档的映射（使用原始unique_docs）
            doc_map = {}
            for doc in unique_docs:
                doc_id = doc.get('id', doc.get('title'))
                doc_map[doc_id] = doc

            # 按reranked_hits的顺序重新排列，优先使用reranked结果
            final_docs_reranked = []
            used_ids = set()

            # 先添加reranked结果中存在于unique_docs的文档
            for hit in reranked_hits:
                hit_id = hit.get('id', hit.get('title'))
                if hit_id in doc_map and hit_id not in used_ids:
                    final_docs_reranked.append(doc_map[hit_id])
                    used_ids.add(hit_id)

            # 再添加unique_docs中未被rerank覆盖的文档（保持原顺序）
            for doc in unique_docs:
                doc_id = doc.get('id', doc.get('title'))
                if doc_id not in used_ids:
                    final_docs_reranked.append(doc)
                    used_ids.add(doc_id)

            final_docs = final_docs_reranked[:10]
            reasoning_steps.append(f"[Final] After global rerank, using top {len(final_docs)} documents.")
        except Exception as e:
            reasoning_steps.append(f"[Final] Global rerank failed: {e}, using original order.")
            final_docs = unique_docs[:10]
    else:
        final_docs = unique_docs[:10]
        reasoning_steps.append(f"[Final] Using {len(final_docs)} documents (no rerank needed).")

    # 构造最终 Prompt 进行回答
    if not final_docs:
        reasoning_steps.append("[Final] No documents retrieved, returning 'I don't know'.")
        return {
            'answer': "I don't know",
            'chain': "\n".join(reasoning_steps),
            'contexts': []
        }

    # 构造文档上下文
    docs_text = "\n\n".join([
        f"Document {i+1}:\nTitle: {doc.get('title', 'N/A')}\nText: {doc.get('paragraph_text', doc.get('text', ''))[:500]}..."
        for i, doc in enumerate(final_docs)
    ])

    # 最终答题 Prompt
    final_prompt = f"""Answer the following question based on the provided documents.

Documents:
{docs_text}

Question: {query}

Answer the question concisely and directly. Output ONLY the answer, do NOT include phrases like 'The answer is' or 'Based on the documents'.

Answer:"""

    # 生成最终答案
    try:
        response = requests.get(
            llm_url,
            params={
                'prompt': final_prompt,
                'max_length': 100,
                'temperature': 0.1,
                'do_sample': False
            },
            timeout=120
        )

        # 解析LLM响应（支持多种格式）
        response_json = response.json()
        if 'generated_texts' in response_json:
            # Llama autobatch server format
            texts = response_json['generated_texts']
            answer = texts[0].strip() if isinstance(texts, list) and len(texts) > 0 else "I don't know"
        elif 'text' in response_json:
            answer = response_json['text'].strip()
        elif 'generated_text' in response_json:
            answer = response_json['generated_text'].strip()
        else:
            answer = "I don't know"

        reasoning_steps.append(f"[Final] Generated answer: {answer}")
    except Exception as e:
        import traceback
        traceback.print_exc()
        reasoning_steps.append(f"[Final] Error generating answer: {e}")
        answer = "I don't know"

    # 返回结果
    return {
        'answer': answer,
        'chain': "\n".join(reasoning_steps),
        'contexts': final_docs
    }