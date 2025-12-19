import requests
import re
import traceback
from typing import List, Dict, Any, Set

def _extract_llm_text(response_json: Dict[str, Any]) -> str:
    """辅助函数：适配多种 LLM API 的响应格式"""
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

def _clean_query(query: str) -> str:
    """清理查询语句，移除多余的标点和指令词"""
    # 移除开头可能的 Search for / Search / Query
    query = re.sub(r'^(search for|search|find|query|look for):?\s*', '', query, flags=re.IGNORECASE)
    # 移除开头结尾的引号和方括号
    query = query.strip().strip('"\'[]').strip()
    return query

def execute_real_multihop(
    query: str,
    retriever_config: dict,
    llm_config: dict,
    dataset_name: str
) -> dict:
    """
    高级 Agentic RAG (Strict Parsing 修订版)
    
    修复:
    1. 解决了 LLM 输出 "Let's analyze..." 废话被当做查询发给检索器的问题。
    2. 引入严格的 Regex 解析，如果格式不对，触发重试。
    """
    
    # === 核心配置参数 ===
    MAX_HOPS = 8
    MAX_RETRIES = 3
    MAX_CONTEXT_DOCS = 40
    
    # 确定 Corpus
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

    # === 状态维护 ===
    all_retrieved_docs: Dict[str, Dict] = {} 
    history_log: List[str] = []
    executed_queries: Set[str] = set()
    reasoning_chain: List[str] = []

    print(f"  [Agent] Start: {query[:60]}... (Corpus: {corpus_name}, Max Hops: {MAX_HOPS})")

    current_context_snippet = "No information gathered yet."
    should_stop_reasoning = False

    # ==============================================================================
    # Stage 1: 多跳推理循环
    # ==============================================================================
    for step in range(MAX_HOPS):
        if should_stop_reasoning:
            break

        feedback_msg = ""
        valid_next_query = None
        action_raw = ""
        thought = ""
        
        # --- 内部循环：规划与自我修正 ---
        for attempt in range(MAX_RETRIES):
            history_str = "\n".join([f"Step {i+1}: {h}" for i, h in enumerate(history_log)]) if history_log else "None"
            
            base_prompt = f"""Task: You are a smart research assistant solving a complex question. 
You have up to {MAX_HOPS} steps.

Question: {query}

Past Actions:
{history_str}

Current Knowledge:
{current_context_snippet}

Instructions:
1. Analyze "Past Actions". DO NOT repeat the same search.
2. If you can answer now, output "Action: Answer".
3. To search, output "Action: Search [Keywords]".
4. OUTPUT ONLY THE FORMAT BELOW. NO CONVERSATIONAL TEXT.

Format:
Thought: <reasoning>
Action: <Search [Query] OR Answer>"""

            if feedback_msg:
                final_prompt = base_prompt + f"\n\n*** FORMAT/LOGIC ERROR ***\n{feedback_msg}\nPlease fix your output format."
            else:
                final_prompt = base_prompt

            try:
                resp = requests.get(
                    llm_url,
                    params={'prompt': final_prompt, 'max_length': 160, 'temperature': 0.2}, 
                    timeout=30
                )
                llm_output = _extract_llm_text(resp.json())
            except Exception as e:
                reasoning_chain.append(f"[Error] LLM Call Failed: {e}")
                should_stop_reasoning = True
                break

            # === 严格解析逻辑 (Critical Fix) ===
            
            # 1. 提取 Thought (允许稍微宽容)
            thought_match = re.search(r'Thought:(.*?)(Action:|$)', llm_output, re.DOTALL | re.IGNORECASE)
            thought = thought_match.group(1).strip() if thought_match else "No thought provided"

            # 2. 提取 Action (必须严格)
            action_match = re.search(r'Action:\s*(.*)', llm_output, re.IGNORECASE) # 不用 DOTALL，只取一行
            
            if action_match:
                # 提取 Action 后面的内容，并只取第一行，去掉多余的换行和废话
                action_raw = action_match.group(1).strip().split('\n')[0]
            else:
                # 没找到 Action: 标签
                # 尝试最后的救命稻草：在全文找 "Search [...]" 结构
                fallback_search = re.search(r'Search\s*\[(.*?)\]', llm_output, re.IGNORECASE)
                if fallback_search:
                    action_raw = f"Search [{fallback_search.group(1)}]"
                else:
                    # 格式彻底错误：既没有 Action: 也没有 Search []
                    # 绝对不能把 llm_output 当作 query！
                    print(f"    [Format Warning] Invalid output: {llm_output[:40]}...")
                    feedback_msg = "Format Error: You MUST output 'Action: Search [Query]' or 'Action: Answer'. Do not output conversational fillers like 'Let's analyze...'."
                    continue # 触发重试

            # Case A: 决定回答
            if "Answer" in action_raw and "Search" not in action_raw:
                reasoning_chain.append(f"[Step {step+1}] Thought: {thought}")
                reasoning_chain.append(f"[Step {step+1}] Action: Answer")
                should_stop_reasoning = True
                valid_next_query = None
                break

            # Case B: 决定搜索
            # 兼容 "Search [X]" 和 "Search X"
            search_match = re.search(r'Search\s*\[?(.*?)\]?$', action_raw, re.IGNORECASE)
            if search_match:
                temp_query = search_match.group(1)
            else:
                # 只有当 action_raw 本身以 Search 开头时才这样处理
                if action_raw.lower().startswith("search"):
                    temp_query = action_raw[6:].strip()
                else:
                    # 这通常意味着解析到了奇怪的东西，触发重试
                    feedback_msg = "Format Error: Your Action must start with 'Search' or 'Answer'."
                    continue

            temp_query = _clean_query(temp_query)

            # 验证 Query 有效性
            if len(temp_query) < 2 or "analyze" in temp_query.lower() or "step" in temp_query.lower():
                # 这里额外拦截一下 "Let's analyze" 这种漏网之鱼
                feedback_msg = f"Invalid Query: '{temp_query}' looks like conversational text, not a search entity. Please output a specific entity."
                continue
            
            # 检测循环
            if temp_query.lower() in executed_queries:
                print(f"    [Correction] Loop detected: '{temp_query}'. Retry {attempt+1}/{MAX_RETRIES}")
                feedback_msg = f"ERROR: You have ALREADY searched for '{temp_query}'. Search for something NEW."
            else:
                valid_next_query = temp_query
                break 

        if not should_stop_reasoning and not valid_next_query:
            reasoning_chain.append(f"[Step {step+1}] Failed to generate valid action. Stopping.")
            break

        if should_stop_reasoning:
            break

        # --- 执行搜索 ---
        reasoning_chain.append(f"[Step {step+1}] Thought: {thought}")
        reasoning_chain.append(f"[Step {step+1}] Action: Search [{valid_next_query}]")
        
        executed_queries.add(valid_next_query.lower())
        
        try:
            r_resp = requests.post(
                retriever_url,
                json={
                    "retrieval_method": "retrieve_from_elasticsearch",
                    "query_text": valid_next_query,
                    "rerank_query_text": valid_next_query,
                    "max_hits_count": 8,
                    "max_buffer_count": 40,
                    "corpus_name": corpus_name,
                    "document_type": "title_paragraph_text",
                    "retrieval_backend": "hybrid"
                }, timeout=120
            )
            hits = r_resp.json().get('retrieval', [])
        except Exception as e:
            reasoning_chain.append(f"[Error] Retriever Failed: {e}")
            hits = []

        new_info_snippets = []
        for h in hits:
            doc_key = f"{h.get('title')} {h.get('paragraph_text')[:30]}"
            if doc_key not in all_retrieved_docs:
                all_retrieved_docs[doc_key] = h
                snippet = f"{h.get('title')}: {h.get('paragraph_text')[:150]}..."
                new_info_snippets.append(snippet)

        reasoning_chain.append(f"[Step {step+1}] Found {len(hits)} hits, {len(new_info_snippets)} new.")

        if new_info_snippets:
            top_titles = [h['title'] for h in hits[:2]]
            history_log.append(f"Searched '{valid_next_query}' -> Found: {', '.join(top_titles)}")
            current_context_snippet = "\n".join(new_info_snippets)
        else:
            history_log.append(f"Searched '{valid_next_query}' -> Found NOTHING. Try a different entity or keyword.")
            current_context_snippet = "Last search returned no relevant results."

    # ==============================================================================
    # Stage 2: 最终答案生成
    # ==============================================================================
    
    final_docs = list(all_retrieved_docs.values())
    if not final_docs:
        return {
            'answer': "I don't know",
            'chain': "\n".join(reasoning_chain),
            'contexts': []
        }

    # Head + Mid + Tail Context Selection
    if len(final_docs) > MAX_CONTEXT_DOCS:
        num_tail = 15
        num_head = 10
        num_mid = MAX_CONTEXT_DOCS - num_tail - num_head
        
        mid_docs = final_docs[num_head:-num_tail]
        if len(mid_docs) > num_mid:
            step_size = max(1, len(mid_docs) // num_mid)
            selected_mid = [mid_docs[i] for i in range(0, len(mid_docs), step_size)][:num_mid]
        else:
            selected_mid = mid_docs
            
        selected_docs = final_docs[:num_head] + selected_mid + final_docs[-num_tail:]
    else:
        selected_docs = final_docs

    context_text = "\n\n".join([
        f"Document [{i+1}] {d.get('title', 'Unknown')}:\n{d.get('paragraph_text', '')}" 
        for i, d in enumerate(selected_docs)
    ])

    final_prompt = f"""Use the following documents to answer the user's question. 
If the documents do not contain the answer, say "I don't know".
Keep the answer concise.

Documents:
{context_text}

Question: {query}
Answer:"""

    try:
        resp = requests.get(
            llm_url,
            params={'prompt': final_prompt, 'max_length': 64, 'temperature': 0.1},
            timeout=30
        )
        final_answer = _extract_llm_text(resp.json())
        final_answer = final_answer.split('\n')[0].strip()
    except:
        final_answer = "Error generating answer"

    return {
        'answer': final_answer,
        'chain': "\n".join(reasoning_chain),
        'contexts': selected_docs
    }