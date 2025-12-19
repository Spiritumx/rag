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
    query = re.sub(r'^(search for|search|find|query):?\s*', '', query, flags=re.IGNORECASE)
    return query.strip().strip('"').strip("'")

def execute_real_multihop(
    query: str,
    retriever_config: dict,
    llm_config: dict,
    dataset_name: str
) -> dict:
    """
    高级 Agentic RAG (Self-Correction + Deep Reasoning 版)
    
    特性:
    1. MAX_HOPS = 8: 允许更深层的推理。
    2. Self-Correction: 遇到重复查询时，在单步内进行 Prompt 修正重试，而不是直接失败。
    3. Optimized Context: 最终生成时采用 Head+Mid+Tail 采样策略，防止长窗口下的上下文丢失。
    """
    
    # === 核心配置参数 ===
    MAX_HOPS = 8           # 增加推理深度，允许更多次跳转
    MAX_RETRIES = 3        # 单步内遇到死循环时的最大修正次数
    MAX_CONTEXT_DOCS = 40  # 最终生成答案时提供给 LLM 的最大文档数
    
    # 确定 Corpus (Elasticsearch Index)
    dataset_to_corpus = {
        'hotpotqa': 'hotpotqa',
        'musique': 'musique',
        '2wikimultihopqa': '2wikimultihopqa',
        'iirc': 'iirc',
        'wiki': 'wiki'
    }
    corpus_name = dataset_to_corpus.get(dataset_name.lower(), 'wiki')
    
    # API 地址（处理可能已包含协议前缀的情况）
    retriever_host = retriever_config['host']
    if not retriever_host.startswith('http'):
        retriever_host = f"http://{retriever_host}"
    retriever_url = f"{retriever_host}:{retriever_config['port']}/retrieve/"

    llm_host = llm_config['host']
    if not llm_host.startswith('http'):
        llm_host = f"http://{llm_host}"
    llm_url = f"{llm_host}:{llm_config['port']}/generate"

    # === 状态维护 ===
    # 存储所有检索到的文档，Key为 title+text 摘要，Value为完整文档对象
    all_retrieved_docs: Dict[str, Dict] = {} 
    # 历史动作日志，用于 Prompt
    history_log: List[str] = []
    # 已执行过的查询集合 (归一化后)，用于检测死循环
    executed_queries: Set[str] = set()
    # 推理链日志，用于最终输出
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

        # --- 内部循环：规划与自我修正 (Planning with Self-Correction) ---
        feedback_msg = ""     # 用于存储给 LLM 的纠错反馈
        valid_next_query = None
        action_raw = ""
        thought = ""
        
        for attempt in range(MAX_RETRIES):
            # 构造历史记录字符串
            history_str = "\n".join([f"Step {i+1}: {h}" for i, h in enumerate(history_log)]) if history_log else "None"
            
            # 基础 Prompt
            base_prompt = f"""Task: You are a smart research assistant solving a complex question. 
You have up to {MAX_HOPS} steps to gather information.

Question: {query}

Past Actions & Results:
{history_str}

Current Knowledge Summary:
{current_context_snippet}

Instructions:
1. Analyze the "Past Actions". DO NOT repeat the same search query.
2. If you have enough information to answer the User Question, output "Action: Answer".
3. If you need more information, output "Action: Search [Keywords]".
4. Focus on finding bridge entities (e.g., if asked for "director of the film starring X", first search "film starring X", then "director of [Film Name]").

Format:
Thought: <your reasoning>
Action: <Search [Query] OR Answer>"""

            # 如果有反馈信息（说明上一次生成重复了），将其追加到 Prompt 末尾，加强语气
            if feedback_msg:
                final_prompt = base_prompt + f"\n\n*** PREVIOUS ATTEMPT FAILED ***\n{feedback_msg}\nPlease try a DIFFERENT angle or entity."
            else:
                final_prompt = base_prompt

            try:
                # 稍微调高 temperature (0.2)，在重试时增加一点随机性，有助于打破死循环
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

            # 解析 LLM 输出
            thought_match = re.search(r'Thought:(.*?)(Action:|$)', llm_output, re.DOTALL | re.IGNORECASE)
            action_match = re.search(r'Action:(.*)', llm_output, re.DOTALL | re.IGNORECASE)

            thought = thought_match.group(1).strip() if thought_match else "No thought"
            action_raw = action_match.group(1).strip() if action_match else llm_output.strip()

            # Case A: 决定回答
            if "Answer" in action_raw and "Search" not in action_raw:
                reasoning_chain.append(f"[Step {step+1}] Thought: {thought}")
                reasoning_chain.append(f"[Step {step+1}] Agent decided to Answer.")
                should_stop_reasoning = True
                valid_next_query = None
                break

            # Case B: 决定搜索 -> 提取 Query
            search_match = re.search(r'Search\s*\[?(.*?)\]?$', action_raw, re.IGNORECASE)
            if search_match:
                temp_query = search_match.group(1)
            else:
                temp_query = action_raw.replace("Search", "").strip()

            temp_query = _clean_query(temp_query)

            # 验证 Query 有效性
            if not temp_query or len(temp_query) < 2:
                feedback_msg = "Your previous action generated an empty query."
                continue
            
            # 检测循环 (Loop Detection)
            if temp_query.lower() in executed_queries:
                print(f"    [Correction] Loop detected: '{temp_query}'. Asking LLM to rewrite... (Attempt {attempt+1}/{MAX_RETRIES})")
                feedback_msg = f"ERROR: You just tried to search for '{temp_query}', but you have ALREADY searched for this in the past steps. You MUST generate a DIFFERENT query."
                # 继续下一次 attempt 循环，不 break
            else:
                # 成功生成唯一 Query
                valid_next_query = temp_query
                break 

        # 如果退出重试循环后，依然没有有效查询且没决定停止，说明重试多次失败
        if not should_stop_reasoning and not valid_next_query:
            reasoning_chain.append(f"[Step {step+1}] Failed to generate unique query after {MAX_RETRIES} attempts. Stopping.")
            break

        if should_stop_reasoning:
            break

        # --- 执行搜索 (Execution) ---
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
                    "max_hits_count": 8,  # 单步检索 8 条，防止文档库膨胀过快
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

        # 处理并去重文档
        new_info_snippets = []
        for h in hits:
            doc_key = f"{h.get('title')} {h.get('paragraph_text')[:30]}"
            if doc_key not in all_retrieved_docs:
                all_retrieved_docs[doc_key] = h
                snippet = f"{h.get('title')}: {h.get('paragraph_text')[:150]}..."
                new_info_snippets.append(snippet)

        reasoning_chain.append(f"[Step {step+1}] Found {len(hits)} hits, {len(new_info_snippets)} new.")

        # 更新 Prompt 上下文
        if new_info_snippets:
            # 记录简单的历史摘要
            top_titles = [h['title'] for h in hits[:2]]
            history_log.append(f"Searched '{valid_next_query}' -> Found info on: {', '.join(top_titles)}")
            # 更新 Current Knowledge Summary (给 LLM 看最近的发现)
            current_context_snippet = "\n".join(new_info_snippets)
        else:
            history_log.append(f"Searched '{valid_next_query}' -> Found NOTHING. Try a different entity or keyword.")
            current_context_snippet = "Last search returned no relevant results."

    # ==============================================================================
    # Stage 2: 最终答案生成 (Final Answer Generation)
    # ==============================================================================
    
    final_docs = list(all_retrieved_docs.values())
    if not final_docs:
        return {
            'answer': "I don't know",
            'chain': "\n".join(reasoning_chain),
            'contexts': []
        }

    # --- 优化文档选择策略 (Head + Mid + Tail) ---
    # 目的：保留最早的背景信息(Head) 和 最新的直接答案信息(Tail)，并从中间采样防止断链
    if len(final_docs) > MAX_CONTEXT_DOCS:
        num_tail = 15   # 最近的 15 个（通常包含最终答案）
        num_head = 10   # 最早的 10 个（通常包含主语定义）
        num_mid = MAX_CONTEXT_DOCS - num_tail - num_head # 中间采样数 (15)
        
        mid_docs = final_docs[num_head:-num_tail]
        
        # 均匀采样中间部分
        if len(mid_docs) > num_mid:
            step_size = max(1, len(mid_docs) // num_mid)
            selected_mid = [mid_docs[i] for i in range(0, len(mid_docs), step_size)][:num_mid]
        else:
            selected_mid = mid_docs
            
        selected_docs = final_docs[:num_head] + selected_mid + final_docs[-num_tail:]
    else:
        selected_docs = final_docs

    # 构造最终 Prompt
    context_text = "\n\n".join([
        f"Document [{i+1}] {d.get('title', 'Unknown')}:\n{d.get('paragraph_text', '')}" 
        for i, d in enumerate(selected_docs)
    ])

    final_prompt = f"""Use the following documents to answer the user's question. 
If the documents do not contain the answer, say "I don't know".
Keep the answer concise (e.g., entity name, date, location).

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
        # 清理答案，只取第一行
        final_answer = final_answer.split('\n')[0].strip()
    except:
        final_answer = "Error generating answer"

    return {
        'answer': final_answer,
        'chain': "\n".join(reasoning_chain),
        'contexts': selected_docs
    }