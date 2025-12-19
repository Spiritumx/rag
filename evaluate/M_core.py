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
    """
    强力清洗查询语句，转化为关键词搜索模式。
    """
    # 1. 移除常见的自然语言引导词
    query = re.sub(r'^(search for|search|find|query|look for|what is|who is|where is|tell me about):?\s*', '', query, flags=re.IGNORECASE)
    
    # 2. 移除问号和引号
    query = query.replace('?', '').replace('"', '').replace("'", "").strip()
    
    # 3. 如果 query 依然很长且包含 stop words，可以考虑保留实体（这里简单处理，依赖 LLM 的自觉性）
    return query

def execute_real_multihop(
    query: str,
    retriever_config: dict,
    llm_config: dict,
    dataset_name: str
) -> dict:
    """
    Expert Agentic RAG (One-Shot CoT Version)
    
    改进点:
    1. One-Shot Example: 在 Prompt 中包含一个完美的推理范例，大幅减少格式错误。
    2. Explicit Decomposition: 强制模型寻找 "Bridge Entity"。
    3. Keyword Enforcement: 引导模型生成关键词而非句子。
    """
    
    # === 配置 ===
    MAX_HOPS = 6           # 6步通常足够，太多容易发散
    MAX_RETRIES = 3        # 单步重试
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

    # === 状态 ===
    all_retrieved_docs: Dict[str, Dict] = {} 
    history_log: List[str] = []
    executed_queries: Set[str] = set()
    reasoning_chain: List[str] = []

    print(f"  [Agent] Start: {query[:60]}... (Corpus: {corpus_name})")

    current_context_snippet = "No information gathered yet."
    should_stop_reasoning = False

    # ==============================================================================
    # Stage 1: 推理循环
    # ==============================================================================
    for step in range(MAX_HOPS):
        if should_stop_reasoning:
            break

        # --- 重试循环 ---
        valid_next_query = None
        feedback_msg = ""
        thought = ""
        action_raw = ""

        for attempt in range(MAX_RETRIES):
            history_str = "\n".join([f"Hop {i+1}: {h}" for i, h in enumerate(history_log)]) if history_log else "None"
            
            # === 核心 Prompt：包含 One-Shot 示例 ===
            base_prompt = f"""You are an expert multi-hop question answering agent.
Your goal is to break down a complex question into simple Keyword Searches to find the answer.

*** EXAMPLE ***
Question: "The director of the film 'Titanic' was born in which country?"
Hop 1:
Thought: I need to find the director of 'Titanic' first. The bridge entity is the director's name.
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

Instructions:
1. Identify the missing "Bridge Entity" needed to proceed.
2. If you know the answer based on "Current Info", output "Action: Answer".
3. Otherwise, output a KEYWORD SEARCH. Do NOT output sentences like "Search for the...".
4. STRICT FORMAT:
Thought: <short reasoning>
Action: <Search [Keywords] OR Answer>"""

            if feedback_msg:
                final_prompt = base_prompt + f"\n\n*** ERROR IN PREVIOUS ATTEMPT ***\n{feedback_msg}\nPlease fix."
            else:
                final_prompt = base_prompt

            try:
                # 调低 temperature，让它严格模仿 Example
                resp = requests.get(
                    llm_url,
                    params={'prompt': final_prompt, 'max_length': 128, 'temperature': 0.1}, 
                    timeout=30
                )
                llm_output = _extract_llm_text(resp.json())
            except Exception as e:
                reasoning_chain.append(f"[Error] LLM: {e}")
                should_stop_reasoning = True
                break

            # === 解析 ===
            thought_match = re.search(r'Thought:(.*?)(Action:|$)', llm_output, re.DOTALL | re.IGNORECASE)
            thought = thought_match.group(1).strip() if thought_match else ""

            action_match = re.search(r'Action:\s*(.*)', llm_output, re.IGNORECASE)
            if not action_match:
                # 尝试找 Search [...]
                fallback = re.search(r'Search\s*\[(.*?)\]', llm_output, re.IGNORECASE)
                if fallback:
                    action_raw = f"Search [{fallback.group(1)}]"
                else:
                    feedback_msg = "Format Error: Use 'Action: Search [Keywords]' or 'Action: Answer'."
                    continue
            else:
                action_raw = action_match.group(1).strip().split('\n')[0]

            # Case A: Answer
            if "Answer" in action_raw and "Search" not in action_raw:
                reasoning_chain.append(f"[Hop {step+1}] Thought: {thought}")
                reasoning_chain.append(f"[Hop {step+1}] Action: Answer")
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
                feedback_msg = "Format Error: Action must start with Search or Answer."
                continue
            
            # 清洗 & 验证
            temp_query = _clean_query(raw_q)
            
            if len(temp_query) < 2:
                feedback_msg = "Query too short."
                continue
            
            if temp_query.lower() in executed_queries:
                feedback_msg = f"Duplicate query: '{temp_query}'. Try a different entity or synonym."
                # 增加重试时的随机性，帮助跳出局部最优
                continue 

            valid_next_query = temp_query
            break

        # --- 执行 ---
        if should_stop_reasoning:
            break
            
        if not valid_next_query:
            reasoning_chain.append(f"[Hop {step+1}] Failed to generate action. Stopping.")
            break

        reasoning_chain.append(f"[Hop {step+1}] Thought: {thought}")
        reasoning_chain.append(f"[Hop {step+1}] Action: Search [{valid_next_query}]")
        executed_queries.add(valid_next_query.lower())

        try:
            r_resp = requests.post(
                retriever_url,
                json={
                    "retrieval_method": "retrieve_from_elasticsearch",
                    "query_text": valid_next_query,
                    "rerank_query_text": valid_next_query,
                    "max_hits_count": 5,  # 减少噪音，只看 Top 5
                    "max_buffer_count": 20,
                    "corpus_name": corpus_name,
                    "document_type": "title_paragraph_text",
                    "retrieval_backend": "hybrid"
                }, timeout=120
            )
            hits = r_resp.json().get('retrieval', [])
        except Exception as e:
            reasoning_chain.append(f"[Error] Retrieve: {e}")
            hits = []

        # 处理结果
        new_snippets = []
        for h in hits:
            title = h.get('title', '')
            paragraph_text = h.get('paragraph_text', '')
            if not title or not paragraph_text:
                continue
            key = f"{title} {paragraph_text[:20]}"
            if key not in all_retrieved_docs:
                all_retrieved_docs[key] = h
                # 摘要：Title + Content
                content_preview = paragraph_text[:200] if len(paragraph_text) > 200 else paragraph_text
                new_snippets.append(f"Title: {title}\nContent: {content_preview}...")

        reasoning_chain.append(f"[Hop {step+1}] Found {len(hits)} hits.")

        if new_snippets:
            # 更新 History: 这里的 History 必须简洁，告诉模型找到了什么实体
            titles = [h.get('title', 'Unknown') for h in hits[:2] if h.get('title')]
            history_log.append(f"Searched '{valid_next_query}' -> Found docs about: {', '.join(titles)}")
            
            # 更新 Context: 只给最近的一步，加上一步的关键信息
            current_context_snippet = "\n---\n".join(new_snippets[:3]) # 只保留 Top 3 防止 Context 污染
        else:
            history_log.append(f"Searched '{valid_next_query}' -> No results.")
            current_context_snippet = "No relevant documents found for the last query."

    # ==============================================================================
    # Stage 2: 生成答案
    # ==============================================================================
    final_docs = list(all_retrieved_docs.values())
    if not final_docs:
        return {'answer': "I don't know", 'chain': "\n".join(reasoning_chain), 'contexts': []}

    # 简单截取 Top 20 (因为之前每步只取了 Top 5，这里的 docs 质量较高)
    if len(final_docs) > MAX_CONTEXT_DOCS:
        selected_docs = final_docs[:10] + final_docs[-10:] # 头尾结合
    else:
        selected_docs = final_docs

    context_str = "\n\n".join([
        f"[{i+1}] {d.get('title', 'Unknown')}: {d.get('paragraph_text', '')}"
        for i, d in enumerate(selected_docs)
        if d.get('title') and d.get('paragraph_text')
    ])

    final_prompt = f"""Based on the documents below, answer the question.
If the answer is not present, say "I don't know".
Be very concise (entity name, date, or place).

Documents:
{context_str}

Question: {query}
Answer:"""

    try:
        resp = requests.get(
            llm_url,
            params={'prompt': final_prompt, 'max_length': 50, 'temperature': 0.1},
            timeout=30
        )
        ans = _extract_llm_text(resp.json()).split('\n')[0].strip()
    except:
        ans = "Error"

    return {
        'answer': ans,
        'chain': "\n".join(reasoning_chain),
        'contexts': selected_docs
    }