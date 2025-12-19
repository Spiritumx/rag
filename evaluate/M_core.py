import requests
import re
import traceback
from typing import List, Dict, Any, Set

def _extract_llm_text(response_json: Dict[str, Any]) -> str:
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
    """
    清洗对话式废话（如 "I need to find"），保留长实体和关键词。
    """
    original_query = query
    query = query.strip().strip('"\'[]').lower()
    
    # 定义垃圾短语 (LLM 喜欢说的废话)
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
            
    # 移除多余空格
    query = " ".join(query.split())
    
    # 如果清洗后变空了（说明全是废话），或者清洗出了问题，回退到原始 query（小写化）
    if len(query) < 1:
        return original_query.strip().lower()

    return query

def _is_semantically_similar(new_query: str, history_queries: Set[str], threshold: float = 0.9) -> bool:
    """
    语义去重 (Jaccard)。阈值设为 0.9，避免误伤长实体的细微差别。
    """
    new_tokens = set(new_query.lower().split())
    if not new_tokens: return False
    
    for old_q in history_queries:
        old_tokens = set(old_q.lower().split())
        if not old_tokens: continue
        
        intersection = new_tokens.intersection(old_tokens)
        union = new_tokens.union(old_tokens)
        similarity = len(intersection) / len(union)
        
        if similarity >= threshold:
            return True
    return False

def execute_real_multihop(
    query: str,
    retriever_config: dict,
    llm_config: dict,
    dataset_name: str
) -> dict:
    """
    Expert Agentic RAG (Warm-Start Version)
    
    Updates:
    1. Step 0: Initial Retrieval using the original query.
    2. Relaxed Cleaning: No longer truncates long queries, just removes chatty phrases.
    3. Context Injection: Keeps the logic of feeding investigation logs to final generation.
    """
    
    # === 配置 ===
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

    # === 状态 ===
    all_retrieved_docs: Dict[str, Dict] = {} 
    history_log: List[str] = []
    executed_queries: Set[str] = set()
    reasoning_chain: List[str] = []

    print(f"  [Agent] Start: {query[:60]}... (Corpus: {corpus_name})")

    # ==============================================================================
    # Step 0: 初始检索 (Warm Start)
    # ==============================================================================
    print("  [Agent] Step 0: Initial Retrieval with original query...")
    try:
        r_resp = requests.post(
            retriever_url,
            json={
                "retrieval_method": "retrieve_from_elasticsearch",
                "query_text": query,  # 直接用原始问题搜
                "rerank_query_text": query,
                "max_hits_count": 8,  # 初始检索多拿一点，建立背景
                "max_buffer_count": 40,
                "corpus_name": corpus_name,
                "document_type": "title_paragraph_text",
                "retrieval_backend": "hybrid"
            }, timeout=30
        )
        hits = r_resp.json().get('retrieval', [])
        
        # 处理结果
        init_snippets = []
        for h in hits:
            key = f"{h.get('title')} {h.get('paragraph_text')[:20]}"
            if key not in all_retrieved_docs:
                all_retrieved_docs[key] = h
                init_snippets.append(f"Title: {h['title']}\nContent: {h['paragraph_text'][:200]}...")
        
        if init_snippets:
            current_context_snippet = "\n---\n".join(init_snippets[:3])
            titles = [h['title'] for h in hits[:3]]
            # 写入历史
            history_log.append(f"Initial Search '{query}' -> Found docs: {', '.join(titles)}")
            executed_queries.add(query.lower()) # 标记原问题已搜过（小写）
            executed_queries.add(_remove_conversational_phrases(query.lower()))
        else:
            current_context_snippet = "Initial search returned no direct results."
            history_log.append(f"Initial Search '{query}' -> No direct results.")
            
        reasoning_chain.append(f"[Step 0] Initial Search Found {len(hits)} hits.")
        
    except Exception as e:
        print(f"  [Agent] Initial search failed: {e}")
        current_context_snippet = "Initial search failed."

    # ==============================================================================
    # Stage 1: 推理循环
    # ==============================================================================
    should_stop_reasoning = False
    
    for step in range(MAX_HOPS):
        if should_stop_reasoning:
            break

        valid_next_query = None
        feedback_msg = ""
        thought = ""
        action_raw = ""

        for attempt in range(MAX_RETRIES):
            history_str = "\n".join([f"Hop {i+1}: {h}" for i, h in enumerate(history_log)]) if history_log else "None"
            
            # Prompt 微调：强调基于 Current Info (Step 0 的结果) 进行分解
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

            try:
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
                feedback_msg = "Format Error."
                continue
            
            # 清洗对话式废话，保留长实体
            clean_q = _remove_conversational_phrases(raw_q)
            
            if len(clean_q) < 2:
                feedback_msg = "Query too short."
                continue
            
            # 语义去重 (阈值 0.9)
            if _is_semantically_similar(clean_q, executed_queries):
                print(f"    [Loop Prevented] '{clean_q}' similar to history.")
                feedback_msg = f"Loop Detected: You already searched '{clean_q}'. Try a different angle."
                continue 

            valid_next_query = clean_q
            break

        if should_stop_reasoning:
            break
            
        if not valid_next_query:
            reasoning_chain.append(f"[Hop {step+1}] Failed to generate valid query. Stopping.")
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
                    "max_hits_count": 5, 
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

        new_snippets = []
        for h in hits:
            key = f"{h.get('title')} {h.get('paragraph_text')[:20]}"
            if key not in all_retrieved_docs:
                all_retrieved_docs[key] = h
                new_snippets.append(f"Title: {h['title']}\nContent: {h['paragraph_text'][:200]}...")

        reasoning_chain.append(f"[Hop {step+1}] Found {len(hits)} hits.")

        if new_snippets:
            titles = [h['title'] for h in hits[:2]]
            history_log.append(f"Searched '{valid_next_query}' -> Found: {', '.join(titles)}")
            current_context_snippet = "\n---\n".join(new_snippets[:3]) 
        else:
            history_log.append(f"Searched '{valid_next_query}' -> No results.")
            current_context_snippet = "No relevant documents found. Try different keywords."

    # ==============================================================================
    # Stage 2: 生成答案
    # ==============================================================================
    final_docs = list(all_retrieved_docs.values())
    
    if not final_docs:
        return {'answer': "I don't know", 'chain': "\n".join(reasoning_chain), 'contexts': []}

    if len(final_docs) > MAX_CONTEXT_DOCS:
        selected_docs = final_docs[:15] + final_docs[-15:]
    else:
        selected_docs = final_docs

    context_str = "\n\n".join([f"[{i+1}] {d['title']}: {d['paragraph_text']}" for i, d in enumerate(selected_docs)])
    
    clean_history = [line for line in reasoning_chain if "Thought:" in line or "Action:" in line or "Found" in line]
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

    try:
        resp = requests.get(
            llm_url,
            params={'prompt': final_prompt, 'max_length': 300, 'temperature': 0.1},
            timeout=40
        )
        llm_output = _extract_llm_text(resp.json())
        
        reasoning_chain.append(f"[Final Generation] Raw: {llm_output[:100]}...")

        ans_match = re.search(r'Answer:\s*(.*)', llm_output, re.DOTALL | re.IGNORECASE)
        if ans_match:
            final_answer = ans_match.group(1).strip().split('\n')[0].strip()
        else:
            lines = [l for l in llm_output.split('\n') if l.strip()]
            final_answer = lines[-1] if lines else "I don't know"
            
    except Exception as e:
        reasoning_chain.append(f"[Final Error] {e}")
        final_answer = "Error"

    return {
        'answer': final_answer,
        'chain': "\n".join(reasoning_chain),
        'contexts': selected_docs
    }