import requests
import re
import traceback
import os
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

def _remove_conversational_phrases(query: str) -> str:
    """
    [强力清洗] 移除对话式废话，保留长实体。
    """
    original_query = query
    query = query.strip().strip('"\'[]').lower()
    
    # 垃圾短语黑名单
    stop_phrases = [
        "i need to find", "i need to identify", "we need to", "try to find",
        "search for", "looking for", "find out", "figure out",
        "tell me", "what is", "who is", "where is", "when was",
        "the name of", "information about", "details regarding",
        "associated with", "related to", "step 1", "next step",
        "first step", "bridge entity", "logical step"
    ]
    
    for phrase in stop_phrases:
        if phrase in query:
            query = query.replace(phrase, " ")
            
    # 移除多余空格
    query = " ".join(query.split())
    
    # 如果清洗后变太短，回退到原始 query (防止误删实体)
    if len(query) < 2:
        return original_query.strip().lower()  # 🔧 修复：保持小写一致性

    return query

def _truncate_at_action(text: str) -> str:
    """
    [物理截断] 防止 LLM 自行生成多跳幻觉。
    """
    lines = text.split('\n')
    truncated_lines = []
    found_action = False
    
    for line in lines:
        truncated_lines.append(line)
        if "Action:" in line:
            found_action = True
            break 
            
    if found_action:
        return "\n".join(truncated_lines)
    return text

def _verify_answer_evidence(thought: str, context_docs: List[Dict]) -> bool:
    """
    [宽松证据校验] 
    1. 如果没有文档，拦截。
    2. 如果模型承认没找到，拦截。
    3. 否则放行（信任模型的语义理解）。
    """
    if not context_docs:
        return False
        
    negative_signals = [
        "not found", "no information", "didn't find", "unable to find", 
        "doesn't mention", "no mention", "unknown", "can't answer",
        "i don't know"
    ]
    thought_lower = thought.lower()
    
    # 检查是否包含否定词
    for neg in negative_signals:
        if neg in thought_lower:
            # 简单检查是否有转折 (but found...)，如果没有转折，则视为失败
            if "but" not in thought_lower and "however" not in thought_lower:
                return False

    return True

def _is_semantically_similar(new_query: str, history_queries: Set[str], threshold: float = 0.9) -> bool:
    """语义去重 (Jaccard)"""
    new_tokens = set(new_query.lower().split())
    if not new_tokens: return False
    
    for old_q in history_queries:
        old_tokens = set(old_q.lower().split())
        if not old_tokens: continue
        
        intersection = new_tokens.intersection(old_tokens)
        union = new_tokens.union(old_tokens)
        if len(union) == 0: continue
        
        if len(intersection) / len(union) >= threshold:
            return True
    return False

def load_prompt_template(dataset_name: str, stage: str) -> str:
    """
    Load M strategy prompt template from file.

    Args:
        dataset_name: e.g., 'hotpotqa', 'squad'
        stage: 'stage0', 'stage1', or 'stage2'

    Returns:
        Prompt template string
    """
    stage_files = {
        'stage0': 'stage0_logic_decomposition.txt',
        'stage1': 'stage1_iterative_reasoning_cot.txt',
        'stage2': 'stage2_final_answer_direct.txt'
    }

    prompt_path = f"prompts/{dataset_name}/{stage_files[stage]}"

    # Fallback to hotpotqa if file doesn't exist
    if not os.path.exists(prompt_path):
        prompt_path = f"prompts/hotpotqa/{stage_files[stage]}"
        print(f"  [Warning] Using fallback prompt: {prompt_path}")

    with open(prompt_path, 'r', encoding='utf-8') as f:
        return f.read().strip()

def _generate_logical_plan(query: str, llm_url: str, dataset_name: str) -> str:
    """
    [Stage 0] 逻辑拆解模块
    将自然语言转化为逻辑表达式，指导后续搜索。
    """
    template = load_prompt_template(dataset_name, 'stage0')
    prompt = template.format(query=query)

    try:
        resp = requests.get(
            llm_url,
            params={'prompt': prompt, 'max_length': 256, 'temperature': 0.1},
            timeout=60  # 增加超时到60秒
        )
        llm_output = _extract_llm_text(resp.json()).strip()

        # 🔧 鲁棒解析：跳过引导性废话，提取实际的逻辑路径
        # 查找包含 "->" 的行（这是逻辑路径的标志）
        lines = llm_output.split('\n')
        for line in lines:
            line = line.strip()
            if '->' in line and not any(skip in line.lower() for skip in [
                'convert', 'break it down', 'step by step', 'logical path',
                "here's", "let me", "i'll", "to answer"
            ]):
                plan = line
                print(f"  [Logic] Plan: {plan}")
                return plan

        # 如果没找到标准格式，取第一个非空行（去掉引导性句子）
        for line in lines:
            line = line.strip()
            if line and not any(skip in line.lower() for skip in [
                'convert', 'break it down', 'step by step',
                "here's the", "here is the", "let me", "i'll"
            ]):
                plan = line
                print(f"  [Logic] Plan: {plan}")
                return plan

        # 最后兜底：返回整个输出的第一行
        plan = lines[0].strip() if lines else "Decompose the question step by step."
        print(f"  [Logic] Plan: {plan}")
        return plan

    except Exception as e:  # 🔧 修复：只捕获 Exception，不捕获系统异常
        print(f"  [Logic] Failed to generate plan: {e}")
        return "Decompose the question step by step."

def execute_real_multihop(
    query: str,
    retriever_config: dict,
    llm_config: dict,
    dataset_name: str
) -> dict:
    """
    Expert Agentic RAG (Logic-Guided Version)
    """
    
    # === 配置 ===
    MAX_HOPS = 6
    MAX_RETRIES = 3
    MAX_CONTEXT_DOCS = 40
    
    dataset_to_corpus = {
        'hotpotqa': 'hotpotqa', 'musique': 'musique',
        '2wikimultihopqa': '2wikimultihopqa', 'iirc': 'iirc', 'wiki': 'wiki'
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
    # Stage 0: 逻辑规划 & 初始检索
    # ==============================================================================
    
    # 0.a 生成逻辑规划 (Query Rewriting / Decomposition)
    logical_plan = _generate_logical_plan(query, llm_url, dataset_name)
    reasoning_chain.append(f"[Step 0] Logical Plan: {logical_plan}")

    # 0.b Warm Start (Initial Retrieval)
    print("  [Agent] Step 0: Initial Retrieval...")
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
            executed_queries.add(_remove_conversational_phrases(query))  # 🔧 修复：函数重命名
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
            
            # Prompt 核心：注入逻辑规划 (Logical Strategy)
            template = load_prompt_template(dataset_name, 'stage1')
            base_prompt = template.format(
                logical_plan=logical_plan,
                query=query,
                history_str=history_str,
                current_context_snippet=current_context_snippet
            )

            if feedback_msg:
                final_prompt = base_prompt + f"\n\n*** ERROR ***\n{feedback_msg}"
            else:
                final_prompt = base_prompt

            try:
                resp = requests.get(
                    llm_url,
                    params={'prompt': final_prompt, 'max_length': 128, 'temperature': 0.1},
                    timeout=60  # 增加超时到60秒
                )
                raw_output = _extract_llm_text(resp.json())
                
                # [Critical] 物理截断
                llm_output = _truncate_at_action(raw_output)
                
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
                # [Critical] 宽松证据校验
                final_docs_check = list(all_retrieved_docs.values())
                if _verify_answer_evidence(thought, final_docs_check):
                    reasoning_chain.append(f"[Hop {step+1}] Thought: {thought}")
                    reasoning_chain.append(f"[Hop {step+1}] Action: Answer")
                    should_stop_reasoning = True
                    valid_next_query = None
                    break
                else:
                    print(f"    [Evidence Check Failed] Agent was unsure or had no docs.")
                    feedback_msg = "You are trying to Answer, but you haven't gathered enough information (or said 'not found'). Please SEARCH for the missing entity."
                    continue

            # Case B: Search
            search_match = re.search(r'Search\s*\[?(.*?)\]?$', action_raw, re.IGNORECASE)
            if search_match:
                raw_q = search_match.group(1)
            elif action_raw.lower().startswith("search"):
                raw_q = action_raw[6:].strip()
            else:
                feedback_msg = "Format Error."
                continue
            
            # 强力清洗
            clean_q = _remove_conversational_phrases(raw_q)  # 🔧 修复：函数重命名
            
            if len(clean_q) < 2:
                feedback_msg = "Query too short."
                continue
            
            # 语义去重
            if _is_semantically_similar(clean_q, executed_queries):
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
        executed_queries.add(valid_next_query.lower())  # 🔧 修复：显式转小写确保一致性

        # Search
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
    # Stage 2: Final Generation (Injection + CoT + Logic Plan)
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

    template = load_prompt_template(dataset_name, 'stage2')
    final_prompt = template.format(
        logical_plan=logical_plan,
        investigation_log=investigation_log,
        context_str=context_str,
        query=query
    )

    try:
        resp = requests.get(
            llm_url,
            params={'prompt': final_prompt, 'max_length': 300, 'temperature': 0.1},
            timeout=90  # 增加超时到90秒（最终生成需要更长时间）
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