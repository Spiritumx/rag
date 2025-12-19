import requests
import traceback
from typing import List, Dict, Any

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

def execute_real_multihop(
    query: str,
    retriever_config: dict,
    llm_config: dict,
    dataset_name: str
) -> dict:
    """
    高级 Agentic RAG：包含"检查-重写-检索"闭环逻辑
    解决 Fester Hollow 问题的核心：如果第一步没搜到，第二步会重写查询去专门搜它。

    Args:
        query: 用户问题
        retriever_config: Retriever 配置 (host, port)
        llm_config: LLM 配置 (host, port)
        dataset_name: 数据集名称，用于确定 corpus_name
    """
    
    # 状态维护
    context_text = ""
    seen_ids = set()
    final_docs = []
    reasoning_steps = []

    # 参数
    MAX_HOPS = 4  # 通常 2-3 跳足够，给多一次容错
    MAX_CONTEXT_DOCS = 40

    # 根据数据集名称确定 corpus_name（对应 ES 索引名）
    dataset_to_corpus = {
        'hotpotqa': 'hotpotqa',
        'musique': 'musique',
        '2wikimultihopqa': '2wikimultihopqa',
        'iirc': 'iirc',
        'wiki': 'wiki'
    }

    # 使用 wiki 作为默认索引（适用于未知数据集）
    dataset_lower = dataset_name.lower()
    if dataset_lower in dataset_to_corpus:
        corpus_name = dataset_to_corpus[dataset_lower]
    else:
        corpus_name = 'wiki'
        print(f"  [Agent] Warning: Unknown dataset '{dataset_name}', using default 'wiki' corpus")

    retriever_url = f"http://{retriever_config['host']}:{retriever_config['port']}/retrieve/"
    llm_url = f"http://{llm_config['host']}:{llm_config['port']}/generate"

    print(f"  [Agent] Corpus: {corpus_name} | Starting reasoning for: {query[:50]}...")

    for step in range(MAX_HOPS):
        # ==============================================================================
        # Step A: 自适应规划 (Adaptive Planning with Reflection)
        # ==============================================================================
        # 这个 Prompt 是核心：教模型判断当前 Context 是否足够，以及缺失什么
        
        prompt_content = f"""Task: You are a multi-hop QA agent. Analyze the Request and the Known Information. Determine the Next Search Query.

Strategy:
1. **Check**: Look at the Known Information. Did we find the entities mentioned in the Request?
2. **Reformulate**: 
   - If a key entity (e.g., "Fester Hollow") is MISSING from Known Information, search specifically for it (e.g., "location of Fester Hollow").
   - If the entity IS FOUND, use that info to jump to the next hop (e.g., "mountains near Portland Pennsylvania").
   - If you have the final answer, output "DONE".

Example 1 (Success Path):
Request: Who is the CEO of the company that acquired WhatsApp?
Known: WhatsApp was acquired by Facebook in 2014.
Thought: I found the acquiring company (Facebook). Now I need its CEO.
Next Search Query: CEO of Facebook

Example 2 (Recovery Path - The "Fester Hollow" Logic):
Request: What mountains are in the state containing Fester Hollow?
Known: [Docs about "Uncle Fester", "Sleepy Hollow"... nothing about the place "Fester Hollow"]
Thought: I have not found the location of "Fester Hollow" yet. The current docs are irrelevant. I need to find where Fester Hollow is first.
Next Search Query: location of Fester Hollow Pennsylvania

Example 3 (Complex):
Request: When was the author of Harry Potter born?
Known: None.
Thought: I need to find the author of Harry Potter first.
Next Search Query: author of Harry Potter

Current Task:
Request: {query}
Known: {context_text if context_text else "No relevant documents found yet."}

Instructions:
- Output your 'Thought' first, then the 'Next Search Query'.
- Keep the query specific and keyword-rich.
"""

        # 调用 LLM
        try:
            response = requests.get(
                llm_url,
                params={'prompt': prompt_content, 'max_length': 100, 'temperature': 0.1, 'do_sample': False},
                timeout=60
            )
            llm_output = _extract_llm_text(response.json())
        except Exception as e:
            reasoning_steps.append(f"[Hop {step+1}] LLM Error: {e}")
            break

        # 解析 Output (格式：Thought: ... Next Search Query: ...)
        thought = ""
        next_query = ""
        
        # 简单的解析逻辑
        if "Next Search Query:" in llm_output:
            parts = llm_output.split("Next Search Query:")
            thought = parts[0].replace("Thought:", "").strip()
            next_query = parts[1].strip()
        else:
            # Fallback: 如果模型没按格式输出，假设全是 query
            next_query = llm_output.strip()

        reasoning_steps.append(f"[Hop {step+1}] Thought: {thought}")
        reasoning_steps.append(f"[Hop {step+1}] Query: {next_query}")

        # ==============================================================================
        # Step B: 终止检查
        # ==============================================================================
        if "DONE" in next_query or len(next_query) < 2:
            reasoning_steps.append(f"[Hop {step+1}] Agent decided to stop.")
            break

        # ==============================================================================
        # Step C: 检索 (Retrieval)
        # ==============================================================================
        try:
            # 这里的关键：Query 已经由 LLM 重写过了，如果是"Fester Hollow location"，BM25 就能搜到了
            response = requests.post(
                retriever_url,
                json={
                    "retrieval_method": "retrieve_from_elasticsearch",
                    "query_text": next_query,        # 使用重写后的 Query
                    "rerank_query_text": next_query, # Rerank 也专注当前意图
                    "max_hits_count": 10,
                    "max_buffer_count": 60,
                    "corpus_name": corpus_name,
                    "document_type": "title_paragraph_text",
                    "retrieval_backend": "hybrid"
                }, timeout=30
            )
            hits = response.json().get('retrieval', [])
        except Exception as e:
            reasoning_steps.append(f"[Hop {step+1}] Retriever Error: {e}")
            hits = []

        reasoning_steps.append(f"[Hop {step+1}] Found {len(hits)} docs.")

        # ==============================================================================
        # Step D: 上下文累积 (Accumulation)
        # ==============================================================================
        new_docs_added = 0
        current_step_snippets = []
        
        for h in hits:
            doc_id = h.get('id', h.get('title'))
            if doc_id not in seen_ids:
                seen_ids.add(doc_id)
                final_docs.append(h)
                
                # 构造用于下一次 Prompt 的摘要
                # 截取前 200 字符，避免 Prompt 爆炸
                snippet = f"Title: {h.get('title', '')} | Content: {h.get('paragraph_text', '')[:200]}..."
                current_step_snippets.append(snippet)
                new_docs_added += 1

        # 更新 Context Text (给 LLM 看的)
        if current_step_snippets:
            context_text += f"\n\n--- Retrieved in Step {step+1} ---\n" + "\n".join(current_step_snippets)
        
        # 滚动窗口：保持 final_docs 不超过限制 (用于最后生成)
        if len(final_docs) > MAX_CONTEXT_DOCS:
            final_docs = final_docs[-MAX_CONTEXT_DOCS:]

        # 如果这一步啥都没搜到，强制终止防止死循环
        if new_docs_added == 0:
            reasoning_steps.append(f"[Hop {step+1}] No new info found. Stopping.")
            break

    # ==============================================================================
    # Step E: 最终生成 (Generation)
    # ==============================================================================
    # 全局 Rerank (可选，建议保留以提升 Top-10 质量)
    if len(final_docs) > 10:
        try:
            # 用原始问题对所有积累的文档进行最后一次清洗
            response = requests.post(
                retriever_url,
                json={
                    "retrieval_method": "retrieve_from_elasticsearch",
                    "query_text": query,
                    "rerank_query_text": query,
                    "max_hits_count": 10,
                    "max_buffer_count": 60,
                    "corpus_name": corpus_name,
                    "document_type": "title_paragraph_text",
                    "retrieval_backend": "hybrid"
                }, timeout=30
            )
            # 这里不仅用检索结果，更重要的是利用 Reranker 对我们手头的 final_docs 进行重排
            # 但如果你没有单独的 Rerank 接口，就用混合策略：
            # 简单截取最后几轮的文档（因为它们通常包含最终答案）
            final_docs = final_docs[-10:] 
        except:
            final_docs = final_docs[-10:]

    docs_text = "\n\n".join([
        f"[{i+1}] {d.get('title', '')}: {d.get('paragraph_text', '')[:500]}" 
        for i, d in enumerate(final_docs)
    ])

    final_prompt = f"""Answer the question based on the documents.
Documents:
{docs_text}

Question: {query}
Answer (concise):"""

    try:
        response = requests.get(
            llm_url,
            params={'prompt': final_prompt, 'max_length': 50, 'temperature': 0.1, 'do_sample': False},
            timeout=60
        )
        answer = _extract_llm_text(response.json())
    except:
        answer = "I don't know"

    return {
        'answer': answer,
        'chain': "\n".join(reasoning_steps),
        'contexts': final_docs
    }