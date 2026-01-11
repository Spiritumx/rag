"""
MI-RA-ToT: Mutual Information Tree-of-Thought Reasoning (Innovation 3)

Replaces baseline's linear greedy search with beam search tree exploration.
Uses mutual information scoring (novelty + relevance) for path pruning.
"""

import requests
import re
import traceback
from typing import List, Dict, Any, Set, Optional
from dataclasses import dataclass, field
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# Reuse helper functions from baseline M_core.py
# ============================================================================

def _extract_llm_text(response_json: Dict[str, Any]) -> str:
    """Extract text from various LLM API response formats."""
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
    """Remove conversational fluff, keep entities."""
    original_query = query
    query = query.strip().strip('"\'[]').lower()

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

    query = " ".join(query.split())

    if len(query) < 2:
        return original_query.strip().lower()

    return query


def _truncate_at_action(text: str) -> str:
    """Physical truncation to prevent LLM hallucination."""
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


def _is_semantically_similar(new_query: str, history_queries: Set[str], threshold: float = 0.9) -> bool:
    """Semantic deduplication using Jaccard similarity."""
    new_tokens = set(new_query.lower().split())
    if not new_tokens:
        return False

    for old_q in history_queries:
        old_tokens = set(old_q.lower().split())
        if not old_tokens:
            continue

        intersection = new_tokens.intersection(old_tokens)
        union = new_tokens.union(old_tokens)
        if len(union) == 0:
            continue

        if len(intersection) / len(union) >= threshold:
            return True
    return False


# ============================================================================
# Innovation 3: Tree-of-Thought Components
# ============================================================================

@dataclass
class TreeNode:
    """
    Represents a state in the reasoning tree.

    Each node contains:
    - thought: The reasoning text
    - query: The search query to execute
    - contexts: Retrieved documents at this node
    - score: Cumulative MI-based score
    - parent: Parent node (None for root)
    - depth: Depth in tree (0 for root)
    - children: List of child nodes
    """
    thought: str
    query: str
    contexts: List[Dict]
    score: float
    parent: Optional['TreeNode'] = None
    depth: int = 0
    children: List['TreeNode'] = field(default_factory=list)
    action_type: str = "search"  # "search" or "answer"

    def add_child(self, child: 'TreeNode'):
        """Add a child node to this node."""
        self.children.append(child)
        child.parent = self
        child.depth = self.depth + 1

    def get_path_to_root(self) -> List['TreeNode']:
        """Get path from this node to root."""
        path = []
        current = self
        while current is not None:
            path.append(current)
            current = current.parent
        return list(reversed(path))

    def get_all_contexts(self) -> List[Dict]:
        """Get all contexts from root to this node."""
        path = self.get_path_to_root()
        all_contexts = []
        seen_keys = set()

        for node in path:
            for ctx in node.contexts:
                key = f"{ctx.get('title')} {ctx.get('paragraph_text', '')[:20]}"
                if key not in seen_keys:
                    all_contexts.append(ctx)
                    seen_keys.add(key)

        return all_contexts


class MutualInformationScorer:
    """
    Computes MI gain using novelty + relevance hybrid.

    Formula: MI_gain = α × relevance(doc, question) + β × novelty(doc, existing_context)
    - α (mi_alpha): Weight for relevance (default 0.7)
    - β (mi_beta): Weight for novelty (default 0.3)
    """

    def __init__(self, reranker_model=None, mi_alpha: float = 0.7, mi_beta: float = 0.3):
        """
        Initialize MI scorer.

        Args:
            reranker_model: Cross-encoder model for relevance scoring
            mi_alpha: Weight for relevance component (0-1)
            mi_beta: Weight for novelty component (0-1)
        """
        self.reranker = reranker_model
        self.mi_alpha = mi_alpha
        self.mi_beta = mi_beta

        if self.reranker is None:
            logger.warning("No reranker provided, using simple scoring")

    def calculate_relevance(self, question: str, doc: Dict) -> float:
        """
        Calculate relevance of document to question.

        Args:
            question: The original question
            doc: Document dictionary with 'paragraph_text'

        Returns:
            Relevance score (0-1)
        """
        if self.reranker is None:
            # Fallback: Simple keyword overlap
            q_tokens = set(question.lower().split())
            d_tokens = set(doc.get('paragraph_text', '').lower().split())
            if not q_tokens or not d_tokens:
                return 0.0
            overlap = len(q_tokens.intersection(d_tokens))
            return min(overlap / len(q_tokens), 1.0)

        try:
            # Use cross-encoder for semantic relevance
            text = doc.get('paragraph_text', '')
            if not text:
                return 0.0

            score = self.reranker.predict([(question, text)])[0]
            # Normalize to [0, 1]
            return max(0.0, min(1.0, float(score)))
        except Exception as e:
            logger.warning(f"Relevance scoring failed: {e}")
            return 0.5

    def calculate_novelty(self, new_doc: Dict, existing_contexts: List[Dict]) -> float:
        """
        Calculate novelty of new document w.r.t. existing contexts.

        Args:
            new_doc: New document to score
            existing_contexts: List of already retrieved documents

        Returns:
            Novelty score (0-1), higher = more novel
        """
        if not existing_contexts:
            return 1.0  # First document is always novel

        new_text = new_doc.get('paragraph_text', '').lower()
        new_tokens = set(new_text.split())

        if not new_tokens:
            return 0.0

        max_similarity = 0.0
        for old_doc in existing_contexts:
            old_text = old_doc.get('paragraph_text', '').lower()
            old_tokens = set(old_text.split())

            if not old_tokens:
                continue

            # Jaccard similarity
            intersection = new_tokens.intersection(old_tokens)
            union = new_tokens.union(old_tokens)

            if len(union) > 0:
                similarity = len(intersection) / len(union)
                max_similarity = max(max_similarity, similarity)

        # Novelty = 1 - max_similarity
        novelty = 1.0 - max_similarity
        return novelty

    def calculate_mi_gain(
        self,
        question: str,
        new_docs: List[Dict],
        existing_contexts: List[Dict]
    ) -> float:
        """
        Calculate MI gain for a set of new documents.

        Args:
            question: Original question
            new_docs: New documents retrieved
            existing_contexts: Already retrieved documents

        Returns:
            MI gain score (0-1)
        """
        if not new_docs:
            return 0.0

        total_score = 0.0
        for doc in new_docs:
            relevance = self.calculate_relevance(question, doc)
            novelty = self.calculate_novelty(doc, existing_contexts)

            # Hybrid score: α × relevance + β × novelty
            mi_score = self.mi_alpha * relevance + self.mi_beta * novelty
            total_score += mi_score

        # Average MI gain across all new docs
        avg_mi = total_score / len(new_docs)
        return avg_mi


class BeamSearchToT:
    """
    Beam search Tree-of-Thought reasoning engine.

    Explores multiple reasoning paths in parallel, pruning low-scoring branches.
    """

    def __init__(
        self,
        retriever_config: dict,
        llm_config: dict,
        dataset_name: str,
        beam_width: int = 3,
        max_depth: int = 4,
        candidates_per_node: int = 3,
        mi_alpha: float = 0.7,
        mi_beta: float = 0.3,
        reranker_model=None,
    ):
        """
        Initialize BeamSearchToT.

        Args:
            retriever_config: Retriever endpoint config
            llm_config: LLM endpoint config
            dataset_name: Dataset name for corpus mapping
            beam_width: Number of paths to keep per layer (B)
            max_depth: Maximum tree depth
            candidates_per_node: k candidate queries per node
            mi_alpha: Relevance weight in MI scoring
            mi_beta: Novelty weight in MI scoring
            reranker_model: Optional reranker for relevance scoring
        """
        self.retriever_config = retriever_config
        self.llm_config = llm_config
        self.dataset_name = dataset_name
        self.beam_width = beam_width
        self.max_depth = max_depth
        self.candidates_per_node = candidates_per_node

        # Corpus mapping
        dataset_to_corpus = {
            'hotpotqa': 'hotpotqa', 'musique': 'musique',
            '2wikimultihopqa': '2wikimultihopqa', 'iirc': 'iirc', 'wiki': 'wiki'
        }
        self.corpus_name = dataset_to_corpus.get(dataset_name.lower(), 'wiki')

        # URLs
        self.retriever_url = f"http://{retriever_config['host']}:{retriever_config['port']}/retrieve/"
        self.llm_url = f"http://{llm_config['host']}:{llm_config['port']}/generate"

        # MI scorer
        self.mi_scorer = MutualInformationScorer(
            reranker_model=reranker_model,
            mi_alpha=mi_alpha,
            mi_beta=mi_beta,
        )

        # State tracking
        self.executed_queries: Set[str] = set()
        self.reasoning_chain: List[str] = []

    def _retrieve_documents(self, query: str, max_hits: int = 5) -> List[Dict]:
        """Execute retrieval for a query."""
        try:
            response = requests.post(
                self.retriever_url,
                json={
                    "retrieval_method": "retrieve_from_elasticsearch",
                    "query_text": query,
                    "rerank_query_text": query,
                    "max_hits_count": max_hits,
                    "max_buffer_count": 20,
                    "corpus_name": self.corpus_name,
                    "document_type": "title_paragraph_text",
                    "retrieval_backend": "hybrid"
                },
                timeout=120
            )
            hits = response.json().get('retrieval', [])
            return hits
        except Exception as e:
            logger.error(f"Retrieval failed for '{query}': {e}")
            return []

    def _generate_candidate_queries(
        self,
        question: str,
        node: TreeNode,
        k: int
    ) -> List[str]:
        """
        Generate k candidate next-step queries from a node.

        Args:
            question: Original question
            node: Current tree node
            k: Number of candidates to generate

        Returns:
            List of candidate queries (cleaned)
        """
        # Build context from path
        path = node.get_path_to_root()
        history_str = "\n".join([
            f"Step {i}: {n.thought[:100]}..." for i, n in enumerate(path) if n.thought
        ]) if len(path) > 1 else "None"

        # Get recent contexts
        all_contexts = node.get_all_contexts()
        context_snippets = [
            f"Title: {c['title']}\nContent: {c.get('paragraph_text', '')[:150]}..."
            for c in all_contexts[:3]
        ]
        context_str = "\n---\n".join(context_snippets) if context_snippets else "No context yet."

        prompt = f"""You are an expert research agent exploring multiple reasoning paths.
Generate {k} DIFFERENT next search queries to find the answer.

Question: {question}

Past Steps:
{history_str}

Current Info:
{context_str}

*** RULES ***
1. Generate {k} diverse search queries (keywords only, not questions)
2. Each query should explore a different aspect
3. Output format: "Query 1: ...\nQuery 2: ...\nQuery 3: ..."

Output:"""

        try:
            response = requests.get(
                self.llm_url,
                params={'prompt': prompt, 'max_length': 128, 'temperature': 0.7},  # Higher temp for diversity
                timeout=60
            )
            raw_output = _extract_llm_text(response.json())

            # Parse queries
            candidates = []
            for line in raw_output.split('\n'):
                match = re.search(r'Query\s*\d*:\s*(.*)', line, re.IGNORECASE)
                if match:
                    raw_q = match.group(1).strip()
                    clean_q = _remove_conversational_phrases(raw_q)
                    if len(clean_q) >= 2 and not _is_semantically_similar(clean_q, self.executed_queries):
                        candidates.append(clean_q)
                        if len(candidates) >= k:
                            break

            # Fallback: If not enough candidates, generate simple variations
            if len(candidates) < k:
                logger.warning(f"Only generated {len(candidates)}/{k} candidates, using fallback")
                # Simple fallback: Reuse last query with minor variations (not ideal, but prevents failure)
                while len(candidates) < k:
                    candidates.append(candidates[0] if candidates else question[:30])

            return candidates[:k]

        except Exception as e:
            logger.error(f"Failed to generate candidate queries: {e}")
            # Emergency fallback
            return [question[:30]] * k

    def _can_answer_question(self, node: TreeNode, question: str) -> bool:
        """
        Check if current node has enough information to answer.

        Args:
            node: Current node
            question: Original question

        Returns:
            True if node can answer, False otherwise
        """
        all_contexts = node.get_all_contexts()
        if len(all_contexts) < 2:
            return False  # Need at least 2 documents for multi-hop

        # Check if we've reached sufficient depth
        if node.depth >= self.max_depth - 1:
            return True  # Force answer at max depth

        # Simple heuristic: Can answer if have enough context
        return len(all_contexts) >= 5

    def search(self, question: str) -> Dict:
        """
        Execute beam search Tree-of-Thought.

        Args:
            question: The question to answer

        Returns:
            Dict with 'answer', 'chain', 'contexts'
        """
        logger.info(f"[ToT] Starting beam search for: {question[:60]}...")
        self.reasoning_chain.append(f"[ToT] Question: {question}")
        self.reasoning_chain.append(f"[ToT] Beam Width: {self.beam_width}, Max Depth: {self.max_depth}")

        # Initialize root node with initial retrieval
        logger.info("[ToT] Step 0: Initial retrieval...")
        initial_hits = self._retrieve_documents(question, max_hits=8)
        self.executed_queries.add(question.lower())

        root = TreeNode(
            thought="Initial exploration",
            query=question,
            contexts=initial_hits,
            score=0.0,
            depth=0
        )
        self.reasoning_chain.append(f"[ToT] Root: Retrieved {len(initial_hits)} initial documents")

        # Beam: Start with root
        current_beam = [root]

        # Beam search loop
        for depth in range(1, self.max_depth + 1):
            logger.info(f"[ToT] Depth {depth}: Expanding {len(current_beam)} nodes...")
            candidates = []

            # Expand each node in current beam
            for node_idx, node in enumerate(current_beam):
                # Check if this node can answer
                if self._can_answer_question(node, question):
                    logger.info(f"[ToT] Node {node_idx} can answer at depth {depth}")
                    # Keep it as a candidate to potentially be selected
                    # But also try to expand it to see if we can get better answer

                # Generate k candidate next queries
                candidate_queries = self._generate_candidate_queries(question, node, self.candidates_per_node)

                logger.info(f"[ToT] Node {node_idx} generated {len(candidate_queries)} candidates")

                # Execute retrieval for each candidate
                for query in candidate_queries:
                    # Retrieve documents
                    new_docs = self._retrieve_documents(query, max_hits=5)

                    if not new_docs:
                        continue  # Skip if no results

                    # Calculate MI gain
                    existing_contexts = node.get_all_contexts()
                    mi_gain = self.mi_scorer.calculate_mi_gain(question, new_docs, existing_contexts)

                    # Create new node
                    new_score = node.score + mi_gain
                    child = TreeNode(
                        thought=f"Search for: {query}",
                        query=query,
                        contexts=new_docs,
                        score=new_score,
                        parent=node,
                        depth=depth
                    )
                    node.add_child(child)
                    candidates.append(child)

                    self.executed_queries.add(query.lower())
                    logger.info(f"[ToT]   Query: '{query}' | MI Gain: {mi_gain:.3f} | Total Score: {new_score:.3f}")

            if not candidates:
                logger.warning(f"[ToT] No candidates generated at depth {depth}, stopping")
                break

            # Prune: Keep top-B nodes by score
            current_beam = sorted(candidates, key=lambda n: n.score, reverse=True)[:self.beam_width]

            self.reasoning_chain.append(
                f"[ToT] Depth {depth}: {len(candidates)} candidates → Pruned to top {len(current_beam)}"
            )

            # Log top beam nodes
            for i, node in enumerate(current_beam):
                self.reasoning_chain.append(f"  Beam[{i}]: Score={node.score:.3f}, Query='{node.query[:50]}'")

        # Select best node from final beam
        if not current_beam:
            logger.error("[ToT] No valid paths found!")
            return {
                'answer': "I don't know",
                'chain': "\n".join(self.reasoning_chain),
                'contexts': []
            }

        best_node = current_beam[0]
        logger.info(f"[ToT] Best path score: {best_node.score:.3f}, Depth: {best_node.depth}")

        # Generate final answer from best path
        return self._generate_final_answer(question, best_node)

    def _generate_final_answer(self, question: str, node: TreeNode) -> Dict:
        """
        Generate final answer using contexts from best path.

        Args:
            question: Original question
            node: Best leaf node from beam search

        Returns:
            Dict with 'answer', 'chain', 'contexts'
        """
        logger.info("[ToT] Generating final answer from best path...")

        # Get all contexts from path
        all_contexts = node.get_all_contexts()

        if not all_contexts:
            return {
                'answer': "I don't know",
                'chain': "\n".join(self.reasoning_chain),
                'contexts': []
            }

        # Smart selection: First 15 + Last 15 (like baseline)
        MAX_CONTEXT_DOCS = 40
        if len(all_contexts) > MAX_CONTEXT_DOCS:
            selected_docs = all_contexts[:15] + all_contexts[-15:]
        else:
            selected_docs = all_contexts

        # Build path summary
        path = node.get_path_to_root()
        path_summary = "\n".join([
            f"Step {i}: {n.thought} (Query: {n.query})"
            for i, n in enumerate(path) if n.depth > 0
        ])

        context_str = "\n\n".join([
            f"[{i+1}] {d['title']}: {d.get('paragraph_text', '')}"
            for i, d in enumerate(selected_docs)
        ])

        final_prompt = f"""Task: Answer the question based on the investigation path and retrieved documents.

*** INVESTIGATION PATH ***
{path_summary}

*** RETRIEVED DOCUMENTS ***
{context_str}

*** INSTRUCTIONS ***
1. Synthesize information from multiple documents
2. Provide a concise answer (Entity, Date, or Name only)

*** QUESTION ***
{question}

*** FORMAT ***
Thought: <analysis>
Answer: <concise answer>
"""

        try:
            response = requests.get(
                self.llm_url,
                params={'prompt': final_prompt, 'max_length': 300, 'temperature': 0.1},
                timeout=90
            )
            llm_output = _extract_llm_text(response.json())
            self.reasoning_chain.append(f"[ToT Final] Raw: {llm_output[:100]}...")

            ans_match = re.search(r'Answer:\s*(.*)', llm_output, re.DOTALL | re.IGNORECASE)
            if ans_match:
                final_answer = ans_match.group(1).strip().split('\n')[0].strip()
            else:
                lines = [l for l in llm_output.split('\n') if l.strip()]
                final_answer = lines[-1] if lines else "I don't know"
        except Exception as e:
            self.reasoning_chain.append(f"[ToT Final Error] {e}")
            final_answer = "Error"

        self.reasoning_chain.append(f"[ToT Final] Answer: {final_answer}")

        return {
            'answer': final_answer,
            'chain': "\n".join(self.reasoning_chain),
            'contexts': selected_docs
        }


# ============================================================================
# Main Entry Point
# ============================================================================

def execute_tot_multihop(
    query: str,
    retriever_config: dict,
    llm_config: dict,
    dataset_name: str,
    beam_width: int = 3,
    max_depth: int = 4,
    mi_alpha: float = 0.7,
    mi_beta: float = 0.3,
    reranker_model=None,
) -> dict:
    """
    Execute MI-RA-ToT multi-hop reasoning.

    Args:
        query: Question to answer
        retriever_config: Retriever endpoint config
        llm_config: LLM endpoint config
        dataset_name: Dataset name
        beam_width: Beam width (default 3)
        max_depth: Max tree depth (default 4)
        mi_alpha: Relevance weight (default 0.7)
        mi_beta: Novelty weight (default 0.3)
        reranker_model: Optional reranker for MI scoring (default None)

    Returns:
        Dict with 'answer', 'chain', 'contexts'
    """
    tot_searcher = BeamSearchToT(
        retriever_config=retriever_config,
        llm_config=llm_config,
        dataset_name=dataset_name,
        beam_width=beam_width,
        max_depth=max_depth,
        mi_alpha=mi_alpha,
        mi_beta=mi_beta,
        reranker_model=reranker_model,
    )

    result = tot_searcher.search(query)
    return result


# Example usage
if __name__ == "__main__":
    # Test with simple question
    test_query = "Who is the director of Titanic?"

    retriever_config = {"host": "localhost", "port": 8002}
    llm_config = {"host": "localhost", "port": 8000}

    print("=" * 80)
    print("Testing MI-RA-ToT")
    print("=" * 80)
    print(f"Question: {test_query}")
    print(f"Beam Width: 3, Max Depth: 4")
    print("=" * 80)

    result = execute_tot_multihop(
        query=test_query,
        retriever_config=retriever_config,
        llm_config=llm_config,
        dataset_name="hotpotqa",
        beam_width=2,  # Smaller for testing
        max_depth=2,
    )

    print("\nAnswer:", result['answer'])
    print("\nReasoning Chain:")
    print(result['chain'])
    print("\nContexts:", len(result['contexts']))
