"""
Query Analyzer for Adaptive Retrieval (Innovation 1)

Analyzes queries to compute:
1. Lexical specificity (entity density)
2. Semantic abstractness
3. Dynamic hybrid retrieval weights
"""

import re
from typing import Dict, List
import logging
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class QueryAnalyzer:
    """
    Analyzes queries and computes dynamic weights for hybrid retrieval.

    Uses NER + semantic heuristics to determine:
    - lexical_score: Entity density (0-1)
    - semantic_score: Abstractness level (0-1)

    Then maps to optimal weights for {bm25, splade, dense}.
    """

    def __init__(self, device="cuda", ner_model="en_core_web_sm"):
        """
        Initialize the query analyzer.

        Args:
            device: Device for model inference (cuda/cpu)
            ner_model: spaCy model name for NER
        """
        self.device = device
        self.ner_model_name = ner_model
        self.nlp = None

        # Abstract question keywords
        self.abstract_keywords = {
            'how', 'why', 'relationship', 'significance', 'reason', 'effect',
            'impact', 'influence', 'consequence', 'meaning', 'purpose',
            'explain', 'describe', 'compare', 'contrast', 'analyze',
            'evaluate', 'discuss', 'difference', 'similarity', 'connection'
        }

        # Concrete question keywords
        self.concrete_keywords = {
            'who', 'what', 'when', 'where', 'which', 'name',
            'date', 'year', 'location', 'place', 'person', 'title'
        }

        # Load NER model lazily
        self._load_ner_model()

    def _load_ner_model(self):
        """Lazy load spaCy NER model."""
        if self.nlp is None:
            try:
                import spacy
                logger.info(f"Loading spaCy model: {self.ner_model_name}")
                self.nlp = spacy.load(self.ner_model_name)
                logger.info("spaCy model loaded successfully")
            except Exception as e:
                logger.warning(f"Failed to load spaCy model: {e}")
                logger.warning("Falling back to heuristic-based NER")
                self.nlp = None

    def _detect_entities_spacy(self, query: str) -> List[str]:
        """Detect named entities using spaCy NER."""
        if self.nlp is None:
            return []

        doc = self.nlp(query)
        entities = [ent.text for ent in doc.ents]
        return entities

    def _detect_entities_heuristic(self, query: str) -> List[str]:
        """
        Heuristic-based entity detection (fallback).

        Detects:
        - Capitalized phrases (proper nouns)
        - Quoted strings
        - Numbers with context (years, dates)
        """
        entities = []

        # Quoted strings
        quoted = re.findall(r'"([^"]+)"', query)
        entities.extend(quoted)

        # Capitalized sequences (likely proper nouns)
        # Match sequences of capitalized words
        capitalized = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', query)
        entities.extend(capitalized)

        # Numbers (years, dates, IDs)
        numbers = re.findall(r'\b\d{4}\b|\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b', query)
        entities.extend(numbers)

        return list(set(entities))  # Remove duplicates

    def _calculate_lexical_specificity(self, query: str, entities: List[str]) -> float:
        """
        Calculate lexical specificity score (0-1).

        Higher score = more entity-dense = favor BM25/SPLADE
        Lower score = more general = favor Dense

        Formula: entity_ratio * avg_entity_length_bonus
        """
        tokens = query.split()
        if not tokens:
            return 0.5  # Neutral

        # Entity ratio: What fraction of tokens are part of entities?
        entity_token_count = sum(len(ent.split()) for ent in entities)
        entity_ratio = min(entity_token_count / len(tokens), 1.0)

        # Bonus for longer entities (more specific)
        if entities:
            avg_entity_length = sum(len(ent.split()) for ent in entities) / len(entities)
            length_bonus = min(avg_entity_length / 3.0, 1.0)  # Cap at 3+ tokens
        else:
            length_bonus = 0.0

        # Combined score
        lexical_score = 0.7 * entity_ratio + 0.3 * length_bonus

        return min(lexical_score, 1.0)

    def _calculate_semantic_abstractness(self, query: str) -> float:
        """
        Calculate semantic abstractness score (0-1).

        Higher score = more abstract = favor Dense
        Lower score = more concrete = favor BM25/SPLADE

        Based on presence of abstract vs concrete keywords.
        """
        query_lower = query.lower()
        tokens = set(query_lower.split())

        # Count abstract keywords
        abstract_count = len(tokens.intersection(self.abstract_keywords))

        # Count concrete keywords
        concrete_count = len(tokens.intersection(self.concrete_keywords))

        # Calculate abstractness
        if abstract_count + concrete_count == 0:
            # No clear signal, check for question patterns
            if any(q in query_lower for q in ['how does', 'why does', 'what is the relationship']):
                return 0.8  # Likely abstract
            elif any(q in query_lower for q in ['what is the name', 'who is', 'when did']):
                return 0.2  # Likely concrete
            else:
                return 0.5  # Neutral

        # Abstract ratio
        abstract_ratio = abstract_count / (abstract_count + concrete_count)

        # Boost for multi-word abstract phrases
        if 'how does' in query_lower or 'why does' in query_lower:
            abstract_ratio = min(abstract_ratio + 0.2, 1.0)

        return abstract_ratio

    def analyze(self, query: str) -> Dict[str, float]:
        """
        Analyze a query and return lexical and semantic scores.

        Args:
            query: The input question/query string

        Returns:
            Dict with keys: lexical_score, semantic_score, entity_count
        """
        # Detect entities
        if self.nlp is not None:
            entities = self._detect_entities_spacy(query)
        else:
            entities = self._detect_entities_heuristic(query)

        # Calculate scores
        lexical_score = self._calculate_lexical_specificity(query, entities)
        semantic_score = self._calculate_semantic_abstractness(query)

        return {
            'lexical_score': lexical_score,
            'semantic_score': semantic_score,
            'entity_count': len(entities),
            'entities': entities  # For debugging
        }

    def get_dynamic_weights(self, query: str, temperature: float = 1.0) -> Dict[str, float]:
        """
        Compute dynamic retrieval weights based on query analysis using Temperature-scaled Softmax.

        Args:
            query: The input question/query string
            temperature: Softmax temperature for weight smoothing (default: 1.0)
                        - Higher temperature → more uniform weights
                        - Lower temperature → more extreme weights

        Returns:
            Dict with keys: bm25, splade, dense (weights sum to 1.0)

        Weight Logic (Temperature-scaled Softmax):
        - lexical_score drives sparse retrieval preference (BM25)
        - semantic_score drives dense retrieval preference (Dense)
        - SPLADE bridges both lexical and semantic characteristics

        Formula:
            logit_sparse = α × lexical_score
            logit_dense  = α × semantic_score
            logit_splade = (logit_sparse + logit_dense) / 2
            weights = softmax([logit_sparse, logit_splade, logit_dense] / temperature)
        """
        analysis = self.analyze(query)
        lexical_score = analysis['lexical_score']
        semantic_score = analysis['semantic_score']

        # Scaling factor for logits (controls sensitivity)
        alpha = 2.0

        # Build logits based on query characteristics
        # - High lexical_score (entity-dense) → higher sparse logit
        # - High semantic_score (abstract) → higher dense logit
        logit_sparse = alpha * lexical_score   # BM25: favors entity-dense queries
        logit_dense = alpha * semantic_score   # Dense: favors abstract/semantic queries
        logit_splade = (logit_sparse + logit_dense) / 2  # SPLADE: balanced hybrid

        # Apply temperature-scaled softmax
        logits = np.array([logit_sparse, logit_splade, logit_dense])
        scaled_logits = logits / temperature
        exp_logits = np.exp(scaled_logits - np.max(scaled_logits))  # Numerical stability
        softmax_weights = exp_logits / np.sum(exp_logits)

        bm25_weight = float(softmax_weights[0])
        splade_weight = float(softmax_weights[1])
        dense_weight = float(softmax_weights[2])

        # Determine dominant strategy for logging
        max_idx = np.argmax(softmax_weights)
        strategy_names = ["lexical-dominant", "hybrid-balanced", "semantic-dominant"]
        strategy = strategy_names[max_idx]

        weights = {
            'bm25': bm25_weight,
            'splade': splade_weight,
            'dense': dense_weight,
            'strategy': strategy,
            'analysis': analysis,
            'logits': {
                'sparse': float(logit_sparse),
                'splade': float(logit_splade),
                'dense': float(logit_dense)
            }
        }

        logger.info(f"[QueryAnalyzer] Query: '{query[:60]}...'")
        logger.info(f"[QueryAnalyzer] Lexical: {lexical_score:.2f}, Semantic: {semantic_score:.2f}")
        logger.info(f"[QueryAnalyzer] Logits: Sparse={logit_sparse:.2f}, SPLADE={logit_splade:.2f}, Dense={logit_dense:.2f}")
        logger.info(f"[QueryAnalyzer] Strategy: {strategy}")
        logger.info(f"[QueryAnalyzer] Weights: BM25={bm25_weight:.3f}, SPLADE={splade_weight:.3f}, Dense={dense_weight:.3f}")

        return weights


# Example usage
if __name__ == "__main__":
    analyzer = QueryAnalyzer()

    # Test queries
    test_queries = [
        "Who is the director of the movie Titanic?",  # Entity-dense, concrete
        "What is the significance of the French Revolution?",  # Abstract
        "When did World War II end?",  # Concrete, entity-dense
        "How does photosynthesis work?",  # Abstract
        "Name of the capital of France",  # Concrete
        "What is the relationship between temperature and pressure?",  # Abstract
    ]

    print("=" * 80)
    print("Query Analyzer - Test Results")
    print("=" * 80)

    for query in test_queries:
        print(f"\nQuery: {query}")
        weights = analyzer.get_dynamic_weights(query)
        print(f"  Strategy: {weights['strategy']}")
        print(f"  Logits: Sparse={weights['logits']['sparse']:.2f}, SPLADE={weights['logits']['splade']:.2f}, Dense={weights['logits']['dense']:.2f}")
        print(f"  Weights: BM25={weights['bm25']:.3f}, SPLADE={weights['splade']:.3f}, Dense={weights['dense']:.3f}")
        print(f"  Analysis: Lexical={weights['analysis']['lexical_score']:.2f}, Semantic={weights['analysis']['semantic_score']:.2f}")
        if weights['analysis']['entities']:
            print(f"  Entities: {weights['analysis']['entities']}")
        print("-" * 80)
