"""
Confidence Verifier for Cascading Dynamic Routing (Innovation 2)

Performs posterior verification of answers by scoring them against retrieved contexts.
Used to determine when to cascade from initial strategies to more robust MI-RA-ToT.
"""

from typing import List, Dict, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ConfidenceVerifier:
    """
    Verifies answer confidence using cross-encoder scoring.

    Given a (question, answer, contexts) tuple, computes a confidence score
    by checking if the answer is well-supported by the retrieved documents.

    Low confidence triggers cascading to a more robust strategy.
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        device: str = "cuda",
        threshold: float = 0.6,
        max_contexts: int = 3,
    ):
        """
        Initialize the confidence verifier.

        Args:
            model_name: HuggingFace model name for cross-encoder
            device: Device for inference (cuda/cpu)
            threshold: Confidence threshold for cascading (default: 0.6)
            max_contexts: Number of top contexts to verify against (default: 3)
        """
        self.model_name = model_name
        self.device = device
        self.threshold = threshold
        self.max_contexts = max_contexts
        self.model = None

        # Load model lazily
        self._load_model()

    def _load_model(self):
        """Lazy load cross-encoder model."""
        if self.model is None:
            try:
                import os
                model_path = self.model_name
                # 必须转为绝对路径，否则 transformers 会把多层相对路径当作 HuggingFace repo ID
                if not os.path.isabs(model_path):
                    # 从项目根目录解析
                    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
                    model_path = os.path.join(project_root, model_path)

                from sentence_transformers import CrossEncoder
                print(f"[ConfidenceVerifier] Loading model: {model_path} (device={self.device})")
                self.model = CrossEncoder(model_path, device=self.device)
                print(f"[ConfidenceVerifier] Model loaded successfully")
            except Exception as e:
                print(f"[ConfidenceVerifier] Failed to load model: {e}")
                import traceback
                traceback.print_exc()
                logger.warning("Confidence verification will be disabled")
                self.model = None

    def verify(
        self,
        question: str,
        answer: str,
        contexts: List[Dict],
        return_detailed: bool = False,
    ) -> float:
        """
        Verify answer confidence against retrieved contexts.

        Args:
            question: The original question
            answer: The generated answer
            contexts: List of retrieved documents (each with 'paragraph_text' key)
            return_detailed: If True, return dict with scores per context

        Returns:
            Confidence score (0-1) or detailed dict if return_detailed=True
        """
        if self.model is None:
            # 模型未加载时，返回高置信度以跳过级联，避免所有问题被错误地全部级联
            logger.warning("Confidence verifier model not loaded, returning high score 0.9 (skip cascade)")
            return 0.9 if not return_detailed else {"confidence": 0.9, "scores": []}

        if not contexts:
            logger.warning("No contexts provided for verification, returning low confidence 0.0")
            return 0.0 if not return_detailed else {"confidence": 0.0, "scores": []}

        if not answer or len(answer.strip()) < 2:
            logger.warning("Empty or very short answer, returning low confidence 0.0")
            return 0.0 if not return_detailed else {"confidence": 0.0, "scores": []}

        # Create QA pair for verification
        qa_text = f"Question: {question}\nAnswer: {answer}"

        # Score against top contexts
        scores = []
        top_contexts = contexts[: self.max_contexts]

        for idx, ctx in enumerate(top_contexts):
            context_text = ctx.get("paragraph_text", "")
            if not context_text:
                continue

            try:
                # Cross-encoder scoring: How well does context support the QA pair?
                score = self.model.predict([(qa_text, context_text)])[0]
                scores.append(float(score))
            except Exception as e:
                logger.warning(f"Failed to score context {idx}: {e}")
                continue

        # Compute confidence: Use max score (best supporting document)
        if not scores:
            confidence = 0.0
        else:
            confidence = max(scores)

        # Normalize to [0, 1] range (cross-encoder scores can be outside this range)
        confidence = max(0.0, min(1.0, confidence))

        if return_detailed:
            return {
                "confidence": confidence,
                "scores": scores,
                "num_contexts": len(top_contexts),
                "max_score": confidence,
                "avg_score": sum(scores) / len(scores) if scores else 0.0,
            }
        else:
            return confidence

    def should_cascade(
        self,
        confidence: float,
        strategy: Optional[str] = None,
    ) -> bool:
        """
        Determine if we should cascade to a more robust strategy.

        Args:
            confidence: Confidence score from verify()
            strategy: Optional strategy name (e.g., 'S-Sparse', 'S-Hybrid')
                     Some strategies may never cascade (e.g., 'M' is already robust)

        Returns:
            True if should cascade, False otherwise
        """
        # Never cascade from M strategies (already most robust)
        if strategy and strategy.startswith("M"):
            return False

        # Cascade if confidence below threshold
        return confidence < self.threshold

    def batch_verify(
        self,
        questions: List[str],
        answers: List[str],
        contexts_list: List[List[Dict]],
    ) -> List[float]:
        """
        Batch verification for multiple QA pairs.

        Args:
            questions: List of questions
            answers: List of answers
            contexts_list: List of context lists (one per question)

        Returns:
            List of confidence scores
        """
        if len(questions) != len(answers) or len(questions) != len(contexts_list):
            raise ValueError("Questions, answers, and contexts must have same length")

        confidences = []
        for q, a, ctxs in zip(questions, answers, contexts_list):
            confidence = self.verify(q, a, ctxs)
            confidences.append(confidence)

        return confidences

    def get_statistics(self, confidences: List[float]) -> Dict[str, float]:
        """
        Compute statistics for a batch of confidence scores.

        Args:
            confidences: List of confidence scores

        Returns:
            Dict with statistics (mean, std, min, max, below_threshold_rate)
        """
        if not confidences:
            return {
                "mean": 0.0,
                "std": 0.0,
                "min": 0.0,
                "max": 0.0,
                "below_threshold_rate": 0.0,
            }

        import numpy as np

        return {
            "mean": float(np.mean(confidences)),
            "std": float(np.std(confidences)),
            "min": float(np.min(confidences)),
            "max": float(np.max(confidences)),
            "below_threshold_rate": sum(c < self.threshold for c in confidences)
            / len(confidences),
        }

    def update_threshold(self, new_threshold: float):
        """
        Update the confidence threshold dynamically.

        Args:
            new_threshold: New threshold value (0-1)
        """
        if not 0.0 <= new_threshold <= 1.0:
            raise ValueError(f"Threshold must be in [0, 1], got {new_threshold}")

        old_threshold = self.threshold
        self.threshold = new_threshold
        logger.info(
            f"Updated confidence threshold: {old_threshold:.2f} → {new_threshold:.2f}"
        )


# Example usage and testing
if __name__ == "__main__":
    verifier = ConfidenceVerifier(threshold=0.6)

    # Test case 1: Well-supported answer
    question1 = "Who is the director of Titanic?"
    answer1 = "James Cameron"
    contexts1 = [
        {
            "paragraph_text": "Titanic is a 1997 American epic romance and disaster film directed by James Cameron.",
            "title": "Titanic (1997 film)",
        },
        {
            "paragraph_text": "James Cameron is a Canadian filmmaker known for directing Titanic and Avatar.",
            "title": "James Cameron",
        },
    ]

    print("=" * 80)
    print("Test Case 1: Well-supported answer")
    print("=" * 80)
    result1 = verifier.verify(question1, answer1, contexts1, return_detailed=True)
    print(f"Question: {question1}")
    print(f"Answer: {answer1}")
    print(f"Confidence: {result1['confidence']:.3f}")
    print(f"Scores per context: {result1['scores']}")
    print(f"Should cascade: {verifier.should_cascade(result1['confidence'])}")
    print()

    # Test case 2: Poorly-supported answer
    question2 = "What is the capital of France?"
    answer2 = "London"  # Wrong answer
    contexts2 = [
        {
            "paragraph_text": "Paris is the capital and most populous city of France.",
            "title": "Paris",
        },
        {
            "paragraph_text": "London is the capital of the United Kingdom and England.",
            "title": "London",
        },
    ]

    print("=" * 80)
    print("Test Case 2: Poorly-supported answer (wrong)")
    print("=" * 80)
    result2 = verifier.verify(question2, answer2, contexts2, return_detailed=True)
    print(f"Question: {question2}")
    print(f"Answer: {answer2}")
    print(f"Confidence: {result2['confidence']:.3f}")
    print(f"Scores per context: {result2['scores']}")
    print(f"Should cascade: {verifier.should_cascade(result2['confidence'])}")
    print()

    # Test case 3: Unsupported answer (no relevant contexts)
    question3 = "What is quantum entanglement?"
    answer3 = "A phenomenon where particles become correlated."
    contexts3 = [
        {
            "paragraph_text": "The Eiffel Tower is located in Paris, France.",
            "title": "Eiffel Tower",
        },
        {
            "paragraph_text": "Soccer is a popular sport played worldwide.",
            "title": "Soccer",
        },
    ]

    print("=" * 80)
    print("Test Case 3: Answer with irrelevant contexts")
    print("=" * 80)
    result3 = verifier.verify(question3, answer3, contexts3, return_detailed=True)
    print(f"Question: {question3}")
    print(f"Answer: {answer3}")
    print(f"Confidence: {result3['confidence']:.3f}")
    print(f"Scores per context: {result3['scores']}")
    print(f"Should cascade: {verifier.should_cascade(result3['confidence'])}")
    print()

    # Batch test
    print("=" * 80)
    print("Batch Test: Statistics")
    print("=" * 80)
    all_confidences = [
        result1["confidence"],
        result2["confidence"],
        result3["confidence"],
    ]
    stats = verifier.get_statistics(all_confidences)
    print(f"Statistics for {len(all_confidences)} verifications:")
    print(f"  Mean: {stats['mean']:.3f}")
    print(f"  Std: {stats['std']:.3f}")
    print(f"  Min: {stats['min']:.3f}")
    print(f"  Max: {stats['max']:.3f}")
    print(
        f"  Below threshold ({verifier.threshold}): {stats['below_threshold_rate']:.1%}"
    )
