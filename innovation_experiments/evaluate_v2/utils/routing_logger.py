"""
Routing Logger for Cascading Dynamic Routing (Innovation 2)

Tracks cascade routing decisions for analysis and paper metrics.
Logs: initial strategy, confidence score, final strategy, cascade flag.
"""

import json
import csv
from pathlib import Path
from typing import Dict, List, Optional
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


class RoutingLogger:
    """
    Logs routing decisions for cascade analysis.

    Tracks each question's journey through the cascade:
    - Initial strategy from classifier
    - Confidence score from verification
    - Whether cascade was triggered
    - Final strategy used
    """

    def __init__(self):
        """Initialize the routing logger."""
        self.routing_decisions: Dict[str, Dict] = {}
        self.statistics = defaultdict(int)

    def log_decision(
        self,
        question_id: str,
        initial_action: str,
        confidence: float,
        final_action: str,
        cascaded: bool,
        question_text: Optional[str] = None,
        dataset: Optional[str] = None,
    ):
        """
        Log a routing decision for a single question.

        Args:
            question_id: Unique question identifier
            initial_action: Strategy predicted by classifier (e.g., 'S-Hybrid')
            confidence: Confidence score from verifier (0-1)
            final_action: Actual strategy executed (e.g., 'M-ToT' if cascaded)
            cascaded: Whether cascade was triggered
            question_text: Optional question text for debugging
            dataset: Optional dataset name
        """
        self.routing_decisions[question_id] = {
            "initial_action": initial_action,
            "confidence": confidence,
            "final_action": final_action,
            "cascaded": cascaded,
            "question_text": question_text,
            "dataset": dataset,
        }

        # Update statistics
        self.statistics["total"] += 1
        self.statistics[f"initial_{initial_action}"] += 1
        self.statistics[f"final_{final_action}"] += 1

        if cascaded:
            self.statistics["total_cascaded"] += 1
            self.statistics[f"cascaded_from_{initial_action}"] += 1

        # Log to console (only cascade decisions at INFO, rest at DEBUG)
        if cascaded:
            logger.info(
                f"[Cascade] {question_id} | {initial_action} → {final_action} | Confidence: {confidence:.3f}"
            )
        else:
            logger.debug(
                f"[Routing] {question_id} | {initial_action} → {final_action} | Confidence: {confidence:.3f}"
            )

    def save(self, output_path: str):
        """
        Save routing decisions to CSV file.

        Args:
            output_path: Path to output CSV file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Write to CSV
        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)

            # Header
            writer.writerow(
                [
                    "question_id",
                    "initial_action",
                    "confidence",
                    "final_action",
                    "cascaded",
                    "dataset",
                    "question_text",
                ]
            )

            # Rows
            for qid, decision in self.routing_decisions.items():
                writer.writerow(
                    [
                        qid,
                        decision["initial_action"],
                        f"{decision['confidence']:.4f}",
                        decision["final_action"],
                        decision["cascaded"],
                        decision.get("dataset", ""),
                        decision.get("question_text", ""),
                    ]
                )

        logger.info(f"Saved routing log to: {output_path}")
        logger.info(f"  Total decisions: {len(self.routing_decisions)}")
        logger.info(
            f"  Cascaded: {self.statistics.get('total_cascaded', 0)} ({self.statistics.get('total_cascaded', 0) / max(1, self.statistics['total']) * 100:.1f}%)"
        )

    def save_json(self, output_path: str):
        """
        Save routing decisions to JSON file.

        Args:
            output_path: Path to output JSON file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "decisions": self.routing_decisions,
            "statistics": dict(self.statistics),
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        logger.info(f"Saved routing log (JSON) to: {output_path}")

    def get_statistics(self) -> Dict:
        """
        Get statistics about routing decisions.

        Returns:
            Dict with cascade statistics
        """
        total = self.statistics.get("total", 0)
        if total == 0:
            return {}

        cascaded = self.statistics.get("total_cascaded", 0)
        cascade_rate = cascaded / total

        # Per-action cascade rates
        cascade_by_action = {}
        for action in ["Z", "S-Sparse", "S-Dense", "S-Hybrid"]:
            initial_count = self.statistics.get(f"initial_{action}", 0)
            cascaded_count = self.statistics.get(f"cascaded_from_{action}", 0)
            if initial_count > 0:
                cascade_by_action[action] = cascaded_count / initial_count

        return {
            "total_questions": total,
            "total_cascaded": cascaded,
            "cascade_rate": cascade_rate,
            "cascade_by_initial_action": cascade_by_action,
        }

    def get_confidence_distribution(self, bins: int = 10) -> Dict[str, List[int]]:
        """
        Get distribution of confidence scores.

        Args:
            bins: Number of bins for histogram (default: 10)

        Returns:
            Dict with histogram data (bin_edges, counts)
        """
        if not self.routing_decisions:
            return {"bin_edges": [], "counts": []}

        import numpy as np

        confidences = [d["confidence"] for d in self.routing_decisions.values()]

        counts, bin_edges = np.histogram(confidences, bins=bins, range=(0.0, 1.0))

        return {
            "bin_edges": bin_edges.tolist(),
            "counts": counts.tolist(),
        }

    def get_cascade_examples(
        self, category: str = "cascaded", limit: int = 10
    ) -> List[Dict]:
        """
        Get example questions for a specific category.

        Args:
            category: 'cascaded' or 'direct'
            limit: Maximum number of examples to return

        Returns:
            List of example routing decisions
        """
        examples = []

        for qid, decision in self.routing_decisions.items():
            if category == "cascaded" and decision["cascaded"]:
                examples.append({"question_id": qid, **decision})
            elif category == "direct" and not decision["cascaded"]:
                examples.append({"question_id": qid, **decision})

            if len(examples) >= limit:
                break

        return examples

    def merge(self, other: "RoutingLogger"):
        """
        Merge routing decisions from another logger.

        Args:
            other: Another RoutingLogger instance
        """
        # Merge decisions
        for qid, decision in other.routing_decisions.items():
            if qid in self.routing_decisions:
                logger.warning(f"Duplicate question_id {qid}, overwriting")
            self.routing_decisions[qid] = decision

        # Merge statistics
        for key, value in other.statistics.items():
            self.statistics[key] += value

        logger.info(f"Merged routing log: {len(other.routing_decisions)} decisions added")

    def reset(self):
        """Clear all routing decisions and statistics."""
        self.routing_decisions.clear()
        self.statistics.clear()
        logger.info("Routing logger reset")

    def print_summary(self):
        """Print a summary of routing decisions to console."""
        stats = self.get_statistics()

        print("\n" + "=" * 80)
        print("Routing Logger Summary")
        print("=" * 80)
        print(f"Total questions: {stats.get('total_questions', 0)}")
        print(
            f"Cascaded: {stats.get('total_cascaded', 0)} ({stats.get('cascade_rate', 0) * 100:.1f}%)"
        )
        print(f"Direct: {stats.get('total_questions', 0) - stats.get('total_cascaded', 0)}")
        print()

        print("Cascade rate by initial action:")
        for action, rate in stats.get("cascade_by_initial_action", {}).items():
            print(f"  {action}: {rate * 100:.1f}%")
        print("=" * 80)


# Example usage and testing
if __name__ == "__main__":
    logger = RoutingLogger()

    # Simulate some routing decisions
    print("Simulating routing decisions...")

    # Case 1: S-Hybrid with high confidence → No cascade
    logger.log_decision(
        question_id="q001",
        initial_action="S-Hybrid",
        confidence=0.85,
        final_action="S-Hybrid",
        cascaded=False,
        question_text="What is the capital of France?",
        dataset="squad",
    )

    # Case 2: S-Hybrid with low confidence → Cascade to M-ToT
    logger.log_decision(
        question_id="q002",
        initial_action="S-Hybrid",
        confidence=0.42,
        final_action="M-ToT",
        cascaded=True,
        question_text="Who directed the movie whose main actor was born in the same city as the inventor of the telephone?",
        dataset="hotpotqa",
    )

    # Case 3: S-Sparse with low confidence → Cascade
    logger.log_decision(
        question_id="q003",
        initial_action="S-Sparse",
        confidence=0.38,
        final_action="M-ToT",
        cascaded=True,
        question_text="What is the relationship between X and Y?",
        dataset="musique",
    )

    # Case 4: S-Dense with high confidence → No cascade
    logger.log_decision(
        question_id="q004",
        initial_action="S-Dense",
        confidence=0.91,
        final_action="S-Dense",
        cascaded=False,
        question_text="Explain the concept of photosynthesis.",
        dataset="nq",
    )

    # Case 5: M strategy → Never cascades (already robust)
    logger.log_decision(
        question_id="q005",
        initial_action="M",
        confidence=1.0,  # M strategies don't get verified
        final_action="M",
        cascaded=False,
        question_text="Complex multi-hop question.",
        dataset="2wikimultihopqa",
    )

    # Print summary
    logger.print_summary()

    # Save to files
    output_dir = Path("test_outputs")
    output_dir.mkdir(exist_ok=True)

    logger.save(output_dir / "routing_log.csv")
    logger.save_json(output_dir / "routing_log.json")

    # Get examples
    print("\n" + "=" * 80)
    print("Cascade Examples:")
    print("=" * 80)
    cascade_examples = logger.get_cascade_examples(category="cascaded", limit=3)
    for ex in cascade_examples:
        print(
            f"  {ex['question_id']}: {ex['initial_action']} → {ex['final_action']} (conf={ex['confidence']:.2f})"
        )

    print("\n" + "=" * 80)
    print("Direct Examples:")
    print("=" * 80)
    direct_examples = logger.get_cascade_examples(category="direct", limit=3)
    for ex in direct_examples:
        print(
            f"  {ex['question_id']}: {ex['initial_action']} → {ex['final_action']} (conf={ex['confidence']:.2f})"
        )

    # Get confidence distribution
    print("\n" + "=" * 80)
    print("Confidence Distribution:")
    print("=" * 80)
    dist = logger.get_confidence_distribution(bins=5)
    for i, (edge, count) in enumerate(zip(dist["bin_edges"][:-1], dist["counts"])):
        next_edge = dist["bin_edges"][i + 1]
        print(f"  [{edge:.1f}, {next_edge:.1f}): {count} questions")
