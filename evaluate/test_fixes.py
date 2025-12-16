#!/usr/bin/env python3
"""
Quick test script to verify the fixes for answer truncation issue.
Tests on a small subset of squad dataset.
"""

import subprocess
import sys
import os


def run_command(description, command):
    """Run a command and handle errors."""
    print(f"\n{'='*60}")
    print(f"{description}")
    print(f"{'='*60}")

    try:
        result = subprocess.run(
            command,
            shell=True,
            check=True,
            text=True,
            capture_output=False
        )
        print(f"✓ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ {description} failed with error code {e.returncode}")
        return False


def main():
    print("\n" + "="*60)
    print("Testing Fixes for Answer Truncation Issue")
    print("="*60)
    print("\nThis script will:")
    print("  1. Regenerate predictions with fixed code")
    print("  2. Evaluate the new predictions")
    print("  3. Run diagnostics to check improvements")
    print("  4. Analyze retrieval quality")
    print()

    # Change to project root
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    # Step 1: Generate predictions
    success = run_command(
        "Step 1: Running Stage 2 (Generation) on squad dataset",
        "python evaluate/stage2_generate.py --config evaluate/config.yaml --datasets squad"
    )

    if not success:
        print("\n✗ Generation failed. Please check:")
        print("  - LLM server is running (http://localhost:8000)")
        print("  - Retriever server is running (http://localhost:8001)")
        sys.exit(1)

    # Step 2: Evaluate
    success = run_command(
        "Step 2: Running Stage 3 (Evaluation) on squad dataset",
        "python evaluate/stage3_evaluate.py --config evaluate/config.yaml --datasets squad"
    )

    if not success:
        print("\n✗ Evaluation failed. Check the error above.")
        sys.exit(1)

    # Step 3: Diagnostics
    success = run_command(
        "Step 3: Running diagnostics to check for improvements",
        "python evaluate/diagnose_predictions.py --config evaluate/config.yaml --datasets squad --samples 20"
    )

    # Step 4: Retrieval Analysis
    success = run_command(
        "Step 4: Analyzing retrieval quality",
        "python evaluate/analyze_retrieval.py --config evaluate/config.yaml --datasets squad --samples 20"
    )

    # Summary
    print("\n" + "="*60)
    print("✓ Test Complete!")
    print("="*60)
    print()
    print("Expected improvements:")
    print("  ✓ F1 score should increase from ~0.10 to >0.30")
    print("  ✓ Truncated answers should decrease from 60%+ to <10%")
    print("  ✓ Check the diagnostic output above for details")
    print()
    print("Retrieval Analysis:")
    print("  - Shows if predictions matching Wikipedia titles are retrieval errors")
    print("  - Displays relevance of retrieved documents to questions")
    print("  - Identifies empty retrievals and irrelevant documents")
    print()
    print("If F1 is still low (<0.20), check:")
    print("  1. LLM server logs for errors")
    print("  2. evaluate/outputs/stage2_predictions/squad_predictions.json")
    print("  3. evaluate/outputs/stage2_predictions/squad_predictions_chains.txt")
    print("  4. evaluate/outputs/stage2_predictions/squad_predictions_contexts.json")
    print()


if __name__ == "__main__":
    main()
