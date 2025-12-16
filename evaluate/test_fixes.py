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

    # Step 4: Retrieval Quality Analysis (compare retrieved vs gold contexts)
    success = run_command(
        "Step 4: Analyzing retrieval quality (Retrieved vs Gold Contexts)",
        "python evaluate/analyze_retrieval_quality.py --config evaluate/config.yaml --datasets squad --sample-size 50 --num-samples 3"
    )

    # Summary
    print("\n" + "="*60)
    print("✓ Test Complete!")
    print("="*60)
    print()
    print("Expected improvements:")
    print("  ✓ EM (Exact Match) should be >0.10")
    print("  ✓ F1 score should increase from ~0.10 to >0.30")
    print("  ✓ ACC (Accuracy) should be >0.40 (checks if prediction contains gold answer)")
    print("  ✓ Recall should be >0.50 (checks if gold answer tokens are in prediction)")
    print("  ✓ Truncated answers should decrease from 60%+ to <10%")
    print("  ✓ Check the diagnostic output above for details")
    print()
    print("Retrieval Quality Analysis:")
    print("  - Precision: 检索到的上下文中有多少是标准答案中的上下文")
    print("  - Recall:    标准答案上下文中有多少被成功检索到")
    print("  - Hit@K:     至少检索到一个标准上下文的比例")
    print("  - 如果 Recall 很低（<0.3），说明检索器可能需要优化")
    print("  - 如果 Precision 很低但 Recall 高，说明检索到了太多无关文档")
    print("  - 详细报告保存在: evaluate/outputs/analysis/")
    print()
    print("If F1 is still low (<0.20), check:")
    print("  1. LLM server logs for errors")
    print("  2. evaluate/outputs/stage2_predictions/squad_predictions.json")
    print("  3. evaluate/outputs/stage2_predictions/squad_predictions_chains.txt")
    print("  4. evaluate/outputs/stage2_predictions/squad_predictions_contexts.json")
    print()


if __name__ == "__main__":
    main()
