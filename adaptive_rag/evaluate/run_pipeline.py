"""
Adaptive-RAG Evaluation Pipeline Runner
Run all stages (classify, generate, evaluate) in sequence.

Usage:
    python -m adaptive_rag.evaluate.run_pipeline
    python -m adaptive_rag.evaluate.run_pipeline --stages 1 2 3
    python -m adaptive_rag.evaluate.run_pipeline --datasets squad hotpotqa
"""

import os
import sys
import time
import shutil
import argparse
import requests
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from evaluate.utils.config_loader import ConfigLoader


def clean_outputs(config: dict, stages: list, datasets: list = None):
    """Clean output files for specified stages and datasets."""
    print("\n" + "="*60)
    print("CLEANING OUTPUT FILES")
    print("="*60)

    stage1_dir = config['outputs']['stage1_dir']
    stage2_dir = config['outputs']['stage2_dir']
    stage3_dir = config['outputs']['stage3_dir']

    if datasets is None:
        datasets = config['datasets']

    cleaned_count = 0

    # Clean stage 1 outputs
    if 1 in stages:
        for dataset_name in datasets:
            file_path = os.path.join(stage1_dir, f'{dataset_name}_classifications.jsonl')
            if os.path.exists(file_path):
                os.remove(file_path)
                print(f"  Removed: {file_path}")
                cleaned_count += 1

    # Clean stage 2 outputs
    if 2 in stages:
        for dataset_name in datasets:
            # Main predictions file
            file_path = os.path.join(stage2_dir, f'{dataset_name}_predictions.json')
            if os.path.exists(file_path):
                os.remove(file_path)
                print(f"  Removed: {file_path}")
                cleaned_count += 1

            # Chains file
            chains_path = os.path.join(stage2_dir, f'{dataset_name}_predictions_chains.txt')
            if os.path.exists(chains_path):
                os.remove(chains_path)
                print(f"  Removed: {chains_path}")
                cleaned_count += 1

            # Contexts file
            contexts_path = os.path.join(stage2_dir, f'{dataset_name}_predictions_contexts.json')
            if os.path.exists(contexts_path):
                os.remove(contexts_path)
                print(f"  Removed: {contexts_path}")
                cleaned_count += 1

    # Clean stage 3 outputs
    if 3 in stages:
        metrics_path = os.path.join(stage3_dir, 'overall_metrics.json')
        if os.path.exists(metrics_path):
            os.remove(metrics_path)
            print(f"  Removed: {metrics_path}")
            cleaned_count += 1

        report_path = os.path.join(stage3_dir, 'detailed_report.txt')
        if os.path.exists(report_path):
            os.remove(report_path)
            print(f"  Removed: {report_path}")
            cleaned_count += 1

    if cleaned_count == 0:
        print("  No files to clean")
    else:
        print(f"\n  Cleaned {cleaned_count} files")


def check_services(config: dict) -> bool:
    """Check if required services are available."""
    print("\n" + "="*60)
    print("SERVICE CHECK")
    print("="*60)

    all_ok = True

    # Check LLM service
    llm_url = f"http://{config['llm']['server_host']}:{config['llm']['server_port']}/generate"
    try:
        response = requests.get(
            llm_url,
            params={'prompt': 'test', 'max_length': 10},
            timeout=10
        )
        print(f"  LLM service ({llm_url}): OK")
    except Exception as e:
        print(f"  LLM service ({llm_url}): FAILED - {e}")
        all_ok = False

    # Check Retriever service
    retriever_url = f"http://{config['retriever']['host']}:{config['retriever']['port']}/retrieve/"
    try:
        response = requests.post(
            retriever_url,
            json={
                "retrieval_method": "retrieve_from_elasticsearch",
                "query_text": "test",
                "max_hits_count": 1,
                "corpus_name": "wiki",
                "document_type": "title_paragraph_text",
                "retrieval_backend": "hybrid"
            },
            timeout=10
        )
        print(f"  Retriever service ({retriever_url}): OK")
    except Exception as e:
        print(f"  Retriever service ({retriever_url}): FAILED - {e}")
        all_ok = False

    return all_ok


def run_stage1(config_path: str, datasets: list = None):
    """Run Stage 1: Classification."""
    from adaptive_rag.evaluate.stage1_classify import AdaptiveStage1Classifier

    config = ConfigLoader.load_config(config_path)
    classifier = AdaptiveStage1Classifier(config)
    classifier.run(datasets=datasets)


def run_stage2(config_path: str, datasets: list = None):
    """Run Stage 2: Generation."""
    from adaptive_rag.evaluate.stage2_generate import AdaptiveStage2Generator

    config = ConfigLoader.load_config(config_path)
    generator = AdaptiveStage2Generator(config)
    generator.run(datasets=datasets)


def run_stage3(config_path: str, datasets: list = None):
    """Run Stage 3: Evaluation."""
    from adaptive_rag.evaluate.stage3_evaluate import AdaptiveStage3Evaluator

    config = ConfigLoader.load_config(config_path)
    evaluator = AdaptiveStage3Evaluator(config)
    evaluator.run(datasets=datasets)


def main():
    parser = argparse.ArgumentParser(description="Adaptive-RAG Evaluation Pipeline")
    parser.add_argument('--config', default='adaptive_rag/evaluate/config.yaml',
                       help='Path to config file')
    parser.add_argument('--stages', nargs='+', type=int, default=[1, 2, 3],
                       help='Stages to run (default: 1 2 3)')
    parser.add_argument('--datasets', nargs='+',
                       help='Datasets to process (default: all in config)')
    parser.add_argument('--skip-service-check', action='store_true',
                       help='Skip service availability check')
    parser.add_argument('--clean', action='store_true',
                       help='Clean output files before running (remove old results)')

    args = parser.parse_args()

    print("\n" + "="*60)
    print("ADAPTIVE-RAG EVALUATION PIPELINE")
    print("="*60)
    print(f"Config: {args.config}")
    print(f"Stages: {args.stages}")
    print(f"Datasets: {args.datasets or 'all'}")

    # Load config
    config = ConfigLoader.load_config(args.config)

    # Clean outputs if requested
    if args.clean:
        clean_outputs(config, args.stages, args.datasets)

    # Service check (if not skipped and running stages that need services)
    if not args.skip_service_check and (2 in args.stages):
        if not check_services(config):
            print("\nServices not available. Please start the services and try again.")
            print("Or use --skip-service-check to skip this check (Stage 2 will fail).")
            sys.exit(1)

    # Run stages
    start_time = time.time()

    if 1 in args.stages:
        print("\n" + "="*60)
        print("RUNNING STAGE 1: CLASSIFICATION")
        print("="*60)
        stage_start = time.time()
        run_stage1(args.config, args.datasets)
        print(f"\nStage 1 completed in {time.time() - stage_start:.2f}s")

    if 2 in args.stages:
        print("\n" + "="*60)
        print("RUNNING STAGE 2: GENERATION")
        print("="*60)
        stage_start = time.time()
        run_stage2(args.config, args.datasets)
        print(f"\nStage 2 completed in {time.time() - stage_start:.2f}s")

    if 3 in args.stages:
        print("\n" + "="*60)
        print("RUNNING STAGE 3: EVALUATION")
        print("="*60)
        stage_start = time.time()
        run_stage3(args.config, args.datasets)
        print(f"\nStage 3 completed in {time.time() - stage_start:.2f}s")

    total_time = time.time() - start_time
    print("\n" + "="*60)
    print("PIPELINE COMPLETE")
    print("="*60)
    print(f"Total time: {total_time:.2f}s ({total_time/60:.2f} min)")


if __name__ == '__main__':
    main()
