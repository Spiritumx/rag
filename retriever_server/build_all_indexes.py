#!/usr/bin/env python
"""
Build all dataset indexes with configurable options.

Usage:
    python build_all_indexes.py --all              # BM25 + Dense + SPLADE
    python build_all_indexes.py --bm25-only        # BM25 only (fastest)
    python build_all_indexes.py --with-dense       # BM25 + Dense
    python build_all_indexes.py --with-splade      # BM25 + SPLADE
    python build_all_indexes.py --datasets wiki hotpotqa  # Specific datasets
"""

import subprocess
import sys
import time
import argparse
from pathlib import Path
from typing import List, Dict

# ANSI color codes
class Colors:
    GREEN = '\033[0;32m'
    BLUE = '\033[0;34m'
    YELLOW = '\033[1;33m'
    RED = '\033[0;31m'
    NC = '\033[0m'  # No Color
    BOLD = '\033[1m'


# All available datasets
ALL_DATASETS = [
    "hotpotqa",
    "iirc",
    "2wikimultihopqa",
    "musique",
    "wiki",
]


def print_colored(message: str, color: str = Colors.NC):
    """Print colored message"""
    print(f"{color}{message}{Colors.NC}")


def print_header(message: str):
    """Print section header"""
    print()
    print_colored("=" * 80, Colors.BLUE)
    print_colored(message, Colors.BLUE)
    print_colored("=" * 80, Colors.BLUE)
    print()


def check_elasticsearch() -> bool:
    """Check if Elasticsearch is running"""
    script_dir = Path(__file__).parent
    es_script = script_dir / "es.sh"
    
    try:
        result = subprocess.run(
            [str(es_script), "status"],
            capture_output=True,
            text=True,
            check=False
        )
        return "running" in result.stdout.lower()
    except Exception as e:
        print_colored(f"Failed to check Elasticsearch status: {e}", Colors.RED)
        return False


def start_elasticsearch() -> bool:
    """Start Elasticsearch"""
    script_dir = Path(__file__).parent
    es_script = script_dir / "es.sh"
    
    print_colored("Starting Elasticsearch...", Colors.YELLOW)
    try:
        subprocess.run([str(es_script), "start"], check=True)
        return True
    except Exception as e:
        print_colored(f"Failed to start Elasticsearch: {e}", Colors.RED)
        return False


def build_index(dataset: str, args: List[str]) -> tuple[bool, float]:
    """
    Build index for a dataset.
    
    Returns:
        (success, time_taken)
    """
    script_dir = Path(__file__).parent
    build_script = script_dir / "build_index.py"
    
    cmd = [sys.executable, str(build_script), dataset] + args + ["--force"]
    
    print_colored(f"Command: {' '.join(cmd)}", Colors.YELLOW)
    print()
    
    start_time = time.time()
    
    try:
        subprocess.run(cmd, check=True)
        end_time = time.time()
        return True, end_time - start_time
    except subprocess.CalledProcessError:
        end_time = time.time()
        return False, end_time - start_time


def format_time(seconds: float) -> str:
    """Format seconds to human readable time"""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes}m {secs}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        return f"{hours}h {minutes}m"


def main():
    parser = argparse.ArgumentParser(
        description="Build all dataset indexes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Build all datasets with all features
  python build_all_indexes.py --all

  # Build all datasets with BM25 only (fastest)
  python build_all_indexes.py --bm25-only

  # Build specific datasets
  python build_all_indexes.py --datasets wiki hotpotqa --with-splade

  # Build all with Dense + SPLADE
  python build_all_indexes.py --with-dense --with-splade
        """
    )
    
    # Index mode options
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument("--bm25-only", action="store_true", 
                           help="Build with BM25 only (fastest)")
    mode_group.add_argument("--all", action="store_true", 
                           help="Build with BM25 + Dense + SPLADE")
    
    # Individual features
    parser.add_argument("--with-dense", action="store_true",
                       help="Include dense embeddings (HNSW)")
    parser.add_argument("--with-splade", action="store_true",
                       help="Include SPLADE vectors")
    
    # Dataset selection
    parser.add_argument("--datasets", nargs="+", choices=ALL_DATASETS + ["all"],
                       default=["all"],
                       help="Datasets to build (default: all)")
    
    # Other options
    parser.add_argument("--skip-es-check", action="store_true",
                       help="Skip Elasticsearch status check")
    parser.add_argument("--continue-on-error", action="store_true",
                       help="Continue building even if one dataset fails")
    
    args = parser.parse_args()
    
    # Determine datasets to build
    if "all" in args.datasets:
        datasets = ALL_DATASETS
    else:
        datasets = args.datasets
    
    # Determine index arguments
    index_args = []
    mode_description = "BM25"
    
    if args.all:
        index_args = ["--use-dense", "--use-splade"]
        mode_description = "BM25 + Dense + SPLADE"
    elif args.bm25_only:
        index_args = []
        mode_description = "BM25 only"
    else:
        if args.with_dense:
            index_args.append("--use-dense")
            mode_description += " + Dense"
        if args.with_splade:
            index_args.append("--use-splade")
            mode_description += " + SPLADE"
    
    # Print configuration
    print_header("Build All Dataset Indexes")
    print_colored(f"Mode: {mode_description}", Colors.YELLOW)
    print_colored(f"Datasets: {', '.join(datasets)}", Colors.YELLOW)
    print_colored(f"Total: {len(datasets)} dataset(s)", Colors.YELLOW)
    
    # Check Elasticsearch
    if not args.skip_es_check:
        print()
        print_colored("Checking Elasticsearch...", Colors.BLUE)
        if not check_elasticsearch():
            print_colored("Elasticsearch is not running!", Colors.RED)
            if not start_elasticsearch():
                print_colored("Failed to start Elasticsearch. Exiting.", Colors.RED)
                return 1
            print_colored("Elasticsearch started successfully", Colors.GREEN)
        else:
            print_colored("Elasticsearch is running", Colors.GREEN)
    
    # Track results
    results: Dict[str, tuple[bool, float]] = {}
    total_start = time.time()
    
    # Build each dataset
    print_header("Starting Index Building")
    
    for i, dataset in enumerate(datasets, 1):
        print()
        print_colored("=" * 80, Colors.BLUE)
        print_colored(f"[{i}/{len(datasets)}] Building index: {dataset}", Colors.BOLD)
        print_colored("=" * 80, Colors.BLUE)
        print()
        
        success, time_taken = build_index(dataset, index_args)
        results[dataset] = (success, time_taken)
        
        print()
        if success:
            print_colored(f"✓ Successfully built index: {dataset}", Colors.GREEN)
            print_colored(f"  Time taken: {format_time(time_taken)}", Colors.GREEN)
        else:
            print_colored(f"✗ Failed to build index: {dataset}", Colors.RED)
            print_colored(f"  Time taken: {format_time(time_taken)}", Colors.RED)
            
            if not args.continue_on_error:
                print()
                response = input("Continue with remaining datasets? (y/n) ").lower()
                if response != 'y':
                    print_colored("Aborted by user", Colors.YELLOW)
                    break
    
    # Print summary
    total_time = time.time() - total_start
    successful = [d for d, (s, _) in results.items() if s]
    failed = [d for d, (s, _) in results.items() if not s]
    
    print_header("Build Summary")
    print_colored(f"Total time: {format_time(total_time)}", Colors.YELLOW)
    print()
    print_colored(f"Successful: {len(successful)}", Colors.GREEN)
    for dataset in successful:
        _, time_taken = results[dataset]
        print_colored(f"  ✓ {dataset} ({format_time(time_taken)})", Colors.GREEN)
    
    print()
    print_colored(f"Failed: {len(failed)}", Colors.RED)
    for dataset in failed:
        _, time_taken = results[dataset]
        print_colored(f"  ✗ {dataset} ({format_time(time_taken)})", Colors.RED)
    
    print()
    
    # Exit code
    if failed:
        print_colored(f"✗ {len(failed)} dataset(s) failed to build", Colors.RED)
        return 1
    else:
        print_colored(f"✓ All {len(successful)} dataset(s) built successfully!", Colors.GREEN)
        return 0


if __name__ == "__main__":
    sys.exit(main())

