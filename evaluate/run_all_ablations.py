"""
Run all ablation experiments and generate summary report.
Executes experiments for: Z, S-Sparse, S-Dense, S-Hybrid, M strategies.
"""

import os
import sys
import json
import argparse
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluate.utils.config_loader import ConfigLoader
from evaluate.run_ablation import AblationExperiment, VALID_STRATEGIES


def check_ablation_complete(strategy: str, datasets: list) -> bool:
    """
    Check if an ablation experiment is already complete.

    Args:
        strategy: Strategy name
        datasets: List of datasets

    Returns:
        True if all datasets have predictions
    """
    base_dir = f"evaluate/outputs/ablation_{strategy}"
    pred_dir = os.path.join(base_dir, "stage2_predictions")

    if not os.path.exists(pred_dir):
        return False

    for dataset in datasets:
        pred_file = os.path.join(pred_dir, f"{dataset}_predictions.json")
        if not os.path.exists(pred_file):
            return False

    return True


def run_all_ablations(config, datasets=None, strategies=None, force=False):
    """
    Run all ablation experiments sequentially.

    Args:
        config: Configuration dictionary
        datasets: List of datasets to process (None = all)
        strategies: List of strategies to test (None = all)
        force: Force re-run even if complete

    Returns:
        Dictionary of results for each strategy
    """
    if strategies is None:
        strategies = VALID_STRATEGIES

    if datasets is None:
        datasets = config['datasets']

    results = {}

    print("\n" + "="*80)
    print("RUNNING ALL ABLATION EXPERIMENTS")
    print("="*80)
    print(f"Strategies: {strategies}")
    print(f"Datasets: {datasets}")
    print(f"Force re-run: {force}")
    print("="*80)

    for i, strategy in enumerate(strategies, 1):
        print(f"\n{'#'*80}")
        print(f"# ABLATION {i}/{len(strategies)}: {strategy}")
        print(f"{'#'*80}")

        # Check if already complete
        if not force and check_ablation_complete(strategy, datasets):
            print(f"  ✓ Already complete, loading existing results...")
            metrics_path = os.path.join(
                f"evaluate/outputs/ablation_{strategy}",
                "stage3_metrics",
                "overall_metrics.json"
            )
            if os.path.exists(metrics_path):
                with open(metrics_path, 'r', encoding='utf-8') as f:
                    results[strategy] = json.load(f)
                print(f"  ✓ Loaded results for {strategy}")
                continue

        try:
            experiment = AblationExperiment(config, strategy)
            experiment.run(datasets=datasets)

            # Load results
            metrics_path = os.path.join(
                f"evaluate/outputs/ablation_{strategy}",
                "stage3_metrics",
                "overall_metrics.json"
            )
            if os.path.exists(metrics_path):
                with open(metrics_path, 'r', encoding='utf-8') as f:
                    results[strategy] = json.load(f)
            else:
                results[strategy] = {'error': 'Metrics file not found'}

        except Exception as e:
            print(f"Error running ablation for {strategy}: {e}")
            import traceback
            traceback.print_exc()
            results[strategy] = {'error': str(e)}

    return results


def generate_summary_report(results: dict, output_path: str):
    """
    Generate summary report comparing all ablation experiments.

    Args:
        results: Dictionary of results for each strategy
        output_path: Path to save the report
    """
    print("\n" + "="*80)
    print("ABLATION EXPERIMENTS SUMMARY")
    print("="*80)

    # Collect overall metrics
    summary_data = []
    for strategy in VALID_STRATEGIES:
        if strategy in results:
            res = results[strategy]
            if 'overall' in res:
                overall = res['overall']
                summary_data.append({
                    'strategy': strategy,
                    'em': overall.get('em', 0),
                    'f1': overall.get('f1', 0),
                    'count': overall.get('count', 0)
                })
            elif 'error' in res:
                summary_data.append({
                    'strategy': strategy,
                    'em': 0,
                    'f1': 0,
                    'count': 0,
                    'error': res['error']
                })

    # Print summary table
    print(f"\n{'Strategy':<12} {'EM':>10} {'F1':>10} {'Count':>10}")
    print("-" * 45)
    for row in summary_data:
        if 'error' in row:
            print(f"{row['strategy']:<12} {'ERROR':>10} {row['error'][:20]}")
        else:
            print(f"{row['strategy']:<12} {row['em']:>10.4f} {row['f1']:>10.4f} {row['count']:>10}")
    print("-" * 45)

    # Per-dataset breakdown
    print("\n\nPER-DATASET RESULTS:")
    print("="*80)

    datasets = set()
    for strategy, res in results.items():
        if isinstance(res, dict):
            for key in res.keys():
                if key not in ['overall', 'error']:
                    datasets.add(key)

    for dataset in sorted(datasets):
        print(f"\n{dataset}:")
        print(f"{'Strategy':<12} {'EM':>10} {'F1':>10}")
        print("-" * 35)
        for strategy in VALID_STRATEGIES:
            if strategy in results and dataset in results[strategy]:
                metrics = results[strategy][dataset]
                if 'overall' in metrics:
                    overall = metrics['overall']
                    print(f"{strategy:<12} {overall.get('em', 0):>10.4f} {overall.get('f1', 0):>10.4f}")

    # Save detailed report
    report = {
        'timestamp': datetime.now().isoformat(),
        'summary': summary_data,
        'detailed_results': results
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\n✓ Summary report saved to: {output_path}")

    # Also save as text report
    txt_path = output_path.replace('.json', '.txt')
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("ABLATION EXPERIMENTS SUMMARY REPORT\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*80 + "\n\n")

        f.write("OVERALL RESULTS:\n")
        f.write("-"*45 + "\n")
        f.write(f"{'Strategy':<12} {'EM':>10} {'F1':>10} {'Count':>10}\n")
        f.write("-"*45 + "\n")
        for row in summary_data:
            if 'error' in row:
                f.write(f"{row['strategy']:<12} ERROR: {row['error']}\n")
            else:
                f.write(f"{row['strategy']:<12} {row['em']:>10.4f} {row['f1']:>10.4f} {row['count']:>10}\n")
        f.write("-"*45 + "\n\n")

        f.write("\nPER-DATASET RESULTS:\n")
        f.write("="*80 + "\n")
        for dataset in sorted(datasets):
            f.write(f"\n{dataset}:\n")
            f.write(f"{'Strategy':<12} {'EM':>10} {'F1':>10}\n")
            f.write("-"*35 + "\n")
            for strategy in VALID_STRATEGIES:
                if strategy in results and dataset in results[strategy]:
                    metrics = results[strategy][dataset]
                    if 'overall' in metrics:
                        overall = metrics['overall']
                        f.write(f"{strategy:<12} {overall.get('em', 0):>10.4f} {overall.get('f1', 0):>10.4f}\n")
            f.write("\n")

    print(f"✓ Text report saved to: {txt_path}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run all ablation experiments and generate summary"
    )
    parser.add_argument(
        '--config',
        default='evaluate/config.yaml',
        help='Path to config file'
    )
    parser.add_argument(
        '--datasets',
        nargs='+',
        help='Datasets to process (default: all in config)'
    )
    parser.add_argument(
        '--strategies',
        nargs='+',
        choices=VALID_STRATEGIES,
        help=f'Strategies to test (default: all). Options: {VALID_STRATEGIES}'
    )
    parser.add_argument(
        '--output',
        default='evaluate/outputs/ablation_summary.json',
        help='Path for summary report'
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Force re-run even if already complete'
    )

    args = parser.parse_args()

    # Load config
    config = ConfigLoader.load_config(args.config)

    # Run all ablations
    results = run_all_ablations(
        config,
        datasets=args.datasets,
        strategies=args.strategies,
        force=args.force
    )

    # Generate summary report
    generate_summary_report(results, args.output)

    print("\n" + "="*80)
    print("✓ ALL ABLATION EXPERIMENTS COMPLETE")
    print("="*80)


if __name__ == '__main__':
    main()
