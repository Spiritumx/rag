"""
Master Analysis Runner

Runs all three analysis scripts:
1. compare_metrics.py - A/B comparison (baseline vs V2)
2. analyze_cascade.py - Cascade routing analysis (Innovation 2)
3. analyze_weights.py - Adaptive weight patterns (Innovation 1)

Generates comprehensive summary report for paper.
"""

import os
import sys
import subprocess
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


class MasterAnalyzer:
    """Run all analysis scripts and generate summary."""

    def __init__(self, output_dir='innovation_experiments/analysis_results'):
        self.output_dir = output_dir
        self.results = {}

    def run_comparison_analysis(self, datasets=None) -> bool:
        """
        Run baseline vs V2 comparison.

        Args:
            datasets: List of datasets (None = all)

        Returns:
            True if successful
        """
        print("\n" + "="*80)
        print("STEP 1: A/B COMPARISON (Baseline vs V2)")
        print("="*80)

        try:
            from compare_metrics import MetricsComparator

            comparator = MetricsComparator()

            # Load metrics
            if not comparator.load_metrics():
                print("ERROR: Could not load metrics for comparison")
                return False

            # Print summary
            comparator.print_summary()

            # Save results
            md_path, json_path = comparator.save_comparison(self.output_dir)

            self.results['comparison'] = {
                'status': 'success',
                'md_path': md_path,
                'json_path': json_path
            }

            return True

        except Exception as e:
            print(f"ERROR in comparison analysis: {e}")
            import traceback
            traceback.print_exc()
            self.results['comparison'] = {'status': 'failed', 'error': str(e)}
            return False

    def run_cascade_analysis(self, datasets=None) -> bool:
        """
        Run cascade routing analysis.

        Args:
            datasets: List of datasets (None = all)

        Returns:
            True if successful
        """
        print("\n" + "="*80)
        print("STEP 2: CASCADE ROUTING ANALYSIS (Innovation 2)")
        print("="*80)

        try:
            from analyze_cascade import CascadeAnalyzer

            analyzer = CascadeAnalyzer()

            # Load cascade logs
            if not analyzer.load_cascade_logs(datasets=datasets):
                print("ERROR: Could not load cascade logs")
                return False

            # Perform analysis
            cascade_rate = analyzer.analyze_cascade_rate()
            print(f"\nOverall Cascade Rate: {cascade_rate['overall']['rate']:.2%} "
                  f"({cascade_rate['overall']['cascaded']}/{cascade_rate['overall']['total']})")

            print("\nBy Action:")
            for action in ['Z', 'S-Sparse', 'S-Dense', 'S-Hybrid', 'M']:
                if action in cascade_rate['by_action']:
                    stats = cascade_rate['by_action'][action]
                    print(f"  {action:12s}: {stats['rate']:6.2%} ({stats['cascaded']}/{stats['total']})")

            # Save results
            report_path, json_path = analyzer.save_analysis(self.output_dir)

            self.results['cascade'] = {
                'status': 'success',
                'report_path': report_path,
                'json_path': json_path,
                'cascade_rate': cascade_rate['overall']['rate']
            }

            return True

        except Exception as e:
            print(f"ERROR in cascade analysis: {e}")
            import traceback
            traceback.print_exc()
            self.results['cascade'] = {'status': 'failed', 'error': str(e)}
            return False

    def run_weights_analysis(self, datasets=None) -> bool:
        """
        Run adaptive weights analysis.

        Args:
            datasets: List of datasets (None = all)

        Returns:
            True if successful
        """
        print("\n" + "="*80)
        print("STEP 3: ADAPTIVE WEIGHTS ANALYSIS (Innovation 1)")
        print("="*80)

        try:
            from analyze_weights import AdaptiveWeightAnalyzer

            analyzer = AdaptiveWeightAnalyzer()

            # Initialize QueryAnalyzer
            analyzer.initialize_query_analyzer()

            # Load test queries
            if not analyzer.load_test_queries(datasets=datasets):
                print("ERROR: Could not load test queries")
                return False

            # Perform analysis
            char_analysis = analyzer.analyze_query_characteristics()

            if 'total' in char_analysis['overall']:
                print(f"\nTotal Queries Analyzed: {char_analysis['overall']['total']}")

            # Save results
            report_path, json_path = analyzer.save_analysis(self.output_dir)

            self.results['weights'] = {
                'status': 'success',
                'report_path': report_path,
                'json_path': json_path
            }

            return True

        except Exception as e:
            print(f"ERROR in weights analysis: {e}")
            import traceback
            traceback.print_exc()
            self.results['weights'] = {'status': 'failed', 'error': str(e)}
            return False

    def generate_master_summary(self) -> str:
        """
        Generate master summary report.

        Returns:
            Markdown formatted summary
        """
        md = []
        md.append("# Innovation Experiments Analysis Summary\n")
        md.append(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

        md.append("## Analysis Status\n")

        # Comparison
        comp_result = self.results.get('comparison', {})
        comp_status = "✅ Success" if comp_result.get('status') == 'success' else "❌ Failed"
        md.append(f"1. **A/B Comparison (Baseline vs V2)**: {comp_status}")
        if comp_result.get('status') == 'success':
            md.append(f"   - Report: `{comp_result['md_path']}`")
            md.append(f"   - Data: `{comp_result['json_path']}`")
        md.append("")

        # Cascade
        cascade_result = self.results.get('cascade', {})
        cascade_status = "✅ Success" if cascade_result.get('status') == 'success' else "❌ Failed"
        md.append(f"2. **Cascade Routing Analysis (Innovation 2)**: {cascade_status}")
        if cascade_result.get('status') == 'success':
            md.append(f"   - Report: `{cascade_result['report_path']}`")
            md.append(f"   - Data: `{cascade_result['json_path']}`")
            if 'cascade_rate' in cascade_result:
                md.append(f"   - Overall Cascade Rate: {cascade_result['cascade_rate']:.2%}")
        md.append("")

        # Weights
        weights_result = self.results.get('weights', {})
        weights_status = "✅ Success" if weights_result.get('status') == 'success' else "❌ Failed"
        md.append(f"3. **Adaptive Weights Analysis (Innovation 1)**: {weights_status}")
        if weights_result.get('status') == 'success':
            md.append(f"   - Report: `{weights_result['report_path']}`")
            md.append(f"   - Data: `{weights_result['json_path']}`")
        md.append("")

        md.append("## Three Innovations Evaluated\n")
        md.append("### Innovation 1: Adaptive Retrieval")
        md.append("- **Mechanism**: Dynamic hybrid weight adjustment based on query analysis")
        md.append("- **Key Components**: QueryAnalyzer (NER + semantic classifier)")
        md.append("- **Expected Benefit**: Better match between query type and retrieval strategy\n")

        md.append("### Innovation 2: Cascading Dynamic Routing")
        md.append("- **Mechanism**: Confidence-based fallback to stronger strategies")
        md.append("- **Key Components**: ConfidenceVerifier (cross-encoder), RoutingLogger")
        md.append("- **Expected Benefit**: Improved accuracy on low-confidence predictions\n")

        md.append("### Innovation 3: MI-RA-ToT (Mutual Information Tree-of-Thought)")
        md.append("- **Mechanism**: Beam search multi-hop reasoning with MI scoring")
        md.append("- **Key Components**: BeamSearchToT, MutualInformationScorer")
        md.append("- **Expected Benefit**: Better reasoning paths than greedy search\n")

        md.append("## Next Steps for Paper\n")
        md.append("1. Review all individual analysis reports")
        md.append("2. Extract key findings and metrics for paper")
        md.append("3. Create visualizations (confidence distribution, weight patterns, cascade flow)")
        md.append("4. Conduct ablation study (test each innovation individually)")
        md.append("5. Statistical significance testing (if needed)")
        md.append("6. Write discussion section on innovation effectiveness")

        md.append("\n---\n")
        md.append("**All analysis results available in:** `innovation_experiments/analysis_results/`")

        return "\n".join(md)

    def save_master_summary(self):
        """Save master summary report."""
        os.makedirs(self.output_dir, exist_ok=True)

        summary = self.generate_master_summary()
        summary_path = os.path.join(self.output_dir, 'MASTER_SUMMARY.md')

        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write(summary)

        print(f"\n✓ Master summary saved to {summary_path}")
        return summary_path

    def run_all(self, datasets=None):
        """
        Run all analyses.

        Args:
            datasets: List of datasets to analyze
        """
        print("="*80)
        print("INNOVATION EXPERIMENTS - COMPREHENSIVE ANALYSIS")
        print("="*80)
        print(f"Output directory: {self.output_dir}")
        print(f"Datasets: {datasets if datasets else 'all'}")
        print("="*80)

        # Step 1: Comparison
        self.run_comparison_analysis(datasets=datasets)

        # Step 2: Cascade
        self.run_cascade_analysis(datasets=datasets)

        # Step 3: Weights
        self.run_weights_analysis(datasets=datasets)

        # Generate master summary
        print("\n" + "="*80)
        print("GENERATING MASTER SUMMARY")
        print("="*80)

        summary_path = self.save_master_summary()

        # Final summary
        print("\n" + "="*80)
        print("ANALYSIS COMPLETE")
        print("="*80)

        success_count = sum(1 for r in self.results.values() if r.get('status') == 'success')
        total_count = len(self.results)

        print(f"\n✓ Completed {success_count}/{total_count} analyses successfully")
        print(f"\n📊 Master summary: {summary_path}")
        print(f"📂 All results: {self.output_dir}/")

        if success_count < total_count:
            print("\n⚠️  Some analyses failed. Check error messages above.")
            return False

        print("\n✅ All analyses completed successfully!")
        return True


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Run all innovation experiments analyses",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all analyses on all datasets
  python innovation_experiments/run_all_analysis.py

  # Run for specific datasets
  python innovation_experiments/run_all_analysis.py --datasets squad hotpotqa

  # Custom output directory
  python innovation_experiments/run_all_analysis.py --output-dir my_results

Requirements:
  - Baseline pipeline must have been run (evaluate/outputs/)
  - V2 pipeline must have been run (innovation_experiments/evaluate_v2/outputs_v2/)
  - Cascade logs must exist (cascade_analysis/)
        """
    )

    parser.add_argument('--datasets', nargs='+',
                       help='Datasets to analyze (default: all available)')
    parser.add_argument('--output-dir',
                       default='innovation_experiments/analysis_results',
                       help='Directory to save all analysis results')

    args = parser.parse_args()

    # Create master analyzer
    analyzer = MasterAnalyzer(output_dir=args.output_dir)

    # Run all analyses
    try:
        success = analyzer.run_all(datasets=args.datasets)
        sys.exit(0 if success else 1)

    except KeyboardInterrupt:
        print("\n\nAnalysis interrupted by user")
        sys.exit(1)

    except Exception as e:
        print(f"\n\nFatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
