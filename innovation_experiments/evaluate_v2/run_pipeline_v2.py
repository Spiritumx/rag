"""
Main pipeline orchestrator for RAG + Innovations Evaluation (V2).

Runs all three stages with innovations:
- Stage 1: Classification (SHARED with baseline)
- Stage 2: Generation with Adaptive Retrieval + Cascading + MI-RA-ToT
- Stage 3: Evaluation with cascade analysis
"""

import os
import sys
import argparse
import logging
from datetime import datetime

# Add paths for imports
# Add both /root/graduateRAG and /root/graduateRAG/innovation_experiments
base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, base_dir)  # For 'from evaluate.utils...'
sys.path.insert(0, os.path.join(base_dir, 'innovation_experiments'))  # For 'from evaluate_v2...'

from evaluate.utils.config_loader import ConfigLoader
from evaluate.utils.llama_generator import LlamaServerManager
from evaluate_v2.utils.service_checker import ServiceChecker  # V2 version (port 8002)
from evaluate.stage1_classify import Stage1Classifier  # SHARED with baseline
from evaluate_v2.stage2_generate_v2 import Stage2GeneratorV2  # V2 with innovations
from evaluate_v2.stage3_evaluate_v2 import Stage3EvaluatorV2  # V2 with cascade analysis


class PipelineRunnerV2:
    """
    Main orchestrator for the 3-stage evaluation pipeline with innovations.

    Innovations:
    1. Adaptive Retrieval (Port 8002, dynamic weights)
    2. Cascading Dynamic Routing (confidence-based fallback)
    3. MI-RA-ToT (beam search multi-hop reasoning)
    """

    def __init__(self, config_path='innovation_experiments/evaluate_v2/config_v2.yaml'):
        self.config = ConfigLoader.load_config(config_path)
        self.service_checker = ServiceChecker(self.config)
        self.llama_manager = None

        # Setup logging
        self.setup_logging()

    def setup_logging(self):
        """Setup logging configuration."""
        log_file = self.config.get('execution', {}).get(
            'log_file',
            'innovation_experiments/evaluate_v2/pipeline_v2.log'
        )
        log_dir = os.path.dirname(log_file)

        if log_dir:
            os.makedirs(log_dir, exist_ok=True)

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )

        # Suppress noisy third-party library logs
        for lib in ('transformers', 'tokenizers', 'huggingface_hub',
                     'sentence_transformers', 'urllib3', 'filelock'):
            logging.getLogger(lib).setLevel(logging.ERROR)

        self.logger = logging.getLogger(__name__)

    def run(self, stages=None, datasets=None):
        """
        Run the evaluation pipeline with innovations.

        Args:
            stages: List of stages to run [1, 2, 3] or None for all
            datasets: List of datasets or None for all in config
        """
        if stages is None:
            stages = [1, 2, 3]

        start_time = datetime.now()

        self.logger.info("="*80)
        self.logger.info("RAG + INNOVATIONS EVALUATION PIPELINE (V2)")
        self.logger.info("="*80)
        self.logger.info(f"Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        self.logger.info(f"Stages to run: {stages}")
        self.logger.info(f"Datasets: {datasets if datasets else 'all'}")
        self.logger.info("")
        self.logger.info("Innovations Enabled:")
        self.logger.info("  1. Adaptive Retrieval (Port 8002)")
        self.logger.info("  2. Cascading Dynamic Routing (Confidence-based)")
        self.logger.info("  3. MI-RA-ToT (Beam Search Reasoning)")
        self.logger.info("="*80)

        try:
            # Pre-flight checks
            if self.config.get('execution', {}).get('service_check', True):
                # Only check services needed for the stages being run
                if 2 in stages:
                    self.logger.info("\nRunning service checks (V2)...")
                    self.service_checker.check_all()

            # Auto-start LLM server if needed
            if 2 in stages and self.config.get('llm', {}).get('auto_start_server', False):
                self.logger.info("\nAuto-starting LLM server...")
                self.llama_manager = LlamaServerManager(self.config)
                self.llama_manager.start_server()

            # Stage 1: Classification (SHARED with baseline)
            if 1 in stages:
                self.logger.info("\n" + "="*80)
                self.logger.info("STAGE 1: CLASSIFICATION (Shared with Baseline)")
                self.logger.info("="*80)
                self.logger.info("Note: Stage 1 results are shared with baseline for fair comparison")
                classifier = Stage1Classifier(self.config)
                classifier.run(datasets=datasets)

            # Stage 2: Generation with Innovations
            if 2 in stages:
                self.logger.info("\n" + "="*80)
                self.logger.info("STAGE 2: GENERATION (V2 with Innovations)")
                self.logger.info("="*80)
                generator = Stage2GeneratorV2(self.config)
                generator.run(datasets=datasets)

            # Stage 3: Evaluation with Cascade Analysis
            if 3 in stages:
                self.logger.info("\n" + "="*80)
                self.logger.info("STAGE 3: EVALUATION (V2 with Cascade Analysis)")
                self.logger.info("="*80)
                evaluator = Stage3EvaluatorV2(self.config)
                evaluator.run(datasets=datasets)

            # Done
            end_time = datetime.now()
            duration = end_time - start_time

            self.logger.info("\n" + "="*80)
            self.logger.info("PIPELINE (V2) COMPLETE")
            self.logger.info("="*80)
            self.logger.info(f"End time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
            self.logger.info(f"Total duration: {duration}")
            self.logger.info("")
            self.logger.info("Results saved to:")
            self.logger.info(f"  Predictions: {self.config['outputs']['stage2_dir']}")
            self.logger.info(f"  Metrics: {self.config['outputs']['stage3_dir']}")
            self.logger.info(f"  Cascade Analysis: {self.config['outputs']['cascade_dir']}")

        except KeyboardInterrupt:
            self.logger.warning("\n\nPipeline interrupted by user")
            raise

        except Exception as e:
            self.logger.error(f"\n\nPipeline failed with error: {e}")
            import traceback
            traceback.print_exc()
            raise

        finally:
            # Stop LLM server if we started it
            if self.llama_manager:
                self.llama_manager.stop_server()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run RAG + Innovations Evaluation Pipeline (V2)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all stages for all datasets
  python innovation_experiments/evaluate_v2/run_pipeline_v2.py

  # Run only generation and evaluation (reuse Stage 1 from baseline)
  python innovation_experiments/evaluate_v2/run_pipeline_v2.py --stages 2 3

  # Run for specific datasets
  python innovation_experiments/evaluate_v2/run_pipeline_v2.py --stages 2 3 --datasets squad hotpotqa

  # Use custom config
  python innovation_experiments/evaluate_v2/run_pipeline_v2.py --config my_config_v2.yaml

  # Skip service checks
  python innovation_experiments/evaluate_v2/run_pipeline_v2.py --no-service-check

Important Notes:
  - Stage 1 (Classification) is SHARED with baseline for fair A/B comparison
  - Stage 2 uses Port 8002 retriever (adaptive weights + cascading + ToT)
  - Stage 3 generates cascade analysis metrics for paper
  - Make sure V2 retriever server is running on port 8002!
        """
    )

    parser.add_argument(
        '--config',
        default='innovation_experiments/evaluate_v2/config_v2.yaml',
        help='Path to config file (default: config_v2.yaml)'
    )

    parser.add_argument(
        '--stages',
        nargs='+',
        type=int,
        choices=[1, 2, 3],
        help='Stages to run: 1=Classification, 2=Generation, 3=Evaluation (default: all)'
    )

    parser.add_argument(
        '--datasets',
        nargs='+',
        help='Datasets to process (default: all in config)'
    )

    parser.add_argument(
        '--no-service-check',
        action='store_true',
        help='Skip service health checks'
    )

    args = parser.parse_args()

    # Load and modify config
    runner = PipelineRunnerV2(args.config)

    if args.no_service_check:
        runner.config['execution']['service_check'] = False

    # Run pipeline
    try:
        runner.run(stages=args.stages, datasets=args.datasets)
    except KeyboardInterrupt:
        print("\n\nExiting...")
        sys.exit(1)
    except Exception:
        sys.exit(1)


if __name__ == '__main__':
    main()
