"""
Main pipeline orchestrator for RAG + Classifier evaluation.
Runs all three stages: Classification → Generation → Evaluation
"""

import os
import sys
import argparse
import logging
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluate.utils.config_loader import ConfigLoader
from evaluate.utils.service_checker import ServiceChecker
from evaluate.utils.llama_generator import LlamaServerManager
from evaluate.stage1_classify import Stage1Classifier
from evaluate.stage2_generate import Stage2Generator
from evaluate.stage3_evaluate import Stage3Evaluator


class PipelineRunner:
    """Main orchestrator for the 3-stage evaluation pipeline."""

    def __init__(self, config_path='evaluate/config.yaml'):
        self.config = ConfigLoader.load_config(config_path)
        self.service_checker = ServiceChecker(self.config)
        self.llama_manager = None

        # Setup logging
        self.setup_logging()

    def setup_logging(self):
        """Setup logging configuration."""
        log_file = self.config.get('execution', {}).get('log_file', 'evaluate/pipeline.log')
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

        self.logger = logging.getLogger(__name__)

    def run(self, stages=None, datasets=None):
        """
        Run the evaluation pipeline.

        Args:
            stages: List of stages to run [1, 2, 3] or None for all
            datasets: List of datasets or None for all in config
        """
        if stages is None:
            stages = [1, 2, 3]

        start_time = datetime.now()

        self.logger.info("="*80)
        self.logger.info("RAG + CLASSIFIER EVALUATION PIPELINE")
        self.logger.info("="*80)
        self.logger.info(f"Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        self.logger.info(f"Stages to run: {stages}")
        self.logger.info(f"Datasets: {datasets if datasets else 'all'}")

        try:
            # Pre-flight checks
            if self.config.get('execution', {}).get('service_check', True):
                # Only check services needed for the stages being run
                if 2 in stages:
                    self.logger.info("\nRunning service checks...")
                    self.service_checker.check_all()

            # Auto-start LLM server if needed
            if 2 in stages and self.config.get('llm', {}).get('auto_start_server', False):
                self.logger.info("\nAuto-starting LLM server...")
                self.llama_manager = LlamaServerManager(self.config)
                self.llama_manager.start_server()

            # Stage 1: Classification
            if 1 in stages:
                self.logger.info("\n" + "="*80)
                self.logger.info("STAGE 1: CLASSIFICATION")
                self.logger.info("="*80)
                classifier = Stage1Classifier(self.config)
                classifier.run(datasets=datasets)

            # Stage 2: Generation
            if 2 in stages:
                self.logger.info("\n" + "="*80)
                self.logger.info("STAGE 2: GENERATION")
                self.logger.info("="*80)
                generator = Stage2Generator(self.config)
                generator.run(datasets=datasets)

            # Stage 3: Evaluation
            if 3 in stages:
                self.logger.info("\n" + "="*80)
                self.logger.info("STAGE 3: EVALUATION")
                self.logger.info("="*80)
                evaluator = Stage3Evaluator(self.config)
                evaluator.run(datasets=datasets)

            # Done
            end_time = datetime.now()
            duration = end_time - start_time

            self.logger.info("\n" + "="*80)
            self.logger.info("PIPELINE COMPLETE")
            self.logger.info("="*80)
            self.logger.info(f"End time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
            self.logger.info(f"Total duration: {duration}")

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
        description="Run RAG + Classifier Evaluation Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all stages for all datasets
  python evaluate/run_pipeline.py

  # Run only classification stage
  python evaluate/run_pipeline.py --stages 1

  # Run generation and evaluation for specific datasets
  python evaluate/run_pipeline.py --stages 2 3 --datasets squad hotpotqa

  # Use custom config
  python evaluate/run_pipeline.py --config my_config.yaml

  # Skip service checks
  python evaluate/run_pipeline.py --no-service-check
        """
    )

    parser.add_argument(
        '--config',
        default='evaluate/config.yaml',
        help='Path to config file (default: evaluate/config.yaml)'
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
    runner = PipelineRunner(args.config)

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
