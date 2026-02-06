"""
Ablation Experiment: Run single-strategy experiments.
Forces all questions to use the same retrieval strategy for comparison.
"""

import os
import sys
import json
import copy
import argparse

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluate.utils.config_loader import ConfigLoader
from evaluate.utils.data_loader import DataLoader
from evaluate.stage2_generate import Stage2Generator
from evaluate.stage3_evaluate import Stage3Evaluator


VALID_STRATEGIES = ['Z', 'S-Sparse', 'S-Dense', 'S-Hybrid', 'M']


class AblationExperiment:
    """Run ablation experiment with a single forced strategy."""

    def __init__(self, config, strategy: str):
        """
        Initialize ablation experiment.

        Args:
            config: Configuration dictionary
            strategy: Strategy to force (Z, S-Sparse, S-Dense, S-Hybrid, M)
        """
        if strategy not in VALID_STRATEGIES:
            raise ValueError(f"Invalid strategy: {strategy}. Must be one of {VALID_STRATEGIES}")

        self.base_config = config
        self.strategy = strategy
        self.data_loader = DataLoader(config)

        # Create ablation-specific config with modified output paths
        self.config = self._create_ablation_config(config, strategy)

    def _create_ablation_config(self, config: dict, strategy: str) -> dict:
        """Create config with ablation-specific output paths."""
        ablation_config = copy.deepcopy(config)

        # Modify output paths for this ablation
        base_ablation_dir = f"evaluate/outputs/ablation_{strategy}"
        ablation_config['outputs'] = {
            'stage1_dir': os.path.join(base_ablation_dir, 'stage1_classifications'),
            'stage2_dir': os.path.join(base_ablation_dir, 'stage2_predictions'),
            'stage3_dir': os.path.join(base_ablation_dir, 'stage3_metrics'),
            'analysis_dir': os.path.join(base_ablation_dir, 'analysis'),
        }

        # Create directories
        for dir_path in ablation_config['outputs'].values():
            os.makedirs(dir_path, exist_ok=True)

        return ablation_config

    def generate_fake_classifications(self, datasets=None):
        """
        Generate fake classification results with all questions assigned to the forced strategy.

        Args:
            datasets: List of datasets to process (None = all)
        """
        if datasets is None:
            datasets = self.base_config['datasets']

        print(f"\n{'='*60}")
        print(f"GENERATING FAKE CLASSIFICATIONS (Strategy: {self.strategy})")
        print(f"{'='*60}")

        for dataset_name in datasets:
            print(f"\nProcessing dataset: {dataset_name}")

            # Load test data
            test_data = self.data_loader.load_test_data(dataset_name)

            # Generate fake classifications
            classifications = {}
            for item in test_data:
                qid = item['question_id']
                classifications[qid] = {
                    'question_id': qid,
                    'question_text': item['question_text'],
                    'predicted_action': self.strategy,
                    'full_response': f"[ABLATION] Forced strategy: {self.strategy}",
                    'dataset': dataset_name
                }

            # Save to ablation output directory
            output_path = os.path.join(
                self.config['outputs']['stage1_dir'],
                f'{dataset_name}_classifications.jsonl'
            )
            with open(output_path, 'w', encoding='utf-8') as f:
                for item in classifications.values():
                    f.write(json.dumps(item, ensure_ascii=False) + '\n')

            print(f"  Saved {len(classifications)} fake classifications to {output_path}")

    def run(self, datasets=None):
        """
        Run the full ablation experiment.

        Args:
            datasets: List of datasets to process (None = all)
        """
        if datasets is None:
            datasets = self.base_config['datasets']

        print("\n" + "="*60)
        print(f"ABLATION EXPERIMENT: {self.strategy}")
        print("="*60)
        print(f"Strategy: {self.strategy}")
        print(f"Datasets: {datasets}")
        print(f"Output dir: evaluate/outputs/ablation_{self.strategy}/")
        print("="*60)

        # Step 1: Generate fake classifications
        self.generate_fake_classifications(datasets)

        # Step 2: Run Stage 2 (Generation)
        print("\n" + "="*60)
        print("Running Stage 2: Generation")
        print("="*60)
        generator = Stage2Generator(self.config)
        generator.run(datasets=datasets)

        # Step 3: Run Stage 3 (Evaluation)
        print("\n" + "="*60)
        print("Running Stage 3: Evaluation")
        print("="*60)
        evaluator = Stage3Evaluator(self.config)
        evaluator.run(datasets=datasets)

        print("\n" + "="*60)
        print(f"✓ ABLATION EXPERIMENT COMPLETE: {self.strategy}")
        print(f"  Results saved to: evaluate/outputs/ablation_{self.strategy}/")
        print("="*60)


def main():
    """Main entry point for ablation experiment."""
    parser = argparse.ArgumentParser(
        description="Run ablation experiment with a single forced strategy"
    )
    parser.add_argument(
        '--strategy',
        required=True,
        choices=VALID_STRATEGIES,
        help=f"Strategy to force: {VALID_STRATEGIES}"
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

    args = parser.parse_args()

    # Load config
    config = ConfigLoader.load_config(args.config)

    # Run ablation experiment
    experiment = AblationExperiment(config, args.strategy)
    experiment.run(datasets=args.datasets)


if __name__ == '__main__':
    main()
