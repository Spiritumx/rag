"""
Result management for saving/loading intermediate results.
"""

import json
import os
from typing import Dict, Any


class ResultManager:
    """Manage saving/loading of results with resume capability."""

    def __init__(self, config):
        self.config = config
        self.stage1_dir = config['outputs']['stage1_dir']
        self.stage2_dir = config['outputs']['stage2_dir']
        self.stage3_dir = config['outputs']['stage3_dir']

        # Create output directories
        os.makedirs(self.stage1_dir, exist_ok=True)
        os.makedirs(self.stage2_dir, exist_ok=True)
        os.makedirs(self.stage3_dir, exist_ok=True)

    def get_stage1_output_path(self, dataset_name: str) -> str:
        """Get path for stage 1 classification results."""
        return os.path.join(self.stage1_dir, f'{dataset_name}_classifications.jsonl')

    def get_stage2_output_path(self, dataset_name: str) -> str:
        """Get path for stage 2 prediction results."""
        return os.path.join(self.stage2_dir, f'{dataset_name}_predictions.json')

    def get_stage3_output_path(self) -> str:
        """Get path for stage 3 metrics."""
        return os.path.join(self.stage3_dir, 'overall_metrics.json')

    def load_existing_results(self, path: str) -> Dict:
        """
        Load existing results from previous run.

        Args:
            path: Path to results file

        Returns:
            Dictionary of existing results (empty if file doesn't exist)
        """
        if not os.path.exists(path):
            return {}

        try:
            if path.endswith('.jsonl'):
                # JSONL format (for stage 1)
                results = {}
                with open(path, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            item = json.loads(line)
                            results[item['question_id']] = item
                return results
            else:
                # JSON format (for stage 2 and 3)
                with open(path, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception as e:
            print(f"Warning: Failed to load existing results from {path}: {e}")
            return {}

    def save_results(self, path: str, results: Dict):
        """
        Save results to file.

        Args:
            path: Output file path
            results: Results dictionary to save
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)

        if path.endswith('.jsonl'):
            # JSONL format (for stage 1)
            with open(path, 'w', encoding='utf-8') as f:
                for item in results.values():
                    f.write(json.dumps(item, ensure_ascii=False) + '\n')
        else:
            # JSON format (for stage 2 and 3)
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)

    def load_stage1_results(self, dataset_name: str) -> Dict:
        """Load classification results for a dataset."""
        path = self.get_stage1_output_path(dataset_name)
        return self.load_existing_results(path)

    def load_stage2_results(self, dataset_name: str) -> Dict:
        """Load prediction results for a dataset."""
        path = self.get_stage2_output_path(dataset_name)
        return self.load_existing_results(path)

    def save_stage1_results(self, dataset_name: str, results: Dict):
        """Save classification results for a dataset."""
        path = self.get_stage1_output_path(dataset_name)
        self.save_results(path, results)

    def save_stage2_results(self, dataset_name: str, results: Dict):
        """Save prediction results for a dataset."""
        path = self.get_stage2_output_path(dataset_name)
        self.save_results(path, results)
