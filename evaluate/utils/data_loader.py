"""
Dataset loading utilities.
"""

import json
import os
from typing import List, Dict


class DataLoader:
    """Load test datasets."""

    def __init__(self, config):
        self.config = config
        self.base_path = config['data']['base_path']
        self.test_file = config['data']['test_file']

    def load_test_data(self, dataset_name: str) -> List[Dict]:
        """
        Load test data for a specific dataset.

        Args:
            dataset_name: Name of the dataset (e.g., 'squad', 'hotpotqa')

        Returns:
            List of test examples
        """
        test_path = os.path.join(self.base_path, dataset_name, self.test_file)

        if not os.path.exists(test_path):
            raise FileNotFoundError(f"Test data not found: {test_path}")

        data = []
        with open(test_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    data.append(item)

        print(f"Loaded {len(data)} examples from {dataset_name}")
        return data

    def load_all_datasets(self) -> Dict[str, List[Dict]]:
        """
        Load all configured datasets.

        Returns:
            Dictionary mapping dataset name to list of examples
        """
        all_data = {}
        for dataset_name in self.config['datasets']:
            all_data[dataset_name] = self.load_test_data(dataset_name)

        total_examples = sum(len(examples) for examples in all_data.values())
        print(f"Loaded {total_examples} total examples from {len(all_data)} datasets")

        return all_data
