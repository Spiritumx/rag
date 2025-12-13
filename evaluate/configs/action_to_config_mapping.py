"""
Action to config mapping for routing questions to appropriate RAG pipelines.
"""

import os


class ActionConfigMapper:
    """Map classifier actions to JSONNET config files."""

    def __init__(self, config):
        self.config = config
        self.base_config_dir = "evaluate/configs/llama_configs"

        # Ensure config directory exists
        os.makedirs(self.base_config_dir, exist_ok=True)

    def get_config_path(self, action: str, dataset: str = None) -> str:
        """
        Map action to appropriate config file.

        Action mapping:
        - Z (Zero retrieval): Direct generation, no retrieval
        - S-Sparse (Single BM25): Single-hop with BM25 retrieval
        - S-Dense (Single HNSW): Single-hop with dense retrieval
        - S-Hybrid (Single Hybrid): Single-hop with hybrid retrieval
        - M (Multi-hop): Multi-hop reasoning with iterative retrieval

        Args:
            action: Predicted action label
            dataset: Dataset name (optional, for dataset-specific configs)

        Returns:
            Path to JSONNET config file
        """
        # Action to config template mapping
        # Use generic configs that work for all datasets
        mapping = {
            'Z': 'zero_retrieval.jsonnet',
            'S-Sparse': 'single_bm25.jsonnet',
            'S-Dense': 'single_dense.jsonnet',
            'S-Hybrid': 'single_hybrid.jsonnet',
            'M': 'multi_hop.jsonnet',
        }

        config_filename = mapping.get(action)
        if not config_filename:
            raise ValueError(f"Unknown action: {action}")

        config_path = os.path.join(self.base_config_dir, config_filename)

        # Check if config exists
        if not os.path.exists(config_path):
            raise FileNotFoundError(
                f"Config file not found: {config_path}\n"
                f"Please ensure all Llama configs are created in {self.base_config_dir}"
            )

        return config_path

    def get_all_required_configs(self) -> list:
        """
        Get list of all required config files.

        Returns:
            List of config filenames that should exist
        """
        return [
            'zero_retrieval.jsonnet',
            'single_bm25.jsonnet',
            'single_dense.jsonnet',
            'single_hybrid.jsonnet',
            'multi_hop.jsonnet',
        ]

    def check_configs_exist(self) -> bool:
        """
        Check if all required configs exist.

        Returns:
            True if all configs exist, False otherwise
        """
        all_exist = True
        for config_file in self.get_all_required_configs():
            config_path = os.path.join(self.base_config_dir, config_file)
            if not os.path.exists(config_path):
                print(f"Missing config: {config_path}")
                all_exist = False

        return all_exist
