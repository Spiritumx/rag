"""
Action to config mapping for V2 pipeline with innovations.

Maps classifier actions to JSONNET config files in llama_configs_v2/ directory.
"""

import os


class ActionConfigMapperV2:
    """Map classifier actions to V2 JSONNET config files."""

    def __init__(self, config):
        self.config = config
        self.base_config_dir = "innovation_experiments/evaluate_v2/configs/llama_configs_v2"

        # Ensure config directory exists
        os.makedirs(self.base_config_dir, exist_ok=True)

    def get_config_path(self, action: str, dataset: str = None) -> str:
        """
        Map action to appropriate V2 config file.

        Action mapping:
        - Z (Zero retrieval): Direct generation, no retrieval
        - S-Sparse (Single SPLADE): Single-hop with SPLADE retrieval
        - S-Dense (Single HNSW): Single-hop with dense retrieval
        - S-Hybrid (Single Hybrid): Single-hop with ADAPTIVE hybrid retrieval (Innovation 1)
        - M (Multi-hop): MI-RA-ToT reasoning (Innovation 3)

        NOTE: All non-M actions support cascading (Innovation 2)

        Args:
            action: Predicted action label
            dataset: Dataset name (optional, for dataset-specific configs)

        Returns:
            Path to JSONNET config file
        """
        # Action to config template mapping
        # V2 configs use same names as baseline for compatibility
        mapping = {
            'Z': 'zero_retrieval.jsonnet',
            'S-Sparse': 'single_splade.jsonnet',
            'S-Dense': 'single_dense.jsonnet',
            'S-Hybrid': 'single_hybrid.jsonnet',  # Uses port 8002 with adaptive weights
            'M': 'multi_hop.jsonnet',  # MI-RA-ToT handled in stage2_generate_v2.py
        }

        config_filename = mapping.get(action)
        if not config_filename:
            raise ValueError(f"Unknown action: {action}")

        config_path = os.path.join(self.base_config_dir, config_filename)

        # Check if config exists
        if not os.path.exists(config_path):
            raise FileNotFoundError(
                f"Config file not found: {config_path}\n"
                f"Please ensure all V2 Llama configs are created in {self.base_config_dir}\n"
                f"You can copy from evaluate/configs/llama_configs/ and update retriever port to 8002"
            )

        return config_path

    def get_all_required_configs(self) -> list:
        """
        Get list of all required V2 config files.

        Returns:
            List of config filenames that should exist
        """
        return [
            'zero_retrieval.jsonnet',
            'single_splade.jsonnet',
            'single_dense.jsonnet',
            'single_hybrid.jsonnet',
            'multi_hop.jsonnet',
        ]

    def check_configs_exist(self) -> bool:
        """
        Check if all required V2 configs exist.

        Returns:
            True if all configs exist, False otherwise
        """
        all_exist = True
        missing_configs = []

        for config_file in self.get_all_required_configs():
            config_path = os.path.join(self.base_config_dir, config_file)
            if not os.path.exists(config_path):
                print(f"Missing V2 config: {config_path}")
                missing_configs.append(config_file)
                all_exist = False

        if not all_exist:
            print(f"\nTo create missing configs:")
            print(f"1. Copy from baseline: evaluate/configs/llama_configs/")
            print(f"2. Update retriever address to use port 8002")
            print(f"3. Place in: {self.base_config_dir}")

        return all_exist

    def get_baseline_config_dir(self) -> str:
        """Get path to baseline config directory for reference."""
        return "evaluate/configs/llama_configs"
