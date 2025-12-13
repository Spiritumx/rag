"""
Configuration loader for the evaluation pipeline.
"""

import yaml
import os
from pathlib import Path


class ConfigLoader:
    """Load and validate YAML configuration."""

    @staticmethod
    def load_config(config_path='evaluate/config.yaml'):
        """
        Load configuration from YAML file.

        Args:
            config_path: Path to config YAML file

        Returns:
            dict: Configuration dictionary
        """
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

        # Validate required sections
        required_sections = ['classifier', 'llm', 'retriever', 'datasets', 'data', 'outputs']
        for section in required_sections:
            if section not in config:
                raise ValueError(f"Missing required config section: {section}")

        return config

    @staticmethod
    def save_config(config, output_path):
        """Save configuration to YAML file."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False)
