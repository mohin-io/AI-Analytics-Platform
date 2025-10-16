"""
Configuration management for the Unified AI Analytics Platform

This module handles loading and managing configuration settings from files and environment variables.
"""

import os
import yaml
import json
from pathlib import Path
from typing import Any, Dict, Optional
from dataclasses import dataclass, field


@dataclass
class Config:
    """
    Configuration class for the Unified AI Analytics Platform.

    This class manages all configuration settings for the platform, including
    paths, model parameters, API settings, and more. It can load configurations
    from YAML files, JSON files, or environment variables.

    Attributes:
        data_dir: Directory for storing datasets
        model_dir: Directory for saving trained models
        log_dir: Directory for log files
        mlflow_tracking_uri: URI for MLflow tracking server
        random_seed: Random seed for reproducibility
        test_size: Default test set size for train/test splits
        cv_folds: Number of cross-validation folds
        n_jobs: Number of parallel jobs (-1 for all cores)
        optuna_n_trials: Number of trials for hyperparameter optimization
        api_host: API server host
        api_port: API server port
        dashboard_port: Dashboard port
        max_training_time: Maximum training time per model in seconds
        cache_enabled: Whether to cache preprocessing results

    Example:
        >>> config = Config()
        >>> config.load_from_yaml("config/settings.yaml")
        >>> print(config.data_dir)
        'data/raw'
    """

    # Directory paths
    data_dir: str = "data/raw"
    processed_data_dir: str = "data/processed"
    model_dir: str = "models"
    log_dir: str = "logs"
    checkpoint_dir: str = "checkpoints"

    # MLflow settings
    mlflow_tracking_uri: str = "mlruns"
    mlflow_experiment_name: str = "unified_ai_experiments"

    # General ML settings
    random_seed: int = 42
    test_size: float = 0.2
    validation_size: float = 0.1
    cv_folds: int = 5
    n_jobs: int = -1

    # AutoML settings
    optuna_n_trials: int = 50
    optuna_timeout: Optional[int] = 3600  # 1 hour
    early_stopping_rounds: int = 10

    # API settings
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_title: str = "Unified AI Analytics Platform API"
    api_version: str = "v1"

    # Dashboard settings
    dashboard_port: int = 8501
    dashboard_title: str = "Unified AI Analytics Platform"

    # Training settings
    max_training_time: int = 3600  # seconds
    batch_size: int = 32
    epochs: int = 100
    learning_rate: float = 0.001

    # Preprocessing settings
    missing_value_strategy: str = "median"  # mean, median, mode, knn, mice
    scaling_strategy: str = "standard"  # standard, minmax, robust
    encoding_strategy: str = "onehot"  # onehot, label, target

    # Feature engineering
    feature_selection_enabled: bool = True
    feature_selection_method: str = "mutual_info"  # mutual_info, chi2, rfe
    max_features: Optional[int] = None

    # Performance settings
    cache_enabled: bool = True
    verbose: int = 1

    # Model-specific settings
    model_configs: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Initialize directories after dataclass initialization."""
        self._create_directories()

    def _create_directories(self) -> None:
        """
        Create necessary directories if they don't exist.

        This method creates all the directories specified in the configuration
        to ensure the platform has the necessary folder structure for operation.
        """
        dirs = [
            self.data_dir,
            self.processed_data_dir,
            self.model_dir,
            self.log_dir,
            self.checkpoint_dir,
        ]
        for dir_path in dirs:
            Path(dir_path).mkdir(parents=True, exist_ok=True)

    def load_from_yaml(self, config_path: str) -> None:
        """
        Load configuration from a YAML file.

        This method reads a YAML configuration file and updates the configuration
        object with the values from the file. Existing values are overwritten.

        Args:
            config_path: Path to the YAML configuration file

        Example:
            >>> config = Config()
            >>> config.load_from_yaml("config/production.yaml")
        """
        config_file = Path(config_path)
        if not config_file.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        with open(config_file, 'r') as f:
            config_data = yaml.safe_load(f)

        self._update_from_dict(config_data)
        self._create_directories()

    def load_from_json(self, config_path: str) -> None:
        """
        Load configuration from a JSON file.

        Args:
            config_path: Path to the JSON configuration file

        Example:
            >>> config = Config()
            >>> config.load_from_json("config/settings.json")
        """
        config_file = Path(config_path)
        if not config_file.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        with open(config_file, 'r') as f:
            config_data = json.load(f)

        self._update_from_dict(config_data)
        self._create_directories()

    def load_from_env(self, prefix: str = "UNIFIED_AI_") -> None:
        """
        Load configuration from environment variables.

        This method reads environment variables with a specific prefix and
        updates the configuration. Useful for containerized deployments.

        Args:
            prefix: Prefix for environment variables (default: "UNIFIED_AI_")

        Example:
            >>> os.environ["UNIFIED_AI_API_PORT"] = "8080"
            >>> config = Config()
            >>> config.load_from_env()
            >>> print(config.api_port)
            8080
        """
        for key, value in os.environ.items():
            if key.startswith(prefix):
                config_key = key[len(prefix):].lower()
                if hasattr(self, config_key):
                    # Convert to appropriate type
                    current_value = getattr(self, config_key)
                    if isinstance(current_value, bool):
                        setattr(self, config_key, value.lower() in ('true', '1', 'yes'))
                    elif isinstance(current_value, int):
                        setattr(self, config_key, int(value))
                    elif isinstance(current_value, float):
                        setattr(self, config_key, float(value))
                    else:
                        setattr(self, config_key, value)

    def _update_from_dict(self, config_dict: Dict[str, Any]) -> None:
        """
        Update configuration from a dictionary.

        Args:
            config_dict: Dictionary containing configuration values
        """
        for key, value in config_dict.items():
            if hasattr(self, key):
                setattr(self, key, value)

    def save_to_yaml(self, output_path: str) -> None:
        """
        Save current configuration to a YAML file.

        Args:
            output_path: Path where the YAML file will be saved

        Example:
            >>> config = Config()
            >>> config.api_port = 9000
            >>> config.save_to_yaml("config/custom_config.yaml")
        """
        config_dict = self.to_dict()
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)

    def save_to_json(self, output_path: str) -> None:
        """
        Save current configuration to a JSON file.

        Args:
            output_path: Path where the JSON file will be saved
        """
        config_dict = self.to_dict()
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, 'w') as f:
            json.dump(config_dict, f, indent=2)

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to a dictionary.

        Returns:
            Dictionary representation of the configuration
        """
        return {
            key: value for key, value in self.__dict__.items()
            if not key.startswith('_')
        }

    def __repr__(self) -> str:
        """String representation of the configuration."""
        config_items = [f"{k}={v}" for k, v in self.to_dict().items()]
        return f"Config({', '.join(config_items[:5])}...)"
