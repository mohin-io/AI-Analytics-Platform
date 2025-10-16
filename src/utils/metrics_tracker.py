"""
Metrics tracking utility for the Unified AI Analytics Platform

This module provides utilities for tracking and storing experiment metrics.
"""

import time
from typing import Any, Dict, List, Optional
from datetime import datetime
import pandas as pd
from pathlib import Path


class MetricsTracker:
    """
    Track and store metrics for machine learning experiments.

    This class provides functionality to track various metrics during model
    training and evaluation, including execution time, accuracy, loss, and
    custom metrics. It can save metrics to CSV files for later analysis.

    Attributes:
        metrics: Dictionary storing all tracked metrics
        start_time: Timestamp when tracking started
        experiment_name: Name of the current experiment

    Example:
        >>> tracker = MetricsTracker(experiment_name="xgboost_experiment")
        >>> tracker.start_timer()
        >>> # ... train model ...
        >>> tracker.log_metric("accuracy", 0.95)
        >>> tracker.log_metric("f1_score", 0.93)
        >>> elapsed = tracker.stop_timer()
        >>> tracker.save_metrics("results/metrics.csv")
    """

    def __init__(self, experiment_name: str = "default_experiment"):
        """
        Initialize the MetricsTracker.

        Args:
            experiment_name: Name to identify this experiment
        """
        self.experiment_name = experiment_name
        self.metrics: Dict[str, List[Any]] = {
            "timestamp": [],
            "experiment_name": [],
        }
        self.start_time: Optional[float] = None
        self.experiment_start_time = datetime.now()

    def start_timer(self) -> None:
        """
        Start the timer for measuring execution time.

        This method records the current time, which can be used later to
        calculate elapsed time using stop_timer().

        Example:
            >>> tracker = MetricsTracker()
            >>> tracker.start_timer()
            >>> # ... perform operations ...
            >>> elapsed = tracker.stop_timer()
        """
        self.start_time = time.time()

    def stop_timer(self) -> float:
        """
        Stop the timer and return elapsed time.

        Returns:
            Elapsed time in seconds since start_timer() was called

        Raises:
            RuntimeError: If timer was not started

        Example:
            >>> tracker = MetricsTracker()
            >>> tracker.start_timer()
            >>> time.sleep(2)
            >>> elapsed = tracker.stop_timer()
            >>> print(f"Execution took {elapsed:.2f} seconds")
        """
        if self.start_time is None:
            raise RuntimeError("Timer was not started. Call start_timer() first.")

        elapsed = time.time() - self.start_time
        self.log_metric("execution_time", elapsed)
        self.start_time = None
        return elapsed

    def log_metric(self, metric_name: str, value: Any) -> None:
        """
        Log a single metric value.

        This method records a metric value with a timestamp. If the metric
        doesn't exist yet, it creates a new entry. The method automatically
        handles the timestamp and experiment name.

        Args:
            metric_name: Name of the metric (e.g., "accuracy", "loss")
            value: Value to log (numeric or string)

        Example:
            >>> tracker = MetricsTracker("model_v1")
            >>> tracker.log_metric("accuracy", 0.95)
            >>> tracker.log_metric("precision", 0.92)
            >>> tracker.log_metric("model_type", "RandomForest")
        """
        if metric_name not in self.metrics:
            self.metrics[metric_name] = []

        # Ensure all metric lists have the same length by padding with None
        current_length = len(self.metrics["timestamp"])
        target_length = current_length + 1

        for key in self.metrics:
            while len(self.metrics[key]) < target_length - 1:
                self.metrics[key].append(None)

        # Add the new metric
        self.metrics["timestamp"].append(datetime.now())
        self.metrics["experiment_name"].append(self.experiment_name)
        self.metrics[metric_name].append(value)

    def log_metrics(self, metrics_dict: Dict[str, Any]) -> None:
        """
        Log multiple metrics at once.

        This is a convenience method to log multiple metrics with a single
        call, useful when you have a dictionary of metrics from model evaluation.

        Args:
            metrics_dict: Dictionary of metric names and values

        Example:
            >>> tracker = MetricsTracker()
            >>> results = {"accuracy": 0.95, "f1": 0.93, "precision": 0.94}
            >>> tracker.log_metrics(results)
        """
        for metric_name, value in metrics_dict.items():
            self.log_metric(metric_name, value)

    def get_metric(self, metric_name: str) -> List[Any]:
        """
        Retrieve all logged values for a specific metric.

        Args:
            metric_name: Name of the metric to retrieve

        Returns:
            List of all values logged for this metric

        Example:
            >>> tracker = MetricsTracker()
            >>> tracker.log_metric("loss", 0.5)
            >>> tracker.log_metric("loss", 0.3)
            >>> losses = tracker.get_metric("loss")
            >>> print(losses)
            [0.5, 0.3]
        """
        return self.metrics.get(metric_name, [])

    def get_latest_metric(self, metric_name: str) -> Optional[Any]:
        """
        Get the most recently logged value for a metric.

        Args:
            metric_name: Name of the metric

        Returns:
            The most recent value, or None if metric doesn't exist

        Example:
            >>> tracker = MetricsTracker()
            >>> tracker.log_metric("accuracy", 0.90)
            >>> tracker.log_metric("accuracy", 0.95)
            >>> latest = tracker.get_latest_metric("accuracy")
            >>> print(latest)
            0.95
        """
        values = self.get_metric(metric_name)
        return values[-1] if values else None

    def get_all_metrics(self) -> Dict[str, List[Any]]:
        """
        Get all tracked metrics.

        Returns:
            Dictionary containing all metrics and their values

        Example:
            >>> tracker = MetricsTracker()
            >>> tracker.log_metric("accuracy", 0.95)
            >>> tracker.log_metric("loss", 0.1)
            >>> all_metrics = tracker.get_all_metrics()
            >>> print(all_metrics.keys())
            dict_keys(['timestamp', 'experiment_name', 'accuracy', 'loss'])
        """
        return self.metrics.copy()

    def to_dataframe(self) -> pd.DataFrame:
        """
        Convert tracked metrics to a pandas DataFrame.

        This method creates a DataFrame where each row represents a logged
        entry with all associated metrics. Useful for analysis and visualization.

        Returns:
            DataFrame containing all metrics

        Example:
            >>> tracker = MetricsTracker()
            >>> tracker.log_metric("accuracy", 0.95)
            >>> tracker.log_metric("loss", 0.1)
            >>> df = tracker.to_dataframe()
            >>> print(df)
        """
        # Ensure all columns have the same length
        max_length = max(len(v) for v in self.metrics.values()) if self.metrics else 0

        aligned_metrics = {}
        for key, values in self.metrics.items():
            aligned_metrics[key] = values + [None] * (max_length - len(values))

        return pd.DataFrame(aligned_metrics)

    def save_metrics(
        self,
        file_path: str,
        format: str = "csv"
    ) -> None:
        """
        Save tracked metrics to a file.

        This method exports all tracked metrics to a file in the specified
        format. Supported formats include CSV and JSON.

        Args:
            file_path: Path where the file will be saved
            format: Output format ("csv" or "json")

        Example:
            >>> tracker = MetricsTracker("experiment_1")
            >>> tracker.log_metric("accuracy", 0.95)
            >>> tracker.save_metrics("results/experiment_1_metrics.csv")
        """
        output_path = Path(file_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        df = self.to_dataframe()

        if format.lower() == "csv":
            df.to_csv(file_path, index=False)
        elif format.lower() == "json":
            df.to_json(file_path, orient="records", date_format="iso", indent=2)
        else:
            raise ValueError(f"Unsupported format: {format}. Use 'csv' or 'json'.")

    def load_metrics(self, file_path: str, format: str = "csv") -> None:
        """
        Load metrics from a file.

        This method loads previously saved metrics from a file, allowing you
        to continue tracking or analyze past experiments.

        Args:
            file_path: Path to the metrics file
            format: File format ("csv" or "json")

        Example:
            >>> tracker = MetricsTracker()
            >>> tracker.load_metrics("results/previous_metrics.csv")
            >>> print(tracker.get_latest_metric("accuracy"))
        """
        if format.lower() == "csv":
            df = pd.read_csv(file_path)
        elif format.lower() == "json":
            df = pd.read_json(file_path)
        else:
            raise ValueError(f"Unsupported format: {format}. Use 'csv' or 'json'.")

        # Convert DataFrame back to metrics dictionary
        self.metrics = {col: df[col].tolist() for col in df.columns}

    def reset(self) -> None:
        """
        Reset all tracked metrics.

        This method clears all tracked metrics and resets the tracker to its
        initial state. Useful when starting a new experiment.

        Example:
            >>> tracker = MetricsTracker()
            >>> tracker.log_metric("accuracy", 0.95)
            >>> tracker.reset()
            >>> print(len(tracker.get_all_metrics()))
            2  # Only timestamp and experiment_name remain
        """
        self.metrics = {
            "timestamp": [],
            "experiment_name": [],
        }
        self.start_time = None
        self.experiment_start_time = datetime.now()

    def summary(self) -> Dict[str, Any]:
        """
        Get a summary of tracked metrics.

        This method provides statistical summaries (mean, min, max, etc.) for
        all numeric metrics, giving a quick overview of the experiment.

        Returns:
            Dictionary containing metric summaries

        Example:
            >>> tracker = MetricsTracker()
            >>> for acc in [0.90, 0.92, 0.95]:
            ...     tracker.log_metric("accuracy", acc)
            >>> summary = tracker.summary()
            >>> print(summary["accuracy"])
            {'mean': 0.923..., 'min': 0.90, 'max': 0.95, ...}
        """
        df = self.to_dataframe()
        summary_dict = {}

        for col in df.columns:
            if col not in ["timestamp", "experiment_name"]:
                try:
                    numeric_values = pd.to_numeric(df[col], errors='coerce').dropna()
                    if len(numeric_values) > 0:
                        summary_dict[col] = {
                            "mean": float(numeric_values.mean()),
                            "std": float(numeric_values.std()),
                            "min": float(numeric_values.min()),
                            "max": float(numeric_values.max()),
                            "count": int(len(numeric_values)),
                        }
                except:
                    pass

        return summary_dict

    def __repr__(self) -> str:
        """String representation of the tracker."""
        num_metrics = len(self.metrics) - 2  # Exclude timestamp and experiment_name
        num_entries = len(self.metrics.get("timestamp", []))
        return f"MetricsTracker(experiment='{self.experiment_name}', metrics={num_metrics}, entries={num_entries})"
