"""
Performance Visualization Tools

Visualize model performance metrics and comparisons.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, List, Dict


class PerformanceVisualizer:
    """
    Visualize model performance metrics.
    """

    def __init__(self):
        """Initialize performance visualizer."""
        sns.set_style('whitegrid')

    def plot_metrics_over_time(self, metrics_df: pd.DataFrame,
                              metrics: List[str],
                              save_path: Optional[str] = None):
        """
        Plot metrics over time/epochs.

        Args:
            metrics_df: DataFrame with timestamp and metrics
            metrics: List of metric names to plot
            save_path: Path to save plot
        """
        n_metrics = len(metrics)
        fig, axes = plt.subplots(n_metrics, 1, figsize=(12, 4 * n_metrics))
        if n_metrics == 1:
            axes = [axes]

        for i, metric in enumerate(metrics):
            if metric in metrics_df.columns:
                axes[i].plot(metrics_df.index, metrics_df[metric],
                           marker='o', linewidth=2, markersize=4)
                axes[i].set_ylabel(metric.upper())
                axes[i].set_title(f'{metric.upper()} Over Time')
                axes[i].grid(alpha=0.3)

        axes[-1].set_xlabel('Time/Epoch')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()

    def plot_metric_distribution(self, metrics: Dict[str, List[float]],
                                save_path: Optional[str] = None):
        """
        Plot distribution of metrics across runs.

        Args:
            metrics: Dict mapping metric names to lists of values
            save_path: Path to save plot
        """
        data = []
        for metric_name, values in metrics.items():
            for value in values:
                data.append({'Metric': metric_name, 'Value': value})

        df = pd.DataFrame(data)

        plt.figure(figsize=(10, 6))
        sns.violinplot(data=df, x='Metric', y='Value')
        plt.title('Metric Distributions Across Runs')
        plt.xticks(rotation=45)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()

    def plot_training_history(self, history: Dict[str, List[float]],
                             save_path: Optional[str] = None):
        """
        Plot training history with train/val curves.

        Args:
            history: Dict with 'train_loss', 'val_loss', etc.
            save_path: Path to save plot
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Loss curves
        if 'train_loss' in history and 'val_loss' in history:
            axes[0].plot(history['train_loss'], label='Train', linewidth=2)
            axes[0].plot(history['val_loss'], label='Validation', linewidth=2)
            axes[0].set_xlabel('Epoch')
            axes[0].set_ylabel('Loss')
            axes[0].set_title('Training and Validation Loss')
            axes[0].legend()
            axes[0].grid(alpha=0.3)

        # Accuracy curves
        if 'train_acc' in history and 'val_acc' in history:
            axes[1].plot(history['train_acc'], label='Train', linewidth=2)
            axes[1].plot(history['val_acc'], label='Validation', linewidth=2)
            axes[1].set_xlabel('Epoch')
            axes[1].set_ylabel('Accuracy')
            axes[1].set_title('Training and Validation Accuracy')
            axes[1].legend()
            axes[1].grid(alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
