"""
Model comparison and benchmarking for the Unified AI Analytics Platform

This module provides tools to compare multiple models, create leaderboards,
and visualize comparative performance.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


class ModelComparator:
    """
    Compare multiple models and generate comparative analysis.

    This class facilitates comparing multiple trained models on the same
    dataset, creating leaderboards, and visualizing comparative performance
    across different metrics.

    Example:
        >>> comparator = ModelComparator()
        >>> results = {
        ...     'Random Forest': {'accuracy': 0.95, 'f1_score': 0.94},
        ...     'XGBoost': {'accuracy': 0.96, 'f1_score': 0.95},
        ...     'Logistic Regression': {'accuracy': 0.92, 'f1_score': 0.91}
        ... }
        >>> comparison_df = comparator.create_comparison_table(results)
        >>> comparator.plot_metric_comparison(comparison_df, 'accuracy')
    """

    def __init__(self, output_dir: str = 'comparison_results'):
        """
        Initialize ModelComparator.

        Args:
            output_dir: Directory to save comparison visualizations
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def create_comparison_table(
        self,
        results: Dict[str, Dict[str, float]]
    ) -> pd.DataFrame:
        """
        Create a comparison table from model results.

        Converts a dictionary of model results into a pandas DataFrame
        for easy comparison and analysis. Automatically sorts models by
        the first metric.

        Args:
            results: Dictionary with model names as keys and metrics dicts as values
                    Example: {'Model1': {'accuracy': 0.95, 'f1': 0.94}, ...}

        Returns:
            DataFrame with models as rows and metrics as columns

        Example:
            >>> results = {
            ...     'Random Forest': {'accuracy': 0.95, 'f1_score': 0.94, 'training_time': 5.2},
            ...     'XGBoost': {'accuracy': 0.96, 'f1_score': 0.95, 'training_time': 3.8}
            ... }
            >>> df = comparator.create_comparison_table(results)
            >>> print(df)
        """
        comparison_df = pd.DataFrame(results).T
        comparison_df.index.name = 'Model'

        # Round numerical values for better display
        comparison_df = comparison_df.round(4)

        return comparison_df

    def rank_models(
        self,
        comparison_df: pd.DataFrame,
        metric: str,
        ascending: bool = False
    ) -> pd.DataFrame:
        """
        Rank models by a specific metric.

        Sorts models based on a chosen metric and adds a rank column.
        By default, higher values are better (use ascending=True for
        metrics like error where lower is better).

        Args:
            comparison_df: DataFrame from create_comparison_table()
            metric: Metric to rank by
            ascending: If True, lower values rank higher

        Returns:
            DataFrame sorted by the metric with rank column

        Example:
            >>> ranked = comparator.rank_models(comparison_df, 'accuracy')
            >>> print(ranked[['accuracy', 'rank']])
        """
        if metric not in comparison_df.columns:
            raise ValueError(f"Metric '{metric}' not found in results")

        ranked_df = comparison_df.sort_values(metric, ascending=ascending).copy()
        ranked_df['rank'] = range(1, len(ranked_df) + 1)

        return ranked_df

    def get_best_model(
        self,
        comparison_df: pd.DataFrame,
        metric: str,
        ascending: bool = False
    ) -> str:
        """
        Get the name of the best model for a specific metric.

        Args:
            comparison_df: DataFrame from create_comparison_table()
            metric: Metric to optimize
            ascending: If True, lower is better

        Returns:
            Name of the best performing model

        Example:
            >>> best_model = comparator.get_best_model(comparison_df, 'f1_score')
            >>> print(f"Best model: {best_model}")
        """
        ranked = self.rank_models(comparison_df, metric, ascending)
        return ranked.index[0]

    def plot_metric_comparison(
        self,
        comparison_df: pd.DataFrame,
        metric: str,
        save_path: Optional[str] = None,
        figsize: tuple = (10, 6)
    ) -> None:
        """
        Plot comparison of models for a specific metric.

        Creates a bar chart comparing all models on a single metric.
        Models are sorted by performance for easy visualization.

        Args:
            comparison_df: DataFrame from create_comparison_table()
            metric: Metric to visualize
            save_path: Optional path to save the plot
            figsize: Figure size

        Example:
            >>> comparator.plot_metric_comparison(
            ...     comparison_df,
            ...     'accuracy',
            ...     save_path='accuracy_comparison.png'
            ... )
        """
        if metric not in comparison_df.columns:
            raise ValueError(f"Metric '{metric}' not found in results")

        # Sort by metric
        sorted_df = comparison_df.sort_values(metric, ascending=False)

        plt.figure(figsize=figsize)
        bars = plt.bar(range(len(sorted_df)), sorted_df[metric])

        # Color bars - best model in green, others in blue
        bars[0].set_color('green')
        for bar in bars[1:]:
            bar.set_color('steelblue')

        plt.xticks(range(len(sorted_df)), sorted_df.index, rotation=45, ha='right')
        plt.ylabel(metric.replace('_', ' ').title())
        plt.title(f'Model Comparison: {metric.replace("_", " ").title()}')
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()

        if save_path:
            plt.savefig(self.output_dir / save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def plot_multi_metric_comparison(
        self,
        comparison_df: pd.DataFrame,
        metrics: List[str],
        save_path: Optional[str] = None,
        figsize: tuple = (12, 6)
    ) -> None:
        """
        Plot comparison across multiple metrics.

        Creates a grouped bar chart showing how each model performs
        across multiple metrics. Useful for understanding trade-offs.

        Args:
            comparison_df: DataFrame from create_comparison_table()
            metrics: List of metrics to compare
            save_path: Optional path to save the plot
            figsize: Figure size

        Example:
            >>> comparator.plot_multi_metric_comparison(
            ...     comparison_df,
            ...     metrics=['accuracy', 'precision', 'recall', 'f1_score']
            ... )
        """
        # Validate metrics
        for metric in metrics:
            if metric not in comparison_df.columns:
                raise ValueError(f"Metric '{metric}' not found in results")

        # Select only specified metrics
        plot_df = comparison_df[metrics]

        # Create grouped bar chart
        ax = plot_df.plot(kind='bar', figsize=figsize, width=0.8)
        plt.title('Multi-Metric Model Comparison')
        plt.xlabel('Model')
        plt.ylabel('Score')
        plt.xticks(rotation=45, ha='right')
        plt.legend(title='Metrics', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()

        if save_path:
            plt.savefig(self.output_dir / save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def plot_heatmap(
        self,
        comparison_df: pd.DataFrame,
        save_path: Optional[str] = None,
        figsize: tuple = (12, 8)
    ) -> None:
        """
        Plot heatmap of all models and metrics.

        Creates a color-coded heatmap showing model performance across
        all metrics. Darker colors indicate better performance.

        Args:
            comparison_df: DataFrame from create_comparison_table()
            save_path: Optional path to save the plot
            figsize: Figure size

        Example:
            >>> comparator.plot_heatmap(comparison_df, 'comparison_heatmap.png')
        """
        plt.figure(figsize=figsize)

        # Normalize data for better color representation
        normalized_df = (comparison_df - comparison_df.min()) / (comparison_df.max() - comparison_df.min())

        sns.heatmap(
            normalized_df.T,
            annot=comparison_df.T,
            fmt='.3f',
            cmap='RdYlGn',
            cbar_kws={'label': 'Normalized Score'},
            linewidths=0.5
        )

        plt.title('Model Performance Heatmap')
        plt.xlabel('Model')
        plt.ylabel('Metric')
        plt.tight_layout()

        if save_path:
            plt.savefig(self.output_dir / save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def plot_training_time_comparison(
        self,
        results: Dict[str, Dict[str, Any]],
        save_path: Optional[str] = None,
        figsize: tuple = (10, 6)
    ) -> None:
        """
        Plot training time comparison across models.

        Visualizes how long each model took to train, helping identify
        efficiency trade-offs.

        Args:
            results: Dictionary with model names and metrics including training_time
            save_path: Optional path to save the plot
            figsize: Figure size

        Example:
            >>> results = {
            ...     'Random Forest': {'training_time': 5.2},
            ...     'XGBoost': {'training_time': 3.8},
            ...     'SVM': {'training_time': 45.6}
            ... }
            >>> comparator.plot_training_time_comparison(results)
        """
        # Extract training times
        training_times = {
            model: metrics.get('training_time', 0)
            for model, metrics in results.items()
        }

        # Sort by training time
        sorted_times = dict(sorted(training_times.items(), key=lambda x: x[1]))

        plt.figure(figsize=figsize)
        bars = plt.barh(list(sorted_times.keys()), list(sorted_times.values()))

        # Color code - green for fastest, red for slowest
        colors = plt.cm.RdYlGn_r(np.linspace(0.3, 0.9, len(bars)))
        for bar, color in zip(bars, colors):
            bar.set_color(color)

        plt.xlabel('Training Time (seconds)')
        plt.title('Model Training Time Comparison')
        plt.grid(axis='x', alpha=0.3)
        plt.tight_layout()

        if save_path:
            plt.savefig(self.output_dir / save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def generate_leaderboard(
        self,
        comparison_df: pd.DataFrame,
        primary_metric: str,
        ascending: bool = False
    ) -> pd.DataFrame:
        """
        Generate a ranked leaderboard of models.

        Creates a comprehensive leaderboard showing model rankings,
        highlighting the best model and providing all metrics.

        Args:
            comparison_df: DataFrame from create_comparison_table()
            primary_metric: Main metric to rank by
            ascending: If True, lower values rank higher

        Returns:
            Leaderboard DataFrame with rankings

        Example:
            >>> leaderboard = comparator.generate_leaderboard(
            ...     comparison_df,
            ...     primary_metric='f1_score'
            ... )
            >>> print(leaderboard)
        """
        leaderboard = self.rank_models(comparison_df, primary_metric, ascending)

        # Add medal emoji for top 3
        def add_medal(rank):
            if rank == 1:
                return "1st"
            elif rank == 2:
                return "2nd"
            elif rank == 3:
                return "3rd"
            else:
                return f"{rank}th"

        leaderboard['Rank'] = leaderboard['rank'].apply(add_medal)
        leaderboard = leaderboard.drop('rank', axis=1)

        # Reorder columns to put Rank first
        cols = ['Rank'] + [col for col in leaderboard.columns if col != 'Rank']
        leaderboard = leaderboard[cols]

        return leaderboard

    def export_comparison(
        self,
        comparison_df: pd.DataFrame,
        output_path: str,
        format: str = 'csv'
    ) -> None:
        """
        Export comparison results to file.

        Args:
            comparison_df: DataFrame to export
            output_path: Path to save the file
            format: Output format ('csv', 'excel', 'html', 'markdown')

        Example:
            >>> comparator.export_comparison(
            ...     comparison_df,
            ...     'model_comparison.csv',
            ...     format='csv'
            ... )
        """
        output_file = self.output_dir / output_path

        if format == 'csv':
            comparison_df.to_csv(output_file)
        elif format == 'excel':
            comparison_df.to_excel(output_file)
        elif format == 'html':
            comparison_df.to_html(output_file)
        elif format == 'markdown':
            comparison_df.to_markdown(output_file)
        else:
            raise ValueError(f"Unsupported format: {format}")

    def summary_statistics(
        self,
        comparison_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Generate summary statistics across all models.

        Computes mean, std, min, and max for each metric across all models.

        Args:
            comparison_df: DataFrame from create_comparison_table()

        Returns:
            DataFrame with summary statistics

        Example:
            >>> stats = comparator.summary_statistics(comparison_df)
            >>> print(stats)
        """
        return comparison_df.describe()
