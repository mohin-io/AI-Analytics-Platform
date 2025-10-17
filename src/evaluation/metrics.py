"""
Model evaluation metrics for the Unified AI Analytics Platform

This module provides comprehensive evaluation metrics for classification,
regression, and clustering tasks, along with visualization utilities.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List, Union, Tuple
from sklearn.metrics import (
    # Classification metrics
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, precision_recall_curve,
    confusion_matrix, classification_report, log_loss,
    matthews_corrcoef, cohen_kappa_score,
    # Regression metrics
    mean_absolute_error, mean_squared_error, r2_score,
    mean_absolute_percentage_error, median_absolute_error,
    explained_variance_score,
    # Clustering metrics
    silhouette_score, davies_bouldin_score, calinski_harabasz_score
)
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


class MetricsCalculator:
    """
    Calculate evaluation metrics for different ML tasks.

    This class provides methods to compute various performance metrics
    for classification, regression, and clustering tasks. It automatically
    selects appropriate metrics based on the task type.

    Example:
        >>> calculator = MetricsCalculator()
        >>> metrics = calculator.calculate_classification_metrics(y_true, y_pred, y_proba)
        >>> print(metrics)
    """

    @staticmethod
    def calculate_classification_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: Optional[np.ndarray] = None,
        average: str = 'weighted'
    ) -> Dict[str, float]:
        """
        Calculate comprehensive classification metrics.

        This method computes all relevant classification metrics including
        accuracy, precision, recall, F1-score, ROC-AUC, and more. It handles
        both binary and multi-class classification.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_proba: Predicted probabilities (optional, needed for ROC-AUC)
            average: Averaging method for multi-class ('micro', 'macro', 'weighted')

        Returns:
            Dictionary containing all calculated metrics

        Example:
            >>> y_true = np.array([0, 1, 1, 0, 1])
            >>> y_pred = np.array([0, 1, 0, 0, 1])
            >>> metrics = MetricsCalculator.calculate_classification_metrics(y_true, y_pred)
            >>> print(f"Accuracy: {metrics['accuracy']:.3f}")
        """
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average=average, zero_division=0),
            'recall': recall_score(y_true, y_pred, average=average, zero_division=0),
            'f1_score': f1_score(y_true, y_pred, average=average, zero_division=0),
            'matthews_corrcoef': matthews_corrcoef(y_true, y_pred),
            'cohen_kappa': cohen_kappa_score(y_true, y_pred),
        }

        # Add probability-based metrics if available
        if y_proba is not None:
            try:
                metrics['log_loss'] = log_loss(y_true, y_proba)

                # ROC-AUC for binary classification
                n_classes = len(np.unique(y_true))
                if n_classes == 2:
                    metrics['roc_auc'] = roc_auc_score(y_true, y_proba[:, 1])
                else:
                    # Multi-class ROC-AUC
                    metrics['roc_auc_ovr'] = roc_auc_score(
                        y_true, y_proba, multi_class='ovr', average=average
                    )
                    metrics['roc_auc_ovo'] = roc_auc_score(
                        y_true, y_proba, multi_class='ovo', average=average
                    )
            except Exception as e:
                pass  # Skip if probabilities are not compatible

        return metrics

    @staticmethod
    def calculate_regression_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> Dict[str, float]:
        """
        Calculate comprehensive regression metrics.

        Computes all relevant regression metrics including MAE, MSE, RMSE,
        R², MAPE, and more. These metrics help assess prediction accuracy
        and model performance.

        Args:
            y_true: True values
            y_pred: Predicted values

        Returns:
            Dictionary containing all calculated metrics

        Example:
            >>> y_true = np.array([3, -0.5, 2, 7])
            >>> y_pred = np.array([2.5, 0.0, 2, 8])
            >>> metrics = MetricsCalculator.calculate_regression_metrics(y_true, y_pred)
            >>> print(f"R²: {metrics['r2']:.3f}")
        """
        metrics = {
            'mae': mean_absolute_error(y_true, y_pred),
            'mse': mean_squared_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'r2': r2_score(y_true, y_pred),
            'median_absolute_error': median_absolute_error(y_true, y_pred),
            'explained_variance': explained_variance_score(y_true, y_pred),
        }

        # Add MAPE if no zeros in y_true
        if not np.any(y_true == 0):
            metrics['mape'] = mean_absolute_percentage_error(y_true, y_pred)

        # Add additional metrics
        residuals = y_true - y_pred
        metrics['mean_residual'] = np.mean(residuals)
        metrics['std_residual'] = np.std(residuals)

        return metrics

    @staticmethod
    def calculate_clustering_metrics(
        X: np.ndarray,
        labels: np.ndarray
    ) -> Dict[str, float]:
        """
        Calculate clustering evaluation metrics.

        Computes metrics that assess the quality of clustering without
        ground truth labels. Includes silhouette score, Davies-Bouldin index,
        and Calinski-Harabasz score.

        Args:
            X: Feature matrix
            labels: Cluster labels

        Returns:
            Dictionary containing clustering metrics

        Example:
            >>> from sklearn.cluster import KMeans
            >>> X = np.random.rand(100, 5)
            >>> kmeans = KMeans(n_clusters=3).fit(X)
            >>> metrics = MetricsCalculator.calculate_clustering_metrics(X, kmeans.labels_)
        """
        metrics = {
            'silhouette_score': silhouette_score(X, labels),
            'davies_bouldin_score': davies_bouldin_score(X, labels),
            'calinski_harabasz_score': calinski_harabasz_score(X, labels),
            'n_clusters': len(np.unique(labels)),
        }

        return metrics


class Evaluator:
    """
    Comprehensive model evaluation with metrics and visualizations.

    This class provides a unified interface for evaluating ML models,
    computing metrics, and generating visualizations. It automatically
    detects the task type and applies appropriate evaluation methods.

    Attributes:
        task: Type of ML task ('classification', 'regression', 'clustering')
        output_dir: Directory for saving visualizations

    Example:
        >>> evaluator = Evaluator(task='classification')
        >>> metrics = evaluator.evaluate(y_true, y_pred, y_proba)
        >>> evaluator.plot_confusion_matrix(y_true, y_pred, save_path='cm.png')
    """

    def __init__(
        self,
        task: str = 'classification',
        output_dir: str = 'evaluation_results'
    ):
        """
        Initialize the Evaluator.

        Args:
            task: Type of task ('classification', 'regression', 'clustering')
            output_dir: Directory to save visualizations
        """
        self.task = task
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.calculator = MetricsCalculator()

    def evaluate(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: Optional[np.ndarray] = None,
        X: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Evaluate model performance with appropriate metrics.

        Automatically selects and computes metrics based on the task type.
        For classification, includes accuracy, F1, ROC-AUC, etc. For regression,
        includes MAE, RMSE, R², etc.

        Args:
            y_true: True labels/values
            y_pred: Predicted labels/values
            y_proba: Predicted probabilities (for classification)
            X: Feature matrix (for clustering)

        Returns:
            Dictionary of metrics

        Example:
            >>> evaluator = Evaluator(task='classification')
            >>> metrics = evaluator.evaluate(y_test, predictions, probabilities)
            >>> for metric, value in metrics.items():
            ...     print(f"{metric}: {value:.4f}")
        """
        if self.task == 'classification':
            return self.calculator.calculate_classification_metrics(
                y_true, y_pred, y_proba
            )
        elif self.task == 'regression':
            return self.calculator.calculate_regression_metrics(y_true, y_pred)
        elif self.task == 'clustering':
            if X is None:
                raise ValueError("Feature matrix X required for clustering evaluation")
            return self.calculator.calculate_clustering_metrics(X, y_pred)
        else:
            raise ValueError(f"Unknown task: {self.task}")

    def plot_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        labels: Optional[List[str]] = None,
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (8, 6)
    ) -> None:
        """
        Plot confusion matrix heatmap.

        Creates a visual representation of the confusion matrix showing
        true positives, false positives, true negatives, and false negatives.
        Useful for understanding classification errors.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            labels: Class labels for display
            save_path: Path to save the plot
            figsize: Figure size

        Example:
            >>> evaluator = Evaluator(task='classification')
            >>> evaluator.plot_confusion_matrix(
            ...     y_test, predictions,
            ...     labels=['Class A', 'Class B'],
            ...     save_path='confusion_matrix.png'
            ... )
        """
        cm = confusion_matrix(y_true, y_pred)

        plt.figure(figsize=figsize)
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=labels if labels else 'auto',
            yticklabels=labels if labels else 'auto'
        )
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()

        if save_path:
            plt.savefig(self.output_dir / save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def plot_roc_curve(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (8, 6)
    ) -> None:
        """
        Plot ROC curve for binary classification.

        The ROC curve shows the trade-off between true positive rate and
        false positive rate at different classification thresholds. The area
        under the curve (AUC) indicates overall model performance.

        Args:
            y_true: True binary labels
            y_proba: Predicted probabilities for positive class
            save_path: Path to save the plot
            figsize: Figure size

        Example:
            >>> evaluator = Evaluator(task='classification')
            >>> evaluator.plot_roc_curve(y_test, probabilities[:, 1], 'roc.png')
        """
        fpr, tpr, thresholds = roc_curve(y_true, y_proba)
        roc_auc = roc_auc_score(y_true, y_proba)

        plt.figure(figsize=figsize)
        plt.plot(fpr, tpr, color='darkorange', lw=2,
                label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--',
                label='Random Classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.grid(alpha=0.3)
        plt.tight_layout()

        if save_path:
            plt.savefig(self.output_dir / save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def plot_precision_recall_curve(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (8, 6)
    ) -> None:
        """
        Plot precision-recall curve.

        Shows the trade-off between precision and recall for different
        thresholds. Particularly useful for imbalanced datasets.

        Args:
            y_true: True binary labels
            y_proba: Predicted probabilities
            save_path: Path to save the plot
            figsize: Figure size
        """
        precision, recall, thresholds = precision_recall_curve(y_true, y_proba)

        plt.figure(figsize=figsize)
        plt.plot(recall, precision, color='blue', lw=2)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.grid(alpha=0.3)
        plt.tight_layout()

        if save_path:
            plt.savefig(self.output_dir / save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def plot_residuals(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (12, 5)
    ) -> None:
        """
        Plot residual analysis for regression.

        Creates two plots:
        1. Residuals vs Predicted values (check for heteroscedasticity)
        2. Residuals distribution (check for normality)

        Args:
            y_true: True values
            y_pred: Predicted values
            save_path: Path to save the plot
            figsize: Figure size
        """
        residuals = y_true - y_pred

        fig, axes = plt.subplots(1, 2, figsize=figsize)

        # Residuals vs Predicted
        axes[0].scatter(y_pred, residuals, alpha=0.5)
        axes[0].axhline(y=0, color='r', linestyle='--')
        axes[0].set_xlabel('Predicted Values')
        axes[0].set_ylabel('Residuals')
        axes[0].set_title('Residuals vs Predicted Values')
        axes[0].grid(alpha=0.3)

        # Residuals distribution
        axes[1].hist(residuals, bins=30, edgecolor='black', alpha=0.7)
        axes[1].set_xlabel('Residuals')
        axes[1].set_ylabel('Frequency')
        axes[1].set_title('Distribution of Residuals')
        axes[1].grid(alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(self.output_dir / save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def plot_actual_vs_predicted(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (8, 8)
    ) -> None:
        """
        Plot actual vs predicted values for regression.

        Creates a scatter plot with a diagonal line showing perfect predictions.
        Points closer to the line indicate better predictions.

        Args:
            y_true: True values
            y_pred: Predicted values
            save_path: Path to save the plot
            figsize: Figure size
        """
        plt.figure(figsize=figsize)
        plt.scatter(y_true, y_pred, alpha=0.5)

        # Perfect prediction line
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)

        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.title('Actual vs Predicted Values')
        plt.grid(alpha=0.3)
        plt.tight_layout()

        if save_path:
            plt.savefig(self.output_dir / save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def plot_feature_importance(
        self,
        feature_names: List[str],
        importance_values: np.ndarray,
        top_n: int = 20,
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (10, 8)
    ) -> None:
        """
        Plot feature importance.

        Creates a horizontal bar chart showing the most important features
        according to the model. Useful for understanding which features
        contribute most to predictions.

        Args:
            feature_names: Names of features
            importance_values: Importance scores
            top_n: Number of top features to display
            save_path: Path to save the plot
            figsize: Figure size

        Example:
            >>> importance = model.get_feature_importance()
            >>> evaluator.plot_feature_importance(
            ...     feature_names=X.columns,
            ...     importance_values=importance,
            ...     top_n=15
            ... )
        """
        # Create DataFrame and sort
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance_values
        }).sort_values('importance', ascending=False).head(top_n)

        plt.figure(figsize=figsize)
        plt.barh(range(len(importance_df)), importance_df['importance'])
        plt.yticks(range(len(importance_df)), importance_df['feature'])
        plt.xlabel('Importance')
        plt.title(f'Top {top_n} Feature Importance')
        plt.gca().invert_yaxis()
        plt.grid(axis='x', alpha=0.3)
        plt.tight_layout()

        if save_path:
            plt.savefig(self.output_dir / save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def generate_classification_report(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        target_names: Optional[List[str]] = None
    ) -> str:
        """
        Generate detailed classification report.

        Returns:
            String containing precision, recall, f1-score for each class
        """
        return classification_report(y_true, y_pred, target_names=target_names)
