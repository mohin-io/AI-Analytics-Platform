"""
Model Visualization Tools

Visualize model internals, decision boundaries, and interpretability.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Any, Optional, List
from sklearn.decomposition import PCA


class ModelVisualizer:
    """
    Visualize model structure and decision boundaries.
    """

    def __init__(self, model: Any):
        """
        Initialize model visualizer.

        Args:
            model: Trained model to visualize
        """
        self.model = model

    def plot_decision_boundary_2d(self, X: np.ndarray, y: np.ndarray,
                                  feature_indices: tuple = (0, 1),
                                  resolution: int = 100,
                                  save_path: Optional[str] = None):
        """
        Plot 2D decision boundary.

        Args:
            X: Feature matrix
            y: Labels
            feature_indices: Indices of 2 features to plot
            resolution: Grid resolution
            save_path: Path to save plot
        """
        # Extract 2 features
        X_2d = X[:, list(feature_indices)]

        # Create mesh grid
        x_min, x_max = X_2d[:, 0].min() - 0.5, X_2d[:, 0].max() + 0.5
        y_min, y_max = X_2d[:, 1].min() - 0.5, X_2d[:, 1].max() + 0.5

        xx, yy = np.meshgrid(
            np.linspace(x_min, x_max, resolution),
            np.linspace(y_min, y_max, resolution)
        )

        # Create full feature vector with other features at mean
        X_mean = X.mean(axis=0)
        grid_points = np.tile(X_mean, (resolution * resolution, 1))
        grid_points[:, feature_indices[0]] = xx.ravel()
        grid_points[:, feature_indices[1]] = yy.ravel()

        # Predict on grid
        Z = self.model.predict(grid_points)
        Z = Z.reshape(xx.shape)

        # Plot
        plt.figure(figsize=(10, 8))
        plt.contourf(xx, yy, Z, alpha=0.3, cmap='viridis')
        scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y, cmap='viridis',
                            edgecolors='black', s=50)
        plt.colorbar(scatter)
        plt.xlabel(f'Feature {feature_indices[0]}')
        plt.ylabel(f'Feature {feature_indices[1]}')
        plt.title('Decision Boundary')

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()

    def plot_weight_distribution(self, save_path: Optional[str] = None):
        """
        Plot distribution of model weights.

        Args:
            save_path: Path to save plot
        """
        if not hasattr(self.model, 'coef_'):
            print("Model does not have weights to visualize")
            return

        weights = self.model.coef_.ravel()

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        # Histogram
        axes[0].hist(weights, bins=50, edgecolor='black', alpha=0.7)
        axes[0].set_xlabel('Weight Value')
        axes[0].set_ylabel('Frequency')
        axes[0].set_title('Weight Distribution')
        axes[0].grid(alpha=0.3)

        # Box plot
        axes[1].boxplot(weights, vert=True)
        axes[1].set_ylabel('Weight Value')
        axes[1].set_title('Weight Statistics')
        axes[1].grid(alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
