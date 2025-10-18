"""
Model Pruning

Remove unnecessary parameters to reduce model size and improve speed.
"""

import numpy as np
from typing import Any, Dict, Optional
from enum import Enum
from sklearn.base import clone


class PruningStrategy(Enum):
    """Pruning strategies."""
    MAGNITUDE = "magnitude"  # Prune by weight magnitude
    RANDOM = "random"  # Random pruning
    STRUCTURED = "structured"  # Structured pruning (entire neurons/filters)


class ModelPruner:
    """
    Prune model weights to reduce size and improve inference speed.

    Supports magnitude-based and structured pruning for sklearn models.
    """

    def __init__(self, strategy: PruningStrategy = PruningStrategy.MAGNITUDE,
                 sparsity: float = 0.5):
        """
        Initialize model pruner.

        Args:
            strategy: Pruning strategy
            sparsity: Target sparsity level (0.5 = 50% of weights pruned)
        """
        self.strategy = strategy
        self.sparsity = sparsity
        self.masks: Dict[str, np.ndarray] = {}

    def prune_model(self, model: Any) -> Any:
        """
        Prune a trained model.

        Args:
            model: Trained model

        Returns:
            Pruned model
        """
        pruned_model = clone(model)

        # Prune weights
        if hasattr(model, 'coef_'):
            pruned_coef, mask = self._prune_weights(model.coef_)
            pruned_model.coef_ = pruned_coef
            self.masks['coef_'] = mask

        if hasattr(model, 'intercept_'):
            # Usually don't prune bias terms, but can if needed
            pruned_model.intercept_ = model.intercept_

        return pruned_model

    def _prune_weights(self, weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Prune weight array.

        Args:
            weights: Weights to prune

        Returns:
            Tuple of (pruned_weights, mask)
        """
        if self.strategy == PruningStrategy.MAGNITUDE:
            return self._magnitude_pruning(weights)
        elif self.strategy == PruningStrategy.RANDOM:
            return self._random_pruning(weights)
        elif self.strategy == PruningStrategy.STRUCTURED:
            return self._structured_pruning(weights)
        else:
            return self._magnitude_pruning(weights)

    def _magnitude_pruning(self, weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Prune weights with smallest absolute values.

        Args:
            weights: Weight array

        Returns:
            Pruned weights and mask
        """
        # Calculate threshold
        threshold = np.percentile(np.abs(weights), self.sparsity * 100)

        # Create mask
        mask = np.abs(weights) >= threshold

        # Apply mask
        pruned_weights = weights * mask

        return pruned_weights, mask

    def _random_pruning(self, weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Randomly prune weights.

        Args:
            weights: Weight array

        Returns:
            Pruned weights and mask
        """
        # Random mask
        mask = np.random.random(weights.shape) > self.sparsity

        # Apply mask
        pruned_weights = weights * mask

        return pruned_weights, mask

    def _structured_pruning(self, weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Structured pruning - remove entire rows/columns.

        Args:
            weights: Weight array (2D)

        Returns:
            Pruned weights and mask
        """
        if weights.ndim == 1:
            # Can't do structured pruning on 1D
            return self._magnitude_pruning(weights)

        # Calculate L2 norm of each row
        row_norms = np.linalg.norm(weights, axis=1)

        # Determine how many rows to prune
        n_rows_to_prune = int(weights.shape[0] * self.sparsity)

        # Get indices of rows with smallest norms
        rows_to_prune = np.argsort(row_norms)[:n_rows_to_prune]

        # Create mask
        mask = np.ones_like(weights, dtype=bool)
        mask[rows_to_prune, :] = False

        # Apply mask
        pruned_weights = weights * mask

        return pruned_weights, mask

    def get_sparsity_stats(self, model: Any) -> Dict[str, float]:
        """
        Calculate sparsity statistics for a model.

        Args:
            model: Model to analyze

        Returns:
            Dictionary with sparsity statistics
        """
        stats = {}

        if hasattr(model, 'coef_'):
            total_params = model.coef_.size
            zero_params = np.sum(model.coef_ == 0)
            stats['coef_sparsity'] = zero_params / total_params
            stats['coef_total_params'] = total_params
            stats['coef_zero_params'] = zero_params

        return stats

    def iterative_pruning(self, model: Any, X: np.ndarray, y: np.ndarray,
                         n_iterations: int = 5,
                         eval_fn: Optional[callable] = None) -> Any:
        """
        Iteratively prune model with fine-tuning.

        Args:
            model: Model to prune
            X: Training data
            y: Training labels
            n_iterations: Number of prune-train cycles
            eval_fn: Optional evaluation function

        Returns:
            Pruned and fine-tuned model
        """
        current_model = model
        sparsity_per_iter = 1 - (1 - self.sparsity) ** (1 / n_iterations)

        for i in range(n_iterations):
            print(f"Pruning iteration {i+1}/{n_iterations}")

            # Update sparsity for this iteration
            original_sparsity = self.sparsity
            self.sparsity = sparsity_per_iter

            # Prune
            current_model = self.prune_model(current_model)

            # Fine-tune on same data
            current_model.fit(X, y)

            # Evaluate
            if eval_fn:
                score = eval_fn(current_model)
                print(f"Score after iteration {i+1}: {score:.4f}")

            # Restore original sparsity
            self.sparsity = original_sparsity

        return current_model


class GradualPruner:
    """
    Gradual pruning that slowly increases sparsity during training.

    Useful for maintaining model accuracy while pruning.
    """

    def __init__(self, initial_sparsity: float = 0.0,
                 final_sparsity: float = 0.9,
                 pruning_frequency: int = 100):
        """
        Initialize gradual pruner.

        Args:
            initial_sparsity: Starting sparsity
            final_sparsity: Target sparsity
            pruning_frequency: Steps between pruning updates
        """
        self.initial_sparsity = initial_sparsity
        self.final_sparsity = final_sparsity
        self.pruning_frequency = pruning_frequency
        self.current_sparsity = initial_sparsity
        self.step = 0

    def update_sparsity(self, total_steps: int):
        """
        Update sparsity based on current step.

        Uses polynomial decay schedule.

        Args:
            total_steps: Total training steps
        """
        self.step += 1

        if self.step % self.pruning_frequency == 0:
            # Polynomial decay: s_t = s_f + (s_i - s_f)(1 - t/T)^3
            progress = min(self.step / total_steps, 1.0)
            self.current_sparsity = self.final_sparsity + \
                (self.initial_sparsity - self.final_sparsity) * (1 - progress) ** 3

    def get_current_sparsity(self) -> float:
        """Get current sparsity level."""
        return self.current_sparsity
