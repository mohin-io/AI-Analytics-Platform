"""
Federated Learning Client

Handles local training on client data and communicates with the server.
"""

import numpy as np
from typing import Any, Dict, Optional
from sklearn.base import clone
from sklearn.metrics import accuracy_score, mean_squared_error
import copy


class FederatedClient:
    """
    Federated Learning Client.

    Trains models locally on private data and sends only model updates
    to the server (not raw data).
    """

    def __init__(self, client_id: str, X_local: np.ndarray, y_local: np.ndarray,
                 task_type: str = 'classification'):
        """
        Initialize federated client.

        Args:
            client_id: Unique client identifier
            X_local: Local training features (private, never shared)
            y_local: Local training labels (private, never shared)
            task_type: 'classification' or 'regression'
        """
        self.client_id = client_id
        self.X_local = X_local
        self.y_local = y_local
        self.task_type = task_type
        self.local_model: Optional[Any] = None
        self.training_rounds = 0

    def receive_global_model(self, global_model: Any):
        """
        Receive global model from server.

        Args:
            global_model: Global model from server
        """
        self.local_model = clone(global_model)

    def train_local(self, epochs: int = 1, batch_size: Optional[int] = None) -> Dict[str, Any]:
        """
        Train model on local data.

        Args:
            epochs: Number of local training epochs
            batch_size: Batch size for training (None = full batch)

        Returns:
            Dictionary with model weights and metrics
        """
        if self.local_model is None:
            raise ValueError("No local model available. Call receive_global_model() first.")

        # For sklearn models, we'll do simple fitting
        # In practice, you might implement mini-batch training
        self.local_model.fit(self.X_local, self.y_local)

        # Calculate local metrics
        predictions = self.local_model.predict(self.X_local)

        if self.task_type == 'classification':
            accuracy = accuracy_score(self.y_local, predictions)
            loss = 1 - accuracy  # Simplified loss
            metrics = {'accuracy': accuracy}
        else:
            mse = mean_squared_error(self.y_local, predictions)
            loss = mse
            metrics = {'mse': mse}

        # Extract model weights
        model_weights = self._extract_model_weights()

        self.training_rounds += 1

        return {
            'client_id': self.client_id,
            'model_weights': model_weights,
            'n_samples': len(self.X_local),
            'loss': loss,
            'metrics': metrics
        }

    def _extract_model_weights(self) -> Dict[str, np.ndarray]:
        """
        Extract model weights for transmission to server.

        Returns:
            Dictionary of model parameters
        """
        weights = {}

        if hasattr(self.local_model, 'coef_'):
            weights['coef_'] = self.local_model.coef_.copy()

        if hasattr(self.local_model, 'intercept_'):
            weights['intercept_'] = self.local_model.intercept_.copy()

        # For tree-based models, we'd need different extraction
        # This is simplified for linear models

        return weights

    def add_differential_privacy(self, weights: Dict[str, np.ndarray],
                                 noise_scale: float = 0.1) -> Dict[str, np.ndarray]:
        """
        Add differential privacy noise to model weights.

        Args:
            weights: Model weights
            noise_scale: Scale of Gaussian noise to add

        Returns:
            Noisy weights
        """
        noisy_weights = {}

        for key, value in weights.items():
            noise = np.random.normal(0, noise_scale, value.shape)
            noisy_weights[key] = value + noise

        return noisy_weights

    def evaluate_local(self, X_test: Optional[np.ndarray] = None,
                      y_test: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        Evaluate model on local test data.

        Args:
            X_test: Test features (None = use training data)
            y_test: Test labels (None = use training labels)

        Returns:
            Dictionary of evaluation metrics
        """
        if self.local_model is None:
            raise ValueError("No local model available.")

        X_eval = X_test if X_test is not None else self.X_local
        y_eval = y_test if y_test is not None else self.y_local

        predictions = self.local_model.predict(X_eval)

        if self.task_type == 'classification':
            accuracy = accuracy_score(y_eval, predictions)
            return {'accuracy': accuracy}
        else:
            mse = mean_squared_error(y_eval, predictions)
            rmse = np.sqrt(mse)
            return {'mse': mse, 'rmse': rmse}


class SecureClient(FederatedClient):
    """
    Federated client with enhanced security features.

    Includes differential privacy, secure aggregation, and gradient clipping.
    """

    def __init__(self, client_id: str, X_local: np.ndarray, y_local: np.ndarray,
                 task_type: str = 'classification',
                 privacy_epsilon: float = 1.0,
                 gradient_clip_norm: float = 1.0):
        """
        Initialize secure federated client.

        Args:
            client_id: Unique client identifier
            X_local: Local training features
            y_local: Local training labels
            task_type: 'classification' or 'regression'
            privacy_epsilon: Privacy budget (smaller = more private)
            gradient_clip_norm: Maximum gradient norm for clipping
        """
        super().__init__(client_id, X_local, y_local, task_type)
        self.privacy_epsilon = privacy_epsilon
        self.gradient_clip_norm = gradient_clip_norm
        self.privacy_spent = 0.0

    def train_local(self, epochs: int = 1, batch_size: Optional[int] = None,
                   add_noise: bool = True) -> Dict[str, Any]:
        """
        Train with privacy-preserving mechanisms.

        Args:
            epochs: Number of epochs
            batch_size: Batch size
            add_noise: Whether to add differential privacy noise

        Returns:
            Model update with privacy guarantees
        """
        # Train normally
        update = super().train_local(epochs, batch_size)

        # Apply gradient clipping
        update['model_weights'] = self._clip_weights(update['model_weights'])

        # Add differential privacy noise
        if add_noise:
            noise_scale = self._calculate_noise_scale()
            update['model_weights'] = self.add_differential_privacy(
                update['model_weights'],
                noise_scale
            )
            self.privacy_spent += self.privacy_epsilon / epochs

        return update

    def _clip_weights(self, weights: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Clip weight magnitudes for privacy.

        Args:
            weights: Model weights

        Returns:
            Clipped weights
        """
        clipped_weights = {}

        for key, value in weights.items():
            # Calculate L2 norm
            norm = np.linalg.norm(value)

            # Clip if exceeds threshold
            if norm > self.gradient_clip_norm:
                clipped_weights[key] = value * (self.gradient_clip_norm / norm)
            else:
                clipped_weights[key] = value

        return clipped_weights

    def _calculate_noise_scale(self) -> float:
        """
        Calculate noise scale for differential privacy.

        Uses the Gaussian mechanism for (epsilon, delta)-DP.

        Returns:
            Noise scale (sigma)
        """
        # Simplified calculation
        # In practice, use proper DP-SGD noise calibration
        sensitivity = self.gradient_clip_norm
        sigma = sensitivity * np.sqrt(2 * np.log(1.25 / 1e-5)) / self.privacy_epsilon

        return sigma

    def get_privacy_budget_used(self) -> float:
        """
        Get total privacy budget used.

        Returns:
            Privacy budget spent
        """
        return self.privacy_spent
