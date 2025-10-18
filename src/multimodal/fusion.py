"""
Multi-Modal Fusion Strategies

Implements various strategies for combining features from different modalities.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import warnings


class FusionStrategy(Enum):
    """Fusion strategies for multi-modal learning."""
    EARLY = "early"  # Concatenate features before model
    LATE = "late"  # Combine predictions from separate models
    HYBRID = "hybrid"  # Combine both early and late fusion
    ATTENTION = "attention"  # Attention-based weighted fusion
    TENSOR = "tensor"  # Tensor fusion


class ModalityFusion:
    """
    Fuse features from multiple modalities.

    Supports various fusion strategies including early fusion (feature concatenation),
    late fusion (decision fusion), and hybrid approaches.
    """

    def __init__(self, strategy: FusionStrategy = FusionStrategy.EARLY,
                 normalize: bool = True,
                 weights: Optional[Dict[str, float]] = None):
        """
        Initialize modality fusion.

        Args:
            strategy: Fusion strategy to use
            normalize: Whether to normalize features before fusion
            weights: Weights for each modality (for weighted fusion)
        """
        self.strategy = strategy
        self.normalize = normalize
        self.weights = weights or {}
        self.scalers: Dict[str, StandardScaler] = {}
        self.modality_names: List[str] = []

    def fit(self, modality_data: Dict[str, np.ndarray]) -> 'ModalityFusion':
        """
        Fit the fusion module.

        Args:
            modality_data: Dictionary mapping modality names to feature arrays

        Returns:
            Self
        """
        self.modality_names = list(modality_data.keys())

        # Fit scalers for normalization
        if self.normalize:
            for modality, features in modality_data.items():
                scaler = StandardScaler()
                scaler.fit(features)
                self.scalers[modality] = scaler

        # Set default weights if not provided
        if not self.weights:
            n_modalities = len(self.modality_names)
            self.weights = {mod: 1.0 / n_modalities for mod in self.modality_names}

        return self

    def transform(self, modality_data: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Transform multi-modal data using the fusion strategy.

        Args:
            modality_data: Dictionary mapping modality names to feature arrays

        Returns:
            Fused feature array
        """
        if self.strategy == FusionStrategy.EARLY:
            return self._early_fusion(modality_data)
        elif self.strategy == FusionStrategy.ATTENTION:
            return self._attention_fusion(modality_data)
        elif self.strategy == FusionStrategy.TENSOR:
            return self._tensor_fusion(modality_data)
        else:
            # Default to early fusion
            return self._early_fusion(modality_data)

    def fit_transform(self, modality_data: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Fit and transform in one step.

        Args:
            modality_data: Dictionary mapping modality names to feature arrays

        Returns:
            Fused feature array
        """
        return self.fit(modality_data).transform(modality_data)

    def _early_fusion(self, modality_data: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Early fusion: Concatenate features from all modalities.

        Args:
            modality_data: Dictionary of modality features

        Returns:
            Concatenated feature array
        """
        fused_features = []

        for modality in self.modality_names:
            if modality not in modality_data:
                warnings.warn(f"Modality '{modality}' not found in data")
                continue

            features = modality_data[modality]

            # Normalize if required
            if self.normalize and modality in self.scalers:
                features = self.scalers[modality].transform(features)

            # Apply modality weight
            weight = self.weights.get(modality, 1.0)
            features = features * weight

            fused_features.append(features)

        if not fused_features:
            raise ValueError("No valid modalities found")

        return np.hstack(fused_features)

    def _attention_fusion(self, modality_data: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Attention-based fusion: Learn attention weights for each modality.

        This is a simplified version using learned static weights.

        Args:
            modality_data: Dictionary of modality features

        Returns:
            Attention-weighted fused features
        """
        # Normalize features
        normalized_features = {}
        for modality in self.modality_names:
            if modality not in modality_data:
                continue

            features = modality_data[modality]

            if self.normalize and modality in self.scalers:
                features = self.scalers[modality].transform(features)

            normalized_features[modality] = features

        # Calculate attention scores based on feature variance
        attention_scores = {}
        total_score = 0

        for modality, features in normalized_features.items():
            # Use variance as proxy for information content
            score = np.var(features, axis=0).mean()
            attention_scores[modality] = score
            total_score += score

        # Normalize attention scores
        for modality in attention_scores:
            attention_scores[modality] /= (total_score + 1e-10)

        # Apply attention weights and concatenate
        fused_features = []
        for modality, features in normalized_features.items():
            weight = attention_scores[modality]
            weighted_features = features * weight
            fused_features.append(weighted_features)

        return np.hstack(fused_features)

    def _tensor_fusion(self, modality_data: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Tensor fusion: Compute outer products between modality features.

        Note: This can create very high-dimensional representations.
        Use with caution on high-dimensional modalities.

        Args:
            modality_data: Dictionary of modality features

        Returns:
            Tensor fused features
        """
        # Normalize features
        normalized_features = []
        for modality in self.modality_names:
            if modality not in modality_data:
                continue

            features = modality_data[modality]

            if self.normalize and modality in self.scalers:
                features = self.scalers[modality].transform(features)

            normalized_features.append(features)

        if len(normalized_features) < 2:
            return normalized_features[0] if normalized_features else np.array([])

        # Compute pairwise outer products
        # For simplicity, we'll do element-wise products for pairs
        n_samples = normalized_features[0].shape[0]

        # Start with first modality
        result = normalized_features[0]

        # Iteratively fuse with remaining modalities
        for i in range(1, len(normalized_features)):
            # Element-wise multiplication (simplified tensor fusion)
            # Repeat features to match dimensions
            feat_a = result
            feat_b = normalized_features[i]

            # Ensure same number of features by padding/truncating
            n_feat_a = feat_a.shape[1]
            n_feat_b = feat_b.shape[1]

            if n_feat_a < n_feat_b:
                padding = np.zeros((n_samples, n_feat_b - n_feat_a))
                feat_a = np.hstack([feat_a, padding])
            elif n_feat_b < n_feat_a:
                padding = np.zeros((n_samples, n_feat_a - n_feat_b))
                feat_b = np.hstack([feat_b, padding])

            # Element-wise product
            tensor_product = feat_a * feat_b

            # Concatenate original and product
            result = np.hstack([result, feat_b, tensor_product])

        return result

    def fuse_predictions(self, modality_predictions: Dict[str, np.ndarray],
                        method: str = 'average') -> np.ndarray:
        """
        Late fusion: Combine predictions from different modality-specific models.

        Args:
            modality_predictions: Dictionary mapping modality to predictions
            method: Fusion method ('average', 'weighted', 'max', 'voting')

        Returns:
            Fused predictions
        """
        if method == 'average':
            # Simple averaging
            all_preds = list(modality_predictions.values())
            return np.mean(all_preds, axis=0)

        elif method == 'weighted':
            # Weighted averaging using modality weights
            weighted_sum = None
            total_weight = 0

            for modality, predictions in modality_predictions.items():
                weight = self.weights.get(modality, 1.0)

                if weighted_sum is None:
                    weighted_sum = predictions * weight
                else:
                    weighted_sum += predictions * weight

                total_weight += weight

            return weighted_sum / total_weight

        elif method == 'max':
            # Take maximum prediction
            all_preds = list(modality_predictions.values())
            return np.max(all_preds, axis=0)

        elif method == 'voting':
            # Majority voting (for classification)
            all_preds = np.array(list(modality_predictions.values()))
            # Get most common prediction
            from scipy import stats
            return stats.mode(all_preds, axis=0)[0].squeeze()

        else:
            raise ValueError(f"Unknown fusion method: {method}")


class MultiModalEnsemble(BaseEstimator):
    """
    Ensemble model for multi-modal learning.

    Trains separate models for each modality and fuses their predictions.
    """

    def __init__(self, models: Dict[str, Any],
                 fusion_strategy: str = 'weighted',
                 modality_weights: Optional[Dict[str, float]] = None):
        """
        Initialize multi-modal ensemble.

        Args:
            models: Dictionary mapping modality names to model instances
            fusion_strategy: How to fuse predictions ('average', 'weighted', 'voting')
            modality_weights: Weights for each modality in fusion
        """
        self.models = models
        self.fusion_strategy = fusion_strategy
        self.modality_weights = modality_weights or {}
        self.fusion_module = ModalityFusion(weights=modality_weights)

    def fit(self, modality_data: Dict[str, np.ndarray], y: np.ndarray) -> 'MultiModalEnsemble':
        """
        Fit models for each modality.

        Args:
            modality_data: Dictionary mapping modality names to feature arrays
            y: Labels

        Returns:
            Self
        """
        for modality, model in self.models.items():
            if modality not in modality_data:
                warnings.warn(f"Modality '{modality}' not found in training data")
                continue

            X_modality = modality_data[modality]
            model.fit(X_modality, y)

        return self

    def predict(self, modality_data: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Predict using all modality models and fuse results.

        Args:
            modality_data: Dictionary mapping modality names to feature arrays

        Returns:
            Fused predictions
        """
        predictions = {}

        for modality, model in self.models.items():
            if modality not in modality_data:
                warnings.warn(f"Modality '{modality}' not found in prediction data")
                continue

            X_modality = modality_data[modality]
            pred = model.predict(X_modality)
            predictions[modality] = pred

        # Fuse predictions
        return self.fusion_module.fuse_predictions(predictions, method=self.fusion_strategy)

    def predict_proba(self, modality_data: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Predict class probabilities using all modality models and fuse results.

        Args:
            modality_data: Dictionary mapping modality names to feature arrays

        Returns:
            Fused probability predictions
        """
        probabilities = {}

        for modality, model in self.models.items():
            if modality not in modality_data:
                continue

            if not hasattr(model, 'predict_proba'):
                warnings.warn(f"Model for modality '{modality}' does not support predict_proba")
                continue

            X_modality = modality_data[modality]
            proba = model.predict_proba(X_modality)
            probabilities[modality] = proba

        if not probabilities:
            raise ValueError("No models with predict_proba available")

        # Fuse probabilities
        return self.fusion_module.fuse_predictions(probabilities, method=self.fusion_strategy)


class CrossModalAttention:
    """
    Cross-modal attention mechanism.

    Allows one modality to attend to features from another modality.
    """

    def __init__(self, embed_dim: int = 128):
        """
        Initialize cross-modal attention.

        Args:
            embed_dim: Embedding dimension for attention computation
        """
        self.embed_dim = embed_dim
        self.query_weights: Optional[np.ndarray] = None
        self.key_weights: Optional[np.ndarray] = None
        self.value_weights: Optional[np.ndarray] = None

    def fit(self, source_features: np.ndarray, target_features: np.ndarray):
        """
        Fit attention weights.

        Args:
            source_features: Source modality features (query)
            target_features: Target modality features (key/value)
        """
        # Initialize random projection matrices (simplified)
        n_source_feat = source_features.shape[1]
        n_target_feat = target_features.shape[1]

        np.random.seed(42)
        self.query_weights = np.random.randn(n_source_feat, self.embed_dim) * 0.01
        self.key_weights = np.random.randn(n_target_feat, self.embed_dim) * 0.01
        self.value_weights = np.random.randn(n_target_feat, self.embed_dim) * 0.01

    def transform(self, source_features: np.ndarray,
                 target_features: np.ndarray) -> np.ndarray:
        """
        Apply cross-modal attention.

        Args:
            source_features: Source modality features
            target_features: Target modality features

        Returns:
            Attention-weighted features
        """
        if self.query_weights is None:
            raise ValueError("Must fit attention module first")

        # Compute query, key, value
        Q = source_features @ self.query_weights  # (n_samples, embed_dim)
        K = target_features @ self.key_weights    # (n_samples, embed_dim)
        V = target_features @ self.value_weights  # (n_samples, embed_dim)

        # Compute attention scores
        scores = Q @ K.T / np.sqrt(self.embed_dim)  # (n_samples, n_samples)

        # Apply softmax
        attention_weights = self._softmax(scores, axis=1)

        # Apply attention to values
        attended_features = attention_weights @ V  # (n_samples, embed_dim)

        # Concatenate with source features
        return np.hstack([source_features, attended_features])

    def _softmax(self, x: np.ndarray, axis: int = -1) -> np.ndarray:
        """Compute softmax."""
        exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
        return exp_x / np.sum(exp_x, axis=axis, keepdims=True)
