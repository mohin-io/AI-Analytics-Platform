"""
Advanced Voting Ensemble Methods

Implements weighted and adaptive voting strategies.
"""

import numpy as np
from typing import List, Optional, Dict, Any
from sklearn.base import BaseEstimator, clone
from sklearn.model_selection import cross_val_score
from scipy import stats


class WeightedVotingEnsemble(BaseEstimator):
    """
    Weighted voting ensemble.

    Combines predictions from multiple models using learned or specified weights.
    """

    def __init__(self, models: List[Any], weights: Optional[List[float]] = None,
                 task_type: str = 'classification', voting: str = 'soft'):
        """
        Initialize weighted voting ensemble.

        Args:
            models: List of model instances
            weights: Model weights (None = equal weights, 'auto' = learn from CV)
            task_type: 'classification' or 'regression'
            voting: 'soft' (average probabilities) or 'hard' (majority vote)
        """
        self.models = models
        self.weights = weights
        self.task_type = task_type
        self.voting = voting
        self.fitted_models: List[Any] = []
        self.learned_weights: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'WeightedVotingEnsemble':
        """
        Fit weighted voting ensemble.

        Args:
            X: Training features
            y: Training labels

        Returns:
            Self
        """
        # Learn weights if not provided
        if self.weights is None or self.weights == 'auto':
            self.learned_weights = self._learn_weights(X, y)
        else:
            self.learned_weights = np.array(self.weights)

        # Normalize weights
        self.learned_weights = self.learned_weights / np.sum(self.learned_weights)

        # Fit all models
        self.fitted_models = []
        for model in self.models:
            fitted_model = clone(model)
            fitted_model.fit(X, y)
            self.fitted_models.append(fitted_model)

        return self

    def _learn_weights(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Learn weights based on cross-validation performance.

        Args:
            X: Features
            y: Labels

        Returns:
            Array of weights
        """
        weights = []

        for model in self.models:
            # Evaluate model using cross-validation
            scores = cross_val_score(model, X, y, cv=5)
            avg_score = np.mean(scores)
            weights.append(avg_score)

        # Convert to numpy array and ensure positive
        weights = np.maximum(np.array(weights), 0)

        # Avoid division by zero
        if np.sum(weights) == 0:
            weights = np.ones(len(weights))

        return weights

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using weighted voting.

        Args:
            X: Features

        Returns:
            Predictions
        """
        if self.task_type == 'classification':
            if self.voting == 'soft':
                return self._soft_voting_predict(X)
            else:
                return self._hard_voting_predict(X)
        else:
            return self._regression_predict(X)

    def _soft_voting_predict(self, X: np.ndarray) -> np.ndarray:
        """Soft voting for classification."""
        # Collect probability predictions
        all_probas = []

        for model in self.fitted_models:
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(X)
                all_probas.append(proba)
            else:
                # Convert predictions to one-hot probabilities
                pred = model.predict(X)
                classes = np.unique(pred)
                proba = np.zeros((len(pred), len(classes)))
                for i, p in enumerate(pred):
                    proba[i, np.where(classes == p)[0][0]] = 1.0
                all_probas.append(proba)

        # Weighted average of probabilities
        weighted_proba = np.zeros_like(all_probas[0])
        for i, proba in enumerate(all_probas):
            weighted_proba += proba * self.learned_weights[i]

        # Return class with highest probability
        return np.argmax(weighted_proba, axis=1)

    def _hard_voting_predict(self, X: np.ndarray) -> np.ndarray:
        """Hard voting for classification."""
        # Collect predictions
        all_preds = []

        for model in self.fitted_models:
            pred = model.predict(X)
            all_preds.append(pred)

        all_preds = np.array(all_preds)

        # Weighted voting
        final_preds = []
        for i in range(X.shape[0]):
            # Get predictions from all models for this sample
            sample_preds = all_preds[:, i]

            # Count weighted votes
            unique_preds = np.unique(sample_preds)
            vote_counts = {}

            for pred_class in unique_preds:
                # Sum weights of models that predicted this class
                vote_counts[pred_class] = np.sum(
                    self.learned_weights[sample_preds == pred_class]
                )

            # Select class with most votes
            final_pred = max(vote_counts, key=vote_counts.get)
            final_preds.append(final_pred)

        return np.array(final_preds)

    def _regression_predict(self, X: np.ndarray) -> np.ndarray:
        """Weighted average for regression."""
        predictions = []

        for model in self.fitted_models:
            pred = model.predict(X)
            predictions.append(pred)

        predictions = np.array(predictions)

        # Weighted average
        weighted_pred = np.zeros(X.shape[0])
        for i, pred in enumerate(predictions):
            weighted_pred += pred * self.learned_weights[i]

        return weighted_pred

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities.

        Args:
            X: Features

        Returns:
            Weighted average of class probabilities
        """
        all_probas = []

        for model in self.fitted_models:
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(X)
                all_probas.append(proba)

        if not all_probas:
            raise AttributeError("No models support predict_proba")

        # Weighted average of probabilities
        weighted_proba = np.zeros_like(all_probas[0])
        for i, proba in enumerate(all_probas):
            weighted_proba += proba * self.learned_weights[i]

        return weighted_proba

    def get_model_weights(self) -> Dict[int, float]:
        """
        Get learned model weights.

        Returns:
            Dictionary mapping model index to weight
        """
        if self.learned_weights is None:
            return {}

        return {i: weight for i, weight in enumerate(self.learned_weights)}


class AdaptiveVoting(BaseEstimator):
    """
    Adaptive voting ensemble that adjusts weights based on sample difficulty.

    Uses different model weights for different regions of the feature space.
    """

    def __init__(self, models: List[Any], task_type: str = 'classification',
                 n_bins: int = 5):
        """
        Initialize adaptive voting ensemble.

        Args:
            models: List of model instances
            task_type: 'classification' or 'regression'
            n_bins: Number of bins for adaptive weighting
        """
        self.models = models
        self.task_type = task_type
        self.n_bins = n_bins
        self.fitted_models: List[Any] = []
        self.adaptive_weights: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'AdaptiveVoting':
        """
        Fit adaptive voting ensemble.

        Args:
            X: Training features
            y: Training labels

        Returns:
            Self
        """
        # Fit all models
        self.fitted_models = []
        for model in self.models:
            fitted_model = clone(model)
            fitted_model.fit(X, y)
            self.fitted_models.append(fitted_model)

        # Learn adaptive weights based on prediction confidence
        self._learn_adaptive_weights(X, y)

        return self

    def _learn_adaptive_weights(self, X: np.ndarray, y: np.ndarray):
        """
        Learn adaptive weights for different confidence levels.

        Args:
            X: Features
            y: Labels
        """
        n_models = len(self.fitted_models)
        self.adaptive_weights = np.ones((self.n_bins, n_models))

        # Get predictions and confidence scores
        confidences = []
        predictions = []

        for model in self.fitted_models:
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(X)
                # Max probability as confidence
                conf = np.max(proba, axis=1)
                pred = model.predict(X)
            else:
                # Use predictions directly
                pred = model.predict(X)
                conf = np.ones(len(pred))  # Uniform confidence

            confidences.append(conf)
            predictions.append(pred)

        confidences = np.array(confidences).T  # (n_samples, n_models)
        predictions = np.array(predictions).T  # (n_samples, n_models)

        # Bin samples by average confidence
        avg_confidence = np.mean(confidences, axis=1)
        bins = np.percentile(avg_confidence, np.linspace(0, 100, self.n_bins + 1))

        # Calculate weights for each bin
        for i in range(self.n_bins):
            lower = bins[i]
            upper = bins[i + 1] if i < self.n_bins - 1 else np.inf

            # Get samples in this bin
            mask = (avg_confidence >= lower) & (avg_confidence < upper)

            if np.sum(mask) == 0:
                continue

            # Calculate accuracy for each model in this bin
            for j in range(n_models):
                correct = predictions[mask, j] == y[mask]
                accuracy = np.mean(correct) if len(correct) > 0 else 0.5
                self.adaptive_weights[i, j] = accuracy

        # Normalize weights
        for i in range(self.n_bins):
            weight_sum = np.sum(self.adaptive_weights[i])
            if weight_sum > 0:
                self.adaptive_weights[i] /= weight_sum

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using adaptive voting.

        Args:
            X: Features

        Returns:
            Predictions
        """
        # Get predictions and confidences
        confidences = []
        predictions = []

        for model in self.fitted_models:
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(X)
                conf = np.max(proba, axis=1)
                pred = np.argmax(proba, axis=1)
            else:
                pred = model.predict(X)
                conf = np.ones(len(pred))

            confidences.append(conf)
            predictions.append(pred)

        confidences = np.array(confidences).T
        predictions = np.array(predictions).T

        # Determine bin for each sample
        avg_confidence = np.mean(confidences, axis=1)

        # Assign adaptive weights based on confidence
        final_predictions = []

        for i in range(len(X)):
            # Determine bin
            confidence = avg_confidence[i]
            bin_idx = min(int(confidence * self.n_bins), self.n_bins - 1)

            # Get weights for this bin
            weights = self.adaptive_weights[bin_idx]

            # Weighted voting
            if self.task_type == 'classification':
                # Count weighted votes
                sample_preds = predictions[i]
                unique_preds = np.unique(sample_preds)
                vote_counts = {}

                for pred_class in unique_preds:
                    vote_counts[pred_class] = np.sum(weights[sample_preds == pred_class])

                final_pred = max(vote_counts, key=vote_counts.get)
            else:
                # Weighted average for regression
                final_pred = np.sum(predictions[i] * weights)

            final_predictions.append(final_pred)

        return np.array(final_predictions)
