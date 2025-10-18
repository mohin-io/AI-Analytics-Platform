"""
Stacking Ensemble Methods

Implements stacked generalization where base models' predictions
are used as features for a meta-learner.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Any, Tuple
from sklearn.base import BaseEstimator, clone
from sklearn.model_selection import cross_val_predict, KFold
from sklearn.linear_model import LogisticRegression, Ridge
import warnings


class StackingEnsemble(BaseEstimator):
    """
    Stacking ensemble with cross-validation.

    Trains base models using cross-validation and uses their out-of-fold
    predictions as features for a meta-learner.
    """

    def __init__(self, base_models: List[Any], meta_model: Optional[Any] = None,
                 task_type: str = 'classification', cv: int = 5,
                 use_original_features: bool = False):
        """
        Initialize stacking ensemble.

        Args:
            base_models: List of base model instances
            meta_model: Meta-learner model (None = default LogisticRegression/Ridge)
            task_type: 'classification' or 'regression'
            cv: Number of cross-validation folds for generating meta-features
            use_original_features: Whether to include original features in meta-learner
        """
        self.base_models = base_models
        self.meta_model = meta_model
        self.task_type = task_type
        self.cv = cv
        self.use_original_features = use_original_features

        # Initialize meta-model if not provided
        if self.meta_model is None:
            if task_type == 'classification':
                self.meta_model = LogisticRegression(max_iter=1000)
            else:
                self.meta_model = Ridge()

        self.fitted_base_models: List[Any] = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'StackingEnsemble':
        """
        Fit stacking ensemble.

        Args:
            X: Training features
            y: Training labels

        Returns:
            Self
        """
        n_samples = X.shape[0]
        n_base_models = len(self.base_models)

        # Generate out-of-fold predictions for meta-features
        meta_features = np.zeros((n_samples, n_base_models))

        for i, base_model in enumerate(self.base_models):
            # Get out-of-fold predictions using cross-validation
            if self.task_type == 'classification' and hasattr(base_model, 'predict_proba'):
                # Use predict_proba for classification
                oof_pred = cross_val_predict(
                    base_model, X, y, cv=self.cv, method='predict_proba'
                )
                # Use probability of positive class (or first class if binary)
                if oof_pred.ndim > 1 and oof_pred.shape[1] > 1:
                    meta_features[:, i] = oof_pred[:, 1] if oof_pred.shape[1] == 2 else oof_pred[:, 0]
                else:
                    meta_features[:, i] = oof_pred.ravel()
            else:
                # Use predict for regression or models without predict_proba
                oof_pred = cross_val_predict(base_model, X, y, cv=self.cv)
                meta_features[:, i] = oof_pred

        # Train base models on full data
        self.fitted_base_models = []
        for base_model in self.base_models:
            fitted_model = clone(base_model)
            fitted_model.fit(X, y)
            self.fitted_base_models.append(fitted_model)

        # Prepare meta-features
        if self.use_original_features:
            meta_X = np.hstack([X, meta_features])
        else:
            meta_X = meta_features

        # Train meta-model
        self.meta_model.fit(meta_X, y)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using the stacking ensemble.

        Args:
            X: Features

        Returns:
            Predictions
        """
        # Get predictions from base models
        meta_features = self._get_meta_features(X)

        # Prepare meta-features
        if self.use_original_features:
            meta_X = np.hstack([X, meta_features])
        else:
            meta_X = meta_features

        # Predict with meta-model
        return self.meta_model.predict(meta_X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities.

        Args:
            X: Features

        Returns:
            Class probabilities
        """
        if not hasattr(self.meta_model, 'predict_proba'):
            raise AttributeError("Meta-model does not support predict_proba")

        # Get predictions from base models
        meta_features = self._get_meta_features(X)

        # Prepare meta-features
        if self.use_original_features:
            meta_X = np.hstack([X, meta_features])
        else:
            meta_X = meta_features

        # Predict probabilities with meta-model
        return self.meta_model.predict_proba(meta_X)

    def _get_meta_features(self, X: np.ndarray) -> np.ndarray:
        """
        Generate meta-features from base model predictions.

        Args:
            X: Features

        Returns:
            Meta-features array
        """
        n_samples = X.shape[0]
        n_base_models = len(self.fitted_base_models)
        meta_features = np.zeros((n_samples, n_base_models))

        for i, model in enumerate(self.fitted_base_models):
            if self.task_type == 'classification' and hasattr(model, 'predict_proba'):
                proba = model.predict_proba(X)
                # Use probability of positive class
                if proba.ndim > 1 and proba.shape[1] > 1:
                    meta_features[:, i] = proba[:, 1] if proba.shape[1] == 2 else proba[:, 0]
                else:
                    meta_features[:, i] = proba.ravel()
            else:
                pred = model.predict(X)
                meta_features[:, i] = pred

        return meta_features


class MultiLevelStacking(BaseEstimator):
    """
    Multi-level stacking ensemble.

    Stacks multiple layers of models, where each layer uses predictions
    from the previous layer as features.
    """

    def __init__(self, layers: List[List[Any]], meta_model: Optional[Any] = None,
                 task_type: str = 'classification', cv: int = 5):
        """
        Initialize multi-level stacking.

        Args:
            layers: List of model lists, where each list is a stacking layer
            meta_model: Final meta-learner
            task_type: 'classification' or 'regression'
            cv: Number of CV folds
        """
        self.layers = layers
        self.meta_model = meta_model
        self.task_type = task_type
        self.cv = cv

        # Initialize meta-model if not provided
        if self.meta_model is None:
            if task_type == 'classification':
                self.meta_model = LogisticRegression(max_iter=1000)
            else:
                self.meta_model = Ridge()

        self.fitted_layers: List[List[Any]] = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'MultiLevelStacking':
        """
        Fit multi-level stacking ensemble.

        Args:
            X: Training features
            y: Training labels

        Returns:
            Self
        """
        current_X = X.copy()
        self.fitted_layers = []

        # Train each layer
        for layer_idx, layer_models in enumerate(self.layers):
            fitted_layer = []
            layer_meta_features = np.zeros((len(X), len(layer_models)))

            # Train each model in the layer
            for model_idx, model in enumerate(layer_models):
                # Get out-of-fold predictions
                if self.task_type == 'classification' and hasattr(model, 'predict_proba'):
                    oof_pred = cross_val_predict(
                        model, current_X, y, cv=self.cv, method='predict_proba'
                    )
                    if oof_pred.ndim > 1 and oof_pred.shape[1] > 1:
                        layer_meta_features[:, model_idx] = (
                            oof_pred[:, 1] if oof_pred.shape[1] == 2 else oof_pred[:, 0]
                        )
                    else:
                        layer_meta_features[:, model_idx] = oof_pred.ravel()
                else:
                    oof_pred = cross_val_predict(model, current_X, y, cv=self.cv)
                    layer_meta_features[:, model_idx] = oof_pred

                # Fit model on full data
                fitted_model = clone(model)
                fitted_model.fit(current_X, y)
                fitted_layer.append(fitted_model)

            self.fitted_layers.append(fitted_layer)

            # Use layer predictions as features for next layer
            current_X = np.hstack([current_X, layer_meta_features])

        # Train final meta-model
        self.meta_model.fit(current_X, y)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using multi-level stacking.

        Args:
            X: Features

        Returns:
            Predictions
        """
        current_X = X.copy()

        # Pass through each layer
        for fitted_layer in self.fitted_layers:
            layer_predictions = []

            for model in fitted_layer:
                if self.task_type == 'classification' and hasattr(model, 'predict_proba'):
                    proba = model.predict_proba(current_X)
                    if proba.ndim > 1 and proba.shape[1] > 1:
                        pred = proba[:, 1] if proba.shape[1] == 2 else proba[:, 0]
                    else:
                        pred = proba.ravel()
                else:
                    pred = model.predict(current_X)

                layer_predictions.append(pred)

            # Append layer predictions to features
            current_X = np.hstack([current_X, np.column_stack(layer_predictions)])

        # Final prediction with meta-model
        return self.meta_model.predict(current_X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities.

        Args:
            X: Features

        Returns:
            Class probabilities
        """
        if not hasattr(self.meta_model, 'predict_proba'):
            raise AttributeError("Meta-model does not support predict_proba")

        current_X = X.copy()

        # Pass through each layer
        for fitted_layer in self.fitted_layers:
            layer_predictions = []

            for model in fitted_layer:
                if self.task_type == 'classification' and hasattr(model, 'predict_proba'):
                    proba = model.predict_proba(current_X)
                    if proba.ndim > 1 and proba.shape[1] > 1:
                        pred = proba[:, 1] if proba.shape[1] == 2 else proba[:, 0]
                    else:
                        pred = proba.ravel()
                else:
                    pred = model.predict(current_X)

                layer_predictions.append(pred)

            # Append layer predictions to features
            current_X = np.hstack([current_X, np.column_stack(layer_predictions)])

        # Final prediction with meta-model
        return self.meta_model.predict_proba(current_X)


class FeatureWeightedStacking(BaseEstimator):
    """
    Stacking with feature-weighted meta-learner.

    Learns importance weights for both base model predictions and original features.
    """

    def __init__(self, base_models: List[Any], task_type: str = 'classification',
                 cv: int = 5, alpha: float = 1.0):
        """
        Initialize feature-weighted stacking.

        Args:
            base_models: List of base models
            task_type: 'classification' or 'regression'
            cv: Number of CV folds
            alpha: Regularization strength for meta-learner
        """
        self.base_models = base_models
        self.task_type = task_type
        self.cv = cv
        self.alpha = alpha

        # Meta-model with regularization
        if task_type == 'classification':
            self.meta_model = LogisticRegression(C=1.0/alpha, max_iter=1000)
        else:
            self.meta_model = Ridge(alpha=alpha)

        self.fitted_base_models: List[Any] = []
        self.feature_weights: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'FeatureWeightedStacking':
        """
        Fit feature-weighted stacking.

        Args:
            X: Training features
            y: Training labels

        Returns:
            Self
        """
        n_samples, n_features = X.shape
        n_base_models = len(self.base_models)

        # Generate meta-features
        meta_features = np.zeros((n_samples, n_base_models))

        for i, base_model in enumerate(self.base_models):
            if self.task_type == 'classification' and hasattr(base_model, 'predict_proba'):
                oof_pred = cross_val_predict(
                    base_model, X, y, cv=self.cv, method='predict_proba'
                )
                if oof_pred.ndim > 1 and oof_pred.shape[1] > 1:
                    meta_features[:, i] = oof_pred[:, 1] if oof_pred.shape[1] == 2 else oof_pred[:, 0]
                else:
                    meta_features[:, i] = oof_pred.ravel()
            else:
                oof_pred = cross_val_predict(base_model, X, y, cv=self.cv)
                meta_features[:, i] = oof_pred

        # Fit base models on full data
        self.fitted_base_models = []
        for base_model in self.base_models:
            fitted_model = clone(base_model)
            fitted_model.fit(X, y)
            self.fitted_base_models.append(fitted_model)

        # Combine original features and meta-features
        combined_features = np.hstack([X, meta_features])

        # Fit meta-model
        self.meta_model.fit(combined_features, y)

        # Extract feature weights
        if hasattr(self.meta_model, 'coef_'):
            self.feature_weights = np.abs(self.meta_model.coef_)
            if self.feature_weights.ndim > 1:
                self.feature_weights = np.mean(self.feature_weights, axis=0)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions.

        Args:
            X: Features

        Returns:
            Predictions
        """
        # Get meta-features
        meta_features = np.zeros((X.shape[0], len(self.fitted_base_models)))

        for i, model in enumerate(self.fitted_base_models):
            if self.task_type == 'classification' and hasattr(model, 'predict_proba'):
                proba = model.predict_proba(X)
                if proba.ndim > 1 and proba.shape[1] > 1:
                    meta_features[:, i] = proba[:, 1] if proba.shape[1] == 2 else proba[:, 0]
                else:
                    meta_features[:, i] = proba.ravel()
            else:
                meta_features[:, i] = model.predict(X)

        # Combine features
        combined_features = np.hstack([X, meta_features])

        return self.meta_model.predict(combined_features)

    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """
        Get feature importance from meta-learner.

        Returns:
            Dictionary with feature importance
        """
        if self.feature_weights is None:
            return None

        importance = {}

        # Original feature importance
        n_original = self.feature_weights.shape[0] - len(self.fitted_base_models)
        for i in range(n_original):
            importance[f'feature_{i}'] = self.feature_weights[i]

        # Base model importance
        for i, _ in enumerate(self.fitted_base_models):
            importance[f'base_model_{i}'] = self.feature_weights[n_original + i]

        return importance
