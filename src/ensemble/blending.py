"""
Blending Ensemble Methods

Similar to stacking but uses a hold-out validation set instead of cross-validation.
"""

import numpy as np
from typing import List, Optional, Any
from sklearn.base import BaseEstimator, clone
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, Ridge


class BlendingEnsemble(BaseEstimator):
    """
    Blending ensemble.

    Trains base models on training set and uses their predictions on a
    hold-out validation set to train the meta-learner.
    """

    def __init__(self, base_models: List[Any], meta_model: Optional[Any] = None,
                 task_type: str = 'classification', blend_ratio: float = 0.2,
                 use_original_features: bool = False):
        """
        Initialize blending ensemble.

        Args:
            base_models: List of base model instances
            meta_model: Meta-learner (None = default)
            task_type: 'classification' or 'regression'
            blend_ratio: Ratio of data to use for blending (validation set)
            use_original_features: Include original features in meta-learner
        """
        self.base_models = base_models
        self.meta_model = meta_model
        self.task_type = task_type
        self.blend_ratio = blend_ratio
        self.use_original_features = use_original_features

        if self.meta_model is None:
            if task_type == 'classification':
                self.meta_model = LogisticRegression(max_iter=1000)
            else:
                self.meta_model = Ridge()

        self.fitted_base_models: List[Any] = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'BlendingEnsemble':
        """
        Fit blending ensemble.

        Args:
            X: Training features
            y: Training labels

        Returns:
            Self
        """
        # Split data into train and blend sets
        X_train, X_blend, y_train, y_blend = train_test_split(
            X, y, test_size=self.blend_ratio, random_state=42, stratify=y if self.task_type == 'classification' else None
        )

        # Train base models on training set
        self.fitted_base_models = []
        blend_features = np.zeros((len(X_blend), len(self.base_models)))

        for i, base_model in enumerate(self.base_models):
            # Clone and fit model
            fitted_model = clone(base_model)
            fitted_model.fit(X_train, y_train)
            self.fitted_base_models.append(fitted_model)

            # Get predictions on blend set
            if self.task_type == 'classification' and hasattr(fitted_model, 'predict_proba'):
                proba = fitted_model.predict_proba(X_blend)
                if proba.ndim > 1 and proba.shape[1] > 1:
                    blend_features[:, i] = proba[:, 1] if proba.shape[1] == 2 else proba[:, 0]
                else:
                    blend_features[:, i] = proba.ravel()
            else:
                blend_features[:, i] = fitted_model.predict(X_blend)

        # Prepare meta-features for blending
        if self.use_original_features:
            meta_X = np.hstack([X_blend, blend_features])
        else:
            meta_X = blend_features

        # Train meta-model on blend set
        self.meta_model.fit(meta_X, y_blend)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions.

        Args:
            X: Features

        Returns:
            Predictions
        """
        # Get predictions from base models
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

        # Prepare meta-features
        if self.use_original_features:
            meta_X = np.hstack([X, meta_features])
        else:
            meta_X = meta_features

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

        # Prepare meta-features
        if self.use_original_features:
            meta_X = np.hstack([X, meta_features])
        else:
            meta_X = meta_features

        return self.meta_model.predict_proba(meta_X)
