"""
Regression model implementations for the Unified AI Analytics Platform

This module provides wrapper classes for popular regression algorithms,
ensuring consistent interface while exposing algorithm-specific functionality.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Union
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    mean_absolute_percentage_error
)
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor
import time

from src.models.base import SupervisedModel


class LinearRegressionModel(SupervisedModel):
    """
    Linear Regression model.

    Fits a linear model with coefficients to minimize residual sum of squares.
    Simple, interpretable, and fast.

    Best for:
    - Linear relationships
    - Baseline comparisons
    - When interpretability matters
    - Small to medium datasets

    Example:
        >>> model = LinearRegressionModel()
        >>> model.train(X_train, y_train)
        >>> predictions = model.predict(X_test)
    """

    def __init__(self, **kwargs: Any):
        super().__init__(model_name="Linear Regression", task_type="regression")
        self.model = LinearRegression(**kwargs)

    def train(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray]
    ) -> 'LinearRegressionModel':
        """Train the linear regression model."""
        start_time = time.time()
        self.model.fit(X, y)
        self.training_time = time.time() - start_time
        self.is_fitted = True
        self.metadata["trained_at"] = pd.Timestamp.now().isoformat()
        return self

    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Make predictions."""
        if not self.is_fitted:
            raise RuntimeError("Model must be trained before making predictions")
        return self.model.predict(X)

    def evaluate(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray]
    ) -> Dict[str, float]:
        """Evaluate model with regression metrics."""
        predictions = self.predict(X)

        metrics = {
            "mae": mean_absolute_error(y, predictions),
            "mse": mean_squared_error(y, predictions),
            "rmse": np.sqrt(mean_squared_error(y, predictions)),
            "r2": r2_score(y, predictions),
        }

        # Add MAPE if no zeros in y
        if not np.any(y == 0):
            metrics["mape"] = mean_absolute_percentage_error(y, predictions)

        return metrics


class RidgeRegressionModel(SupervisedModel):
    """
    Ridge Regression (L2 regularization).

    Linear regression with L2 penalty to prevent overfitting.
    Shrinks coefficients but doesn't set them to zero.

    Best for:
    - Multicollinearity in features
    - Preventing overfitting
    - When all features are relevant

    Example:
        >>> model = RidgeRegressionModel(alpha=1.0)
        >>> model.train(X_train, y_train)
    """

    def __init__(self, alpha: float = 1.0, **kwargs: Any):
        super().__init__(model_name="Ridge Regression", task_type="regression")
        self.model = Ridge(alpha=alpha, **kwargs)

    def train(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray]
    ) -> 'RidgeRegressionModel':
        """Train the Ridge regression model."""
        start_time = time.time()
        self.model.fit(X, y)
        self.training_time = time.time() - start_time
        self.is_fitted = True
        self.metadata["trained_at"] = pd.Timestamp.now().isoformat()
        return self

    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Make predictions."""
        if not self.is_fitted:
            raise RuntimeError("Model must be trained before making predictions")
        return self.model.predict(X)

    def evaluate(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray]
    ) -> Dict[str, float]:
        """Evaluate model with regression metrics."""
        predictions = self.predict(X)

        metrics = {
            "mae": mean_absolute_error(y, predictions),
            "mse": mean_squared_error(y, predictions),
            "rmse": np.sqrt(mean_squared_error(y, predictions)),
            "r2": r2_score(y, predictions),
        }

        if not np.any(y == 0):
            metrics["mape"] = mean_absolute_percentage_error(y, predictions)

        return metrics


class LassoRegressionModel(SupervisedModel):
    """
    Lasso Regression (L1 regularization).

    Linear regression with L1 penalty. Can set coefficients to exactly zero,
    effectively performing feature selection.

    Best for:
    - Feature selection
    - Sparse models
    - When some features are irrelevant

    Example:
        >>> model = LassoRegressionModel(alpha=0.1)
        >>> model.train(X_train, y_train)
        >>> # Check which features were selected
        >>> selected = model.model.coef_ != 0
    """

    def __init__(self, alpha: float = 1.0, max_iter: int = 1000, **kwargs: Any):
        super().__init__(model_name="Lasso Regression", task_type="regression")
        self.model = Lasso(alpha=alpha, max_iter=max_iter, **kwargs)

    def train(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray]
    ) -> 'LassoRegressionModel':
        """Train the Lasso regression model."""
        start_time = time.time()
        self.model.fit(X, y)
        self.training_time = time.time() - start_time
        self.is_fitted = True
        self.metadata["trained_at"] = pd.Timestamp.now().isoformat()
        return self

    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Make predictions."""
        if not self.is_fitted:
            raise RuntimeError("Model must be trained before making predictions")
        return self.model.predict(X)

    def evaluate(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray]
    ) -> Dict[str, float]:
        """Evaluate model with regression metrics."""
        predictions = self.predict(X)

        metrics = {
            "mae": mean_absolute_error(y, predictions),
            "mse": mean_squared_error(y, predictions),
            "rmse": np.sqrt(mean_squared_error(y, predictions)),
            "r2": r2_score(y, predictions),
        }

        if not np.any(y == 0):
            metrics["mape"] = mean_absolute_percentage_error(y, predictions)

        return metrics


class ElasticNetModel(SupervisedModel):
    """
    ElasticNet Regression (L1 + L2 regularization).

    Combines L1 and L2 penalties, getting benefits of both Ridge and Lasso.

    Best for:
    - When you want both regularization and feature selection
    - Many correlated features
    - Balance between Ridge and Lasso

    Example:
        >>> model = ElasticNetModel(alpha=1.0, l1_ratio=0.5)
        >>> model.train(X_train, y_train)
    """

    def __init__(
        self,
        alpha: float = 1.0,
        l1_ratio: float = 0.5,
        max_iter: int = 1000,
        **kwargs: Any
    ):
        super().__init__(model_name="ElasticNet", task_type="regression")
        self.model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=max_iter, **kwargs)

    def train(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray]
    ) -> 'ElasticNetModel':
        """Train the ElasticNet model."""
        start_time = time.time()
        self.model.fit(X, y)
        self.training_time = time.time() - start_time
        self.is_fitted = True
        self.metadata["trained_at"] = pd.Timestamp.now().isoformat()
        return self

    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Make predictions."""
        if not self.is_fitted:
            raise RuntimeError("Model must be trained before making predictions")
        return self.model.predict(X)

    def evaluate(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray]
    ) -> Dict[str, float]:
        """Evaluate model with regression metrics."""
        predictions = self.predict(X)

        metrics = {
            "mae": mean_absolute_error(y, predictions),
            "mse": mean_squared_error(y, predictions),
            "rmse": np.sqrt(mean_squared_error(y, predictions)),
            "r2": r2_score(y, predictions),
        }

        if not np.any(y == 0):
            metrics["mape"] = mean_absolute_percentage_error(y, predictions)

        return metrics


class RandomForestRegressorModel(SupervisedModel):
    """
    Random Forest regressor.

    Ensemble of decision trees for regression. Reduces overfitting and
    provides feature importance.

    Example:
        >>> model = RandomForestRegressorModel(n_estimators=100)
        >>> model.train(X_train, y_train)
        >>> metrics = model.evaluate(X_test, y_test)
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: Optional[int] = None,
        random_state: int = 42,
        n_jobs: int = -1,
        **kwargs: Any
    ):
        super().__init__(model_name="Random Forest Regressor", task_type="regression")
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=n_jobs,
            **kwargs
        )

    def train(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray]
    ) -> 'RandomForestRegressorModel':
        """Train the Random Forest regressor."""
        start_time = time.time()
        self.model.fit(X, y)
        self.training_time = time.time() - start_time
        self.is_fitted = True
        self.metadata["trained_at"] = pd.Timestamp.now().isoformat()
        return self

    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Make predictions."""
        if not self.is_fitted:
            raise RuntimeError("Model must be trained before making predictions")
        return self.model.predict(X)

    def evaluate(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray]
    ) -> Dict[str, float]:
        """Evaluate model with regression metrics."""
        predictions = self.predict(X)

        metrics = {
            "mae": mean_absolute_error(y, predictions),
            "mse": mean_squared_error(y, predictions),
            "rmse": np.sqrt(mean_squared_error(y, predictions)),
            "r2": r2_score(y, predictions),
        }

        if not np.any(y == 0):
            metrics["mape"] = mean_absolute_percentage_error(y, predictions)

        return metrics

    def get_feature_importance(self) -> np.ndarray:
        """Get feature importance scores."""
        if not self.is_fitted:
            raise RuntimeError("Model must be trained first")
        return self.model.feature_importances_


class XGBoostRegressorModel(SupervisedModel):
    """
    XGBoost regressor.

    Gradient boosting implementation optimized for speed and performance.

    Example:
        >>> model = XGBoostRegressorModel(n_estimators=100, learning_rate=0.1)
        >>> model.train(X_train, y_train)
    """

    def __init__(
        self,
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        max_depth: int = 6,
        random_state: int = 42,
        n_jobs: int = -1,
        **kwargs: Any
    ):
        super().__init__(model_name="XGBoost Regressor", task_type="regression")
        self.model = xgb.XGBRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=n_jobs,
            **kwargs
        )

    def train(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray]
    ) -> 'XGBoostRegressorModel':
        """Train the XGBoost regressor."""
        start_time = time.time()
        self.model.fit(X, y)
        self.training_time = time.time() - start_time
        self.is_fitted = True
        self.metadata["trained_at"] = pd.Timestamp.now().isoformat()
        return self

    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Make predictions."""
        if not self.is_fitted:
            raise RuntimeError("Model must be trained before making predictions")
        return self.model.predict(X)

    def evaluate(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray]
    ) -> Dict[str, float]:
        """Evaluate model with regression metrics."""
        predictions = self.predict(X)

        metrics = {
            "mae": mean_absolute_error(y, predictions),
            "mse": mean_squared_error(y, predictions),
            "rmse": np.sqrt(mean_squared_error(y, predictions)),
            "r2": r2_score(y, predictions),
        }

        if not np.any(y == 0):
            metrics["mape"] = mean_absolute_percentage_error(y, predictions)

        return metrics

    def get_feature_importance(self) -> np.ndarray:
        """Get feature importance scores."""
        if not self.is_fitted:
            raise RuntimeError("Model must be trained first")
        return self.model.feature_importances_


class LightGBMRegressorModel(SupervisedModel):
    """
    LightGBM regressor.

    Fast gradient boosting framework for regression tasks.

    Example:
        >>> model = LightGBMRegressorModel(n_estimators=100)
        >>> model.train(X_train, y_train)
    """

    def __init__(
        self,
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        num_leaves: int = 31,
        random_state: int = 42,
        n_jobs: int = -1,
        **kwargs: Any
    ):
        super().__init__(model_name="LightGBM Regressor", task_type="regression")
        self.model = lgb.LGBMRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            num_leaves=num_leaves,
            random_state=random_state,
            n_jobs=n_jobs,
            verbosity=-1,
            **kwargs
        )

    def train(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray]
    ) -> 'LightGBMRegressorModel':
        """Train the LightGBM regressor."""
        start_time = time.time()
        self.model.fit(X, y)
        self.training_time = time.time() - start_time
        self.is_fitted = True
        self.metadata["trained_at"] = pd.Timestamp.now().isoformat()
        return self

    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Make predictions."""
        if not self.is_fitted:
            raise RuntimeError("Model must be trained before making predictions")
        return self.model.predict(X)

    def evaluate(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray]
    ) -> Dict[str, float]:
        """Evaluate model with regression metrics."""
        predictions = self.predict(X)

        metrics = {
            "mae": mean_absolute_error(y, predictions),
            "mse": mean_squared_error(y, predictions),
            "rmse": np.sqrt(mean_squared_error(y, predictions)),
            "r2": r2_score(y, predictions),
        }

        if not np.any(y == 0):
            metrics["mape"] = mean_absolute_percentage_error(y, predictions)

        return metrics

    def get_feature_importance(self) -> np.ndarray:
        """Get feature importance scores."""
        if not self.is_fitted:
            raise RuntimeError("Model must be trained first")
        return self.model.feature_importances_


class CatBoostRegressorModel(SupervisedModel):
    """
    CatBoost regressor.

    Gradient boosting with native categorical feature support.

    Example:
        >>> model = CatBoostRegressorModel(iterations=100)
        >>> model.train(X_train, y_train)
    """

    def __init__(
        self,
        iterations: int = 100,
        learning_rate: float = 0.1,
        depth: int = 6,
        random_state: int = 42,
        verbose: bool = False,
        **kwargs: Any
    ):
        super().__init__(model_name="CatBoost Regressor", task_type="regression")
        self.model = CatBoostRegressor(
            iterations=iterations,
            learning_rate=learning_rate,
            depth=depth,
            random_state=random_state,
            verbose=verbose,
            **kwargs
        )

    def train(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray]
    ) -> 'CatBoostRegressorModel':
        """Train the CatBoost regressor."""
        start_time = time.time()
        self.model.fit(X, y)
        self.training_time = time.time() - start_time
        self.is_fitted = True
        self.metadata["trained_at"] = pd.Timestamp.now().isoformat()
        return self

    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Make predictions."""
        if not self.is_fitted:
            raise RuntimeError("Model must be trained before making predictions")
        return self.model.predict(X)

    def evaluate(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray]
    ) -> Dict[str, float]:
        """Evaluate model with regression metrics."""
        predictions = self.predict(X)

        metrics = {
            "mae": mean_absolute_error(y, predictions),
            "mse": mean_squared_error(y, predictions),
            "rmse": np.sqrt(mean_squared_error(y, predictions)),
            "r2": r2_score(y, predictions),
        }

        if not np.any(y == 0):
            metrics["mape"] = mean_absolute_percentage_error(y, predictions)

        return metrics

    def get_feature_importance(self) -> np.ndarray:
        """Get feature importance scores."""
        if not self.is_fitted:
            raise RuntimeError("Model must be trained first")
        return self.model.feature_importances_


class SVMRegressorModel(SupervisedModel):
    """
    Support Vector Machine regressor.

    Example:
        >>> model = SVMRegressorModel(kernel='rbf', C=1.0)
        >>> model.train(X_train, y_train)
    """

    def __init__(
        self,
        C: float = 1.0,
        kernel: str = 'rbf',
        gamma: str = 'scale',
        **kwargs: Any
    ):
        super().__init__(model_name="SVM Regressor", task_type="regression")
        self.model = SVR(C=C, kernel=kernel, gamma=gamma, **kwargs)

    def train(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray]
    ) -> 'SVMRegressorModel':
        """Train the SVM regressor."""
        start_time = time.time()
        self.model.fit(X, y)
        self.training_time = time.time() - start_time
        self.is_fitted = True
        self.metadata["trained_at"] = pd.Timestamp.now().isoformat()
        return self

    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Make predictions."""
        if not self.is_fitted:
            raise RuntimeError("Model must be trained before making predictions")
        return self.model.predict(X)

    def evaluate(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray]
    ) -> Dict[str, float]:
        """Evaluate model with regression metrics."""
        predictions = self.predict(X)

        metrics = {
            "mae": mean_absolute_error(y, predictions),
            "mse": mean_squared_error(y, predictions),
            "rmse": np.sqrt(mean_squared_error(y, predictions)),
            "r2": r2_score(y, predictions),
        }

        if not np.any(y == 0):
            metrics["mape"] = mean_absolute_percentage_error(y, predictions)

        return metrics
