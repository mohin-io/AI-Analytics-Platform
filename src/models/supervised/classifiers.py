"""
Classification model implementations for the Unified AI Analytics Platform

This module provides wrapper classes for popular classification algorithms,
ensuring a consistent interface across all models while exposing algorithm-specific
parameters and functionality.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Union
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, log_loss, classification_report, confusion_matrix
)
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
import time

from src.models.base import SupervisedModel


class LogisticRegressionModel(SupervisedModel):
    """
    Logistic Regression classifier.

    Logistic regression is a linear model for binary and multi-class classification.
    It models the probability of the target class using a logistic (sigmoid) function.
    Despite its name, it's a classification algorithm, not regression.

    Best for:
    - Linearly separable data
    - Binary and multi-class problems
    - When interpretability is important (coefficients show feature importance)
    - Baseline model for comparison

    Attributes:
        model: sklearn LogisticRegression instance
        C: Inverse of regularization strength (smaller = stronger regularization)
        max_iter: Maximum iterations for solver convergence
        solver: Algorithm for optimization

    Example:
        >>> model = LogisticRegressionModel(C=1.0, max_iter=1000)
        >>> model.train(X_train, y_train)
        >>> predictions = model.predict(X_test)
        >>> probs = model.predict_proba(X_test)
    """

    def __init__(
        self,
        C: float = 1.0,
        max_iter: int = 1000,
        solver: str = 'lbfgs',
        random_state: int = 42,
        **kwargs: Any
    ):
        """
        Initialize Logistic Regression model.

        Args:
            C: Inverse regularization strength (smaller = more regularization)
            max_iter: Maximum iterations for convergence
            solver: Optimization algorithm ('lbfgs', 'liblinear', 'saga')
            random_state: Random seed for reproducibility
            **kwargs: Additional parameters passed to LogisticRegression
        """
        super().__init__(model_name="Logistic Regression", task_type="classification")
        self.model = LogisticRegression(
            C=C,
            max_iter=max_iter,
            solver=solver,
            random_state=random_state,
            **kwargs
        )

    def train(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray]
    ) -> 'LogisticRegressionModel':
        """Train the logistic regression model."""
        start_time = time.time()
        self.model.fit(X, y)
        self.training_time = time.time() - start_time
        self.is_fitted = True
        self.metadata["trained_at"] = pd.Timestamp.now().isoformat()
        return self

    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Make class predictions."""
        if not self.is_fitted:
            raise RuntimeError("Model must be trained before making predictions")
        return self.model.predict(X)

    def evaluate(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray]
    ) -> Dict[str, float]:
        """
        Evaluate model performance with comprehensive metrics.

        Returns a dictionary containing:
        - accuracy: Overall correctness
        - precision: Positive predictive value
        - recall: True positive rate
        - f1_score: Harmonic mean of precision and recall
        - roc_auc: Area under ROC curve (if binary classification)
        - log_loss: Cross-entropy loss
        """
        predictions = self.predict(X)
        probabilities = self.predict_proba(X)

        metrics = {
            "accuracy": accuracy_score(y, predictions),
            "precision": precision_score(y, predictions, average='weighted', zero_division=0),
            "recall": recall_score(y, predictions, average='weighted', zero_division=0),
            "f1_score": f1_score(y, predictions, average='weighted', zero_division=0),
            "log_loss": log_loss(y, probabilities)
        }

        # Add ROC-AUC for binary classification
        if len(np.unique(y)) == 2:
            metrics["roc_auc"] = roc_auc_score(y, probabilities[:, 1])

        return metrics


class RandomForestClassifierModel(SupervisedModel):
    """
    Random Forest classifier.

    Random Forest is an ensemble of decision trees trained on random subsets
    of the data. It reduces overfitting compared to single decision trees and
    provides feature importance measures.

    Best for:
    - Non-linear relationships
    - High-dimensional data
    - When feature importance is needed
    - Robust to outliers and missing values

    The algorithm works by:
    1. Creating multiple decision trees from random data subsets
    2. Each tree votes for a class
    3. Final prediction is the majority vote

    Example:
        >>> model = RandomForestClassifierModel(n_estimators=100, max_depth=10)
        >>> model.train(X_train, y_train)
        >>> accuracy = model.evaluate(X_test, y_test)['accuracy']
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        random_state: int = 42,
        n_jobs: int = -1,
        **kwargs: Any
    ):
        """
        Initialize Random Forest classifier.

        Args:
            n_estimators: Number of trees in the forest
            max_depth: Maximum depth of trees (None = unlimited)
            min_samples_split: Minimum samples to split a node
            min_samples_leaf: Minimum samples in a leaf node
            random_state: Random seed
            n_jobs: Number of CPU cores to use (-1 = all)
            **kwargs: Additional sklearn parameters
        """
        super().__init__(model_name="Random Forest", task_type="classification")
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state,
            n_jobs=n_jobs,
            **kwargs
        )

    def train(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray]
    ) -> 'RandomForestClassifierModel':
        """Train the Random Forest model."""
        start_time = time.time()
        self.model.fit(X, y)
        self.training_time = time.time() - start_time
        self.is_fitted = True
        self.metadata["trained_at"] = pd.Timestamp.now().isoformat()
        return self

    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Make class predictions."""
        if not self.is_fitted:
            raise RuntimeError("Model must be trained before making predictions")
        return self.model.predict(X)

    def evaluate(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray]
    ) -> Dict[str, float]:
        """Evaluate model with comprehensive metrics."""
        predictions = self.predict(X)
        probabilities = self.predict_proba(X)

        metrics = {
            "accuracy": accuracy_score(y, predictions),
            "precision": precision_score(y, predictions, average='weighted', zero_division=0),
            "recall": recall_score(y, predictions, average='weighted', zero_division=0),
            "f1_score": f1_score(y, predictions, average='weighted', zero_division=0),
            "log_loss": log_loss(y, probabilities)
        }

        if len(np.unique(y)) == 2:
            metrics["roc_auc"] = roc_auc_score(y, probabilities[:, 1])

        return metrics

    def get_feature_importance(self) -> np.ndarray:
        """
        Get feature importance scores.

        Returns:
            Array of feature importance values (sum to 1.0)

        Example:
            >>> model.train(X_train, y_train)
            >>> importance = model.get_feature_importance()
            >>> top_features = np.argsort(importance)[-10:]  # Top 10 features
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be trained first")
        return self.model.feature_importances_


class XGBoostClassifierModel(SupervisedModel):
    """
    XGBoost classifier.

    XGBoost (eXtreme Gradient Boosting) is a highly efficient and scalable
    implementation of gradient boosting. It's one of the most powerful
    algorithms for structured/tabular data and wins many ML competitions.

    Key advantages:
    - Handles missing values automatically
    - Built-in regularization (L1 and L2)
    - Parallel processing
    - Tree pruning using max_depth
    - Cross-validation built-in

    Best for:
    - Structured/tabular data
    - When high accuracy is critical
    - Medium to large datasets
    - Competition-winning performance

    Example:
        >>> model = XGBoostClassifierModel(n_estimators=100, learning_rate=0.1)
        >>> model.train(X_train, y_train)
        >>> metrics = model.evaluate(X_test, y_test)
    """

    def __init__(
        self,
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        max_depth: int = 6,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        random_state: int = 42,
        n_jobs: int = -1,
        **kwargs: Any
    ):
        """
        Initialize XGBoost classifier.

        Args:
            n_estimators: Number of boosting rounds
            learning_rate: Step size shrinkage (lower = more conservative)
            max_depth: Maximum tree depth
            subsample: Fraction of samples for each tree
            colsample_bytree: Fraction of features for each tree
            random_state: Random seed
            n_jobs: Number of CPU cores
            **kwargs: Additional XGBoost parameters
        """
        super().__init__(model_name="XGBoost", task_type="classification")
        self.model = xgb.XGBClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            random_state=random_state,
            n_jobs=n_jobs,
            **kwargs
        )

    def train(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray]
    ) -> 'XGBoostClassifierModel':
        """Train the XGBoost model."""
        start_time = time.time()
        self.model.fit(X, y)
        self.training_time = time.time() - start_time
        self.is_fitted = True
        self.metadata["trained_at"] = pd.Timestamp.now().isoformat()
        return self

    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Make class predictions."""
        if not self.is_fitted:
            raise RuntimeError("Model must be trained before making predictions")
        return self.model.predict(X)

    def evaluate(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray]
    ) -> Dict[str, float]:
        """Evaluate model with comprehensive metrics."""
        predictions = self.predict(X)
        probabilities = self.predict_proba(X)

        metrics = {
            "accuracy": accuracy_score(y, predictions),
            "precision": precision_score(y, predictions, average='weighted', zero_division=0),
            "recall": recall_score(y, predictions, average='weighted', zero_division=0),
            "f1_score": f1_score(y, predictions, average='weighted', zero_division=0),
            "log_loss": log_loss(y, probabilities)
        }

        if len(np.unique(y)) == 2:
            metrics["roc_auc"] = roc_auc_score(y, probabilities[:, 1])

        return metrics

    def get_feature_importance(self) -> np.ndarray:
        """Get feature importance scores."""
        if not self.is_fitted:
            raise RuntimeError("Model must be trained first")
        return self.model.feature_importances_


class LightGBMClassifierModel(SupervisedModel):
    """
    LightGBM classifier.

    LightGBM is Microsoft's gradient boosting framework that uses tree-based
    learning. It's faster and more memory-efficient than XGBoost for large datasets.

    Key features:
    - Leaf-wise tree growth (vs level-wise)
    - Faster training speed
    - Lower memory usage
    - Better accuracy on some datasets
    - Handles categorical features natively

    Best for:
    - Large datasets (>10k rows)
    - When training speed matters
    - Memory-constrained environments
    - Categorical features

    Example:
        >>> model = LightGBMClassifierModel(n_estimators=100, num_leaves=31)
        >>> model.train(X_train, y_train)
        >>> predictions = model.predict(X_test)
    """

    def __init__(
        self,
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        num_leaves: int = 31,
        max_depth: int = -1,
        random_state: int = 42,
        n_jobs: int = -1,
        **kwargs: Any
    ):
        """
        Initialize LightGBM classifier.

        Args:
            n_estimators: Number of boosting rounds
            learning_rate: Shrinkage rate
            num_leaves: Maximum leaves in one tree
            max_depth: Maximum tree depth (-1 = no limit)
            random_state: Random seed
            n_jobs: Number of CPU cores
            **kwargs: Additional LightGBM parameters
        """
        super().__init__(model_name="LightGBM", task_type="classification")
        self.model = lgb.LGBMClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            num_leaves=num_leaves,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=n_jobs,
            verbosity=-1,  # Suppress warnings
            **kwargs
        )

    def train(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray]
    ) -> 'LightGBMClassifierModel':
        """Train the LightGBM model."""
        start_time = time.time()
        self.model.fit(X, y)
        self.training_time = time.time() - start_time
        self.is_fitted = True
        self.metadata["trained_at"] = pd.Timestamp.now().isoformat()
        return self

    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Make class predictions."""
        if not self.is_fitted:
            raise RuntimeError("Model must be trained before making predictions")
        return self.model.predict(X)

    def evaluate(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray]
    ) -> Dict[str, float]:
        """Evaluate model with comprehensive metrics."""
        predictions = self.predict(X)
        probabilities = self.predict_proba(X)

        metrics = {
            "accuracy": accuracy_score(y, predictions),
            "precision": precision_score(y, predictions, average='weighted', zero_division=0),
            "recall": recall_score(y, predictions, average='weighted', zero_division=0),
            "f1_score": f1_score(y, predictions, average='weighted', zero_division=0),
            "log_loss": log_loss(y, probabilities)
        }

        if len(np.unique(y)) == 2:
            metrics["roc_auc"] = roc_auc_score(y, probabilities[:, 1])

        return metrics

    def get_feature_importance(self) -> np.ndarray:
        """Get feature importance scores."""
        if not self.is_fitted:
            raise RuntimeError("Model must be trained first")
        return self.model.feature_importances_


class CatBoostClassifierModel(SupervisedModel):
    """
    CatBoost classifier.

    CatBoost (Categorical Boosting) is Yandex's gradient boosting library
    with native support for categorical features. It often requires less
    hyperparameter tuning than XGBoost and LightGBM.

    Key features:
    - Handles categorical features automatically
    - Reduces overfitting with ordered boosting
    - Fast prediction speed
    - Good default parameters
    - Built-in cross-validation

    Best for:
    - Datasets with many categorical features
    - When minimal tuning is desired
    - Production deployments (fast inference)

    Example:
        >>> model = CatBoostClassifierModel(iterations=100, depth=6)
        >>> model.train(X_train, y_train)
        >>> metrics = model.evaluate(X_test, y_test)
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
        """
        Initialize CatBoost classifier.

        Args:
            iterations: Number of boosting iterations
            learning_rate: Step size
            depth: Tree depth
            random_state: Random seed
            verbose: Whether to print training progress
            **kwargs: Additional CatBoost parameters
        """
        super().__init__(model_name="CatBoost", task_type="classification")
        self.model = CatBoostClassifier(
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
    ) -> 'CatBoostClassifierModel':
        """Train the CatBoost model."""
        start_time = time.time()
        self.model.fit(X, y)
        self.training_time = time.time() - start_time
        self.is_fitted = True
        self.metadata["trained_at"] = pd.Timestamp.now().isoformat()
        return self

    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Make class predictions."""
        if not self.is_fitted:
            raise RuntimeError("Model must be trained before making predictions")
        return self.model.predict(X)

    def evaluate(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray]
    ) -> Dict[str, float]:
        """Evaluate model with comprehensive metrics."""
        predictions = self.predict(X)
        probabilities = self.predict_proba(X)

        metrics = {
            "accuracy": accuracy_score(y, predictions),
            "precision": precision_score(y, predictions, average='weighted', zero_division=0),
            "recall": recall_score(y, predictions, average='weighted', zero_division=0),
            "f1_score": f1_score(y, predictions, average='weighted', zero_division=0),
            "log_loss": log_loss(y, probabilities)
        }

        if len(np.unique(y)) == 2:
            metrics["roc_auc"] = roc_auc_score(y, probabilities[:, 1])

        return metrics

    def get_feature_importance(self) -> np.ndarray:
        """Get feature importance scores."""
        if not self.is_fitted:
            raise RuntimeError("Model must be trained first")
        return self.model.feature_importances_


class SVMClassifierModel(SupervisedModel):
    """
    Support Vector Machine classifier.

    SVM finds the hyperplane that best separates classes with maximum margin.
    It can handle non-linear boundaries using kernel functions.

    Best for:
    - Small to medium datasets
    - High-dimensional data
    - Clear margin of separation
    - Binary classification (can be extended to multi-class)

    Note: SVMs can be slow on large datasets (>10k samples).

    Example:
        >>> model = SVMClassifierModel(kernel='rbf', C=1.0)
        >>> model.train(X_train, y_train)
        >>> predictions = model.predict(X_test)
    """

    def __init__(
        self,
        C: float = 1.0,
        kernel: str = 'rbf',
        gamma: str = 'scale',
        probability: bool = True,
        random_state: int = 42,
        **kwargs: Any
    ):
        """
        Initialize SVM classifier.

        Args:
            C: Regularization parameter
            kernel: Kernel type ('linear', 'rbf', 'poly')
            gamma: Kernel coefficient
            probability: Enable probability estimates
            random_state: Random seed
            **kwargs: Additional sklearn parameters
        """
        super().__init__(model_name="SVM", task_type="classification")
        self.model = SVC(
            C=C,
            kernel=kernel,
            gamma=gamma,
            probability=probability,
            random_state=random_state,
            **kwargs
        )

    def train(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray]
    ) -> 'SVMClassifierModel':
        """Train the SVM model."""
        start_time = time.time()
        self.model.fit(X, y)
        self.training_time = time.time() - start_time
        self.is_fitted = True
        self.metadata["trained_at"] = pd.Timestamp.now().isoformat()
        return self

    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Make class predictions."""
        if not self.is_fitted:
            raise RuntimeError("Model must be trained before making predictions")
        return self.model.predict(X)

    def evaluate(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray]
    ) -> Dict[str, float]:
        """Evaluate model with comprehensive metrics."""
        predictions = self.predict(X)
        probabilities = self.predict_proba(X)

        metrics = {
            "accuracy": accuracy_score(y, predictions),
            "precision": precision_score(y, predictions, average='weighted', zero_division=0),
            "recall": recall_score(y, predictions, average='weighted', zero_division=0),
            "f1_score": f1_score(y, predictions, average='weighted', zero_division=0),
            "log_loss": log_loss(y, probabilities)
        }

        if len(np.unique(y)) == 2:
            metrics["roc_auc"] = roc_auc_score(y, probabilities[:, 1])

        return metrics


class KNNClassifierModel(SupervisedModel):
    """
    K-Nearest Neighbors classifier.

    KNN is a simple, instance-based learning algorithm that classifies
    samples based on the majority class of k nearest neighbors.

    Best for:
    - Small datasets
    - Low-dimensional data
    - Non-linear boundaries
    - Baseline comparisons

    Note: Can be slow on large datasets as it needs to compute distances
    to all training samples for each prediction.

    Example:
        >>> model = KNNClassifierModel(n_neighbors=5)
        >>> model.train(X_train, y_train)
        >>> predictions = model.predict(X_test)
    """

    def __init__(
        self,
        n_neighbors: int = 5,
        weights: str = 'uniform',
        metric: str = 'minkowski',
        n_jobs: int = -1,
        **kwargs: Any
    ):
        """
        Initialize KNN classifier.

        Args:
            n_neighbors: Number of neighbors to use
            weights: Weight function ('uniform' or 'distance')
            metric: Distance metric
            n_jobs: Number of CPU cores
            **kwargs: Additional sklearn parameters
        """
        super().__init__(model_name="KNN", task_type="classification")
        self.model = KNeighborsClassifier(
            n_neighbors=n_neighbors,
            weights=weights,
            metric=metric,
            n_jobs=n_jobs,
            **kwargs
        )

    def train(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray]
    ) -> 'KNNClassifierModel':
        """Train the KNN model."""
        start_time = time.time()
        self.model.fit(X, y)
        self.training_time = time.time() - start_time
        self.is_fitted = True
        self.metadata["trained_at"] = pd.Timestamp.now().isoformat()
        return self

    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Make class predictions."""
        if not self.is_fitted:
            raise RuntimeError("Model must be trained before making predictions")
        return self.model.predict(X)

    def evaluate(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray]
    ) -> Dict[str, float]:
        """Evaluate model with comprehensive metrics."""
        predictions = self.predict(X)
        probabilities = self.predict_proba(X)

        metrics = {
            "accuracy": accuracy_score(y, predictions),
            "precision": precision_score(y, predictions, average='weighted', zero_division=0),
            "recall": recall_score(y, predictions, average='weighted', zero_division=0),
            "f1_score": f1_score(y, predictions, average='weighted', zero_division=0),
            "log_loss": log_loss(y, probabilities)
        }

        if len(np.unique(y)) == 2:
            metrics["roc_auc"] = roc_auc_score(y, probabilities[:, 1])

        return metrics


class NaiveBayesModel(SupervisedModel):
    """
    Naive Bayes classifier.

    Naive Bayes applies Bayes' theorem with the "naive" assumption of
    feature independence. Despite this simplifying assumption, it works
    well in practice, especially for text classification.

    Best for:
    - Text classification
    - Small datasets
    - Real-time predictions (very fast)
    - Baseline models

    Variants:
    - Gaussian: Continuous features (assumes Gaussian distribution)
    - Multinomial: Discrete counts (good for text)
    - Bernoulli: Binary features

    Example:
        >>> model = NaiveBayesModel(variant='gaussian')
        >>> model.train(X_train, y_train)
        >>> predictions = model.predict(X_test)
    """

    def __init__(self, variant: str = 'gaussian', **kwargs: Any):
        """
        Initialize Naive Bayes classifier.

        Args:
            variant: Type of Naive Bayes ('gaussian', 'multinomial', 'bernoulli')
            **kwargs: Additional parameters for the specific variant
        """
        super().__init__(model_name=f"Naive Bayes ({variant})", task_type="classification")

        if variant == 'gaussian':
            self.model = GaussianNB(**kwargs)
        elif variant == 'multinomial':
            self.model = MultinomialNB(**kwargs)
        elif variant == 'bernoulli':
            self.model = BernoulliNB(**kwargs)
        else:
            raise ValueError(f"Unknown variant: {variant}")

    def train(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray]
    ) -> 'NaiveBayesModel':
        """Train the Naive Bayes model."""
        start_time = time.time()
        self.model.fit(X, y)
        self.training_time = time.time() - start_time
        self.is_fitted = True
        self.metadata["trained_at"] = pd.Timestamp.now().isoformat()
        return self

    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Make class predictions."""
        if not self.is_fitted:
            raise RuntimeError("Model must be trained before making predictions")
        return self.model.predict(X)

    def evaluate(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray]
    ) -> Dict[str, float]:
        """Evaluate model with comprehensive metrics."""
        predictions = self.predict(X)
        probabilities = self.predict_proba(X)

        metrics = {
            "accuracy": accuracy_score(y, predictions),
            "precision": precision_score(y, predictions, average='weighted', zero_division=0),
            "recall": recall_score(y, predictions, average='weighted', zero_division=0),
            "f1_score": f1_score(y, predictions, average='weighted', zero_division=0),
            "log_loss": log_loss(y, probabilities)
        }

        if len(np.unique(y)) == 2:
            metrics["roc_auc"] = roc_auc_score(y, probabilities[:, 1])

        return metrics
