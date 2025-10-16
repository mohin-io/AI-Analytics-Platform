"""
Base model classes for the Unified AI Analytics Platform

This module defines abstract base classes that all model implementations inherit from.
These base classes ensure a consistent interface across all models and provide
common functionality for training, prediction, and model management.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Union
import pandas as pd
import numpy as np
from datetime import datetime
import joblib
from pathlib import Path


class BaseModel(ABC):
    """
    Abstract base class for all machine learning models.

    This class defines the interface that all models must implement,
    ensuring consistency across different algorithms. It provides
    common functionality for model lifecycle management including
    training, prediction, saving, and loading.

    All model classes in the platform inherit from this base class
    and implement the abstract methods according to their specific
    algorithm requirements.

    Attributes:
        model: The underlying ML model object
        model_name: Human-readable name of the model
        is_fitted: Whether the model has been trained
        training_time: Time taken to train the model (in seconds)
        metadata: Dictionary storing model metadata

    Example:
        >>> class MyModel(BaseModel):
        ...     def train(self, X, y):
        ...         # Implementation
        ...         pass
        ...     def predict(self, X):
        ...         # Implementation
        ...         pass
        ...     def evaluate(self, X, y):
        ...         # Implementation
        ...         pass
    """

    def __init__(self, model_name: str = "BaseModel"):
        """
        Initialize the base model.

        Args:
            model_name: Name identifier for the model

        The initialization sets up basic attributes that track the model's
        state and training history. These attributes are used throughout
        the model's lifecycle for monitoring and management.
        """
        self.model: Optional[Any] = None
        self.model_name = model_name
        self.is_fitted = False
        self.training_time: float = 0.0
        self.metadata: Dict[str, Any] = {
            "model_name": model_name,
            "created_at": datetime.now().isoformat(),
            "trained_at": None,
            "version": "1.0.0"
        }

    @abstractmethod
    def train(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray]
    ) -> 'BaseModel':
        """
        Train the model on the provided data.

        This is an abstract method that must be implemented by all subclasses.
        Each model type implements its own training logic while maintaining
        this consistent interface.

        Args:
            X: Training features (n_samples, n_features)
            y: Training target values (n_samples,)

        Returns:
            Self (for method chaining)

        Raises:
            NotImplementedError: If not implemented by subclass

        Example:
            >>> model = MyModel()
            >>> model.train(X_train, y_train)
            >>> # Model is now fitted and ready for predictions
        """
        raise NotImplementedError("Subclasses must implement train()")

    @abstractmethod
    def predict(
        self,
        X: Union[pd.DataFrame, np.ndarray]
    ) -> np.ndarray:
        """
        Make predictions on new data.

        This abstract method must be implemented to return predictions
        for the given input features. The specific prediction logic depends
        on the model type (classification, regression, etc.).

        Args:
            X: Features to predict on (n_samples, n_features)

        Returns:
            Array of predictions (n_samples,)

        Raises:
            NotImplementedError: If not implemented by subclass
            RuntimeError: If model hasn't been trained yet

        Example:
            >>> model = MyModel()
            >>> model.train(X_train, y_train)
            >>> predictions = model.predict(X_test)
        """
        raise NotImplementedError("Subclasses must implement predict()")

    @abstractmethod
    def evaluate(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray]
    ) -> Dict[str, float]:
        """
        Evaluate the model on test data.

        This method computes relevant performance metrics for the model.
        The specific metrics depend on the task type (classification metrics
        like accuracy/F1 for classifiers, RMSE/R² for regressors, etc.).

        Args:
            X: Test features
            y: True target values

        Returns:
            Dictionary of metric names and values

        Raises:
            NotImplementedError: If not implemented by subclass

        Example:
            >>> model = MyModel()
            >>> model.train(X_train, y_train)
            >>> metrics = model.evaluate(X_test, y_test)
            >>> print(f"Accuracy: {metrics['accuracy']}")
        """
        raise NotImplementedError("Subclasses must implement evaluate()")

    def save(self, file_path: str) -> None:
        """
        Save the model to disk.

        This method serializes the entire model object including the trained
        parameters, metadata, and configuration. It uses joblib for efficient
        serialization, especially for models containing large numpy arrays.

        Args:
            file_path: Path where the model will be saved

        Raises:
            RuntimeError: If model hasn't been trained yet

        Example:
            >>> model = MyModel()
            >>> model.train(X_train, y_train)
            >>> model.save("models/my_model.pkl")
        """
        if not self.is_fitted:
            raise RuntimeError("Cannot save a model that hasn't been trained")

        # Create directory if it doesn't exist
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)

        # Save the entire object
        joblib.dump(self, file_path)

    @classmethod
    def load(cls, file_path: str) -> 'BaseModel':
        """
        Load a saved model from disk.

        This class method deserializes a previously saved model and returns
        a fully initialized model object ready for predictions.

        Args:
            file_path: Path to the saved model file

        Returns:
            Loaded model instance

        Raises:
            FileNotFoundError: If the file doesn't exist

        Example:
            >>> model = MyModel.load("models/my_model.pkl")
            >>> predictions = model.predict(X_new)
        """
        if not Path(file_path).exists():
            raise FileNotFoundError(f"Model file not found: {file_path}")

        return joblib.load(file_path)

    def get_params(self) -> Dict[str, Any]:
        """
        Get model parameters.

        This method returns the configuration parameters of the model.
        For scikit-learn compatible models, it uses get_params(). For
        custom models, it returns an empty dict by default.

        Returns:
            Dictionary of parameter names and values

        Example:
            >>> model = MyModel(learning_rate=0.01, n_estimators=100)
            >>> params = model.get_params()
            >>> print(params)
            {'learning_rate': 0.01, 'n_estimators': 100}
        """
        if hasattr(self.model, 'get_params'):
            return self.model.get_params()
        return {}

    def set_params(self, **params: Any) -> 'BaseModel':
        """
        Set model parameters.

        This method allows updating model parameters after initialization.
        Useful for hyperparameter tuning and experimentation.

        Args:
            **params: Parameter names and values to set

        Returns:
            Self (for method chaining)

        Example:
            >>> model = MyModel()
            >>> model.set_params(learning_rate=0.001, n_estimators=200)
            >>> model.train(X_train, y_train)
        """
        if hasattr(self.model, 'set_params'):
            self.model.set_params(**params)
        return self

    def get_metadata(self) -> Dict[str, Any]:
        """
        Get model metadata.

        Returns comprehensive information about the model including
        name, training time, creation date, and any custom metadata.

        Returns:
            Dictionary containing model metadata

        Example:
            >>> model = MyModel()
            >>> model.train(X_train, y_train)
            >>> metadata = model.get_metadata()
            >>> print(f"Training time: {metadata['training_time']} seconds")
        """
        return {
            **self.metadata,
            "is_fitted": self.is_fitted,
            "training_time": self.training_time,
            "params": self.get_params()
        }

    def __repr__(self) -> str:
        """String representation of the model."""
        status = "fitted" if self.is_fitted else "not fitted"
        return f"{self.model_name}({status})"


class SupervisedModel(BaseModel):
    """
    Base class for supervised learning models.

    This class extends BaseModel with additional functionality specific
    to supervised learning tasks (classification and regression). It provides
    common methods like predict_proba for classifiers and additional
    evaluation capabilities.

    Supervised models learn from labeled data (features X and target y)
    and make predictions on new unlabeled data.

    Example:
        >>> class MyClassifier(SupervisedModel):
        ...     def __init__(self):
        ...         super().__init__(model_name="MyClassifier")
        ...         self.task_type = "classification"
    """

    def __init__(
        self,
        model_name: str = "SupervisedModel",
        task_type: str = "classification"
    ):
        """
        Initialize supervised model.

        Args:
            model_name: Name of the model
            task_type: Type of supervised task
                - 'classification': Predicting discrete classes
                - 'regression': Predicting continuous values

        The task_type determines which evaluation metrics and prediction
        methods are appropriate for the model.
        """
        super().__init__(model_name)
        self.task_type = task_type
        self.metadata["task_type"] = task_type

    def predict_proba(
        self,
        X: Union[pd.DataFrame, np.ndarray]
    ) -> np.ndarray:
        """
        Predict class probabilities (for classification models).

        This method returns probability estimates for each class rather than
        hard predictions. Not all models support probability prediction;
        those that don't will raise NotImplementedError.

        Args:
            X: Features to predict on

        Returns:
            Array of shape (n_samples, n_classes) with probability estimates

        Raises:
            NotImplementedError: If model doesn't support probability prediction
            RuntimeError: If model hasn't been trained

        Example:
            >>> model = MyClassifier()
            >>> model.train(X_train, y_train)
            >>> probs = model.predict_proba(X_test)
            >>> print(f"Probability of class 1: {probs[:, 1]}")
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be trained before making predictions")

        if self.task_type != "classification":
            raise NotImplementedError("predict_proba only available for classification")

        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X)
        else:
            raise NotImplementedError(
                f"{self.model_name} does not support probability prediction"
            )

    def score(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray]
    ) -> float:
        """
        Return the default score for the model.

        For classification, this is typically accuracy. For regression,
        it's typically R² score. This provides a quick single-number
        assessment of model performance.

        Args:
            X: Test features
            y: True target values

        Returns:
            Score value (higher is better)

        Example:
            >>> model = MyModel()
            >>> model.train(X_train, y_train)
            >>> score = model.score(X_test, y_test)
            >>> print(f"Model score: {score:.3f}")
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be trained before scoring")

        if hasattr(self.model, 'score'):
            return self.model.score(X, y)
        else:
            # Fallback: compute basic metric
            predictions = self.predict(X)
            if self.task_type == "classification":
                from sklearn.metrics import accuracy_score
                return accuracy_score(y, predictions)
            else:
                from sklearn.metrics import r2_score
                return r2_score(y, predictions)


class UnsupervisedModel(BaseModel):
    """
    Base class for unsupervised learning models.

    This class extends BaseModel for unsupervised learning tasks like
    clustering and dimensionality reduction. Unlike supervised models,
    these don't require target labels and focus on finding patterns
    in the data itself.

    Example:
        >>> class MyClusterer(UnsupervisedModel):
        ...     def __init__(self):
        ...         super().__init__(model_name="MyClusterer")
        ...         self.algorithm_type = "clustering"
    """

    def __init__(
        self,
        model_name: str = "UnsupervisedModel",
        algorithm_type: str = "clustering"
    ):
        """
        Initialize unsupervised model.

        Args:
            model_name: Name of the model
            algorithm_type: Type of unsupervised algorithm
                - 'clustering': Group similar data points
                - 'dimensionality_reduction': Reduce feature space
                - 'anomaly_detection': Identify outliers

        The algorithm_type helps categorize the model and determines
        which evaluation metrics are appropriate.
        """
        super().__init__(model_name)
        self.algorithm_type = algorithm_type
        self.metadata["algorithm_type"] = algorithm_type

    def train(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Optional[Union[pd.Series, np.ndarray]] = None
    ) -> 'UnsupervisedModel':
        """
        Train the unsupervised model.

        Note that unsupervised models don't use the y parameter, but it's
        included in the signature for API consistency. Most unsupervised
        algorithms use fit() rather than train(), so subclasses should
        implement the appropriate method.

        Args:
            X: Training features
            y: Ignored (kept for API consistency)

        Returns:
            Self (for method chaining)
        """
        raise NotImplementedError("Subclasses must implement train()")

    def transform(
        self,
        X: Union[pd.DataFrame, np.ndarray]
    ) -> np.ndarray:
        """
        Transform data using the fitted model.

        This method is relevant for dimensionality reduction algorithms
        (like PCA, t-SNE) that transform data into a new representation.
        For clustering algorithms, this typically returns cluster assignments.

        Args:
            X: Data to transform

        Returns:
            Transformed data

        Example:
            >>> model = MyDimensionalityReducer()
            >>> model.train(X_train)
            >>> X_reduced = model.transform(X_test)
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be trained before transforming")

        if hasattr(self.model, 'transform'):
            return self.model.transform(X)
        else:
            raise NotImplementedError(
                f"{self.model_name} does not support transform()"
            )

    def fit_transform(
        self,
        X: Union[pd.DataFrame, np.ndarray]
    ) -> np.ndarray:
        """
        Fit the model and transform data in one step.

        This is a convenience method that combines training and transformation.
        It's more efficient than calling train() and transform() separately
        for some algorithms.

        Args:
            X: Data to fit and transform

        Returns:
            Transformed data

        Example:
            >>> model = MyDimensionalityReducer()
            >>> X_reduced = model.fit_transform(X_train)
        """
        self.train(X)
        return self.transform(X)
