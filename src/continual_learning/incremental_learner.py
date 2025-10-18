"""
Incremental and Online Learning

Implements strategies for continual learning including:
- Incremental batch learning
- Online learning (sample-by-sample)
- Class-incremental learning
- Task-incremental learning
"""

import numpy as np
import pandas as pd
from typing import Optional, List, Dict, Any, Tuple
from sklearn.base import BaseEstimator, clone
from sklearn.linear_model import SGDClassifier, SGDRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import warnings


class IncrementalLearner:
    """
    Incremental learning wrapper for models.

    Supports batch incremental learning where models are updated
    with new batches of data while retaining previous knowledge.
    """

    def __init__(self, base_model: BaseEstimator, task_type: str = 'classification',
                 warm_start: bool = True):
        """
        Initialize incremental learner.

        Args:
            base_model: Base sklearn model (must support partial_fit or warm_start)
            task_type: 'classification' or 'regression'
            warm_start: Whether to use warm start for models that support it
        """
        self.base_model = base_model
        self.task_type = task_type
        self.warm_start = warm_start
        self.model = None
        self.classes_ = None
        self.n_batches_seen = 0
        self.training_history: List[Dict[str, Any]] = []

    def initial_fit(self, X: np.ndarray, y: np.ndarray) -> 'IncrementalLearner':
        """
        Initial training on first batch of data.

        Args:
            X: Feature matrix
            y: Labels

        Returns:
            Self
        """
        # Clone the base model
        self.model = clone(self.base_model)

        # Enable warm start if supported
        if self.warm_start and hasattr(self.model, 'warm_start'):
            self.model.warm_start = True

        # Store classes for classification
        if self.task_type == 'classification':
            self.classes_ = np.unique(y)

        # Fit the model
        self.model.fit(X, y)

        self.n_batches_seen = 1
        self.training_history.append({
            'batch': self.n_batches_seen,
            'samples': len(X),
            'action': 'initial_fit'
        })

        return self

    def partial_fit(self, X: np.ndarray, y: np.ndarray,
                   classes: Optional[np.ndarray] = None) -> 'IncrementalLearner':
        """
        Update model with new batch of data.

        Args:
            X: Feature matrix
            y: Labels
            classes: All possible classes (for partial_fit models)

        Returns:
            Self
        """
        if self.model is None:
            return self.initial_fit(X, y)

        # Update classes if new ones appear
        if self.task_type == 'classification':
            new_classes = np.unique(y)
            if self.classes_ is not None:
                self.classes_ = np.unique(np.concatenate([self.classes_, new_classes]))
            else:
                self.classes_ = new_classes

        # Use partial_fit if available
        if hasattr(self.model, 'partial_fit'):
            if self.task_type == 'classification':
                self.model.partial_fit(X, y, classes=classes or self.classes_)
            else:
                self.model.partial_fit(X, y)
        elif self.warm_start and hasattr(self.model, 'warm_start'):
            # Use warm start for models like Random Forest
            self.model.fit(X, y)
        else:
            warnings.warn("Model does not support incremental learning. "
                        "Using full retraining.")
            self.model.fit(X, y)

        self.n_batches_seen += 1
        self.training_history.append({
            'batch': self.n_batches_seen,
            'samples': len(X),
            'action': 'partial_fit'
        })

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions.

        Args:
            X: Feature matrix

        Returns:
            Predictions
        """
        if self.model is None:
            raise ValueError("Model not fitted. Call initial_fit() or partial_fit() first.")

        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities.

        Args:
            X: Feature matrix

        Returns:
            Class probabilities
        """
        if self.model is None:
            raise ValueError("Model not fitted. Call initial_fit() or partial_fit() first.")

        if not hasattr(self.model, 'predict_proba'):
            raise AttributeError("Model does not support predict_proba")

        return self.model.predict_proba(X)

    def get_training_history(self) -> pd.DataFrame:
        """
        Get training history.

        Returns:
            DataFrame with training history
        """
        return pd.DataFrame(self.training_history)


class OnlineLearner:
    """
    Online learning for streaming data.

    Updates model sample-by-sample or in mini-batches as data streams in.
    """

    def __init__(self, task_type: str = 'classification',
                 learning_rate: float = 0.01,
                 loss: str = 'log_loss',
                 mini_batch_size: int = 1):
        """
        Initialize online learner.

        Args:
            task_type: 'classification' or 'regression'
            learning_rate: Learning rate for SGD
            loss: Loss function ('log_loss', 'hinge', 'squared_error', etc.)
            mini_batch_size: Size of mini-batches (1 = pure online)
        """
        self.task_type = task_type
        self.learning_rate = learning_rate
        self.loss = loss
        self.mini_batch_size = mini_batch_size

        # Initialize SGD model
        if task_type == 'classification':
            self.model = SGDClassifier(
                loss=loss,
                learning_rate='constant',
                eta0=learning_rate,
                random_state=42
            )
        else:
            self.model = SGDRegressor(
                loss='squared_error' if loss == 'squared_error' else 'huber',
                learning_rate='constant',
                eta0=learning_rate,
                random_state=42
            )

        self.classes_ = None
        self.n_samples_seen = 0
        self.mini_batch_buffer: List[Tuple[np.ndarray, np.ndarray]] = []

    def partial_fit(self, X: np.ndarray, y: np.ndarray) -> 'OnlineLearner':
        """
        Update model with new sample(s).

        Args:
            X: Feature matrix (can be single sample or batch)
            y: Labels

        Returns:
            Self
        """
        # Ensure 2D array
        if X.ndim == 1:
            X = X.reshape(1, -1)

        # Update classes
        if self.task_type == 'classification':
            new_classes = np.unique(y)
            if self.classes_ is not None:
                self.classes_ = np.unique(np.concatenate([self.classes_, new_classes]))
            else:
                self.classes_ = new_classes

        # Add to mini-batch buffer
        self.mini_batch_buffer.append((X, y))

        # Update model when buffer is full
        if len(self.mini_batch_buffer) >= self.mini_batch_size:
            X_batch = np.vstack([x for x, _ in self.mini_batch_buffer])
            y_batch = np.concatenate([y for _, y in self.mini_batch_buffer])

            if self.task_type == 'classification':
                self.model.partial_fit(X_batch, y_batch, classes=self.classes_)
            else:
                self.model.partial_fit(X_batch, y_batch)

            self.n_samples_seen += len(X_batch)
            self.mini_batch_buffer = []

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions.

        Args:
            X: Feature matrix

        Returns:
            Predictions
        """
        # Ensure 2D array
        if X.ndim == 1:
            X = X.reshape(1, -1)

        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities.

        Args:
            X: Feature matrix

        Returns:
            Class probabilities
        """
        # Ensure 2D array
        if X.ndim == 1:
            X = X.reshape(1, -1)

        if not hasattr(self.model, 'predict_proba'):
            raise AttributeError("Model does not support predict_proba")

        return self.model.predict_proba(X)


class ClassIncrementalLearner:
    """
    Class-incremental learning.

    Learns new classes incrementally without forgetting previous classes.
    Uses a multi-head approach where each task has its own output head.
    """

    def __init__(self, base_model_class: type, task_type: str = 'classification'):
        """
        Initialize class-incremental learner.

        Args:
            base_model_class: Class of base model to use
            task_type: 'classification' or 'regression'
        """
        self.base_model_class = base_model_class
        self.task_type = task_type
        self.models: Dict[int, Any] = {}  # task_id -> model
        self.class_to_task: Dict[Any, int] = {}  # class_label -> task_id
        self.task_classes: Dict[int, List[Any]] = {}  # task_id -> [classes]
        self.current_task_id = 0

    def learn_task(self, X: np.ndarray, y: np.ndarray,
                  task_id: Optional[int] = None) -> 'ClassIncrementalLearner':
        """
        Learn a new task with potentially new classes.

        Args:
            X: Feature matrix
            y: Labels
            task_id: Task identifier (auto-incremented if None)

        Returns:
            Self
        """
        if task_id is None:
            task_id = self.current_task_id
            self.current_task_id += 1

        # Get classes for this task
        classes = np.unique(y)

        # Update class-to-task mapping
        for cls in classes:
            self.class_to_task[cls] = task_id

        self.task_classes[task_id] = classes.tolist()

        # Train model for this task
        model = self.base_model_class()
        model.fit(X, y)

        self.models[task_id] = model

        return self

    def predict(self, X: np.ndarray, task_id: Optional[int] = None) -> np.ndarray:
        """
        Make predictions.

        Args:
            X: Feature matrix
            task_id: Specific task to use (None = determine from data)

        Returns:
            Predictions
        """
        if task_id is not None:
            # Use specific task model
            if task_id not in self.models:
                raise ValueError(f"Task {task_id} not found")
            return self.models[task_id].predict(X)
        else:
            # Need to determine task from predictions of all models
            # This is a simplified approach - in practice, you'd use task boundaries
            if len(self.models) == 1:
                return list(self.models.values())[0].predict(X)
            else:
                # Use the most recent model as default
                latest_task = max(self.models.keys())
                return self.models[latest_task].predict(X)

    def predict_with_task_inference(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict with automatic task inference.

        Args:
            X: Feature matrix

        Returns:
            Tuple of (predictions, task_ids)
        """
        # Get predictions from all models
        all_predictions = {}
        all_probas = {}

        for task_id, model in self.models.items():
            predictions = model.predict(X)
            all_predictions[task_id] = predictions

            if hasattr(model, 'predict_proba'):
                probas = model.predict_proba(X)
                all_probas[task_id] = probas

        # Select prediction with highest confidence
        if all_probas:
            # Use probabilities for task selection
            final_predictions = []
            final_tasks = []

            for i in range(len(X)):
                best_task = None
                best_confidence = -1

                for task_id, probas in all_probas.items():
                    max_proba = np.max(probas[i])
                    if max_proba > best_confidence:
                        best_confidence = max_proba
                        best_task = task_id

                final_predictions.append(all_predictions[best_task][i])
                final_tasks.append(best_task)

            return np.array(final_predictions), np.array(final_tasks)
        else:
            # Use most recent model
            latest_task = max(self.models.keys())
            return all_predictions[latest_task], np.full(len(X), latest_task)

    def get_known_classes(self) -> List[Any]:
        """
        Get all classes the model has seen.

        Returns:
            List of all known classes
        """
        return list(self.class_to_task.keys())

    def get_task_info(self) -> Dict[int, List[Any]]:
        """
        Get information about all tasks.

        Returns:
            Dictionary mapping task_id to list of classes
        """
        return self.task_classes.copy()
