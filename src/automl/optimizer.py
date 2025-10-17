"""
AutoML optimizer for automated model selection and hyperparameter tuning

This module uses Optuna for Bayesian hyperparameter optimization and
automatic algorithm selection.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List, Callable
import optuna
from optuna.samplers import TPESampler
from sklearn.model_selection import cross_val_score
import warnings

from src.models.supervised import (
    LogisticRegressionModel,
    RandomForestClassifierModel,
    XGBoostClassifierModel,
    LightGBMClassifierModel,
    LinearRegressionModel,
    RandomForestRegressorModel,
    XGBoostRegressorModel,
    LightGBMRegressorModel
)


class HyperparameterOptimizer:
    """
    Optimize hyperparameters using Optuna.

    This class provides Bayesian optimization for hyperparameter tuning
    using Optuna's Tree-structured Parzen Estimator (TPE).

    Example:
        >>> optimizer = HyperparameterOptimizer(
        ...     model_class=XGBoostClassifierModel,
        ...     param_space={'n_estimators': (50, 300), 'learning_rate': (0.01, 0.3)},
        ...     task='classification'
        ... )
        >>> best_params = optimizer.optimize(X_train, y_train, n_trials=100)
    """

    def __init__(
        self,
        model_class: Any,
        param_space: Dict[str, Any],
        task: str = 'classification',
        cv: int = 5,
        scoring: Optional[str] = None,
        random_state: int = 42
    ):
        """
        Initialize hyperparameter optimizer.

        Args:
            model_class: Model class to optimize
            param_space: Dictionary defining parameter search space
            task: 'classification' or 'regression'
            cv: Number of cross-validation folds
            scoring: Scoring metric (None = default)
            random_state: Random seed
        """
        self.model_class = model_class
        self.param_space = param_space
        self.task = task
        self.cv = cv
        self.scoring = scoring or ('accuracy' if task == 'classification' else 'r2')
        self.random_state = random_state
        self.best_params: Optional[Dict] = None
        self.best_score: Optional[float] = None
        self.study: Optional[optuna.Study] = None

    def _objective(
        self,
        trial: optuna.Trial,
        X: np.ndarray,
        y: np.ndarray
    ) -> float:
        """
        Objective function for Optuna optimization.

        Args:
            trial: Optuna trial object
            X: Training features
            y: Training target

        Returns:
            Cross-validation score
        """
        # Suggest hyperparameters
        params = {}
        for param_name, param_range in self.param_space.items():
            if isinstance(param_range, tuple) and len(param_range) == 2:
                if isinstance(param_range[0], int):
                    params[param_name] = trial.suggest_int(param_name, *param_range)
                elif isinstance(param_range[0], float):
                    params[param_name] = trial.suggest_float(param_name, *param_range)
            elif isinstance(param_range, list):
                params[param_name] = trial.suggest_categorical(param_name, param_range)

        # Create model with suggested parameters
        model = self.model_class(**params)

        # Evaluate with cross-validation
        try:
            model.train(X, y)
            scores = cross_val_score(
                model.model,
                X, y,
                cv=self.cv,
                scoring=self.scoring,
                n_jobs=-1
            )
            return scores.mean()
        except Exception as e:
            return float('-inf')

    def optimize(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_trials: int = 100,
        timeout: Optional[int] = None,
        show_progress: bool = True
    ) -> Dict[str, Any]:
        """
        Run hyperparameter optimization.

        Args:
            X: Training features
            y: Training target
            n_trials: Number of optimization trials
            timeout: Time limit in seconds
            show_progress: Show progress bar

        Returns:
            Dictionary of best hyperparameters

        Example:
            >>> optimizer = HyperparameterOptimizer(
            ...     model_class=XGBoostClassifierModel,
            ...     param_space={'n_estimators': (50, 300), 'learning_rate': (0.01, 0.3)},
            ...     task='classification'
            ... )
            >>> best_params = optimizer.optimize(X_train, y_train, n_trials=50)
            >>> print(f"Best parameters: {best_params}")
        """
        # Create study
        sampler = TPESampler(seed=self.random_state)
        self.study = optuna.create_study(
            direction='maximize',
            sampler=sampler
        )

        # Optimize
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.study.optimize(
                lambda trial: self._objective(trial, X, y),
                n_trials=n_trials,
                timeout=timeout,
                show_progress_bar=show_progress
            )

        self.best_params = self.study.best_params
        self.best_score = self.study.best_value

        return self.best_params

    def get_best_model(self) -> Any:
        """
        Get model with best hyperparameters.

        Returns:
            Model instance with optimized parameters
        """
        if self.best_params is None:
            raise ValueError("Must run optimize() first")

        return self.model_class(**self.best_params)


class AutoMLOptimizer:
    """
    Automated machine learning optimizer.

    Automatically selects the best algorithm and tunes hyperparameters.
    Tests multiple algorithms and returns the best performing one.

    Example:
        >>> automl = AutoMLOptimizer(task='classification', time_budget=3600)
        >>> best_model = automl.fit(X_train, y_train)
        >>> predictions = best_model.predict(X_test)
    """

    def __init__(
        self,
        task: str = 'classification',
        time_budget: int = 3600,
        n_trials_per_model: int = 50,
        cv: int = 5,
        scoring: Optional[str] = None,
        random_state: int = 42
    ):
        """
        Initialize AutoML optimizer.

        Args:
            task: 'classification' or 'regression'
            time_budget: Total time budget in seconds
            n_trials_per_model: Optuna trials per model
            cv: Cross-validation folds
            scoring: Scoring metric
            random_state: Random seed
        """
        self.task = task
        self.time_budget = time_budget
        self.n_trials_per_model = n_trials_per_model
        self.cv = cv
        self.scoring = scoring or ('accuracy' if task == 'classification' else 'r2')
        self.random_state = random_state

        self.best_model = None
        self.best_score = float('-inf')
        self.best_algorithm = None
        self.results = {}

        # Define algorithms to try
        if task == 'classification':
            self.algorithms = {
                'LogisticRegression': (LogisticRegressionModel, {
                    'C': (0.001, 100.0),
                    'max_iter': (100, 1000)
                }),
                'RandomForest': (RandomForestClassifierModel, {
                    'n_estimators': (50, 300),
                    'max_depth': (3, 20),
                    'min_samples_split': (2, 20)
                }),
                'XGBoost': (XGBoostClassifierModel, {
                    'n_estimators': (50, 300),
                    'learning_rate': (0.01, 0.3),
                    'max_depth': (3, 10),
                    'subsample': (0.6, 1.0)
                }),
                'LightGBM': (LightGBMClassifierModel, {
                    'n_estimators': (50, 300),
                    'learning_rate': (0.01, 0.3),
                    'num_leaves': (20, 100)
                })
            }
        else:  # regression
            self.algorithms = {
                'LinearRegression': (LinearRegressionModel, {}),
                'RandomForest': (RandomForestRegressorModel, {
                    'n_estimators': (50, 300),
                    'max_depth': (3, 20),
                    'min_samples_split': (2, 20)
                }),
                'XGBoost': (XGBoostRegressorModel, {
                    'n_estimators': (50, 300),
                    'learning_rate': (0.01, 0.3),
                    'max_depth': (3, 10),
                    'subsample': (0.6, 1.0)
                }),
                'LightGBM': (LightGBMRegressorModel, {
                    'n_estimators': (50, 300),
                    'learning_rate': (0.01, 0.3),
                    'num_leaves': (20, 100)
                })
            }

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        algorithms: Optional[List[str]] = None
    ) -> Any:
        """
        Fit AutoML optimizer.

        Tests multiple algorithms with hyperparameter optimization and
        returns the best performing model.

        Args:
            X: Training features
            y: Training target
            algorithms: Specific algorithms to try (None = all)

        Returns:
            Best trained model

        Example:
            >>> automl = AutoMLOptimizer(task='classification', n_trials_per_model=30)
            >>> best_model = automl.fit(X_train, y_train)
            >>> print(f"Best algorithm: {automl.best_algorithm}")
            >>> print(f"Best score: {automl.best_score:.4f}")
        """
        algorithms_to_try = algorithms or list(self.algorithms.keys())

        print(f"AutoML: Testing {len(algorithms_to_try)} algorithms...")

        for algo_name in algorithms_to_try:
            if algo_name not in self.algorithms:
                print(f"Warning: Unknown algorithm '{algo_name}', skipping")
                continue

            model_class, param_space = self.algorithms[algo_name]

            print(f"\nOptimizing {algo_name}...")

            if param_space:
                # Optimize hyperparameters
                optimizer = HyperparameterOptimizer(
                    model_class=model_class,
                    param_space=param_space,
                    task=self.task,
                    cv=self.cv,
                    scoring=self.scoring,
                    random_state=self.random_state
                )

                best_params = optimizer.optimize(
                    X, y,
                    n_trials=self.n_trials_per_model,
                    show_progress=False
                )

                score = optimizer.best_score
                model = optimizer.get_best_model()
            else:
                # No hyperparameters to tune
                model = model_class()
                model.train(X, y)
                scores = cross_val_score(
                    model.model, X, y,
                    cv=self.cv,
                    scoring=self.scoring,
                    n_jobs=-1
                )
                score = scores.mean()
                best_params = {}

            self.results[algo_name] = {
                'score': score,
                'params': best_params
            }

            print(f"{algo_name} score: {score:.4f}")

            # Update best model
            if score > self.best_score:
                self.best_score = score
                self.best_algorithm = algo_name
                self.best_model = model

        print(f"\n{'='*60}")
        print(f"Best Algorithm: {self.best_algorithm}")
        print(f"Best Score: {self.best_score:.4f}")
        print(f"{'='*60}\n")

        # Train best model on full data
        self.best_model.train(X, y)

        return self.best_model

    def get_leaderboard(self) -> pd.DataFrame:
        """
        Get leaderboard of all tested algorithms.

        Returns:
            DataFrame with algorithm rankings

        Example:
            >>> automl.fit(X_train, y_train)
            >>> leaderboard = automl.get_leaderboard()
            >>> print(leaderboard)
        """
        if not self.results:
            raise ValueError("Must run fit() first")

        leaderboard_data = []
        for algo_name, result in self.results.items():
            leaderboard_data.append({
                'Algorithm': algo_name,
                'Score': result['score'],
                'Best': '✓' if algo_name == self.best_algorithm else ''
            })

        df = pd.DataFrame(leaderboard_data)
        df = df.sort_values('Score', ascending=False).reset_index(drop=True)
        df['Rank'] = range(1, len(df) + 1)

        return df[['Rank', 'Algorithm', 'Score', 'Best']]

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using the best model.

        Args:
            X: Features to predict on

        Returns:
            Predictions
        """
        if self.best_model is None:
            raise ValueError("Must run fit() first")

        return self.best_model.predict(X)

    def get_best_params(self) -> Dict[str, Any]:
        """Get best hyperparameters for the winning algorithm."""
        if self.best_algorithm is None:
            raise ValueError("Must run fit() first")

        return self.results[self.best_algorithm]['params']
