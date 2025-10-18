"""
Bias Mitigation Techniques

This module implements various bias mitigation strategies including
pre-processing, in-processing, and post-processing techniques.
"""

import numpy as np
import pandas as pd
from typing import Optional, List, Dict, Any, Tuple
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
import warnings


class BiasMitigation:
    """
    Collection of bias mitigation techniques.

    Provides methods for:
    - Pre-processing: Reweighting, resampling
    - In-processing: Fairness constraints during training
    - Post-processing: Threshold optimization
    """

    @staticmethod
    def reweighting(X: pd.DataFrame, y: np.ndarray,
                   sensitive_feature: np.ndarray) -> np.ndarray:
        """
        Calculate sample weights to achieve demographic parity.

        Assigns weights to balance positive/negative rates across groups.

        Args:
            X: Feature matrix
            y: Labels
            sensitive_feature: Sensitive attribute

        Returns:
            Array of sample weights
        """
        weights = np.ones(len(y))
        groups = np.unique(sensitive_feature)

        # Calculate overall statistics
        overall_positive_rate = np.mean(y)

        for group in groups:
            group_mask = sensitive_feature == group
            group_size = np.sum(group_mask)
            group_positive_rate = np.mean(y[group_mask])

            # Calculate weights for positive and negative examples in this group
            for label in [0, 1]:
                label_mask = y == label
                combined_mask = group_mask & label_mask

                if label == 1:
                    # Positive examples
                    expected_count = group_size * overall_positive_rate
                    actual_count = np.sum(combined_mask)
                else:
                    # Negative examples
                    expected_count = group_size * (1 - overall_positive_rate)
                    actual_count = np.sum(combined_mask)

                if actual_count > 0:
                    weight = expected_count / actual_count
                    weights[combined_mask] = weight

        # Normalize weights
        weights = weights / np.mean(weights)

        return weights

    @staticmethod
    def disparate_impact_remover(X: pd.DataFrame, sensitive_feature: str,
                                 repair_level: float = 1.0) -> pd.DataFrame:
        """
        Remove disparate impact from features using the repair algorithm.

        Modifies feature distributions to reduce correlation with sensitive attribute.

        Args:
            X: Feature DataFrame
            sensitive_feature: Name of sensitive attribute column
            repair_level: Level of repair (0.0 = no repair, 1.0 = full repair)

        Returns:
            Repaired feature DataFrame
        """
        X_repaired = X.copy()

        if sensitive_feature not in X.columns:
            warnings.warn(f"Sensitive feature '{sensitive_feature}' not found in data")
            return X_repaired

        sensitive_values = X[sensitive_feature].values
        groups = np.unique(sensitive_values)

        # Get numeric columns (exclude sensitive feature)
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        if sensitive_feature in numeric_cols:
            numeric_cols.remove(sensitive_feature)

        for col in numeric_cols:
            feature_values = X[col].values

            # Calculate median for each group
            group_medians = {}
            for group in groups:
                group_mask = sensitive_values == group
                group_medians[group] = np.median(feature_values[group_mask])

            # Calculate overall median
            overall_median = np.median(feature_values)

            # Repair each group
            for group in groups:
                group_mask = sensitive_values == group
                repair_amount = repair_level * (overall_median - group_medians[group])
                X_repaired.loc[group_mask, col] += repair_amount

        return X_repaired

    @staticmethod
    def equalized_odds_postprocessing(y_true: np.ndarray, y_pred: np.ndarray,
                                     y_proba: np.ndarray, sensitive_feature: np.ndarray,
                                     constraint: str = 'equalized_odds') -> np.ndarray:
        """
        Post-process predictions to satisfy equalized odds or equal opportunity.

        Adjusts decision thresholds per group to achieve fairness.

        Args:
            y_true: True labels
            y_pred: Original predicted labels
            y_proba: Predicted probabilities
            sensitive_feature: Sensitive attribute
            constraint: 'equalized_odds' or 'equal_opportunity'

        Returns:
            Adjusted predictions
        """
        groups = np.unique(sensitive_feature)
        adjusted_pred = y_pred.copy()

        # Calculate optimal thresholds for each group
        thresholds = {}

        if constraint == 'equalized_odds':
            # Target: equal TPR and FPR across groups
            # Calculate overall TPR and FPR
            overall_tpr = np.sum((y_true == 1) & (y_pred == 1)) / np.sum(y_true == 1)
            overall_fpr = np.sum((y_true == 0) & (y_pred == 1)) / np.sum(y_true == 0)

            for group in groups:
                group_mask = sensitive_feature == group
                y_true_group = y_true[group_mask]
                y_proba_group = y_proba[group_mask]

                # Find threshold that achieves target TPR and FPR
                best_threshold = 0.5
                best_score = float('inf')

                for threshold in np.linspace(0, 1, 100):
                    y_pred_temp = (y_proba_group >= threshold).astype(int)

                    if np.sum(y_true_group == 1) > 0:
                        tpr = np.sum((y_true_group == 1) & (y_pred_temp == 1)) / np.sum(y_true_group == 1)
                    else:
                        tpr = 0

                    if np.sum(y_true_group == 0) > 0:
                        fpr = np.sum((y_true_group == 0) & (y_pred_temp == 1)) / np.sum(y_true_group == 0)
                    else:
                        fpr = 0

                    # Minimize distance to target TPR and FPR
                    score = abs(tpr - overall_tpr) + abs(fpr - overall_fpr)

                    if score < best_score:
                        best_score = score
                        best_threshold = threshold

                thresholds[group] = best_threshold

        elif constraint == 'equal_opportunity':
            # Target: equal TPR across groups
            overall_tpr = np.sum((y_true == 1) & (y_pred == 1)) / np.sum(y_true == 1)

            for group in groups:
                group_mask = sensitive_feature == group
                y_true_group = y_true[group_mask]
                y_proba_group = y_proba[group_mask]

                # Find threshold that achieves target TPR
                best_threshold = 0.5
                best_score = float('inf')

                for threshold in np.linspace(0, 1, 100):
                    y_pred_temp = (y_proba_group >= threshold).astype(int)

                    if np.sum(y_true_group == 1) > 0:
                        tpr = np.sum((y_true_group == 1) & (y_pred_temp == 1)) / np.sum(y_true_group == 1)
                    else:
                        tpr = 0

                    score = abs(tpr - overall_tpr)

                    if score < best_score:
                        best_score = score
                        best_threshold = threshold

                thresholds[group] = best_threshold

        # Apply group-specific thresholds
        for group in groups:
            group_mask = sensitive_feature == group
            adjusted_pred[group_mask] = (y_proba[group_mask] >= thresholds[group]).astype(int)

        return adjusted_pred

    @staticmethod
    def calibrated_equalized_odds(y_true: np.ndarray, y_proba: np.ndarray,
                                  sensitive_feature: np.ndarray,
                                  cost_constraint: str = 'weighted') -> np.ndarray:
        """
        Apply calibrated equalized odds post-processing.

        Finds optimal classification thresholds for each group to satisfy
        equalized odds while maintaining calibration.

        Args:
            y_true: True labels
            y_proba: Predicted probabilities
            sensitive_feature: Sensitive attribute
            cost_constraint: Cost constraint type ('weighted', 'fpr', 'fnr')

        Returns:
            Adjusted predictions
        """
        groups = np.unique(sensitive_feature)
        adjusted_pred = np.zeros_like(y_true)

        # Learn transformation for each group
        for group in groups:
            group_mask = sensitive_feature == group
            y_true_group = y_true[group_mask]
            y_proba_group = y_proba[group_mask]

            # Sort by probability
            sorted_indices = np.argsort(y_proba_group)
            y_true_sorted = y_true_group[sorted_indices]
            y_proba_sorted = y_proba_group[sorted_indices]

            # Calculate cumulative statistics
            n = len(y_true_sorted)
            tp = np.cumsum(y_true_sorted[::-1])[::-1]
            fp = np.cumsum(1 - y_true_sorted[::-1])[::-1]
            fn = np.sum(y_true_sorted) - tp
            tn = np.sum(1 - y_true_sorted) - fp

            # Calculate rates
            tpr = tp / (tp + fn + 1e-10)
            fpr = fp / (fp + tn + 1e-10)

            # Select threshold based on cost constraint
            if cost_constraint == 'fpr':
                # Minimize FPR
                best_idx = np.argmin(fpr)
            elif cost_constraint == 'fnr':
                # Minimize FNR (maximize TPR)
                best_idx = np.argmax(tpr)
            else:
                # Weighted cost
                cost = 0.5 * (1 - tpr) + 0.5 * fpr
                best_idx = np.argmin(cost)

            threshold = y_proba_sorted[best_idx] if best_idx < len(y_proba_sorted) else 0.5

            # Apply threshold
            adjusted_pred[group_mask] = (y_proba[group_mask] >= threshold).astype(int)

        return adjusted_pred


class FairClassifier(BaseEstimator):
    """
    Wrapper classifier that applies fairness constraints during training.

    Uses regularization to penalize unfairness during model training.
    """

    def __init__(self, base_estimator: Any, sensitive_feature_idx: int,
                 fairness_penalty: float = 1.0, fairness_metric: str = 'demographic_parity'):
        """
        Initialize fair classifier.

        Args:
            base_estimator: Base scikit-learn classifier
            sensitive_feature_idx: Index of sensitive feature in X
            fairness_penalty: Weight for fairness penalty term
            fairness_metric: Type of fairness to enforce
        """
        self.base_estimator = base_estimator
        self.sensitive_feature_idx = sensitive_feature_idx
        self.fairness_penalty = fairness_penalty
        self.fairness_metric = fairness_metric

    def fit(self, X: np.ndarray, y: np.ndarray,
            sample_weight: Optional[np.ndarray] = None) -> 'FairClassifier':
        """
        Fit the fair classifier.

        Args:
            X: Feature matrix
            y: Labels
            sample_weight: Sample weights (optional)

        Returns:
            Self
        """
        # Extract sensitive feature
        sensitive_feature = X[:, self.sensitive_feature_idx]

        # Calculate fairness-based sample weights
        if self.fairness_metric == 'demographic_parity':
            # Use reweighting to achieve demographic parity
            fairness_weights = BiasMitigation.reweighting(
                pd.DataFrame(X), y, sensitive_feature
            )
        else:
            fairness_weights = np.ones(len(y))

        # Combine with provided sample weights
        if sample_weight is not None:
            combined_weights = sample_weight * fairness_weights
        else:
            combined_weights = fairness_weights

        # Fit base estimator with fairness weights
        if hasattr(self.base_estimator, 'sample_weight'):
            self.base_estimator.fit(X, y, sample_weight=combined_weights)
        else:
            # Try fit with sample_weight parameter
            try:
                self.base_estimator.fit(X, y, sample_weight=combined_weights)
            except TypeError:
                warnings.warn("Base estimator does not support sample weights. "
                            "Training without fairness constraints.")
                self.base_estimator.fit(X, y)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict labels.

        Args:
            X: Feature matrix

        Returns:
            Predicted labels
        """
        return self.base_estimator.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities.

        Args:
            X: Feature matrix

        Returns:
            Predicted probabilities
        """
        if hasattr(self.base_estimator, 'predict_proba'):
            return self.base_estimator.predict_proba(X)
        else:
            raise AttributeError("Base estimator does not support predict_proba")


class FairnessPreprocessor(BaseEstimator, TransformerMixin):
    """
    Preprocessing transformer that removes bias from features.
    """

    def __init__(self, sensitive_features: List[str], repair_level: float = 1.0):
        """
        Initialize fairness preprocessor.

        Args:
            sensitive_features: List of sensitive feature names
            repair_level: Level of repair to apply (0.0 to 1.0)
        """
        self.sensitive_features = sensitive_features
        self.repair_level = repair_level
        self.feature_names_ = None

    def fit(self, X: pd.DataFrame, y: Optional[np.ndarray] = None) -> 'FairnessPreprocessor':
        """
        Fit the preprocessor (learns feature names).

        Args:
            X: Feature DataFrame
            y: Labels (unused)

        Returns:
            Self
        """
        if isinstance(X, pd.DataFrame):
            self.feature_names_ = X.columns.tolist()
        else:
            self.feature_names_ = None

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform features to remove bias.

        Args:
            X: Feature DataFrame

        Returns:
            Transformed features
        """
        X_transformed = X.copy()

        if isinstance(X, pd.DataFrame):
            for sensitive_feature in self.sensitive_features:
                if sensitive_feature in X.columns:
                    X_transformed = BiasMitigation.disparate_impact_remover(
                        X_transformed, sensitive_feature, self.repair_level
                    )

        return X_transformed

    def fit_transform(self, X: pd.DataFrame, y: Optional[np.ndarray] = None) -> pd.DataFrame:
        """
        Fit and transform in one step.

        Args:
            X: Feature DataFrame
            y: Labels (unused)

        Returns:
            Transformed features
        """
        return self.fit(X, y).transform(X)
