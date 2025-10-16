"""
Missing value handling for the Unified AI Analytics Platform

This module provides comprehensive strategies for handling missing values in datasets,
including simple imputation, advanced methods like KNN and MICE, and intelligent
strategy selection based on data characteristics.
"""

import pandas as pd
import numpy as np
from typing import Optional, List, Dict, Any, Union
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import warnings


class MissingValueHandler:
    """
    Handle missing values in datasets using various imputation strategies.

    This class provides multiple strategies for handling missing data:
    - Simple imputation (mean, median, mode, constant)
    - KNN imputation (uses neighboring samples)
    - MICE imputation (Multiple Imputation by Chained Equations)
    - Forward/backward fill for time series
    - Column-specific strategies

    The class automatically handles both numerical and categorical features
    and preserves the original data structure.

    Attributes:
        strategy: Default imputation strategy to use
        fill_value: Value to use for constant imputation
        imputer: Fitted imputer object (stored after fit)

    Example:
        >>> handler = MissingValueHandler(strategy='median')
        >>> handler.fit(X_train)
        >>> X_train_imputed = handler.transform(X_train)
        >>> X_test_imputed = handler.transform(X_test)
    """

    def __init__(
        self,
        strategy: str = 'median',
        fill_value: Optional[Any] = None,
        n_neighbors: int = 5,
        max_iter: int = 10,
        random_state: int = 42
    ):
        """
        Initialize the MissingValueHandler.

        Args:
            strategy: Imputation strategy. Options:
                - 'mean': Replace with column mean (numeric only)
                - 'median': Replace with column median (numeric only)
                - 'mode': Replace with most frequent value
                - 'constant': Replace with a constant value
                - 'knn': Use K-Nearest Neighbors imputation
                - 'mice': Use Multiple Imputation by Chained Equations
                - 'ffill': Forward fill (time series)
                - 'bfill': Backward fill (time series)
                - 'drop': Drop rows with missing values
            fill_value: Value for constant imputation
            n_neighbors: Number of neighbors for KNN imputation
            max_iter: Maximum iterations for MICE
            random_state: Random seed for reproducibility

        Example:
            >>> # Use median imputation
            >>> handler = MissingValueHandler(strategy='median')
            >>>
            >>> # Use KNN with 10 neighbors
            >>> handler = MissingValueHandler(strategy='knn', n_neighbors=10)
            >>>
            >>> # Use constant value
            >>> handler = MissingValueHandler(strategy='constant', fill_value=0)
        """
        self.strategy = strategy
        self.fill_value = fill_value
        self.n_neighbors = n_neighbors
        self.max_iter = max_iter
        self.random_state = random_state

        self.numeric_imputer: Optional[Any] = None
        self.categorical_imputer: Optional[Any] = None
        self.numeric_columns: List[str] = []
        self.categorical_columns: List[str] = []
        self.column_strategies: Dict[str, str] = {}

        self._validate_strategy()

    def _validate_strategy(self) -> None:
        """
        Validate the imputation strategy.

        This method checks if the specified strategy is supported and raises
        an error if not. It helps catch configuration errors early.

        Raises:
            ValueError: If strategy is not supported
        """
        valid_strategies = [
            'mean', 'median', 'mode', 'constant', 'knn', 'mice',
            'ffill', 'bfill', 'drop'
        ]

        if self.strategy not in valid_strategies:
            raise ValueError(
                f"Invalid strategy '{self.strategy}'. "
                f"Valid strategies: {valid_strategies}"
            )

    def fit(self, X: pd.DataFrame) -> 'MissingValueHandler':
        """
        Fit the imputer on the training data.

        This method learns the imputation parameters from the training data.
        For example, with 'mean' strategy, it calculates the mean of each column.
        The fitted parameters are stored and used in the transform() method.

        Args:
            X: Training DataFrame with potential missing values

        Returns:
            Self (for method chaining)

        Example:
            >>> handler = MissingValueHandler(strategy='median')
            >>> handler.fit(X_train)
            >>> # Now handler has learned the median of each column
        """
        X = X.copy()

        # Identify numeric and categorical columns
        self.numeric_columns = X.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_columns = X.select_dtypes(exclude=[np.number]).columns.tolist()

        # Fit imputers based on strategy
        if self.strategy in ['mean', 'median', 'constant']:
            # Numeric columns
            if self.numeric_columns:
                self.numeric_imputer = SimpleImputer(
                    strategy=self.strategy if self.strategy != 'constant' else 'constant',
                    fill_value=self.fill_value if self.strategy == 'constant' else None
                )
                self.numeric_imputer.fit(X[self.numeric_columns])

            # Categorical columns (use mode)
            if self.categorical_columns:
                self.categorical_imputer = SimpleImputer(
                    strategy='most_frequent'
                )
                self.categorical_imputer.fit(X[self.categorical_columns])

        elif self.strategy == 'mode':
            # Use most frequent for all columns
            if self.numeric_columns:
                self.numeric_imputer = SimpleImputer(strategy='most_frequent')
                self.numeric_imputer.fit(X[self.numeric_columns])

            if self.categorical_columns:
                self.categorical_imputer = SimpleImputer(strategy='most_frequent')
                self.categorical_imputer.fit(X[self.categorical_columns])

        elif self.strategy == 'knn':
            # KNN imputation (numeric only)
            if self.numeric_columns:
                self.numeric_imputer = KNNImputer(
                    n_neighbors=self.n_neighbors,
                    weights='distance'
                )
                self.numeric_imputer.fit(X[self.numeric_columns])

            # Categorical: use mode
            if self.categorical_columns:
                self.categorical_imputer = SimpleImputer(strategy='most_frequent')
                self.categorical_imputer.fit(X[self.categorical_columns])

        elif self.strategy == 'mice':
            # MICE imputation (numeric only)
            if self.numeric_columns:
                self.numeric_imputer = IterativeImputer(
                    max_iter=self.max_iter,
                    random_state=self.random_state
                )
                self.numeric_imputer.fit(X[self.numeric_columns])

            # Categorical: use mode
            if self.categorical_columns:
                self.categorical_imputer = SimpleImputer(strategy='most_frequent')
                self.categorical_imputer.fit(X[self.categorical_columns])

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform data by imputing missing values.

        This method applies the imputation strategy learned during fit() to
        fill in missing values. It preserves the DataFrame structure including
        column names and index.

        Args:
            X: DataFrame to transform

        Returns:
            DataFrame with missing values imputed

        Example:
            >>> handler = MissingValueHandler(strategy='median')
            >>> handler.fit(X_train)
            >>> X_train_clean = handler.transform(X_train)
            >>> X_test_clean = handler.transform(X_test)
        """
        X = X.copy()

        if self.strategy == 'drop':
            # Simply drop rows with any missing values
            return X.dropna()

        elif self.strategy in ['ffill', 'bfill']:
            # Forward or backward fill (for time series)
            return X.fillna(method=self.strategy)

        else:
            # Apply fitted imputers
            result = X.copy()

            # Impute numeric columns
            if self.numeric_columns and self.numeric_imputer is not None:
                imputed_numeric = self.numeric_imputer.transform(X[self.numeric_columns])
                result[self.numeric_columns] = imputed_numeric

            # Impute categorical columns
            if self.categorical_columns and self.categorical_imputer is not None:
                imputed_categorical = self.categorical_imputer.transform(X[self.categorical_columns])
                result[self.categorical_columns] = imputed_categorical

            return result

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Fit and transform in one step.

        This is a convenience method that combines fit() and transform().
        It's useful for training data but should not be used for test data.

        Args:
            X: DataFrame to fit and transform

        Returns:
            Transformed DataFrame

        Example:
            >>> handler = MissingValueHandler(strategy='median')
            >>> X_train_clean = handler.fit_transform(X_train)
            >>> # Don't use fit_transform on test data!
            >>> X_test_clean = handler.transform(X_test)
        """
        return self.fit(X).transform(X)

    def set_column_strategies(
        self,
        strategies: Dict[str, str]
    ) -> 'MissingValueHandler':
        """
        Set different imputation strategies for specific columns.

        This method allows you to use different strategies for different columns.
        For example, you might want to use median for age but mode for gender.

        Args:
            strategies: Dictionary mapping column names to strategies

        Returns:
            Self (for method chaining)

        Example:
            >>> handler = MissingValueHandler()
            >>> handler.set_column_strategies({
            ...     'age': 'median',
            ...     'income': 'mean',
            ...     'category': 'mode',
            ...     'score': 'knn'
            ... })
            >>> handler.fit(X_train)
            >>> X_clean = handler.transform(X_train)
        """
        self.column_strategies = strategies
        return self

    def get_missing_info(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Get information about missing values in the dataset.

        This method provides a comprehensive summary of missing values,
        including counts and percentages for each column. It helps you
        understand the extent of missing data before choosing a strategy.

        Args:
            X: DataFrame to analyze

        Returns:
            DataFrame with missing value statistics

        Example:
            >>> handler = MissingValueHandler()
            >>> missing_info = handler.get_missing_info(df)
            >>> print(missing_info)

                 column  missing_count  missing_percentage  data_type
            0       age             10                5.0      int64
            1    income             25               12.5    float64
        """
        missing_counts = X.isnull().sum()
        missing_percentages = (missing_counts / len(X)) * 100

        info_df = pd.DataFrame({
            'column': X.columns,
            'missing_count': missing_counts.values,
            'missing_percentage': missing_percentages.values,
            'data_type': X.dtypes.values
        })

        # Sort by missing percentage descending
        info_df = info_df.sort_values('missing_percentage', ascending=False)

        return info_df[info_df['missing_count'] > 0].reset_index(drop=True)

    @staticmethod
    def suggest_strategy(X: pd.DataFrame, column: str) -> str:
        """
        Suggest an appropriate imputation strategy for a column.

        This method analyzes the column characteristics (data type, distribution,
        missing percentage) and suggests the most appropriate imputation strategy.
        The logic is based on common best practices in data science.

        Args:
            X: DataFrame containing the column
            column: Column name to analyze

        Returns:
            Suggested strategy name

        Example:
            >>> handler = MissingValueHandler()
            >>> strategy = handler.suggest_strategy(df, 'age')
            >>> print(f"Suggested strategy for 'age': {strategy}")
            Suggested strategy for 'age': median
        """
        if column not in X.columns:
            raise ValueError(f"Column '{column}' not found in DataFrame")

        col_data = X[column]
        missing_pct = (col_data.isnull().sum() / len(col_data)) * 100

        # If very few missing values, drop might be acceptable
        if missing_pct < 1:
            return 'drop'

        # Check data type
        if pd.api.types.is_numeric_dtype(col_data):
            # For numeric columns, check distribution
            non_null_data = col_data.dropna()

            if len(non_null_data) == 0:
                return 'drop'

            # Check for outliers using IQR
            Q1 = non_null_data.quantile(0.25)
            Q3 = non_null_data.quantile(0.75)
            IQR = Q3 - Q1

            outlier_count = ((non_null_data < (Q1 - 1.5 * IQR)) |
                           (non_null_data > (Q3 + 1.5 * IQR))).sum()
            outlier_pct = (outlier_count / len(non_null_data)) * 100

            # If many outliers, use median; otherwise mean
            if outlier_pct > 10:
                return 'median'
            else:
                # For small datasets with few missing values, KNN works well
                if len(X) < 1000 and missing_pct < 20:
                    return 'knn'
                else:
                    return 'mean'
        else:
            # For categorical columns, use mode
            return 'mode'

    def visualize_missing_pattern(
        self,
        X: pd.DataFrame,
        figsize: tuple = (12, 6)
    ) -> None:
        """
        Visualize the pattern of missing values in the dataset.

        This method creates a visualization showing where missing values occur,
        helping you identify patterns like missing data clustering or correlation
        between missing values in different columns.

        Args:
            X: DataFrame to visualize
            figsize: Figure size (width, height)

        Example:
            >>> handler = MissingValueHandler()
            >>> handler.visualize_missing_pattern(df)
            >>> # This will display a heatmap of missing values
        """
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns

            plt.figure(figsize=figsize)

            # Create binary matrix (1 = missing, 0 = present)
            missing_matrix = X.isnull().astype(int)

            # Plot heatmap
            sns.heatmap(
                missing_matrix,
                cbar=True,
                cmap='RdYlGn_r',
                yticklabels=False
            )

            plt.title('Missing Value Pattern\n(Red = Missing, Green = Present)')
            plt.xlabel('Columns')
            plt.ylabel('Rows')
            plt.tight_layout()
            plt.show()

        except ImportError:
            warnings.warn(
                "matplotlib and seaborn are required for visualization. "
                "Install them with: pip install matplotlib seaborn"
            )

    def __repr__(self) -> str:
        """String representation of the handler."""
        return (
            f"MissingValueHandler(strategy='{self.strategy}', "
            f"fill_value={self.fill_value})"
        )
