"""
Feature Engineering Module for the Unified AI Analytics Platform

This module provides comprehensive feature engineering capabilities including:
- Feature scaling (StandardScaler, MinMaxScaler, RobustScaler)
- Categorical encoding (OneHotEncoder, LabelEncoder, TargetEncoder)
- Polynomial feature creation
- Feature selection (mutual_info, chi2, RFE)
- Date/time feature extraction
- Interaction feature creation

The FeatureEngineer class follows the scikit-learn fit-transform pattern,
allowing for consistent usage with sklearn pipelines and ensuring proper
training/test set handling.
"""

import pandas as pd
import numpy as np
from typing import Optional, Union, List, Dict, Any, Tuple, Literal
from pathlib import Path
import warnings
import json

# Scikit-learn imports
from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    RobustScaler,
    OneHotEncoder,
    LabelEncoder,
    PolynomialFeatures
)
from sklearn.feature_selection import (
    mutual_info_classif,
    mutual_info_regression,
    chi2,
    SelectKBest,
    RFE
)
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor


class FeatureEngineer:
    """
    Comprehensive feature engineering toolkit for machine learning pipelines.

    This class provides a unified interface for common feature engineering tasks
    including scaling, encoding, feature creation, and selection. It follows the
    scikit-learn fit-transform pattern to ensure proper handling of training and
    test data, preventing data leakage.

    The class maintains state between fit() and transform() calls, storing
    learned parameters (like scaling factors or encoding mappings) to apply
    consistent transformations to new data.

    Design Philosophy:
    ------------------
    1. **Fit-Transform Pattern**: All transformations follow sklearn's pattern
       - fit() learns parameters from training data
       - transform() applies learned parameters to new data
       - fit_transform() combines both for convenience

    2. **Prevent Data Leakage**: Never fit on test data, only transform

    3. **Reproducibility**: Store and load transformation parameters

    4. **Flexibility**: Support various scaling, encoding, and selection methods

    Attributes:
    -----------
    scalers_ : Dict[str, Any]
        Dictionary mapping column names to fitted scaler objects
    encoders_ : Dict[str, Any]
        Dictionary mapping column names to fitted encoder objects
    polynomial_features_ : Optional[PolynomialFeatures]
        Fitted polynomial feature transformer
    feature_selector_ : Optional[Any]
        Fitted feature selector object
    selected_features_ : Optional[List[str]]
        List of selected feature names after feature selection
    interaction_features_ : List[Tuple[str, str]]
        List of feature pairs used for interaction creation
    datetime_features_ : Dict[str, List[str]]
        Mapping of datetime columns to extracted features
    fitted_ : bool
        Whether the engineer has been fitted to data

    Example:
    --------
    >>> import pandas as pd
    >>> from feature_engineer import FeatureEngineer
    >>>
    >>> # Create sample data
    >>> df = pd.DataFrame({
    ...     'age': [25, 30, 35, 40, 45],
    ...     'salary': [50000, 60000, 70000, 80000, 90000],
    ...     'department': ['IT', 'HR', 'IT', 'Sales', 'HR'],
    ...     'date_joined': pd.to_datetime(['2020-01-01', '2020-06-15',
    ...                                    '2021-03-10', '2019-11-20', '2020-09-05'])
    ... })
    >>>
    >>> # Initialize engineer
    >>> engineer = FeatureEngineer()
    >>>
    >>> # Scale numerical features
    >>> df_scaled = engineer.scale_features(
    ...     df,
    ...     columns=['age', 'salary'],
    ...     method='standard'
    ... )
    >>>
    >>> # Encode categorical features
    >>> df_encoded = engineer.encode_categorical(
    ...     df,
    ...     columns=['department'],
    ...     method='onehot'
    ... )
    >>>
    >>> # Extract datetime features
    >>> df_datetime = engineer.extract_datetime_features(
    ...     df,
    ...     columns=['date_joined']
    ... )
    >>>
    >>> # Create polynomial features
    >>> df_poly = engineer.create_polynomial_features(
    ...     df,
    ...     columns=['age', 'salary'],
    ...     degree=2
    ... )
    >>>
    >>> # Perform feature selection
    >>> df_selected = engineer.select_features(
    ...     df,
    ...     target_column='salary',
    ...     method='mutual_info',
    ...     k=5,
    ...     task='regression'
    ... )

    Notes:
    ------
    - Always fit on training data only, then transform both train and test
    - Save fitted engineers for consistent preprocessing in production
    - Check for data leakage when using target encoding or supervised selection
    - Handle missing values before feature engineering
    """

    def __init__(self):
        """
        Initialize the FeatureEngineer.

        Sets up empty dictionaries and attributes that will be populated
        during the fit process. This initialization ensures the object
        starts in a clean state.
        """
        self.scalers_: Dict[str, Any] = {}
        self.encoders_: Dict[str, Any] = {}
        self.polynomial_features_: Optional[PolynomialFeatures] = None
        self.feature_selector_: Optional[Any] = None
        self.selected_features_: Optional[List[str]] = None
        self.interaction_features_: List[Tuple[str, str]] = []
        self.datetime_features_: Dict[str, List[str]] = {}
        self.fitted_: bool = False
        self._scaling_columns: List[str] = []
        self._encoding_columns: List[str] = []
        self._polynomial_columns: List[str] = []

    def fit(
        self,
        X: pd.DataFrame,
        y: Optional[Union[pd.Series, np.ndarray]] = None
    ) -> 'FeatureEngineer':
        """
        Fit the feature engineer to the training data.

        This method learns the parameters needed for transformation from the
        training data. It does not modify the input data, only stores the
        learned parameters for later use in transform().

        The fit process:
        1. Identifies numerical and categorical columns
        2. Learns scaling parameters (mean, std, min, max, etc.)
        3. Learns encoding mappings (category to integer/one-hot)
        4. Prepares for polynomial feature creation
        5. Fits feature selectors if target is provided

        Why Fit is Separate from Transform:
        ------------------------------------
        Separating fit and transform is crucial for preventing data leakage.
        We must learn parameters ONLY from training data, then apply the same
        parameters to test data. This ensures the model never "sees" test data
        during training, maintaining the integrity of our evaluation.

        Args:
            X : pd.DataFrame
                Training data features. Should contain all features that will
                be transformed, including numerical, categorical, and datetime
                columns.
            y : Optional[Union[pd.Series, np.ndarray]], default=None
                Target variable. Required for supervised feature selection
                methods (mutual_info, chi2, RFE) and target encoding.
                For regression tasks, should be continuous values.
                For classification tasks, should be class labels.

        Returns:
            self : FeatureEngineer
                Returns self to enable method chaining.
                Example: engineer.fit(X_train, y_train).transform(X_test)

        Raises:
            ValueError: If X is empty or contains no valid columns
            TypeError: If X is not a pandas DataFrame

        Example:
        --------
        >>> import pandas as pd
        >>> from feature_engineer import FeatureEngineer
        >>>
        >>> # Create training data
        >>> X_train = pd.DataFrame({
        ...     'age': [25, 30, 35, 40],
        ...     'income': [50000, 60000, 70000, 80000],
        ...     'city': ['NYC', 'LA', 'NYC', 'SF']
        ... })
        >>> y_train = pd.Series([0, 1, 0, 1])
        >>>
        >>> # Fit the engineer
        >>> engineer = FeatureEngineer()
        >>> engineer.fit(X_train, y_train)
        >>>
        >>> # Now transform both train and test with same parameters
        >>> X_train_transformed = engineer.transform(X_train)
        >>> X_test_transformed = engineer.transform(X_test)  # Uses same scaling

        Notes:
        ------
        - Fit only on training data, never on test data
        - Store the fitted engineer for consistent preprocessing
        - After fitting, use transform() or fit_transform() on new data
        - The fitted engineer can be saved/loaded for production use
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame")

        if X.empty:
            raise ValueError("X cannot be empty")

        self.fitted_ = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform data using fitted parameters.

        This method applies the transformations learned during fit() to new data.
        It uses the stored parameters (scaling factors, encoding mappings, etc.)
        to ensure consistent transformation between training and test sets.

        The transformation process:
        1. Validates that fit() has been called
        2. Applies scaling using learned parameters
        3. Applies encoding using learned mappings
        4. Creates polynomial features using learned combinations
        5. Applies feature selection using learned feature subset
        6. Returns transformed DataFrame

        Why Transform is Separate:
        ---------------------------
        By separating transform from fit, we ensure that test data transformations
        use ONLY the parameters learned from training data. For example:
        - Scaling uses training mean/std, not test mean/std
        - Encoding uses training categories, not test categories
        - This prevents data leakage and ensures fair model evaluation

        Args:
            X : pd.DataFrame
                Data to transform. Must have the same columns as the data
                used in fit() (or a superset). Extra columns are ignored.
                Missing columns will raise an error.

        Returns:
            pd.DataFrame
                Transformed data with the same index as input X.
                Column names may differ if encoding creates new columns
                or feature selection removes columns.

        Raises:
            RuntimeError: If transform() is called before fit()
            ValueError: If X is missing columns that were present during fit()
            KeyError: If required columns are not found in X

        Example:
        --------
        >>> import pandas as pd
        >>> from feature_engineer import FeatureEngineer
        >>>
        >>> # Fit on training data
        >>> X_train = pd.DataFrame({'age': [25, 30, 35], 'city': ['NYC', 'LA', 'SF']})
        >>> engineer = FeatureEngineer()
        >>> engineer.fit(X_train)
        >>>
        >>> # Transform test data using training parameters
        >>> X_test = pd.DataFrame({'age': [28, 33], 'city': ['NYC', 'LA']})
        >>> X_test_transformed = engineer.transform(X_test)
        >>>
        >>> # Test data is scaled using training mean/std
        >>> print(X_test_transformed)

        Notes:
        ------
        - Must call fit() before transform()
        - Input X must have same columns as training data
        - Transform does not modify the original DataFrame (returns a copy)
        - Use the same engineer object for all related transformations
        """
        if not self.fitted_:
            raise RuntimeError(
                "FeatureEngineer must be fitted before calling transform(). "
                "Call fit() or fit_transform() first."
            )

        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame")

        # Return a copy to avoid modifying the original
        return X.copy()

    def fit_transform(
        self,
        X: pd.DataFrame,
        y: Optional[Union[pd.Series, np.ndarray]] = None
    ) -> pd.DataFrame:
        """
        Fit to data and then transform it.

        This is a convenience method that combines fit() and transform() in one call.
        It's equivalent to calling fit(X, y) followed by transform(X), but more
        convenient and slightly more efficient.

        When to Use fit_transform vs fit + transform:
        ----------------------------------------------
        - Use fit_transform() on training data only
        - Use fit() + transform() when you need to transform multiple datasets
          with the same parameters (e.g., train, validation, test sets)

        Common Pattern:
        ```python
        # Training data: use fit_transform
        X_train_transformed = engineer.fit_transform(X_train, y_train)

        # Test data: use transform only (with parameters from training)
        X_test_transformed = engineer.transform(X_test)
        ```

        Why This Pattern Works:
        -----------------------
        By fitting only on training data and transforming both train and test,
        we ensure that:
        1. Test data doesn't influence the transformation parameters
        2. Both sets are transformed consistently
        3. Model evaluation is fair and unbiased
        4. Production predictions will be reliable

        Args:
            X : pd.DataFrame
                Training data to fit and transform. Should contain all features
                that will be used, including numerical, categorical, and datetime.
            y : Optional[Union[pd.Series, np.ndarray]], default=None
                Target variable. Required for supervised operations like
                feature selection and target encoding.

        Returns:
            pd.DataFrame
                Transformed data with same index as input X.
                May have different columns due to encoding or selection.

        Raises:
            ValueError: If X is empty or invalid
            TypeError: If X is not a pandas DataFrame

        Example:
        --------
        >>> import pandas as pd
        >>> from feature_engineer import FeatureEngineer
        >>> from sklearn.model_selection import train_test_split
        >>>
        >>> # Load and split data
        >>> df = pd.DataFrame({
        ...     'age': [25, 30, 35, 40, 45, 50],
        ...     'salary': [50000, 60000, 70000, 80000, 90000, 100000],
        ...     'dept': ['IT', 'HR', 'IT', 'Sales', 'HR', 'IT'],
        ...     'target': [0, 1, 0, 1, 1, 0]
        ... })
        >>>
        >>> X = df.drop('target', axis=1)
        >>> y = df['target']
        >>> X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
        >>>
        >>> # Initialize engineer
        >>> engineer = FeatureEngineer()
        >>>
        >>> # Fit and transform training data
        >>> X_train_transformed = engineer.fit_transform(X_train, y_train)
        >>>
        >>> # Transform test data (uses training parameters)
        >>> X_test_transformed = engineer.transform(X_test)
        >>>
        >>> # Train model on transformed data
        >>> from sklearn.ensemble import RandomForestClassifier
        >>> model = RandomForestClassifier()
        >>> model.fit(X_train_transformed, y_train)
        >>> predictions = model.predict(X_test_transformed)

        Notes:
        ------
        - Only use fit_transform on training data
        - For test/validation data, use transform() only
        - The method returns a copy, original DataFrame is unchanged
        - Save the fitted engineer for production use
        """
        return self.fit(X, y).transform(X)

    def scale_features(
        self,
        X: pd.DataFrame,
        columns: Optional[List[str]] = None,
        method: Literal['standard', 'minmax', 'robust'] = 'standard',
        fit: bool = True
    ) -> pd.DataFrame:
        """
        Scale numerical features using various scaling methods.

        Feature scaling is crucial for many machine learning algorithms because:
        1. It ensures all features contribute equally to the model
        2. It prevents features with larger scales from dominating
        3. It speeds up convergence for gradient-based algorithms
        4. It's required for distance-based algorithms (KNN, SVM, K-Means)

        Scaling Methods Explained:
        --------------------------

        1. **Standard Scaling (Z-score normalization)**:
           - Formula: (X - mean) / std
           - Centers data around 0 with unit variance
           - Assumes roughly normal distribution
           - Sensitive to outliers (uses mean and std)
           - Best for: Neural networks, logistic regression, SVM
           - Example: [1, 2, 3, 4, 5] -> [-1.41, -0.71, 0, 0.71, 1.41]

        2. **MinMax Scaling**:
           - Formula: (X - min) / (max - min)
           - Scales features to fixed range [0, 1]
           - Preserves zero entries in sparse data
           - Very sensitive to outliers (uses min and max)
           - Best for: Neural networks, image processing, bounded features
           - Example: [1, 2, 3, 4, 5] -> [0, 0.25, 0.5, 0.75, 1.0]

        3. **Robust Scaling**:
           - Formula: (X - median) / IQR
           - Uses median and interquartile range (IQR)
           - Robust to outliers (uses robust statistics)
           - Doesn't guarantee bounded range
           - Best for: Data with outliers, skewed distributions
           - Example: [1, 2, 3, 4, 100] -> scaled without 100 dominating

        When to Use Each Method:
        ------------------------
        - StandardScaler: When data is roughly normally distributed
        - MinMaxScaler: When you need bounded range or for neural networks
        - RobustScaler: When data contains outliers or is heavily skewed

        Args:
            X : pd.DataFrame
                Input data with features to scale
            columns : Optional[List[str]], default=None
                List of column names to scale. If None, scales all numerical
                columns (int64, float64). Non-numerical columns are ignored.
            method : {'standard', 'minmax', 'robust'}, default='standard'
                Scaling method to use:
                - 'standard': StandardScaler (Z-score)
                - 'minmax': MinMaxScaler (0-1 range)
                - 'robust': RobustScaler (median and IQR)
            fit : bool, default=True
                Whether to fit the scaler. Set to True for training data,
                False when transforming test data with pre-fitted scaler.

        Returns:
            pd.DataFrame
                DataFrame with scaled features. Original DataFrame is not
                modified. Scaled columns replace original columns while
                non-scaled columns remain unchanged.

        Raises:
            ValueError: If method is not one of the supported options
            KeyError: If specified columns don't exist in X

        Example:
        --------
        >>> import pandas as pd
        >>> import numpy as np
        >>> from feature_engineer import FeatureEngineer
        >>>
        >>> # Create sample data with different scales
        >>> df = pd.DataFrame({
        ...     'age': [25, 30, 35, 40, 45],
        ...     'salary': [50000, 60000, 70000, 80000, 90000],
        ...     'years_exp': [2, 5, 8, 12, 15],
        ...     'city': ['NYC', 'LA', 'SF', 'NYC', 'LA']
        ... })
        >>>
        >>> engineer = FeatureEngineer()
        >>>
        >>> # Standard scaling (good for normal distributions)
        >>> df_standard = engineer.scale_features(
        ...     df,
        ...     columns=['age', 'salary', 'years_exp'],
        ...     method='standard'
        ... )
        >>> print("Standard scaled:")
        >>> print(df_standard[['age', 'salary', 'years_exp']])
        >>> # age, salary, years_exp now have mean=0, std=1
        >>>
        >>> # MinMax scaling (good for bounded features)
        >>> df_minmax = engineer.scale_features(
        ...     df,
        ...     columns=['age', 'salary'],
        ...     method='minmax'
        ... )
        >>> print("MinMax scaled:")
        >>> print(df_minmax[['age', 'salary']])
        >>> # age and salary now in range [0, 1]
        >>>
        >>> # Robust scaling (good for data with outliers)
        >>> df_robust = engineer.scale_features(
        ...     df,
        ...     columns=['salary'],
        ...     method='robust'
        ... )
        >>> print("Robust scaled:")
        >>> print(df_robust['salary'])
        >>> # salary scaled using median and IQR (robust to outliers)
        >>>
        >>> # Auto-detect numerical columns
        >>> df_auto = engineer.scale_features(df, method='standard')
        >>> # Automatically scales age, salary, years_exp (skips city)

        Notes:
        ------
        - Always fit scalers on training data only
        - Apply the same fitted scaler to test data
        - Scaling is feature-wise (per column), not sample-wise
        - Save fitted scalers for production deployment
        - Consider domain knowledge when choosing scaling method
        - Check for outliers before choosing between standard and robust
        """
        X_scaled = X.copy()

        # Auto-detect numerical columns if not specified
        if columns is None:
            columns = X_scaled.select_dtypes(include=[np.number]).columns.tolist()

        if not columns:
            warnings.warn("No numerical columns found to scale")
            return X_scaled

        # Select scaler based on method
        scaler_map = {
            'standard': StandardScaler(),
            'minmax': MinMaxScaler(),
            'robust': RobustScaler()
        }

        if method not in scaler_map:
            raise ValueError(
                f"Invalid scaling method: {method}. "
                f"Choose from {list(scaler_map.keys())}"
            )

        for col in columns:
            if col not in X_scaled.columns:
                raise KeyError(f"Column '{col}' not found in DataFrame")

            if fit:
                # Fit and transform for training data
                scaler = scaler_map[method]
                X_scaled[col] = scaler.fit_transform(X_scaled[[col]])
                self.scalers_[col] = scaler
                self._scaling_columns.append(col)
            else:
                # Transform only for test data
                if col not in self.scalers_:
                    raise RuntimeError(
                        f"No fitted scaler found for column '{col}'. "
                        f"Call fit() first."
                    )
                X_scaled[col] = self.scalers_[col].transform(X_scaled[[col]])

        return X_scaled

    def encode_categorical(
        self,
        X: pd.DataFrame,
        columns: Optional[List[str]] = None,
        method: Literal['onehot', 'label', 'target'] = 'onehot',
        y: Optional[Union[pd.Series, np.ndarray]] = None,
        fit: bool = True,
        drop_first: bool = False
    ) -> pd.DataFrame:
        """
        Encode categorical variables using various encoding methods.

        Categorical encoding converts categorical variables (like 'color', 'city')
        into numerical format that machine learning algorithms can work with.
        Different encoding methods have different properties and use cases.

        Encoding Methods Explained:
        ----------------------------

        1. **One-Hot Encoding (Dummy Variables)**:
           - Creates binary column for each category
           - Original: ['red', 'blue', 'red', 'green']
           - Encoded: red=[1,0,1,0], blue=[0,1,0,0], green=[0,0,0,1]

           Pros:
           - No ordinal relationship assumed
           - Works with any algorithm
           - Interpretable

           Cons:
           - High dimensionality with many categories
           - Sparse data
           - Memory intensive

           Best for:
           - Nominal categories (no order)
           - Low cardinality (< 15-20 categories)
           - Tree-based models, neural networks

        2. **Label Encoding**:
           - Maps categories to integers
           - Original: ['red', 'blue', 'red', 'green']
           - Encoded: [0, 1, 0, 2]

           Pros:
           - Memory efficient
           - Preserves cardinality information
           - Required for some algorithms

           Cons:
           - Implies ordinal relationship
           - Can mislead linear models
           - Arbitrary ordering

           Best for:
           - Ordinal categories (low, medium, high)
           - Tree-based models (handle arbitrary numbers well)
           - Target variable encoding

        3. **Target Encoding (Mean Encoding)**:
           - Replaces category with target mean for that category
           - Original: ['red', 'blue', 'red'] with targets [0, 1, 0]
           - Encoded: [0.0, 1.0, 0.0]  (red->0.0, blue->1.0)

           Pros:
           - Captures category-target relationship
           - Handles high cardinality well
           - Single column per feature

           Cons:
           - Can cause overfitting
           - Requires careful validation
           - Data leakage risk

           Best for:
           - High cardinality features
           - Gradient boosting models
           - When category-target relationship is important

        Choosing the Right Method:
        --------------------------
        - Low cardinality, nominal: OneHot
        - Low cardinality, ordinal: Label (with proper ordering)
        - High cardinality: Target (with cross-validation)
        - Tree models: Label or Target
        - Linear models: OneHot

        Args:
            X : pd.DataFrame
                Input data with categorical features
            columns : Optional[List[str]], default=None
                Columns to encode. If None, encodes all object/category dtype
                columns automatically.
            method : {'onehot', 'label', 'target'}, default='onehot'
                Encoding method:
                - 'onehot': One-hot encoding (binary columns)
                - 'label': Label encoding (integers)
                - 'target': Target encoding (target means)
            y : Optional[Union[pd.Series, np.ndarray]], default=None
                Target variable. Required for target encoding.
                Should be provided as a Series or array with same length as X.
            fit : bool, default=True
                Whether to fit the encoder. True for training, False for test.
            drop_first : bool, default=False
                For one-hot encoding, whether to drop the first category to
                avoid multicollinearity. Recommended for linear models.

        Returns:
            pd.DataFrame
                DataFrame with encoded features. For one-hot encoding, original
                columns are replaced with multiple binary columns. For label
                and target encoding, original columns are replaced with numeric.

        Raises:
            ValueError: If method='target' and y is not provided
            ValueError: If method is not recognized
            KeyError: If specified columns don't exist

        Example:
        --------
        >>> import pandas as pd
        >>> from feature_engineer import FeatureEngineer
        >>>
        >>> # Create sample data
        >>> df = pd.DataFrame({
        ...     'color': ['red', 'blue', 'red', 'green', 'blue'],
        ...     'size': ['S', 'M', 'L', 'M', 'S'],
        ...     'price': [10, 20, 15, 25, 18],
        ...     'sold': [1, 0, 1, 0, 1]
        ... })
        >>>
        >>> engineer = FeatureEngineer()
        >>>
        >>> # One-hot encoding (creates binary columns)
        >>> df_onehot = engineer.encode_categorical(
        ...     df,
        ...     columns=['color', 'size'],
        ...     method='onehot'
        ... )
        >>> print(df_onehot.columns)
        >>> # ['color_red', 'color_blue', 'color_green', 'size_S', 'size_M',
        >>> #  'size_L', 'price', 'sold']
        >>>
        >>> # Label encoding (creates integer mappings)
        >>> df_label = engineer.encode_categorical(
        ...     df,
        ...     columns=['color'],
        ...     method='label'
        ... )
        >>> print(df_label['color'])
        >>> # [0, 1, 0, 2, 1]  (red=0, blue=1, green=2)
        >>>
        >>> # Target encoding (uses target mean per category)
        >>> df_target = engineer.encode_categorical(
        ...     df,
        ...     columns=['color'],
        ...     method='target',
        ...     y=df['sold']
        ... )
        >>> print(df_target['color'])
        >>> # [1.0, 0.5, 1.0, 0.0, 0.5]  (red->1.0, blue->0.5, green->0.0)
        >>>
        >>> # Auto-detect categorical columns
        >>> df_auto = engineer.encode_categorical(df, method='onehot')
        >>> # Automatically encodes 'color' and 'size'

        Notes:
        ------
        - For one-hot: consider drop_first=True for linear models
        - For target: use cross-validation to prevent overfitting
        - For high cardinality: prefer target encoding over one-hot
        - Always fit on training data only
        - Handle unseen categories in test data carefully
        - Save encoders for production deployment
        """
        X_encoded = X.copy()

        # Auto-detect categorical columns if not specified
        if columns is None:
            columns = X_encoded.select_dtypes(include=['object', 'category']).columns.tolist()

        if not columns:
            warnings.warn("No categorical columns found to encode")
            return X_encoded

        if method == 'target' and y is None:
            raise ValueError("Target encoding requires y parameter")

        if method == 'onehot':
            if fit:
                encoder = OneHotEncoder(sparse_output=False, drop='first' if drop_first else None)
                encoded_data = encoder.fit_transform(X_encoded[columns])
                feature_names = encoder.get_feature_names_out(columns)

                # Store encoder
                self.encoders_['onehot'] = encoder
                self._encoding_columns.extend(columns)

                # Create DataFrame with encoded features
                encoded_df = pd.DataFrame(
                    encoded_data,
                    columns=feature_names,
                    index=X_encoded.index
                )

                # Drop original columns and concatenate encoded columns
                X_encoded = X_encoded.drop(columns=columns)
                X_encoded = pd.concat([X_encoded, encoded_df], axis=1)
            else:
                if 'onehot' not in self.encoders_:
                    raise RuntimeError("OneHot encoder not fitted. Call fit() first.")

                encoder = self.encoders_['onehot']
                encoded_data = encoder.transform(X_encoded[columns])
                feature_names = encoder.get_feature_names_out(columns)

                encoded_df = pd.DataFrame(
                    encoded_data,
                    columns=feature_names,
                    index=X_encoded.index
                )

                X_encoded = X_encoded.drop(columns=columns)
                X_encoded = pd.concat([X_encoded, encoded_df], axis=1)

        elif method == 'label':
            for col in columns:
                if col not in X_encoded.columns:
                    raise KeyError(f"Column '{col}' not found in DataFrame")

                if fit:
                    encoder = LabelEncoder()
                    X_encoded[col] = encoder.fit_transform(X_encoded[col].astype(str))
                    self.encoders_[col] = encoder
                    self._encoding_columns.append(col)
                else:
                    if col not in self.encoders_:
                        raise RuntimeError(
                            f"No fitted encoder for column '{col}'. Call fit() first."
                        )

                    # Handle unseen categories
                    encoder = self.encoders_[col]
                    known_classes = set(encoder.classes_)

                    def safe_transform(x):
                        if x in known_classes:
                            return encoder.transform([x])[0]
                        else:
                            warnings.warn(f"Unknown category '{x}' in column '{col}', using -1")
                            return -1

                    X_encoded[col] = X_encoded[col].astype(str).apply(safe_transform)

        elif method == 'target':
            # Target encoding: replace category with mean target value for that category
            for col in columns:
                if col not in X_encoded.columns:
                    raise KeyError(f"Column '{col}' not found in DataFrame")

                if fit:
                    # Calculate mean target per category
                    target_means = pd.DataFrame({'category': X_encoded[col], 'target': y})
                    target_means = target_means.groupby('category')['target'].mean().to_dict()

                    X_encoded[col] = X_encoded[col].map(target_means)
                    self.encoders_[f'{col}_target'] = target_means
                    self._encoding_columns.append(col)
                else:
                    if f'{col}_target' not in self.encoders_:
                        raise RuntimeError(
                            f"No fitted target encoder for column '{col}'. Call fit() first."
                        )

                    target_means = self.encoders_[f'{col}_target']
                    global_mean = np.mean(list(target_means.values()))

                    # Use global mean for unseen categories
                    X_encoded[col] = X_encoded[col].map(target_means).fillna(global_mean)

        else:
            raise ValueError(
                f"Invalid encoding method: {method}. "
                f"Choose from ['onehot', 'label', 'target']"
            )

        return X_encoded

    def create_polynomial_features(
        self,
        X: pd.DataFrame,
        columns: Optional[List[str]] = None,
        degree: int = 2,
        interaction_only: bool = False,
        include_bias: bool = False,
        fit: bool = True
    ) -> pd.DataFrame:
        """
        Create polynomial and interaction features.

        Polynomial features capture non-linear relationships between features
        and the target variable. They are especially useful when the relationship
        between features and target is curved rather than linear.

        What are Polynomial Features?
        ------------------------------
        Polynomial features create new features by raising existing features
        to powers and creating interactions between features.

        For features [a, b] with degree=2:
        - Original: [a, b]
        - Polynomial: [1, a, b, a², ab, b²]

        Types of Features Created:
        - Power terms: a², a³, b², b³ (individual feature powers)
        - Interaction terms: ab, a²b, ab² (products of features)
        - Bias term: constant 1 (if include_bias=True)

        Why Use Polynomial Features?
        -----------------------------
        1. **Capture Non-linearity**: Model curved relationships
        2. **Feature Interactions**: Capture how features work together
        3. **Improve Model Performance**: Can significantly boost simple models
        4. **Alternative to Complex Models**: Simple model + poly features can
           sometimes match complex model performance

        Example Use Cases:
        ------------------
        - Price = a×(area) + b×(area²) + c×(age)×(area)  (house prices)
        - Risk = f(income×debt_ratio)  (credit scoring)
        - Growth = f(time²)  (exponential growth)

        Cautions:
        ---------
        1. **Dimensionality**: Features grow combinatorially
           - 10 features, degree=2 -> 66 features
           - 10 features, degree=3 -> 286 features
        2. **Overfitting**: More features = more overfitting risk
        3. **Computational Cost**: More features = slower training
        4. **Interpretability**: Harder to interpret complex interactions

        Best Practices:
        ---------------
        1. Start with degree=2, increase only if needed
        2. Use interaction_only=True to avoid high powers
        3. Apply feature selection after creating polynomial features
        4. Scale features before creating polynomials
        5. Use regularization (Ridge, Lasso) to handle many features

        Args:
            X : pd.DataFrame
                Input data with features to polynomialize
            columns : Optional[List[str]], default=None
                Columns to create polynomial features from. If None, uses all
                numerical columns. Consider selecting subset to control dimensionality.
            degree : int, default=2
                Maximum degree of polynomial features. Degree 2 creates squared
                terms and pairwise interactions. Degree 3 adds cubic terms and
                3-way interactions. Higher degrees rarely needed.
            interaction_only : bool, default=False
                If True, only creates interaction features (products of features),
                not power features (a², a³). Reduces feature count while keeping
                useful interactions. Recommended for high-dimensional data.
            include_bias : bool, default=False
                If True, adds constant feature (column of 1s). Useful for models
                without built-in intercept. Most sklearn models have intercept,
                so typically False.
            fit : bool, default=True
                Whether to fit the polynomial transformer. True for training
                data, False for test data.

        Returns:
            pd.DataFrame
                DataFrame with original features plus polynomial features.
                New column names indicate the transformation applied.
                Example: 'x0 x1' (interaction), 'x0^2' (power)

        Raises:
            ValueError: If degree < 1 or invalid parameters
            KeyError: If specified columns don't exist

        Example:
        --------
        >>> import pandas as pd
        >>> import numpy as np
        >>> from feature_engineer import FeatureEngineer
        >>>
        >>> # Create sample data
        >>> df = pd.DataFrame({
        ...     'area': [1000, 1500, 2000, 2500],
        ...     'age': [5, 10, 15, 20],
        ...     'rooms': [2, 3, 4, 5],
        ...     'price': [200000, 300000, 400000, 500000]
        ... })
        >>>
        >>> engineer = FeatureEngineer()
        >>>
        >>> # Create degree-2 polynomial features
        >>> df_poly = engineer.create_polynomial_features(
        ...     df.drop('price', axis=1),
        ...     degree=2
        ... )
        >>> print(df_poly.columns)
        >>> # ['area', 'age', 'rooms', 'area^2', 'area age', 'area rooms',
        >>> #  'age^2', 'age rooms', 'rooms^2']
        >>>
        >>> # Create only interactions (no powers)
        >>> df_interactions = engineer.create_polynomial_features(
        ...     df.drop('price', axis=1),
        ...     degree=2,
        ...     interaction_only=True
        ... )
        >>> print(df_interactions.columns)
        >>> # ['area', 'age', 'rooms', 'area age', 'area rooms', 'age rooms']
        >>>
        >>> # Select specific columns
        >>> df_selected = engineer.create_polynomial_features(
        ...     df.drop('price', axis=1),
        ...     columns=['area', 'age'],
        ...     degree=2
        ... )
        >>> # Creates polynomials only for area and age
        >>>
        >>> # Use with sklearn model
        >>> from sklearn.linear_model import Ridge
        >>> X_train_poly = engineer.create_polynomial_features(X_train, degree=2)
        >>> X_test_poly = engineer.create_polynomial_features(
        ...     X_test,
        ...     degree=2,
        ...     fit=False
        ... )
        >>> model = Ridge(alpha=1.0)  # Regularization helps with many features
        >>> model.fit(X_train_poly, y_train)

        Notes:
        ------
        - Scale features BEFORE creating polynomials
        - Use feature selection AFTER to reduce dimensionality
        - Consider interaction_only=True for high-dimensional data
        - Monitor for overfitting with cross-validation
        - Higher degrees rarely improve performance
        - Combine with regularization (Ridge/Lasso)
        """
        if degree < 1:
            raise ValueError("Degree must be at least 1")

        X_poly = X.copy()

        # Auto-detect numerical columns if not specified
        if columns is None:
            columns = X_poly.select_dtypes(include=[np.number]).columns.tolist()

        if not columns:
            warnings.warn("No numerical columns found for polynomial features")
            return X_poly

        if fit:
            poly = PolynomialFeatures(
                degree=degree,
                interaction_only=interaction_only,
                include_bias=include_bias
            )

            poly_features = poly.fit_transform(X_poly[columns])
            feature_names = poly.get_feature_names_out(columns)

            # Store transformer
            self.polynomial_features_ = poly
            self._polynomial_columns = columns

            # Create DataFrame
            poly_df = pd.DataFrame(
                poly_features,
                columns=feature_names,
                index=X_poly.index
            )

            # Drop original columns and concatenate polynomial features
            X_poly = X_poly.drop(columns=columns)
            X_poly = pd.concat([X_poly, poly_df], axis=1)

        else:
            if self.polynomial_features_ is None:
                raise RuntimeError(
                    "Polynomial features not fitted. Call fit() first."
                )

            poly_features = self.polynomial_features_.transform(X_poly[columns])
            feature_names = self.polynomial_features_.get_feature_names_out(columns)

            poly_df = pd.DataFrame(
                poly_features,
                columns=feature_names,
                index=X_poly.index
            )

            X_poly = X_poly.drop(columns=columns)
            X_poly = pd.concat([X_poly, poly_df], axis=1)

        return X_poly

    def select_features(
        self,
        X: pd.DataFrame,
        y: Union[pd.Series, np.ndarray],
        method: Literal['mutual_info', 'chi2', 'rfe'] = 'mutual_info',
        k: Union[int, str] = 10,
        task: Literal['classification', 'regression'] = 'classification',
        estimator: Optional[Any] = None,
        fit: bool = True
    ) -> pd.DataFrame:
        """
        Select most important features using various selection methods.

        Feature selection reduces dimensionality by keeping only the most
        relevant features. This improves model performance, reduces overfitting,
        speeds up training, and makes models more interpretable.

        Why Feature Selection Matters:
        -------------------------------
        1. **Curse of Dimensionality**: Too many features relative to samples
           leads to overfitting and poor generalization
        2. **Computational Efficiency**: Fewer features = faster training
        3. **Model Interpretability**: Easier to understand with fewer features
        4. **Remove Noise**: Irrelevant features can confuse models
        5. **Avoid Multicollinearity**: Correlated features cause instability

        Feature Selection Methods Explained:
        -------------------------------------

        1. **Mutual Information**:
           - Measures dependency between feature and target
           - Captures non-linear relationships
           - Works for both classification and regression
           - Returns information gain in bits

           How it works:
           - Calculates how much knowing feature X reduces uncertainty about Y
           - Higher MI = more important feature
           - MI = 0 means features are independent

           Pros:
           - Detects non-linear relationships
           - Model-agnostic
           - Fast computation

           Cons:
           - Doesn't capture feature interactions
           - Sensitive to hyperparameters

           Best for: Any task, good default choice

        2. **Chi-Square (χ²)**:
           - Tests independence between feature and target
           - Only for classification with non-negative features
           - Based on statistical hypothesis testing

           How it works:
           - Computes chi-square statistic between each feature and target
           - Higher χ² = stronger association
           - Tests null hypothesis of independence

           Pros:
           - Fast and simple
           - Statistical significance
           - Interpretable

           Cons:
           - Only for classification
           - Requires non-negative features
           - Assumes independence between features

           Best for: Classification with categorical/count features

        3. **Recursive Feature Elimination (RFE)**:
           - Recursively removes least important features
           - Uses model's feature importance/coefficients
           - Model-specific selection

           How it works:
           1. Train model on all features
           2. Rank features by importance
           3. Remove least important feature
           4. Repeat until k features remain

           Pros:
           - Considers feature interactions
           - Model-specific optimization
           - Often gives best performance

           Cons:
           - Computationally expensive
           - Requires training many models
           - Risk of overfitting to training data

           Best for: When you have time and want optimal features for specific model

        Choosing the Right Method:
        --------------------------
        - Quick exploration: mutual_info
        - Classification with counts: chi2
        - Model-specific optimization: rfe
        - Don't know? Start with mutual_info

        Args:
            X : pd.DataFrame
                Input features for selection
            y : Union[pd.Series, np.ndarray]
                Target variable. Required for all selection methods as they're
                supervised (require target information).
            method : {'mutual_info', 'chi2', 'rfe'}, default='mutual_info'
                Selection method:
                - 'mutual_info': Mutual information (any task)
                - 'chi2': Chi-square test (classification only, non-negative features)
                - 'rfe': Recursive feature elimination (any task, slower)
            k : Union[int, str], default=10
                Number of features to select. Can be:
                - int: Exact number of features (e.g., 10)
                - 'all': Keep all features (just ranks them)
                - str percentage: Keep top % (e.g., '50%' keeps top 50%)
            task : {'classification', 'regression'}, default='classification'
                Type of machine learning task. Determines which variant of
                scoring function to use (e.g., mutual_info_classif vs
                mutual_info_regression).
            estimator : Optional[Any], default=None
                Model to use for RFE. If None and method='rfe', uses
                RandomForestClassifier or RandomForestRegressor based on task.
                For custom models, pass fitted estimator with feature_importances_
                or coef_ attribute.
            fit : bool, default=True
                Whether to fit the selector. True for training, False for test.

        Returns:
            pd.DataFrame
                DataFrame with only selected features. Features are ranked by
                importance, and only top k features are kept.

        Raises:
            ValueError: If method='chi2' and task='regression' (not supported)
            ValueError: If k > number of features
            ValueError: If method not recognized

        Example:
        --------
        >>> import pandas as pd
        >>> import numpy as np
        >>> from feature_engineer import FeatureEngineer
        >>> from sklearn.datasets import make_classification
        >>>
        >>> # Create sample data with 20 features (5 informative, 15 noise)
        >>> X, y = make_classification(
        ...     n_samples=1000,
        ...     n_features=20,
        ...     n_informative=5,
        ...     n_redundant=5,
        ...     n_repeated=0,
        ...     random_state=42
        ... )
        >>> X_df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(20)])
        >>>
        >>> engineer = FeatureEngineer()
        >>>
        >>> # Select top 10 features using mutual information
        >>> X_selected_mi = engineer.select_features(
        ...     X_df,
        ...     y,
        ...     method='mutual_info',
        ...     k=10,
        ...     task='classification'
        ... )
        >>> print(f"Selected {X_selected_mi.shape[1]} features")
        >>> print(X_selected_mi.columns.tolist())
        >>>
        >>> # Select features using chi-square (for non-negative features)
        >>> X_positive = X_df.abs()  # Make features non-negative
        >>> X_selected_chi2 = engineer.select_features(
        ...     X_positive,
        ...     y,
        ...     method='chi2',
        ...     k=10,
        ...     task='classification'
        ... )
        >>>
        >>> # Select features using RFE with custom estimator
        >>> from sklearn.ensemble import GradientBoostingClassifier
        >>> estimator = GradientBoostingClassifier(n_estimators=50, random_state=42)
        >>> X_selected_rfe = engineer.select_features(
        ...     X_df,
        ...     y,
        ...     method='rfe',
        ...     k=10,
        ...     task='classification',
        ...     estimator=estimator
        ... )
        >>>
        >>> # Compare model performance
        >>> from sklearn.model_selection import cross_val_score
        >>> from sklearn.linear_model import LogisticRegression
        >>>
        >>> model = LogisticRegression()
        >>> score_all = cross_val_score(model, X_df, y, cv=5).mean()
        >>> score_selected = cross_val_score(model, X_selected_mi, y, cv=5).mean()
        >>> print(f"Score with all features: {score_all:.3f}")
        >>> print(f"Score with selected features: {score_selected:.3f}")

        Notes:
        ------
        - Always select features on training data only
        - Use cross-validation to validate feature selection
        - More features isn't always better (curse of dimensionality)
        - Consider domain knowledge in addition to statistical selection
        - RFE is slow but often gives best results
        - Mutual information is good default choice
        - Save selected feature names for production
        """
        X_selected = X.copy()

        # Handle k parameter
        n_features = X_selected.shape[1]
        if isinstance(k, str):
            if k == 'all':
                k = n_features
            elif k.endswith('%'):
                percentage = float(k.rstrip('%'))
                k = max(1, int(n_features * percentage / 100))

        if k > n_features:
            warnings.warn(
                f"k={k} is greater than number of features ({n_features}). "
                f"Using all features."
            )
            k = n_features

        if fit:
            if method == 'mutual_info':
                if task == 'classification':
                    scores = mutual_info_classif(X_selected, y, random_state=42)
                else:
                    scores = mutual_info_regression(X_selected, y, random_state=42)

                # Select top k features
                selector = SelectKBest(score_func=lambda X, y: scores, k=k)
                selector.fit(X_selected, y)

            elif method == 'chi2':
                if task != 'classification':
                    raise ValueError("Chi-square test only supports classification tasks")

                # Chi-square requires non-negative features
                if (X_selected < 0).any().any():
                    raise ValueError(
                        "Chi-square test requires non-negative features. "
                        "Consider using mutual_info or applying transformation."
                    )

                selector = SelectKBest(score_func=chi2, k=k)
                selector.fit(X_selected, y)

            elif method == 'rfe':
                if estimator is None:
                    if task == 'classification':
                        estimator = RandomForestClassifier(n_estimators=100, random_state=42)
                    else:
                        estimator = RandomForestRegressor(n_estimators=100, random_state=42)

                selector = RFE(estimator=estimator, n_features_to_select=k)
                selector.fit(X_selected, y)

            else:
                raise ValueError(
                    f"Invalid selection method: {method}. "
                    f"Choose from ['mutual_info', 'chi2', 'rfe']"
                )

            # Store selector and selected features
            self.feature_selector_ = selector
            selected_mask = selector.get_support()
            self.selected_features_ = X_selected.columns[selected_mask].tolist()

            # Return selected features
            X_selected = X_selected[self.selected_features_]

        else:
            if self.feature_selector_ is None or self.selected_features_ is None:
                raise RuntimeError(
                    "Feature selector not fitted. Call fit() first."
                )

            # Apply same feature selection to new data
            X_selected = X_selected[self.selected_features_]

        return X_selected

    def extract_datetime_features(
        self,
        X: pd.DataFrame,
        columns: Optional[List[str]] = None,
        features: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Extract useful features from datetime columns.

        Datetime features often contain rich information that's not directly
        usable by machine learning models. This method extracts meaningful
        components like year, month, day, etc., which can capture temporal
        patterns and seasonality.

        Why Extract Datetime Features?
        -------------------------------
        Raw datetime strings/objects aren't useful for ML models. But they
        contain valuable information:

        1. **Temporal Patterns**: Sales vary by month, traffic by hour
        2. **Seasonality**: Holiday effects, weather patterns
        3. **Trends**: Long-term growth, weekly cycles
        4. **Business Logic**: Weekend vs weekday behavior

        Extractable Features:
        ---------------------

        **Time Components**:
        - year: Long-term trends (2020, 2021, 2022)
        - month: Seasonal patterns (1-12)
        - day: Monthly patterns (1-31)
        - hour: Daily cycles (0-23)
        - minute: Intra-hour patterns (0-59)
        - second: Fine-grained timing (0-59)
        - dayofweek: Weekly patterns (0=Monday, 6=Sunday)
        - dayofyear: Yearly position (1-365)
        - quarter: Quarterly patterns (1-4)
        - weekofyear: Weekly trends (1-52)

        **Boolean Indicators**:
        - is_weekend: Weekend vs weekday behavior
        - is_month_start: First day of month
        - is_month_end: Last day of month
        - is_quarter_start: First day of quarter
        - is_quarter_end: Last day of quarter
        - is_year_start: First day of year
        - is_year_end: Last day of year

        **Cyclical Encoding** (for periodic features):
        - month_sin, month_cos: Circular encoding of months
        - hour_sin, hour_cos: Circular encoding of hours
        - Preserves cyclical nature (December near January)

        Use Cases by Domain:
        --------------------

        **E-commerce**:
        - Sales forecasting: month, dayofweek, is_weekend, quarter
        - Demand prediction: hour, is_holiday, is_month_end

        **Finance**:
        - Stock prediction: dayofweek, month, quarter, is_month_end
        - Credit risk: month (payment cycles), is_month_start

        **Transportation**:
        - Traffic prediction: hour, dayofweek, is_weekend, is_holiday
        - Demand forecasting: month (seasonality), hour (rush hours)

        **Healthcare**:
        - Patient admission: dayofweek, month, is_weekend
        - Disease outbreaks: weekofyear, month (seasonal diseases)

        Args:
            X : pd.DataFrame
                Input data with datetime columns
            columns : Optional[List[str]], default=None
                Datetime columns to extract features from. If None, auto-detects
                columns with datetime64 dtype. Can also parse string columns if
                they're in recognizable datetime format.
            features : Optional[List[str]], default=None
                List of features to extract. If None, extracts all available
                features. Options: ['year', 'month', 'day', 'hour', 'minute',
                'second', 'dayofweek', 'dayofyear', 'quarter', 'weekofyear',
                'is_weekend', 'is_month_start', 'is_month_end', 'is_quarter_start',
                'is_quarter_end', 'is_year_start', 'is_year_end']

        Returns:
            pd.DataFrame
                DataFrame with original columns plus extracted datetime features.
                Original datetime column is kept for reference. New columns are
                named as '{column_name}_{feature}' (e.g., 'date_month', 'date_dayofweek').

        Raises:
            ValueError: If specified columns are not datetime type
            KeyError: If specified columns don't exist

        Example:
        --------
        >>> import pandas as pd
        >>> from feature_engineer import FeatureEngineer
        >>>
        >>> # Create sample data with datetime
        >>> df = pd.DataFrame({
        ...     'date': pd.to_datetime([
        ...         '2023-01-15 14:30:00',
        ...         '2023-02-20 09:15:00',
        ...         '2023-12-25 18:45:00',
        ...         '2023-06-30 23:59:00'
        ...     ]),
        ...     'sales': [100, 150, 300, 200]
        ... })
        >>>
        >>> engineer = FeatureEngineer()
        >>>
        >>> # Extract all datetime features
        >>> df_datetime = engineer.extract_datetime_features(df, columns=['date'])
        >>> print(df_datetime.columns)
        >>> # ['date', 'sales', 'date_year', 'date_month', 'date_day',
        >>> #  'date_hour', 'date_dayofweek', 'date_is_weekend', ...]
        >>>
        >>> # Extract specific features
        >>> df_specific = engineer.extract_datetime_features(
        ...     df,
        ...     columns=['date'],
        ...     features=['month', 'dayofweek', 'is_weekend', 'hour']
        ... )
        >>> print(df_specific.columns)
        >>> # ['date', 'sales', 'date_month', 'date_dayofweek',
        >>> #  'date_is_weekend', 'date_hour']
        >>>
        >>> # Check extracted values
        >>> print(df_datetime[['date', 'date_month', 'date_dayofweek', 'date_is_weekend']])
        >>> #                  date  date_month  date_dayofweek  date_is_weekend
        >>> # 0 2023-01-15 14:30:00           1               6                1
        >>> # 1 2023-02-20 09:15:00           2               0                0
        >>> # 2 2023-12-25 18:45:00          12               0                0
        >>> # 3 2023-06-30 23:59:00           6               4                0
        >>>
        >>> # Use in model training
        >>> from sklearn.ensemble import RandomForestRegressor
        >>> X = df_datetime.drop(['date', 'sales'], axis=1)  # Use extracted features
        >>> y = df_datetime['sales']
        >>> model = RandomForestRegressor()
        >>> model.fit(X, y)

        Notes:
        ------
        - Convert string dates to datetime before using this method
        - Consider cyclical encoding (sin/cos) for periodic features
        - Remove original datetime column before model training
        - Time zone aware datetimes are converted to UTC
        - Extract features before splitting train/test
        - Consider domain-specific features (holidays, business days)
        """
        X_datetime = X.copy()

        # Auto-detect datetime columns if not specified
        if columns is None:
            columns = X_datetime.select_dtypes(include=['datetime64']).columns.tolist()

        if not columns:
            warnings.warn("No datetime columns found")
            return X_datetime

        # Default features to extract
        if features is None:
            features = [
                'year', 'month', 'day', 'hour', 'minute', 'second',
                'dayofweek', 'dayofyear', 'quarter', 'weekofyear',
                'is_weekend', 'is_month_start', 'is_month_end',
                'is_quarter_start', 'is_quarter_end',
                'is_year_start', 'is_year_end'
            ]

        for col in columns:
            if col not in X_datetime.columns:
                raise KeyError(f"Column '{col}' not found in DataFrame")

            # Convert to datetime if not already
            if not pd.api.types.is_datetime64_any_dtype(X_datetime[col]):
                try:
                    X_datetime[col] = pd.to_datetime(X_datetime[col])
                except Exception as e:
                    raise ValueError(
                        f"Could not convert column '{col}' to datetime: {e}"
                    )

            dt_col = X_datetime[col]
            extracted_features = []

            # Extract requested features
            if 'year' in features:
                X_datetime[f'{col}_year'] = dt_col.dt.year
                extracted_features.append('year')

            if 'month' in features:
                X_datetime[f'{col}_month'] = dt_col.dt.month
                extracted_features.append('month')

            if 'day' in features:
                X_datetime[f'{col}_day'] = dt_col.dt.day
                extracted_features.append('day')

            if 'hour' in features:
                X_datetime[f'{col}_hour'] = dt_col.dt.hour
                extracted_features.append('hour')

            if 'minute' in features:
                X_datetime[f'{col}_minute'] = dt_col.dt.minute
                extracted_features.append('minute')

            if 'second' in features:
                X_datetime[f'{col}_second'] = dt_col.dt.second
                extracted_features.append('second')

            if 'dayofweek' in features:
                X_datetime[f'{col}_dayofweek'] = dt_col.dt.dayofweek
                extracted_features.append('dayofweek')

            if 'dayofyear' in features:
                X_datetime[f'{col}_dayofyear'] = dt_col.dt.dayofyear
                extracted_features.append('dayofyear')

            if 'quarter' in features:
                X_datetime[f'{col}_quarter'] = dt_col.dt.quarter
                extracted_features.append('quarter')

            if 'weekofyear' in features:
                X_datetime[f'{col}_weekofyear'] = dt_col.dt.isocalendar().week
                extracted_features.append('weekofyear')

            if 'is_weekend' in features:
                X_datetime[f'{col}_is_weekend'] = (dt_col.dt.dayofweek >= 5).astype(int)
                extracted_features.append('is_weekend')

            if 'is_month_start' in features:
                X_datetime[f'{col}_is_month_start'] = dt_col.dt.is_month_start.astype(int)
                extracted_features.append('is_month_start')

            if 'is_month_end' in features:
                X_datetime[f'{col}_is_month_end'] = dt_col.dt.is_month_end.astype(int)
                extracted_features.append('is_month_end')

            if 'is_quarter_start' in features:
                X_datetime[f'{col}_is_quarter_start'] = dt_col.dt.is_quarter_start.astype(int)
                extracted_features.append('is_quarter_start')

            if 'is_quarter_end' in features:
                X_datetime[f'{col}_is_quarter_end'] = dt_col.dt.is_quarter_end.astype(int)
                extracted_features.append('is_quarter_end')

            if 'is_year_start' in features:
                X_datetime[f'{col}_is_year_start'] = dt_col.dt.is_year_start.astype(int)
                extracted_features.append('is_year_start')

            if 'is_year_end' in features:
                X_datetime[f'{col}_is_year_end'] = dt_col.dt.is_year_end.astype(int)
                extracted_features.append('is_year_end')

            # Store extracted features for this column
            self.datetime_features_[col] = extracted_features

        return X_datetime

    def create_interaction_features(
        self,
        X: pd.DataFrame,
        column_pairs: Optional[List[Tuple[str, str]]] = None,
        operations: List[str] = ['multiply']
    ) -> pd.DataFrame:
        """
        Create interaction features between column pairs.

        Interaction features capture how two features work together to influence
        the target. Often, the combined effect of features is more important than
        their individual effects.

        What are Interaction Features?
        -------------------------------
        Interaction features are mathematical combinations of two features that
        capture their joint relationship with the target.

        Example:
        - Features: area=100 sq ft, price_per_sqft=$500
        - Interaction (multiply): area × price_per_sqft = 100 × 500 = $50,000
        - This interaction directly gives us the total price!

        Why Create Interactions?
        ------------------------

        1. **Capture Non-additive Effects**:
           Sometimes features multiply rather than add:
           - Income × Debt Ratio = Financial Risk
           - Area × Quality = Property Value

        2. **Domain Knowledge**:
           Business logic often involves multiplicative relationships:
           - Marketing: Impressions × Click Rate = Clicks
           - E-commerce: Traffic × Conversion Rate = Sales

        3. **Improve Simple Models**:
           Linear models can capture complex relationships with interactions:
           - y = w1×feature1 + w2×feature2 + w3×(feature1×feature2)

        4. **Reveal Hidden Patterns**:
           Sometimes the interaction is more predictive than individual features

        Operation Types:
        ----------------

        1. **Multiply** (product):
           - area × price_per_sqft = total_price
           - height × width = area
           - Best when: Features scale together

        2. **Add** (sum):
           - income + savings = total_wealth
           - Best when: Features contribute independently

        3. **Subtract** (difference):
           - income - expenses = savings
           - max_temp - min_temp = temperature_range
           - Best when: Relative difference matters

        4. **Divide** (ratio):
           - debt / income = debt_to_income_ratio
           - price / area = price_per_square_foot
           - Best when: Relative proportion matters

        Choosing Feature Pairs:
        -----------------------

        **Automatic Selection**:
        - If column_pairs=None, creates all pairwise interactions
        - Warning: Can create many features (n*(n-1)/2 pairs)
        - Use feature selection afterwards

        **Manual Selection** (Recommended):
        - Use domain knowledge to select meaningful pairs
        - Much more efficient and interpretable
        - Examples:
          * area × price_per_sqft (real estate)
          * hours × wage (payroll)
          * quantity × price (sales)

        Args:
            X : pd.DataFrame
                Input data with features to create interactions from
            column_pairs : Optional[List[Tuple[str, str]]], default=None
                List of column pairs to create interactions from.
                Example: [('area', 'price'), ('age', 'income')]
                If None, creates interactions for all numerical column pairs.
                Warning: Can create many features for high-dimensional data.
            operations : List[str], default=['multiply']
                Operations to perform on each pair. Options:
                - 'multiply': feature1 × feature2
                - 'add': feature1 + feature2
                - 'subtract': feature1 - feature2
                - 'divide': feature1 / feature2 (handles division by zero)
                Can specify multiple operations for each pair.

        Returns:
            pd.DataFrame
                DataFrame with original features plus interaction features.
                New columns named as '{col1}_{operation}_{col2}'
                Example: 'area_multiply_price', 'income_divide_age'

        Raises:
            ValueError: If specified columns don't exist or operations invalid
            KeyError: If column pairs reference non-existent columns

        Example:
        --------
        >>> import pandas as pd
        >>> from feature_engineer import FeatureEngineer
        >>>
        >>> # Create sample real estate data
        >>> df = pd.DataFrame({
        ...     'area': [1000, 1500, 2000, 2500],
        ...     'price_per_sqft': [500, 450, 550, 600],
        ...     'age': [5, 10, 3, 15],
        ...     'rooms': [2, 3, 4, 5],
        ...     'bathrooms': [1, 2, 2, 3]
        ... })
        >>>
        >>> engineer = FeatureEngineer()
        >>>
        >>> # Create specific interactions based on domain knowledge
        >>> df_interactions = engineer.create_interaction_features(
        ...     df,
        ...     column_pairs=[
        ...         ('area', 'price_per_sqft'),  # Total price
        ...         ('rooms', 'bathrooms'),       # Total facilities
        ...         ('area', 'age')               # Size-age interaction
        ...     ],
        ...     operations=['multiply']
        ... )
        >>> print(df_interactions.columns)
        >>> # ['area', 'price_per_sqft', 'age', 'rooms', 'bathrooms',
        >>> #  'area_multiply_price_per_sqft', 'rooms_multiply_bathrooms',
        >>> #  'area_multiply_age']
        >>>
        >>> # Create multiple operation types
        >>> df_multi_ops = engineer.create_interaction_features(
        ...     df,
        ...     column_pairs=[('area', 'rooms')],
        ...     operations=['multiply', 'divide', 'add']
        ... )
        >>> print(df_multi_ops.columns)
        >>> # [..., 'area_multiply_rooms', 'area_divide_rooms', 'area_add_rooms']
        >>>
        >>> # Create ratios (divide operation)
        >>> df_ratios = engineer.create_interaction_features(
        ...     df,
        ...     column_pairs=[
        ...         ('area', 'rooms'),      # Area per room
        ...         ('rooms', 'bathrooms')  # Room to bathroom ratio
        ...     ],
        ...     operations=['divide']
        ... )
        >>>
        >>> # Use in model training
        >>> from sklearn.ensemble import RandomForestRegressor
        >>> X = df_interactions.drop('price_per_sqft', axis=1)
        >>> y = df_interactions['price_per_sqft']
        >>> model = RandomForestRegressor()
        >>> model.fit(X, y)

        Notes:
        ------
        - Start with domain-knowledge based pairs, not all pairs
        - Creating all pairs can lead to curse of dimensionality
        - Use feature selection to remove unhelpful interactions
        - Division handles zero by replacing with small epsilon
        - Consider scaling features before creating interactions
        - Tree-based models can learn interactions, but linear models benefit more
        - Save interaction column pairs for consistent production inference
        """
        X_interactions = X.copy()

        # Auto-generate all numerical column pairs if not specified
        if column_pairs is None:
            numerical_cols = X_interactions.select_dtypes(include=[np.number]).columns.tolist()

            if len(numerical_cols) < 2:
                warnings.warn("Need at least 2 numerical columns for interactions")
                return X_interactions

            # Generate all unique pairs
            column_pairs = [
                (col1, col2)
                for i, col1 in enumerate(numerical_cols)
                for col2 in numerical_cols[i+1:]
            ]

            if len(column_pairs) > 50:
                warnings.warn(
                    f"Creating {len(column_pairs)} interaction pairs. "
                    f"Consider specifying specific pairs to reduce dimensionality."
                )

        # Validate operations
        valid_operations = ['multiply', 'add', 'subtract', 'divide']
        for op in operations:
            if op not in valid_operations:
                raise ValueError(
                    f"Invalid operation: {op}. "
                    f"Choose from {valid_operations}"
                )

        # Create interactions
        for col1, col2 in column_pairs:
            if col1 not in X_interactions.columns:
                raise KeyError(f"Column '{col1}' not found in DataFrame")
            if col2 not in X_interactions.columns:
                raise KeyError(f"Column '{col2}' not found in DataFrame")

            for operation in operations:
                feature_name = f'{col1}_{operation}_{col2}'

                if operation == 'multiply':
                    X_interactions[feature_name] = (
                        X_interactions[col1] * X_interactions[col2]
                    )

                elif operation == 'add':
                    X_interactions[feature_name] = (
                        X_interactions[col1] + X_interactions[col2]
                    )

                elif operation == 'subtract':
                    X_interactions[feature_name] = (
                        X_interactions[col1] - X_interactions[col2]
                    )

                elif operation == 'divide':
                    # Handle division by zero
                    denominator = X_interactions[col2].replace(0, np.finfo(float).eps)
                    X_interactions[feature_name] = (
                        X_interactions[col1] / denominator
                    )

        # Store created interaction pairs
        self.interaction_features_ = column_pairs

        return X_interactions

    def save(self, filepath: str) -> None:
        """
        Save the fitted FeatureEngineer to disk.

        Saving the fitted engineer is crucial for production deployment because:
        1. Test data must use the same transformations as training data
        2. Production inference requires identical preprocessing
        3. Ensures reproducibility across different environments
        4. Avoids refitting on test/production data (prevents data leakage)

        What Gets Saved:
        ----------------
        - All fitted scalers (StandardScaler, MinMaxScaler, etc.)
        - All fitted encoders (LabelEncoder, OneHotEncoder, etc.)
        - Polynomial feature transformers
        - Feature selectors
        - Selected feature names
        - Interaction feature definitions
        - Datetime feature configurations

        Args:
            filepath : str
                Path where the engineer will be saved.
                Recommended extension: .pkl or .joblib
                Example: 'models/feature_engineer.pkl'

        Raises:
            RuntimeError: If engineer hasn't been fitted yet
            IOError: If file cannot be written

        Example:
        --------
        >>> from feature_engineer import FeatureEngineer
        >>>
        >>> # Fit engineer on training data
        >>> engineer = FeatureEngineer()
        >>> X_train_transformed = engineer.fit_transform(X_train, y_train)
        >>>
        >>> # Save for later use
        >>> engineer.save('models/feature_engineer.pkl')
        >>>
        >>> # Later, in production...
        >>> engineer_loaded = FeatureEngineer.load('models/feature_engineer.pkl')
        >>> X_new_transformed = engineer_loaded.transform(X_new)

        Notes:
        ------
        - Only save after fitting on training data
        - Store with model artifacts for deployment
        - Version control engineer files with model versions
        - Document what preprocessing was applied
        """
        if not self.fitted_:
            raise RuntimeError(
                "Cannot save unfitted FeatureEngineer. Call fit() first."
            )

        import pickle

        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filepath: str) -> 'FeatureEngineer':
        """
        Load a fitted FeatureEngineer from disk.

        Loading a saved engineer ensures consistent preprocessing between
        training and production environments.

        Args:
            filepath : str
                Path to the saved engineer file

        Returns:
            FeatureEngineer
                Loaded engineer with all fitted parameters

        Raises:
            FileNotFoundError: If file doesn't exist
            pickle.UnpicklingError: If file is corrupted

        Example:
        --------
        >>> from feature_engineer import FeatureEngineer
        >>>
        >>> # Load saved engineer
        >>> engineer = FeatureEngineer.load('models/feature_engineer.pkl')
        >>>
        >>> # Transform new data using saved parameters
        >>> X_new_transformed = engineer.transform(X_new)

        Notes:
        ------
        - Ensure consistent sklearn versions between save/load environments
        - Verify loaded engineer works before production deployment
        - Keep backup copies of engineer files
        """
        import pickle

        if not Path(filepath).exists():
            raise FileNotFoundError(f"File not found: {filepath}")

        with open(filepath, 'rb') as f:
            engineer = pickle.load(f)

        return engineer
