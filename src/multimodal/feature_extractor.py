"""
Multi-Modal Feature Extraction

Extract features from different modalities (text, images, time-series, etc.)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Union
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import StandardScaler
import warnings


class MultiModalFeatureExtractor:
    """
    Extract and process features from multiple data modalities.

    Supports:
    - Tabular/numerical data
    - Text data
    - Time-series data
    - Categorical data
    """

    def __init__(self):
        """Initialize multi-modal feature extractor."""
        self.extractors: Dict[str, Any] = {}
        self.modality_types: Dict[str, str] = {}

    def add_modality(self, name: str, modality_type: str, **extractor_params):
        """
        Add a modality with its feature extractor.

        Args:
            name: Modality name
            modality_type: Type of modality ('tabular', 'text', 'timeseries', 'categorical')
            **extractor_params: Parameters for the extractor
        """
        self.modality_types[name] = modality_type

        if modality_type == 'text':
            # Use TF-IDF for text
            max_features = extractor_params.get('max_features', 1000)
            ngram_range = extractor_params.get('ngram_range', (1, 2))
            self.extractors[name] = TfidfVectorizer(
                max_features=max_features,
                ngram_range=ngram_range,
                stop_words='english'
            )
        elif modality_type == 'tabular':
            # Use standard scaler for tabular
            self.extractors[name] = StandardScaler()
        elif modality_type == 'timeseries':
            # Use statistical features for time series
            self.extractors[name] = TimeSeriesFeatureExtractor(**extractor_params)
        elif modality_type == 'categorical':
            # Use one-hot encoding for categorical
            self.extractors[name] = CategoricalFeatureExtractor(**extractor_params)
        else:
            raise ValueError(f"Unknown modality type: {modality_type}")

    def fit(self, data: Dict[str, Any]) -> 'MultiModalFeatureExtractor':
        """
        Fit all feature extractors.

        Args:
            data: Dictionary mapping modality names to data

        Returns:
            Self
        """
        for modality, extractor in self.extractors.items():
            if modality not in data:
                warnings.warn(f"Modality '{modality}' not found in data")
                continue

            modality_data = data[modality]

            if self.modality_types[modality] == 'text':
                # Fit text vectorizer
                extractor.fit(modality_data)
            elif self.modality_types[modality] == 'tabular':
                # Fit scaler
                extractor.fit(modality_data)
            elif self.modality_types[modality] in ['timeseries', 'categorical']:
                # Fit custom extractor
                extractor.fit(modality_data)

        return self

    def transform(self, data: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """
        Transform all modalities to feature arrays.

        Args:
            data: Dictionary mapping modality names to data

        Returns:
            Dictionary mapping modality names to feature arrays
        """
        features = {}

        for modality, extractor in self.extractors.items():
            if modality not in data:
                warnings.warn(f"Modality '{modality}' not found in data")
                continue

            modality_data = data[modality]

            if self.modality_types[modality] == 'text':
                # Transform text to TF-IDF
                feat = extractor.transform(modality_data).toarray()
            elif self.modality_types[modality] == 'tabular':
                # Scale tabular data
                feat = extractor.transform(modality_data)
            elif self.modality_types[modality] in ['timeseries', 'categorical']:
                # Transform using custom extractor
                feat = extractor.transform(modality_data)
            else:
                feat = modality_data

            features[modality] = feat

        return features

    def fit_transform(self, data: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """
        Fit and transform in one step.

        Args:
            data: Dictionary mapping modality names to data

        Returns:
            Dictionary mapping modality names to feature arrays
        """
        return self.fit(data).transform(data)


class TimeSeriesFeatureExtractor:
    """
    Extract statistical features from time series data.
    """

    def __init__(self, window_size: int = 10):
        """
        Initialize time series feature extractor.

        Args:
            window_size: Window size for rolling statistics
        """
        self.window_size = window_size
        self.feature_names: List[str] = []

    def fit(self, X: np.ndarray) -> 'TimeSeriesFeatureExtractor':
        """
        Fit extractor (learns feature names).

        Args:
            X: Time series data (n_samples, n_timesteps) or (n_samples, n_timesteps, n_features)

        Returns:
            Self
        """
        if X.ndim == 2:
            n_features = 1
        else:
            n_features = X.shape[2]

        # Define feature names
        self.feature_names = []
        for i in range(n_features):
            self.feature_names.extend([
                f'ts{i}_mean',
                f'ts{i}_std',
                f'ts{i}_min',
                f'ts{i}_max',
                f'ts{i}_median',
                f'ts{i}_range',
                f'ts{i}_skew',
                f'ts{i}_kurt'
            ])

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform time series to statistical features.

        Args:
            X: Time series data

        Returns:
            Feature array (n_samples, n_features)
        """
        if X.ndim == 2:
            # Single feature time series
            X = X[:, :, np.newaxis]

        n_samples, n_timesteps, n_features = X.shape
        features = []

        for i in range(n_samples):
            sample_features = []

            for j in range(n_features):
                ts = X[i, :, j]

                # Statistical features
                sample_features.extend([
                    np.mean(ts),
                    np.std(ts),
                    np.min(ts),
                    np.max(ts),
                    np.median(ts),
                    np.max(ts) - np.min(ts),  # range
                    self._skewness(ts),
                    self._kurtosis(ts)
                ])

            features.append(sample_features)

        return np.array(features)

    def _skewness(self, x: np.ndarray) -> float:
        """Calculate skewness."""
        mean = np.mean(x)
        std = np.std(x)
        if std == 0:
            return 0
        return np.mean(((x - mean) / std) ** 3)

    def _kurtosis(self, x: np.ndarray) -> float:
        """Calculate kurtosis."""
        mean = np.mean(x)
        std = np.std(x)
        if std == 0:
            return 0
        return np.mean(((x - mean) / std) ** 4) - 3


class CategoricalFeatureExtractor:
    """
    Extract features from categorical data using various encoding strategies.
    """

    def __init__(self, encoding: str = 'onehot', max_categories: int = 100):
        """
        Initialize categorical feature extractor.

        Args:
            encoding: Encoding strategy ('onehot', 'label', 'frequency')
            max_categories: Maximum number of categories to encode
        """
        self.encoding = encoding
        self.max_categories = max_categories
        self.category_maps: Dict[int, Dict[Any, int]] = {}
        self.frequency_maps: Dict[int, Dict[Any, float]] = {}

    def fit(self, X: Union[np.ndarray, pd.DataFrame]) -> 'CategoricalFeatureExtractor':
        """
        Fit encoder on categorical data.

        Args:
            X: Categorical data

        Returns:
            Self
        """
        if isinstance(X, pd.DataFrame):
            X = X.values

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        n_features = X.shape[1]

        for i in range(n_features):
            feature = X[:, i]
            unique_values = np.unique(feature)

            # Limit to max_categories
            if len(unique_values) > self.max_categories:
                # Keep most frequent categories
                value_counts = pd.Series(feature).value_counts()
                top_categories = value_counts.head(self.max_categories).index.tolist()
                unique_values = np.array(top_categories)

            # Create mappings
            self.category_maps[i] = {val: idx for idx, val in enumerate(unique_values)}

            # Frequency encoding
            if self.encoding == 'frequency':
                value_counts = pd.Series(feature).value_counts(normalize=True)
                self.frequency_maps[i] = value_counts.to_dict()

        return self

    def transform(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Transform categorical data to encoded features.

        Args:
            X: Categorical data

        Returns:
            Encoded feature array
        """
        if isinstance(X, pd.DataFrame):
            X = X.values

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        n_samples, n_features = X.shape

        if self.encoding == 'onehot':
            # One-hot encoding
            encoded_features = []

            for i in range(n_features):
                feature = X[:, i]
                n_categories = len(self.category_maps[i])
                onehot = np.zeros((n_samples, n_categories))

                for j, val in enumerate(feature):
                    if val in self.category_maps[i]:
                        category_idx = self.category_maps[i][val]
                        onehot[j, category_idx] = 1

                encoded_features.append(onehot)

            return np.hstack(encoded_features)

        elif self.encoding == 'label':
            # Label encoding
            encoded = np.zeros_like(X, dtype=int)

            for i in range(n_features):
                for j, val in enumerate(X[:, i]):
                    if val in self.category_maps[i]:
                        encoded[j, i] = self.category_maps[i][val]

            return encoded.astype(float)

        elif self.encoding == 'frequency':
            # Frequency encoding
            encoded = np.zeros_like(X, dtype=float)

            for i in range(n_features):
                for j, val in enumerate(X[:, i]):
                    if val in self.frequency_maps[i]:
                        encoded[j, i] = self.frequency_maps[i][val]

            return encoded

        else:
            raise ValueError(f"Unknown encoding: {self.encoding}")
