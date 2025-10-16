"""
Data loading utilities for the Unified AI Analytics Platform

This module provides functionality to load data from various sources including
files (CSV, JSON, Parquet, Excel), databases, and APIs.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Union, Dict, Any
from sqlalchemy import create_engine
import warnings


class DataLoader:
    """
    Load data from various sources.

    This class provides methods to load data from different file formats,
    databases, and other sources. It automatically detects the file type
    based on the file extension and applies appropriate loading methods.

    Example:
        >>> loader = DataLoader()
        >>> df = loader.load_from_csv("data/train.csv")
        >>> df = loader.load_from_sql("SELECT * FROM users", "sqlite:///database.db")
        >>> df = loader.auto_load("data/dataset.parquet")
    """

    def __init__(self):
        """Initialize the DataLoader."""
        self.supported_formats = {
            '.csv': self.load_from_csv,
            '.tsv': self.load_from_tsv,
            '.json': self.load_from_json,
            '.parquet': self.load_from_parquet,
            '.xlsx': self.load_from_excel,
            '.xls': self.load_from_excel,
        }

    def auto_load(
        self,
        file_path: str,
        **kwargs: Any
    ) -> pd.DataFrame:
        """
        Automatically load data based on file extension.

        This method detects the file format from the extension and calls
        the appropriate loading method. It's the recommended way to load
        files when you want automatic format detection.

        Args:
            file_path: Path to the data file
            **kwargs: Additional arguments passed to the specific loader

        Returns:
            DataFrame containing the loaded data

        Raises:
            ValueError: If file format is not supported
            FileNotFoundError: If file doesn't exist

        Example:
            >>> loader = DataLoader()
            >>> df = loader.auto_load("data/train.csv")
            >>> df = loader.auto_load("data/dataset.parquet")
        """
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        file_ext = path.suffix.lower()

        if file_ext not in self.supported_formats:
            raise ValueError(
                f"Unsupported file format: {file_ext}. "
                f"Supported formats: {list(self.supported_formats.keys())}"
            )

        loader_func = self.supported_formats[file_ext]
        return loader_func(file_path, **kwargs)

    def load_from_csv(
        self,
        file_path: str,
        **kwargs: Any
    ) -> pd.DataFrame:
        """
        Load data from a CSV file.

        This method reads CSV files using pandas read_csv with sensible defaults.
        It automatically infers data types and handles common CSV variations.

        Args:
            file_path: Path to the CSV file
            **kwargs: Additional arguments for pandas.read_csv()
                     (e.g., sep=',', encoding='utf-8', index_col=0)

        Returns:
            DataFrame containing the CSV data

        Example:
            >>> loader = DataLoader()
            >>> df = loader.load_from_csv("data/train.csv")
            >>> df = loader.load_from_csv("data/custom.csv", sep=";", encoding="latin1")
        """
        try:
            df = pd.read_csv(file_path, **kwargs)
            return df
        except Exception as e:
            raise RuntimeError(f"Failed to load CSV file: {e}")

    def load_from_tsv(
        self,
        file_path: str,
        **kwargs: Any
    ) -> pd.DataFrame:
        """
        Load data from a TSV (Tab-Separated Values) file.

        Args:
            file_path: Path to the TSV file
            **kwargs: Additional arguments for pandas.read_csv()

        Returns:
            DataFrame containing the TSV data

        Example:
            >>> loader = DataLoader()
            >>> df = loader.load_from_tsv("data/dataset.tsv")
        """
        kwargs['sep'] = kwargs.get('sep', '\t')
        return self.load_from_csv(file_path, **kwargs)

    def load_from_json(
        self,
        file_path: str,
        orient: str = 'records',
        **kwargs: Any
    ) -> pd.DataFrame:
        """
        Load data from a JSON file.

        This method supports various JSON formats including records, columns,
        index, split, and values orientations.

        Args:
            file_path: Path to the JSON file
            orient: JSON orientation ('records', 'columns', 'index', 'split', 'values')
            **kwargs: Additional arguments for pandas.read_json()

        Returns:
            DataFrame containing the JSON data

        Example:
            >>> loader = DataLoader()
            >>> df = loader.load_from_json("data/data.json", orient='records')
        """
        try:
            df = pd.read_json(file_path, orient=orient, **kwargs)
            return df
        except Exception as e:
            raise RuntimeError(f"Failed to load JSON file: {e}")

    def load_from_parquet(
        self,
        file_path: str,
        **kwargs: Any
    ) -> pd.DataFrame:
        """
        Load data from a Parquet file.

        Parquet is a columnar storage format that is more efficient than CSV
        for large datasets and preserves data types accurately.

        Args:
            file_path: Path to the Parquet file
            **kwargs: Additional arguments for pandas.read_parquet()

        Returns:
            DataFrame containing the Parquet data

        Example:
            >>> loader = DataLoader()
            >>> df = loader.load_from_parquet("data/large_dataset.parquet")
        """
        try:
            df = pd.read_parquet(file_path, **kwargs)
            return df
        except Exception as e:
            raise RuntimeError(f"Failed to load Parquet file: {e}")

    def load_from_excel(
        self,
        file_path: str,
        sheet_name: Union[str, int] = 0,
        **kwargs: Any
    ) -> pd.DataFrame:
        """
        Load data from an Excel file.

        This method can read both .xlsx and .xls formats and supports
        loading specific sheets by name or index.

        Args:
            file_path: Path to the Excel file
            sheet_name: Name or index of the sheet to read
            **kwargs: Additional arguments for pandas.read_excel()

        Returns:
            DataFrame containing the Excel data

        Example:
            >>> loader = DataLoader()
            >>> df = loader.load_from_excel("data/sales.xlsx", sheet_name="Q1")
            >>> df = loader.load_from_excel("data/report.xls", sheet_name=0)
        """
        try:
            df = pd.read_excel(file_path, sheet_name=sheet_name, **kwargs)
            return df
        except Exception as e:
            raise RuntimeError(f"Failed to load Excel file: {e}")

    def load_from_sql(
        self,
        query: str,
        connection_string: str,
        **kwargs: Any
    ) -> pd.DataFrame:
        """
        Load data from a SQL database.

        This method executes a SQL query and returns the results as a DataFrame.
        It supports various databases through SQLAlchemy connection strings.

        Args:
            query: SQL query to execute
            connection_string: SQLAlchemy connection string
                Examples:
                - PostgreSQL: "postgresql://user:password@localhost/dbname"
                - MySQL: "mysql+pymysql://user:password@localhost/dbname"
                - SQLite: "sqlite:///path/to/database.db"
            **kwargs: Additional arguments for pandas.read_sql()

        Returns:
            DataFrame containing the query results

        Example:
            >>> loader = DataLoader()
            >>> df = loader.load_from_sql(
            ...     "SELECT * FROM users WHERE age > 18",
            ...     "postgresql://user:pass@localhost/mydb"
            ... )
        """
        try:
            engine = create_engine(connection_string)
            df = pd.read_sql(query, engine, **kwargs)
            return df
        except Exception as e:
            raise RuntimeError(f"Failed to load from SQL: {e}")

    def load_from_url(
        self,
        url: str,
        file_type: Optional[str] = None,
        **kwargs: Any
    ) -> pd.DataFrame:
        """
        Load data from a URL.

        This method downloads data from a URL and loads it into a DataFrame.
        It can auto-detect the format from the URL or you can specify it explicitly.

        Args:
            url: URL to the data file
            file_type: Optional file type ('csv', 'json', 'parquet', 'excel')
                      If None, attempts to infer from URL
            **kwargs: Additional arguments for the specific loader

        Returns:
            DataFrame containing the data from the URL

        Example:
            >>> loader = DataLoader()
            >>> df = loader.load_from_url(
            ...     "https://example.com/data.csv",
            ...     file_type="csv"
            ... )
        """
        try:
            if file_type is None:
                # Try to infer from URL
                if url.endswith('.csv'):
                    file_type = 'csv'
                elif url.endswith('.json'):
                    file_type = 'json'
                elif url.endswith('.parquet'):
                    file_type = 'parquet'
                elif url.endswith(('.xlsx', '.xls')):
                    file_type = 'excel'
                else:
                    # Default to CSV
                    file_type = 'csv'
                    warnings.warn(
                        f"Could not infer file type from URL, assuming CSV. "
                        f"Specify file_type explicitly if this is incorrect."
                    )

            if file_type == 'csv':
                return pd.read_csv(url, **kwargs)
            elif file_type == 'json':
                return pd.read_json(url, **kwargs)
            elif file_type == 'parquet':
                return pd.read_parquet(url, **kwargs)
            elif file_type == 'excel':
                return pd.read_excel(url, **kwargs)
            else:
                raise ValueError(f"Unsupported file type: {file_type}")

        except Exception as e:
            raise RuntimeError(f"Failed to load from URL: {e}")

    def load_sample_dataset(
        self,
        dataset_name: str
    ) -> pd.DataFrame:
        """
        Load a sample dataset for testing and demonstration.

        This method provides quick access to popular sample datasets like
        Iris, Titanic, Boston Housing, etc. using seaborn or sklearn.

        Args:
            dataset_name: Name of the sample dataset
                Supported: 'iris', 'titanic', 'tips', 'diamonds', 'boston'

        Returns:
            DataFrame containing the sample dataset

        Example:
            >>> loader = DataLoader()
            >>> df = loader.load_sample_dataset('iris')
            >>> df = loader.load_sample_dataset('titanic')
        """
        dataset_name = dataset_name.lower()

        try:
            if dataset_name in ['iris', 'titanic', 'tips', 'diamonds', 'penguins']:
                import seaborn as sns
                df = sns.load_dataset(dataset_name)
                return df

            elif dataset_name == 'boston':
                from sklearn.datasets import load_boston
                warnings.warn(
                    "Boston Housing dataset is deprecated. "
                    "Consider using California Housing instead."
                )
                data = load_boston()
                df = pd.DataFrame(data.data, columns=data.feature_names)
                df['target'] = data.target
                return df

            elif dataset_name == 'california':
                from sklearn.datasets import fetch_california_housing
                data = fetch_california_housing()
                df = pd.DataFrame(data.data, columns=data.feature_names)
                df['target'] = data.target
                return df

            elif dataset_name == 'wine':
                from sklearn.datasets import load_wine
                data = load_wine()
                df = pd.DataFrame(data.data, columns=data.feature_names)
                df['target'] = data.target
                return df

            elif dataset_name == 'breast_cancer':
                from sklearn.datasets import load_breast_cancer
                data = load_breast_cancer()
                df = pd.DataFrame(data.data, columns=data.feature_names)
                df['target'] = data.target
                return df

            else:
                raise ValueError(
                    f"Unknown dataset: {dataset_name}. "
                    f"Supported datasets: iris, titanic, tips, diamonds, penguins, "
                    f"california, wine, breast_cancer"
                )

        except Exception as e:
            raise RuntimeError(f"Failed to load sample dataset '{dataset_name}': {e}")

    def get_data_info(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Get comprehensive information about a DataFrame.

        This method provides a summary of the DataFrame including shape,
        data types, missing values, and memory usage.

        Args:
            df: DataFrame to analyze

        Returns:
            Dictionary containing data information

        Example:
            >>> loader = DataLoader()
            >>> df = loader.load_from_csv("data/train.csv")
            >>> info = loader.get_data_info(df)
            >>> print(info['shape'])
            (1000, 10)
        """
        info = {
            'shape': df.shape,
            'columns': df.columns.tolist(),
            'dtypes': df.dtypes.to_dict(),
            'missing_values': df.isnull().sum().to_dict(),
            'missing_percentage': (df.isnull().sum() / len(df) * 100).to_dict(),
            'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024 / 1024,
            'duplicate_rows': df.duplicated().sum(),
        }

        return info
