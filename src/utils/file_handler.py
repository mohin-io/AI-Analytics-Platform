"""
File handling utilities for the Unified AI Analytics Platform

This module provides utilities for reading, writing, and managing various file formats.
"""

import json
import pickle
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import yaml


class FileHandler:
    """
    Utility class for handling various file operations.

    This class provides methods to read and write different file formats commonly
    used in machine learning projects, including CSV, JSON, Parquet, pickle files,
    and more. It also handles model serialization and deserialization.

    Example:
        >>> handler = FileHandler()
        >>> df = handler.read_csv("data/train.csv")
        >>> handler.save_model(model, "models/my_model.pkl")
        >>> loaded_model = handler.load_model("models/my_model.pkl")
    """

    @staticmethod
    def read_csv(
        file_path: str,
        **kwargs: Any
    ) -> pd.DataFrame:
        """
        Read a CSV file into a pandas DataFrame.

        Args:
            file_path: Path to the CSV file
            **kwargs: Additional arguments to pass to pandas.read_csv()

        Returns:
            DataFrame containing the CSV data

        Example:
            >>> df = FileHandler.read_csv("data/train.csv", index_col=0)
        """
        return pd.read_csv(file_path, **kwargs)

    @staticmethod
    def write_csv(
        df: pd.DataFrame,
        file_path: str,
        **kwargs: Any
    ) -> None:
        """
        Write a DataFrame to a CSV file.

        Args:
            df: DataFrame to write
            file_path: Output file path
            **kwargs: Additional arguments to pass to DataFrame.to_csv()

        Example:
            >>> FileHandler.write_csv(df, "output/results.csv", index=False)
        """
        output_path = Path(file_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(file_path, **kwargs)

    @staticmethod
    def read_json(file_path: str) -> Union[Dict, List]:
        """
        Read a JSON file.

        Args:
            file_path: Path to the JSON file

        Returns:
            Parsed JSON data (dict or list)

        Example:
            >>> config = FileHandler.read_json("config/settings.json")
        """
        with open(file_path, 'r') as f:
            return json.load(f)

    @staticmethod
    def write_json(
        data: Union[Dict, List],
        file_path: str,
        indent: int = 2
    ) -> None:
        """
        Write data to a JSON file.

        Args:
            data: Data to write (dict or list)
            file_path: Output file path
            indent: JSON indentation level

        Example:
            >>> FileHandler.write_json({"key": "value"}, "output/data.json")
        """
        output_path = Path(file_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=indent)

    @staticmethod
    def read_yaml(file_path: str) -> Dict:
        """
        Read a YAML file.

        Args:
            file_path: Path to the YAML file

        Returns:
            Parsed YAML data as a dictionary

        Example:
            >>> settings = FileHandler.read_yaml("config/settings.yaml")
        """
        with open(file_path, 'r') as f:
            return yaml.safe_load(f)

    @staticmethod
    def write_yaml(
        data: Dict,
        file_path: str
    ) -> None:
        """
        Write data to a YAML file.

        Args:
            data: Dictionary to write
            file_path: Output file path

        Example:
            >>> FileHandler.write_yaml({"model": "xgboost"}, "config/model.yaml")
        """
        output_path = Path(file_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, 'w') as f:
            yaml.dump(data, f, default_flow_style=False)

    @staticmethod
    def read_parquet(file_path: str) -> pd.DataFrame:
        """
        Read a Parquet file into a DataFrame.

        Args:
            file_path: Path to the Parquet file

        Returns:
            DataFrame containing the Parquet data

        Example:
            >>> df = FileHandler.read_parquet("data/large_dataset.parquet")
        """
        return pd.read_parquet(file_path)

    @staticmethod
    def write_parquet(
        df: pd.DataFrame,
        file_path: str,
        **kwargs: Any
    ) -> None:
        """
        Write a DataFrame to a Parquet file.

        Parquet is a columnar storage format that is more efficient than CSV
        for large datasets and preserves data types.

        Args:
            df: DataFrame to write
            file_path: Output file path
            **kwargs: Additional arguments to pass to DataFrame.to_parquet()

        Example:
            >>> FileHandler.write_parquet(df, "output/data.parquet")
        """
        output_path = Path(file_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(file_path, **kwargs)

    @staticmethod
    def save_model(
        model: Any,
        file_path: str,
        use_joblib: bool = True
    ) -> None:
        """
        Save a machine learning model to disk.

        This method serializes a model using either joblib (recommended for
        scikit-learn models) or pickle. Joblib is more efficient for models
        with large numpy arrays.

        Args:
            model: The model object to save
            file_path: Output file path
            use_joblib: If True, use joblib; otherwise use pickle

        Example:
            >>> from sklearn.ensemble import RandomForestClassifier
            >>> model = RandomForestClassifier()
            >>> FileHandler.save_model(model, "models/rf_model.pkl")
        """
        output_path = Path(file_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if use_joblib:
            joblib.dump(model, file_path)
        else:
            with open(file_path, 'wb') as f:
                pickle.dump(model, f)

    @staticmethod
    def load_model(
        file_path: str,
        use_joblib: bool = True
    ) -> Any:
        """
        Load a machine learning model from disk.

        Args:
            file_path: Path to the saved model file
            use_joblib: If True, use joblib; otherwise use pickle

        Returns:
            The loaded model object

        Example:
            >>> model = FileHandler.load_model("models/rf_model.pkl")
            >>> predictions = model.predict(X_test)
        """
        if use_joblib:
            return joblib.load(file_path)
        else:
            with open(file_path, 'rb') as f:
                return pickle.load(f)

    @staticmethod
    def save_numpy(
        array: np.ndarray,
        file_path: str
    ) -> None:
        """
        Save a numpy array to disk.

        Args:
            array: Numpy array to save
            file_path: Output file path

        Example:
            >>> arr = np.array([1, 2, 3])
            >>> FileHandler.save_numpy(arr, "data/array.npy")
        """
        output_path = Path(file_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(file_path, array)

    @staticmethod
    def load_numpy(file_path: str) -> np.ndarray:
        """
        Load a numpy array from disk.

        Args:
            file_path: Path to the numpy file

        Returns:
            Loaded numpy array

        Example:
            >>> arr = FileHandler.load_numpy("data/array.npy")
        """
        return np.load(file_path)

    @staticmethod
    def read_excel(
        file_path: str,
        sheet_name: Union[str, int] = 0,
        **kwargs: Any
    ) -> pd.DataFrame:
        """
        Read an Excel file into a DataFrame.

        Args:
            file_path: Path to the Excel file
            sheet_name: Name or index of the sheet to read
            **kwargs: Additional arguments to pass to pandas.read_excel()

        Returns:
            DataFrame containing the Excel data

        Example:
            >>> df = FileHandler.read_excel("data/sales.xlsx", sheet_name="Q1")
        """
        return pd.read_excel(file_path, sheet_name=sheet_name, **kwargs)

    @staticmethod
    def write_excel(
        df: pd.DataFrame,
        file_path: str,
        sheet_name: str = "Sheet1",
        **kwargs: Any
    ) -> None:
        """
        Write a DataFrame to an Excel file.

        Args:
            df: DataFrame to write
            file_path: Output file path
            sheet_name: Name of the sheet
            **kwargs: Additional arguments to pass to DataFrame.to_excel()

        Example:
            >>> FileHandler.write_excel(df, "output/report.xlsx", sheet_name="Results")
        """
        output_path = Path(file_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_excel(file_path, sheet_name=sheet_name, **kwargs)

    @staticmethod
    def file_exists(file_path: str) -> bool:
        """
        Check if a file exists.

        Args:
            file_path: Path to check

        Returns:
            True if file exists, False otherwise

        Example:
            >>> if FileHandler.file_exists("models/my_model.pkl"):
            ...     model = FileHandler.load_model("models/my_model.pkl")
        """
        return Path(file_path).exists()

    @staticmethod
    def create_directory(dir_path: str) -> None:
        """
        Create a directory if it doesn't exist.

        Args:
            dir_path: Directory path to create

        Example:
            >>> FileHandler.create_directory("output/models/version_1")
        """
        Path(dir_path).mkdir(parents=True, exist_ok=True)

    @staticmethod
    def list_files(
        directory: str,
        pattern: str = "*",
        recursive: bool = False
    ) -> List[str]:
        """
        List files in a directory matching a pattern.

        Args:
            directory: Directory to search
            pattern: File pattern (e.g., "*.csv", "model_*.pkl")
            recursive: If True, search subdirectories

        Returns:
            List of file paths matching the pattern

        Example:
            >>> csv_files = FileHandler.list_files("data", pattern="*.csv")
            >>> print(csv_files)
            ['data/train.csv', 'data/test.csv']
        """
        dir_path = Path(directory)
        if recursive:
            return [str(f) for f in dir_path.rglob(pattern)]
        else:
            return [str(f) for f in dir_path.glob(pattern)]
