"""
Data validation utilities for the Unified AI Analytics Platform

This module provides comprehensive data validation functionality to ensure data quality
and integrity before processing. It validates schema, detects data quality issues,
identifies outliers, and generates detailed validation reports.

The DataValidator class is designed to be extensible, allowing users to add custom
validation rules while providing a robust set of built-in validators for common
data quality checks.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Callable, Union, Tuple
from pathlib import Path
from datetime import datetime
import warnings
from dataclasses import dataclass, field
from enum import Enum


class ValidationSeverity(Enum):
    """
    Enum representing validation issue severity levels.

    This helps categorize validation findings by their impact:
    - INFO: Informational messages that don't require action
    - WARNING: Issues that should be reviewed but may be acceptable
    - ERROR: Critical issues that must be addressed before processing
    """
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


@dataclass
class ValidationIssue:
    """
    Data class representing a single validation issue.

    This structure provides a consistent way to record and report validation
    problems, making it easy to filter, sort, and analyze issues.

    Attributes:
        column: Name of the column where the issue was found (None for dataset-level issues)
        issue_type: Type of validation issue (e.g., 'missing_values', 'type_mismatch')
        severity: Severity level of the issue
        message: Human-readable description of the issue
        details: Additional details about the issue (counts, examples, etc.)

    Example:
        >>> issue = ValidationIssue(
        ...     column='age',
        ...     issue_type='missing_values',
        ...     severity=ValidationSeverity.WARNING,
        ...     message='Column has missing values',
        ...     details={'count': 15, 'percentage': 1.5}
        ... )
    """
    column: Optional[str]
    issue_type: str
    severity: ValidationSeverity
    message: str
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ValidationResult:
    """
    Container for validation results.

    This class aggregates all validation findings and provides methods to
    analyze and export the results. It separates issues by severity for
    easier filtering and reporting.

    Attributes:
        is_valid: Overall validation status (False if any errors found)
        issues: List of all validation issues found
        metadata: Additional validation metadata (timestamp, dataset info, etc.)

    Example:
        >>> result = validator.validate(df)
        >>> if not result.is_valid:
        ...     print(f"Found {len(result.get_errors())} errors")
        ...     print(result.get_summary())
    """
    is_valid: bool
    issues: List[ValidationIssue] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_errors(self) -> List[ValidationIssue]:
        """Get all ERROR-level issues."""
        return [issue for issue in self.issues if issue.severity == ValidationSeverity.ERROR]

    def get_warnings(self) -> List[ValidationIssue]:
        """Get all WARNING-level issues."""
        return [issue for issue in self.issues if issue.severity == ValidationSeverity.WARNING]

    def get_info(self) -> List[ValidationIssue]:
        """Get all INFO-level issues."""
        return [issue for issue in self.issues if issue.severity == ValidationSeverity.INFO]

    def get_summary(self) -> str:
        """Generate a human-readable summary of validation results."""
        summary = f"Validation Status: {'PASSED' if self.is_valid else 'FAILED'}\n"
        summary += f"Total Issues: {len(self.issues)}\n"
        summary += f"  - Errors: {len(self.get_errors())}\n"
        summary += f"  - Warnings: {len(self.get_warnings())}\n"
        summary += f"  - Info: {len(self.get_info())}\n"
        return summary


class DataValidator:
    """
    Comprehensive data validation for DataFrames.

    This class provides a rich set of validation methods to ensure data quality
    and integrity. It can validate schemas, check for missing values, detect
    duplicates, identify outliers, and run custom validation rules.

    The validator is designed to be flexible and extensible:
    - Built-in validators handle common data quality checks
    - Custom validation rules can be added for domain-specific requirements
    - Validation results are structured for easy analysis and reporting

    Design Philosophy:
    ==================
    1. **Non-destructive**: The validator never modifies the input data. It only
       analyzes and reports issues, leaving data cleaning decisions to the user.

    2. **Comprehensive**: Validates multiple aspects of data quality in a single
       pass, providing a complete picture of data health.

    3. **Configurable**: Thresholds and rules can be customized to match specific
       data quality requirements.

    4. **Informative**: Provides detailed messages and statistics to help users
       understand and address issues.

    Example:
        >>> # Basic validation
        >>> validator = DataValidator()
        >>> df = pd.read_csv('data.csv')
        >>> result = validator.validate(df)
        >>> print(result.get_summary())

        >>> # Validation with schema
        >>> schema = {
        ...     'age': 'int64',
        ...     'name': 'object',
        ...     'salary': 'float64'
        ... }
        >>> result = validator.validate_schema(df, schema)

        >>> # Custom validation rules
        >>> validator.add_custom_rule(
        ...     'age_range',
        ...     lambda df: df['age'].between(0, 120).all(),
        ...     'Age must be between 0 and 120'
        ... )
        >>> result = validator.validate(df)
    """

    def __init__(
        self,
        missing_threshold: float = 0.0,
        duplicate_threshold: float = 0.0,
        outlier_method: str = 'iqr',
        outlier_threshold: float = 1.5
    ):
        """
        Initialize the DataValidator with configurable thresholds.

        Args:
            missing_threshold: Maximum acceptable percentage of missing values (0-100).
                              Values above this trigger a WARNING. Set to 0 to flag any
                              missing values.
            duplicate_threshold: Maximum acceptable percentage of duplicate rows (0-100).
                                Set to 0 to flag any duplicates.
            outlier_method: Method for outlier detection ('iqr', 'zscore', 'isolation_forest').
                          - 'iqr': Uses Interquartile Range (robust to extreme values)
                          - 'zscore': Uses standard deviation (assumes normal distribution)
                          - 'isolation_forest': Uses ML-based detection (best for complex patterns)
            outlier_threshold: Threshold for outlier detection:
                             - For IQR: multiplier for IQR (1.5 = standard, 3.0 = extreme)
                             - For Z-score: number of standard deviations (3.0 = standard)

        Example:
            >>> # Strict validator - flag any issues
            >>> strict_validator = DataValidator(
            ...     missing_threshold=0.0,
            ...     duplicate_threshold=0.0
            ... )

            >>> # Lenient validator - allow some missing values
            >>> lenient_validator = DataValidator(
            ...     missing_threshold=5.0,  # Allow up to 5% missing
            ...     duplicate_threshold=1.0  # Allow up to 1% duplicates
            ... )
        """
        self.missing_threshold = missing_threshold
        self.duplicate_threshold = duplicate_threshold
        self.outlier_method = outlier_method
        self.outlier_threshold = outlier_threshold

        # Custom validation rules storage
        # Each rule is a tuple of (name, function, error_message)
        self.custom_rules: List[Tuple[str, Callable, str]] = []

    def validate(
        self,
        df: pd.DataFrame,
        schema: Optional[Dict[str, str]] = None,
        check_outliers: bool = True,
        check_duplicates: bool = True,
        check_missing: bool = True,
        check_types: bool = True
    ) -> ValidationResult:
        """
        Perform comprehensive validation on a DataFrame.

        This is the main entry point for validation. It runs all enabled
        validation checks and aggregates the results into a single report.

        The validation process includes:
        1. Schema validation (if schema provided)
        2. Data type validation
        3. Missing value detection
        4. Duplicate row detection
        5. Outlier detection (for numeric columns)
        6. Custom validation rules (if any defined)

        Args:
            df: DataFrame to validate
            schema: Optional schema definition mapping column names to expected types
            check_outliers: Whether to check for outliers in numeric columns
            check_duplicates: Whether to check for duplicate rows
            check_missing: Whether to check for missing values
            check_types: Whether to validate data types

        Returns:
            ValidationResult containing all validation findings

        Example:
            >>> validator = DataValidator()
            >>> result = validator.validate(df)
            >>>
            >>> # Check if validation passed
            >>> if result.is_valid:
            ...     print("Data is valid!")
            ... else:
            ...     print("Validation failed:")
            ...     for error in result.get_errors():
            ...         print(f"  - {error.message}")
            >>>
            >>> # Quick validation with specific checks
            >>> result = validator.validate(
            ...     df,
            ...     check_outliers=False,  # Skip outlier detection
            ...     check_duplicates=True
            ... )
        """
        issues: List[ValidationIssue] = []

        # Add metadata
        metadata = {
            'timestamp': datetime.now().isoformat(),
            'dataset_shape': df.shape,
            'columns': df.columns.tolist(),
            'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024 / 1024
        }

        # Schema validation
        if schema is not None:
            schema_issues = self.validate_schema(df, schema).issues
            issues.extend(schema_issues)

        # Data type validation
        if check_types:
            type_issues = self.check_data_types(df).issues
            issues.extend(type_issues)

        # Missing value validation
        if check_missing:
            missing_issues = self.check_missing_values(df).issues
            issues.extend(missing_issues)

        # Duplicate validation
        if check_duplicates:
            duplicate_issues = self.check_duplicates(df).issues
            issues.extend(duplicate_issues)

        # Outlier detection
        if check_outliers:
            outlier_issues = self.check_outliers(df).issues
            issues.extend(outlier_issues)

        # Custom validation rules
        custom_issues = self._run_custom_rules(df)
        issues.extend(custom_issues)

        # Determine overall validation status
        # Validation fails if there are any ERROR-level issues
        has_errors = any(issue.severity == ValidationSeverity.ERROR for issue in issues)
        is_valid = not has_errors

        return ValidationResult(
            is_valid=is_valid,
            issues=issues,
            metadata=metadata
        )

    def validate_schema(
        self,
        df: pd.DataFrame,
        schema: Dict[str, str]
    ) -> ValidationResult:
        """
        Validate DataFrame against a defined schema.

        This method checks that:
        1. All required columns are present
        2. No unexpected columns exist (optional warning)
        3. Column data types match the schema

        Schema validation is crucial for:
        - Ensuring data contracts between systems
        - Catching data pipeline errors early
        - Maintaining data consistency across environments

        Args:
            df: DataFrame to validate
            schema: Dictionary mapping column names to expected pandas dtypes
                   Examples: 'int64', 'float64', 'object', 'datetime64[ns]', 'bool'

        Returns:
            ValidationResult with schema validation findings

        Raises:
            ValueError: If schema is empty or invalid

        Example:
            >>> schema = {
            ...     'user_id': 'int64',
            ...     'name': 'object',
            ...     'age': 'int64',
            ...     'balance': 'float64',
            ...     'is_active': 'bool'
            ... }
            >>> result = validator.validate_schema(df, schema)
            >>>
            >>> # Check for specific schema violations
            >>> for issue in result.issues:
            ...     if issue.issue_type == 'missing_column':
            ...         print(f"Missing required column: {issue.column}")
        """
        if not schema:
            raise ValueError("Schema cannot be empty")

        issues: List[ValidationIssue] = []

        # Check for missing required columns
        required_columns = set(schema.keys())
        actual_columns = set(df.columns)
        missing_columns = required_columns - actual_columns

        for col in missing_columns:
            issues.append(ValidationIssue(
                column=col,
                issue_type='missing_column',
                severity=ValidationSeverity.ERROR,
                message=f"Required column '{col}' is missing from dataset",
                details={'expected_type': schema[col]}
            ))

        # Check for unexpected columns (informational)
        extra_columns = actual_columns - required_columns
        if extra_columns:
            issues.append(ValidationIssue(
                column=None,
                issue_type='extra_columns',
                severity=ValidationSeverity.INFO,
                message=f"Dataset contains {len(extra_columns)} unexpected columns",
                details={'extra_columns': list(extra_columns)}
            ))

        # Check data types for existing columns
        for col in required_columns.intersection(actual_columns):
            expected_type = schema[col]
            actual_type = str(df[col].dtype)

            # Handle type compatibility (e.g., int64 vs int32)
            if not self._types_compatible(actual_type, expected_type):
                issues.append(ValidationIssue(
                    column=col,
                    issue_type='type_mismatch',
                    severity=ValidationSeverity.ERROR,
                    message=f"Column '{col}' has type '{actual_type}', expected '{expected_type}'",
                    details={
                        'expected_type': expected_type,
                        'actual_type': actual_type
                    }
                ))

        has_errors = any(issue.severity == ValidationSeverity.ERROR for issue in issues)

        return ValidationResult(
            is_valid=not has_errors,
            issues=issues,
            metadata={'schema': schema}
        )

    def check_missing_values(
        self,
        df: pd.DataFrame,
        columns: Optional[List[str]] = None
    ) -> ValidationResult:
        """
        Check for missing values in DataFrame columns.

        This method identifies and quantifies missing values (NaN, None, NaT) in
        the dataset. Missing values can indicate:
        - Data collection issues
        - Integration problems
        - Optional fields in the data model

        The severity of missing value issues depends on the percentage:
        - Above threshold: WARNING
        - Below threshold: INFO

        Args:
            df: DataFrame to check
            columns: Optional list of specific columns to check.
                    If None, checks all columns.

        Returns:
            ValidationResult with missing value information

        Example:
            >>> result = validator.check_missing_values(df)
            >>>
            >>> # Find columns with most missing values
            >>> missing_issues = [
            ...     issue for issue in result.issues
            ...     if issue.issue_type == 'missing_values'
            ... ]
            >>> sorted_issues = sorted(
            ...     missing_issues,
            ...     key=lambda x: x.details['percentage'],
            ...     reverse=True
            ... )
            >>> for issue in sorted_issues[:5]:
            ...     print(f"{issue.column}: {issue.details['percentage']:.2f}% missing")
        """
        issues: List[ValidationIssue] = []

        # Determine which columns to check
        cols_to_check = columns if columns is not None else df.columns.tolist()

        # Check each column for missing values
        for col in cols_to_check:
            if col not in df.columns:
                issues.append(ValidationIssue(
                    column=col,
                    issue_type='column_not_found',
                    severity=ValidationSeverity.ERROR,
                    message=f"Column '{col}' not found in DataFrame",
                    details={}
                ))
                continue

            missing_count = df[col].isnull().sum()

            if missing_count > 0:
                missing_percentage = (missing_count / len(df)) * 100

                # Determine severity based on threshold
                if missing_percentage > self.missing_threshold:
                    severity = ValidationSeverity.WARNING
                else:
                    severity = ValidationSeverity.INFO

                issues.append(ValidationIssue(
                    column=col,
                    issue_type='missing_values',
                    severity=severity,
                    message=f"Column '{col}' has {missing_count} missing values ({missing_percentage:.2f}%)",
                    details={
                        'count': int(missing_count),
                        'percentage': float(missing_percentage),
                        'total_rows': len(df)
                    }
                ))

        # Add summary info
        total_missing = df.isnull().sum().sum()
        if total_missing > 0:
            total_cells = df.shape[0] * df.shape[1]
            overall_percentage = (total_missing / total_cells) * 100

            issues.append(ValidationIssue(
                column=None,
                issue_type='missing_values_summary',
                severity=ValidationSeverity.INFO,
                message=f"Dataset has {total_missing} total missing values ({overall_percentage:.2f}% of all cells)",
                details={
                    'total_missing': int(total_missing),
                    'total_cells': total_cells,
                    'percentage': float(overall_percentage)
                }
            ))

        has_errors = any(issue.severity == ValidationSeverity.ERROR for issue in issues)

        return ValidationResult(
            is_valid=not has_errors,
            issues=issues,
            metadata={'checked_columns': cols_to_check}
        )

    def check_duplicates(
        self,
        df: pd.DataFrame,
        subset: Optional[List[str]] = None,
        keep: str = 'first'
    ) -> ValidationResult:
        """
        Check for duplicate rows in the DataFrame.

        Duplicate rows can indicate:
        - Data ingestion errors (same record inserted multiple times)
        - Missing unique constraints in upstream systems
        - Intentional duplicates (e.g., time-series data)

        This method identifies duplicates and provides statistics to help
        understand the nature and extent of duplication.

        Args:
            df: DataFrame to check
            subset: Optional list of columns to consider for identifying duplicates.
                   If None, considers all columns.
            keep: Which duplicates to mark ('first', 'last', False).
                 - 'first': Mark duplicates except first occurrence as duplicates
                 - 'last': Mark duplicates except last occurrence as duplicates
                 - False: Mark all duplicates as duplicates

        Returns:
            ValidationResult with duplicate information

        Example:
            >>> # Check for complete duplicate rows
            >>> result = validator.check_duplicates(df)
            >>>
            >>> # Check for duplicates based on specific columns (e.g., user_id)
            >>> result = validator.check_duplicates(df, subset=['user_id', 'email'])
            >>>
            >>> # Get actual duplicate rows for inspection
            >>> if result.issues:
            ...     duplicates = df[df.duplicated(subset=['user_id'], keep=False)]
            ...     print(duplicates.head())
        """
        issues: List[ValidationIssue] = []

        # Validate subset columns
        if subset is not None:
            missing_cols = set(subset) - set(df.columns)
            if missing_cols:
                issues.append(ValidationIssue(
                    column=None,
                    issue_type='invalid_subset',
                    severity=ValidationSeverity.ERROR,
                    message=f"Subset columns not found: {missing_cols}",
                    details={'missing_columns': list(missing_cols)}
                ))
                return ValidationResult(is_valid=False, issues=issues, metadata={})

        # Find duplicates
        duplicate_mask = df.duplicated(subset=subset, keep=keep)
        duplicate_count = duplicate_mask.sum()

        if duplicate_count > 0:
            duplicate_percentage = (duplicate_count / len(df)) * 100

            # Determine severity based on threshold
            if duplicate_percentage > self.duplicate_threshold:
                severity = ValidationSeverity.WARNING
            else:
                severity = ValidationSeverity.INFO

            # Get sample of duplicate values
            duplicate_rows = df[duplicate_mask].head(5)

            issues.append(ValidationIssue(
                column=None,
                issue_type='duplicate_rows',
                severity=severity,
                message=f"Found {duplicate_count} duplicate rows ({duplicate_percentage:.2f}%)",
                details={
                    'count': int(duplicate_count),
                    'percentage': float(duplicate_percentage),
                    'total_rows': len(df),
                    'subset_columns': subset if subset else 'all columns',
                    'sample_indices': duplicate_rows.index.tolist()
                }
            ))

        has_errors = any(issue.severity == ValidationSeverity.ERROR for issue in issues)

        return ValidationResult(
            is_valid=not has_errors,
            issues=issues,
            metadata={'subset': subset, 'keep': keep}
        )

    def check_data_types(
        self,
        df: pd.DataFrame,
        expected_numeric: Optional[List[str]] = None,
        expected_categorical: Optional[List[str]] = None,
        expected_datetime: Optional[List[str]] = None
    ) -> ValidationResult:
        """
        Validate that columns have appropriate data types.

        This method checks that columns contain the expected types of data.
        It can also detect potential type issues like:
        - Numeric columns stored as strings
        - Date columns not parsed as datetime
        - High-cardinality object columns that might be IDs

        Args:
            df: DataFrame to validate
            expected_numeric: List of columns that should be numeric (int or float)
            expected_categorical: List of columns that should be categorical/object
            expected_datetime: List of columns that should be datetime

        Returns:
            ValidationResult with data type validation findings

        Example:
            >>> result = validator.check_data_types(
            ...     df,
            ...     expected_numeric=['age', 'salary', 'years_experience'],
            ...     expected_categorical=['department', 'city'],
            ...     expected_datetime=['hire_date', 'last_login']
            ... )
            >>>
            >>> # Find type mismatches
            >>> for issue in result.issues:
            ...     if issue.issue_type == 'type_mismatch':
            ...         print(f"{issue.column}: {issue.message}")
        """
        issues: List[ValidationIssue] = []

        # Check numeric columns
        if expected_numeric:
            for col in expected_numeric:
                if col not in df.columns:
                    issues.append(ValidationIssue(
                        column=col,
                        issue_type='column_not_found',
                        severity=ValidationSeverity.ERROR,
                        message=f"Expected numeric column '{col}' not found",
                        details={}
                    ))
                elif not pd.api.types.is_numeric_dtype(df[col]):
                    issues.append(ValidationIssue(
                        column=col,
                        issue_type='type_mismatch',
                        severity=ValidationSeverity.ERROR,
                        message=f"Column '{col}' should be numeric but is '{df[col].dtype}'",
                        details={
                            'expected': 'numeric',
                            'actual': str(df[col].dtype)
                        }
                    ))

        # Check categorical columns
        if expected_categorical:
            for col in expected_categorical:
                if col not in df.columns:
                    issues.append(ValidationIssue(
                        column=col,
                        issue_type='column_not_found',
                        severity=ValidationSeverity.ERROR,
                        message=f"Expected categorical column '{col}' not found",
                        details={}
                    ))
                elif not pd.api.types.is_object_dtype(df[col]) and not pd.api.types.is_categorical_dtype(df[col]):
                    issues.append(ValidationIssue(
                        column=col,
                        issue_type='type_mismatch',
                        severity=ValidationSeverity.WARNING,
                        message=f"Column '{col}' should be categorical but is '{df[col].dtype}'",
                        details={
                            'expected': 'categorical/object',
                            'actual': str(df[col].dtype)
                        }
                    ))

        # Check datetime columns
        if expected_datetime:
            for col in expected_datetime:
                if col not in df.columns:
                    issues.append(ValidationIssue(
                        column=col,
                        issue_type='column_not_found',
                        severity=ValidationSeverity.ERROR,
                        message=f"Expected datetime column '{col}' not found",
                        details={}
                    ))
                elif not pd.api.types.is_datetime64_any_dtype(df[col]):
                    issues.append(ValidationIssue(
                        column=col,
                        issue_type='type_mismatch',
                        severity=ValidationSeverity.ERROR,
                        message=f"Column '{col}' should be datetime but is '{df[col].dtype}'",
                        details={
                            'expected': 'datetime64',
                            'actual': str(df[col].dtype)
                        }
                    ))

        # Additional type checks: Identify potential issues
        for col in df.columns:
            # Check for numeric columns stored as object
            if pd.api.types.is_object_dtype(df[col]):
                # Try to convert to numeric
                try:
                    pd.to_numeric(df[col].dropna(), errors='raise')
                    issues.append(ValidationIssue(
                        column=col,
                        issue_type='numeric_as_object',
                        severity=ValidationSeverity.INFO,
                        message=f"Column '{col}' is object type but contains numeric values",
                        details={'suggestion': 'Consider converting to numeric type'}
                    ))
                except (ValueError, TypeError):
                    pass

            # Check for high-cardinality categorical columns
            if pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_categorical_dtype(df[col]):
                unique_ratio = df[col].nunique() / len(df)
                if unique_ratio > 0.5:  # More than 50% unique values
                    issues.append(ValidationIssue(
                        column=col,
                        issue_type='high_cardinality',
                        severity=ValidationSeverity.INFO,
                        message=f"Column '{col}' has high cardinality ({df[col].nunique()} unique values)",
                        details={
                            'unique_count': int(df[col].nunique()),
                            'unique_ratio': float(unique_ratio),
                            'suggestion': 'May be an ID field or require special encoding'
                        }
                    ))

        has_errors = any(issue.severity == ValidationSeverity.ERROR for issue in issues)

        return ValidationResult(
            is_valid=not has_errors,
            issues=issues,
            metadata={
                'expected_numeric': expected_numeric,
                'expected_categorical': expected_categorical,
                'expected_datetime': expected_datetime
            }
        )

    def check_outliers(
        self,
        df: pd.DataFrame,
        columns: Optional[List[str]] = None,
        method: Optional[str] = None,
        threshold: Optional[float] = None
    ) -> ValidationResult:
        """
        Detect outliers in numeric columns.

        Outliers can represent:
        - Data entry errors
        - Sensor malfunctions
        - Legitimate extreme values (heavy-tailed distributions)
        - Fraud or anomalous behavior

        This method supports multiple outlier detection techniques:

        1. IQR (Interquartile Range):
           - Most robust to extreme values
           - Works well for skewed distributions
           - Formula: Q1 - threshold * IQR, Q3 + threshold * IQR
           - Standard threshold: 1.5 (outliers), 3.0 (extreme outliers)

        2. Z-Score:
           - Assumes normal distribution
           - Based on standard deviations from mean
           - Formula: |value - mean| / std > threshold
           - Standard threshold: 3.0

        3. Isolation Forest:
           - Machine learning approach
           - Detects outliers in multi-dimensional space
           - Best for complex patterns
           - Requires sklearn

        Args:
            df: DataFrame to check
            columns: Optional list of columns to check. If None, checks all numeric columns.
            method: Outlier detection method ('iqr', 'zscore', 'isolation_forest').
                   If None, uses the instance's default method.
            threshold: Threshold for outlier detection. If None, uses the instance's default.

        Returns:
            ValidationResult with outlier information

        Example:
            >>> # Detect outliers using IQR method
            >>> result = validator.check_outliers(df, method='iqr', threshold=1.5)
            >>>
            >>> # Check specific columns for outliers
            >>> result = validator.check_outliers(
            ...     df,
            ...     columns=['age', 'salary', 'transaction_amount']
            ... )
            >>>
            >>> # Review outlier details
            >>> for issue in result.issues:
            ...     if issue.issue_type == 'outliers_detected':
            ...         print(f"{issue.column}: {issue.details['count']} outliers")
            ...         print(f"  Range: [{issue.details['min']}, {issue.details['max']}]")
        """
        issues: List[ValidationIssue] = []

        # Use instance defaults if not specified
        method = method or self.outlier_method
        threshold = threshold or self.outlier_threshold

        # Get numeric columns to check
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        cols_to_check = columns if columns is not None else numeric_cols

        # Validate that specified columns exist and are numeric
        for col in cols_to_check:
            if col not in df.columns:
                issues.append(ValidationIssue(
                    column=col,
                    issue_type='column_not_found',
                    severity=ValidationSeverity.ERROR,
                    message=f"Column '{col}' not found in DataFrame",
                    details={}
                ))
            elif col not in numeric_cols:
                issues.append(ValidationIssue(
                    column=col,
                    issue_type='non_numeric_column',
                    severity=ValidationSeverity.ERROR,
                    message=f"Column '{col}' is not numeric, cannot check for outliers",
                    details={'dtype': str(df[col].dtype)}
                ))

        # Remove invalid columns
        valid_cols = [col for col in cols_to_check if col in numeric_cols]

        # Detect outliers using specified method
        for col in valid_cols:
            data = df[col].dropna()

            if len(data) == 0:
                continue

            outliers_mask = self._detect_outliers(data, method, threshold)
            outlier_count = outliers_mask.sum()

            if outlier_count > 0:
                outlier_percentage = (outlier_count / len(data)) * 100
                outlier_values = data[outliers_mask]

                issues.append(ValidationIssue(
                    column=col,
                    issue_type='outliers_detected',
                    severity=ValidationSeverity.INFO,
                    message=f"Column '{col}' has {outlier_count} outliers ({outlier_percentage:.2f}%) using {method} method",
                    details={
                        'count': int(outlier_count),
                        'percentage': float(outlier_percentage),
                        'method': method,
                        'threshold': threshold,
                        'min': float(outlier_values.min()),
                        'max': float(outlier_values.max()),
                        'sample_values': outlier_values.head(10).tolist()
                    }
                ))

        has_errors = any(issue.severity == ValidationSeverity.ERROR for issue in issues)

        return ValidationResult(
            is_valid=not has_errors,
            issues=issues,
            metadata={
                'method': method,
                'threshold': threshold,
                'checked_columns': valid_cols
            }
        )

    def check_value_ranges(
        self,
        df: pd.DataFrame,
        ranges: Dict[str, Tuple[Optional[float], Optional[float]]]
    ) -> ValidationResult:
        """
        Validate that numeric columns fall within expected ranges.

        This method checks that values in specified columns are within
        acceptable bounds. This is useful for:
        - Business rule validation (e.g., age between 0 and 120)
        - Data sanity checks (e.g., probability between 0 and 1)
        - Domain constraints (e.g., temperature in valid range)

        Args:
            df: DataFrame to validate
            ranges: Dictionary mapping column names to (min, max) tuples.
                   Use None for unbounded min or max.
                   Examples:
                   - {'age': (0, 120)} - age between 0 and 120
                   - {'probability': (0.0, 1.0)} - probability between 0 and 1
                   - {'temperature': (-50, None)} - temperature at least -50

        Returns:
            ValidationResult with range validation findings

        Example:
            >>> ranges = {
            ...     'age': (0, 120),
            ...     'salary': (0, None),  # No upper bound
            ...     'discount_rate': (0.0, 1.0),
            ...     'temperature': (-273.15, None)  # Absolute zero minimum
            ... }
            >>> result = validator.check_value_ranges(df, ranges)
            >>>
            >>> # Find values outside acceptable ranges
            >>> for issue in result.issues:
            ...     if issue.issue_type == 'value_out_of_range':
            ...         print(f"{issue.column}: {issue.details['count']} values out of range")
        """
        issues: List[ValidationIssue] = []

        for col, (min_val, max_val) in ranges.items():
            if col not in df.columns:
                issues.append(ValidationIssue(
                    column=col,
                    issue_type='column_not_found',
                    severity=ValidationSeverity.ERROR,
                    message=f"Column '{col}' not found in DataFrame",
                    details={'expected_range': (min_val, max_val)}
                ))
                continue

            if not pd.api.types.is_numeric_dtype(df[col]):
                issues.append(ValidationIssue(
                    column=col,
                    issue_type='non_numeric_column',
                    severity=ValidationSeverity.ERROR,
                    message=f"Column '{col}' is not numeric, cannot check range",
                    details={'dtype': str(df[col].dtype)}
                ))
                continue

            # Check for values outside range
            data = df[col].dropna()
            out_of_range = pd.Series([False] * len(data), index=data.index)

            if min_val is not None:
                out_of_range |= data < min_val
            if max_val is not None:
                out_of_range |= data > max_val

            out_of_range_count = out_of_range.sum()

            if out_of_range_count > 0:
                out_of_range_percentage = (out_of_range_count / len(data)) * 100
                out_of_range_values = data[out_of_range]

                issues.append(ValidationIssue(
                    column=col,
                    issue_type='value_out_of_range',
                    severity=ValidationSeverity.ERROR,
                    message=f"Column '{col}' has {out_of_range_count} values outside range [{min_val}, {max_val}]",
                    details={
                        'count': int(out_of_range_count),
                        'percentage': float(out_of_range_percentage),
                        'expected_range': (min_val, max_val),
                        'actual_min': float(data.min()),
                        'actual_max': float(data.max()),
                        'sample_values': out_of_range_values.head(10).tolist()
                    }
                ))

        has_errors = any(issue.severity == ValidationSeverity.ERROR for issue in issues)

        return ValidationResult(
            is_valid=not has_errors,
            issues=issues,
            metadata={'ranges': ranges}
        )

    def check_categorical_values(
        self,
        df: pd.DataFrame,
        allowed_values: Dict[str, List[Any]]
    ) -> ValidationResult:
        """
        Validate that categorical columns contain only allowed values.

        This method checks that categorical columns contain values from a
        predefined set. This is essential for:
        - Enum/category validation
        - Detecting data entry errors
        - Ensuring data consistency
        - Catching upstream data issues

        Args:
            df: DataFrame to validate
            allowed_values: Dictionary mapping column names to lists of allowed values

        Returns:
            ValidationResult with categorical validation findings

        Example:
            >>> allowed_values = {
            ...     'status': ['active', 'inactive', 'pending'],
            ...     'priority': ['low', 'medium', 'high', 'critical'],
            ...     'department': ['sales', 'engineering', 'marketing', 'hr']
            ... }
            >>> result = validator.check_categorical_values(df, allowed_values)
            >>>
            >>> # Find invalid values
            >>> for issue in result.issues:
            ...     if issue.issue_type == 'invalid_categorical_value':
            ...         print(f"{issue.column} has invalid values:")
            ...         print(f"  Found: {issue.details['invalid_values']}")
        """
        issues: List[ValidationIssue] = []

        for col, allowed in allowed_values.items():
            if col not in df.columns:
                issues.append(ValidationIssue(
                    column=col,
                    issue_type='column_not_found',
                    severity=ValidationSeverity.ERROR,
                    message=f"Column '{col}' not found in DataFrame",
                    details={'allowed_values': allowed}
                ))
                continue

            # Get actual values (excluding NaN)
            actual_values = set(df[col].dropna().unique())
            allowed_set = set(allowed)

            # Find invalid values
            invalid_values = actual_values - allowed_set

            if invalid_values:
                invalid_count = df[col].isin(invalid_values).sum()
                invalid_percentage = (invalid_count / len(df)) * 100

                issues.append(ValidationIssue(
                    column=col,
                    issue_type='invalid_categorical_value',
                    severity=ValidationSeverity.ERROR,
                    message=f"Column '{col}' contains {len(invalid_values)} invalid value(s)",
                    details={
                        'invalid_values': list(invalid_values),
                        'allowed_values': allowed,
                        'count': int(invalid_count),
                        'percentage': float(invalid_percentage)
                    }
                ))

        has_errors = any(issue.severity == ValidationSeverity.ERROR for issue in issues)

        return ValidationResult(
            is_valid=not has_errors,
            issues=issues,
            metadata={'allowed_values': allowed_values}
        )

    def add_custom_rule(
        self,
        name: str,
        validation_func: Callable[[pd.DataFrame], bool],
        error_message: str
    ) -> None:
        """
        Add a custom validation rule.

        Custom rules allow you to implement domain-specific validation logic
        that goes beyond the built-in validators. The validation function
        should return True if validation passes, False otherwise.

        Args:
            name: Unique name for the validation rule
            validation_func: Function that takes a DataFrame and returns bool
                            True = validation passed, False = validation failed
            error_message: Message to display when validation fails

        Example:
            >>> # Ensure age is always greater than years of experience
            >>> validator.add_custom_rule(
            ...     'age_vs_experience',
            ...     lambda df: (df['age'] >= df['years_experience']).all(),
            ...     'Age must be greater than or equal to years of experience'
            ... )
            >>>
            >>> # Ensure email domain is from allowed list
            >>> def check_email_domain(df):
            ...     allowed_domains = ['company.com', 'partner.com']
            ...     domains = df['email'].str.extract(r'@(.+)$')[0]
            ...     return domains.isin(allowed_domains).all()
            >>>
            >>> validator.add_custom_rule(
            ...     'email_domain',
            ...     check_email_domain,
            ...     'Email must be from allowed domains'
            ... )
            >>>
            >>> # Run validation with custom rules
            >>> result = validator.validate(df)
        """
        self.custom_rules.append((name, validation_func, error_message))

    def remove_custom_rule(self, name: str) -> bool:
        """
        Remove a custom validation rule by name.

        Args:
            name: Name of the rule to remove

        Returns:
            True if rule was found and removed, False otherwise

        Example:
            >>> validator.remove_custom_rule('age_vs_experience')
        """
        for i, (rule_name, _, _) in enumerate(self.custom_rules):
            if rule_name == name:
                self.custom_rules.pop(i)
                return True
        return False

    def generate_validation_report(
        self,
        result: ValidationResult,
        output_format: str = 'text',
        output_file: Optional[str] = None
    ) -> str:
        """
        Generate a comprehensive validation report.

        This method creates a detailed, human-readable report of validation
        results. The report includes:
        - Summary statistics
        - Issues grouped by severity
        - Detailed information for each issue
        - Metadata about the validation run

        Args:
            result: ValidationResult to generate report from
            output_format: Format of the report ('text', 'html', 'json')
            output_file: Optional file path to save the report

        Returns:
            Report as a string

        Example:
            >>> result = validator.validate(df)
            >>>
            >>> # Generate text report
            >>> report = validator.generate_validation_report(result)
            >>> print(report)
            >>>
            >>> # Save HTML report to file
            >>> report = validator.generate_validation_report(
            ...     result,
            ...     output_format='html',
            ...     output_file='validation_report.html'
            ... )
            >>>
            >>> # Generate JSON report for programmatic processing
            >>> json_report = validator.generate_validation_report(
            ...     result,
            ...     output_format='json'
            ... )
        """
        if output_format == 'text':
            report = self._generate_text_report(result)
        elif output_format == 'html':
            report = self._generate_html_report(result)
        elif output_format == 'json':
            report = self._generate_json_report(result)
        else:
            raise ValueError(f"Unsupported output format: {output_format}")

        # Save to file if requested
        if output_file:
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(report, encoding='utf-8')

        return report

    # Private helper methods

    def _types_compatible(self, actual: str, expected: str) -> bool:
        """
        Check if two pandas dtypes are compatible.

        This method handles type compatibility beyond exact matches, such as:
        - int64 vs int32
        - float64 vs float32
        - datetime64[ns] variations

        Args:
            actual: Actual data type as string
            expected: Expected data type as string

        Returns:
            True if types are compatible, False otherwise
        """
        # Exact match
        if actual == expected:
            return True

        # Integer type compatibility
        if 'int' in expected and 'int' in actual:
            return True

        # Float type compatibility
        if 'float' in expected and 'float' in actual:
            return True

        # Datetime compatibility
        if 'datetime64' in expected and 'datetime64' in actual:
            return True

        return False

    def _detect_outliers(
        self,
        data: pd.Series,
        method: str,
        threshold: float
    ) -> pd.Series:
        """
        Detect outliers in a numeric series using the specified method.

        Args:
            data: Numeric series to check
            method: Detection method ('iqr', 'zscore', 'isolation_forest')
            threshold: Threshold for detection

        Returns:
            Boolean series with True for outliers
        """
        if method == 'iqr':
            Q1 = data.quantile(0.25)
            Q3 = data.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            return (data < lower_bound) | (data > upper_bound)

        elif method == 'zscore':
            mean = data.mean()
            std = data.std()
            if std == 0:
                return pd.Series([False] * len(data), index=data.index)
            z_scores = np.abs((data - mean) / std)
            return z_scores > threshold

        elif method == 'isolation_forest':
            try:
                from sklearn.ensemble import IsolationForest

                # Reshape data for sklearn
                X = data.values.reshape(-1, 1)

                # Fit isolation forest
                iso_forest = IsolationForest(
                    contamination=0.1,  # Expected proportion of outliers
                    random_state=42
                )
                predictions = iso_forest.fit_predict(X)

                # -1 indicates outlier, 1 indicates inlier
                return pd.Series(predictions == -1, index=data.index)

            except ImportError:
                warnings.warn(
                    "sklearn not available. Falling back to IQR method for outlier detection."
                )
                return self._detect_outliers(data, 'iqr', 1.5)

        else:
            raise ValueError(f"Unknown outlier detection method: {method}")

    def _run_custom_rules(self, df: pd.DataFrame) -> List[ValidationIssue]:
        """
        Run all custom validation rules.

        Args:
            df: DataFrame to validate

        Returns:
            List of validation issues from custom rules
        """
        issues: List[ValidationIssue] = []

        for name, validation_func, error_message in self.custom_rules:
            try:
                is_valid = validation_func(df)

                if not is_valid:
                    issues.append(ValidationIssue(
                        column=None,
                        issue_type='custom_rule_violation',
                        severity=ValidationSeverity.ERROR,
                        message=f"Custom rule '{name}' failed: {error_message}",
                        details={'rule_name': name}
                    ))

            except Exception as e:
                issues.append(ValidationIssue(
                    column=None,
                    issue_type='custom_rule_error',
                    severity=ValidationSeverity.ERROR,
                    message=f"Custom rule '{name}' raised an exception: {str(e)}",
                    details={'rule_name': name, 'exception': str(e)}
                ))

        return issues

    def _generate_text_report(self, result: ValidationResult) -> str:
        """Generate a text-based validation report."""
        lines = []
        lines.append("=" * 80)
        lines.append("DATA VALIDATION REPORT")
        lines.append("=" * 80)
        lines.append("")

        # Summary
        lines.append(result.get_summary())
        lines.append("")

        # Metadata
        if result.metadata:
            lines.append("Dataset Information:")
            lines.append("-" * 40)
            if 'timestamp' in result.metadata:
                lines.append(f"Validation Time: {result.metadata['timestamp']}")
            if 'dataset_shape' in result.metadata:
                rows, cols = result.metadata['dataset_shape']
                lines.append(f"Dataset Shape: {rows} rows × {cols} columns")
            if 'memory_usage_mb' in result.metadata:
                lines.append(f"Memory Usage: {result.metadata['memory_usage_mb']:.2f} MB")
            lines.append("")

        # Issues by severity
        errors = result.get_errors()
        if errors:
            lines.append(f"ERRORS ({len(errors)}):")
            lines.append("-" * 40)
            for issue in errors:
                lines.append(f"  [{issue.issue_type}] {issue.message}")
                if issue.column:
                    lines.append(f"    Column: {issue.column}")
                if issue.details:
                    for key, value in issue.details.items():
                        if key not in ['sample_values', 'sample_indices']:
                            lines.append(f"    {key}: {value}")
            lines.append("")

        warnings_list = result.get_warnings()
        if warnings_list:
            lines.append(f"WARNINGS ({len(warnings_list)}):")
            lines.append("-" * 40)
            for issue in warnings_list:
                lines.append(f"  [{issue.issue_type}] {issue.message}")
                if issue.column:
                    lines.append(f"    Column: {issue.column}")
            lines.append("")

        info_list = result.get_info()
        if info_list:
            lines.append(f"INFORMATIONAL ({len(info_list)}):")
            lines.append("-" * 40)
            for issue in info_list[:10]:  # Limit to first 10
                lines.append(f"  [{issue.issue_type}] {issue.message}")
            if len(info_list) > 10:
                lines.append(f"  ... and {len(info_list) - 10} more")
            lines.append("")

        lines.append("=" * 80)

        return "\n".join(lines)

    def _generate_html_report(self, result: ValidationResult) -> str:
        """Generate an HTML validation report."""
        html = []
        html.append("<!DOCTYPE html>")
        html.append("<html><head>")
        html.append("<title>Data Validation Report</title>")
        html.append("<style>")
        html.append("body { font-family: Arial, sans-serif; margin: 20px; }")
        html.append("h1 { color: #333; }")
        html.append(".summary { background: #f0f0f0; padding: 15px; border-radius: 5px; }")
        html.append(".error { color: #d32f2f; }")
        html.append(".warning { color: #f57c00; }")
        html.append(".info { color: #0288d1; }")
        html.append(".issue { margin: 10px 0; padding: 10px; border-left: 3px solid #ccc; }")
        html.append("</style>")
        html.append("</head><body>")

        html.append("<h1>Data Validation Report</h1>")

        # Summary
        html.append(f"<div class='summary'>")
        html.append(f"<h2>Summary</h2>")
        html.append(f"<p><strong>Status:</strong> {'PASSED' if result.is_valid else 'FAILED'}</p>")
        html.append(f"<p><strong>Total Issues:</strong> {len(result.issues)}</p>")
        html.append(f"<p><strong>Errors:</strong> {len(result.get_errors())}</p>")
        html.append(f"<p><strong>Warnings:</strong> {len(result.get_warnings())}</p>")
        html.append(f"<p><strong>Info:</strong> {len(result.get_info())}</p>")
        html.append("</div>")

        # Errors
        errors = result.get_errors()
        if errors:
            html.append("<h2 class='error'>Errors</h2>")
            for issue in errors:
                html.append(f"<div class='issue error'>")
                html.append(f"<strong>[{issue.issue_type}]</strong> {issue.message}")
                if issue.column:
                    html.append(f"<br><em>Column: {issue.column}</em>")
                html.append("</div>")

        # Warnings
        warnings_list = result.get_warnings()
        if warnings_list:
            html.append("<h2 class='warning'>Warnings</h2>")
            for issue in warnings_list:
                html.append(f"<div class='issue warning'>")
                html.append(f"<strong>[{issue.issue_type}]</strong> {issue.message}")
                if issue.column:
                    html.append(f"<br><em>Column: {issue.column}</em>")
                html.append("</div>")

        html.append("</body></html>")

        return "\n".join(html)

    def _generate_json_report(self, result: ValidationResult) -> str:
        """Generate a JSON validation report."""
        import json

        report = {
            'is_valid': result.is_valid,
            'summary': {
                'total_issues': len(result.issues),
                'errors': len(result.get_errors()),
                'warnings': len(result.get_warnings()),
                'info': len(result.get_info())
            },
            'metadata': result.metadata,
            'issues': [
                {
                    'column': issue.column,
                    'issue_type': issue.issue_type,
                    'severity': issue.severity.value,
                    'message': issue.message,
                    'details': issue.details
                }
                for issue in result.issues
            ]
        }

        return json.dumps(report, indent=2, default=str)
