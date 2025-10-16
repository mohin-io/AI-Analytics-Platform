"""
Unit tests for the DataValidator class

This test suite demonstrates how to test the DataValidator functionality.
It covers the main validation methods and edge cases.
"""

import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.preprocessing.data_validator import (
    DataValidator,
    ValidationResult,
    ValidationIssue,
    ValidationSeverity
)


@pytest.fixture
def sample_df():
    """Create a sample DataFrame for testing."""
    return pd.DataFrame({
        'id': [1, 2, 3, 4, 5],
        'name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve'],
        'age': [25, 30, 35, 40, 45],
        'salary': [50000.0, 60000.0, 70000.0, 80000.0, 90000.0],
        'department': ['Sales', 'Engineering', 'Sales', 'HR', 'Engineering']
    })


@pytest.fixture
def df_with_issues():
    """Create a DataFrame with various data quality issues."""
    return pd.DataFrame({
        'id': [1, 2, 3, 4, 5, 5],  # Duplicate ID
        'age': [25, 30, np.nan, 40, 150, 45],  # Missing value and outlier
        'salary': [50000, 60000, 70000, 80000, 90000, 90000],
        'status': ['active', 'active', 'inactive', 'invalid', 'active', 'active']  # Invalid value
    })


class TestDataValidator:
    """Test suite for DataValidator class."""

    def test_initialization(self):
        """Test DataValidator initialization with custom parameters."""
        validator = DataValidator(
            missing_threshold=5.0,
            duplicate_threshold=2.0,
            outlier_method='zscore',
            outlier_threshold=3.0
        )

        assert validator.missing_threshold == 5.0
        assert validator.duplicate_threshold == 2.0
        assert validator.outlier_method == 'zscore'
        assert validator.outlier_threshold == 3.0
        assert len(validator.custom_rules) == 0

    def test_validate_clean_data(self, sample_df):
        """Test validation on clean data with no issues."""
        validator = DataValidator()
        result = validator.validate(sample_df)

        assert isinstance(result, ValidationResult)
        assert result.is_valid is True
        assert len(result.get_errors()) == 0

    def test_validate_schema_success(self, sample_df):
        """Test successful schema validation."""
        schema = {
            'id': 'int64',
            'name': 'object',
            'age': 'int64',
            'salary': 'float64',
            'department': 'object'
        }

        validator = DataValidator()
        result = validator.validate_schema(sample_df, schema)

        assert result.is_valid is True
        assert len(result.get_errors()) == 0

    def test_validate_schema_missing_column(self, sample_df):
        """Test schema validation with missing column."""
        schema = {
            'id': 'int64',
            'name': 'object',
            'email': 'object'  # This column doesn't exist
        }

        validator = DataValidator()
        result = validator.validate_schema(sample_df, schema)

        assert result.is_valid is False
        errors = result.get_errors()
        assert len(errors) > 0
        assert any('email' in error.message for error in errors)

    def test_validate_schema_type_mismatch(self, sample_df):
        """Test schema validation with type mismatch."""
        schema = {
            'id': 'int64',
            'age': 'object'  # age is actually int64, not object
        }

        validator = DataValidator()
        result = validator.validate_schema(sample_df, schema)

        assert result.is_valid is False
        errors = result.get_errors()
        assert any('type' in error.issue_type for error in errors)

    def test_check_missing_values(self, df_with_issues):
        """Test missing value detection."""
        validator = DataValidator(missing_threshold=0.0)
        result = validator.check_missing_values(df_with_issues)

        # Should find missing value in 'age' column
        missing_issues = [i for i in result.issues if i.issue_type == 'missing_values']
        assert len(missing_issues) > 0
        assert any(issue.column == 'age' for issue in missing_issues)

    def test_check_duplicates(self, df_with_issues):
        """Test duplicate detection."""
        validator = DataValidator(duplicate_threshold=0.0)
        result = validator.check_duplicates(df_with_issues)

        # Should find duplicates
        duplicate_issues = [i for i in result.issues if i.issue_type == 'duplicate_rows']
        assert len(duplicate_issues) > 0

    def test_check_duplicates_subset(self):
        """Test duplicate detection with subset of columns."""
        df = pd.DataFrame({
            'id': [1, 2, 3],
            'name': ['Alice', 'Alice', 'Bob'],
            'email': ['alice@a.com', 'alice@b.com', 'bob@c.com']
        })

        validator = DataValidator()
        result = validator.check_duplicates(df, subset=['name'])

        # Should find duplicates based on 'name' column
        duplicate_issues = [i for i in result.issues if i.issue_type == 'duplicate_rows']
        assert len(duplicate_issues) > 0

    def test_check_data_types(self, sample_df):
        """Test data type validation."""
        validator = DataValidator()
        result = validator.check_data_types(
            sample_df,
            expected_numeric=['id', 'age', 'salary'],
            expected_categorical=['name', 'department']
        )

        assert result.is_valid is True

    def test_check_data_types_invalid(self, sample_df):
        """Test data type validation with incorrect types."""
        validator = DataValidator()
        result = validator.check_data_types(
            sample_df,
            expected_numeric=['name'],  # name is object, not numeric
        )

        assert result.is_valid is False
        errors = result.get_errors()
        assert any('type_mismatch' in error.issue_type for error in errors)

    def test_check_outliers_iqr(self, df_with_issues):
        """Test outlier detection using IQR method."""
        validator = DataValidator(outlier_method='iqr', outlier_threshold=1.5)
        result = validator.check_outliers(df_with_issues, columns=['age'])

        # Should detect age=150 as outlier
        outlier_issues = [i for i in result.issues if i.issue_type == 'outliers_detected']
        assert len(outlier_issues) > 0

    def test_check_outliers_zscore(self):
        """Test outlier detection using Z-score method."""
        df = pd.DataFrame({
            'value': [1, 2, 3, 4, 5, 100]  # 100 is clear outlier
        })

        validator = DataValidator(outlier_method='zscore', outlier_threshold=2.0)
        result = validator.check_outliers(df, columns=['value'])

        outlier_issues = [i for i in result.issues if i.issue_type == 'outliers_detected']
        assert len(outlier_issues) > 0

    def test_check_value_ranges(self):
        """Test value range validation."""
        df = pd.DataFrame({
            'age': [25, 30, 150, 40],  # 150 is out of range
            'probability': [0.5, 0.8, 1.5, 0.3]  # 1.5 is out of range
        })

        validator = DataValidator()
        ranges = {
            'age': (0, 120),
            'probability': (0.0, 1.0)
        }
        result = validator.check_value_ranges(df, ranges)

        assert result.is_valid is False
        errors = result.get_errors()
        assert len(errors) == 2  # Both columns have out-of-range values

    def test_check_categorical_values(self, df_with_issues):
        """Test categorical value validation."""
        validator = DataValidator()
        allowed_values = {
            'status': ['active', 'inactive']
        }
        result = validator.check_categorical_values(df_with_issues, allowed_values)

        assert result.is_valid is False
        errors = result.get_errors()
        assert any('invalid_categorical_value' in error.issue_type for error in errors)

    def test_add_custom_rule_pass(self, sample_df):
        """Test adding a custom rule that passes."""
        validator = DataValidator()

        validator.add_custom_rule(
            'age_positive',
            lambda df: (df['age'] > 0).all(),
            'All ages must be positive'
        )

        result = validator.validate(sample_df)
        assert result.is_valid is True

    def test_add_custom_rule_fail(self, sample_df):
        """Test adding a custom rule that fails."""
        validator = DataValidator()

        validator.add_custom_rule(
            'age_under_30',
            lambda df: (df['age'] < 30).all(),
            'All ages must be under 30'
        )

        result = validator.validate(sample_df)
        assert result.is_valid is False
        errors = result.get_errors()
        assert any('custom_rule_violation' in error.issue_type for error in errors)

    def test_remove_custom_rule(self, sample_df):
        """Test removing a custom rule."""
        validator = DataValidator()

        validator.add_custom_rule(
            'test_rule',
            lambda df: False,
            'This rule always fails'
        )

        # Rule should fail validation
        result = validator.validate(sample_df)
        assert result.is_valid is False

        # Remove the rule
        removed = validator.remove_custom_rule('test_rule')
        assert removed is True

        # Now validation should pass
        result = validator.validate(sample_df)
        assert result.is_valid is True

    def test_generate_text_report(self, sample_df):
        """Test text report generation."""
        validator = DataValidator()
        result = validator.validate(sample_df)

        report = validator.generate_validation_report(result, output_format='text')

        assert isinstance(report, str)
        assert 'DATA VALIDATION REPORT' in report
        assert 'Validation Status' in report

    def test_generate_html_report(self, sample_df):
        """Test HTML report generation."""
        validator = DataValidator()
        result = validator.validate(sample_df)

        report = validator.generate_validation_report(result, output_format='html')

        assert isinstance(report, str)
        assert '<!DOCTYPE html>' in report
        assert 'Data Validation Report' in report

    def test_generate_json_report(self, sample_df):
        """Test JSON report generation."""
        import json

        validator = DataValidator()
        result = validator.validate(sample_df)

        report = validator.generate_validation_report(result, output_format='json')

        assert isinstance(report, str)
        # Should be valid JSON
        report_data = json.loads(report)
        assert 'is_valid' in report_data
        assert 'summary' in report_data
        assert 'issues' in report_data

    def test_validation_result_methods(self, df_with_issues):
        """Test ValidationResult helper methods."""
        validator = DataValidator()
        result = validator.validate(df_with_issues)

        # Test getter methods
        errors = result.get_errors()
        warnings = result.get_warnings()
        info = result.get_info()

        assert isinstance(errors, list)
        assert isinstance(warnings, list)
        assert isinstance(info, list)

        # Test summary
        summary = result.get_summary()
        assert isinstance(summary, str)
        assert 'Validation Status' in summary

    def test_empty_dataframe(self):
        """Test validation on empty DataFrame."""
        df = pd.DataFrame()
        validator = DataValidator()

        result = validator.validate(df)
        assert isinstance(result, ValidationResult)

    def test_single_row_dataframe(self):
        """Test validation on DataFrame with single row."""
        df = pd.DataFrame({'a': [1], 'b': [2]})
        validator = DataValidator()

        result = validator.validate(df)
        assert isinstance(result, ValidationResult)

    def test_validation_with_all_checks_disabled(self, sample_df):
        """Test validation with all checks disabled."""
        validator = DataValidator()
        result = validator.validate(
            sample_df,
            check_outliers=False,
            check_duplicates=False,
            check_missing=False,
            check_types=False
        )

        assert isinstance(result, ValidationResult)
        assert result.is_valid is True

    def test_severity_levels(self, df_with_issues):
        """Test that issues have correct severity levels."""
        validator = DataValidator(missing_threshold=0.0)
        result = validator.validate(df_with_issues)

        # Check that severity is properly assigned
        for issue in result.issues:
            assert isinstance(issue.severity, ValidationSeverity)
            assert issue.severity in [
                ValidationSeverity.ERROR,
                ValidationSeverity.WARNING,
                ValidationSeverity.INFO
            ]


class TestValidationIssue:
    """Test ValidationIssue data class."""

    def test_validation_issue_creation(self):
        """Test creating a ValidationIssue."""
        issue = ValidationIssue(
            column='age',
            issue_type='missing_values',
            severity=ValidationSeverity.WARNING,
            message='Column has missing values',
            details={'count': 5}
        )

        assert issue.column == 'age'
        assert issue.issue_type == 'missing_values'
        assert issue.severity == ValidationSeverity.WARNING
        assert issue.message == 'Column has missing values'
        assert issue.details['count'] == 5


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_invalid_schema(self, sample_df):
        """Test validation with invalid schema."""
        validator = DataValidator()

        with pytest.raises(ValueError):
            validator.validate_schema(sample_df, {})  # Empty schema

    def test_invalid_output_format(self, sample_df):
        """Test report generation with invalid format."""
        validator = DataValidator()
        result = validator.validate(sample_df)

        with pytest.raises(ValueError):
            validator.generate_validation_report(result, output_format='invalid')

    def test_check_missing_values_nonexistent_column(self, sample_df):
        """Test missing value check on non-existent column."""
        validator = DataValidator()
        result = validator.check_missing_values(sample_df, columns=['nonexistent'])

        errors = result.get_errors()
        assert len(errors) > 0
        assert any('not found' in error.message for error in errors)

    def test_check_outliers_non_numeric_column(self, sample_df):
        """Test outlier detection on non-numeric column."""
        validator = DataValidator()
        result = validator.check_outliers(sample_df, columns=['name'])

        errors = result.get_errors()
        assert len(errors) > 0
        assert any('not numeric' in error.message for error in errors)

    def test_custom_rule_exception(self, sample_df):
        """Test custom rule that raises an exception."""
        validator = DataValidator()

        def bad_rule(df):
            raise ValueError("This rule crashes")

        validator.add_custom_rule('bad_rule', bad_rule, 'This will fail')

        result = validator.validate(sample_df)
        errors = result.get_errors()
        assert any('custom_rule_error' in error.issue_type for error in errors)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
