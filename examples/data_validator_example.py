"""
Example usage of the DataValidator class

This script demonstrates various validation scenarios using the DataValidator
from the Unified AI Analytics Platform.
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.preprocessing.data_validator import DataValidator, ValidationSeverity


def create_sample_data():
    """Create a sample dataset with various data quality issues."""
    np.random.seed(42)

    data = {
        'user_id': range(1, 101),
        'name': ['User' + str(i) for i in range(1, 101)],
        'age': [np.random.randint(18, 80) if i % 10 != 0 else np.nan for i in range(100)],
        'salary': [np.random.uniform(30000, 150000) if i % 15 != 0 else np.nan for i in range(100)],
        'department': np.random.choice(['Sales', 'Engineering', 'Marketing', 'HR', 'InvalidDept'], 100),
        'years_experience': [np.random.randint(0, 30) for i in range(100)],
        'satisfaction_score': [np.random.uniform(0, 1) for i in range(100)],
        'hire_date': pd.date_range('2010-01-01', periods=100, freq='W')
    }

    df = pd.DataFrame(data)

    # Add some outliers to salary
    df.loc[0, 'salary'] = 500000  # Outlier
    df.loc[1, 'salary'] = 15000   # Outlier

    # Add some duplicate rows
    df.loc[100] = df.loc[50].copy()
    df.loc[101] = df.loc[50].copy()

    # Add some out-of-range values
    df.loc[2, 'satisfaction_score'] = 1.5  # Should be between 0 and 1
    df.loc[3, 'age'] = -5  # Invalid age

    return df


def example_1_basic_validation():
    """Example 1: Basic validation with all checks."""
    print("\n" + "="*80)
    print("EXAMPLE 1: Basic Validation")
    print("="*80)

    df = create_sample_data()
    validator = DataValidator()

    # Run comprehensive validation
    result = validator.validate(df)

    print(result.get_summary())

    # Display errors
    if result.get_errors():
        print("\nErrors found:")
        for error in result.get_errors():
            print(f"  - {error.message}")

    # Display warnings
    if result.get_warnings():
        print("\nWarnings found:")
        for warning in result.get_warnings()[:5]:  # Show first 5
            print(f"  - {warning.message}")


def example_2_schema_validation():
    """Example 2: Validate against a schema."""
    print("\n" + "="*80)
    print("EXAMPLE 2: Schema Validation")
    print("="*80)

    df = create_sample_data()

    # Define expected schema
    schema = {
        'user_id': 'int64',
        'name': 'object',
        'age': 'float64',  # Note: nullable integers become float64
        'salary': 'float64',
        'department': 'object',
        'years_experience': 'int64',
        'satisfaction_score': 'float64',
        'hire_date': 'datetime64[ns]'
    }

    validator = DataValidator()
    result = validator.validate_schema(df, schema)

    print(f"Schema validation: {'PASSED' if result.is_valid else 'FAILED'}")

    if not result.is_valid:
        for issue in result.issues:
            print(f"  - {issue.message}")


def example_3_missing_values():
    """Example 3: Check for missing values."""
    print("\n" + "="*80)
    print("EXAMPLE 3: Missing Value Detection")
    print("="*80)

    df = create_sample_data()

    # Create validator with threshold
    validator = DataValidator(missing_threshold=5.0)  # Warn if > 5% missing

    result = validator.check_missing_values(df)

    print(f"Total issues: {len(result.issues)}")

    # Show missing value statistics
    for issue in result.issues:
        if issue.issue_type == 'missing_values':
            print(f"\nColumn: {issue.column}")
            print(f"  Missing: {issue.details['count']} ({issue.details['percentage']:.2f}%)")
            print(f"  Severity: {issue.severity.value}")


def example_4_duplicate_detection():
    """Example 4: Detect duplicate rows."""
    print("\n" + "="*80)
    print("EXAMPLE 4: Duplicate Detection")
    print("="*80)

    df = create_sample_data()
    validator = DataValidator()

    # Check for complete duplicates
    result = validator.check_duplicates(df)

    for issue in result.issues:
        if issue.issue_type == 'duplicate_rows':
            print(f"Found {issue.details['count']} duplicate rows")
            print(f"Duplicate indices: {issue.details['sample_indices']}")

    # Check for duplicates based on specific columns
    result = validator.check_duplicates(df, subset=['name', 'department'])

    if result.issues:
        print(f"\nDuplicates based on name+department: {result.issues[0].details['count']}")


def example_5_outlier_detection():
    """Example 5: Detect outliers using different methods."""
    print("\n" + "="*80)
    print("EXAMPLE 5: Outlier Detection")
    print("="*80)

    df = create_sample_data()

    # IQR method (default)
    validator_iqr = DataValidator(outlier_method='iqr', outlier_threshold=1.5)
    result_iqr = validator_iqr.check_outliers(df, columns=['salary'])

    print("IQR Method:")
    for issue in result_iqr.issues:
        if issue.issue_type == 'outliers_detected':
            print(f"  Column: {issue.column}")
            print(f"  Outliers: {issue.details['count']} ({issue.details['percentage']:.2f}%)")
            print(f"  Range: [{issue.details['min']:.2f}, {issue.details['max']:.2f}]")

    # Z-score method
    validator_zscore = DataValidator(outlier_method='zscore', outlier_threshold=3.0)
    result_zscore = validator_zscore.check_outliers(df, columns=['salary'])

    print("\nZ-Score Method:")
    for issue in result_zscore.issues:
        if issue.issue_type == 'outliers_detected':
            print(f"  Column: {issue.column}")
            print(f"  Outliers: {issue.details['count']} ({issue.details['percentage']:.2f}%)")


def example_6_value_ranges():
    """Example 6: Validate value ranges."""
    print("\n" + "="*80)
    print("EXAMPLE 6: Value Range Validation")
    print("="*80)

    df = create_sample_data()
    validator = DataValidator()

    # Define acceptable ranges
    ranges = {
        'age': (0, 120),
        'salary': (0, None),  # No upper bound
        'satisfaction_score': (0.0, 1.0),
        'years_experience': (0, 50)
    }

    result = validator.check_value_ranges(df, ranges)

    print(f"Range validation: {'PASSED' if result.is_valid else 'FAILED'}")

    for issue in result.issues:
        if issue.issue_type == 'value_out_of_range':
            print(f"\nColumn: {issue.column}")
            print(f"  Expected: {issue.details['expected_range']}")
            print(f"  Actual: [{issue.details['actual_min']:.2f}, {issue.details['actual_max']:.2f}]")
            print(f"  Out of range: {issue.details['count']} values")


def example_7_categorical_validation():
    """Example 7: Validate categorical values."""
    print("\n" + "="*80)
    print("EXAMPLE 7: Categorical Value Validation")
    print("="*80)

    df = create_sample_data()
    validator = DataValidator()

    # Define allowed values
    allowed_values = {
        'department': ['Sales', 'Engineering', 'Marketing', 'HR']
    }

    result = validator.check_categorical_values(df, allowed_values)

    print(f"Categorical validation: {'PASSED' if result.is_valid else 'FAILED'}")

    for issue in result.issues:
        if issue.issue_type == 'invalid_categorical_value':
            print(f"\nColumn: {issue.column}")
            print(f"  Invalid values found: {issue.details['invalid_values']}")
            print(f"  Count: {issue.details['count']}")


def example_8_custom_rules():
    """Example 8: Add custom validation rules."""
    print("\n" + "="*80)
    print("EXAMPLE 8: Custom Validation Rules")
    print("="*80)

    df = create_sample_data()
    validator = DataValidator()

    # Rule 1: Age should be greater than years of experience
    validator.add_custom_rule(
        'age_vs_experience',
        lambda df: (df['age'].fillna(0) >= df['years_experience']).all(),
        'Age must be greater than or equal to years of experience'
    )

    # Rule 2: High earners should have more experience
    def high_earner_rule(df):
        high_earners = df[df['salary'] > 100000].dropna(subset=['salary'])
        if len(high_earners) > 0:
            return (high_earners['years_experience'] >= 5).all()
        return True

    validator.add_custom_rule(
        'high_earner_experience',
        high_earner_rule,
        'Employees with salary > 100K should have at least 5 years experience'
    )

    # Run validation with custom rules
    result = validator.validate(df)

    print(f"Validation with custom rules: {'PASSED' if result.is_valid else 'FAILED'}")

    for issue in result.issues:
        if issue.issue_type == 'custom_rule_violation':
            print(f"\n{issue.message}")


def example_9_validation_report():
    """Example 9: Generate validation reports."""
    print("\n" + "="*80)
    print("EXAMPLE 9: Validation Reports")
    print("="*80)

    df = create_sample_data()
    validator = DataValidator()

    result = validator.validate(df)

    # Generate text report
    print("\n--- Text Report ---")
    text_report = validator.generate_validation_report(result, output_format='text')
    print(text_report)

    # Save HTML report
    html_report = validator.generate_validation_report(
        result,
        output_format='html',
        output_file='validation_report.html'
    )
    print("\nHTML report saved to: validation_report.html")

    # Generate JSON report
    json_report = validator.generate_validation_report(result, output_format='json')
    print("\n--- JSON Report (first 500 chars) ---")
    print(json_report[:500] + "...")


def example_10_data_type_validation():
    """Example 10: Validate data types."""
    print("\n" + "="*80)
    print("EXAMPLE 10: Data Type Validation")
    print("="*80)

    df = create_sample_data()
    validator = DataValidator()

    result = validator.check_data_types(
        df,
        expected_numeric=['age', 'salary', 'years_experience', 'satisfaction_score'],
        expected_categorical=['name', 'department'],
        expected_datetime=['hire_date']
    )

    print(f"Data type validation: {'PASSED' if result.is_valid else 'FAILED'}")

    # Show any type issues
    for issue in result.issues:
        print(f"\n{issue.issue_type}: {issue.message}")
        if issue.details:
            for key, value in issue.details.items():
                if key not in ['sample_values', 'sample_indices']:
                    print(f"  {key}: {value}")


def main():
    """Run all examples."""
    print("\n" + "="*80)
    print("DATA VALIDATOR EXAMPLES")
    print("Unified AI Analytics Platform")
    print("="*80)

    examples = [
        example_1_basic_validation,
        example_2_schema_validation,
        example_3_missing_values,
        example_4_duplicate_detection,
        example_5_outlier_detection,
        example_6_value_ranges,
        example_7_categorical_validation,
        example_8_custom_rules,
        example_9_validation_report,
        example_10_data_type_validation
    ]

    for example in examples:
        try:
            example()
        except Exception as e:
            print(f"\nError in {example.__name__}: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "="*80)
    print("All examples completed!")
    print("="*80)


if __name__ == "__main__":
    main()
