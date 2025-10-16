# DataValidator Quick Reference

## Import

```python
from src.preprocessing.data_validator import DataValidator, ValidationSeverity
```

## Basic Usage

```python
# Create validator
validator = DataValidator()

# Validate DataFrame
result = validator.validate(df)

# Check if valid
if result.is_valid:
    print("Data is valid!")
else:
    for error in result.get_errors():
        print(error.message)
```

## Common Validation Tasks

### Schema Validation
```python
schema = {'age': 'int64', 'name': 'object', 'salary': 'float64'}
result = validator.validate_schema(df, schema)
```

### Check Missing Values
```python
result = validator.check_missing_values(df)
result = validator.check_missing_values(df, columns=['age', 'salary'])
```

### Check Duplicates
```python
result = validator.check_duplicates(df)
result = validator.check_duplicates(df, subset=['user_id'])
```

### Check Outliers
```python
# IQR method
result = validator.check_outliers(df, method='iqr', threshold=1.5)

# Z-score method
result = validator.check_outliers(df, method='zscore', threshold=3.0)
```

### Value Range Validation
```python
ranges = {
    'age': (0, 120),
    'salary': (0, None),
    'probability': (0.0, 1.0)
}
result = validator.check_value_ranges(df, ranges)
```

### Categorical Validation
```python
allowed = {
    'status': ['active', 'inactive', 'pending'],
    'department': ['sales', 'engineering', 'hr']
}
result = validator.check_categorical_values(df, allowed)
```

### Custom Rules
```python
validator.add_custom_rule(
    'age_check',
    lambda df: (df['age'] > 0).all(),
    'Age must be positive'
)
result = validator.validate(df)
```

## Configuration

```python
validator = DataValidator(
    missing_threshold=5.0,      # Warn if > 5% missing
    duplicate_threshold=1.0,    # Warn if > 1% duplicates
    outlier_method='iqr',       # 'iqr', 'zscore', 'isolation_forest'
    outlier_threshold=1.5       # Threshold for detection
)
```

## Reports

```python
# Text report
report = validator.generate_validation_report(result, output_format='text')
print(report)

# HTML report (saved to file)
validator.generate_validation_report(
    result,
    output_format='html',
    output_file='report.html'
)

# JSON report
json_report = validator.generate_validation_report(result, output_format='json')
```

## Working with Results

```python
result = validator.validate(df)

# Get issues by severity
errors = result.get_errors()      # Critical issues
warnings = result.get_warnings()  # Should review
info = result.get_info()          # Informational

# Get summary
print(result.get_summary())

# Access metadata
print(result.metadata)

# Iterate through all issues
for issue in result.issues:
    print(f"{issue.severity.value}: {issue.message}")
    print(f"  Column: {issue.column}")
    print(f"  Details: {issue.details}")
```

## Validation Issue Properties

```python
issue = result.issues[0]

issue.column        # Column name (or None for dataset-level)
issue.issue_type    # e.g., 'missing_values', 'outliers_detected'
issue.severity      # ValidationSeverity.ERROR/WARNING/INFO
issue.message       # Human-readable description
issue.details       # Dictionary with additional info
```

## Common Issue Types

- `missing_column` - Required column not found
- `extra_columns` - Unexpected columns present
- `type_mismatch` - Data type doesn't match schema
- `missing_values` - Column has missing values
- `duplicate_rows` - Duplicate rows found
- `outliers_detected` - Outliers found in numeric column
- `value_out_of_range` - Values outside acceptable range
- `invalid_categorical_value` - Invalid category value
- `custom_rule_violation` - Custom validation rule failed
- `high_cardinality` - Too many unique values
- `numeric_as_object` - Numeric data stored as object

## Tips

1. **Start with schema validation** to catch structural issues
2. **Use appropriate thresholds** for your data quality requirements
3. **Save reports** for audit trails
4. **Filter results** by severity to focus on critical issues
5. **Combine with data cleaning** to fix identified issues

## Example: Complete Validation Pipeline

```python
from src.preprocessing.data_validator import DataValidator
import pandas as pd

# Load data
df = pd.read_csv('data.csv')

# Define schema
schema = {
    'user_id': 'int64',
    'age': 'int64',
    'salary': 'float64',
    'department': 'object'
}

# Create validator
validator = DataValidator(
    missing_threshold=5.0,
    duplicate_threshold=0.0
)

# Add custom rules
validator.add_custom_rule(
    'salary_age_check',
    lambda df: (df['salary'] > 0).all() and (df['age'] > 18).all(),
    'Salary must be positive and age > 18'
)

# Run validation
result = validator.validate(df, schema=schema)

# Handle results
if not result.is_valid:
    print(f"Validation failed with {len(result.get_errors())} errors")

    # Generate report
    validator.generate_validation_report(
        result,
        output_format='html',
        output_file='validation_report.html'
    )

    # Fix issues or raise error
    raise ValueError("Data validation failed")
else:
    print("Data validation passed!")
    # Continue with data processing
```

## Further Reading

- [Complete Documentation](./DATA_VALIDATOR_README.md)
- [Example Scripts](../../examples/data_validator_example.py)
- [Test Suite](../../tests/test_data_validator.py)
