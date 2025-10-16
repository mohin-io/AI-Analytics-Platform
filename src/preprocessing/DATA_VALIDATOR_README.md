# DataValidator Module

## Overview

The `DataValidator` class provides comprehensive data validation functionality for the Unified AI Analytics Platform. It ensures data quality and integrity before processing by validating schemas, detecting data quality issues, identifying outliers, and generating detailed validation reports.

## Features

### Core Validation Capabilities

1. **Schema Validation**
   - Validates column presence and data types
   - Detects missing required columns
   - Identifies unexpected columns
   - Ensures type compatibility

2. **Data Quality Checks**
   - Missing value detection and quantification
   - Duplicate row identification
   - Data type validation
   - High cardinality detection

3. **Value Validation**
   - Numeric range validation
   - Categorical value validation
   - Outlier detection (IQR, Z-score, Isolation Forest)
   - Custom validation rules

4. **Reporting**
   - Detailed validation reports (Text, HTML, JSON)
   - Issue severity levels (ERROR, WARNING, INFO)
   - Comprehensive statistics and metadata

## Installation

The DataValidator is part of the preprocessing module. No additional installation is required beyond the project dependencies.

```python
from src.preprocessing.data_validator import DataValidator
```

## Quick Start

```python
import pandas as pd
from src.preprocessing.data_validator import DataValidator

# Load your data
df = pd.read_csv('data.csv')

# Create validator
validator = DataValidator()

# Run comprehensive validation
result = validator.validate(df)

# Check results
if result.is_valid:
    print("Data validation passed!")
else:
    print("Validation failed:")
    for error in result.get_errors():
        print(f"  - {error.message}")
```

## Usage Examples

### 1. Basic Validation

```python
# Create validator with custom thresholds
validator = DataValidator(
    missing_threshold=5.0,      # Warn if > 5% missing values
    duplicate_threshold=1.0,    # Warn if > 1% duplicates
    outlier_method='iqr',       # Use IQR for outlier detection
    outlier_threshold=1.5       # Standard IQR threshold
)

# Validate DataFrame
result = validator.validate(df)

# Print summary
print(result.get_summary())
```

### 2. Schema Validation

```python
# Define expected schema
schema = {
    'user_id': 'int64',
    'name': 'object',
    'age': 'float64',
    'salary': 'float64',
    'is_active': 'bool',
    'hire_date': 'datetime64[ns]'
}

# Validate schema
result = validator.validate_schema(df, schema)

if not result.is_valid:
    for issue in result.issues:
        print(f"{issue.severity.value}: {issue.message}")
```

### 3. Missing Value Detection

```python
# Check all columns
result = validator.check_missing_values(df)

# Check specific columns
result = validator.check_missing_values(df, columns=['age', 'salary'])

# Analyze missing values
for issue in result.issues:
    if issue.issue_type == 'missing_values':
        print(f"{issue.column}: {issue.details['percentage']:.2f}% missing")
```

### 4. Duplicate Detection

```python
# Check for complete duplicates
result = validator.check_duplicates(df)

# Check duplicates based on specific columns
result = validator.check_duplicates(df, subset=['user_id', 'email'])

# Get duplicate information
for issue in result.issues:
    if issue.issue_type == 'duplicate_rows':
        print(f"Found {issue.details['count']} duplicates")
        print(f"Sample indices: {issue.details['sample_indices']}")
```

### 5. Outlier Detection

```python
# Using IQR method (robust to extreme values)
validator = DataValidator(outlier_method='iqr', outlier_threshold=1.5)
result = validator.check_outliers(df, columns=['salary', 'age'])

# Using Z-score method (assumes normal distribution)
validator = DataValidator(outlier_method='zscore', outlier_threshold=3.0)
result = validator.check_outliers(df)

# Using Isolation Forest (ML-based, requires sklearn)
validator = DataValidator(outlier_method='isolation_forest')
result = validator.check_outliers(df)

# Analyze outliers
for issue in result.issues:
    if issue.issue_type == 'outliers_detected':
        print(f"{issue.column}:")
        print(f"  Count: {issue.details['count']}")
        print(f"  Range: [{issue.details['min']}, {issue.details['max']}]")
```

### 6. Value Range Validation

```python
# Define acceptable ranges
ranges = {
    'age': (0, 120),
    'salary': (0, None),              # No upper bound
    'satisfaction_score': (0.0, 1.0),
    'temperature': (-273.15, None)    # Absolute zero minimum
}

result = validator.check_value_ranges(df, ranges)

for issue in result.issues:
    if issue.issue_type == 'value_out_of_range':
        print(f"{issue.column}: {issue.details['count']} values out of range")
```

### 7. Categorical Value Validation

```python
# Define allowed values
allowed_values = {
    'status': ['active', 'inactive', 'pending'],
    'priority': ['low', 'medium', 'high', 'critical'],
    'department': ['sales', 'engineering', 'marketing', 'hr']
}

result = validator.check_categorical_values(df, allowed_values)

for issue in result.issues:
    if issue.issue_type == 'invalid_categorical_value':
        print(f"{issue.column}: Invalid values - {issue.details['invalid_values']}")
```

### 8. Custom Validation Rules

```python
# Add custom rule
validator.add_custom_rule(
    'age_vs_experience',
    lambda df: (df['age'] >= df['years_experience']).all(),
    'Age must be greater than or equal to years of experience'
)

# Add complex custom rule
def check_email_domain(df):
    allowed_domains = ['company.com', 'partner.com']
    domains = df['email'].str.extract(r'@(.+)$')[0]
    return domains.isin(allowed_domains).all()

validator.add_custom_rule(
    'email_domain',
    check_email_domain,
    'Email must be from allowed domains'
)

# Run validation with custom rules
result = validator.validate(df)

# Remove custom rule
validator.remove_custom_rule('age_vs_experience')
```

### 9. Data Type Validation

```python
result = validator.check_data_types(
    df,
    expected_numeric=['age', 'salary', 'years_experience'],
    expected_categorical=['department', 'status'],
    expected_datetime=['hire_date', 'last_login']
)

for issue in result.issues:
    if issue.issue_type == 'type_mismatch':
        print(f"{issue.column}: Expected {issue.details['expected']}, got {issue.details['actual']}")
```

### 10. Generating Validation Reports

```python
# Validate data
result = validator.validate(df)

# Generate text report
text_report = validator.generate_validation_report(result, output_format='text')
print(text_report)

# Generate and save HTML report
html_report = validator.generate_validation_report(
    result,
    output_format='html',
    output_file='validation_report.html'
)

# Generate JSON report for programmatic processing
json_report = validator.generate_validation_report(result, output_format='json')
import json
report_data = json.loads(json_report)
```

## API Reference

### DataValidator Class

#### Constructor Parameters

- `missing_threshold` (float, default=0.0): Maximum acceptable percentage of missing values (0-100)
- `duplicate_threshold` (float, default=0.0): Maximum acceptable percentage of duplicate rows (0-100)
- `outlier_method` (str, default='iqr'): Method for outlier detection ('iqr', 'zscore', 'isolation_forest')
- `outlier_threshold` (float, default=1.5): Threshold for outlier detection

#### Main Methods

##### `validate(df, schema=None, check_outliers=True, check_duplicates=True, check_missing=True, check_types=True)`

Perform comprehensive validation on a DataFrame.

**Parameters:**
- `df` (pd.DataFrame): DataFrame to validate
- `schema` (dict, optional): Schema definition mapping column names to expected types
- `check_outliers` (bool): Whether to check for outliers
- `check_duplicates` (bool): Whether to check for duplicate rows
- `check_missing` (bool): Whether to check for missing values
- `check_types` (bool): Whether to validate data types

**Returns:** `ValidationResult`

##### `validate_schema(df, schema)`

Validate DataFrame against a defined schema.

**Parameters:**
- `df` (pd.DataFrame): DataFrame to validate
- `schema` (dict): Dictionary mapping column names to expected pandas dtypes

**Returns:** `ValidationResult`

##### `check_missing_values(df, columns=None)`

Check for missing values in DataFrame columns.

**Parameters:**
- `df` (pd.DataFrame): DataFrame to check
- `columns` (list, optional): Specific columns to check. If None, checks all columns.

**Returns:** `ValidationResult`

##### `check_duplicates(df, subset=None, keep='first')`

Check for duplicate rows in the DataFrame.

**Parameters:**
- `df` (pd.DataFrame): DataFrame to check
- `subset` (list, optional): Columns to consider for identifying duplicates
- `keep` (str): Which duplicates to mark ('first', 'last', False)

**Returns:** `ValidationResult`

##### `check_data_types(df, expected_numeric=None, expected_categorical=None, expected_datetime=None)`

Validate that columns have appropriate data types.

**Parameters:**
- `df` (pd.DataFrame): DataFrame to validate
- `expected_numeric` (list, optional): Columns that should be numeric
- `expected_categorical` (list, optional): Columns that should be categorical/object
- `expected_datetime` (list, optional): Columns that should be datetime

**Returns:** `ValidationResult`

##### `check_outliers(df, columns=None, method=None, threshold=None)`

Detect outliers in numeric columns.

**Parameters:**
- `df` (pd.DataFrame): DataFrame to check
- `columns` (list, optional): Columns to check. If None, checks all numeric columns.
- `method` (str, optional): Outlier detection method
- `threshold` (float, optional): Threshold for outlier detection

**Returns:** `ValidationResult`

##### `check_value_ranges(df, ranges)`

Validate that numeric columns fall within expected ranges.

**Parameters:**
- `df` (pd.DataFrame): DataFrame to validate
- `ranges` (dict): Dictionary mapping column names to (min, max) tuples

**Returns:** `ValidationResult`

##### `check_categorical_values(df, allowed_values)`

Validate that categorical columns contain only allowed values.

**Parameters:**
- `df` (pd.DataFrame): DataFrame to validate
- `allowed_values` (dict): Dictionary mapping column names to lists of allowed values

**Returns:** `ValidationResult`

##### `add_custom_rule(name, validation_func, error_message)`

Add a custom validation rule.

**Parameters:**
- `name` (str): Unique name for the validation rule
- `validation_func` (callable): Function that takes a DataFrame and returns bool
- `error_message` (str): Message to display when validation fails

##### `remove_custom_rule(name)`

Remove a custom validation rule by name.

**Parameters:**
- `name` (str): Name of the rule to remove

**Returns:** `bool` - True if rule was found and removed

##### `generate_validation_report(result, output_format='text', output_file=None)`

Generate a comprehensive validation report.

**Parameters:**
- `result` (ValidationResult): ValidationResult to generate report from
- `output_format` (str): Format of the report ('text', 'html', 'json')
- `output_file` (str, optional): File path to save the report

**Returns:** `str` - Report as a string

### ValidationResult Class

Container for validation results.

#### Attributes

- `is_valid` (bool): Overall validation status
- `issues` (list): List of ValidationIssue objects
- `metadata` (dict): Additional validation metadata

#### Methods

- `get_errors()`: Get all ERROR-level issues
- `get_warnings()`: Get all WARNING-level issues
- `get_info()`: Get all INFO-level issues
- `get_summary()`: Generate a human-readable summary

### ValidationIssue Class

Data class representing a single validation issue.

#### Attributes

- `column` (str, optional): Column where the issue was found
- `issue_type` (str): Type of validation issue
- `severity` (ValidationSeverity): Severity level
- `message` (str): Human-readable description
- `details` (dict): Additional details about the issue

### ValidationSeverity Enum

Enum representing validation issue severity levels.

- `INFO`: Informational messages
- `WARNING`: Issues that should be reviewed
- `ERROR`: Critical issues that must be addressed

## Design Decisions

### 1. Non-Destructive Validation

The validator never modifies the input data. It only analyzes and reports issues, leaving data cleaning decisions to the user. This design ensures:
- Data integrity is preserved
- Users have full control over data transformations
- Validation can be run multiple times safely

### 2. Configurable Thresholds

Thresholds for missing values, duplicates, and outliers can be customized to match specific data quality requirements. This allows:
- Strict validation for critical data
- Lenient validation for exploratory analysis
- Domain-specific quality standards

### 3. Severity Levels

Issues are categorized by severity (ERROR, WARNING, INFO) to help prioritize remediation efforts:
- **ERROR**: Critical issues that must be fixed (type mismatches, invalid values)
- **WARNING**: Issues that should be reviewed (high missing values, many duplicates)
- **INFO**: Informational findings (outliers detected, extra columns)

### 4. Comprehensive Validation

Multiple aspects of data quality are validated in a single pass:
- Schema conformance
- Data type correctness
- Missing value patterns
- Duplicate detection
- Outlier identification
- Value range compliance
- Categorical value validation
- Custom business rules

### 5. Extensibility

Custom validation rules allow domain-specific validations without modifying the core validator:
- Business rule validation
- Cross-column dependencies
- Domain constraints
- Data consistency checks

### 6. Multiple Outlier Detection Methods

Different outlier detection methods suit different data distributions:
- **IQR**: Robust to extreme values, works well for skewed distributions
- **Z-score**: Assumes normal distribution, fast computation
- **Isolation Forest**: ML-based, detects complex multi-dimensional patterns

## Best Practices

### 1. Choose Appropriate Thresholds

```python
# For critical data (e.g., financial transactions)
strict_validator = DataValidator(
    missing_threshold=0.0,
    duplicate_threshold=0.0
)

# For exploratory analysis
lenient_validator = DataValidator(
    missing_threshold=10.0,
    duplicate_threshold=5.0
)
```

### 2. Use Schema Validation Early

Define schemas early in your pipeline to catch data contract violations:

```python
schema = {...}
result = validator.validate_schema(df, schema)
if not result.is_valid:
    raise ValueError("Schema validation failed")
```

### 3. Combine with Data Cleaning

Use validation results to guide data cleaning:

```python
result = validator.validate(df)

# Handle missing values based on severity
for issue in result.get_warnings():
    if issue.issue_type == 'missing_values':
        if issue.details['percentage'] < 5:
            df[issue.column].fillna(df[issue.column].median(), inplace=True)
```

### 4. Document Custom Rules

Always document the business logic behind custom rules:

```python
# Business Rule: Employees cannot be paid less than minimum wage
validator.add_custom_rule(
    'minimum_wage_check',
    lambda df: (df['hourly_rate'] >= 15.00).all(),
    'Hourly rate must be at least $15.00 (minimum wage)'
)
```

### 5. Save Validation Reports

Save validation reports for audit trails and reproducibility:

```python
result = validator.validate(df)
validator.generate_validation_report(
    result,
    output_format='html',
    output_file=f'validation_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.html'
)
```

## Troubleshooting

### Issue: Validation is too slow on large datasets

**Solution:** Disable specific checks or validate a sample:

```python
# Disable expensive checks
result = validator.validate(df, check_outliers=False)

# Or validate a sample
sample_df = df.sample(n=10000, random_state=42)
result = validator.validate(sample_df)
```

### Issue: Too many INFO-level issues cluttering the report

**Solution:** Filter results by severity:

```python
result = validator.validate(df)

# Only show errors and warnings
for issue in result.get_errors() + result.get_warnings():
    print(issue.message)
```

### Issue: Outlier detection with IQR marks too many values as outliers

**Solution:** Increase the IQR threshold:

```python
# Standard threshold is 1.5, increase to 3.0 for extreme outliers only
validator = DataValidator(outlier_method='iqr', outlier_threshold=3.0)
```

### Issue: Custom rule is failing with cryptic errors

**Solution:** Add error handling to custom rules:

```python
def safe_custom_rule(df):
    try:
        # Your validation logic
        return (df['age'] >= df['years_experience']).all()
    except Exception as e:
        print(f"Custom rule error: {e}")
        return False

validator.add_custom_rule('safe_rule', safe_custom_rule, 'Validation failed')
```

## Performance Considerations

- **Large Datasets**: For datasets > 1M rows, consider sampling or disabling outlier detection
- **Many Columns**: Specify columns explicitly in checks to avoid validating all columns
- **Isolation Forest**: Most computationally expensive outlier method; use sparingly on large datasets
- **Custom Rules**: Complex custom rules can slow validation; optimize rule functions

## See Also

- [Data Loader Documentation](./data_loader.py)
- [Feature Engineering Module](./feature_engineer.py)
- [Example Usage](../../examples/data_validator_example.py)

## Contributing

When adding new validation methods:
1. Follow the existing pattern of returning `ValidationResult`
2. Use appropriate severity levels (ERROR, WARNING, INFO)
3. Include comprehensive docstrings with examples
4. Add type hints for all parameters and returns
5. Update this README with the new functionality
