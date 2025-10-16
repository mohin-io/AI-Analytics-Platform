# Quick Start Guide - Unified AI Analytics Platform

This notebook demonstrates the basic usage of the Unified AI Analytics Platform for a complete machine learning workflow.

## Setup

```python
# Import required modules
from src.preprocessing import DataLoader, DataValidator, FeatureEngineer, MissingValueHandler
from src.utils import setup_logger, Config, MetricsTracker
import pandas as pd
import numpy as np

# Setup logging
logger = setup_logger("quickstart_example")
logger.info("Starting Quick Start Example")
```

## Step 1: Load Configuration

```python
# Load configuration
from src.utils import Config

config = Config()
config.load_from_yaml("config/settings.yaml")

print(f"Random seed: {config.random_seed}")
print(f"Test size: {config.test_size}")
print(f"Scaling strategy: {config.scaling_strategy}")
```

## Step 2: Load Data

```python
# Initialize data loader
loader = DataLoader()

# Option 1: Load from file
df = loader.load_from_csv("data/sample/dataset.csv")

# Option 2: Load sample dataset
# df = loader.load_sample_dataset('iris')

# Option 3: Load from URL
# df = loader.load_from_url("https://example.com/data.csv")

# Display basic information
print(f"Dataset shape: {df.shape}")
print(f"\nFirst few rows:")
print(df.head())

# Get data info
info = loader.get_data_info(df)
print(f"\nData Info:")
print(f"Columns: {info['columns']}")
print(f"Missing values: {info['missing_values']}")
```

## Step 3: Validate Data Quality

```python
# Initialize validator
validator = DataValidator(
    missing_threshold=5.0,
    duplicate_threshold=1.0,
    outlier_method='iqr'
)

# Run validation
validation_result = validator.validate(df)

# Check if data is valid
if validation_result.is_valid:
    print("Data validation passed!")
else:
    print("Data validation failed. Issues found:")
    for error in validation_result.get_errors():
        print(f"  - {error.message}")

# Generate HTML report
validator.generate_validation_report(
    validation_result,
    output_format='html',
    output_file='reports/validation_report.html'
)

# Get summary
summary = validation_result.summary()
print(f"\nValidation Summary:")
for key, value in summary.items():
    print(f"{key}: {value}")
```

## Step 4: Handle Missing Values

```python
# Check for missing values
missing_info = MissingValueHandler().get_missing_info(df)
print("Missing Values:")
print(missing_info)

# Initialize missing value handler
missing_handler = MissingValueHandler(strategy='median')

# For training data: fit and transform
df_clean = missing_handler.fit_transform(df)

# Verify no missing values remain
print(f"\nMissing values after imputation: {df_clean.isnull().sum().sum()}")
```

## Step 5: Feature Engineering

```python
# Separate features and target
X = df_clean.drop('target', axis=1)
y = df_clean['target']

# Initialize feature engineer
engineer = FeatureEngineer(
    scaling='standard',
    encoding='onehot',
    handle_unknown='ignore'
)

# Fit on training data
engineer.fit(X, y)

# Transform data
X_transformed = engineer.transform(X)

print(f"Original shape: {X.shape}")
print(f"Transformed shape: {X_transformed.shape}")
print(f"Feature names: {engineer.get_feature_names()}")

# Optional: Create polynomial features
# X_poly = engineer.create_polynomial_features(X, degree=2)

# Optional: Create datetime features
# X_time = engineer.create_datetime_features(X, ['date_column'])

# Optional: Create interaction features
# X_interact = engineer.create_interaction_features(X, [('feature1', 'feature2')])
```

## Step 6: Split Data

```python
from sklearn.model_selection import train_test_split

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X_transformed,
    y,
    test_size=config.test_size,
    random_state=config.random_seed
)

print(f"Training set size: {len(X_train)}")
print(f"Test set size: {len(X_test)}")
print(f"Class distribution (train): {y_train.value_counts().to_dict()}")
```

## Step 7: Track Metrics

```python
# Initialize metrics tracker
tracker = MetricsTracker(experiment_name="quickstart_experiment")

# Start timer
tracker.start_timer()

# Log some metrics
tracker.log_metric("dataset_size", len(df))
tracker.log_metric("num_features", X_transformed.shape[1])
tracker.log_metric("train_size", len(X_train))
tracker.log_metric("test_size", len(X_test))

# You can also log multiple metrics at once
tracker.log_metrics({
    "scaling_method": config.scaling_strategy,
    "encoding_method": config.encoding_strategy,
    "missing_strategy": config.missing_value_strategy
})

# View all tracked metrics
all_metrics = tracker.get_all_metrics()
print("\nTracked Metrics:")
for metric, values in all_metrics.items():
    print(f"{metric}: {values}")
```

## Step 8: Save Processed Data

```python
from src.utils import FileHandler

# Save processed data
FileHandler.write_parquet(X_train, "data/processed/X_train.parquet")
FileHandler.write_parquet(X_test, "data/processed/X_test.parquet")
FileHandler.write_parquet(y_train.to_frame(), "data/processed/y_train.parquet")
FileHandler.write_parquet(y_test.to_frame(), "data/processed/y_test.parquet")

# Save feature engineer for later use
engineer.save("models/feature_engineer.pkl")

# Save missing handler
missing_handler.save("models/missing_handler.pkl")

print("Processed data saved successfully!")
```

## Step 9: Generate Summary Report

```python
# Stop timer
execution_time = tracker.stop_timer()

# Get summary
summary = tracker.summary()
print("\nExperiment Summary:")
for metric, stats in summary.items():
    print(f"\n{metric}:")
    for stat, value in stats.items():
        print(f"  {stat}: {value}")

# Save metrics to file
tracker.save_metrics("reports/quickstart_metrics.csv")

logger.info(f"Quick start example completed in {execution_time:.2f} seconds")
```

## Complete Pipeline Function

Here's a complete function that combines all the steps:

```python
def preprocess_pipeline(
    data_path: str,
    target_column: str,
    config_path: str = "config/settings.yaml"
):
    """
    Complete preprocessing pipeline.

    Args:
        data_path: Path to the input data
        target_column: Name of the target column
        config_path: Path to configuration file

    Returns:
        Tuple of (X_train, X_test, y_train, y_test, metadata)
    """
    # Setup
    logger = setup_logger("preprocessing_pipeline")
    config = Config()
    config.load_from_yaml(config_path)
    tracker = MetricsTracker("preprocessing_pipeline")
    tracker.start_timer()

    # Load data
    logger.info("Loading data...")
    loader = DataLoader()
    df = loader.auto_load(data_path)
    tracker.log_metric("original_shape", str(df.shape))

    # Validate data
    logger.info("Validating data...")
    validator = DataValidator()
    validation_result = validator.validate(df)

    if not validation_result.is_valid:
        logger.warning("Data validation failed")
        for error in validation_result.get_errors():
            logger.warning(f"  - {error.message}")

    # Handle missing values
    logger.info("Handling missing values...")
    missing_handler = MissingValueHandler(strategy=config.missing_value_strategy)
    df_clean = missing_handler.fit_transform(df)
    tracker.log_metric("missing_values_imputed", df.isnull().sum().sum())

    # Engineer features
    logger.info("Engineering features...")
    X = df_clean.drop(target_column, axis=1)
    y = df_clean[target_column]

    engineer = FeatureEngineer(
        scaling=config.scaling_strategy,
        encoding=config.encoding_strategy
    )
    X_transformed = engineer.fit_transform(X, y)
    tracker.log_metric("final_num_features", X_transformed.shape[1])

    # Split data
    logger.info("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_transformed,
        y,
        test_size=config.test_size,
        random_state=config.random_seed
    )

    # Save artifacts
    logger.info("Saving artifacts...")
    FileHandler.write_parquet(X_train, "data/processed/X_train.parquet")
    FileHandler.write_parquet(X_test, "data/processed/X_test.parquet")
    engineer.save("models/feature_engineer.pkl")
    missing_handler.save("models/missing_handler.pkl")

    # Finalize
    execution_time = tracker.stop_timer()
    tracker.save_metrics("reports/preprocessing_metrics.csv")

    metadata = {
        "execution_time": execution_time,
        "original_shape": df.shape,
        "final_shape": X_transformed.shape,
        "train_size": len(X_train),
        "test_size": len(X_test)
    }

    logger.info(f"Pipeline completed in {execution_time:.2f} seconds")

    return X_train, X_test, y_train, y_test, metadata


# Usage
X_train, X_test, y_train, y_test, metadata = preprocess_pipeline(
    data_path="data/sample/dataset.csv",
    target_column="target"
)

print("Pipeline Results:")
print(f"Training set: {metadata['train_size']} samples")
print(f"Test set: {metadata['test_size']} samples")
print(f"Features: {metadata['final_shape'][1]}")
print(f"Execution time: {metadata['execution_time']:.2f}s")
```

## Next Steps

After preprocessing, you can:

1. **Train Models** - Use the supervised learning suite to train multiple algorithms
2. **Evaluate Performance** - Compare models using the evaluation module
3. **Explain Predictions** - Use SHAP/LIME for model interpretability
4. **Optimize Hyperparameters** - Use AutoML for automated tuning
5. **Deploy** - Create an API endpoint or dashboard for predictions

See other example notebooks for these workflows:
- `02_ModelTraining.ipynb` - Training and comparing models
- `03_Hyperparameter Optimization.ipynb` - AutoML and tuning
- `04_ModelExplainability.ipynb` - SHAP and LIME explanations
- `05_Deployment.ipynb` - API and dashboard deployment

## Summary

This quick start guide demonstrated:
- Loading data from multiple sources
- Validating data quality
- Handling missing values
- Feature engineering (scaling, encoding, transformations)
- Data splitting
- Metrics tracking
- Saving processed data and artifacts

All components are modular and can be used independently or as part of a complete pipeline.
