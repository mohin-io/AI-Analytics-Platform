# Unified AI Analytics Platform

**A Machine Learning Model Benchmarking System**

A comprehensive, production-ready platform for training, comparing, and deploying machine learning models across multiple paradigms. Built to demonstrate full-stack ML engineering capabilities with automated pipelines, explainability tools, and interactive dashboards.

---

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Modules](#modules)
- [Usage Examples](#usage-examples)
- [Architecture](#architecture)
- [API Reference](#api-reference)
- [Configuration](#configuration)
- [Testing](#testing)
- [Deployment](#deployment)
- [Contributing](#contributing)
- [License](#license)
- [Roadmap](#roadmap)

---

## Overview

The **Unified AI Analytics Platform** is a self-built system that automates the entire machine learning workflow—from data preprocessing to model deployment. It provides:

- **30+ ML Algorithms**: Classification, regression, clustering, deep learning, NLP, and time series
- **Automated Preprocessing**: Intelligent handling of missing values, outliers, and feature engineering
- **Model Benchmarking**: Compare all algorithms on your dataset with comprehensive metrics
- **Explainable AI**: SHAP, LIME, and feature importance for model interpretability
- **AutoML**: Automated hyperparameter optimization using Optuna
- **REST API**: Deploy models via FastAPI endpoints
- **Interactive Dashboard**: Streamlit-based UI for non-technical users
- **MLOps Integration**: MLflow for experiment tracking and model registry

### Problem Statement

Data scientists typically spend 80% of their time on:
- Manual data preprocessing and cleaning
- Experimenting with multiple algorithms
- Hyperparameter tuning
- Model comparison and selection
- Deployment and monitoring

This platform automates these workflows, allowing focus on insights rather than infrastructure.

---

## Key Features

###  Data Preprocessing Engine
- Load data from CSV, JSON, Parquet, Excel, SQL databases, and URLs
- Automated data validation and quality checks
- Multiple missing value imputation strategies (mean, median, KNN, MICE)
- Intelligent outlier detection (IQR, Z-score, Isolation Forest)
- Feature engineering (scaling, encoding, polynomial features, datetime extraction)

###  Supervised Learning Suite
**Classification Algorithms:**
- Logistic Regression
- Random Forest, Gradient Boosting
- XGBoost, LightGBM, CatBoost
- Support Vector Machines (SVM)
- K-Nearest Neighbors (KNN)
- Naive Bayes (Gaussian, Multinomial, Bernoulli)

**Regression Algorithms:**
- Linear, Ridge, Lasso, ElasticNet
- Random Forest, Gradient Boosting Regressors
- XGBoost, LightGBM, CatBoost Regressors
- SVR (Support Vector Regression)

###  Deep Learning Module
- Feedforward Neural Networks (Tabular data)
- Convolutional Neural Networks (Image classification)
- Recurrent Networks - LSTM/GRU (Time series)
- Autoencoders (Dimensionality reduction, anomaly detection)

###  Unsupervised Learning
- **Clustering**: K-Means, DBSCAN, Hierarchical, GMM
- **Dimensionality Reduction**: PCA, t-SNE, UMAP
- **Anomaly Detection**: Isolation Forest, One-Class SVM

###  Time Series Forecasting
- ARIMA/SARIMA
- Prophet (Facebook)
- Exponential Smoothing
- LSTM-based forecasting
- Temporal Fusion Transformers

###  NLP & Text Analytics
- Text preprocessing (tokenization, stopword removal, lemmatization)
- TF-IDF + Classical ML
- Word embeddings (Word2Vec, GloVe)
- Transformer models (BERT, RoBERTa, DistilBERT)
- Sentiment analysis

###  Explainable AI (XAI)
- SHAP values for any model
- LIME for local explanations
- Partial Dependence Plots
- Feature importance rankings
- Integrated Gradients for deep learning

###  AutoML & Optimization
- Automated algorithm selection
- Bayesian hyperparameter optimization (Optuna)
- Neural architecture search
- Ensemble model creation

###  Model Evaluation
- 20+ evaluation metrics
- Cross-validation support
- Learning curves
- Confusion matrices, ROC curves
- Comparative analysis dashboards

###  Deployment & Monitoring
- REST API with FastAPI
- Interactive Streamlit dashboard
- MLflow experiment tracking
- Model versioning and registry
- Docker containerization
- CI/CD pipelines

---

## Project Structure

```
unified-ai-platform/
├── docs/                          # Documentation
│   ├── PLAN.md                   # Project blueprint
│   ├── API_REFERENCE.md          # API documentation
│   └── SYSTEM_DESIGN.md          # Architecture details
├── src/                           # Source code
│   ├── preprocessing/             # Data preprocessing
│   │   ├── data_loader.py        # Multi-format data loading
│   │   ├── data_validator.py     # Data quality validation
│   │   ├── feature_engineer.py   # Feature engineering
│   │   └── missing_handler.py    # Missing value imputation
│   ├── models/                    # ML model implementations
│   │   ├── base.py               # Base model classes
│   │   ├── supervised/           # Classification & regression
│   │   ├── unsupervised/         # Clustering & dimensionality reduction
│   │   ├── deep_learning/        # Neural networks
│   │   ├── time_series/          # Forecasting models
│   │   └── nlp/                  # NLP models
│   ├── evaluation/                # Model evaluation
│   │   ├── metrics.py            # Evaluation metrics
│   │   └── comparator.py         # Model comparison
│   ├── explainability/            # XAI tools
│   │   ├── shap_explainer.py     # SHAP explanations
│   │   └── lime_explainer.py     # LIME explanations
│   ├── automl/                    # AutoML engine
│   │   └── optimizer.py          # Hyperparameter optimization
│   ├── api/                       # REST API
│   │   └── main.py               # FastAPI application
│   ├── dashboard/                 # Streamlit dashboard
│   │   └── app.py                # Dashboard application
│   └── utils/                     # Utilities
│       ├── logger.py             # Logging configuration
│       ├── config.py             # Configuration management
│       ├── file_handler.py       # File I/O operations
│       └── metrics_tracker.py    # Experiment tracking
├── tests/                         # Test suite
│   ├── unit/                     # Unit tests
│   ├── integration/              # Integration tests
│   └── performance/              # Performance tests
├── notebooks/                     # Jupyter notebooks
│   └── examples/                 # Usage examples
├── data/                          # Data storage
│   ├── raw/                      # Raw data
│   ├── processed/                # Processed data
│   └── sample/                   # Sample datasets
├── models/                        # Saved models
├── logs/                          # Log files
├── config/                        # Configuration files
├── .github/workflows/             # CI/CD pipelines
│   └── ci.yml                    # GitHub Actions workflow
├── requirements.txt               # Python dependencies
├── environment.yml                # Conda environment
├── setup.py                       # Package setup
├── pyproject.toml                 # Project configuration
├── Dockerfile                     # Docker configuration
├── docker-compose.yml             # Docker Compose
├── .gitignore                    # Git ignore rules
├── LICENSE                        # MIT License
├── CONTRIBUTING.md                # Contribution guidelines
├── CODE_OF_CONDUCT.md             # Code of conduct
└── README.md                      # This file
```

---

## Installation

### Prerequisites

- Python 3.9 or higher
- pip or conda package manager
- (Optional) CUDA-capable GPU for deep learning

### Option 1: Using pip

```bash
# Clone the repository
git clone https://github.com/mohin-io/unified-ai-platform.git
cd unified-ai-platform

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install the package in development mode
pip install -e .
```

### Option 2: Using Conda

```bash
# Clone the repository
git clone https://github.com/mohin-io/unified-ai-platform.git
cd unified-ai-platform

# Create conda environment
conda env create -f environment.yml

# Activate environment
conda activate unified-ai-platform
```

### Option 3: Using Docker

```bash
# Pull the Docker image
docker pull mohin-io/unified-ai-platform:latest

# Or build from source
docker build -t unified-ai-platform .

# Run the container
docker run -p 8000:8000 -p 8501:8501 unified-ai-platform
```

---

## Quick Start

### 1. Load and Preprocess Data

```python
from src.preprocessing import DataLoader, DataValidator, FeatureEngineer, MissingValueHandler

# Load data
loader = DataLoader()
df = loader.load_from_csv("data/dataset.csv")

# Validate data quality
validator = DataValidator()
validation_result = validator.validate(df)
print(validation_result.summary())

# Handle missing values
missing_handler = MissingValueHandler(strategy='median')
df_clean = missing_handler.fit_transform(df)

# Engineer features
engineer = FeatureEngineer(scaling='standard', encoding='onehot')
X = engineer.fit_transform(df_clean.drop('target', axis=1))
y = df_clean['target']
```

### 2. Train Multiple Models

```python
from src.models.supervised import (
    LogisticRegressionModel,
    RandomForestClassifierModel,
    XGBoostClassifierModel
)

# Initialize models
models = {
    'Logistic Regression': LogisticRegressionModel(),
    'Random Forest': RandomForestClassifierModel(n_estimators=100),
    'XGBoost': XGBoostClassifierModel()
}

# Train all models
results = {}
for name, model in models.items():
    model.train(X_train, y_train)
    metrics = model.evaluate(X_test, y_test)
    results[name] = metrics
    print(f"{name}: Accuracy = {metrics['accuracy']:.3f}")
```

### 3. Compare and Explain

```python
from src.evaluation import ModelComparator
from src.explainability import SHAPExplainer

# Compare models
comparator = ModelComparator()
comparison_df = comparator.compare_models(results)
comparator.plot_comparison(comparison_df)

# Explain the best model
best_model = models['XGBoost']
explainer = SHAPExplainer(best_model)
shap_values = explainer.explain(X_test)
explainer.plot_summary(shap_values)
```

### 4. Deploy via API

```bash
# Start the API server
uvicorn src.api.main:app --reload

# Make predictions via REST API
curl -X POST "http://localhost:8000/api/v1/predict" \
  -H "Content-Type: application/json" \
  -d '{"features": [[1.5, 2.3, 0.8]]}'
```

### 5. Launch Dashboard

```bash
# Start the Streamlit dashboard
streamlit run src/dashboard/app.py

# Access at http://localhost:8501
```

---

## Modules

### Data Preprocessing Engine

**Location:** [src/preprocessing/](src/preprocessing/)

The preprocessing module handles all data preparation tasks:

- **DataLoader**: Load data from multiple sources (CSV, JSON, SQL, etc.)
- **DataValidator**: Validate data quality and detect issues
- **MissingValueHandler**: Impute missing values using various strategies
- **FeatureEngineer**: Transform features (scaling, encoding, creation)
- **OutlierDetector**: Detect and handle outliers

**Example:**
```python
from src.preprocessing import DataPreprocessor

preprocessor = DataPreprocessor(
    missing_strategy='knn',
    scaling='robust',
    encoding='onehot'
)

X_train_processed = preprocessor.fit_transform(X_train, y_train)
X_test_processed = preprocessor.transform(X_test)
```

### Supervised Learning Suite

**Location:** [src/models/supervised/](src/models/supervised/)

Implementations of classification and regression algorithms:

**Key Classes:**
- `LogisticRegressionModel`
- `RandomForestClassifierModel`
- `XGBoostClassifierModel`
- `LightGBMClassifierModel`
- `CatBoostClassifierModel`
- `SVMClassifierModel`

All models inherit from `SupervisedModel` base class and provide consistent interfaces for training, prediction, and evaluation.

### Model Evaluation

**Location:** [src/evaluation/](src/evaluation/)

Comprehensive model evaluation tools:

**Metrics Supported:**
- Classification: Accuracy, Precision, Recall, F1, ROC-AUC
- Regression: MAE, MSE, RMSE, R², MAPE
- Clustering: Silhouette Score, Davies-Bouldin Index

**Example:**
```python
from src.evaluation import Evaluator

evaluator = Evaluator(task='classification')
metrics = evaluator.evaluate(y_true, y_pred)
evaluator.plot_confusion_matrix(y_true, y_pred)
evaluator.plot_roc_curve(y_true, y_proba)
```

### Explainable AI

**Location:** [src/explainability/](src/explainability/)

Model interpretability tools:

- **SHAP**: TreeExplainer, DeepExplainer, KernelExplainer
- **LIME**: Local explanations for any model
- **Feature Importance**: Rankings and visualizations
- **Partial Dependence**: PD plots for feature effects

**Example:**
```python
from src.explainability import SHAPExplainer

explainer = SHAPExplainer(model)
shap_values = explainer.compute_shap_values(X)
explainer.plot_waterfall(X[0])  # Explain single prediction
explainer.plot_summary(shap_values)  # Global feature importance
```

### AutoML Engine

**Location:** [src/automl/](src/automl/)

Automated machine learning capabilities:

- Automatic algorithm selection
- Hyperparameter optimization (Optuna)
- Feature selection
- Ensemble creation

**Example:**
```python
from src.automl import AutoML

automl = AutoML(task='classification', n_trials=100)
best_model = automl.fit(X_train, y_train)
predictions = best_model.predict(X_test)
```

---

## Usage Examples

### Example 1: Binary Classification

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from src.preprocessing import DataLoader, FeatureEngineer
from src.models.supervised import XGBoostClassifierModel
from src.evaluation import Evaluator

# Load data
loader = DataLoader()
df = loader.load_sample_dataset('breast_cancer')

# Split data
X = df.drop('target', axis=1)
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Preprocess
engineer = FeatureEngineer(scaling='standard')
X_train_scaled = engineer.fit_transform(X_train)
X_test_scaled = engineer.transform(X_test)

# Train model
model = XGBoostClassifierModel(n_estimators=100, learning_rate=0.1)
model.train(X_train_scaled, y_train)

# Evaluate
evaluator = Evaluator(task='classification')
metrics = evaluator.evaluate_classification(
    y_test,
    model.predict(X_test_scaled),
    model.predict_proba(X_test_scaled)
)

print(f"Accuracy: {metrics['accuracy']:.3f}")
print(f"F1 Score: {metrics['f1_score']:.3f}")
print(f"ROC-AUC: {metrics['roc_auc']:.3f}")
```

### Example 2: Regression with AutoML

```python
from src.preprocessing import DataLoader
from src.automl import AutoML

# Load data
loader = DataLoader()
df = loader.load_sample_dataset('california')

X = df.drop('target', axis=1)
y = df['target']

# AutoML - finds best model and hyperparameters
automl = AutoML(
    task='regression',
    time_budget=3600,  # 1 hour
    n_trials=100
)

best_model = automl.fit(X, y)

# Get results
print(f"Best algorithm: {automl.best_algorithm}")
print(f"Best score (R²): {automl.best_score:.3f}")
print(f"Best parameters: {automl.best_params}")

# Make predictions
predictions = best_model.predict(X_new)
```

### Example 3: Model Comparison

```python
from src.models.supervised import (
    LogisticRegressionModel,
    RandomForestClassifierModel,
    XGBoostClassifierModel,
    LightGBMClassifierModel
)
from src.evaluation import ModelComparator

# Initialize multiple models
models = {
    'Logistic Regression': LogisticRegressionModel(),
    'Random Forest': RandomForestClassifierModel(n_estimators=100),
    'XGBoost': XGBoostClassifierModel(),
    'LightGBM': LightGBMClassifierModel()
}

# Train and evaluate all models
results = {}
for name, model in models.items():
    model.train(X_train, y_train)
    metrics = model.evaluate(X_test, y_test)
    results[name] = metrics

# Compare models
comparator = ModelComparator()
comparison_df = comparator.create_comparison_table(results)
print(comparison_df)

# Visualize comparison
comparator.plot_metric_comparison(comparison_df, metric='f1_score')
comparator.plot_training_time_comparison(results)
```

---

## Architecture

### High-Level Architecture

```
┌─────────────┐      ┌──────────────────┐      ┌─────────────┐
│   Data      │─────>│  Preprocessing   │─────>│   Models    │
│   Sources   │      │     Engine       │      │   Training  │
└─────────────┘      └──────────────────┘      └─────────────┘
                                                       │
                                                       v
┌─────────────┐      ┌──────────────────┐      ┌─────────────┐
│  Dashboard  │<─────│    Evaluation    │<─────│    Model    │
│     UI      │      │   & Comparison   │      │  Registry   │
└─────────────┘      └──────────────────┘      └─────────────┘
                              │
                              v
                    ┌──────────────────┐
                    │   Explainability │
                    │      Engine      │
                    └──────────────────┘
```

### Component Interactions

1. **Data Ingestion**: Load data from various sources
2. **Preprocessing**: Clean, validate, and transform data
3. **Model Training**: Train multiple algorithms in parallel
4. **Evaluation**: Compute metrics and compare models
5. **Explainability**: Generate SHAP/LIME explanations
6. **Storage**: Save models to registry with MLflow
7. **Serving**: Deploy via REST API or dashboard
8. **Monitoring**: Track performance and data drift

---

## API Reference

### REST API Endpoints

**Base URL:** `http://localhost:8000/api/v1`

#### Data Management

```
POST   /upload              Upload dataset
GET    /datasets            List all datasets
GET    /datasets/{id}       Get dataset details
DELETE /datasets/{id}       Delete dataset
```

#### Model Training

```
POST   /train               Start training pipeline
GET    /training/status/{id} Check training status
POST   /train/automl        Run AutoML
```

#### Model Management

```
GET    /models              List all trained models
GET    /models/{id}         Get model details
POST   /models/{id}/predict Make predictions
DELETE /models/{id}         Delete model
```

#### Evaluation

```
GET    /models/{id}/metrics Get evaluation metrics
GET    /models/{id}/explain Get SHAP explanations
POST   /compare             Compare multiple models
```

### Example API Usage

```python
import requests

# Upload dataset
with open('data.csv', 'rb') as f:
    response = requests.post(
        'http://localhost:8000/api/v1/upload',
        files={'file': f}
    )
dataset_id = response.json()['dataset_id']

# Train models
response = requests.post(
    'http://localhost:8000/api/v1/train',
    json={
        'dataset_id': dataset_id,
        'task_type': 'classification',
        'algorithms': ['xgboost', 'random_forest', 'logistic_regression']
    }
)
job_id = response.json()['job_id']

# Get best model
response = requests.get(f'http://localhost:8000/api/v1/models/{model_id}')
model_info = response.json()

# Make predictions
response = requests.post(
    f'http://localhost:8000/api/v1/models/{model_id}/predict',
    json={'features': [[1.5, 2.3, 0.8]]}
)
predictions = response.json()['predictions']
```

---

## Configuration

### Configuration File

Create a `config/settings.yaml` file:

```yaml
# Data settings
data:
  raw_dir: "data/raw"
  processed_dir: "data/processed"
  test_size: 0.2
  random_seed: 42

# Model settings
models:
  n_jobs: -1  # Use all CPU cores
  cv_folds: 5

# AutoML settings
automl:
  n_trials: 100
  timeout: 3600  # 1 hour
  early_stopping: 10

# API settings
api:
  host: "0.0.0.0"
  port: 8000
  debug: false

# MLflow settings
mlflow:
  tracking_uri: "mlruns"
  experiment_name: "unified_ai_experiments"

# Logging
logging:
  level: "INFO"
  log_dir: "logs"
```

### Load Configuration

```python
from src.utils import Config

config = Config()
config.load_from_yaml("config/settings.yaml")

# Access settings
print(config.api_host)
print(config.random_seed)
```

---

## Testing

### Run All Tests

```bash
# Run all tests with coverage
pytest tests/ --cov=src --cov-report=html

# Run specific test categories
pytest tests/unit/              # Unit tests only
pytest tests/integration/       # Integration tests only
pytest tests/performance/       # Performance tests only

# Run with verbose output
pytest tests/ -v

# Run specific test file
pytest tests/unit/test_preprocessing.py
```

### Test Structure

```python
# tests/unit/test_data_loader.py
import pytest
from src.preprocessing import DataLoader

def test_load_csv():
    loader = DataLoader()
    df = loader.load_from_csv("tests/data/sample.csv")
    assert len(df) > 0
    assert df.shape[1] > 0

def test_load_invalid_file():
    loader = DataLoader()
    with pytest.raises(FileNotFoundError):
        loader.load_from_csv("nonexistent.csv")
```

---

## Deployment

### Local Deployment

```bash
# Start API server
uvicorn src.api.main:app --host 0.0.0.0 --port 8000

# Start dashboard (in another terminal)
streamlit run src.dashboard/app.py --server.port 8501
```

### Docker Deployment

```bash
# Build image
docker build -t unified-ai-platform .

# Run container
docker run -d \
  -p 8000:8000 \
  -p 8501:8501 \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/models:/app/models \
  unified-ai-platform
```

### Docker Compose

```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

### Cloud Deployment

**AWS ECS:**
```bash
# Build and push to ECR
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin <account-id>.dkr.ecr.us-east-1.amazonaws.com
docker build -t unified-ai-platform .
docker tag unified-ai-platform:latest <account-id>.dkr.ecr.us-east-1.amazonaws.com/unified-ai-platform:latest
docker push <account-id>.dkr.ecr.us-east-1.amazonaws.com/unified-ai-platform:latest

# Deploy to ECS (configure task definition and service)
```

**Google Cloud Run:**
```bash
# Build and deploy
gcloud builds submit --tag gcr.io/<project-id>/unified-ai-platform
gcloud run deploy unified-ai-platform --image gcr.io/<project-id>/unified-ai-platform --platform managed
```

---

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for details.

### Development Setup

```bash
# Clone and setup
git clone https://github.com/mohin-io/unified-ai-platform.git
cd unified-ai-platform

# Install development dependencies
pip install -r requirements.txt
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install

# Run tests
pytest tests/

# Format code
black src/ tests/
isort src/ tests/

# Lint
flake8 src/ tests/
mypy src/
```

### Pull Request Process

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run tests and linters
5. Commit with descriptive messages (`git commit -m 'Add amazing feature'`)
6. Push to your fork (`git push origin feature/amazing-feature`)
7. Open a Pull Request

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Roadmap

### Phase 1 (Current)
- [x] Core preprocessing engine
- [x] Supervised learning models
- [x] Model evaluation framework
- [x] Base explainability tools
- [x] REST API foundation

### Phase 2 (In Progress)
- [ ] Complete deep learning module
- [ ] Advanced NLP capabilities
- [ ] Time series forecasting
- [ ] Enhanced AutoML features
- [ ] Streamlit dashboard

### Phase 3 (Planned)
- [ ] Fairness and bias detection
- [ ] Model monitoring and drift detection
- [ ] Continual learning pipeline
- [ ] Multi-modal learning
- [ ] Advanced ensemble methods

### Phase 4 (Future)
- [ ] Federated learning support
- [ ] Model compression and optimization
- [ ] Edge deployment capabilities
- [ ] Real-time inference optimization
- [ ] Advanced visualization tools

---

## Acknowledgments

- Built with love for the ML/AI community
- Inspired by AutoML frameworks and production ML systems
- Uses best-in-class open-source libraries

---

## Contact

- **Author**: AI/ML Engineering Team
- **GitHub**: [@mohin-io](https://github.com/mohin-io)
- **Project Link**: [https://github.com/mohin-io/unified-ai-platform](https://github.com/mohin-io/unified-ai-platform)

---

**Made with Python, scikit-learn, TensorFlow, PyTorch, and lots of coffee**
