"""
FastAPI application for the Unified AI Analytics Platform

This module provides REST API endpoints for:
- Model training
- Predictions
- Model management
- Evaluation
"""

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import io

from src.preprocessing import DataLoader, FeatureEngineer, MissingValueHandler
from src.models.supervised import (
    XGBoostClassifierModel,
    RandomForestClassifierModel,
    LogisticRegressionModel,
    XGBoostRegressorModel,
    RandomForestRegressorModel,
    LinearRegressionModel
)
from src.evaluation import Evaluator
from src.automl import AutoMLOptimizer

# Initialize FastAPI app
app = FastAPI(
    title="Unified AI Analytics Platform API",
    description="REST API for machine learning model training and prediction",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for request/response
class TrainingRequest(BaseModel):
    """Request model for training."""
    algorithm: str
    task_type: str  # 'classification' or 'regression'
    target_column: str
    test_size: float = 0.2
    hyperparameters: Optional[Dict[str, Any]] = None

class PredictionRequest(BaseModel):
    """Request model for predictions."""
    model_id: str
    features: List[List[float]]

class ModelResponse(BaseModel):
    """Response model for model information."""
    model_id: str
    algorithm: str
    task_type: str
    metrics: Dict[str, float]
    created_at: str

# Global storage (in production, use a database)
models_storage: Dict[str, Any] = {}
datasets_storage: Dict[str, pd.DataFrame] = {}

@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Unified AI Analytics Platform API",
        "version": "1.0.0",
        "docs": "/docs"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}

@app.post("/api/v1/upload")
async def upload_dataset(file: UploadFile = File(...)):
    """
    Upload a dataset.

    Args:
        file: CSV file

    Returns:
        Dataset ID and info
    """
    try:
        # Read file
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))

        # Generate dataset ID
        dataset_id = f"dataset_{len(datasets_storage) + 1}"

        # Store dataset
        datasets_storage[dataset_id] = df

        return {
            "dataset_id": dataset_id,
            "filename": file.filename,
            "shape": df.shape,
            "columns": df.columns.tolist()
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/api/v1/train")
async def train_model(request: TrainingRequest, dataset_id: str):
    """
    Train a machine learning model.

    Args:
        request: Training configuration
        dataset_id: ID of uploaded dataset

    Returns:
        Trained model information
    """
    try:
        # Get dataset
        if dataset_id not in datasets_storage:
            raise HTTPException(status_code=404, detail="Dataset not found")

        df = datasets_storage[dataset_id]

        # Split features and target
        X = df.drop(request.target_column, axis=1)
        y = df[request.target_column]

        # Preprocess
        engineer = FeatureEngineer(scaling='standard', encoding='onehot')
        X_processed = engineer.fit_transform(X, y)

        # Split data
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X_processed, y, test_size=request.test_size, random_state=42
        )

        # Select and train model
        algorithm_map = {
            'xgboost': XGBoostClassifierModel if request.task_type == 'classification' else XGBoostRegressorModel,
            'random_forest': RandomForestClassifierModel if request.task_type == 'classification' else RandomForestRegressorModel,
            'logistic_regression': LogisticRegressionModel,
            'linear_regression': LinearRegressionModel
        }

        if request.algorithm not in algorithm_map:
            raise HTTPException(status_code=400, detail=f"Unknown algorithm: {request.algorithm}")

        model_class = algorithm_map[request.algorithm]
        model = model_class(**(request.hyperparameters or {}))

        # Train
        model.train(X_train, y_train)

        # Evaluate
        metrics = model.evaluate(X_test, y_test)

        # Generate model ID
        model_id = f"model_{len(models_storage) + 1}"

        # Store model
        models_storage[model_id] = {
            'model': model,
            'engineer': engineer,
            'algorithm': request.algorithm,
            'task_type': request.task_type,
            'metrics': metrics,
            'created_at': pd.Timestamp.now().isoformat()
        }

        return {
            "model_id": model_id,
            "algorithm": request.algorithm,
            "task_type": request.task_type,
            "metrics": metrics,
            "training_time": model.training_time
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/predict")
async def predict(request: PredictionRequest):
    """
    Make predictions using a trained model.

    Args:
        request: Prediction request with model_id and features

    Returns:
        Predictions
    """
    try:
        if request.model_id not in models_storage:
            raise HTTPException(status_code=404, detail="Model not found")

        model_info = models_storage[request.model_id]
        model = model_info['model']
        engineer = model_info['engineer']

        # Prepare features
        X = pd.DataFrame(request.features)
        X_processed = engineer.transform(X)

        # Predict
        predictions = model.predict(X_processed).tolist()

        # Get probabilities if classification
        probabilities = None
        if model_info['task_type'] == 'classification':
            try:
                probabilities = model.predict_proba(X_processed).tolist()
            except:
                pass

        return {
            "predictions": predictions,
            "probabilities": probabilities
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/models")
async def list_models():
    """
    List all trained models.

    Returns:
        List of model information
    """
    models_list = []
    for model_id, model_info in models_storage.items():
        models_list.append({
            "model_id": model_id,
            "algorithm": model_info['algorithm'],
            "task_type": model_info['task_type'],
            "metrics": model_info['metrics'],
            "created_at": model_info['created_at']
        })

    return {"models": models_list}

@app.get("/api/v1/models/{model_id}")
async def get_model(model_id: str):
    """
    Get detailed information about a specific model.

    Args:
        model_id: Model identifier

    Returns:
        Model details
    """
    if model_id not in models_storage:
        raise HTTPException(status_code=404, detail="Model not found")

    model_info = models_storage[model_id]
    return {
        "model_id": model_id,
        "algorithm": model_info['algorithm'],
        "task_type": model_info['task_type'],
        "metrics": model_info['metrics'],
        "created_at": model_info['created_at'],
        "metadata": model_info['model'].get_metadata()
    }

@app.delete("/api/v1/models/{model_id}")
async def delete_model(model_id: str):
    """
    Delete a trained model.

    Args:
        model_id: Model identifier

    Returns:
        Deletion confirmation
    """
    if model_id not in models_storage:
        raise HTTPException(status_code=404, detail="Model not found")

    del models_storage[model_id]

    return {"message": f"Model {model_id} deleted successfully"}

@app.post("/api/v1/automl")
async def run_automl(dataset_id: str, target_column: str, task_type: str, n_trials: int = 50):
    """
    Run AutoML to find the best model.

    Args:
        dataset_id: Dataset identifier
        target_column: Name of target column
        task_type: 'classification' or 'regression'
        n_trials: Number of trials per algorithm

    Returns:
        Best model information
    """
    try:
        if dataset_id not in datasets_storage:
            raise HTTPException(status_code=404, detail="Dataset not found")

        df = datasets_storage[dataset_id]

        # Split features and target
        X = df.drop(target_column, axis=1)
        y = df[target_column]

        # Preprocess
        engineer = FeatureEngineer(scaling='standard', encoding='onehot')
        X_processed = engineer.fit_transform(X, y)

        # Run AutoML
        automl = AutoMLOptimizer(
            task=task_type,
            n_trials_per_model=n_trials
        )

        best_model = automl.fit(X_processed, y)

        # Generate model ID
        model_id = f"model_{len(models_storage) + 1}"

        # Store model
        models_storage[model_id] = {
            'model': best_model,
            'engineer': engineer,
            'algorithm': automl.best_algorithm,
            'task_type': task_type,
            'metrics': {'score': automl.best_score},
            'created_at': pd.Timestamp.now().isoformat()
        }

        return {
            "model_id": model_id,
            "best_algorithm": automl.best_algorithm,
            "best_score": automl.best_score,
            "leaderboard": automl.get_leaderboard().to_dict('records')
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
