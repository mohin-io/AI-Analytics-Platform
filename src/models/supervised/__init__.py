"""
Supervised Learning Models

This module contains implementations of supervised learning algorithms
for classification and regression tasks.
"""

from src.models.supervised.classifiers import (
    LogisticRegressionModel,
    RandomForestClassifierModel,
    XGBoostClassifierModel,
    LightGBMClassifierModel,
    CatBoostClassifierModel,
    SVMClassifierModel,
    KNNClassifierModel,
    NaiveBayesModel
)

from src.models.supervised.regressors import (
    LinearRegressionModel,
    RidgeRegressionModel,
    LassoRegressionModel,
    ElasticNetModel,
    RandomForestRegressorModel,
    XGBoostRegressorModel,
    LightGBMRegressorModel,
    CatBoostRegressorModel,
    SVMRegressorModel
)

__all__ = [
    # Classifiers
    "LogisticRegressionModel",
    "RandomForestClassifierModel",
    "XGBoostClassifierModel",
    "LightGBMClassifierModel",
    "CatBoostClassifierModel",
    "SVMClassifierModel",
    "KNNClassifierModel",
    "NaiveBayesModel",
    # Regressors
    "LinearRegressionModel",
    "RidgeRegressionModel",
    "LassoRegressionModel",
    "ElasticNetModel",
    "RandomForestRegressorModel",
    "XGBoostRegressorModel",
    "LightGBMRegressorModel",
    "CatBoostRegressorModel",
    "SVMRegressorModel",
]
