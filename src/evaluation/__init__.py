"""
Model Evaluation Module

This module provides comprehensive model evaluation capabilities including
metrics calculation, visualization, and model comparison.
"""

from src.evaluation.metrics import Evaluator, MetricsCalculator
from src.evaluation.comparator import ModelComparator

__all__ = ["Evaluator", "MetricsCalculator", "ModelComparator"]
