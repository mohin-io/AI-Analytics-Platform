"""
Fairness and Bias Detection Module

This module provides tools for detecting and measuring bias in machine learning models.
"""

from src.fairness.bias_detector import BiasDetector, FairnessMetrics
from src.fairness.mitigation import BiasMitigation

__all__ = [
    'BiasDetector',
    'FairnessMetrics',
    'BiasMitigation'
]
