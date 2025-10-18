"""
Multi-Modal Learning Module

Provides tools for learning from multiple data modalities (text, images, tabular, etc.)
and fusing information from different sources.
"""

from src.multimodal.fusion import ModalityFusion, FusionStrategy
from src.multimodal.feature_extractor import MultiModalFeatureExtractor

__all__ = [
    'ModalityFusion',
    'FusionStrategy',
    'MultiModalFeatureExtractor'
]
