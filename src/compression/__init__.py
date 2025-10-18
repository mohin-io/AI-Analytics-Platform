"""
Model Compression and Optimization Module

Implements techniques to reduce model size and improve inference speed:
- Quantization
- Pruning
- Knowledge distillation
- Model compression
"""

from src.compression.quantization import ModelQuantizer, QuantizationStrategy
from src.compression.pruning import ModelPruner, PruningStrategy
from src.compression.distillation import KnowledgeDistillation

__all__ = [
    'ModelQuantizer',
    'QuantizationStrategy',
    'ModelPruner',
    'PruningStrategy',
    'KnowledgeDistillation'
]
