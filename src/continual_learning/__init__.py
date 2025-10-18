"""
Continual Learning Module

Implements continual/incremental learning strategies for models to adapt
to new data without forgetting previous knowledge.
"""

from src.continual_learning.incremental_learner import IncrementalLearner, OnlineLearner
from src.continual_learning.memory_replay import MemoryReplayBuffer, ReplayStrategy

__all__ = [
    'IncrementalLearner',
    'OnlineLearner',
    'MemoryReplayBuffer',
    'ReplayStrategy'
]
