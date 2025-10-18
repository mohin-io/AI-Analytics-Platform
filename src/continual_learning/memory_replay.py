"""
Memory Replay for Continual Learning

Implements experience replay strategies to prevent catastrophic forgetting.
"""

import numpy as np
import pandas as pd
from typing import Optional, List, Tuple, Dict, Any
from enum import Enum
import random


class ReplayStrategy(Enum):
    """Replay strategies for memory management."""
    RANDOM = "random"  # Random sampling from memory
    RESERVOIR = "reservoir"  # Reservoir sampling
    RING_BUFFER = "ring_buffer"  # First-in-first-out
    BALANCED = "balanced"  # Class-balanced sampling
    HERDING = "herding"  # Mean-of-features based selection


class MemoryReplayBuffer:
    """
    Experience replay buffer for continual learning.

    Stores samples from previous tasks to prevent catastrophic forgetting.
    """

    def __init__(self, max_size: int = 1000,
                 strategy: ReplayStrategy = ReplayStrategy.RANDOM):
        """
        Initialize memory replay buffer.

        Args:
            max_size: Maximum number of samples to store
            strategy: Replay strategy to use
        """
        self.max_size = max_size
        self.strategy = strategy

        self.X_buffer: List[np.ndarray] = []
        self.y_buffer: List[Any] = []
        self.n_samples_seen = 0
        self.class_indices: Dict[Any, List[int]] = {}  # For balanced sampling

    def add(self, X: np.ndarray, y: np.ndarray):
        """
        Add samples to the buffer.

        Args:
            X: Feature matrix
            y: Labels
        """
        if X.ndim == 1:
            X = X.reshape(1, -1)
            y = np.array([y])

        for i in range(len(X)):
            sample_x = X[i]
            sample_y = y[i]

            if self.strategy == ReplayStrategy.RESERVOIR:
                self._reservoir_add(sample_x, sample_y)
            elif self.strategy == ReplayStrategy.RING_BUFFER:
                self._ring_buffer_add(sample_x, sample_y)
            elif self.strategy == ReplayStrategy.BALANCED:
                self._balanced_add(sample_x, sample_y)
            else:  # RANDOM or HERDING
                self._simple_add(sample_x, sample_y)

            self.n_samples_seen += 1

    def _simple_add(self, x: np.ndarray, y: Any):
        """Simple addition to buffer."""
        if len(self.X_buffer) < self.max_size:
            self.X_buffer.append(x)
            self.y_buffer.append(y)

            # Update class indices
            if y not in self.class_indices:
                self.class_indices[y] = []
            self.class_indices[y].append(len(self.X_buffer) - 1)
        else:
            # Replace random sample
            idx = random.randint(0, self.max_size - 1)
            old_y = self.y_buffer[idx]

            # Update class indices
            if old_y in self.class_indices:
                self.class_indices[old_y].remove(idx)

            self.X_buffer[idx] = x
            self.y_buffer[idx] = y

            if y not in self.class_indices:
                self.class_indices[y] = []
            self.class_indices[y].append(idx)

    def _reservoir_add(self, x: np.ndarray, y: Any):
        """Reservoir sampling addition."""
        if len(self.X_buffer) < self.max_size:
            self.X_buffer.append(x)
            self.y_buffer.append(y)

            if y not in self.class_indices:
                self.class_indices[y] = []
            self.class_indices[y].append(len(self.X_buffer) - 1)
        else:
            # Reservoir sampling: keep with probability max_size / n_samples_seen
            j = random.randint(0, self.n_samples_seen)
            if j < self.max_size:
                old_y = self.y_buffer[j]

                # Update class indices
                if old_y in self.class_indices:
                    self.class_indices[old_y].remove(j)

                self.X_buffer[j] = x
                self.y_buffer[j] = y

                if y not in self.class_indices:
                    self.class_indices[y] = []
                self.class_indices[y].append(j)

    def _ring_buffer_add(self, x: np.ndarray, y: Any):
        """Ring buffer (FIFO) addition."""
        if len(self.X_buffer) < self.max_size:
            self.X_buffer.append(x)
            self.y_buffer.append(y)

            if y not in self.class_indices:
                self.class_indices[y] = []
            self.class_indices[y].append(len(self.X_buffer) - 1)
        else:
            # Remove oldest (index 0) and append new
            old_y = self.y_buffer[0]

            # Update class indices
            if old_y in self.class_indices:
                # Remove index 0 and decrement all other indices
                self.class_indices[old_y].remove(0)
                for cls in self.class_indices:
                    self.class_indices[cls] = [i - 1 for i in self.class_indices[cls]]

            self.X_buffer.pop(0)
            self.y_buffer.pop(0)

            self.X_buffer.append(x)
            self.y_buffer.append(y)

            if y not in self.class_indices:
                self.class_indices[y] = []
            self.class_indices[y].append(len(self.X_buffer) - 1)

    def _balanced_add(self, x: np.ndarray, y: Any):
        """Balanced addition - maintain equal samples per class."""
        if y not in self.class_indices:
            self.class_indices[y] = []

        # Calculate samples per class
        n_classes = len(self.class_indices)
        if n_classes == 0:
            samples_per_class = self.max_size
        else:
            samples_per_class = max(1, self.max_size // max(n_classes, 1))

        # Check if this class is at capacity
        if len(self.class_indices[y]) < samples_per_class:
            # Add new sample
            self.X_buffer.append(x)
            self.y_buffer.append(y)
            self.class_indices[y].append(len(self.X_buffer) - 1)
        else:
            # Replace random sample of this class
            idx_in_class = random.randint(0, len(self.class_indices[y]) - 1)
            buffer_idx = self.class_indices[y][idx_in_class]

            self.X_buffer[buffer_idx] = x
            self.y_buffer[buffer_idx] = y

    def sample(self, n_samples: int,
               strategy: Optional[ReplayStrategy] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Sample from the buffer.

        Args:
            n_samples: Number of samples to retrieve
            strategy: Sampling strategy (None = use buffer's strategy)

        Returns:
            Tuple of (X_sample, y_sample)
        """
        if len(self.X_buffer) == 0:
            return np.array([]), np.array([])

        n_samples = min(n_samples, len(self.X_buffer))

        if strategy is None:
            strategy = self.strategy

        if strategy == ReplayStrategy.BALANCED:
            return self._balanced_sample(n_samples)
        else:
            # Random sampling
            indices = random.sample(range(len(self.X_buffer)), n_samples)
            X_sample = np.array([self.X_buffer[i] for i in indices])
            y_sample = np.array([self.y_buffer[i] for i in indices])
            return X_sample, y_sample

    def _balanced_sample(self, n_samples: int) -> Tuple[np.ndarray, np.ndarray]:
        """Sample with balanced class distribution."""
        if len(self.class_indices) == 0:
            return np.array([]), np.array([])

        samples_per_class = max(1, n_samples // len(self.class_indices))
        sampled_X = []
        sampled_y = []

        for cls, indices in self.class_indices.items():
            if len(indices) == 0:
                continue

            n_to_sample = min(samples_per_class, len(indices))
            sampled_indices = random.sample(indices, n_to_sample)

            for idx in sampled_indices:
                sampled_X.append(self.X_buffer[idx])
                sampled_y.append(self.y_buffer[idx])

        # If we haven't reached n_samples, sample more randomly
        while len(sampled_X) < n_samples and len(self.X_buffer) > 0:
            idx = random.randint(0, len(self.X_buffer) - 1)
            sampled_X.append(self.X_buffer[idx])
            sampled_y.append(self.y_buffer[idx])

        return np.array(sampled_X), np.array(sampled_y)

    def get_all(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get all samples from buffer.

        Returns:
            Tuple of (X_all, y_all)
        """
        if len(self.X_buffer) == 0:
            return np.array([]), np.array([])

        return np.array(self.X_buffer), np.array(self.y_buffer)

    def clear(self):
        """Clear the buffer."""
        self.X_buffer = []
        self.y_buffer = []
        self.class_indices = {}
        self.n_samples_seen = 0

    def size(self) -> int:
        """Get current size of buffer."""
        return len(self.X_buffer)

    def get_class_distribution(self) -> Dict[Any, int]:
        """
        Get distribution of classes in buffer.

        Returns:
            Dictionary mapping class to count
        """
        distribution = {}
        for y in self.y_buffer:
            distribution[y] = distribution.get(y, 0) + 1
        return distribution

    def is_full(self) -> bool:
        """Check if buffer is full."""
        return len(self.X_buffer) >= self.max_size


class ReplayMixin:
    """
    Mixin class to add replay capabilities to any learner.

    Can be combined with IncrementalLearner or OnlineLearner.
    """

    def __init__(self, *args, memory_size: int = 1000,
                 replay_strategy: ReplayStrategy = ReplayStrategy.RANDOM,
                 replay_ratio: float = 0.5, **kwargs):
        """
        Initialize replay mixin.

        Args:
            memory_size: Size of replay buffer
            replay_strategy: Strategy for replay sampling
            replay_ratio: Ratio of replay samples to new samples during training
        """
        super().__init__(*args, **kwargs)
        self.memory = MemoryReplayBuffer(max_size=memory_size, strategy=replay_strategy)
        self.replay_ratio = replay_ratio

    def fit_with_replay(self, X_new: np.ndarray, y_new: np.ndarray):
        """
        Fit model on new data combined with replay samples.

        Args:
            X_new: New feature data
            y_new: New labels
        """
        # Add new samples to memory
        self.memory.add(X_new, y_new)

        # Sample from memory
        n_replay = int(len(X_new) * self.replay_ratio)
        if n_replay > 0 and self.memory.size() > 0:
            X_replay, y_replay = self.memory.sample(n_replay)

            # Combine new and replay data
            X_combined = np.vstack([X_new, X_replay])
            y_combined = np.concatenate([y_new, y_replay])
        else:
            X_combined = X_new
            y_combined = y_new

        # Train on combined data
        if hasattr(self, 'partial_fit'):
            return self.partial_fit(X_combined, y_combined)
        elif hasattr(self, 'fit'):
            return self.fit(X_combined, y_combined)
        else:
            raise AttributeError("Object must have 'fit' or 'partial_fit' method")

    def get_memory_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the replay memory.

        Returns:
            Dictionary with memory statistics
        """
        return {
            'size': self.memory.size(),
            'max_size': self.memory.max_size,
            'is_full': self.memory.is_full(),
            'class_distribution': self.memory.get_class_distribution(),
            'samples_seen': self.memory.n_samples_seen
        }
