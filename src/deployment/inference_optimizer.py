"""
Real-Time Inference Optimization

Optimize models for low-latency, high-throughput inference.
"""

import numpy as np
from typing import Any, Dict, Optional, List
import time
from concurrent.futures import ThreadPoolExecutor
import queue


class InferenceOptimizer:
    """
    Optimize model inference for production deployment.

    Features:
    - Batch inference optimization
    - Caching for repeated inputs
    - Async inference
    - Model warm-up
    """

    def __init__(self, model: Any, batch_size: int = 32,
                 use_caching: bool = True, cache_size: int = 1000):
        """
        Initialize inference optimizer.

        Args:
            model: Trained model
            batch_size: Optimal batch size for inference
            use_caching: Enable prediction caching
            cache_size: Maximum cache entries
        """
        self.model = model
        self.batch_size = batch_size
        self.use_caching = use_caching
        self.cache_size = cache_size

        self.prediction_cache: Dict[str, np.ndarray] = {}
        self.cache_hits = 0
        self.cache_misses = 0

        # Warm up model
        self._warm_up()

    def _warm_up(self):
        """Warm up model with dummy inference."""
        if hasattr(self.model, 'coef_'):
            n_features = self.model.coef_.shape[1] if self.model.coef_.ndim > 1 else self.model.coef_.shape[0]
            dummy_input = np.random.randn(1, n_features)
            _ = self.model.predict(dummy_input)

    def predict_single(self, X: np.ndarray, use_cache: bool = True) -> np.ndarray:
        """
        Predict single sample with caching.

        Args:
            X: Input features (single sample)
            use_cache: Whether to use cache

        Returns:
            Prediction
        """
        if use_cache and self.use_caching:
            # Create cache key from input
            cache_key = self._create_cache_key(X)

            if cache_key in self.prediction_cache:
                self.cache_hits += 1
                return self.prediction_cache[cache_key]

            self.cache_misses += 1

        # Make prediction
        prediction = self.model.predict(X.reshape(1, -1))

        # Cache result
        if use_cache and self.use_caching:
            if len(self.prediction_cache) < self.cache_size:
                self.prediction_cache[cache_key] = prediction

        return prediction

    def predict_batch(self, X: np.ndarray) -> np.ndarray:
        """
        Optimized batch prediction.

        Args:
            X: Input features (multiple samples)

        Returns:
            Predictions
        """
        return self.model.predict(X)

    def predict_streaming(self, X_iterator, max_workers: int = 4):
        """
        Async streaming inference.

        Args:
            X_iterator: Iterator over input batches
            max_workers: Number of parallel workers

        Yields:
            Predictions for each batch
        """
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []

            for X_batch in X_iterator:
                future = executor.submit(self.predict_batch, X_batch)
                futures.append(future)

            for future in futures:
                yield future.result()

    def _create_cache_key(self, X: np.ndarray) -> str:
        """Create hash key for caching."""
        return hash(X.tobytes())

    def get_cache_stats(self) -> Dict[str, int]:
        """Get cache statistics."""
        total_queries = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / total_queries if total_queries > 0 else 0

        return {
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'hit_rate': hit_rate,
            'cache_size': len(self.prediction_cache)
        }

    def benchmark(self, X_test: np.ndarray, n_iterations: int = 100) -> Dict[str, float]:
        """
        Benchmark inference performance.

        Args:
            X_test: Test data
            n_iterations: Number of iterations

        Returns:
            Performance metrics
        """
        # Single sample latency
        single_times = []
        for _ in range(n_iterations):
            start = time.time()
            _ = self.predict_single(X_test[0], use_cache=False)
            single_times.append(time.time() - start)

        # Batch throughput
        batch_times = []
        for _ in range(n_iterations // 10):
            start = time.time()
            _ = self.predict_batch(X_test)
            batch_times.append(time.time() - start)

        return {
            'single_latency_ms': np.mean(single_times) * 1000,
            'single_latency_p50_ms': np.percentile(single_times, 50) * 1000,
            'single_latency_p95_ms': np.percentile(single_times, 95) * 1000,
            'single_latency_p99_ms': np.percentile(single_times, 99) * 1000,
            'batch_throughput_samples_per_sec': len(X_test) / np.mean(batch_times),
            'batch_latency_ms': np.mean(batch_times) * 1000
        }


class AsyncInferenceQueue:
    """
    Asynchronous inference queue for high-throughput serving.
    """

    def __init__(self, model: Any, max_queue_size: int = 1000,
                 batch_size: int = 32, timeout: float = 1.0):
        """
        Initialize async inference queue.

        Args:
            model: Model for inference
            max_queue_size: Maximum queue size
            batch_size: Batch size for inference
            timeout: Timeout for batch collection (seconds)
        """
        self.model = model
        self.max_queue_size = max_queue_size
        self.batch_size = batch_size
        self.timeout = timeout

        self.input_queue = queue.Queue(maxsize=max_queue_size)
        self.result_queues: Dict[int, queue.Queue] = {}
        self.request_id = 0

    def submit(self, X: np.ndarray) -> int:
        """
        Submit inference request.

        Args:
            X: Input features

        Returns:
            Request ID
        """
        request_id = self.request_id
        self.request_id += 1

        result_queue = queue.Queue()
        self.result_queues[request_id] = result_queue

        self.input_queue.put((request_id, X))

        return request_id

    def get_result(self, request_id: int, timeout: Optional[float] = None) -> np.ndarray:
        """
        Get inference result.

        Args:
            request_id: Request ID
            timeout: Timeout in seconds

        Returns:
            Prediction result
        """
        result_queue = self.result_queues[request_id]
        result = result_queue.get(timeout=timeout)
        del self.result_queues[request_id]
        return result

    def process_batch(self):
        """Process a batch of requests."""
        batch_requests = []
        batch_inputs = []

        # Collect batch
        start_time = time.time()
        while len(batch_requests) < self.batch_size:
            try:
                timeout_remaining = max(0, self.timeout - (time.time() - start_time))
                request_id, X = self.input_queue.get(timeout=timeout_remaining)
                batch_requests.append(request_id)
                batch_inputs.append(X)
            except queue.Empty:
                break

        if not batch_requests:
            return

        # Batch inference
        X_batch = np.vstack(batch_inputs)
        predictions = self.model.predict(X_batch)

        # Distribute results
        for i, request_id in enumerate(batch_requests):
            self.result_queues[request_id].put(predictions[i])
