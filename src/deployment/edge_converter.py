"""
Edge Model Converter

Convert trained models to formats suitable for edge deployment.
"""

import numpy as np
import json
import pickle
from typing import Any, Dict, Optional, List
from enum import Enum
from dataclasses import dataclass, asdict


class EdgeFormat(Enum):
    """Edge deployment formats."""
    ONNX = "onnx"  # Open Neural Network Exchange
    TFLITE = "tflite"  # TensorFlow Lite
    COREML = "coreml"  # Apple Core ML
    SKLEARN_LITE = "sklearn_lite"  # Lightweight sklearn format
    C_CODE = "c_code"  # C code generation


@dataclass
class EdgeModelMetadata:
    """Metadata for edge models."""
    model_type: str
    input_shape: tuple
    output_shape: tuple
    feature_names: List[str]
    quantized: bool
    compression_ratio: float
    target_platform: str


class EdgeModelConverter:
    """
    Convert ML models to edge-friendly formats.

    Optimizes models for deployment on resource-constrained devices.
    """

    def __init__(self, target_format: EdgeFormat = EdgeFormat.SKLEARN_LITE):
        """
        Initialize edge model converter.

        Args:
            target_format: Target format for conversion
        """
        self.target_format = target_format
        self.metadata: Optional[EdgeModelMetadata] = None

    def convert(self, model: Any, input_shape: tuple,
               feature_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Convert model to edge format.

        Args:
            model: Trained sklearn model
            input_shape: Shape of input data
            feature_names: Names of input features

        Returns:
            Dictionary with converted model and metadata
        """
        if self.target_format == EdgeFormat.SKLEARN_LITE:
            return self._convert_sklearn_lite(model, input_shape, feature_names)
        elif self.target_format == EdgeFormat.C_CODE:
            return self._convert_to_c_code(model, input_shape, feature_names)
        else:
            raise NotImplementedError(f"Format {self.target_format} not implemented")

    def _convert_sklearn_lite(self, model: Any, input_shape: tuple,
                             feature_names: Optional[List[str]]) -> Dict[str, Any]:
        """
        Convert to lightweight sklearn format.

        Extracts only necessary parameters for inference.

        Args:
            model: sklearn model
            input_shape: Input shape
            feature_names: Feature names

        Returns:
            Converted model dictionary
        """
        lite_model = {
            'model_type': type(model).__name__,
            'parameters': {},
            'metadata': {
                'input_shape': input_shape,
                'feature_names': feature_names or []
            }
        }

        # Extract parameters
        if hasattr(model, 'coef_'):
            lite_model['parameters']['coef'] = model.coef_.tolist()

        if hasattr(model, 'intercept_'):
            lite_model['parameters']['intercept'] = model.intercept_.tolist()

        if hasattr(model, 'classes_'):
            lite_model['parameters']['classes'] = model.classes_.tolist()

        # For tree-based models (simplified)
        if hasattr(model, 'tree_'):
            lite_model['parameters']['tree_structure'] = 'simplified'

        return lite_model

    def _convert_to_c_code(self, model: Any, input_shape: tuple,
                          feature_names: Optional[List[str]]) -> Dict[str, Any]:
        """
        Generate C code for model inference.

        Useful for embedded systems without Python runtime.

        Args:
            model: sklearn model
            input_shape: Input shape
            feature_names: Feature names

        Returns:
            Dictionary with C code and headers
        """
        c_code = []
        headers = []

        # Generate header
        headers.append("#include <stdio.h>")
        headers.append("#include <stdlib.h>")
        headers.append("#include <math.h>")
        headers.append("")

        # For linear models
        if hasattr(model, 'coef_') and hasattr(model, 'intercept_'):
            c_code.extend(self._generate_linear_model_c_code(model, input_shape))

        # Generate main function
        c_code.append("")
        c_code.append("// Example usage:")
        c_code.append("// float features[] = {1.0, 2.0, 3.0};")
        c_code.append("// float prediction = predict(features);")

        return {
            'headers': '\n'.join(headers),
            'code': '\n'.join(c_code),
            'model_type': type(model).__name__
        }

    def _generate_linear_model_c_code(self, model: Any, input_shape: tuple) -> List[str]:
        """
        Generate C code for linear models.

        Args:
            model: Linear model
            input_shape: Input shape

        Returns:
            List of C code lines
        """
        code = []

        n_features = model.coef_.shape[1] if model.coef_.ndim > 1 else model.coef_.shape[0]

        # Define coefficients
        code.append(f"// Model coefficients")
        code.append(f"const int N_FEATURES = {n_features};")
        code.append("")

        if model.coef_.ndim > 1:
            # Multi-class
            n_classes = model.coef_.shape[0]
            code.append(f"const int N_CLASSES = {n_classes};")
            code.append("")
            code.append(f"const float COEF[{n_classes}][{n_features}] = {{")
            for i, row in enumerate(model.coef_):
                row_str = ', '.join(f'{val:.6f}f' for val in row)
                code.append(f"    {{{row_str}}},")
            code.append("};")
            code.append("")

            # Intercepts
            code.append(f"const float INTERCEPT[{n_classes}] = {{")
            intercept_str = ', '.join(f'{val:.6f}f' for val in model.intercept_)
            code.append(f"    {intercept_str}")
            code.append("};")

        else:
            # Binary or regression
            code.append(f"const float COEF[{n_features}] = {{")
            coef_str = ', '.join(f'{val:.6f}f' for val in model.coef_)
            code.append(f"    {coef_str}")
            code.append("};")
            code.append("")
            code.append(f"const float INTERCEPT = {model.intercept_[0]:.6f}f;")

        code.append("")

        # Generate prediction function
        if model.coef_.ndim > 1:
            # Multi-class
            code.append("int predict(float* features) {")
            code.append("    float scores[N_CLASSES];")
            code.append("    int max_class = 0;")
            code.append("    float max_score = -INFINITY;")
            code.append("")
            code.append("    for (int c = 0; c < N_CLASSES; c++) {")
            code.append("        scores[c] = INTERCEPT[c];")
            code.append("        for (int f = 0; f < N_FEATURES; f++) {")
            code.append("            scores[c] += COEF[c][f] * features[f];")
            code.append("        }")
            code.append("        if (scores[c] > max_score) {")
            code.append("            max_score = scores[c];")
            code.append("            max_class = c;")
            code.append("        }")
            code.append("    }")
            code.append("    return max_class;")
            code.append("}")
        else:
            # Binary or regression
            code.append("float predict(float* features) {")
            code.append("    float score = INTERCEPT;")
            code.append("    for (int i = 0; i < N_FEATURES; i++) {")
            code.append("        score += COEF[i] * features[i];")
            code.append("    }")
            code.append("    return score;")
            code.append("}")

        return code

    def save_edge_model(self, converted_model: Dict[str, Any], filepath: str):
        """
        Save converted edge model to file.

        Args:
            converted_model: Converted model dictionary
            filepath: Path to save model
        """
        if self.target_format == EdgeFormat.C_CODE:
            # Save as C source file
            with open(filepath, 'w') as f:
                f.write(converted_model['headers'])
                f.write('\n\n')
                f.write(converted_model['code'])
        else:
            # Save as JSON
            with open(filepath, 'w') as f:
                json.dump(converted_model, f, indent=2)

    def load_edge_model(self, filepath: str) -> Dict[str, Any]:
        """
        Load edge model from file.

        Args:
            filepath: Path to model file

        Returns:
            Loaded model dictionary
        """
        if self.target_format == EdgeFormat.C_CODE:
            with open(filepath, 'r') as f:
                return {'code': f.read()}
        else:
            with open(filepath, 'r') as f:
                return json.load(f)


class LitePredictor:
    """
    Lightweight predictor for edge models.

    Minimal dependencies, suitable for embedded systems.
    """

    def __init__(self, model_dict: Dict[str, Any]):
        """
        Initialize lite predictor.

        Args:
            model_dict: Model dictionary from edge converter
        """
        self.model_type = model_dict['model_type']
        self.parameters = model_dict['parameters']
        self.metadata = model_dict.get('metadata', {})

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using lite model.

        Args:
            X: Input features

        Returns:
            Predictions
        """
        if 'coef' in self.parameters:
            return self._linear_predict(X)
        else:
            raise NotImplementedError(f"Prediction for {self.model_type} not implemented")

    def _linear_predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict using linear model parameters.

        Args:
            X: Input features

        Returns:
            Predictions
        """
        coef = np.array(self.parameters['coef'])
        intercept = np.array(self.parameters['intercept'])

        # Compute scores
        scores = X @ coef.T + intercept

        # For classification, return argmax
        if len(scores.shape) > 1 and scores.shape[1] > 1:
            return np.argmax(scores, axis=1)
        else:
            return scores.ravel()


class ModelProfiler:
    """
    Profile model performance on edge devices.

    Measures inference time, memory usage, and accuracy.
    """

    def __init__(self):
        """Initialize model profiler."""
        self.profile_results = {}

    def profile_inference(self, model: Any, X_test: np.ndarray,
                         n_iterations: int = 100) -> Dict[str, float]:
        """
        Profile inference performance.

        Args:
            model: Model to profile
            X_test: Test data
            n_iterations: Number of inference iterations

        Returns:
            Profiling results
        """
        import time

        # Warm-up
        _ = model.predict(X_test[:1])

        # Profile inference time
        start_time = time.time()
        for _ in range(n_iterations):
            _ = model.predict(X_test)
        total_time = time.time() - start_time

        avg_time = total_time / n_iterations
        throughput = len(X_test) / avg_time  # samples per second

        results = {
            'avg_inference_time_ms': avg_time * 1000,
            'throughput_samples_per_sec': throughput,
            'total_samples': len(X_test),
            'iterations': n_iterations
        }

        self.profile_results = results
        return results

    def estimate_memory_usage(self, model: Any) -> Dict[str, int]:
        """
        Estimate model memory footprint.

        Args:
            model: Model to analyze

        Returns:
            Memory usage estimates
        """
        total_bytes = 0
        param_count = 0

        # Count parameters
        if hasattr(model, 'coef_'):
            total_bytes += model.coef_.nbytes
            param_count += model.coef_.size

        if hasattr(model, 'intercept_'):
            total_bytes += model.intercept_.nbytes
            param_count += model.intercept_.size

        return {
            'total_bytes': total_bytes,
            'total_kb': total_bytes / 1024,
            'total_mb': total_bytes / (1024 * 1024),
            'parameter_count': param_count
        }
