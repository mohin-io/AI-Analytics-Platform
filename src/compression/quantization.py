"""
Model Quantization

Reduces model precision to decrease size and speed up inference.
"""

import numpy as np
from typing import Any, Dict, Optional, Tuple
from enum import Enum
import pickle


class QuantizationStrategy(Enum):
    """Quantization strategies."""
    INT8 = "int8"  # 8-bit integer quantization
    INT16 = "int16"  # 16-bit integer quantization
    FLOAT16 = "float16"  # 16-bit floating point
    DYNAMIC = "dynamic"  # Dynamic quantization


class ModelQuantizer:
    """
    Quantize model weights to reduce size and improve inference speed.

    Supports various quantization strategies for sklearn models.
    """

    def __init__(self, strategy: QuantizationStrategy = QuantizationStrategy.INT8):
        """
        Initialize model quantizer.

        Args:
            strategy: Quantization strategy to use
        """
        self.strategy = strategy
        self.scale_factors: Dict[str, float] = {}
        self.zero_points: Dict[str, int] = {}

    def quantize_model(self, model: Any) -> Any:
        """
        Quantize a trained model.

        Args:
            model: Trained sklearn model

        Returns:
            Quantized model
        """
        quantized_model = model

        # Quantize weights
        if hasattr(model, 'coef_'):
            quantized_model.coef_, scale, zero_point = self._quantize_weights(
                model.coef_
            )
            self.scale_factors['coef_'] = scale
            self.zero_points['coef_'] = zero_point

        if hasattr(model, 'intercept_'):
            quantized_model.intercept_, scale, zero_point = self._quantize_weights(
                model.intercept_
            )
            self.scale_factors['intercept_'] = scale
            self.zero_points['intercept_'] = zero_point

        # For tree-based models, quantize feature importances
        if hasattr(model, 'feature_importances_'):
            quantized_model.feature_importances_, scale, zero_point = self._quantize_weights(
                model.feature_importances_
            )
            self.scale_factors['feature_importances_'] = scale
            self.zero_points['feature_importances_'] = zero_point

        return quantized_model

    def _quantize_weights(self, weights: np.ndarray) -> Tuple[np.ndarray, float, int]:
        """
        Quantize weight array.

        Args:
            weights: Weight array to quantize

        Returns:
            Tuple of (quantized_weights, scale_factor, zero_point)
        """
        if self.strategy == QuantizationStrategy.INT8:
            return self._quantize_int8(weights)
        elif self.strategy == QuantizationStrategy.INT16:
            return self._quantize_int16(weights)
        elif self.strategy == QuantizationStrategy.FLOAT16:
            return self._quantize_float16(weights)
        else:
            return self._quantize_int8(weights)

    def _quantize_int8(self, weights: np.ndarray) -> Tuple[np.ndarray, float, int]:
        """
        Quantize to 8-bit integers.

        Args:
            weights: Float weights

        Returns:
            Quantized weights, scale, zero_point
        """
        # Calculate min and max
        w_min = np.min(weights)
        w_max = np.max(weights)

        # Calculate scale and zero point
        qmin, qmax = -128, 127  # int8 range
        scale = (w_max - w_min) / (qmax - qmin)

        if scale == 0:
            scale = 1.0

        zero_point = int(np.round(qmin - w_min / scale))
        zero_point = np.clip(zero_point, qmin, qmax)

        # Quantize
        quantized = np.round(weights / scale + zero_point)
        quantized = np.clip(quantized, qmin, qmax).astype(np.int8)

        return quantized, scale, zero_point

    def _quantize_int16(self, weights: np.ndarray) -> Tuple[np.ndarray, float, int]:
        """
        Quantize to 16-bit integers.

        Args:
            weights: Float weights

        Returns:
            Quantized weights, scale, zero_point
        """
        w_min = np.min(weights)
        w_max = np.max(weights)

        qmin, qmax = -32768, 32767  # int16 range
        scale = (w_max - w_min) / (qmax - qmin)

        if scale == 0:
            scale = 1.0

        zero_point = int(np.round(qmin - w_min / scale))
        zero_point = np.clip(zero_point, qmin, qmax)

        quantized = np.round(weights / scale + zero_point)
        quantized = np.clip(quantized, qmin, qmax).astype(np.int16)

        return quantized, scale, zero_point

    def _quantize_float16(self, weights: np.ndarray) -> Tuple[np.ndarray, float, int]:
        """
        Convert to 16-bit float.

        Args:
            weights: Float32 weights

        Returns:
            Float16 weights, scale=1.0, zero_point=0
        """
        quantized = weights.astype(np.float16)
        return quantized, 1.0, 0

    def dequantize_weights(self, quantized_weights: np.ndarray,
                          param_name: str) -> np.ndarray:
        """
        Dequantize weights back to float32.

        Args:
            quantized_weights: Quantized weights
            param_name: Parameter name ('coef_', 'intercept_', etc.)

        Returns:
            Dequantized weights
        """
        if param_name not in self.scale_factors:
            return quantized_weights.astype(np.float32)

        scale = self.scale_factors[param_name]
        zero_point = self.zero_points[param_name]

        # Dequantize: w_float = scale * (w_quant - zero_point)
        dequantized = scale * (quantized_weights.astype(np.float32) - zero_point)

        return dequantized

    def get_model_size(self, model: Any) -> Dict[str, int]:
        """
        Calculate model size before and after quantization.

        Args:
            model: Model to analyze

        Returns:
            Dictionary with size information
        """
        total_params = 0
        total_bytes_original = 0
        total_bytes_quantized = 0

        if hasattr(model, 'coef_'):
            params = model.coef_.size
            total_params += params

            # Original: float32 = 4 bytes per param
            total_bytes_original += params * 4

            # Quantized size depends on strategy
            if self.strategy == QuantizationStrategy.INT8:
                total_bytes_quantized += params * 1
            elif self.strategy == QuantizationStrategy.INT16:
                total_bytes_quantized += params * 2
            elif self.strategy == QuantizationStrategy.FLOAT16:
                total_bytes_quantized += params * 2

        if hasattr(model, 'intercept_'):
            params = model.intercept_.size
            total_params += params
            total_bytes_original += params * 4

            if self.strategy == QuantizationStrategy.INT8:
                total_bytes_quantized += params * 1
            elif self.strategy == QuantizationStrategy.INT16:
                total_bytes_quantized += params * 2
            elif self.strategy == QuantizationStrategy.FLOAT16:
                total_bytes_quantized += params * 2

        compression_ratio = total_bytes_original / total_bytes_quantized if total_bytes_quantized > 0 else 1.0

        return {
            'total_parameters': total_params,
            'original_size_bytes': total_bytes_original,
            'quantized_size_bytes': total_bytes_quantized,
            'compression_ratio': compression_ratio,
            'size_reduction_percent': (1 - total_bytes_quantized / total_bytes_original) * 100 if total_bytes_original > 0 else 0
        }

    def save_quantized_model(self, model: Any, filepath: str):
        """
        Save quantized model to file.

        Args:
            model: Quantized model
            filepath: Path to save model
        """
        model_data = {
            'model': model,
            'scale_factors': self.scale_factors,
            'zero_points': self.zero_points,
            'strategy': self.strategy
        }

        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)

    def load_quantized_model(self, filepath: str) -> Any:
        """
        Load quantized model from file.

        Args:
            filepath: Path to load model from

        Returns:
            Quantized model
        """
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)

        self.scale_factors = model_data['scale_factors']
        self.zero_points = model_data['zero_points']
        self.strategy = model_data['strategy']

        return model_data['model']


class DynamicQuantizer:
    """
    Dynamic quantization that adapts based on input range.

    Useful for models with varying input distributions.
    """

    def __init__(self, bits: int = 8):
        """
        Initialize dynamic quantizer.

        Args:
            bits: Number of bits for quantization
        """
        self.bits = bits
        self.qmin = -(2 ** (bits - 1))
        self.qmax = 2 ** (bits - 1) - 1

    def quantize_activations(self, activations: np.ndarray) -> Tuple[np.ndarray, float, int]:
        """
        Dynamically quantize activations based on their range.

        Args:
            activations: Activation values

        Returns:
            Quantized activations, scale, zero_point
        """
        a_min = np.min(activations)
        a_max = np.max(activations)

        scale = (a_max - a_min) / (self.qmax - self.qmin)

        if scale == 0:
            scale = 1.0

        zero_point = int(np.round(self.qmin - a_min / scale))
        zero_point = np.clip(zero_point, self.qmin, self.qmax)

        # Quantize
        quantized = np.round(activations / scale + zero_point)
        quantized = np.clip(quantized, self.qmin, self.qmax)

        if self.bits == 8:
            quantized = quantized.astype(np.int8)
        elif self.bits == 16:
            quantized = quantized.astype(np.int16)

        return quantized, scale, zero_point

    def dequantize_activations(self, quantized: np.ndarray,
                               scale: float, zero_point: int) -> np.ndarray:
        """
        Dequantize activations.

        Args:
            quantized: Quantized values
            scale: Scale factor
            zero_point: Zero point

        Returns:
            Dequantized activations
        """
        return scale * (quantized.astype(np.float32) - zero_point)
