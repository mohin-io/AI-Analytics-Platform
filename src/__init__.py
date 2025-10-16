"""
Unified AI Analytics Platform

A comprehensive machine learning model benchmarking and analytics system.
"""

__version__ = "0.1.0"
__author__ = "AI/ML Engineering Team"
__license__ = "MIT"

from src.utils.logger import setup_logger
from src.utils.config import Config

__all__ = ["setup_logger", "Config", "__version__"]