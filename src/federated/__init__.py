"""
Federated Learning Module

Implements federated learning for privacy-preserving distributed training
across multiple clients without sharing raw data.
"""

from src.federated.server import FederatedServer, AggregationStrategy
from src.federated.client import FederatedClient
from src.federated.secure_aggregation import SecureAggregator

__all__ = [
    'FederatedServer',
    'FederatedClient',
    'AggregationStrategy',
    'SecureAggregator'
]
