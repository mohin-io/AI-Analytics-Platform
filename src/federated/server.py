"""
Federated Learning Server

Coordinates federated training across multiple clients using various
aggregation strategies (FedAvg, FedProx, etc.)
"""

import numpy as np
import pickle
from typing import List, Dict, Optional, Any, Callable
from enum import Enum
from dataclasses import dataclass
import copy


class AggregationStrategy(Enum):
    """Aggregation strategies for federated learning."""
    FED_AVG = "fedavg"  # Federated Averaging
    FED_PROX = "fedprox"  # Federated Proximal
    FED_ADAM = "fedadam"  # Federated Adam
    WEIGHTED_AVG = "weighted_avg"  # Weighted by dataset size


@dataclass
class ClientUpdate:
    """Container for client model updates."""
    client_id: str
    model_weights: Dict[str, np.ndarray]
    n_samples: int
    loss: float
    metrics: Dict[str, float]


class FederatedServer:
    """
    Federated Learning Server.

    Coordinates training across multiple clients, aggregates their updates,
    and maintains the global model.
    """

    def __init__(self, initial_model: Any,
                 aggregation_strategy: AggregationStrategy = AggregationStrategy.FED_AVG,
                 min_clients_per_round: int = 2,
                 client_fraction: float = 1.0):
        """
        Initialize federated server.

        Args:
            initial_model: Initial global model (sklearn-compatible)
            aggregation_strategy: Strategy for aggregating client updates
            min_clients_per_round: Minimum number of clients required per round
            client_fraction: Fraction of clients to sample per round
        """
        self.global_model = initial_model
        self.aggregation_strategy = aggregation_strategy
        self.min_clients_per_round = min_clients_per_round
        self.client_fraction = client_fraction

        self.round_number = 0
        self.client_updates: List[ClientUpdate] = []
        self.training_history: List[Dict[str, Any]] = []
        self.registered_clients: List[str] = []

    def register_client(self, client_id: str):
        """
        Register a client with the server.

        Args:
            client_id: Unique client identifier
        """
        if client_id not in self.registered_clients:
            self.registered_clients.append(client_id)

    def get_global_model(self) -> Any:
        """
        Get the current global model.

        Returns:
            Copy of global model
        """
        return copy.deepcopy(self.global_model)

    def receive_update(self, client_update: ClientUpdate):
        """
        Receive and store update from a client.

        Args:
            client_update: Client's model update
        """
        self.client_updates.append(client_update)

    def aggregate_updates(self) -> bool:
        """
        Aggregate client updates into global model.

        Returns:
            True if aggregation successful, False otherwise
        """
        if len(self.client_updates) < self.min_clients_per_round:
            print(f"Not enough client updates: {len(self.client_updates)} < {self.min_clients_per_round}")
            return False

        if self.aggregation_strategy == AggregationStrategy.FED_AVG:
            self._federated_averaging()
        elif self.aggregation_strategy == AggregationStrategy.WEIGHTED_AVG:
            self._weighted_averaging()
        elif self.aggregation_strategy == AggregationStrategy.FED_PROX:
            self._federated_proximal()
        else:
            self._federated_averaging()  # Default to FedAvg

        # Record training history
        self._record_round_metrics()

        # Clear client updates for next round
        self.client_updates = []
        self.round_number += 1

        return True

    def _federated_averaging(self):
        """
        FedAvg: Average model weights across all clients.

        Simple averaging where each client contributes equally regardless
        of dataset size.
        """
        n_clients = len(self.client_updates)

        # Extract model weights (assuming sklearn models with coef_ and intercept_)
        if hasattr(self.global_model, 'coef_'):
            # Linear models
            avg_coef = np.zeros_like(self.global_model.coef_)
            avg_intercept = 0.0

            for update in self.client_updates:
                if 'coef_' in update.model_weights:
                    avg_coef += update.model_weights['coef_']
                if 'intercept_' in update.model_weights:
                    avg_intercept += update.model_weights['intercept_']

            self.global_model.coef_ = avg_coef / n_clients

            if hasattr(self.global_model, 'intercept_'):
                self.global_model.intercept_ = avg_intercept / n_clients

        elif hasattr(self.global_model, 'estimators_'):
            # Ensemble models (more complex, simplified here)
            print("Warning: Ensemble model averaging not fully implemented")

    def _weighted_averaging(self):
        """
        Weighted averaging by dataset size.

        Clients with more data have proportionally more influence on
        the global model update.
        """
        total_samples = sum(update.n_samples for update in self.client_updates)

        if hasattr(self.global_model, 'coef_'):
            # Linear models
            weighted_coef = np.zeros_like(self.global_model.coef_)
            weighted_intercept = 0.0

            for update in self.client_updates:
                weight = update.n_samples / total_samples

                if 'coef_' in update.model_weights:
                    weighted_coef += update.model_weights['coef_'] * weight
                if 'intercept_' in update.model_weights:
                    weighted_intercept += update.model_weights['intercept_'] * weight

            self.global_model.coef_ = weighted_coef

            if hasattr(self.global_model, 'intercept_'):
                self.global_model.intercept_ = weighted_intercept

    def _federated_proximal(self):
        """
        FedProx: Federated averaging with proximal term.

        Similar to FedAvg but adds a proximal term to handle
        heterogeneous data across clients.
        """
        # For now, use weighted averaging as base
        # Full FedProx requires modification during client training
        self._weighted_averaging()

    def _record_round_metrics(self):
        """Record metrics for the current round."""
        avg_loss = np.mean([u.loss for u in self.client_updates])
        total_samples = sum(u.n_samples for u in self.client_updates)

        # Aggregate metrics
        aggregated_metrics = {}
        if self.client_updates and self.client_updates[0].metrics:
            for metric_name in self.client_updates[0].metrics.keys():
                # Weighted average of metrics
                metric_values = [u.metrics.get(metric_name, 0) * u.n_samples
                               for u in self.client_updates]
                aggregated_metrics[metric_name] = sum(metric_values) / total_samples

        self.training_history.append({
            'round': self.round_number,
            'n_clients': len(self.client_updates),
            'total_samples': total_samples,
            'avg_loss': avg_loss,
            'metrics': aggregated_metrics
        })

    def get_training_history(self) -> List[Dict[str, Any]]:
        """
        Get training history.

        Returns:
            List of round metrics
        """
        return self.training_history

    def save_global_model(self, filepath: str):
        """
        Save global model to file.

        Args:
            filepath: Path to save model
        """
        with open(filepath, 'wb') as f:
            pickle.dump(self.global_model, f)

    def load_global_model(self, filepath: str):
        """
        Load global model from file.

        Args:
            filepath: Path to load model from
        """
        with open(filepath, 'rb') as f:
            self.global_model = pickle.load(f)


class FederatedCoordinator:
    """
    High-level coordinator for federated learning workflows.

    Manages the complete federated training process including client
    selection, round scheduling, and convergence monitoring.
    """

    def __init__(self, server: FederatedServer,
                 max_rounds: int = 100,
                 convergence_threshold: float = 0.001):
        """
        Initialize federated coordinator.

        Args:
            server: FederatedServer instance
            max_rounds: Maximum number of training rounds
            convergence_threshold: Threshold for early stopping
        """
        self.server = server
        self.max_rounds = max_rounds
        self.convergence_threshold = convergence_threshold
        self.converged = False

    def run_round(self, client_trainers: List[Callable]) -> bool:
        """
        Execute one round of federated training.

        Args:
            client_trainers: List of client training functions

        Returns:
            True if round completed successfully
        """
        # Select clients for this round
        n_clients = max(
            self.server.min_clients_per_round,
            int(len(client_trainers) * self.server.client_fraction)
        )

        selected_indices = np.random.choice(
            len(client_trainers),
            size=min(n_clients, len(client_trainers)),
            replace=False
        )

        # Get global model for clients
        global_model = self.server.get_global_model()

        # Each selected client trains locally
        for idx in selected_indices:
            client_trainer = client_trainers[idx]
            client_update = client_trainer(global_model)
            self.server.receive_update(client_update)

        # Aggregate updates
        success = self.server.aggregate_updates()

        # Check convergence
        if len(self.server.training_history) >= 2:
            prev_loss = self.server.training_history[-2]['avg_loss']
            curr_loss = self.server.training_history[-1]['avg_loss']
            improvement = abs(prev_loss - curr_loss)

            if improvement < self.convergence_threshold:
                self.converged = True

        return success

    def train(self, client_trainers: List[Callable],
             validation_fn: Optional[Callable] = None) -> Dict[str, Any]:
        """
        Run complete federated training.

        Args:
            client_trainers: List of client training functions
            validation_fn: Optional validation function

        Returns:
            Training results
        """
        for round_num in range(self.max_rounds):
            print(f"Round {round_num + 1}/{self.max_rounds}")

            success = self.run_round(client_trainers)

            if not success:
                print("Round failed, skipping...")
                continue

            # Optional validation
            if validation_fn:
                val_metrics = validation_fn(self.server.get_global_model())
                print(f"Validation metrics: {val_metrics}")

            # Check convergence
            if self.converged:
                print(f"Converged after {round_num + 1} rounds")
                break

            # Print round stats
            if self.server.training_history:
                last_round = self.server.training_history[-1]
                print(f"Round {round_num + 1}: Loss={last_round['avg_loss']:.4f}, "
                      f"Clients={last_round['n_clients']}, "
                      f"Samples={last_round['total_samples']}")

        return {
            'rounds_completed': len(self.server.training_history),
            'converged': self.converged,
            'final_loss': self.server.training_history[-1]['avg_loss'] if self.server.training_history else None,
            'history': self.server.training_history
        }


class PrivacyBudgetTracker:
    """
    Track privacy budget for differential privacy in federated learning.

    Implements epsilon-delta differential privacy tracking.
    """

    def __init__(self, epsilon: float = 1.0, delta: float = 1e-5):
        """
        Initialize privacy budget tracker.

        Args:
            epsilon: Privacy parameter (smaller = more private)
            delta: Probability of privacy breach
        """
        self.epsilon_total = epsilon
        self.delta_total = delta
        self.epsilon_spent = 0.0
        self.delta_spent = 0.0
        self.query_count = 0

    def spend_budget(self, epsilon: float, delta: float = 0.0) -> bool:
        """
        Spend privacy budget.

        Args:
            epsilon: Epsilon to spend
            delta: Delta to spend

        Returns:
            True if budget available, False otherwise
        """
        if self.epsilon_spent + epsilon > self.epsilon_total:
            return False

        if self.delta_spent + delta > self.delta_total:
            return False

        self.epsilon_spent += epsilon
        self.delta_spent += delta
        self.query_count += 1

        return True

    def get_remaining_budget(self) -> Dict[str, float]:
        """
        Get remaining privacy budget.

        Returns:
            Dictionary with remaining epsilon and delta
        """
        return {
            'epsilon_remaining': self.epsilon_total - self.epsilon_spent,
            'delta_remaining': self.delta_total - self.delta_spent,
            'epsilon_spent': self.epsilon_spent,
            'delta_spent': self.delta_spent,
            'queries': self.query_count
        }

    def reset(self):
        """Reset privacy budget."""
        self.epsilon_spent = 0.0
        self.delta_spent = 0.0
        self.query_count = 0
