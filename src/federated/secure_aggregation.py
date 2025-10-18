"""
Secure Aggregation for Federated Learning

Implements secure multi-party computation for aggregating client updates
without revealing individual client data to the server.
"""

import numpy as np
from typing import List, Dict, Tuple
import hashlib
import secrets


class SecureAggregator:
    """
    Secure aggregation using secret sharing.

    Allows server to compute aggregate of client updates without
    seeing individual updates.
    """

    def __init__(self, n_clients: int, threshold: int):
        """
        Initialize secure aggregator.

        Args:
            n_clients: Total number of clients
            threshold: Minimum clients needed for reconstruction
        """
        self.n_clients = n_clients
        self.threshold = threshold
        self.client_shares: Dict[str, List[np.ndarray]] = {}

    def create_shares(self, value: np.ndarray, client_id: str) -> List[np.ndarray]:
        """
        Create secret shares for a value using Shamir's Secret Sharing.

        Args:
            value: Value to share
            client_id: Client identifier

        Returns:
            List of shares (one per client)
        """
        # Simplified secret sharing (for demonstration)
        # In production, use proper Shamir's Secret Sharing

        shares = []
        random_shares = [np.random.randn(*value.shape)
                        for _ in range(self.n_clients - 1)]

        # Last share is computed so sum = value
        last_share = value - sum(random_shares)
        random_shares.append(last_share)

        return random_shares

    def aggregate_shares(self, shares: List[np.ndarray]) -> np.ndarray:
        """
        Aggregate shares to recover original sum.

        Args:
            shares: List of shares from different clients

        Returns:
            Aggregated value
        """
        if len(shares) < self.threshold:
            raise ValueError(f"Not enough shares: {len(shares)} < {self.threshold}")

        # Simple summation (works with our simplified sharing)
        return sum(shares)


class HomomorphicEncryption:
    """
    Simplified homomorphic encryption for secure aggregation.

    Allows computation on encrypted values.
    """

    def __init__(self, key_size: int = 2048):
        """
        Initialize homomorphic encryption.

        Args:
            key_size: Size of encryption key
        """
        self.key_size = key_size
        self.public_key = None
        self.private_key = None
        self._generate_keys()

    def _generate_keys(self):
        """Generate public/private key pair."""
        # Simplified key generation (not secure for production)
        # In production, use libraries like python-paillier
        self.public_key = secrets.randbits(self.key_size)
        self.private_key = secrets.randbits(self.key_size)

    def encrypt(self, value: float) -> int:
        """
        Encrypt a value.

        Args:
            value: Value to encrypt

        Returns:
            Encrypted value
        """
        # Simplified encryption (demonstration only)
        # In production, use Paillier or similar homomorphic scheme
        scaled_value = int(value * 1e6)  # Scale for integer operations
        encrypted = (scaled_value + self.public_key) % (2**self.key_size)
        return encrypted

    def decrypt(self, encrypted_value: int) -> float:
        """
        Decrypt a value.

        Args:
            encrypted_value: Encrypted value

        Returns:
            Decrypted value
        """
        # Simplified decryption
        decrypted = (encrypted_value - self.public_key) % (2**self.key_size)
        # Handle large values (wrap-around)
        if decrypted > 2**(self.key_size - 1):
            decrypted -= 2**self.key_size
        return decrypted / 1e6

    def add_encrypted(self, enc1: int, enc2: int) -> int:
        """
        Add two encrypted values (homomorphic property).

        Args:
            enc1: First encrypted value
            enc2: Second encrypted value

        Returns:
            Encrypted sum
        """
        return (enc1 + enc2) % (2**self.key_size)


class SecureAggregationProtocol:
    """
    Complete secure aggregation protocol.

    Combines secret sharing and masking for privacy-preserving aggregation.
    """

    def __init__(self, client_ids: List[str]):
        """
        Initialize secure aggregation protocol.

        Args:
            client_ids: List of participating client IDs
        """
        self.client_ids = client_ids
        self.n_clients = len(client_ids)
        self.masks: Dict[str, np.ndarray] = {}
        self.masked_updates: Dict[str, np.ndarray] = {}

    def generate_pairwise_masks(self, shape: Tuple) -> Dict[Tuple[str, str], np.ndarray]:
        """
        Generate pairwise masks between clients.

        Each pair of clients shares a random mask that cancels out
        when summed across all clients.

        Args:
            shape: Shape of the values to mask

        Returns:
            Dictionary of pairwise masks
        """
        pairwise_masks = {}

        for i, client_i in enumerate(self.client_ids):
            for j, client_j in enumerate(self.client_ids[i+1:], start=i+1):
                # Generate random mask for this pair
                mask = np.random.randn(*shape)

                # Store with ordering: mask for (i,j) = -mask for (j,i)
                pairwise_masks[(client_i, client_j)] = mask
                pairwise_masks[(client_j, client_i)] = -mask

        return pairwise_masks

    def mask_client_update(self, client_id: str, update: np.ndarray,
                          pairwise_masks: Dict[Tuple[str, str], np.ndarray]) -> np.ndarray:
        """
        Mask a client's update with pairwise masks.

        Args:
            client_id: Client ID
            update: Client's model update
            pairwise_masks: Pairwise masks

        Returns:
            Masked update
        """
        masked_update = update.copy()

        # Add all pairwise masks for this client
        for other_client in self.client_ids:
            if other_client != client_id:
                key = (client_id, other_client)
                if key in pairwise_masks:
                    masked_update += pairwise_masks[key]

        self.masked_updates[client_id] = masked_update
        return masked_update

    def aggregate_masked_updates(self) -> np.ndarray:
        """
        Aggregate masked updates.

        The pairwise masks cancel out, revealing only the sum.

        Returns:
            Aggregated update (sum of original updates)
        """
        if not self.masked_updates:
            raise ValueError("No masked updates available")

        # Sum all masked updates
        # Pairwise masks cancel: mask(i,j) + mask(j,i) = 0
        aggregated = sum(self.masked_updates.values())

        return aggregated

    def secure_aggregate(self, client_updates: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Perform complete secure aggregation.

        Args:
            client_updates: Dictionary mapping client IDs to their updates

        Returns:
            Aggregated update without revealing individual updates
        """
        if not client_updates:
            raise ValueError("No client updates provided")

        # Get shape from first update
        first_update = next(iter(client_updates.values()))
        shape = first_update.shape

        # Generate pairwise masks
        pairwise_masks = self.generate_pairwise_masks(shape)

        # Each client masks their update
        for client_id, update in client_updates.items():
            self.mask_client_update(client_id, update, pairwise_masks)

        # Aggregate masked updates
        aggregated = self.aggregate_masked_updates()

        return aggregated


class DifferentialPrivacyMechanism:
    """
    Differential privacy mechanisms for federated learning.
    """

    @staticmethod
    def gaussian_mechanism(value: np.ndarray, sensitivity: float,
                          epsilon: float, delta: float = 1e-5) -> np.ndarray:
        """
        Add Gaussian noise for (epsilon, delta)-differential privacy.

        Args:
            value: True value
            sensitivity: L2 sensitivity of the computation
            epsilon: Privacy parameter
            delta: Failure probability

        Returns:
            Noisy value with DP guarantee
        """
        # Calculate noise scale using Gaussian mechanism
        sigma = sensitivity * np.sqrt(2 * np.log(1.25 / delta)) / epsilon

        # Add Gaussian noise
        noise = np.random.normal(0, sigma, value.shape)
        noisy_value = value + noise

        return noisy_value

    @staticmethod
    def laplace_mechanism(value: np.ndarray, sensitivity: float,
                         epsilon: float) -> np.ndarray:
        """
        Add Laplace noise for epsilon-differential privacy.

        Args:
            value: True value
            sensitivity: L1 sensitivity
            epsilon: Privacy parameter

        Returns:
            Noisy value with DP guarantee
        """
        # Calculate scale for Laplace noise
        scale = sensitivity / epsilon

        # Add Laplace noise
        noise = np.random.laplace(0, scale, value.shape)
        noisy_value = value + noise

        return noisy_value

    @staticmethod
    def clip_and_add_noise(gradients: np.ndarray, clip_norm: float,
                          noise_multiplier: float) -> np.ndarray:
        """
        Clip gradients and add noise (DP-SGD style).

        Args:
            gradients: Gradient values
            clip_norm: Clipping threshold
            noise_multiplier: Noise scale multiplier

        Returns:
            Clipped and noised gradients
        """
        # Clip gradient norm
        grad_norm = np.linalg.norm(gradients)
        if grad_norm > clip_norm:
            gradients = gradients * (clip_norm / grad_norm)

        # Add Gaussian noise
        noise = np.random.normal(0, clip_norm * noise_multiplier, gradients.shape)
        noisy_gradients = gradients + noise

        return noisy_gradients
