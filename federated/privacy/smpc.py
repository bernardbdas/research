"""Secure Multi-Party Computation: Shamir's (t, n)-secret sharing and Bonawitz SecAgg."""

import random

import numpy as np


class ShamirSecretSharing:
    """Implementation of Shamir's (t, n)-threshold secret sharing scheme over prime field Z_p."""

    def __init__(self, prime: int = 2147483647):  # 2^31 - 1 (Mersenne prime)
        self.p = prime

    def split_secret(
        self, secret: int, threshold: int, num_shares: int
    ) -> list[tuple[int, int]]:
        """Split a secret integer into n evaluation shares with threshold t."""
        if threshold > num_shares:
            raise ValueError("Threshold cannot exceed number of shares.")
        if secret < 0 or secret >= self.p:
            secret = secret % self.p

        # Polynomial coefficients: a_0 = secret, a_1..a_{t-1} random
        coeffs = [secret] + [
            random.randint(1, self.p - 1) for _ in range(threshold - 1)
        ]

        shares = []
        for x in range(1, num_shares + 1):
            y = sum(c * pow(x, idx, self.p) for idx, c in enumerate(coeffs)) % self.p
            shares.append((x, y))
        return shares

    def reconstruct_secret(self, shares: list[tuple[int, int]]) -> int:
        """Reconstruct the secret from any t shares via Lagrange polynomial interpolation."""
        secret = 0

        for i, (xi, yi) in enumerate(shares):
            # Compute Lagrange basis polynomial L_i(0)
            numerator = 1
            denominator = 1
            for j, (xj, _) in enumerate(shares):
                if i != j:
                    numerator = (numerator * (-xj)) % self.p
                    denominator = (denominator * (xi - xj)) % self.p

            inv_denom = pow(denominator, -1, self.p)
            lagrange_basis = (numerator * inv_denom) % self.p
            secret = (secret + yi * lagrange_basis) % self.p

        return secret


def pseudorandom_generator(seed: int, dimension: int) -> np.ndarray:
    """Cryptographic PRG: generates deterministic pseudo-random float vector from integer seed."""
    rng = np.random.RandomState(seed % (2**32))
    return rng.normal(loc=0.0, scale=1.0, size=dimension)


class SecAggClient:
    """Simulated federated client participating in Secure Aggregation protocol."""

    def __init__(self, client_id: int, total_clients: int, param_dim: int):
        self.id = client_id
        self.K = total_clients
        self.dim = param_dim
        # Private key for pairwise Diffie-Hellman
        self.sk = random.randint(1000, 99999)
        # Pairwise seeds negotiated with other clients
        self.pairwise_seeds: dict[int, int] = {}

    def setup_pairwise_seed(self, peer_id: int, peer_sk: int):
        """Simulate Diffie-Hellman shared key derivation: H(sk_u * sk_v)."""
        shared_secret = (self.sk * peer_sk) % 1000000007
        self.pairwise_seeds[peer_id] = shared_secret

    def generate_masked_update(self, raw_weights: np.ndarray) -> np.ndarray:
        """Mask the local weight vector with zero-sum pairwise perturbations."""
        masked_vector = np.copy(raw_weights)

        for peer_id, seed in self.pairwise_seeds.items():
            pairwise_noise = pseudorandom_generator(seed, self.dim)
            if self.id < peer_id:
                masked_vector += pairwise_noise
            else:
                masked_vector -= pairwise_noise

        return masked_vector


def generate_pairwise_masks(
    client_id: int, num_clients: int, param_dim: int
) -> np.ndarray:
    """Generate zero-sum SecAgg pairwise masks for client u."""
    total_mask = np.zeros(param_dim, dtype=np.float32)
    for peer_id in range(num_clients):
        if peer_id == client_id:
            continue
        seed = min(client_id, peer_id) * 10007 + max(client_id, peer_id)
        rng = np.random.RandomState(seed)
        mask = rng.normal(loc=0.0, scale=1.0, size=param_dim).astype(np.float32)
        if client_id < peer_id:
            total_mask += mask
        else:
            total_mask -= mask
    return total_mask
