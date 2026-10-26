"""Paillier additive homomorphic cryptosystem with fixed-point tensor encoding."""

import math
import random

import numpy as np


class PaillierPublicKey:
    """Public encryption key for Paillier cryptosystem."""

    def __init__(self, n: int):
        self.n = n
        self.nsquare = n * n
        self.g = n + 1  # Standard canonical base
        self.max_int = n // 3  # Maximum positive integer before treating as negative

    def encrypt(self, plaintext_int: int) -> int:
        """Encrypt an integer into ciphertext c in Z_{n^2}."""
        m = plaintext_int % self.n
        r = random.randint(2, self.n - 1)
        while math.gcd(r, self.n) != 1:
            r = random.randint(2, self.n - 1)
        # c = (g^m * r^n) mod n^2
        c = (pow(self.g, m, self.nsquare) * pow(r, self.n, self.nsquare)) % self.nsquare
        return c

    def add(self, c1: int, c2: int) -> int:
        """Homomorphic addition: E(m1) * E(m2) mod n^2 = E(m1 + m2)."""
        return (c1 * c2) % self.nsquare

    def multiply(self, c: int, scalar: int) -> int:
        """Homomorphic scalar multiplication: E(m)^k mod n^2 = E(k * m)."""
        return pow(c, scalar, self.nsquare)


class PaillierPrivateKey:
    """Private decryption key for Paillier cryptosystem."""

    def __init__(self, p: int, q: int, pubkey: PaillierPublicKey):
        self.pubkey = pubkey
        self.p = p
        self.q = q
        self.lambda_n = int((p - 1) * (q - 1) // math.gcd(p - 1, q - 1))

        # L(u) = (u - 1) // n
        u = pow(pubkey.g, self.lambda_n, pubkey.nsquare)
        l_val = (u - 1) // pubkey.n
        self.mu = pow(l_val, -1, pubkey.n)

    def decrypt(self, ciphertext: int) -> int:
        """Decrypt ciphertext back to signed integer."""
        u = pow(ciphertext, self.lambda_n, self.pubkey.nsquare)
        l_val = (u - 1) // self.pubkey.n
        m = (l_val * self.mu) % self.pubkey.n

        # Decode negative numbers
        if m > self.pubkey.max_int:
            m -= self.pubkey.n
        return m


def generate_paillier_keypair(
    prime_bits: int = 128,
) -> tuple[PaillierPublicKey, PaillierPrivateKey]:
    """Generate a Paillier keypair with primes p, q of specified bit-length."""

    def is_prime(num: int) -> bool:
        if num < 2:
            return False
        for p in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31]:
            if num % p == 0:
                return num == p
        d = num - 1
        s = 0
        while d % 2 == 0:
            d //= 2
            s += 1
        for _ in range(16):
            a = random.randint(2, num - 2)
            x = pow(a, d, num)
            if x == 1 or x == num - 1:
                continue
            for _ in range(s - 1):
                x = pow(x, 2, num)
                if x == num - 1:
                    break
            else:
                return False
        return True

    def get_prime(bits: int) -> int:
        while True:
            candidate = random.getrandbits(bits) | (1 << (bits - 1)) | 1
            if is_prime(candidate):
                return candidate

    p = get_prime(prime_bits)
    q = get_prime(prime_bits)
    while p == q:
        q = get_prime(prime_bits)

    pub = PaillierPublicKey(n=p * q)
    priv = PaillierPrivateKey(p=p, q=q, pubkey=pub)
    return pub, priv


class HomomorphicTensorCodec:
    """Encodes continuous floating-point tensors into fixed-point integers for Paillier encryption."""

    def __init__(self, pubkey: PaillierPublicKey, precision_scale: int = 1000):
        self.pubkey = pubkey
        self.scale = precision_scale

    def encrypt_vector(self, float_vec: np.ndarray) -> list[int]:
        """Convert float vector -> fixed-point integers -> Paillier ciphertexts."""
        int_vec = np.round(float_vec * self.scale).astype(np.int64)
        return [self.pubkey.encrypt(int(val)) for val in int_vec]

    def decrypt_vector(
        self, ciphertexts: list[int], privkey: PaillierPrivateKey
    ) -> np.ndarray:
        """Decrypt ciphertexts -> fixed-point integers -> float vector."""
        int_vals = [privkey.decrypt(c) for c in ciphertexts]
        return np.array(int_vals, dtype=np.float32) / self.scale

    def homomorphic_sum(self, cipher_vectors: list[list[int]]) -> list[int]:
        """Blindly aggregate encrypted model vectors across K clients without decryption."""
        num_params = len(cipher_vectors[0])
        accumulated = list(cipher_vectors[0])
        for vec in cipher_vectors[1:]:
            for idx in range(num_params):
                accumulated[idx] = self.pubkey.add(accumulated[idx], vec[idx])
        return accumulated
