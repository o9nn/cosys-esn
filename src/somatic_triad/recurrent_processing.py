"""P-5: Recurrent Processing — Sparse reservoir, spectral radius, nonlinear dynamics."""
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from typing import Optional
from cosmos_core import BaseCosmosService, ServiceConfig, ServiceMessage, create_message


def _activation(x: np.ndarray, func: str) -> np.ndarray:
    if func == "tanh":
        return np.tanh(x)
    elif func == "relu":
        return np.maximum(0.0, x)
    elif func == "sigmoid":
        return 1.0 / (1.0 + np.exp(-x))
    return x


class RecurrentProcessingService(BaseCosmosService):
    """P-5: Recurrent Processing. Sparse recurrent connections and nonlinear dynamics."""

    def __init__(self, config: ServiceConfig, reservoir_dim: int, input_dim: int,
                 spectral_radius: float = 0.95, sparsity: float = 0.1,
                 leak_rate: float = 0.3, activation: str = "tanh",
                 random_seed: Optional[int] = 42):
        super().__init__(config)
        self.reservoir_dim = reservoir_dim
        self.input_dim = input_dim
        self.spectral_radius = spectral_radius
        self.sparsity = sparsity
        self.leak_rate = leak_rate
        self.activation = activation
        self.random_seed = random_seed
        self.state: np.ndarray = np.zeros(reservoir_dim)
        self.W_in: Optional[np.ndarray] = None
        self.W: Optional[sp.csr_matrix] = None

    async def initialize(self) -> None:
        rng = np.random.default_rng(self.random_seed)
        self.W_in = rng.uniform(-1, 1, (self.reservoir_dim, self.input_dim))
        self.W = self._create_sparse_reservoir(rng)
        self._scale_spectral_radius()
        self.state = np.zeros(self.reservoir_dim)
        self.log("info", f"RecurrentProcessingService initialized: dim={self.reservoir_dim}, "
                         f"sr={self.spectral_radius}, sparsity={self.sparsity}")
        self.initialized = True

    def _create_sparse_reservoir(self, rng: np.random.Generator) -> sp.csr_matrix:
        N = self.reservoir_dim
        W = sp.random(N, N, density=self.sparsity, format="csr", random_state=int(rng.integers(0, 2**31)))
        W.data = rng.uniform(-1, 1, W.data.shape)
        return W

    def _scale_spectral_radius(self) -> None:
        if self.W.nnz == 0:
            return
        try:
            k = min(6, self.W.shape[0] - 2)
            if k < 1:
                eigenvalues = np.linalg.eigvals(self.W.toarray())
                current_radius = np.max(np.abs(eigenvalues))
            else:
                eigenvalues = spla.eigs(self.W, k=k, which="LM", return_eigenvectors=False)
                current_radius = np.max(np.abs(eigenvalues))
            if current_radius > 0:
                self.W = self.W * (self.spectral_radius / current_radius)
        except Exception:
            eigenvalues = np.linalg.eigvals(self.W.toarray())
            current_radius = np.max(np.abs(eigenvalues))
            if current_radius > 0:
                self.W = self.W * (self.spectral_radius / current_radius)

    def update(self, u: np.ndarray) -> np.ndarray:
        """Update reservoir state: x(t+1) = (1-α)x(t) + α·f(W_in·u + W·x(t))"""
        pre = self.W_in @ u + self.W @ self.state
        activated = _activation(pre, self.activation)
        self.state = (1 - self.leak_rate) * self.state + self.leak_rate * activated
        return self.state

    async def process(self, message: ServiceMessage) -> Optional[ServiceMessage]:
        if message.type not in ("MEMBRANE_INPUT", "PREPROCESSED_INPUT"):
            return None
        u = np.asarray(message.payload, dtype=float)
        state = self.update(u)
        return create_message("RESERVOIR_STATE", state.copy(), self.config.service_name)

    async def shutdown(self) -> None:
        self.log("info", "RecurrentProcessingService shutdown")
