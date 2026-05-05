"""PD-2: Dynamics Development — Spectral tuning, topology optimization."""
import numpy as np
import scipy.sparse as sp
from typing import Optional, TYPE_CHECKING
from cosmos_core import BaseCosmosService, ServiceConfig, ServiceMessage, create_message

if TYPE_CHECKING:
    from somatic_triad.recurrent_processing import RecurrentProcessingService


def _small_world_reservoir(N: int, k: int, p: float, rng: np.random.Generator) -> sp.csr_matrix:
    """Watts-Strogatz small-world graph as a sparse matrix."""
    rows, cols, data = [], [], []
    # Ring lattice
    for i in range(N):
        for j in range(1, k // 2 + 1):
            rows.append(i)
            cols.append((i + j) % N)
            rows.append(i)
            cols.append((i - j) % N)
    # Rewire
    new_rows, new_cols, new_data = [], [], []
    for r, c in zip(rows, cols):
        if rng.random() < p:
            new_c = rng.integers(0, N)
            new_rows.append(r)
            new_cols.append(int(new_c))
        else:
            new_rows.append(r)
            new_cols.append(c)
        new_data.append(rng.uniform(-1, 1))
    return sp.csr_matrix((new_data, (new_rows, new_cols)), shape=(N, N))


def _scale_free_reservoir(N: int, m: int, rng: np.random.Generator) -> sp.csr_matrix:
    """Barabasi-Albert scale-free graph."""
    rows, cols, data = [], [], []
    degrees = np.ones(N)
    for i in range(m, N):
        probs = degrees[:i] / degrees[:i].sum()
        targets = rng.choice(i, size=min(m, i), replace=False, p=probs)
        for t in targets:
            rows.append(i)
            cols.append(int(t))
            data.append(rng.uniform(-1, 1))
            degrees[i] += 1
            degrees[t] += 1
    if not rows:
        return sp.csr_matrix((N, N))
    return sp.csr_matrix((data, (rows, cols)), shape=(N, N))


class DynamicsDevelopmentService(BaseCosmosService):
    """PD-2: Dynamics Development. Spectral tuning and topology optimization."""

    def __init__(self, config: ServiceConfig):
        super().__init__(config)
        self._reservoir_service = None

    def attach_reservoir(self, reservoir_service) -> None:
        self._reservoir_service = reservoir_service

    async def initialize(self) -> None:
        self.log("info", "DynamicsDevelopmentService initialized")
        self.initialized = True

    def build_topology(self, N: int, topology: str = "random", sparsity: float = 0.1,
                       seed: Optional[int] = 42) -> sp.csr_matrix:
        rng = np.random.default_rng(seed)
        if topology == "small_world":
            k = max(2, int(N * sparsity))
            return _small_world_reservoir(N, k=k, p=0.1, rng=rng)
        elif topology == "scale_free":
            m = max(1, int(N * sparsity / 2))
            return _scale_free_reservoir(N, m=m, rng=rng)
        else:  # random
            W = sp.random(N, N, density=sparsity, format="csr", random_state=int(rng.integers(0, 2**31)))
            W.data = rng.uniform(-1, 1, W.data.shape)
            return W

    async def process(self, message: ServiceMessage) -> Optional[ServiceMessage]:
        return None

    async def shutdown(self) -> None:
        self.log("info", "DynamicsDevelopmentService shutdown")
