"""T-7: State Treasury — Echo memory store, temporal pattern library."""
import numpy as np
from typing import Dict, List, Optional
from cosmos_core import BaseCosmosService, ServiceConfig, ServiceMessage, create_message


class StateTreasuryService(BaseCosmosService):
    """T-7: State Treasury. Echo memory store and temporal pattern retrieval."""

    def __init__(self, config: ServiceConfig, capacity: int = 500, compress: bool = False,
                 compress_dim: int = 50):
        super().__init__(config)
        self.capacity = capacity
        self.compress = compress
        self.compress_dim = compress_dim
        self._store: List[np.ndarray] = []
        self._labels: List[str] = []
        self._proj: Optional[np.ndarray] = None

    async def initialize(self) -> None:
        self.log("info", "StateTreasuryService initialized")
        self.initialized = True

    def store(self, state: np.ndarray, label: str = "") -> None:
        if self.compress and self._proj is None and state.shape[0] > self.compress_dim:
            rng = np.random.default_rng(2)
            self._proj = rng.standard_normal((self.compress_dim, state.shape[0])) / np.sqrt(state.shape[0])
        stored = (self._proj @ state) if self._proj is not None else state.copy()
        self._store.append(stored)
        self._labels.append(label)
        if len(self._store) > self.capacity:
            self._store.pop(0)
            self._labels.pop(0)

    def retrieve(self, query: np.ndarray, top_k: int = 1) -> List[np.ndarray]:
        if not self._store:
            return []
        q = (self._proj @ query) if self._proj is not None else query
        sims = [float(np.dot(q, s) / (np.linalg.norm(q) * np.linalg.norm(s) + 1e-8))
                for s in self._store]
        idx = np.argsort(sims)[-top_k:][::-1]
        return [self._store[i] for i in idx]

    async def process(self, message: ServiceMessage) -> Optional[ServiceMessage]:
        if message.type == "RESERVOIR_STATE":
            state = np.asarray(message.payload, dtype=float)
            self.store(state)
            return message
        return None

    async def shutdown(self) -> None:
        self.log("info", "StateTreasuryService shutdown")
