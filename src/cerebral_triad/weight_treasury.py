"""T-7: Weight Treasury — Output weight storage, pattern memory, model save/load."""
import numpy as np
from pathlib import Path
from typing import Optional, Dict
from cosmos_core import BaseCosmosService, ServiceConfig, ServiceMessage, create_message


class WeightTreasuryService(BaseCosmosService):
    """T-7: Weight Treasury. Output weight storage and model persistence."""

    def __init__(self, config: ServiceConfig):
        super().__init__(config)
        self._weights: Dict[str, np.ndarray] = {}
        self._metadata: Dict[str, dict] = {}

    async def initialize(self) -> None:
        self.log("info", "WeightTreasuryService initialized")
        self.initialized = True

    def store(self, name: str, weights: np.ndarray, metadata: Optional[dict] = None) -> None:
        self._weights[name] = weights.copy()
        self._metadata[name] = metadata or {}
        self.log("info", f"Stored weights '{name}' shape={weights.shape}")

    def retrieve(self, name: str) -> Optional[np.ndarray]:
        return self._weights.get(name)

    def save(self, path: str) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        np.savez(str(p), **self._weights)
        self.log("info", f"Weights saved to {path}")

    def load(self, path: str) -> None:
        data = np.load(path)
        for k in data.files:
            self._weights[k] = data[k]
        self.log("info", f"Weights loaded from {path}")

    async def process(self, message: ServiceMessage) -> Optional[ServiceMessage]:
        return None

    async def shutdown(self) -> None:
        self.log("info", "WeightTreasuryService shutdown")
