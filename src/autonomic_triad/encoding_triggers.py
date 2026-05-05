"""T-7: Encoding Triggers — Sparse coding (k-WTA), feature extraction, thresholds."""
import numpy as np
from typing import Optional
from cosmos_core import BaseCosmosService, ServiceConfig, ServiceMessage, create_message


class EncodingTriggersService(BaseCosmosService):
    """T-7: Encoding Triggers. Sparse coding and feature extraction."""

    def __init__(self, config: ServiceConfig, k: int = 5, threshold: Optional[float] = None,
                 output_dim: Optional[int] = None):
        super().__init__(config)
        self.k = k                          # k-WTA: keep top-k activations
        self.threshold = threshold          # Hard threshold
        self.output_dim = output_dim        # Random projection output dim
        self._proj: Optional[np.ndarray] = None

    def _init_projection(self, input_dim: int) -> None:
        if self.output_dim and self._proj is None:
            rng = np.random.default_rng(1)
            self._proj = rng.standard_normal((self.output_dim, input_dim)) / np.sqrt(input_dim)

    def kwta(self, x: np.ndarray) -> np.ndarray:
        """Keep top-k activations, zero out the rest."""
        out = np.zeros_like(x)
        idx = np.argsort(np.abs(x))[-self.k:]
        out[idx] = x[idx]
        return out

    async def initialize(self) -> None:
        self.log("info", "EncodingTriggersService initialized")
        self.initialized = True

    async def process(self, message: ServiceMessage) -> Optional[ServiceMessage]:
        if message.type not in ("PREPROCESSED_INPUT", "VALIDATED_INPUT", "ORGANIZED_INPUT",
                                "TRANSFORMED_INPUT"):
            return None
        x = np.asarray(message.payload, dtype=float)
        if x.ndim > 1:
            x = x.flatten()
        if self.output_dim:
            self._init_projection(x.shape[0])
            x = np.tanh(self._proj @ x)
        if self.threshold is not None:
            x = np.where(np.abs(x) >= self.threshold, x, 0.0)
        else:
            x = self.kwta(x)
        return create_message("ENCODED_INPUT", x, self.config.service_name)

    async def shutdown(self) -> None:
        self.log("info", "EncodingTriggersService shutdown")
