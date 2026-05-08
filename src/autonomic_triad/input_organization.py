"""O-4: Input Organization — Temporal batching, sliding windows, padding."""
import numpy as np
from typing import List, Optional
from cosmos_core import BaseCosmosService, ServiceConfig, ServiceMessage, create_message


class InputOrganizationService(BaseCosmosService):
    """O-4: Input Organization. Temporal batching and sliding window management."""

    def __init__(self, config: ServiceConfig, window_size: int = 1, stride: int = 1,
                 pad_mode: str = "zero"):
        super().__init__(config)
        self.window_size = window_size
        self.stride = stride
        self.pad_mode = pad_mode
        self._history: List[np.ndarray] = []

    async def initialize(self) -> None:
        self.log("info", "InputOrganizationService initialized")
        self.initialized = True

    def create_windows(self, sequence: np.ndarray) -> np.ndarray:
        """Create sliding windows from a 1D or 2D sequence."""
        n = len(sequence)
        if n < self.window_size:
            if self.pad_mode == "zero":
                pad = np.zeros((self.window_size - n, *sequence.shape[1:]))
            else:
                pad = np.tile(sequence[:1], (self.window_size - n, *([1] * (sequence.ndim - 1))))
            sequence = np.concatenate([pad, sequence], axis=0)
            n = self.window_size
        windows = []
        for i in range(0, n - self.window_size + 1, self.stride):
            windows.append(sequence[i:i + self.window_size])
        return np.array(windows)

    async def process(self, message: ServiceMessage) -> Optional[ServiceMessage]:
        if message.type not in ("PREPROCESSED_INPUT", "VALIDATED_INPUT", "TRANSFORMED_INPUT"):
            return None
        x = np.asarray(message.payload, dtype=float)
        self._history.append(x)
        if len(self._history) > self.window_size:
            self._history = self._history[-self.window_size:]
        batch = np.array(self._history)
        return create_message("ORGANIZED_INPUT", batch, self.config.service_name)

    async def shutdown(self) -> None:
        self.log("info", "InputOrganizationService shutdown")
