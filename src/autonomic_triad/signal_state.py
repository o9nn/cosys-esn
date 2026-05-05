"""S-8: Signal State — Circular buffer, temporal windowing."""
import numpy as np
from collections import deque
from typing import Optional
from cosmos_core import BaseCosmosService, ServiceConfig, ServiceMessage, create_message


class SignalStateService(BaseCosmosService):
    """S-8: Signal State. Manages input buffers and temporal windowing."""

    def __init__(self, config: ServiceConfig, window_size: int = 10):
        super().__init__(config)
        self.window_size = window_size
        self.buffer: deque = deque(maxlen=window_size)

    async def initialize(self) -> None:
        self.log("info", "SignalStateService initialized")
        self.initialized = True

    async def process(self, message: ServiceMessage) -> Optional[ServiceMessage]:
        if message.type not in ("VALIDATED_INPUT", "PREPROCESSED_INPUT"):
            return None
        self.buffer.append(message.payload)
        window = np.array(list(self.buffer))
        return create_message("BUFFERED_INPUT", window, self.config.service_name)

    def get_window(self) -> np.ndarray:
        return np.array(list(self.buffer))

    async def shutdown(self) -> None:
        self.log("info", "SignalStateService shutdown")
