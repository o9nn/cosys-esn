"""M-1: Input Monitoring — Validates inputs, detects noise, guards NaN/Inf."""
import numpy as np
from typing import Optional
from cosmos_core import BaseCosmosService, ServiceConfig, ServiceMessage, create_message


class InputMonitoringService(BaseCosmosService):
    """M-1: Input Monitoring. Validates input, detects noise, scales signals."""

    def __init__(self, config: ServiceConfig, input_dim: int, input_scaling: float = 1.0):
        super().__init__(config)
        self.input_dim = input_dim
        self.input_scaling = input_scaling

    async def initialize(self) -> None:
        self.log("info", "InputMonitoringService initialized")
        self.initialized = True

    async def process(self, message: ServiceMessage) -> Optional[ServiceMessage]:
        if message.type != "RAW_INPUT":
            return None
        raw = np.asarray(message.payload, dtype=float)
        if raw.ndim != 1 or raw.shape[0] != self.input_dim:
            self.log("error", f"Invalid input shape {raw.shape}, expected ({self.input_dim},)")
            return None
        if not np.all(np.isfinite(raw)):
            self.log("warning", "Input contains NaN/Inf — clipping")
            raw = np.nan_to_num(raw, nan=0.0, posinf=1.0, neginf=-1.0)
        noise_level = np.std(raw)
        scaled = raw * self.input_scaling
        msg = create_message("VALIDATED_INPUT", scaled, self.config.service_name)
        msg.metadata["noise_level"] = float(noise_level)
        return msg

    async def shutdown(self) -> None:
        self.log("info", "InputMonitoringService shutdown")
