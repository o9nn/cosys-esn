"""M-1: Membrane Interface — Input scaling with state-dependent adaptation."""
import numpy as np
from typing import Optional
from cosmos_core import BaseCosmosService, ServiceConfig, ServiceMessage, create_message


class MembraneInterfaceService(BaseCosmosService):
    """M-1: Membrane Interface. Input scaling and boundary conditions."""

    def __init__(self, config: ServiceConfig, input_scaling: float = 1.0,
                 adaptive: bool = False):
        super().__init__(config)
        self.input_scaling = input_scaling
        self.adaptive = adaptive
        self._state_norm_history: list = []

    async def initialize(self) -> None:
        self.log("info", "MembraneInterfaceService initialized")
        self.initialized = True

    def update_state_feedback(self, state_norm: float) -> None:
        """Receive state-norm feedback for adaptive scaling."""
        self._state_norm_history.append(state_norm)
        if self.adaptive and len(self._state_norm_history) >= 10:
            recent = np.mean(self._state_norm_history[-10:])
            if recent > 2.0:
                self.input_scaling *= 0.95
            elif recent < 0.5:
                self.input_scaling *= 1.05
            self.input_scaling = np.clip(self.input_scaling, 0.01, 10.0)

    async def process(self, message: ServiceMessage) -> Optional[ServiceMessage]:
        if message.type not in ("PREPROCESSED_INPUT", "ENCODED_INPUT", "VALIDATED_INPUT",
                                "ORGANIZED_INPUT", "RAW_INPUT"):
            return None
        x = np.asarray(message.payload, dtype=float)
        if x.ndim > 1:
            x = x.flatten()
        scaled = x * self.input_scaling
        return create_message("MEMBRANE_INPUT", scaled, self.config.service_name)

    async def shutdown(self) -> None:
        self.log("info", "MembraneInterfaceService shutdown")
