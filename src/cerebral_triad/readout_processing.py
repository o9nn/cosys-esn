"""P-5: Readout Processing — Linear combination of reservoir states."""
import numpy as np
from typing import Optional
from cosmos_core import BaseCosmosService, ServiceConfig, ServiceMessage, create_message


class ReadoutProcessingService(BaseCosmosService):
    """P-5: Readout Processing. Linear combination of reservoir states for output."""

    def __init__(self, config: ServiceConfig, reservoir_dim: int, output_dim: int):
        super().__init__(config)
        self.reservoir_dim = reservoir_dim
        self.output_dim = output_dim
        self.W_out: Optional[np.ndarray] = None

    async def initialize(self) -> None:
        self.W_out = np.zeros((self.output_dim, self.reservoir_dim + 1))
        self.log("info", "ReadoutProcessingService initialized")
        self.initialized = True

    def train(self, states: np.ndarray, targets: np.ndarray, ridge_lambda: float = 1e-6) -> None:
        """Ridge regression: W_out = (X^T X + λI)^-1 X^T Y"""
        X = np.hstack([states, np.ones((states.shape[0], 1))])
        Y = targets
        Id = np.eye(X.shape[1])
        self.W_out = np.linalg.solve(X.T @ X + ridge_lambda * Id, X.T @ Y).T
        self.log("info", f"Output weights trained: {self.W_out.shape}")

    async def process(self, message: ServiceMessage) -> Optional[ServiceMessage]:
        if message.type != "RESERVOIR_STATE":
            return None
        if self.W_out is None or not np.any(self.W_out):
            self.log("warning", "ReadoutProcessingService: W_out not trained, output will be zeros")
        state = np.asarray(message.payload, dtype=float)
        state_bias = np.append(state, 1.0)
        output = self.W_out @ state_bias
        return create_message("READOUT_OUTPUT", output, self.config.service_name)

    async def shutdown(self) -> None:
        self.log("info", "ReadoutProcessingService shutdown")
