"""M-1: Target Interface — Loss computation, feedback reception, error routing."""
import numpy as np
from typing import Optional, Callable
from cosmos_core import BaseCosmosService, ServiceConfig, ServiceMessage, create_message


class TargetInterfaceService(BaseCosmosService):
    """M-1: Target Interface. Computes loss and routes error signals."""

    def __init__(self, config: ServiceConfig, loss: str = "mse"):
        super().__init__(config)
        self.loss_name = loss
        self._loss_history: list = []

    def compute_loss(self, prediction: np.ndarray, target: np.ndarray) -> float:
        if self.loss_name == "mse":
            return float(np.mean((prediction - target) ** 2))
        elif self.loss_name == "mae":
            return float(np.mean(np.abs(prediction - target)))
        return float(np.mean((prediction - target) ** 2))

    async def initialize(self) -> None:
        self.log("info", "TargetInterfaceService initialized")
        self.initialized = True

    async def process(self, message: ServiceMessage) -> Optional[ServiceMessage]:
        if message.type == "PREDICTION_WITH_TARGET":
            pred = np.asarray(message.payload["prediction"], dtype=float)
            target = np.asarray(message.payload["target"], dtype=float)
            loss = self.compute_loss(pred, target)
            self._loss_history.append(loss)
            result = create_message("LOSS_SIGNAL", {"loss": loss, "error": pred - target},
                                    self.config.service_name)
            return result
        return None

    async def shutdown(self) -> None:
        self.log("info", "TargetInterfaceService shutdown")
