"""S-8: Output Delivery — Prediction generation, classification, confidence scoring."""
import numpy as np
from typing import Optional
from cosmos_core import BaseCosmosService, ServiceConfig, ServiceMessage, create_message


class OutputDeliveryService(BaseCosmosService):
    """S-8: Output Delivery. Generates predictions and confidence scores."""

    def __init__(self, config: ServiceConfig, task: str = "regression"):
        super().__init__(config)
        self.task = task  # "regression", "classification"
        self._prediction_history: list = []

    def confidence_score(self, output: np.ndarray) -> float:
        """Proxy confidence: inverse of output magnitude variance."""
        if len(self._prediction_history) < 2:
            return 1.0
        recent = np.array(self._prediction_history[-10:])
        return float(1.0 / (1.0 + np.std(recent)))

    async def initialize(self) -> None:
        self.log("info", "OutputDeliveryService initialized")
        self.initialized = True

    async def process(self, message: ServiceMessage) -> Optional[ServiceMessage]:
        if message.type not in ("READOUT_OUTPUT", "FINAL_OUTPUT"):
            return None
        if isinstance(message.payload, dict):
            output = np.asarray(message.payload.get("output", []), dtype=float)
        else:
            output = np.asarray(message.payload, dtype=float)
        self._prediction_history.append(output.tolist())
        if self.task == "classification":
            prediction = int(np.argmax(output))
            probs = np.exp(output) / np.sum(np.exp(output))
            confidence = float(np.max(probs))
        else:
            prediction = output
            confidence = self.confidence_score(output)
        result = create_message("DELIVERED_OUTPUT",
                                {"prediction": prediction, "confidence": confidence},
                                self.config.service_name)
        return result

    async def shutdown(self) -> None:
        self.log("info", "OutputDeliveryService shutdown")
