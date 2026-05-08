"""O-4: Output Organization — Signal structuring, response formatting."""
import numpy as np
from typing import Optional
from cosmos_core import BaseCosmosService, ServiceConfig, ServiceMessage, create_message


class OutputOrganizationService(BaseCosmosService):
    """O-4: Output Organization. Structures and formats the final output."""

    def __init__(self, config: ServiceConfig, output_scaling: float = 1.0):
        super().__init__(config)
        self.output_scaling = output_scaling

    async def initialize(self) -> None:
        self.log("info", "OutputOrganizationService initialized")
        self.initialized = True

    async def process(self, message: ServiceMessage) -> Optional[ServiceMessage]:
        if message.type not in ("READOUT_OUTPUT", "DELIVERED_OUTPUT"):
            return None
        if isinstance(message.payload, dict):
            output = np.asarray(message.payload.get("prediction", message.payload.get("output", [])),
                                dtype=float)
        else:
            output = np.asarray(message.payload, dtype=float)
        output = output * self.output_scaling
        return create_message("FINAL_OUTPUT",
                              {"output": output, "timestamp": message.timestamp,
                               "source_triad": "cerebral"},
                              self.config.service_name)

    async def shutdown(self) -> None:
        self.log("info", "OutputOrganizationService shutdown")
