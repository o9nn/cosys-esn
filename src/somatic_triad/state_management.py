"""S-8: State Management — Reservoir state vector, echo persistence, history."""
import numpy as np
from typing import Optional, List
from cosmos_core import BaseCosmosService, ServiceConfig, ServiceMessage, create_message


class StateManagementService(BaseCosmosService):
    """S-8: State Management. Maintains reservoir state and echo persistence."""

    def __init__(self, config: ServiceConfig, reservoir_dim: int, max_history: int = 1000):
        super().__init__(config)
        self.reservoir_dim = reservoir_dim
        self.max_history = max_history
        self.state: np.ndarray = np.zeros(reservoir_dim)
        self.state_history: List[np.ndarray] = []

    async def initialize(self) -> None:
        self.state = np.zeros(self.reservoir_dim)
        self.state_history = []
        self.log("info", "StateManagementService initialized")
        self.initialized = True

    def update_state(self, new_state: np.ndarray) -> None:
        self.state = new_state
        self.state_history.append(new_state.copy())
        if len(self.state_history) > self.max_history:
            self.state_history.pop(0)

    def reset(self) -> None:
        self.state = np.zeros(self.reservoir_dim)

    async def process(self, message: ServiceMessage) -> Optional[ServiceMessage]:
        if message.type == "RESERVOIR_STATE":
            self.update_state(np.asarray(message.payload, dtype=float))
            return message
        return None

    async def shutdown(self) -> None:
        self.log("info", "StateManagementService shutdown")
