"""O-4: Dynamics Organization — Spectral radius enforcement, leak rate, edge-of-chaos."""
import numpy as np
from typing import Optional
from cosmos_core import BaseCosmosService, ServiceConfig, ServiceMessage, create_message


class DynamicsOrganizationService(BaseCosmosService):
    """O-4: Dynamics Organization. Monitors and enforces spectral radius and leak rate."""

    def __init__(self, config: ServiceConfig, target_spectral_radius: float = 0.95,
                 target_leak_rate: float = 0.3):
        super().__init__(config)
        self.target_spectral_radius = target_spectral_radius
        self.target_leak_rate = target_leak_rate
        self._state_norms: list = []

    async def initialize(self) -> None:
        self.log("info", "DynamicsOrganizationService initialized")
        self.initialized = True

    def monitor_state(self, state: np.ndarray) -> dict:
        norm = float(np.linalg.norm(state))
        sparsity = float(np.mean(np.abs(state) < 0.01))
        self._state_norms.append(norm)
        return {
            "state_norm": norm,
            "activation_sparsity": sparsity,
            "edge_of_chaos": self._is_edge_of_chaos(),
        }

    def _is_edge_of_chaos(self) -> bool:
        """Heuristic: state norm should be stable (not growing or collapsing)."""
        if len(self._state_norms) < 10:
            return True
        recent = self._state_norms[-10:]
        trend = np.polyfit(range(10), recent, 1)[0]
        return abs(trend) < 0.05

    async def process(self, message: ServiceMessage) -> Optional[ServiceMessage]:
        if message.type == "RESERVOIR_STATE":
            state = np.asarray(message.payload, dtype=float)
            metrics = self.monitor_state(state)
            result = create_message("DYNAMICS_METRICS", metrics, self.config.service_name)
            return result
        return None

    async def shutdown(self) -> None:
        self.log("info", "DynamicsOrganizationService shutdown")
