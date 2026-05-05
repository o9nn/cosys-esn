"""PD-2: Learning Director — Ridge regression, RLS, FORCE learning."""
import numpy as np
from typing import Optional, List
from cosmos_core import BaseCosmosService, ServiceConfig, ServiceMessage, create_message


class LearningDirectorService(BaseCosmosService):
    """PD-2: Learning Director. Coordinates training algorithms."""

    def __init__(self, config: ServiceConfig, method: str = "ridge",
                 ridge_lambda: float = 1e-6, rls_forgetting: float = 0.99):
        super().__init__(config)
        self.method = method  # "ridge", "rls", "force"
        self.ridge_lambda = ridge_lambda
        self.rls_forgetting = rls_forgetting
        # RLS state
        self._P: Optional[np.ndarray] = None
        self._W_out: Optional[np.ndarray] = None

    async def initialize(self) -> None:
        self.log("info", f"LearningDirectorService initialized (method={self.method})")
        self.initialized = True

    def batch_train(self, states: np.ndarray, targets: np.ndarray) -> np.ndarray:
        """Batch ridge regression."""
        X = np.hstack([states, np.ones((states.shape[0], 1))])
        Y = targets
        Id = np.eye(X.shape[1])
        W_out = np.linalg.solve(X.T @ X + self.ridge_lambda * Id, X.T @ Y).T
        self._W_out = W_out
        self.log("info", f"Batch training complete: W_out shape {W_out.shape}")
        return W_out

    def rls_init(self, feature_dim: int, output_dim: int) -> None:
        """Initialize RLS matrices."""
        self._P = (1.0 / self.ridge_lambda) * np.eye(feature_dim + 1)
        self._W_out = np.zeros((output_dim, feature_dim + 1))

    def rls_update(self, state: np.ndarray, target: np.ndarray) -> np.ndarray:
        """One-step RLS update."""
        x = np.append(state, 1.0)
        lam = self.rls_forgetting
        Px = self._P @ x
        gain = Px / (lam + x @ Px)
        error = target - self._W_out @ x
        self._W_out += np.outer(error, gain)
        self._P = (self._P - np.outer(gain, Px)) / lam
        return self._W_out

    def force_update(self, state: np.ndarray, error: np.ndarray) -> np.ndarray:
        """FORCE learning update."""
        if self._P is None or self._W_out is None:
            raise RuntimeError("Call rls_init first")
        x = np.append(state, 1.0)
        Px = self._P @ x
        gain = Px / (1.0 + x @ Px)
        self._W_out -= np.outer(error, gain)
        self._P -= np.outer(gain, Px)
        return self._W_out

    async def process(self, message: ServiceMessage) -> Optional[ServiceMessage]:
        return None

    async def shutdown(self) -> None:
        self.log("info", "LearningDirectorService shutdown")
