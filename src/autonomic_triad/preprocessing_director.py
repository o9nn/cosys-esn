"""PD-2: Preprocessing Director — Normalization, PCA dimensionality reduction."""
import numpy as np
from typing import Optional
from cosmos_core import BaseCosmosService, ServiceConfig, ServiceMessage, create_message


class PreprocessingDirectorService(BaseCosmosService):
    """PD-2: Preprocessing Director. Normalization and dimensionality reduction."""

    def __init__(self, config: ServiceConfig, method: str = "zscore", n_components: Optional[int] = None):
        super().__init__(config)
        self.method = method  # "zscore", "minmax", "none"
        self.n_components = n_components
        self._mean: Optional[np.ndarray] = None
        self._std: Optional[np.ndarray] = None
        self._min: Optional[np.ndarray] = None
        self._max: Optional[np.ndarray] = None
        self._pca_components: Optional[np.ndarray] = None  # (n_components, input_dim)

    def fit(self, data: np.ndarray) -> None:
        """Fit normalization and optional PCA on training data. data: (n_samples, input_dim)"""
        if self.method == "zscore":
            self._mean = np.mean(data, axis=0)
            self._std = np.std(data, axis=0) + 1e-8
        elif self.method == "minmax":
            self._min = np.min(data, axis=0)
            self._max = np.max(data, axis=0) + 1e-8
        if self.n_components is not None and self.n_components < data.shape[1]:
            centered = data - np.mean(data, axis=0)
            _, _, Vt = np.linalg.svd(centered, full_matrices=False)
            self._pca_components = Vt[:self.n_components]

    def transform(self, x: np.ndarray) -> np.ndarray:
        if self.method == "zscore" and self._mean is not None:
            x = (x - self._mean) / self._std
        elif self.method == "minmax" and self._min is not None:
            x = (x - self._min) / (self._max - self._min)
        if self._pca_components is not None:
            x = self._pca_components @ x
        return x

    async def initialize(self) -> None:
        self.log("info", "PreprocessingDirectorService initialized")
        self.initialized = True

    async def process(self, message: ServiceMessage) -> Optional[ServiceMessage]:
        if message.type not in ("VALIDATED_INPUT", "RAW_INPUT"):
            return None
        x = np.asarray(message.payload, dtype=float)
        x = self.transform(x)
        return create_message("PREPROCESSED_INPUT", x, self.config.service_name)

    async def shutdown(self) -> None:
        self.log("info", "PreprocessingDirectorService shutdown")
