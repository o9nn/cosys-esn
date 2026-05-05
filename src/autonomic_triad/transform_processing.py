"""P-5: Transform Processing — Fourier, wavelet, embedding generation."""
import numpy as np
from typing import Optional
from cosmos_core import BaseCosmosService, ServiceConfig, ServiceMessage, create_message


class TransformProcessingService(BaseCosmosService):
    """P-5: Transform Processing. Fourier/wavelet transforms and embeddings."""

    def __init__(self, config: ServiceConfig, transform: str = "none", embed_dim: Optional[int] = None):
        super().__init__(config)
        self.transform = transform  # "none", "fft", "dct"
        self.embed_dim = embed_dim
        self._embed_matrix: Optional[np.ndarray] = None

    def _init_embedding(self, input_dim: int) -> None:
        if self.embed_dim and self._embed_matrix is None:
            rng = np.random.default_rng(0)
            self._embed_matrix = rng.standard_normal((self.embed_dim, input_dim)) / np.sqrt(input_dim)

    async def initialize(self) -> None:
        self.log("info", "TransformProcessingService initialized")
        self.initialized = True

    async def process(self, message: ServiceMessage) -> Optional[ServiceMessage]:
        if message.type not in ("PREPROCESSED_INPUT", "VALIDATED_INPUT", "RAW_INPUT"):
            return None
        x = np.asarray(message.payload, dtype=float)
        if self.transform == "fft":
            x = np.abs(np.fft.rfft(x))
        elif self.transform == "dct":
            # Type-2 DCT via FFT
            N = len(x)
            v = np.concatenate([x[::2], x[1::2][::-1]])
            X = np.fft.fft(v)
            k = np.arange(N)
            w = 2 * np.exp(-1j * np.pi * k / (2 * N))
            x = np.real(w * X)[:N // 2 + 1]
        if self.embed_dim:
            self._init_embedding(x.shape[0])
            x = np.tanh(self._embed_matrix @ x)
        return create_message("TRANSFORMED_INPUT", x, self.config.service_name)

    async def shutdown(self) -> None:
        self.log("info", "TransformProcessingService shutdown")
