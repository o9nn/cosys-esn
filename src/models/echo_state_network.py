"""
EchoStateNetwork: Standalone Echo State Network wrapping all three triads.
Implements synchronous update/train/predict using scipy.sparse for the W matrix.
"""
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from dataclasses import dataclass, field
from typing import Optional, Literal
from pathlib import Path


@dataclass
class ESNConfig:
    n_inputs: int = 10
    n_reservoir: int = 500
    n_outputs: int = 1
    spectral_radius: float = 0.9
    input_scaling: float = 0.5
    leak_rate: float = 0.3
    sparsity: float = 0.1           # connection density (NOT sparsity)
    activation: Literal["tanh", "relu", "sigmoid"] = "tanh"
    topology: Literal["random", "small_world", "scale_free"] = "random"
    ridge_lambda: float = 1e-6
    random_seed: Optional[int] = 42
    washout: int = 100              # initial timesteps to discard during training


def _activation_fn(x: np.ndarray, func: str) -> np.ndarray:
    if func == "tanh":
        return np.tanh(x)
    elif func == "relu":
        return np.maximum(0.0, x)
    elif func == "sigmoid":
        return 1.0 / (1.0 + np.exp(-x))
    return x


def _build_random_reservoir(N: int, density: float, rng: np.random.Generator) -> sp.csr_matrix:
    W = sp.random(N, N, density=density, format="csr",
                  random_state=int(rng.integers(0, 2**31)))
    W.data = rng.uniform(-1, 1, W.data.shape)
    return W


def _build_small_world_reservoir(N: int, density: float, rewire_prob: float,
                                  rng: np.random.Generator) -> sp.csr_matrix:
    k = max(2, int(N * density))
    rows, cols, data = [], [], []
    for i in range(N):
        for j in range(1, k // 2 + 1):
            nb = (i + j) % N
            if rng.random() < rewire_prob:
                nb = int(rng.integers(0, N))
            rows.append(i); cols.append(nb); data.append(rng.uniform(-1, 1))
            nb2 = (i - j) % N
            if rng.random() < rewire_prob:
                nb2 = int(rng.integers(0, N))
            rows.append(i); cols.append(nb2); data.append(rng.uniform(-1, 1))
    return sp.csr_matrix((data, (rows, cols)), shape=(N, N))


def _build_scale_free_reservoir(N: int, density: float, rng: np.random.Generator) -> sp.csr_matrix:
    m = max(1, int(N * density / 2))
    rows, cols, data = [], [], []
    degrees = np.ones(N)
    for i in range(m, N):
        p = degrees[:i] / degrees[:i].sum()
        targets = rng.choice(i, size=min(m, i), replace=False, p=p)
        for t in targets:
            rows.append(i); cols.append(int(t)); data.append(rng.uniform(-1, 1))
            degrees[i] += 1; degrees[int(t)] += 1
    if not rows:
        return sp.csr_matrix((N, N))
    return sp.csr_matrix((data, (rows, cols)), shape=(N, N))


def _scale_spectral_radius(W: sp.csr_matrix, target_sr: float) -> sp.csr_matrix:
    if W.nnz == 0:
        return W
    k = min(6, W.shape[0] - 2)
    try:
        if k < 1:
            raise ValueError
        eigs = spla.eigs(W, k=k, which="LM", return_eigenvectors=False)
        current_sr = float(np.max(np.abs(eigs)))
    except Exception:
        eigs = np.linalg.eigvals(W.toarray())
        current_sr = float(np.max(np.abs(eigs)))
    if current_sr > 0:
        W = W * (target_sr / current_sr)
    return W


class EchoStateNetwork:
    """
    Cosmos System 5 Echo State Network.

    Wraps Autonomic (input), Somatic (reservoir), and Cerebral (readout) triads
    as synchronous operations for efficient time-series processing.

    Attributes
    ----------
    config : ESNConfig
    W_in : np.ndarray  — (n_reservoir, n_inputs) input weight matrix
    W    : sp.csr_matrix — (n_reservoir, n_reservoir) sparse recurrent weights
    W_out: np.ndarray  — (n_outputs, n_reservoir+1) output weights (trained)
    x    : np.ndarray  — (n_reservoir,) current reservoir state
    """

    def __init__(self, config: Optional[ESNConfig] = None, **kwargs):
        if config is None:
            config = ESNConfig(**kwargs)
        self.config = config
        self._rng = np.random.default_rng(config.random_seed)
        self.W_in: Optional[np.ndarray] = None
        self.W: Optional[sp.csr_matrix] = None
        self.W_out: Optional[np.ndarray] = None
        self.x: np.ndarray = np.zeros(config.n_reservoir)
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        cfg = self.config
        # Input weights (Autonomic triad: M-1 Membrane Interface)
        self.W_in = self._rng.uniform(-1, 1, (cfg.n_reservoir, cfg.n_inputs)) * cfg.input_scaling

        # Recurrent weights (Somatic triad: P-5 Recurrent Processing)
        if cfg.topology == "small_world":
            W = _build_small_world_reservoir(cfg.n_reservoir, cfg.sparsity, 0.1, self._rng)
        elif cfg.topology == "scale_free":
            W = _build_scale_free_reservoir(cfg.n_reservoir, cfg.sparsity, self._rng)
        else:
            W = _build_random_reservoir(cfg.n_reservoir, cfg.sparsity, self._rng)
        self.W = _scale_spectral_radius(W, cfg.spectral_radius)

        # Output weights (Cerebral triad: P-5 Readout) — initialized to zeros
        self.W_out = np.zeros((cfg.n_outputs, cfg.n_reservoir + 1))

    def reset_state(self) -> None:
        """Reset reservoir state to zero."""
        self.x = np.zeros(self.config.n_reservoir)

    def update(self, u: np.ndarray) -> np.ndarray:
        """
        Update reservoir state (Somatic triad processing).

        x(t+1) = (1-α)x(t) + α·f(W_in·u + W·x(t))
        """
        alpha = self.config.leak_rate
        pre = self.W_in @ u + self.W @ self.x
        activated = _activation_fn(pre, self.config.activation)
        self.x = (1 - alpha) * self.x + alpha * activated
        return self.x

    def train(self, inputs: np.ndarray, targets: np.ndarray,
              ridge_param: Optional[float] = None) -> np.ndarray:
        """
        Train output weights using ridge regression (Cerebral triad: PD-2 Learning Director).

        Parameters
        ----------
        inputs  : (n_samples, n_inputs)
        targets : (n_samples, n_outputs)
        ridge_param : regularisation λ (defaults to config.ridge_lambda)

        Returns
        -------
        W_out : (n_outputs, n_reservoir+1)
        """
        if ridge_param is None:
            ridge_param = self.config.ridge_lambda
        washout = self.config.washout
        self.reset_state()

        states = []
        for u in inputs:
            self.update(u)
            states.append(self.x.copy())

        states = np.array(states)
        # Discard washout period
        states = states[washout:]
        targets_w = targets[washout:]

        # Add bias
        X = np.hstack([states, np.ones((states.shape[0], 1))])
        Y = targets_w

        # Ridge regression (numerically stable via lstsq or solve)
        reg = ridge_param * np.eye(X.shape[1])
        self.W_out = np.linalg.solve(X.T @ X + reg, X.T @ Y).T  # (n_outputs, n_reservoir+1)
        return self.W_out

    def predict(self, u: np.ndarray) -> np.ndarray:
        """
        Update reservoir and generate prediction (Cerebral triad: P-5 Readout).

        Parameters
        ----------
        u : (n_inputs,) input vector

        Returns
        -------
        output : (n_outputs,)
        """
        self.update(u)
        x_bias = np.append(self.x, 1.0)
        return self.W_out @ x_bias

    def run(self, inputs: np.ndarray) -> np.ndarray:
        """
        Run prediction over a sequence without updating state between calls.

        Parameters
        ----------
        inputs : (n_samples, n_inputs)

        Returns
        -------
        outputs : (n_samples, n_outputs)
        """
        return np.array([self.predict(u) for u in inputs])

    def save(self, path: str) -> None:
        """Save model to .npz file."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        np.savez(str(p),
                 W_in=self.W_in,
                 W_data=self.W.data,
                 W_indices=self.W.indices,
                 W_indptr=self.W.indptr,
                 W_shape=np.array(self.W.shape),
                 W_out=self.W_out)

    @classmethod
    def load(cls, path: str, config: ESNConfig) -> "EchoStateNetwork":
        """Load model from .npz file."""
        esn = cls(config)
        data = np.load(path)
        esn.W_in = data["W_in"]
        shape = tuple(data["W_shape"])
        esn.W = sp.csr_matrix((data["W_data"], data["W_indices"], data["W_indptr"]), shape=shape)
        esn.W_out = data["W_out"]
        return esn
