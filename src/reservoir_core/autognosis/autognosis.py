"""
ReservoirAutognosis: Self-awareness system for Echo State Networks.

Four monitoring layers:
1. Self-Monitoring    — real-time state metrics
2. Self-Modeling      — capacity and quality estimation
3. Meta-Cognitive     — performance prediction and anomaly detection
4. Self-Optimization  — adaptive parameter tuning
"""
import numpy as np
import scipy.sparse.linalg as spla
from typing import Dict, List, Optional, TYPE_CHECKING
from models.echo_state_network import EchoStateNetwork

if TYPE_CHECKING:
    pass


class ReservoirAutognosis:
    """
    Self-awareness system for echo state networks.

    Monitors reservoir health in real-time and triggers adaptive
    parameter adjustments when degradation is detected.
    """

    def __init__(self, esn: EchoStateNetwork, window_size: int = 100):
        self.esn = esn
        self.window_size = window_size
        self.metrics_history: List[Dict] = []
        self._baseline_norms: List[float] = []
        self._confidence_history: List[float] = []

    # =========================================================================
    # Layer 1: Self-Monitoring
    # =========================================================================

    def state_norm_tracking(self) -> float:
        """Track current reservoir state norm."""
        return float(np.linalg.norm(self.esn.x))

    def spectral_analysis(self) -> float:
        """Compute current effective spectral radius."""
        W = self.esn.W
        if W.nnz == 0:
            return 0.0
        try:
            k = min(6, W.shape[0] - 2)
            if k < 1:
                raise ValueError
            eigs = spla.eigs(W, k=k, which="LM", return_eigenvectors=False)
            return float(np.max(np.abs(eigs)))
        except Exception:
            eigs = np.linalg.eigvals(W.toarray())
            return float(np.max(np.abs(eigs)))

    def echo_index(self) -> float:
        """
        Compute echo index as proxy for memory capacity.
        Higher = longer memory (fading memory measurement).
        Estimated as 1 / leak_rate when no history is available.
        """
        alpha = self.esn.config.leak_rate
        return float(1.0 / (alpha + 1e-8))

    def activation_statistics(self) -> Dict[str, float]:
        """Compute statistics of current reservoir activation."""
        x = self.esn.x
        return {
            "mean": float(np.mean(x)),
            "std": float(np.std(x)),
            "sparsity": float(np.mean(np.abs(x) < 0.1)),
            "max_abs": float(np.max(np.abs(x))),
        }

    def monitor(self) -> Dict:
        """Full self-monitoring pass — call after each update step."""
        metrics = {
            "state_norm": self.state_norm_tracking(),
            "spectral_radius": self.spectral_analysis(),
            "echo_index": self.echo_index(),
            **self.activation_statistics(),
        }
        self.metrics_history.append(metrics)
        if len(self.metrics_history) > self.window_size:
            self.metrics_history.pop(0)
        return metrics

    # =========================================================================
    # Layer 2: Self-Modeling
    # =========================================================================

    def capacity_estimation(self, test_inputs: Optional[np.ndarray] = None) -> float:
        """
        Estimate total information processing capacity.
        Approximated as: MC_estimate + nonlinear_estimate
        """
        mc = self._compute_memory_capacity_estimate()
        nc = self._compute_nonlinear_capacity_estimate()
        return mc + nc

    def _compute_memory_capacity_estimate(self) -> float:
        """Proxy: echo_index * n_reservoir * (1 - sparsity)."""
        cfg = self.esn.config
        return self.echo_index() * cfg.n_reservoir * cfg.sparsity

    def _compute_nonlinear_capacity_estimate(self) -> float:
        """Proxy: reservoir_dim * (1 - mean activation)."""
        cfg = self.esn.config
        x = self.esn.x
        return float(cfg.n_reservoir * np.mean(np.abs(np.tanh(x) - x)))

    def kernel_quality(self) -> Dict[str, float]:
        """
        Estimate kernel quality: separation vs approximation balance.
        Returns separation_score and approximation_score (both 0..1).
        """
        if len(self.metrics_history) < 2:
            return {"separation": 0.5, "approximation": 0.5}
        norms = [m["state_norm"] for m in self.metrics_history]
        variation = float(np.std(norms) / (np.mean(norms) + 1e-8))
        separation = min(1.0, variation)
        approximation = 1.0 - separation
        return {"separation": separation, "approximation": approximation}

    def memory_depth(self) -> int:
        """Estimate effective echo horizon in timesteps."""
        alpha = self.esn.config.leak_rate
        sr = self.esn.config.spectral_radius
        # Effective horizon: -1/log(α * sr)
        val = alpha * sr
        if val <= 0 or val >= 1:
            return int(1.0 / (alpha + 1e-8))
        return int(-1.0 / np.log(val))

    def nonlinearity_profile(self) -> Dict[str, float]:
        """Profile the nonlinearity of current reservoir."""
        x = self.esn.x
        lin = x
        nonlin = np.tanh(x)
        distortion = float(np.mean(np.abs(nonlin - lin)))
        return {
            "distortion": distortion,
            "saturation_fraction": float(np.mean(np.abs(x) > 2.0)),
        }

    # =========================================================================
    # Layer 3: Meta-Cognitive
    # =========================================================================

    def performance_prediction(self) -> float:
        """
        Predict expected performance (proxy: stability of state norms).
        Returns 0.0 (poor) to 1.0 (excellent).
        """
        if len(self.metrics_history) < 5:
            return 0.5
        recent_norms = [m["state_norm"] for m in self.metrics_history[-10:]]
        stability = 1.0 / (1.0 + np.std(recent_norms))
        return float(np.clip(stability, 0.0, 1.0))

    def confidence_scoring(self, output: Optional[np.ndarray] = None) -> float:
        """Score prediction confidence based on output stability."""
        self._confidence_history.append(self.performance_prediction())
        if len(self._confidence_history) < 2:
            return 0.5
        return float(np.mean(self._confidence_history[-5:]))

    def anomaly_detection(self, threshold_sigma: float = 3.0) -> bool:
        """Detect anomalous reservoir states (out-of-distribution)."""
        if len(self.metrics_history) < 10:
            return False
        norms = [m["state_norm"] for m in self.metrics_history]
        mu = np.mean(norms[:-1])
        sigma = np.std(norms[:-1]) + 1e-8
        current = norms[-1]
        return bool(abs(current - mu) > threshold_sigma * sigma)

    def adaptation_trigger_check(self) -> bool:
        """Determine if reservoir needs adaptation."""
        if len(self.metrics_history) < 10:
            return False
        recent = self.metrics_history[-10:]
        norm_trend = np.polyfit(range(10), [m["state_norm"] for m in recent], 1)[0]
        return abs(norm_trend) > 0.1

    def should_adapt(self) -> bool:
        return self.adaptation_trigger_check()

    # =========================================================================
    # Layer 4: Self-Optimization
    # =========================================================================

    def tune_spectral_radius(self, target: Optional[float] = None) -> float:
        """Adaptively tune spectral radius toward target."""
        current_sr = self.spectral_analysis()
        if target is None:
            # Auto-target based on performance
            if self.performance_prediction() < 0.5:
                target = min(0.99, current_sr * 1.02)
            else:
                target = max(0.5, current_sr * 0.99)
        if current_sr > 0:
            self.esn.W = self.esn.W * (target / current_sr)
        self.esn.config.spectral_radius = target
        return target

    def adapt_leak_rate(self, delta: float = 0.01) -> float:
        """Adapt leak rate based on memory depth requirements."""
        if len(self.metrics_history) < 10:
            return self.esn.config.leak_rate
        norm_trend = float(np.polyfit(range(len(self.metrics_history[-10:])),
                                      [m["state_norm"] for m in self.metrics_history[-10:]], 1)[0])
        if norm_trend > 0.1:
            self.esn.config.leak_rate = max(0.01, self.esn.config.leak_rate - delta)
        elif norm_trend < -0.1:
            self.esn.config.leak_rate = min(1.0, self.esn.config.leak_rate + delta)
        return self.esn.config.leak_rate

    def optimize_input_scaling(self) -> float:
        """Adjust input scaling based on activation statistics."""
        stats = self.activation_statistics()
        if (stats.get("saturation_fraction", 0)) > 0.3:
            self.esn.config.input_scaling = max(0.01, self.esn.config.input_scaling * 0.95)
            self.esn.W_in *= 0.95
        elif stats["mean"] < 0.01:
            self.esn.config.input_scaling = min(10.0, self.esn.config.input_scaling * 1.05)
            self.esn.W_in *= 1.05
        return self.esn.config.input_scaling

    def refine_topology(self) -> str:
        """Suggest topology refinement based on kernel quality."""
        kq = self.kernel_quality()
        if kq["separation"] < 0.2:
            return "increase_sparsity"
        elif kq["approximation"] < 0.2:
            return "decrease_sparsity"
        return "no_change"
