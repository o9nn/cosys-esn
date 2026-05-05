"""
ReservoirSystem5: 60-step deterministic state machine.

Implements the Cosmos System 5 model with:
- 3 Universal Modes (3-step cycle): Exploration, Exploitation, Adaptation
- 4 Reservoir Compartments (5-step staggered): Input, Core, Memory, Output
- 60-step cycle (LCM of 3 and 20)
"""
import numpy as np
from enum import Enum
from dataclasses import dataclass, field
from typing import List, Optional, Dict
from models.echo_state_network import EchoStateNetwork, ESNConfig


class UniversalMode(Enum):
    EXPLORATION = "exploration"    # High spectral radius ~0.99
    EXPLOITATION = "exploitation"  # Stable dynamics ~0.85
    ADAPTATION = "adaptation"      # Learning phase ~0.90


# Mode-specific spectral radii
MODE_SPECTRAL_RADIUS: Dict[UniversalMode, float] = {
    UniversalMode.EXPLORATION: 0.99,
    UniversalMode.EXPLOITATION: 0.85,
    UniversalMode.ADAPTATION: 0.90,
}

# Mode cycle (3-step)
_MODE_CYCLE = [UniversalMode.EXPLORATION, UniversalMode.EXPLOITATION, UniversalMode.ADAPTATION]


@dataclass
class ReservoirCompartment:
    """A sub-reservoir compartment within the Somatic Triad."""
    name: str
    state: np.ndarray = field(default_factory=lambda: np.array([]))
    history: List[np.ndarray] = field(default_factory=list)

    def initialize(self, dim: int) -> None:
        self.state = np.zeros(dim)
        self.history = []

    def update(self, new_state: np.ndarray) -> None:
        self.state = new_state
        self.history.append(new_state.copy())
        if len(self.history) > 200:
            self.history.pop(0)


class ReservoirSystem5:
    """
    Implements the 60-step reservoir cycle with triadic dynamics.

    The cycle length is LCM(3, 20) = 60:
    - 3-step Universal Mode cycle (Exploration, Exploitation, Adaptation)
    - 20-step Particular Compartment cycle (4 compartments × 5 steps each)
    """

    CYCLE_LENGTH = 60

    def __init__(self, config: Optional[ESNConfig] = None, n_reservoir: int = 500):
        if config is None:
            config = ESNConfig(n_reservoir=n_reservoir)
        self.config = config
        self.esn = EchoStateNetwork(config)

        # Universal Sets: Global Reservoir Modes
        self.current_mode = UniversalMode.EXPLORATION

        # Particular Sets: Reservoir Compartments
        self.compartments: List[ReservoirCompartment] = [
            ReservoirCompartment("InputCompartment"),
            ReservoirCompartment("CoreReservoir"),
            ReservoirCompartment("MemoryCompartment"),
            ReservoirCompartment("OutputCompartment"),
        ]
        for c in self.compartments:
            c.initialize(config.n_reservoir)

        # Compartment state indices (0-indexed)
        self._compartment_states = np.zeros((4, config.n_reservoir))
        self._step_count = 0

    def get_mode(self, t: int) -> UniversalMode:
        """Get current Universal Mode based on 3-step cycle."""
        return _MODE_CYCLE[t % 3]

    def _apply_mode_spectral_radius(self, mode: UniversalMode) -> None:
        """Temporarily rescale spectral radius based on mode."""
        target_sr = MODE_SPECTRAL_RADIUS[mode]
        # Only scale if significantly different from current
        current_sr = self.config.spectral_radius
        if abs(target_sr - current_sr) > 0.01:
            self.esn.W = self.esn.W * (target_sr / current_sr)

    def _compartment_convolution(self, t: int) -> np.ndarray:
        """
        Concurrency convolution for compartment state transitions.

        S_i(t+1) = (S_i(t) + Σ_{j≠i} S_j(t) + mode_phase(t)) mod 4
        """
        mode_phase = t % 3
        new_states = np.zeros_like(self._compartment_states)
        for i in range(4):
            s_sum = np.sum(self._compartment_states, axis=0) + mode_phase
            new_states[i] = np.tanh(self._compartment_states[i] + s_sum * 0.1)
        return new_states

    def echo_step(self, u: np.ndarray, t: int) -> np.ndarray:
        """
        Execute one step of the 60-step echo cycle.

        Parameters
        ----------
        u : input vector
        t : current time step

        Returns
        -------
        output state of the active compartment
        """
        self._step_count = t % self.CYCLE_LENGTH

        # Universal mode (3-step cycle)
        mode = self.get_mode(t)
        self.current_mode = mode

        # Apply mode-dependent spectral radius
        self._apply_mode_spectral_radius(mode)

        # Particular compartment (5-step staggered within 20-step block)
        p_idx = (t // 3) % 4
        compartment = self.compartments[p_idx]

        # Update ESN state
        state = self.esn.update(u)

        # Update compartment
        compartment.update(state)
        self._compartment_states[p_idx] = state

        # Apply compartment convolution (rest step every 5th in compartment cycle)
        if (t % 5) == 4:
            self._compartment_states = self._compartment_convolution(t)

        return compartment.state

    def run_cycle(self, input_sequence: np.ndarray) -> List[np.ndarray]:
        """
        Run a full 60-step cycle.

        Parameters
        ----------
        input_sequence : (n_steps, n_inputs) or will be tiled to 60 steps

        Returns
        -------
        List of 60 compartment state outputs
        """
        n = len(input_sequence)
        outputs = []
        for t in range(self.CYCLE_LENGTH):
            u = input_sequence[t % n]
            outputs.append(self.echo_step(u, t))
        return outputs

    def get_state_summary(self) -> dict:
        return {
            "step": self._step_count,
            "mode": self.current_mode.value,
            "compartment_norms": [float(np.linalg.norm(c.state)) for c in self.compartments],
        }
