"""
EchoBeatsLoop: 12-step cognitive loop with 3 concurrent streams.

Three streams are phased 120° apart (4 steps in 12-step cycle):
  Stream 1: phase_offset=0
  Stream 2: phase_offset=4
  Stream 3: phase_offset=8
"""
import numpy as np
from enum import Enum
from typing import List, Dict, Any, Optional
from models.echo_state_network import EchoStateNetwork


class Phase(Enum):
    PERCEIVE = "perceive"
    ATTEND = "attend"
    FRAME = "frame"
    REASON = "reason"
    INTEND = "intend"
    EXECUTE = "execute"
    EVALUATE = "evaluate"
    INTEGRATE = "integrate"


# 12-step phase sequence (each position maps to a Phase)
PHASE_SEQUENCE = [
    Phase.PERCEIVE,
    Phase.ATTEND,
    Phase.FRAME,
    Phase.REASON,
    Phase.PERCEIVE,
    Phase.INTEND,
    Phase.PERCEIVE,
    Phase.EXECUTE,
    Phase.PERCEIVE,
    Phase.EVALUATE,
    Phase.PERCEIVE,
    Phase.INTEGRATE,
]


class CognitiveStream:
    """
    A single cognitive stream with 120° phase offset.

    Each stream maintains its own echo state, perception buffer, and
    internal representation.
    """

    def __init__(self, esn: EchoStateNetwork, phase_offset: int = 0,
                 stream_id: int = 0):
        self.esn = esn
        self.phase_offset = phase_offset
        self.stream_id = stream_id
        self._state: np.ndarray = np.zeros(esn.config.n_reservoir)
        self._perception_buffer: List[np.ndarray] = []
        self._attention_weights: Optional[np.ndarray] = None
        self._frame: Optional[np.ndarray] = None
        self._intention: Optional[np.ndarray] = None
        self._execution_output: Optional[np.ndarray] = None
        self._evaluation: Optional[float] = None

    def get_phase(self, step_idx: int) -> Phase:
        return PHASE_SEQUENCE[(step_idx + self.phase_offset) % 12]

    def perceive(self, input_data: np.ndarray) -> np.ndarray:
        """PERCEIVE: Update reservoir state with input."""
        self._state = self.esn.update(input_data)
        self._perception_buffer.append(self._state.copy())
        if len(self._perception_buffer) > 12:
            self._perception_buffer.pop(0)
        return self._state

    def attend(self) -> np.ndarray:
        """ATTEND: Compute attention weights over perception buffer."""
        if not self._perception_buffer:
            self._attention_weights = np.ones(1)
            return self._state
        buffer = np.array(self._perception_buffer)
        norms = np.linalg.norm(buffer, axis=1) + 1e-8
        self._attention_weights = norms / norms.sum()
        attended = (buffer * self._attention_weights[:, np.newaxis]).sum(axis=0)
        return attended

    def frame(self) -> np.ndarray:
        """FRAME: Create a compressed representation."""
        if not self._perception_buffer:
            self._frame = self._state.copy()
            return self._frame
        buffer = np.array(self._perception_buffer)
        self._frame = np.mean(buffer, axis=0)
        return self._frame

    def reason(self) -> np.ndarray:
        """REASON: Apply reasoning via nonlinear transformation of frame."""
        if self._frame is None:
            self._frame = self._state.copy()
        reasoning = np.tanh(self._frame * 2.0)
        return reasoning

    def intend(self) -> np.ndarray:
        """INTEND: Form intention as combination of reasoning and state."""
        reasoning = self.reason()
        self._intention = 0.7 * reasoning + 0.3 * self._state
        return self._intention

    def execute(self) -> np.ndarray:
        """EXECUTE: Execute intention through readout layer."""
        if self._intention is None:
            self._intention = self._state.copy()
        x_bias = np.append(self._intention, 1.0)
        self._execution_output = self.esn.W_out @ x_bias
        return self._execution_output

    def evaluate(self) -> float:
        """EVALUATE: Evaluate execution quality (proxy: output magnitude)."""
        if self._execution_output is None:
            self._evaluation = 0.0
        else:
            self._evaluation = float(np.linalg.norm(self._execution_output))
        return self._evaluation

    def integrate(self) -> np.ndarray:
        """INTEGRATE: Integrate stream outputs into updated state."""
        if self._execution_output is not None:
            # Blend execution output back into reservoir state
            scale = min(1.0, 1.0 / (np.linalg.norm(self._execution_output) + 1e-8))
            feedback = np.zeros(self.esn.config.n_reservoir)
            # Project output back into reservoir space via W_out transpose
            if self._execution_output.shape[0] == self.esn.config.n_outputs:
                feedback = self.esn.W_out[:, :-1].T @ self._execution_output
                feedback = np.tanh(feedback * scale * 0.1)
            self._state = 0.9 * self._state + 0.1 * feedback
        return self._state


class EchoBeatsLoop:
    """
    12-step cognitive loop with 3 concurrent streams.

    Phases are interleaved across streams at 120° (4-step) offsets:
      Step:    1  2  3  4  5  6  7  8  9  10 11 12
      Stream1: P  A  F  R  P  I  P  E  P  Ev P  In
      Stream2: R  P  I  P  E  P  Ev P  In P  A  F
      Stream3: E  P  Ev P  In P  A  F  R  P  I  P
    """

    CYCLE_LENGTH = 12

    def __init__(self, esn: EchoStateNetwork):
        self.esn = esn
        self.streams = [
            CognitiveStream(esn, phase_offset=0, stream_id=0),
            CognitiveStream(esn, phase_offset=4, stream_id=1),
            CognitiveStream(esn, phase_offset=8, stream_id=2),
        ]

    def step(self, step_idx: int, input_data: np.ndarray) -> Dict[str, Any]:
        """
        Execute one step of the 12-step loop for all 3 streams.

        Parameters
        ----------
        step_idx  : current step (0-11)
        input_data: (n_inputs,) input for this step

        Returns
        -------
        Dict with 'stream_0', 'stream_1', 'stream_2' results and 'integrated_state'
        """
        results = {}
        stream_outputs = []

        for i, stream in enumerate(self.streams):
            phase = stream.get_phase(step_idx)
            key = f"stream_{i}"

            if phase == Phase.PERCEIVE:
                out = stream.perceive(input_data)
            elif phase == Phase.ATTEND:
                out = stream.attend()
            elif phase == Phase.FRAME:
                out = stream.frame()
            elif phase == Phase.REASON:
                out = stream.reason()
            elif phase == Phase.INTEND:
                out = stream.intend()
            elif phase == Phase.EXECUTE:
                out = stream.execute()
            elif phase == Phase.EVALUATE:
                out = stream.evaluate()
            elif phase == Phase.INTEGRATE:
                out = stream.integrate()
            else:
                out = stream._state

            results[key] = {"phase": phase.value, "output": out}
            if isinstance(out, np.ndarray):
                stream_outputs.append(out)

        # Integration: combine all stream state outputs that share the same shape
        if stream_outputs:
            reference_shape = stream_outputs[0].shape
            same_shape_outputs = [o for o in stream_outputs if o.shape == reference_shape]
            if same_shape_outputs:
                combined = np.mean(same_shape_outputs, axis=0)
            else:
                combined = stream_outputs[0]
            results["integrated_state"] = combined
        else:
            results["integrated_state"] = np.zeros(1)

        return results

    def run_cycle(self, input_sequence: np.ndarray) -> List[Dict[str, Any]]:
        """
        Run complete 12-step cycle.

        Parameters
        ----------
        input_sequence : (n_steps, n_inputs) — will be tiled to 12 steps

        Returns
        -------
        List of 12 step result dicts
        """
        n = len(input_sequence)
        outputs = []
        for step_idx in range(self.CYCLE_LENGTH):
            u = input_sequence[step_idx % n]
            outputs.append(self.step(step_idx, u))
        return outputs
