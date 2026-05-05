"""
StreamManager: Manages 3 concurrent EchoBeats streams.
"""
from typing import List, Optional
import numpy as np
from models.echo_state_network import EchoStateNetwork
from reservoir_core.echo_beats.echo_beats import EchoBeatsLoop, CognitiveStream


class StreamManager:
    """Coordinates 3 concurrent echo streams."""

    def __init__(self, esn: EchoStateNetwork):
        self.loop = EchoBeatsLoop(esn)
        self._step_count = 0

    def step(self, input_data: np.ndarray) -> dict:
        result = self.loop.step(self._step_count % 12, input_data)
        self._step_count += 1
        return result

    def run_cycle(self, input_sequence: np.ndarray) -> list:
        return self.loop.run_cycle(input_sequence)
