"""Tests for the 12-step EchoBeats cognitive loop."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import pytest
from models.echo_state_network import EchoStateNetwork, ESNConfig
from reservoir_core.echo_beats.echo_beats import EchoBeatsLoop, CognitiveStream, Phase, PHASE_SEQUENCE


def make_esn():
    config = ESNConfig(n_inputs=3, n_reservoir=30, n_outputs=1,
                       washout=5, random_seed=42)
    esn = EchoStateNetwork(config)
    # Train with dummy data so W_out is not zeros
    X = np.random.randn(30, 3)
    y = np.random.randn(30, 1)
    esn.train(X, y)
    return esn


def test_phase_sequence_length():
    assert len(PHASE_SEQUENCE) == 12


def test_stream_phase_offsets():
    esn = make_esn()
    loop = EchoBeatsLoop(esn)
    # Stream 0: step 0 should be PERCEIVE (offset=0, PHASE_SEQUENCE[0])
    assert loop.streams[0].get_phase(0) == Phase.PERCEIVE
    # Stream 1: step 0 should be PHASE_SEQUENCE[4] (offset=4)
    assert loop.streams[1].get_phase(0) == PHASE_SEQUENCE[4]
    # Stream 2: step 0 should be PHASE_SEQUENCE[8] (offset=8)
    assert loop.streams[2].get_phase(0) == PHASE_SEQUENCE[8]


def test_cognitive_stream_perceive():
    esn = make_esn()
    stream = CognitiveStream(esn, phase_offset=0, stream_id=0)
    u = np.array([0.1, 0.2, 0.3])
    state = stream.perceive(u)
    assert state.shape == (30,)
    assert len(stream._perception_buffer) == 1


def test_echo_beats_step():
    esn = make_esn()
    loop = EchoBeatsLoop(esn)
    u = np.array([0.1, 0.2, 0.3])
    result = loop.step(0, u)
    assert "stream_0" in result
    assert "stream_1" in result
    assert "stream_2" in result
    assert "integrated_state" in result


def test_echo_beats_run_cycle():
    esn = make_esn()
    loop = EchoBeatsLoop(esn)
    input_seq = np.random.randn(12, 3)
    outputs = loop.run_cycle(input_seq)
    assert len(outputs) == 12
    for step_out in outputs:
        assert "stream_0" in step_out
        assert "integrated_state" in step_out
