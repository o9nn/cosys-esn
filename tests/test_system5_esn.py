"""
Tests for ReservoirSystem5 (system5_esn.py):
- UniversalMode cycling
- ReservoirCompartment state updates
- echo_step output shape
- run_cycle length and type
- get_state_summary keys
"""
import numpy as np
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from models.system5_esn import (
    ReservoirSystem5,
    UniversalMode,
    ReservoirCompartment,
    MODE_SPECTRAL_RADIUS,
    _MODE_CYCLE,
)
from models.echo_state_network import ESNConfig


# ---------------------------------------------------------------------------
# UniversalMode
# ---------------------------------------------------------------------------

def test_universal_mode_values():
    assert UniversalMode.EXPLORATION.value == "exploration"
    assert UniversalMode.EXPLOITATION.value == "exploitation"
    assert UniversalMode.ADAPTATION.value == "adaptation"


def test_mode_spectral_radii_range():
    for mode, sr in MODE_SPECTRAL_RADIUS.items():
        assert 0.5 <= sr <= 1.1, f"{mode} SR out of range: {sr}"


def test_mode_cycle_length():
    assert len(_MODE_CYCLE) == 3


# ---------------------------------------------------------------------------
# ReservoirCompartment
# ---------------------------------------------------------------------------

def test_compartment_initialize():
    c = ReservoirCompartment("test")
    c.initialize(dim=10)
    assert c.state.shape == (10,)
    assert np.allclose(c.state, 0)


def test_compartment_update_and_history():
    c = ReservoirCompartment("test")
    c.initialize(dim=5)
    new_state = np.ones(5)
    c.update(new_state)
    assert np.allclose(c.state, new_state)
    assert len(c.history) == 1


def test_compartment_history_capped_at_200():
    c = ReservoirCompartment("test")
    c.initialize(dim=4)
    for i in range(250):
        c.update(np.full(4, float(i)))
    assert len(c.history) <= 200


# ---------------------------------------------------------------------------
# ReservoirSystem5
# ---------------------------------------------------------------------------

@pytest.fixture
def sys5():
    config = ESNConfig(n_inputs=3, n_reservoir=60, n_outputs=1, random_seed=0)
    return ReservoirSystem5(config=config)


def test_system5_default_init():
    rs = ReservoirSystem5(n_reservoir=50)
    assert len(rs.compartments) == 4
    assert rs.CYCLE_LENGTH == 60


def test_system5_get_mode_cycling(sys5):
    """Mode should cycle through 3 values."""
    modes = [sys5.get_mode(t) for t in range(6)]
    assert modes == [
        UniversalMode.EXPLORATION,
        UniversalMode.EXPLOITATION,
        UniversalMode.ADAPTATION,
        UniversalMode.EXPLORATION,
        UniversalMode.EXPLOITATION,
        UniversalMode.ADAPTATION,
    ]


def test_system5_echo_step_output_shape(sys5):
    u = np.random.randn(3)
    out = sys5.echo_step(u, t=0)
    assert out.shape == (60,)


def test_system5_echo_step_updates_mode(sys5):
    u = np.random.randn(3)
    sys5.echo_step(u, t=0)
    assert sys5.current_mode == UniversalMode.EXPLORATION
    sys5.echo_step(u, t=1)
    assert sys5.current_mode == UniversalMode.EXPLOITATION
    sys5.echo_step(u, t=2)
    assert sys5.current_mode == UniversalMode.ADAPTATION


def test_system5_run_cycle_length(sys5):
    seq = np.random.randn(12, 3)
    outputs = sys5.run_cycle(seq)
    assert len(outputs) == 60


def test_system5_run_cycle_output_shapes(sys5):
    seq = np.random.randn(5, 3)
    outputs = sys5.run_cycle(seq)
    for out in outputs:
        assert out.shape == (60,)


def test_system5_get_state_summary_keys(sys5):
    u = np.random.randn(3)
    sys5.echo_step(u, t=5)
    summary = sys5.get_state_summary()
    assert "step" in summary
    assert "mode" in summary
    assert "compartment_norms" in summary
    assert len(summary["compartment_norms"]) == 4


def test_system5_compartment_norms_finite(sys5):
    seq = np.random.randn(20, 3)
    sys5.run_cycle(seq)
    summary = sys5.get_state_summary()
    for norm in summary["compartment_norms"]:
        assert np.isfinite(norm)
