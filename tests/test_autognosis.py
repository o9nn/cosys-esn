"""Tests for ReservoirAutognosis self-awareness system."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import pytest
from models.echo_state_network import EchoStateNetwork, ESNConfig
from reservoir_core.autognosis.autognosis import ReservoirAutognosis


def make_esn_with_data():
    config = ESNConfig(n_inputs=2, n_reservoir=50, n_outputs=1,
                       spectral_radius=0.9, leak_rate=0.3, random_seed=42)
    esn = EchoStateNetwork(config)
    # Run some inputs to populate state
    inputs = np.random.randn(50, 2)
    for u in inputs:
        esn.update(u)
    return esn


def test_autognosis_state_norm():
    esn = make_esn_with_data()
    ag = ReservoirAutognosis(esn)
    norm = ag.state_norm_tracking()
    assert norm >= 0.0


def test_autognosis_spectral_analysis():
    esn = make_esn_with_data()
    ag = ReservoirAutognosis(esn)
    sr = ag.spectral_analysis()
    assert 0.0 < sr < 2.0


def test_autognosis_echo_index():
    esn = make_esn_with_data()
    ag = ReservoirAutognosis(esn)
    idx = ag.echo_index()
    assert idx > 0.0


def test_autognosis_activation_statistics():
    esn = make_esn_with_data()
    ag = ReservoirAutognosis(esn)
    stats = ag.activation_statistics()
    assert "mean" in stats
    assert "std" in stats
    assert "sparsity" in stats


def test_autognosis_monitor():
    esn = make_esn_with_data()
    ag = ReservoirAutognosis(esn)
    metrics = ag.monitor()
    assert "state_norm" in metrics
    assert "spectral_radius" in metrics
    assert len(ag.metrics_history) == 1


def test_autognosis_capacity_estimation():
    esn = make_esn_with_data()
    ag = ReservoirAutognosis(esn)
    cap = ag.capacity_estimation()
    assert cap >= 0.0


def test_autognosis_memory_depth():
    esn = make_esn_with_data()
    ag = ReservoirAutognosis(esn)
    depth = ag.memory_depth()
    assert depth >= 0


def test_autognosis_performance_prediction():
    esn = make_esn_with_data()
    ag = ReservoirAutognosis(esn)
    # Feed some monitoring history
    for _ in range(10):
        esn.update(np.random.randn(2))
        ag.monitor()
    perf = ag.performance_prediction()
    assert 0.0 <= perf <= 1.0


def test_autognosis_anomaly_detection():
    esn = make_esn_with_data()
    ag = ReservoirAutognosis(esn)
    # Build baseline
    for _ in range(15):
        esn.update(np.random.randn(2) * 0.1)
        ag.monitor()
    # Normal state — probably no anomaly
    result = ag.anomaly_detection(threshold_sigma=3.0)
    assert isinstance(result, (bool, np.bool_))


def test_autognosis_should_adapt():
    esn = make_esn_with_data()
    ag = ReservoirAutognosis(esn)
    for _ in range(15):
        ag.monitor()
    result = ag.should_adapt()
    assert isinstance(result, (bool, np.bool_))


def test_autognosis_tune_spectral_radius():
    esn = make_esn_with_data()
    ag = ReservoirAutognosis(esn)
    new_sr = ag.tune_spectral_radius(target=0.85)
    assert abs(new_sr - 0.85) < 0.01
