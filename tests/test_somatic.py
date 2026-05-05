"""Unit tests for all 6 Somatic Triad services."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import asyncio
import numpy as np
import pytest
from cosmos_core import ServiceConfig, Triad, Polarity, ServicePosition, Dimension, create_message
from somatic_triad.membrane_interface import MembraneInterfaceService
from somatic_triad.state_management import StateManagementService
from somatic_triad.recurrent_processing import RecurrentProcessingService
from somatic_triad.dynamics_organization import DynamicsOrganizationService
from somatic_triad.dynamics_development import DynamicsDevelopmentService
from somatic_triad.state_treasury import StateTreasuryService


def make_cfg(name, pos):
    return ServiceConfig(name, Triad.SOMATIC, pos, Polarity.SOMATIC, Dimension.COMMITMENT)


# ---------- M-1: MembraneInterfaceService ----------

def test_membrane_interface_scaling():
    svc = MembraneInterfaceService(make_cfg("m1", ServicePosition.M1), input_scaling=0.5)
    asyncio.run(svc.initialize())
    msg = create_message("PREPROCESSED_INPUT", np.array([2.0, 4.0]), "ext")
    result = asyncio.run(svc.process(msg))
    np.testing.assert_array_almost_equal(result.payload, [1.0, 2.0])


def test_membrane_interface_adaptive():
    svc = MembraneInterfaceService(make_cfg("m1", ServicePosition.M1),
                                   input_scaling=1.0, adaptive=True)
    asyncio.run(svc.initialize())
    # Feed many high-norm states → should reduce scaling
    for _ in range(15):
        svc.update_state_feedback(10.0)
    assert svc.input_scaling < 1.0


# ---------- S-8: StateManagementService ----------

def test_state_management():
    svc = StateManagementService(make_cfg("s8", ServicePosition.S8), reservoir_dim=5)
    asyncio.run(svc.initialize())
    assert svc.state.shape == (5,)
    new_state = np.ones(5)
    svc.update_state(new_state)
    np.testing.assert_array_equal(svc.state, np.ones(5))
    assert len(svc.state_history) == 1


# ---------- P-5: RecurrentProcessingService ----------

def test_recurrent_processing_update():
    svc = RecurrentProcessingService(
        make_cfg("p5", ServicePosition.P5),
        reservoir_dim=20, input_dim=3,
        spectral_radius=0.9, sparsity=0.2,
        leak_rate=0.3, random_seed=0,
    )
    asyncio.run(svc.initialize())
    msg = create_message("MEMBRANE_INPUT", np.array([0.1, 0.2, 0.3]), "ext")
    result = asyncio.run(svc.process(msg))
    assert result is not None
    assert result.type == "RESERVOIR_STATE"
    assert result.payload.shape == (20,)


def test_recurrent_processing_echo_state():
    """Verify echo state property: two different starts converge to same state."""
    svc = RecurrentProcessingService(
        make_cfg("p5", ServicePosition.P5),
        reservoir_dim=50, input_dim=2,
        spectral_radius=0.9, sparsity=0.1,
        random_seed=42,
    )
    asyncio.run(svc.initialize())
    u_seq = [np.array([np.sin(t * 0.1), np.cos(t * 0.1)]) for t in range(100)]

    svc.state = np.zeros(50)
    for u in u_seq:
        svc.update(u)
    state1 = svc.state.copy()

    svc.state = np.random.randn(50) * 0.01
    for u in u_seq:
        svc.update(u)
    state2 = svc.state.copy()

    np.testing.assert_allclose(state1, state2, atol=0.1)


# ---------- O-4: DynamicsOrganizationService ----------

def test_dynamics_organization():
    svc = DynamicsOrganizationService(make_cfg("o4", ServicePosition.O4))
    asyncio.run(svc.initialize())
    state = np.random.randn(50)
    metrics = svc.monitor_state(state)
    assert "state_norm" in metrics
    assert metrics["state_norm"] >= 0


# ---------- PD-2: DynamicsDevelopmentService ----------

def test_dynamics_development_topologies():
    svc = DynamicsDevelopmentService(make_cfg("pd2", ServicePosition.PD2))
    asyncio.run(svc.initialize())
    for topo in ["random", "small_world", "scale_free"]:
        W = svc.build_topology(N=30, topology=topo, sparsity=0.1, seed=0)
        assert W.shape == (30, 30)


# ---------- T-7: StateTreasuryService ----------

def test_state_treasury_store_retrieve():
    svc = StateTreasuryService(make_cfg("t7", ServicePosition.T7), capacity=5)
    asyncio.run(svc.initialize())
    states = [np.random.randn(10) for _ in range(5)]
    for s in states:
        svc.store(s)
    assert len(svc._store) == 5

    query = states[0].copy()
    retrieved = svc.retrieve(query, top_k=1)
    assert len(retrieved) == 1
    # Retrieved state should be similar to query
    sim = np.dot(query, retrieved[0]) / (np.linalg.norm(query) * np.linalg.norm(retrieved[0]) + 1e-8)
    assert sim > 0.9
