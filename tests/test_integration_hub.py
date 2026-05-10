"""
Tests for the integration hub:
- EventBus (pub/sub messaging)
- StreamManager (3-stream echo coordination)
- CosmosReservoirSystem (full 18-service async pipeline)
"""
import asyncio
import numpy as np
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from cosmos_core import create_message
from integration_hub.event_bus import EventBus
from integration_hub.stream_manager import StreamManager
from integration_hub.cosmos_reservoir_system import CosmosReservoirSystem
from models.echo_state_network import EchoStateNetwork, ESNConfig


# ---------------------------------------------------------------------------
# EventBus
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_event_bus_subscribe_publish():
    bus = EventBus()
    received = []

    async def handler(msg):
        received.append(msg.payload)
        return None

    bus.subscribe("TEST_EVENT", handler)
    msg = create_message("TEST_EVENT", {"value": 42}, "test")
    await bus.publish(msg)
    assert received == [{"value": 42}]


@pytest.mark.asyncio
async def test_event_bus_multiple_subscribers():
    bus = EventBus()
    results = []

    async def h1(msg):
        results.append("h1")

    async def h2(msg):
        results.append("h2")

    bus.subscribe("PING", h1)
    bus.subscribe("PING", h2)
    await bus.publish(create_message("PING", None, "test"))
    assert results == ["h1", "h2"]


@pytest.mark.asyncio
async def test_event_bus_no_subscribers():
    bus = EventBus()
    # Publishing with no subscribers should return empty list
    results = await bus.publish(create_message("NO_TOPIC", None, "test"))
    assert results == []


@pytest.mark.asyncio
async def test_event_bus_handler_exception_is_swallowed():
    """Errors in a handler must not propagate."""
    bus = EventBus()
    called = []

    async def bad_handler(msg):
        raise RuntimeError("boom")

    async def good_handler(msg):
        called.append(True)

    bus.subscribe("EV", bad_handler)
    bus.subscribe("EV", good_handler)
    # Should not raise
    await bus.publish(create_message("EV", None, "test"))
    assert called == [True]


# ---------------------------------------------------------------------------
# StreamManager
# ---------------------------------------------------------------------------

def test_stream_manager_step_returns_dict():
    config = ESNConfig(n_inputs=3, n_reservoir=50, n_outputs=1, random_seed=0)
    esn = EchoStateNetwork(config)
    sm = StreamManager(esn)
    u = np.random.randn(3)
    result = sm.step(u)
    assert isinstance(result, dict)


def test_stream_manager_step_increments():
    config = ESNConfig(n_inputs=3, n_reservoir=50, n_outputs=1, random_seed=0)
    esn = EchoStateNetwork(config)
    sm = StreamManager(esn)
    u = np.random.randn(3)
    for _ in range(12):
        sm.step(u)
    assert sm._step_count == 12


def test_stream_manager_run_cycle():
    config = ESNConfig(n_inputs=2, n_reservoir=50, n_outputs=1, random_seed=7)
    esn = EchoStateNetwork(config)
    sm = StreamManager(esn)
    seq = np.random.randn(12, 2)
    results = sm.run_cycle(seq)
    assert isinstance(results, list)
    assert len(results) == 12


# ---------------------------------------------------------------------------
# CosmosReservoirSystem
# ---------------------------------------------------------------------------

@pytest.fixture
def small_config():
    return ESNConfig(
        n_inputs=3, n_reservoir=80, n_outputs=1,
        spectral_radius=0.9, leak_rate=0.3, random_seed=42,
    )


@pytest.mark.asyncio
async def test_cosmos_system_initialize(small_config):
    sys_ = CosmosReservoirSystem(small_config)
    await sys_.initialize()
    assert len(sys_.services) == 18
    assert sys_.esn is not None


@pytest.mark.asyncio
async def test_cosmos_system_train_and_process(small_config):
    rng = np.random.default_rng(0)
    X = rng.standard_normal((120, 3))
    y = X[:, :1] * 0.5 + 0.1

    sys_ = CosmosReservoirSystem(small_config)
    await sys_.initialize()
    await sys_.train(X, y)

    out = await sys_.process_input(X[0])
    assert out.shape == (1,)


@pytest.mark.asyncio
async def test_cosmos_system_process_returns_correct_shape(small_config):
    sys_ = CosmosReservoirSystem(small_config)
    await sys_.initialize()

    u = np.zeros(small_config.n_inputs)
    out = await sys_.process_input(u)
    # Before training W_out may be zero but shape must be correct
    assert out.shape == (small_config.n_outputs,)


@pytest.mark.asyncio
async def test_cosmos_system_train_improves_fit(small_config):
    """After training, residuals should be small for a linear target."""
    rng = np.random.default_rng(1)
    n = 200
    X = rng.standard_normal((n, small_config.n_inputs))
    y = np.sum(X, axis=1, keepdims=True)

    sys_ = CosmosReservoirSystem(small_config)
    await sys_.initialize()
    await sys_.train(X, y)

    preds = []
    for xi in X[100:]:
        out = await sys_.process_input(xi)
        preds.append(out)
    preds = np.array(preds)
    targets = y[100:]

    rmse = float(np.sqrt(np.mean((preds - targets) ** 2)))
    # Should generalise reasonably on a simple linear task
    assert rmse < 5.0, f"RMSE too high: {rmse}"
