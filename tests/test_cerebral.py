"""Unit tests for all 6 Cerebral Triad services."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import asyncio
import numpy as np
import pytest
from cosmos_core import ServiceConfig, Triad, Polarity, ServicePosition, Dimension, create_message
from cerebral_triad.target_interface import TargetInterfaceService
from cerebral_triad.output_delivery import OutputDeliveryService
from cerebral_triad.readout_processing import ReadoutProcessingService
from cerebral_triad.output_organization import OutputOrganizationService
from cerebral_triad.learning_director import LearningDirectorService
from cerebral_triad.weight_treasury import WeightTreasuryService


def make_cfg(name, pos):
    return ServiceConfig(name, Triad.CEREBRAL, pos, Polarity.SOMATIC, Dimension.COMMITMENT)


# ---------- M-1: TargetInterfaceService ----------

def test_target_interface_mse():
    svc = TargetInterfaceService(make_cfg("m1", ServicePosition.M1), loss="mse")
    asyncio.run(svc.initialize())
    pred = np.array([1.0, 2.0])
    target = np.array([1.5, 2.5])
    loss = svc.compute_loss(pred, target)
    assert abs(loss - 0.25) < 1e-6


def test_target_interface_mae():
    svc = TargetInterfaceService(make_cfg("m1", ServicePosition.M1), loss="mae")
    asyncio.run(svc.initialize())
    loss = svc.compute_loss(np.array([1.0]), np.array([2.0]))
    assert abs(loss - 1.0) < 1e-6


# ---------- S-8: OutputDeliveryService ----------

def test_output_delivery_regression():
    svc = OutputDeliveryService(make_cfg("s8", ServicePosition.S8), task="regression")
    asyncio.run(svc.initialize())
    msg = create_message("READOUT_OUTPUT", np.array([0.5, 1.5]), "ext")
    result = asyncio.run(svc.process(msg))
    assert result is not None
    assert "prediction" in result.payload
    assert "confidence" in result.payload


def test_output_delivery_classification():
    svc = OutputDeliveryService(make_cfg("s8", ServicePosition.S8), task="classification")
    asyncio.run(svc.initialize())
    msg = create_message("READOUT_OUTPUT", np.array([0.1, 0.9, 0.5]), "ext")
    result = asyncio.run(svc.process(msg))
    assert result.payload["prediction"] == 1
    assert 0.0 <= result.payload["confidence"] <= 1.0


# ---------- P-5: ReadoutProcessingService ----------

def test_readout_processing_train():
    svc = ReadoutProcessingService(make_cfg("p5", ServicePosition.P5),
                                   reservoir_dim=10, output_dim=1)
    asyncio.run(svc.initialize())
    states = np.random.randn(50, 10)
    targets = np.random.randn(50, 1)
    svc.train(states, targets, ridge_lambda=1e-4)
    assert svc.W_out.shape == (1, 11)


def test_readout_processing_predict():
    svc = ReadoutProcessingService(make_cfg("p5", ServicePosition.P5),
                                   reservoir_dim=5, output_dim=2)
    asyncio.run(svc.initialize())
    states = np.random.randn(30, 5)
    targets = np.random.randn(30, 2)
    svc.train(states, targets)
    state = np.random.randn(5)
    msg = create_message("RESERVOIR_STATE", state, "ext")
    result = asyncio.run(svc.process(msg))
    assert result.type == "READOUT_OUTPUT"
    assert result.payload.shape == (2,)


# ---------- O-4: OutputOrganizationService ----------

def test_output_organization():
    svc = OutputOrganizationService(make_cfg("o4", ServicePosition.O4))
    asyncio.run(svc.initialize())
    msg = create_message("READOUT_OUTPUT", np.array([1.0, 2.0]), "ext")
    result = asyncio.run(svc.process(msg))
    assert result.type == "FINAL_OUTPUT"
    assert "output" in result.payload


# ---------- PD-2: LearningDirectorService ----------

def test_learning_director_ridge():
    svc = LearningDirectorService(make_cfg("pd2", ServicePosition.PD2), method="ridge")
    asyncio.run(svc.initialize())
    X = np.random.randn(100, 20)
    Y = np.random.randn(100, 2)
    W_out = svc.batch_train(X, Y)
    assert W_out.shape == (2, 21)


def test_learning_director_rls():
    svc = LearningDirectorService(make_cfg("pd2", ServicePosition.PD2), method="rls")
    asyncio.run(svc.initialize())
    svc.rls_init(feature_dim=10, output_dim=1)
    for _ in range(20):
        state = np.random.randn(10)
        target = np.array([np.sum(state[:3])])
        svc.rls_update(state, target)
    assert svc._W_out.shape == (1, 11)


# ---------- T-7: WeightTreasuryService ----------

def test_weight_treasury(tmp_path):
    svc = WeightTreasuryService(make_cfg("t7", ServicePosition.T7))
    asyncio.run(svc.initialize())
    W = np.random.randn(3, 5)
    svc.store("test_weights", W)
    retrieved = svc.retrieve("test_weights")
    np.testing.assert_array_equal(W, retrieved)

    path = str(tmp_path / "weights.npz")
    svc.save(path)

    svc2 = WeightTreasuryService(make_cfg("t7b", ServicePosition.T7))
    asyncio.run(svc2.initialize())
    svc2.load(path)
    np.testing.assert_array_equal(svc2.retrieve("test_weights"), W)
