"""Unit tests for all 6 Autonomic Triad services."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import asyncio
import numpy as np
import pytest
from cosmos_core import ServiceConfig, Triad, Polarity, ServicePosition, Dimension, create_message
from autonomic_triad.input_monitoring import InputMonitoringService
from autonomic_triad.signal_state import SignalStateService
from autonomic_triad.preprocessing_director import PreprocessingDirectorService
from autonomic_triad.transform_processing import TransformProcessingService
from autonomic_triad.input_organization import InputOrganizationService
from autonomic_triad.encoding_triggers import EncodingTriggersService


def make_cfg(name, pos):
    return ServiceConfig(name, Triad.AUTONOMIC, pos, Polarity.SYMPATHETIC, Dimension.PERFORMANCE)


# ---------- M-1: InputMonitoringService ----------

def test_input_monitoring_valid():
    svc = InputMonitoringService(make_cfg("m1", ServicePosition.M1), input_dim=3)
    asyncio.run(svc.initialize())
    msg = create_message("RAW_INPUT", np.array([1.0, 2.0, 3.0]), "ext")
    result = asyncio.run(svc.process(msg))
    assert result is not None
    assert result.type == "VALIDATED_INPUT"
    np.testing.assert_array_almost_equal(result.payload, [1.0, 2.0, 3.0])


def test_input_monitoring_wrong_dim():
    svc = InputMonitoringService(make_cfg("m1", ServicePosition.M1), input_dim=3)
    asyncio.run(svc.initialize())
    msg = create_message("RAW_INPUT", np.array([1.0, 2.0]), "ext")
    result = asyncio.run(svc.process(msg))
    assert result is None


def test_input_monitoring_nan():
    svc = InputMonitoringService(make_cfg("m1", ServicePosition.M1), input_dim=3)
    asyncio.run(svc.initialize())
    msg = create_message("RAW_INPUT", np.array([1.0, np.nan, 3.0]), "ext")
    result = asyncio.run(svc.process(msg))
    assert result is not None
    assert np.all(np.isfinite(result.payload))


def test_input_monitoring_scaling():
    svc = InputMonitoringService(make_cfg("m1", ServicePosition.M1), input_dim=2, input_scaling=2.0)
    asyncio.run(svc.initialize())
    msg = create_message("RAW_INPUT", np.array([1.0, 1.0]), "ext")
    result = asyncio.run(svc.process(msg))
    np.testing.assert_array_almost_equal(result.payload, [2.0, 2.0])


# ---------- S-8: SignalStateService ----------

def test_signal_state_buffer():
    svc = SignalStateService(make_cfg("s8", ServicePosition.S8), window_size=3)
    asyncio.run(svc.initialize())
    result = None
    for i in range(5):
        msg = create_message("VALIDATED_INPUT", np.array([float(i)]), "ext")
        result = asyncio.run(svc.process(msg))
    # Only 3 items in buffer after 5 inserts
    assert len(svc.buffer) == 3
    assert result.type == "BUFFERED_INPUT"


# ---------- PD-2: PreprocessingDirectorService ----------

def test_preprocessing_zscore():
    svc = PreprocessingDirectorService(make_cfg("pd2", ServicePosition.PD2), method="zscore")
    asyncio.run(svc.initialize())
    data = np.random.randn(100, 4)
    svc.fit(data)
    x = np.array([1.0, 2.0, 3.0, 4.0])
    result = svc.transform(x)
    assert result.shape == (4,)


def test_preprocessing_minmax():
    svc = PreprocessingDirectorService(make_cfg("pd2", ServicePosition.PD2), method="minmax")
    asyncio.run(svc.initialize())
    data = np.array([[0.0, 0.0], [1.0, 2.0], [2.0, 4.0]])
    svc.fit(data)
    result = svc.transform(np.array([1.0, 2.0]))
    assert np.all(result >= 0.0) and np.all(result <= 1.0)


def test_preprocessing_pca():
    svc = PreprocessingDirectorService(make_cfg("pd2", ServicePosition.PD2),
                                       method="zscore", n_components=2)
    asyncio.run(svc.initialize())
    data = np.random.randn(50, 5)
    svc.fit(data)
    result = svc.transform(data[0])
    assert result.shape == (2,)


# ---------- P-5: TransformProcessingService ----------

def test_transform_none():
    svc = TransformProcessingService(make_cfg("p5", ServicePosition.P5))
    asyncio.run(svc.initialize())
    msg = create_message("PREPROCESSED_INPUT", np.array([1.0, 2.0, 3.0]), "ext")
    result = asyncio.run(svc.process(msg))
    assert result.type == "TRANSFORMED_INPUT"
    np.testing.assert_array_almost_equal(result.payload, [1.0, 2.0, 3.0])


def test_transform_fft():
    svc = TransformProcessingService(make_cfg("p5", ServicePosition.P5), transform="fft")
    asyncio.run(svc.initialize())
    x = np.sin(2 * np.pi * np.arange(16) / 16)
    msg = create_message("PREPROCESSED_INPUT", x, "ext")
    result = asyncio.run(svc.process(msg))
    assert result is not None
    assert len(result.payload) == 9  # rfft of length 16 -> 9 components


# ---------- O-4: InputOrganizationService ----------

def test_input_organization():
    svc = InputOrganizationService(make_cfg("o4", ServicePosition.O4), window_size=3)
    asyncio.run(svc.initialize())
    result = None
    for i in range(5):
        msg = create_message("PREPROCESSED_INPUT", np.array([float(i), float(i)]), "ext")
        result = asyncio.run(svc.process(msg))
    assert result.type == "ORGANIZED_INPUT"
    assert result.payload.shape[0] <= 3


# ---------- T-7: EncodingTriggersService ----------

def test_encoding_kwta():
    svc = EncodingTriggersService(make_cfg("t7", ServicePosition.T7), k=2)
    asyncio.run(svc.initialize())
    msg = create_message("PREPROCESSED_INPUT", np.array([1.0, 5.0, 3.0, 2.0, 4.0]), "ext")
    result = asyncio.run(svc.process(msg))
    assert result.type == "ENCODED_INPUT"
    # Only 2 non-zero values
    assert np.sum(result.payload != 0) == 2


def test_encoding_threshold():
    svc = EncodingTriggersService(make_cfg("t7", ServicePosition.T7), threshold=3.0)
    asyncio.run(svc.initialize())
    msg = create_message("PREPROCESSED_INPUT", np.array([1.0, 5.0, 2.0, 4.0]), "ext")
    result = asyncio.run(svc.process(msg))
    assert result.payload[0] == 0.0   # 1.0 < 3.0
    assert result.payload[1] == 5.0   # 5.0 >= 3.0
