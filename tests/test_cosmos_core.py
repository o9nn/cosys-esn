"""Tests for cosmos_core primitives."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pytest
from cosmos_core import (
    Triad, Polarity, ServicePosition, Dimension,
    ServiceConfig, ServiceMessage, create_message, setup_logging,
    BaseCosmosService, TriadicCoordinator,
)


def test_enums():
    assert Triad.AUTONOMIC.value == "autonomic"
    assert Polarity.SOMATIC.value == "somatic"
    assert ServicePosition.M1.value == "M-1"
    assert ServicePosition.P5.value == "P-5"
    assert Dimension.COMMITMENT.value == "commitment"


def test_service_config():
    cfg = ServiceConfig("test", Triad.SOMATIC, ServicePosition.P5,
                        Polarity.SOMATIC, Dimension.COMMITMENT)
    assert cfg.service_name == "test"
    assert cfg.triad == Triad.SOMATIC


def test_create_message():
    msg = create_message("TEST", {"data": 1}, "src", "dst")
    assert msg.type == "TEST"
    assert msg.payload == {"data": 1}
    assert msg.source == "src"
    assert msg.target == "dst"


def test_triadic_coordinator():
    class DummyService(BaseCosmosService):
        async def initialize(self): pass
        async def process(self, m): return None
        async def shutdown(self): pass

    coord = TriadicCoordinator()
    cfg = ServiceConfig("s1", Triad.AUTONOMIC, ServicePosition.M1,
                        Polarity.SYMPATHETIC, Dimension.PERFORMANCE)
    svc = DummyService(cfg)
    coord.register_service(svc)
    assert len(coord.get_services()) == 1
    assert len(coord.get_by_triad(Triad.AUTONOMIC)) == 1
    assert len(coord.get_by_triad(Triad.SOMATIC)) == 0
