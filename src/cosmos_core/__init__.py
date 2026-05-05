"""
COSYS Core: Shared primitives for the Cosmos System 5 architecture.
"""
import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional


class Triad(Enum):
    AUTONOMIC = "autonomic"
    SOMATIC = "somatic"
    CEREBRAL = "cerebral"


class Polarity(Enum):
    SYMPATHETIC = "sympathetic"
    PARASYMPATHETIC = "parasympathetic"
    SOMATIC = "somatic"


class ServicePosition(Enum):
    M1 = "M-1"
    S8 = "S-8"
    P5 = "P-5"
    O4 = "O-4"
    PD2 = "PD-2"
    T7 = "T-7"


class Dimension(Enum):
    PERFORMANCE = "performance"
    COMMITMENT = "commitment"
    POTENTIAL = "potential"


@dataclass
class ServiceConfig:
    service_name: str
    triad: Triad
    position: ServicePosition
    polarity: Polarity
    dimension: Dimension


@dataclass
class ServiceMessage:
    type: str
    payload: Any
    source: str
    target: Optional[str] = None
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)


def create_message(msg_type: str, payload: Any, source: str, target: Optional[str] = None) -> ServiceMessage:
    return ServiceMessage(type=msg_type, payload=payload, source=source, target=target)


def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )


class BaseCosmosService:
    def __init__(self, config: ServiceConfig):
        self.config = config
        self.initialized = False
        self._logger = logging.getLogger(config.service_name)

    def log(self, level: str, msg: str) -> None:
        getattr(self._logger, level, self._logger.info)(msg)

    async def initialize(self) -> None:
        raise NotImplementedError

    async def process(self, message: ServiceMessage) -> Optional[ServiceMessage]:
        raise NotImplementedError

    async def shutdown(self) -> None:
        raise NotImplementedError


class TriadicCoordinator:
    def __init__(self):
        self._services: List[BaseCosmosService] = []

    def register_service(self, service: BaseCosmosService) -> None:
        self._services.append(service)

    def get_services(self) -> List[BaseCosmosService]:
        return list(self._services)

    def get_by_triad(self, triad: Triad) -> List[BaseCosmosService]:
        return [s for s in self._services if s.config.triad == triad]
