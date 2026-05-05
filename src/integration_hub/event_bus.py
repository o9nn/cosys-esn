"""
EventBus: Async pub/sub for inter-service communication.
"""
import asyncio
from typing import Callable, Dict, List
from cosmos_core import ServiceMessage


class EventBus:
    """Async publish/subscribe event bus."""

    def __init__(self):
        self._subscribers: Dict[str, List[Callable]] = {}

    def subscribe(self, topic: str, handler: Callable) -> None:
        self._subscribers.setdefault(topic, []).append(handler)

    async def publish(self, message: ServiceMessage) -> List[ServiceMessage]:
        handlers = self._subscribers.get(message.type, [])
        results = []
        for handler in handlers:
            try:
                result = await handler(message)
                if result is not None:
                    results.append(result)
            except Exception:
                pass
        return results
