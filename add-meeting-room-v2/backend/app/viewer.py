import asyncio
import logging
from typing import Any

from fastapi import WebSocket
from starlette.websockets import WebSocketState


logger = logging.getLogger(__name__)


class ViewerHub:
    def __init__(self) -> None:
        self._viewers: set[WebSocket] = set()
        self._history: list[dict[str, Any]] = []
        self._lock = asyncio.Lock()

    async def connect(self, websocket: WebSocket) -> None:
        await websocket.accept()
        async with self._lock:
            self._viewers.add(websocket)
            history = list(self._history)

        for event in history:
            await self._send_json(websocket, event)

    async def disconnect(self, websocket: WebSocket) -> None:
        async with self._lock:
            self._viewers.discard(websocket)

    async def broadcast(self, event: dict[str, Any]) -> None:
        async with self._lock:
            self._history.append(event)
            viewers = list(self._viewers)

        stale: list[WebSocket] = []
        for viewer in viewers:
            try:
                await self._send_json(viewer, event)
            except Exception:
                logger.info("viewer send failed; dropping connection", exc_info=True)
                stale.append(viewer)

        if stale:
            async with self._lock:
                for viewer in stale:
                    self._viewers.discard(viewer)

    async def _send_json(self, websocket: WebSocket, event: dict[str, Any]) -> None:
        if websocket.client_state != WebSocketState.CONNECTED:
            raise RuntimeError("viewer websocket is not connected")
        await websocket.send_json(event)


viewer_hub = ViewerHub()
