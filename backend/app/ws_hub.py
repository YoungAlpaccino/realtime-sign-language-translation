"""WebSocket hub: fan-in keypoint/gloss streams, fan-out to subscribers per session."""
from __future__ import annotations

import asyncio
from typing import Any


class WSHub:
    """Tracks connected clients per session and broadcasts frames.

    A session is a many-to-many channel: edge runners (or browser clients)
    publish `{t, keypoints, partial_gloss}` frames, and every other client
    subscribed to the same session id receives them.
    """

    def __init__(self) -> None:
        self._sessions: dict[str, set] = {}
        self._lock = asyncio.Lock()

    async def connect(self, session_id: str, ws) -> None:  # noqa: ANN001
        """Register a client connection under a session."""
        async with self._lock:
            self._sessions.setdefault(session_id, set()).add(ws)

    async def disconnect(self, session_id: str, ws) -> None:  # noqa: ANN001
        """Remove a connection; drop the session when empty."""
        async with self._lock:
            conns = self._sessions.get(session_id)
            if not conns:
                return
            conns.discard(ws)
            if not conns:
                self._sessions.pop(session_id, None)

    async def broadcast(self, session_id: str, message: dict, exclude: Any = None) -> None:
        """Fan-out a {t, keypoints, partial_gloss} message to session subscribers."""
        async with self._lock:
            targets = [c for c in self._sessions.get(session_id, set()) if c is not exclude]
        dead = []
        for c in targets:
            try:
                await c.send_json(message)
            except Exception:  # noqa: BLE001 - connection died mid-broadcast
                dead.append(c)
        for c in dead:
            await self.disconnect(session_id, c)

    def session_count(self, session_id: str) -> int:
        """Number of live connections for a session (diagnostics)."""
        return len(self._sessions.get(session_id, set()))
