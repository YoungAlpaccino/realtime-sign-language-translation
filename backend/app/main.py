"""FastAPI entrypoint for the sign-rt backend.

WS hub fans keypoint/gloss streams to subscribers; serves ONNX + LM artifacts;
runs gloss->text translation on demand; persists transcripts in SQLite.
"""
from __future__ import annotations

import json
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from pydantic import BaseModel
from sqlmodel import select

from .auth import create_token, verify_token
from .models import Session, Transcript, engine, get_session, init_db
from .ws_hub import WSHub

ARTIFACTS_DIR = Path(__file__).resolve().parents[2] / "artifacts"

hub = WSHub()


@asynccontextmanager
async def lifespan(app: FastAPI):  # noqa: ANN201
    init_db()
    yield


app = FastAPI(title="sign-rt backend", version="0.0.1", lifespan=lifespan)


@app.get("/api/health")
async def health() -> dict[str, str]:
    """Liveness probe."""
    return {"status": "ok"}


class TokenRequest(BaseModel):
    subject: str = "demo-user"


@app.post("/api/token")
async def issue_token(req: TokenRequest) -> dict[str, str]:
    """Dev helper: mint a JWT for a subject. (Replace with real auth in prod.)"""
    return {"token": create_token(req.subject)}


class TranslateRequest(BaseModel):
    gloss: str
    target_lang: str = "en"


class TranslateResponse(BaseModel):
    text: str
    source: str = "placeholder"


@app.post("/api/translate", response_model=TranslateResponse)
async def translate(req: TranslateRequest) -> TranslateResponse:
    """Gloss -> text (SLT).

    Placeholder until the SLT Transformer is trained (Phase 5): returns a
    lightly-templated rendering of the gloss sequence so the surface is wired
    end-to-end. Swap in `core.sign_rt.slt` once checkpoints exist.
    """
    glosses = [g for g in req.gloss.replace("\n", " ").split() if g]
    text = " ".join(glosses).lower().capitalize()
    if text and not text.endswith("."):
        text += "."
    return TranslateResponse(text=text, source="placeholder")


@app.get("/api/sessions/{session_id}/transcripts")
async def list_transcripts(session_id: str) -> list[Transcript]:
    """Return stored transcript records for a session."""
    with get_session() as db:
        rows = db.exec(
            select(Transcript).where(Transcript.session_id == session_id).order_by(Transcript.t_start)
        ).all()
        return list(rows)


@app.get("/api/artifacts/{name}")
async def get_artifact(name: str) -> FileResponse:
    """Serve an ONNX / KenLM artifact by name (e.g. onnx/recognizer_ctr_gcn.onnx)."""
    # Prevent path traversal; resolve under ARTIFACTS_DIR only.
    target = (ARTIFACTS_DIR / name).resolve()
    if not str(target).startswith(str(ARTIFACTS_DIR.resolve())) or not target.is_file():
        raise HTTPException(status_code=404, detail="artifact not found")
    return FileResponse(target)


def _ensure_session(session_id: str, keypoints_only: bool) -> None:
    with get_session() as db:
        if db.get(Session, session_id) is None:
            db.add(Session(id=session_id, keypoints_only=keypoints_only))
            db.commit()


def _persist_segment(session_id: str, msg: dict) -> None:
    with get_session() as db:
        db.add(
            Transcript(
                session_id=session_id,
                t_start=float(msg.get("t_start", 0.0)),
                t_end=float(msg.get("t_end", 0.0)),
                gloss=str(msg.get("gloss", "")),
                text=str(msg.get("text", "")),
                peaks=json.dumps(msg.get("peaks", [])),
            )
        )
        db.commit()


@app.websocket("/ws")
async def ws_endpoint(
    ws: WebSocket,
    session: str = Query(..., description="session id to join"),
    token: str = Query(..., description="JWT"),
) -> None:
    """Join a session: fan-in frames from publishers, fan-out to subscribers.

    Frame shapes:
      {"type": "frame",   "t": float, "keypoints": [...], "partial_gloss": [...]}
      {"type": "segment", "t_start": float, "t_end": float, "gloss": str, ...}
    """
    try:
        verify_token(token)
    except ValueError:
        await ws.close(code=4401)  # unauthorized
        return

    await ws.accept()
    _ensure_session(session, keypoints_only=True)
    await hub.connect(session, ws)
    try:
        while True:
            msg = await ws.receive_json()
            if msg.get("type") == "segment":
                _persist_segment(session, msg)
            await hub.broadcast(session, msg, exclude=ws)
    except WebSocketDisconnect:
        pass
    finally:
        await hub.disconnect(session, ws)


# Re-export for convenience / tests.
__all__ = ["app", "hub", "engine"]
