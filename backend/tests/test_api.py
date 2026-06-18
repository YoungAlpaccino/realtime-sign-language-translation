"""Backend API + WebSocket tests."""
from __future__ import annotations

import os

from fastapi.testclient import TestClient


def _client(tmp_path) -> TestClient:
    os.environ["SIGN_RT_DB_URL"] = f"sqlite:///{tmp_path / 'api_test.db'}"
    # Import after env is set so the engine binds to the temp DB.
    from backend.app.main import app

    return TestClient(app)


def test_health_and_token_and_translate(tmp_path):
    with _client(tmp_path) as c:
        assert c.get("/api/health").json() == {"status": "ok"}
        tok = c.post("/api/token", json={"subject": "u1"}).json()["token"]
        assert isinstance(tok, str) and len(tok) > 20
        r = c.post("/api/translate", json={"gloss": "HELLO WORLD"}).json()
        assert r["text"] == "Hello world."


def test_ws_fanout_and_persistence(tmp_path):
    with _client(tmp_path) as c:
        tok = c.post("/api/token", json={"subject": "u1"}).json()["token"]
        url = f"/ws?session=s1&token={tok}"
        with c.websocket_connect(url) as a, c.websocket_connect(url) as b:
            a.send_json({"type": "frame", "t": 0.1, "keypoints": [1, 2, 3]})
            assert b.receive_json()["t"] == 0.1
            a.send_json(
                {"type": "segment", "t_start": 0.0, "t_end": 1.0,
                 "gloss": "HELLO WORLD", "text": "hello world", "peaks": [0.2, 0.6]}
            )
            b.receive_json()

        rows = c.get("/api/sessions/s1/transcripts").json()
        assert len(rows) == 1
        assert rows[0]["gloss"] == "HELLO WORLD"
        assert rows[0]["peaks"] == "[0.2, 0.6]"


def test_ws_rejects_bad_token(tmp_path):
    with _client(tmp_path) as c:
        import pytest
        from starlette.websockets import WebSocketDisconnect

        with pytest.raises(WebSocketDisconnect):
            with c.websocket_connect("/ws?session=s1&token=bogus") as z:
                z.receive_json()


def test_artifact_path_traversal_blocked(tmp_path):
    with _client(tmp_path) as c:
        assert c.get("/api/artifacts/../../etc/passwd").status_code in (404, 400)


def test_artifact_serves_nested_path(tmp_path):
    """Regression: {name:path} must match nested subpaths like onnx/foo.onnx."""
    from backend.app.main import ARTIFACTS_DIR

    sub = ARTIFACTS_DIR / "onnx"
    sub.mkdir(parents=True, exist_ok=True)
    f = sub / "_pytest_probe.bin"
    f.write_bytes(b"hello-artifact")
    try:
        with _client(tmp_path) as c:
            r = c.get("/api/artifacts/onnx/_pytest_probe.bin")
            assert r.status_code == 200
            assert r.content == b"hello-artifact"
    finally:
        f.unlink(missing_ok=True)
