"""Python<->TS parity: generate a fixture from Python, verify the TS ports match.

The fixture is consumed by frontend/scripts/parity.ts, run under Node's
type-stripping. The test is skipped if Node is unavailable.
"""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from sign_rt.io.keypoint_codec import encode
from sign_rt.pose.holistic import L_SHOULDER, N_KEYPOINTS, R_SHOULDER, normalize_keypoints
from sign_rt.seq.ctc_head import ctc_collapse

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "frontend" / "src" / "core_port" / "__fixtures__" / "parity.json"
PARITY_SCRIPT = ROOT / "frontend" / "scripts" / "parity.ts"


def _build_fixture() -> dict:
    rng = np.random.RandomState(7)

    collapse = []
    for ids in ([0, 0, 1, 1, 2, 0, 2, 2], [3, 3, 3], [0, 0, 0], [1, 0, 1, 0, 1]):
        collapse.append({"input": ids, "blank": 0, "expected": ctc_collapse(ids, 0)})

    normalize = []
    for _ in range(3):
        k = (rng.rand(N_KEYPOINTS, 3).astype(np.float32) - 0.3) * 2
        k[L_SHOULDER, :2] = [0.45, 0.5]
        k[R_SHOULDER, :2] = [0.55, 0.5]
        nk = normalize_keypoints(k)
        normalize.append({"input": k.reshape(-1).tolist(), "expected": nk.reshape(-1).tolist()})

    codec = []
    for _ in range(2):
        k = (rng.rand(N_KEYPOINTS, 3).astype(np.float32) - 0.5) * 4
        payload = encode(k)
        decoded = (np.frombuffer(payload[6:], dtype="<i2").astype(np.float32) / 4096.0)
        codec.append({"payload_hex": payload.hex(), "expected": decoded.tolist()})

    return {"collapse": collapse, "normalize": normalize, "codec": codec}


def test_write_parity_fixture():
    FIXTURE.parent.mkdir(parents=True, exist_ok=True)
    FIXTURE.write_text(json.dumps(_build_fixture()), encoding="utf-8")
    assert FIXTURE.is_file()


@pytest.mark.skipif(shutil.which("node") is None, reason="node not available")
def test_cross_language_parity():
    FIXTURE.parent.mkdir(parents=True, exist_ok=True)
    FIXTURE.write_text(json.dumps(_build_fixture()), encoding="utf-8")
    proc = subprocess.run(
        ["node", "--experimental-strip-types", str(PARITY_SCRIPT)],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, f"TS parity failed:\nSTDOUT {proc.stdout}\nSTDERR {proc.stderr}"
