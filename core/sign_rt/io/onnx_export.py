"""ONNX export/load with a round-trip parity check across surfaces."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

OPSET = 17


def export_onnx(model: torch.nn.Module, dummy: torch.Tensor, out_path: str | Path) -> Path:
    """Export a model to ONNX with a dynamic time axis.

    One graph, three surfaces (edge / server / browser). Returns the written path.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    model.eval()
    torch.onnx.export(
        model,
        dummy,
        str(out_path),
        input_names=["keypoints"],
        output_names=["log_probs"],
        # Time axis (dim 2 of input, dim 1 of output) is dynamic for streaming.
        dynamic_axes={"keypoints": {0: "B", 2: "T"}, "log_probs": {0: "B", 1: "T"}},
        opset_version=OPSET,
        do_constant_folding=True,
    )
    return out_path


def check_parity(onnx_path: str | Path, dummy: torch.Tensor, ref_logits: torch.Tensor) -> float:
    """Run the ONNX graph and return max |Δlogit| vs the PyTorch reference.

    CI gate: max logit Δ must be < 1e-2 across surfaces.
    """
    import onnxruntime as ort

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    (out,) = sess.run(["log_probs"], {"keypoints": dummy.detach().cpu().numpy().astype(np.float32)})
    return float(np.abs(out - ref_logits.detach().cpu().numpy()).max())
