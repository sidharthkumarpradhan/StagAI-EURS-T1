"""Track 2 submission package (Track 1 lives in ``macvi.eurs.track1``)."""

from __future__ import annotations

import shutil
from pathlib import Path

from macvi.eurs.onnx_tools import check


def track2_package(onnx_path: str | Path, out_dir: str | Path) -> Path:
    report = check(onnx_path)
    if not report.ok:
        raise ValueError(f"not Track 2 compliant: {report.problems}")
    out = Path(out_dir) / Path(onnx_path).name
    out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(onnx_path, out)
    return out
