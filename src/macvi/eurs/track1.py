"""Track 1 submission: refit on every train scene, predict every Dev image, verify, zip.

Official rules (quoted from the challenge page):
- "a single .zip archive containing two folders at its root: restored/ and seg_masks/"
- "Segmentation masks must be single-channel PNGs with integer pixel values 0-4"
- "Every output must be resized to exactly the resolution the evaluator scores at (longest
  side capped at 1024px, aspect-preserved from the original scene)"; "a mismatched size is
  rejected outright"
- "Run the starter kit's resize_for_submission.py as the mandatory last step"
- "A submission must include both outputs for every Dev-split image."

The Dev split is released as degraded images only (``eurs_dev_degraded.zip``); its ground
truth stays on the server. So the model trains on ``eurs_train`` and predicts the Dev images.

The official resize script (read on 2 Oct 2026) takes ``--degraded-dir --restored-dir
--masks-dir --output-dir`` and requires an output for every file in ``--degraded-dir``.
``official_resize`` recognises that CLI from ``--help``; any other CLI needs
``track1.resize_cmd``. A zip built without the official step is named ``*_UNVERIFIED.zip``
so it cannot be mistaken for an upload.
"""

from __future__ import annotations

import hashlib
import os
import shlex
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

from macvi.eurs import fusion
from macvi.eurs.data import read_rgb
from macvi.eurs.evaluate import predict_torch, scoring_size

RESIZE_URL = "https://macvi.org/EURS_resize_for_submission.py"
OFFICIAL_FLAGS = ("--degraded-dir", "--restored-dir", "--masks-dir", "--output-dir")


def predict_dev(model, images: Iterable[Path], size: tuple[int, int], out_dir: Path, device: str = "cpu",
                tta: bool = False) -> Path:
    """Write restored/<stem>.png and seg_masks/<stem>.png for every image at the scoring size."""
    if out_dir.exists():
        shutil.rmtree(out_dir)  # never zip stale files from an earlier run
    (out_dir / "restored").mkdir(parents=True, exist_ok=True)
    (out_dir / "seg_masks").mkdir(parents=True, exist_ok=True)
    for path in images:
        img = read_rgb(Path(path))
        h, w = scoring_size(*img.shape[:2])
        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
        restored, mask = predict_torch(model, img, size, device, tta, path=path)
        cv2.imwrite(str(out_dir / "restored" / f"{Path(path).stem}.png"), cv2.cvtColor(restored, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(out_dir / "seg_masks" / f"{Path(path).stem}.png"), mask)
    return out_dir


def verify(pred_dir: Path, images: Iterable[Path], num_classes: int = 5) -> list[str]:
    """Every rule we can check locally; an empty list means the folder is upload-shaped."""
    problems = []
    for path in images:
        stem = Path(path).stem
        h, w = scoring_size(*cv2.imread(str(path), cv2.IMREAD_UNCHANGED).shape[:2])
        r = cv2.imread(str(pred_dir / "restored" / f"{stem}.png"), cv2.IMREAD_UNCHANGED)
        m = cv2.imread(str(pred_dir / "seg_masks" / f"{stem}.png"), cv2.IMREAD_UNCHANGED)
        if r is None or m is None:
            problems.append(f"{stem}: missing restored or mask")
            continue
        if r.shape[:2] != (h, w) or m.shape[:2] != (h, w):
            problems.append(f"{stem}: size {r.shape[:2]}/{m.shape[:2]} != {(h, w)}")
        if r.ndim != 3 or r.shape[2] != 3:
            problems.append(f"{stem}: restored image is not 3-channel")
        if m.ndim != 2:
            problems.append(f"{stem}: mask is not single-channel")
        elif m.max() >= num_classes or m.dtype != np.uint8:
            problems.append(f"{stem}: mask values/dtype invalid ({m.dtype}, max {m.max()})")
    expected = {Path(p).stem for p in images}
    for sub in ("restored", "seg_masks"):
        extra = {p.stem for p in (pred_dir / sub).iterdir()} - expected if (pred_dir / sub).is_dir() else set()
        if extra:
            problems.append(f"{len(extra)} unexpected files in {sub}/, e.g. {sorted(extra)[:3]}")
    return problems


def degraded_dir(images: list[Path], staging: Path) -> Path:
    """A folder holding exactly ``images``: the official script wants an output for every file in it.

    The Dev images' own folder is used when it contains nothing else; otherwise they are copied.
    """
    parents = {Path(p).parent for p in images}
    if len(parents) == 1:
        parent = parents.pop()
        if sorted(p.name for p in parent.iterdir() if p.is_file()) == sorted(Path(p).name for p in images):
            return parent
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    for p in images:
        shutil.copy2(p, staging / Path(p).name)
    return staging


def official_resize(pred_dir: Path, images: list[Path], cmd_template: str | None = None,
                    cache_dir: Path = Path("external")) -> tuple[Path | None, str]:
    """Run the organisers' resize script on ``pred_dir``. Returns (output dir or None, log).

    ``cmd_template`` placeholders: {python} {script} {src} (our restored/ + seg_masks/ folder),
    {dst} (output folder) and {degraded} (folder of exactly the Dev images).
    """
    script = cache_dir / "EURS_resize_for_submission.py"
    try:
        if not script.exists():
            script.parent.mkdir(parents=True, exist_ok=True)
            urllib.request.urlretrieve(RESIZE_URL, script)
    except OSError as exc:
        return None, f"official script unreachable ({type(exc).__name__}); run this step in Colab"
    help_text = subprocess.run([sys.executable, str(script), "--help"], capture_output=True, text=True)
    help_text = help_text.stdout + help_text.stderr
    if cmd_template is None and not all(flag in help_text for flag in OFFICIAL_FLAGS):
        return None, f"unrecognised CLI; set track1.resize_cmd using this help text:\n{help_text}"
    dst = pred_dir.with_name(pred_dir.name + "_official")
    if dst.exists():
        shutil.rmtree(dst)
    degraded = degraded_dir(images, pred_dir.with_name(pred_dir.name + "_degraded"))
    if cmd_template is None:
        argv = [sys.executable, str(script), "--degraded-dir", str(degraded), "--restored-dir", str(pred_dir / "restored"),
                "--masks-dir", str(pred_dir / "seg_masks"), "--output-dir", str(dst)]
    else:
        cmd = cmd_template.format(python=sys.executable, script=script, src=pred_dir, dst=dst, degraded=degraded)
        argv = shlex.split(cmd, posix=os.name != "nt")
    proc = subprocess.run(argv, capture_output=True, text=True)
    if proc.returncode != 0:
        return None, f"official resize failed:\n{proc.stdout}\n{proc.stderr}"
    return dst, proc.stdout


def zip_dir(pred_dir: Path, zip_path: Path) -> Path:
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for sub in ("restored", "seg_masks"):
            for f in sorted((pred_dir / sub).glob("*.png")):
                zf.write(f, f"{sub}/{f.name}")
    return zip_path


MAX_ZIP_BYTES = 1000 * 10**6  # macvi.org: "maximum upload size is 1024 MB" (refused at 1.38 GB, 6 Oct 2026)


def quantize_png(data: bytes, step: int) -> bytes:
    """An RGB PNG with every channel rounded to the nearest multiple of ``step`` (pure)."""
    import io

    from PIL import Image

    a = np.asarray(Image.open(io.BytesIO(data)).convert("RGB")).astype(np.float32)
    out = io.BytesIO()
    Image.fromarray(np.clip(np.round(a / step) * step, 0, 255).astype(np.uint8)).save(out, "PNG")
    return out.getvalue()


def fit_zip(zip_path: Path, max_bytes: int = MAX_ZIP_BYTES, steps: Iterable[int] = (2, 3, 4, 5, 6, 8)) -> dict:
    """Make an upload zip fit the site's size limit: rewrite restored/ with the smallest colour step that fits.

    1024x768 restored PNGs average ~1.1 MB, so 1,200 Dev images (1.38 GB) exceed the limit. Rounding to a
    step of 3 adds an MSE of ~0.7 (PSNR -0.006 dB at our 21 dB) and cuts PNG size by a third, measured on
    track1_0a6952a5c0fb. Masks are copied unchanged. Returns {"bytes", "quant_step"} (step 1: untouched).
    """
    size = zip_path.stat().st_size
    if size <= max_bytes:
        return {"bytes": size, "quant_step": 1}
    for step in steps:
        tmp = zip_path.with_name(zip_path.name + ".fit")
        done: dict[bytes, bytes] = {}  # key-affine zips repeat one PNG per scene 200 times: quantize each once
        with zipfile.ZipFile(zip_path) as src, zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as dst:
            for info in src.infolist():
                data = src.read(info)
                if info.filename.startswith("restored/") and info.filename.endswith(".png"):
                    digest = hashlib.sha1(data).digest()
                    if digest not in done:
                        done[digest] = quantize_png(data, step)
                    data = done[digest]
                dst.writestr(info.filename, data)
        if tmp.stat().st_size <= max_bytes:
            os.replace(tmp, zip_path)
            return {"bytes": zip_path.stat().st_size, "quant_step": step}
        tmp.unlink()
    raise ValueError(f"{zip_path.name}: still over {max_bytes} bytes at colour step {max(steps)}")


def zip_problems(zip_path: Path, images: int | None = None) -> list[str]:
    """Shape check of an upload zip without unpacking it: only restored/<stem>.png and
    seg_masks/<stem>.png at the root, the same stems in both, ``images`` of each if given."""
    try:
        with zipfile.ZipFile(zip_path) as zf:
            names = [n for n in zf.namelist() if not n.endswith("/")]
    except (OSError, zipfile.BadZipFile) as exc:
        return [f"unreadable zip: {exc}"]
    stems: dict[str, set[str]] = {"restored": set(), "seg_masks": set()}
    problems = []
    for n in names:
        parts = n.split("/")
        if len(parts) != 2 or parts[0] not in stems or not parts[1].endswith(".png"):
            problems.append(f"unexpected member {n}")
        else:
            stems[parts[0]].add(parts[1][:-4])
    if stems["restored"] != stems["seg_masks"]:
        problems.append(f"{len(stems['restored'] ^ stems['seg_masks'])} stems lack a restored image or a mask")
    if images is not None and len(stems["restored"]) != images:
        problems.append(f"{len(stems['restored'])} restored images, expected {images}")
    if not names:
        problems.append("empty zip")
    return problems


def build_submission(cfg: dict, data_root: Path, out_dir: Path, dev_root: Path, refit: bool = True, model=None,
                     resize_cmd: str | None = None, device: str | None = None, save_epoch: int | None = None,
                     deadline: float | None = None) -> dict:
    """Refit on every train scene, predict all Dev images, verify, run the official resize, zip.

    ``deadline`` (a ``time.time()`` value) ends the refit early with its last epoch (remote jobs).
    """
    import torch

    from macvi.common.config import get, with_overrides
    from macvi.eurs.data import list_images
    from macvi.eurs.train import train

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    images = list_images(dev_root)  # fail before the refit, not after hours of training
    if refit or model is None:
        # Same schedule as selection; keep the epoch that won on held-out scenes (if known).
        overrides = {"data.val_fraction": 0, "train.save_epoch": save_epoch}
        model, _ = train(with_overrides(cfg, overrides), data_root, out_dir / "refit", deadline=deadline)
    model = model.to(device)
    fusion_cfg = fusion.settings(cfg)  # per-scene fusion (track1.fusion; owner-approved, disclosed below)
    tta = bool(get(cfg, "track1.tta", True))
    if fusion_cfg is None:
        pred, fused = predict_dev(model, images, tuple(get(cfg, "data.size")), out_dir / "pred", device, tta=tta), {}
    else:
        pred = out_dir / "pred"
        fused = fusion.predict_dev(model, images, tuple(get(cfg, "data.size")), pred, fusion_cfg, device, tta=tta)
    problems = verify(pred, images, get(cfg, "model.num_classes", 5))
    if problems:
        raise ValueError(f"Track 1 outputs failed verification: {problems[:5]}")
    official, log = official_resize(pred, images, resize_cmd)
    if official is not None:
        problems = verify(official, images, get(cfg, "model.num_classes", 5))
        if problems:
            raise ValueError(f"outputs after the official resize failed verification: {problems[:5]}")
    final = official or pred
    zip_path = zip_dir(final, out_dir / ("track1_submission.zip" if official else "track1_submission_UNVERIFIED.zip"))
    fit = fit_zip(zip_path, int(float(get(cfg, "track1.max_zip_mb", MAX_ZIP_BYTES / 10**6)) * 10**6))
    return {"zip": str(zip_path), "images": len(images), "official_resize": official is not None, "log": log, **fit,
            **disclosure(cfg, model), "fusion_stats": fused}


def disclosure(cfg: dict, model) -> dict:
    """What the zip was built with, for job.json and the upload description: file-name metadata
    ("Meta" flag on macvi.org) and per-scene fusion across the Dev versions."""
    from macvi.common.config import get

    uses_meta = bool(getattr(model, "uses_meta", False) and get(cfg, "model.use_meta", True))
    return {"metadata": uses_meta, "fusion": fusion.settings(cfg) is not None}
