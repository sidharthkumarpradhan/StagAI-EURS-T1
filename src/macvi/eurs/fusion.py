"""Track 1 per-scene fusion: one prediction per scene from all its pixel-aligned versions.

Every EURS scene is ONE clean image and ONE mask rendered ~200 times through the image-formation
model (10 water types x 20 depths, names ``<scene>_<water>_<depth>``; measured 3 Oct 2026 for
train and the 6 Dev scenes). Predicting each version alone throws that away: the darkest versions
(~17% have SSIM < 0.1 against their clean image) carry almost no signal, the clearest ones nearly
all of it. Fusion, Track 1 only and behind ``track1.fusion`` (owner-approved trial, disclosed in
the run metadata and in docs/research.md; it does not carry over to Track 2's single-image ONNX):

- segmentation: average the softmax of every version (optionally confidence-weighted) and write
  the argmax for every version;
- restoration: a robust weighted mean of the restored versions. Weights come from agreement with
  the consensus (iteratively reweighted: a version far from the others is a bad one), computed on
  a small copy, optionally keeping only the ``top_k`` best versions. Pixel alignment keeps edges.

Images whose name carries no scene (``meta.scene_of`` is None) are predicted alone.
``evaluate`` scores held-out train scenes with and without fusion from one prediction pass.
"""

from __future__ import annotations

import shutil
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np

from macvi.eurs import meta as meta_mod
from macvi.eurs.data import Sample, read_rgb
from macvi.eurs.evaluate import _finish, load_scored, predict_probs, scoring_size
from macvi.eurs.metrics import Accumulator


@dataclass
class FusionConfig:
    restore: str = "consistency"  # "consistency" (agreement-weighted) or "mean"
    top_k: int | None = None      # restoration: keep only the k highest-weight versions
    iterations: int = 3           # reweighting rounds for "consistency"
    seg: str = "mean"             # "mean" softmax, or "confidence" (weighted by mean max-probability)
    small_side: int = 256         # longest side of the copy the weights are computed on


def settings(cfg: dict[str, Any]) -> FusionConfig | None:
    """The run's fusion settings, or None when ``track1.fusion`` is off (the default)."""
    t1 = cfg.get("track1") or {}
    return FusionConfig(**(t1.get("fusion_options") or {})) if t1.get("fusion") else None


def scene_groups(paths: Iterable[Path]) -> list[list[Path]]:
    """Images grouped by the scene prefix of their names; unparseable names stay alone."""
    groups: dict[str, list[Path]] = defaultdict(list)
    for p in paths:
        scene = meta_mod.scene_of(p)
        groups[f"scene:{scene}" if scene is not None else f"single:{Path(p).stem}"].append(Path(p))
    return [groups[k] for k in sorted(groups)]


def _small(img: np.ndarray, side: int) -> np.ndarray:
    h, w = img.shape[:2]
    s = min(1.0, side / max(h, w))
    out = cv2.resize(img, (max(1, round(w * s)), max(1, round(h * s))), interpolation=cv2.INTER_AREA) if s < 1 else img
    return out.astype(np.float32) / 255.0


def restoration_weights(images: list[np.ndarray], cfg: FusionConfig) -> np.ndarray:
    """Normalised weight per restored version (uint8 HxWx3 each)."""
    n = len(images)
    if cfg.restore == "mean" or n < 3:
        w = np.ones(n, np.float64)
    elif cfg.restore == "consistency":
        small = np.stack([_small(im, cfg.small_side) for im in images])
        consensus = np.median(small, axis=0)
        for _ in range(max(1, cfg.iterations)):
            err = ((small - consensus) ** 2).mean(axis=(1, 2, 3)).astype(np.float64)
            w = 1.0 / (err + 1e-5)
            consensus = np.tensordot(w / w.sum(), small, axes=1)
    else:
        raise ValueError(f"unknown fusion restore mode {cfg.restore!r}")
    if cfg.top_k and cfg.top_k < n:
        w[np.argsort(-w)[cfg.top_k:]] = 0.0
    return w / w.sum()


class SceneFuser:
    """Accumulates the predictions of one scene's versions, then fuses them."""

    def __init__(self, cfg: FusionConfig):
        self.cfg, self.restored, self.prob_sum, self.shape = cfg, [], None, None

    def add(self, restored: np.ndarray, probs: np.ndarray) -> None:
        if self.shape is None:
            self.shape = restored.shape
        elif restored.shape != self.shape:
            raise ValueError(f"versions of one scene differ in size: {restored.shape} vs {self.shape}")
        self.restored.append(restored)
        w = float(probs.max(0).mean()) if self.cfg.seg == "confidence" else 1.0
        self.prob_sum = probs * w if self.prob_sum is None else self.prob_sum + probs * w

    def result(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(fused restored uint8 HxWx3, fused mask uint8 HxW, restoration weights)."""
        weights = restoration_weights(self.restored, self.cfg)
        acc = np.zeros(self.shape, np.float32)
        for w, im in zip(weights, self.restored):
            if w > 0:
                acc += np.float32(w) * im.astype(np.float32)
        fused = np.clip(acc + 0.5, 0, 255).astype(np.uint8)
        return fused, self.prob_sum.argmax(0).astype(np.uint8), weights


def evaluate(model, samples: Iterable[Sample], size: tuple[int, int], cfg: FusionConfig, device: str = "cpu",
             max_side: int = 1024, tta: bool = False) -> tuple[dict[str, float], dict[str, float]]:
    """(single-image metrics, fused metrics) on labelled samples, from one prediction pass."""
    single, fused = evaluate_variants(model, samples, size, {"fused": cfg}, device, max_side, tta)
    return single, fused["fused"]


def evaluate_variants(model, samples: Iterable[Sample], size: tuple[int, int], variants: dict[str, FusionConfig],
                      device: str = "cpu", max_side: int = 1024,
                      tta: bool = False) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
    """Single-image metrics and the metrics of every fusion variant, from one prediction pass.

    Scenes come from ``Sample.scene`` (the layout's scene key: every version shares its clean
    image and mask); each fused output counts once per version, as in the Dev zip."""
    by_scene: dict[str, list[Sample]] = defaultdict(list)
    for s in samples:
        by_scene[s.scene].append(s)
    single = Accumulator(model.cfg.num_classes)
    fused = {name: Accumulator(model.cfg.num_classes) for name in variants}
    for scene in sorted(by_scene):
        fusers, clean, mask = {name: SceneFuser(c) for name, c in variants.items()}, None, None
        for s in by_scene[scene]:
            deg, clean, mask = load_scored(s, max_side)
            restored, probs = predict_probs(model, deg, size, device, tta, path=s.degraded)
            single.add(restored, clean, probs.argmax(0).astype(np.uint8), mask)
            for fuser in fusers.values():
                fuser.add(restored, probs)
        for name, fuser in fusers.items():
            restored, pred, _ = fuser.result()
            fused[name].add(restored, clean, pred, mask, repeat=len(by_scene[scene]))
    return _finish(single), {name: _finish(acc) for name, acc in fused.items()}


def predict_dev(model, images: Iterable[Path], size: tuple[int, int], out_dir: Path, cfg: FusionConfig,
                device: str = "cpu", tta: bool = False) -> dict[str, Any]:
    """Fused twin of ``track1.predict_dev``: restored/ and seg_masks/ for every image at the scoring
    size, one fused output per scene. Returns counts for the run metadata."""
    if out_dir.exists():
        shutil.rmtree(out_dir)
    (out_dir / "restored").mkdir(parents=True, exist_ok=True)
    (out_dir / "seg_masks").mkdir(parents=True, exist_ok=True)
    groups = scene_groups(images)
    for group in groups:
        fuser = SceneFuser(cfg)
        for path in group:
            img = read_rgb(path)
            h, w = scoring_size(*img.shape[:2])
            img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
            fuser.add(*predict_probs(model, img, size, device, tta, path=path))
        restored, mask, _ = fuser.result()
        bgr = cv2.cvtColor(restored, cv2.COLOR_RGB2BGR)
        for path in group:
            cv2.imwrite(str(out_dir / "restored" / f"{path.stem}.png"), bgr)
            cv2.imwrite(str(out_dir / "seg_masks" / f"{path.stem}.png"), mask)
    fused = [g for g in groups if len(g) > 1]
    return {"fused_scenes": len(fused), "fused_images": sum(map(len, fused)),
            "single_images": sum(len(g) for g in groups if len(g) == 1)}
