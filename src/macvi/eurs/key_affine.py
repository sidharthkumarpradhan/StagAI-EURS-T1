"""Track 1 restoration from file-name metadata: one colour affine per (water type, depth), fused per scene.

Measured 6 Oct 2026 on the 4 held-out train scenes of f053fbd99f50 (tools/workbench/key_affine_v100.py,
fuse_eval_v100.py): the degraded -> clean map of an EURS image is close to one global 3x4 colour affine
(oracle 26.9 dB), and that affine depends on the (water, depth) in the name far more than on the scene.
Fitted on train scenes and picked by name alone it restores a single image to 25.0 dB (the teacher's
restorer: 16.2 dB); water types 1-3 reach ~34 dB. Every version of a scene shares one clean image, so
the scene is restored once from its ``top_k`` best keys (lowest train MSE) and written for every version:
top-10 + unsharp 0.5 gave PSNR 34.1 / SSIM 0.957 / sharpness 0.874 (aimslabs_t1_13: 27.2 / 0.884 / 0.862).

Track 1 only (it needs the whole scene and the names); disclosed like ``fusion``.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable

import cv2
import numpy as np

from macvi.eurs import meta

Key = tuple[int, int]


def key_of(name) -> Key | None:
    m = meta.parse(name)
    return None if m is None else (m.water, m.depth)


def _design(img: np.ndarray) -> np.ndarray:
    x = img.reshape(-1, 3).astype(np.float64) / 255
    return np.concatenate([x, np.ones((len(x), 1))], 1)


class Fitter:
    """Accumulates per-key normal equations from (degraded, clean) uint8 pairs; ``solve`` gives 4x3 affines."""

    def __init__(self, pixels: int | None = 20000, seed: int = 0):
        self.pixels, self.rng = pixels, np.random.default_rng(seed)
        self.xx: dict[Key, np.ndarray] = defaultdict(lambda: np.zeros((4, 4)))
        self.xy: dict[Key, np.ndarray] = defaultdict(lambda: np.zeros((4, 3)))

    def add(self, key: Key, degraded: np.ndarray, clean: np.ndarray) -> None:
        x, y = _design(degraded), clean.reshape(-1, 3).astype(np.float64) / 255
        if self.pixels and len(x) > self.pixels:
            idx = self.rng.choice(len(x), self.pixels, replace=False)
            x, y = x[idx], y[idx]
        self.xx[key] += x.T @ x
        self.xy[key] += x.T @ y

    def solve(self, ridge: float = 1e-6) -> dict[Key, np.ndarray]:
        return {k: np.linalg.solve(self.xx[k] + ridge * np.eye(4), self.xy[k]) for k in self.xx}


def restore(img: np.ndarray, affine: np.ndarray) -> np.ndarray:
    """Apply a 4x3 affine to a uint8 RGB image; float32 in [0, 1]."""
    x = img.reshape(-1, 3).astype(np.float32) / 255
    a = affine.astype(np.float32)
    return np.clip(x @ a[:3] + a[3], 0, 1).reshape(img.shape)


def load_table(path) -> dict[Key, np.ndarray]:
    """Per-key affines as saved by tools/workbench/key_affine_v100.py: ``{"A": {(water, depth): 4x3}}``."""
    return np.load(path, allow_pickle=True).item()["A"]


def map_input(img: np.ndarray, key: Key | None, table: dict[Key, np.ndarray]) -> np.ndarray:
    """The keyed colour map of one uint8 image (uint8), the segmenter input of the Track 1 judge and packager;
    the image itself when its key is unknown or not in ``table`` (clean images, re-renderings)."""
    if key is None or key not in table:
        return img
    return (restore(img, table[key]) * 255 + 0.5).astype(np.uint8)


def best_keys(errors: dict[Key, float], top_k: int) -> list[Key]:
    """The ``top_k`` keys with the lowest (train) restoration error."""
    return sorted(errors, key=errors.get)[:top_k]


def unsharp(img: np.ndarray, amount: float, sigma: float = 1.0) -> np.ndarray:
    if amount == 0:
        return img
    return np.clip(img + amount * (img - cv2.GaussianBlur(img, (0, 0), sigma)), 0, 1)


def fuse_scene(versions: Iterable[tuple[Key, np.ndarray]], affines: dict[Key, np.ndarray], keys: Iterable[Key],
               amount: float = 0.5) -> np.ndarray:
    """One restored uint8 image for a scene: mean of the affine-restored versions whose key is in ``keys``."""
    keep, acc, n = set(keys), None, 0
    for k, img in versions:
        if k in keep and k in affines:
            r = restore(img, affines[k])
            acc = r if acc is None else acc + r
            n += 1
    if n == 0:
        raise ValueError("no version of this scene has a selected key")
    return (unsharp(acc / n, amount) * 255 + 0.5).astype(np.uint8)
