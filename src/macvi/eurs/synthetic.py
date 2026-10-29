"""Synthetic EURS-like data for offline end-to-end tests (no real data in the sandbox).

Clean scenes contain random blobs of the five classes; degraded versions follow the standard
underwater image formation model I = J * t + B * (1 - t) with per-channel attenuation, so the
restoration task is learnable and the pipeline can be exercised end to end on CPU.

``write_grid_dataset`` mimics the release's structure exactly: pixel-aligned versions of one
clean image per scene on a (water type, depth) grid, named ``<scene>_<water>_<depth>.png``.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

CLASS_COLORS = np.array([[40, 90, 120], [220, 120, 90], [240, 200, 60], [150, 150, 170], [90, 200, 120]], np.uint8)


def make_scene(rng: np.random.Generator, h: int, w: int) -> tuple[np.ndarray, np.ndarray]:
    mask = np.zeros((h, w), np.uint8)
    for cls in rng.permutation(np.arange(1, 5)):
        for _ in range(rng.integers(1, 3)):
            center = (int(rng.integers(0, w)), int(rng.integers(0, h)))
            axes = (int(rng.integers(w // 12, w // 4)), int(rng.integers(h // 12, h // 4)))
            cv2.ellipse(mask, center, axes, float(rng.uniform(0, 180)), 0, 360, int(cls), -1)
    clean = CLASS_COLORS[mask].astype(np.float32)
    clean += rng.normal(0, 12, clean.shape)  # texture
    return np.clip(clean, 0, 255).astype(np.uint8), mask


def _depth_field(rng: np.random.Generator, h: int, w: int) -> np.ndarray:
    """Smooth random scene depth: a vertical gradient (farther towards the top) plus low-frequency noise."""
    ramp = np.linspace(1.0, 0.0, h, dtype=np.float32)[:, None] * np.float32(rng.uniform(0.0, 0.8))
    noise = cv2.resize(rng.random((4, 6)).astype(np.float32), (w, h), interpolation=cv2.INTER_CUBIC) * np.float32(rng.uniform(0.0, 0.5))
    return np.clip(np.float32(rng.uniform(0.3, 1.0)) + ramp + noise, 0.1, 3.0).astype(np.float32)


def degrade(clean: np.ndarray, rng: np.random.Generator, spatial: bool = True) -> np.ndarray:
    """Underwater formation model I = J*t + B*(1-t), t = exp(-beta*d), with per-channel beta
    (red attenuates fastest) and, if ``spatial``, a per-pixel depth map as in RSUIGM-style data.
    Computed in float32 (float64 doubled the cost for no visible difference in 8-bit output)."""
    j = clean.astype(np.float32) / 255.0
    beta = np.array([rng.uniform(1.2, 2.5), rng.uniform(0.4, 0.9), rng.uniform(0.2, 0.6)], np.float32)
    depth = _depth_field(rng, *j.shape[:2])[..., None] if spatial else np.float32(rng.uniform(0.3, 1.0))
    t = np.exp(-beta[None, None, :] * depth)
    backlight = np.array([rng.uniform(0.0, 0.15), rng.uniform(0.35, 0.6), rng.uniform(0.45, 0.75)], np.float32)[None, None, :]
    return np.clip((j * t + backlight * (1 - t)) * 255, 0, 255).astype(np.uint8)


def water_params(water: int) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic (beta, backlight) per water type 1..10: clear blue water (1) to dark brown
    coastal water (10), as in the release's Jerlov grid (red attenuates fastest everywhere)."""
    t = (min(max(water, 1), 10) - 1) / 9.0
    beta = np.array([1.2 + 1.8 * t, 0.3 + 1.4 * t, 0.15 + 1.8 * t], np.float32)
    blue, brown = np.array([0.05, 0.35, 0.55]), np.array([0.30, 0.28, 0.10])
    backlight = ((1 - t) * blue + t * brown) * (1.0 - 0.5 * t)
    return beta, backlight.astype(np.float32)


def degrade_grid(clean: np.ndarray, water: int, depth: int, depth_map: np.ndarray) -> np.ndarray:
    """RSUIGM-like rendering of one grid cell: the scene's own depth map scaled by ``depth`` (5..100)."""
    j = clean.astype(np.float32) / 255.0
    beta, backlight = water_params(water)
    t = np.exp(-beta[None, None, :] * (depth_map[..., None] * depth / 50.0))
    return np.clip((j * t + backlight[None, None, :] * (1 - t)) * 255, 0, 255).astype(np.uint8)


def write_grid_dataset(root: str | Path, scenes: int = 6, waters=(1, 3, 5, 7, 9), depths=(5, 30, 60, 100),
                       size: tuple[int, int] = (64, 96), seed: int = 0, prefix: str = "scene",
                       degraded_only: bool = False) -> Path:
    """Release-shaped data: every scene is ONE clean image + mask with pixel-aligned degraded versions
    named ``<scene>_<water>_<depth>.png`` (one depth map per scene), like train and Dev.

    ``degraded_only`` writes just the degraded images (a Dev-like split) under ``root/degraded``.
    """
    root, rng = Path(root), np.random.default_rng(seed)
    subs = ("degraded",) if degraded_only else ("clean", "degraded", "masks")
    for sub in subs:
        (root / sub).mkdir(parents=True, exist_ok=True)
    for s in range(scenes):
        clean, mask = make_scene(rng, *size)
        name = f"{prefix}{s:03d}"
        depth_map = _depth_field(rng, *size).astype(np.float32)
        if not degraded_only:
            cv2.imwrite(str(root / "clean" / f"{name}.png"), cv2.cvtColor(clean, cv2.COLOR_RGB2BGR))
            cv2.imwrite(str(root / "masks" / f"{name}.png"), mask)
        for w in waters:
            for d in depths:
                deg = degrade_grid(clean, w, d, depth_map)
                cv2.imwrite(str(root / "degraded" / f"{name}_{w}_{d}.png"), cv2.cvtColor(deg, cv2.COLOR_RGB2BGR))
    return root


def write_dataset(root: str | Path, scenes: int = 12, versions: int = 3, size: tuple[int, int] = (96, 160),
                  seed: int = 0) -> Path:
    root, rng = Path(root), np.random.default_rng(seed)
    for sub in ("clean", "degraded", "masks"):
        (root / sub).mkdir(parents=True, exist_ok=True)
    for s in range(scenes):
        clean, mask = make_scene(rng, *size)
        name = f"scene{s:03d}"
        cv2.imwrite(str(root / "clean" / f"{name}.png"), cv2.cvtColor(clean, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(root / "masks" / f"{name}.png"), mask)
        for v in range(versions):
            deg = degrade(clean, rng)
            cv2.imwrite(str(root / "degraded" / f"{name}_v{v}.png"), cv2.cvtColor(deg, cv2.COLOR_RGB2BGR))
    return root
