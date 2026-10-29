"""EURS data: discovery, scene-wise splits and a torch Dataset.

The released folder layout is configured, not hard-coded. Each sample is a degraded image
paired with its scene's clean image and mask. ``scene_regex`` (or the degraded image's folder)
extracts the scene id so that all degraded versions of a scene land in the same split
(otherwise validation leaks).

``python -m macvi.cli eurs-layout <train root>`` infers the layout (``macvi.eurs.layout``).
"""

from __future__ import annotations

import random
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


@dataclass
class Layout:
    degraded_glob: str = "degraded/*.png"
    clean_glob: str = "clean/*.png"
    mask_glob: str = "masks/*.png"
    scene_regex: str = r"^(?P<scene>[^_.]+)"
    mask_palette: list[list[int]] | None = None  # RGB colour per class id, if masks are colour-coded
    degraded_scene: str = "regex"  # "parent": a degraded image's scene is its folder name (degraded/<scene>/*.png)
    clean_regex: str | None = None  # scene of clean/mask files, if it differs from scene_regex (reef_01 vs reef_01_7)


@dataclass(frozen=True)
class Sample:
    scene: str
    degraded: Path
    clean: Path
    mask: Path


def _scene(path: Path, pattern: re.Pattern) -> str:
    match = pattern.search(path.stem)
    if not match:
        raise ValueError(f"scene_regex does not match {path.name}")
    return match.group("scene")


def list_images(root: str | Path) -> list[Path]:
    """Every image under ``root`` (recursive), e.g. the degraded-only Dev split.

    Submissions are flat folders keyed by file stem, so duplicate stems are an error.
    """
    images = sorted(p for p in Path(root).rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS
                    and not any(part.startswith(".") for part in p.relative_to(root).parts))
    if not images:
        raise FileNotFoundError(f"no images under {root}")
    stems = [p.stem for p in images]
    if len(set(stems)) != len(stems):
        dup = sorted({s for s in stems if stems.count(s) > 1})[:3]
        raise ValueError(f"duplicate image stems under {root}, e.g. {dup}")
    return images


def discover(root: str | Path, layout: Layout = Layout(), cache_dir: str | Path | None = None) -> list[Sample]:
    """Pair every degraded image with its scene's clean image and mask.

    Colour-coded masks (``layout.mask_palette``) are converted once to class-index PNGs in
    ``cache_dir`` (default ``<root>/.macvi_masks``) so the rest of the pipeline sees ids 0..C-1.
    """
    root, pattern = Path(root), re.compile(layout.scene_regex)
    per_scene = re.compile(layout.clean_regex) if layout.clean_regex else pattern
    clean = {_scene(p, per_scene): p for p in sorted(root.glob(layout.clean_glob))}
    masks = {_scene(p, per_scene): p for p in sorted(root.glob(layout.mask_glob))}
    if layout.mask_palette:
        cache = Path(cache_dir or root / ".macvi_masks")
        cache.mkdir(parents=True, exist_ok=True)
        for scene, path in list(masks.items()):
            out = cache / f"{scene}.png"
            if not out.exists():
                cv2.imwrite(str(out), palette_to_index(read_rgb(path), layout.mask_palette))
            masks[scene] = out
    samples = []
    for p in sorted(root.glob(layout.degraded_glob)):
        s = p.parent.name if layout.degraded_scene == "parent" else _scene(p, pattern)
        if s in clean and s in masks:
            samples.append(Sample(s, p, clean[s], masks[s]))
    if not samples:
        raise FileNotFoundError(f"no complete samples under {root} with {layout}")
    return samples


def split_by_scene(samples: list[Sample], val_fraction: float = 0.2, seed: int = 0,
                   fold: int = 0) -> tuple[list[Sample], list[Sample]]:
    """Hold out whole scenes. ``fold`` rotates which block of scenes is held out (k-fold CV).

    RSUIGM-style data has few scenes (the 2025 release: 30 scenes x 200 degradations), so a
    random image split would leak and a single small fold is noisy; compare configs on the
    same fold and use other folds to confirm a winner.
    """
    by_scene: dict[str, list[Sample]] = defaultdict(list)
    for s in samples:
        by_scene[s.scene].append(s)
    scenes = sorted(by_scene)
    random.Random(seed).shuffle(scenes)
    if val_fraction <= 0:  # final fit on all data (e.g. a Track 1 submission)
        return list(samples), []
    n_val = max(1, round(len(scenes) * val_fraction))
    start = (fold * n_val) % len(scenes)
    val_scenes = set((scenes + scenes)[start:start + n_val])
    train = [s for s in samples if s.scene not in val_scenes]
    val = [s for s in samples if s.scene in val_scenes]
    return train, val


def cap_per_scene(samples: list[Sample], k: int | None, seed: int = 0) -> list[Sample]:
    """Deterministic subset of at most ``k`` degraded versions per scene (fast per-epoch validation)."""
    if not k:
        return samples
    by_scene: dict[str, list[Sample]] = defaultdict(list)
    for s in samples:
        by_scene[s.scene].append(s)
    rng = random.Random(seed)
    out = []
    for scene in sorted(by_scene):
        group = sorted(by_scene[scene], key=lambda s: s.degraded.name)
        out += group if len(group) <= k else rng.sample(group, k)
    return out


def scene_classes(samples: list[Sample], num_classes: int, min_fraction: float = 1e-3) -> dict[str, set[int]]:
    """Classes covering at least ``min_fraction`` of each scene's mask (specks do not count)."""
    out = {}
    for s in {s.scene: s for s in samples}.values():
        counts = np.bincount(read_mask(s.mask).ravel(), minlength=256)[:num_classes]
        out[s.scene] = {c for c in range(num_classes) if counts[c] >= min_fraction * max(counts.sum(), 1)}
    return out


def scene_balanced_weights(samples: list[Sample], class_boost: list[float] | None = None,
                           presence: dict[str, set[int]] | None = None) -> list[float]:
    """Sampling weights giving every scene the same total probability, whatever its version count.

    ``class_boost`` (one multiplier per class) oversamples scenes holding rare classes: a scene's
    weight is the largest boost among the classes present in it (``scene_classes``), at least 1.
    In the release elasmobranch is in 2 of 20 scenes and "other" in 3, and each Qseg class counts
    equally, so those few scenes decide two of five per-class scores.
    """
    counts: dict[str, int] = defaultdict(int)
    for s in samples:
        counts[s.scene] += 1
    scene_w = {scene: 1.0 for scene in counts}
    if class_boost:
        presence = presence if presence is not None else scene_classes(samples, len(class_boost))
        scene_w = {scene: max([1.0, *(class_boost[c] for c in presence[scene])]) for scene in counts}
    return [scene_w[s.scene] / counts[s.scene] for s in samples]


def water_boost_weights(samples: list[Sample], weights: list[float], boost: list[float]) -> list[float]:
    """Shift each scene's draws toward the water types in ``boost`` (one multiplier per type 1..10).

    A sample's weight is multiplied by its water type's boost (1 when the file name carries none) and
    every scene is rescaled to its old total, so scene balance (and ``class_boost``) is unchanged."""
    from macvi.eurs import meta
    if len(boost) != meta.WATER_TYPES:
        raise ValueError(f"train.water_boost needs {meta.WATER_TYPES} values, got {len(boost)}")
    new = []
    for s, w in zip(samples, weights):
        m = meta.parse(s.degraded)
        new.append(w * (boost[m.water - 1] if m else 1.0))
    old_tot, new_tot = defaultdict(float), defaultdict(float)
    for s, w, n in zip(samples, weights, new):
        old_tot[s.scene] += w
        new_tot[s.scene] += n
    return [n * old_tot[s.scene] / new_tot[s.scene] if new_tot[s.scene] else w
            for s, w, n in zip(samples, weights, new)]


def pixel_class_weights(samples: list[Sample], num_classes: int) -> list[float]:
    """Inverse-sqrt class frequency over the distinct training masks, normalised to mean 1."""
    pixels = np.zeros(num_classes, np.float64)
    for path in sorted({s.mask for s in samples}):
        pixels += np.bincount(read_mask(path).ravel(), minlength=256)[:num_classes]
    w = 1.0 / np.sqrt(np.maximum(pixels / max(pixels.sum(), 1.0), 1e-6))
    return (w / w.mean()).tolist()


def read_rgb(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def palette_to_index(rgb: np.ndarray, palette: list[list[int]]) -> np.ndarray:
    out = np.full(rgb.shape[:2], 255, np.uint8)
    for cls, colour in enumerate(palette):
        out[np.all(rgb == np.asarray(colour, np.uint8), axis=-1)] = cls
    if (out == 255).any():
        bad = {tuple(c) for c in rgb[out == 255][:5].tolist()}
        raise ValueError(f"mask colours not in palette, e.g. {bad}")
    return out


def read_mask(path: Path) -> np.ndarray:
    """Class-index mask. Grey masks stored as 3 identical channels are accepted; true colour
    masks must be converted with ``Layout.mask_palette`` (silently taking one channel would
    corrupt the labels)."""
    mask = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if mask is None:
        raise FileNotFoundError(path)
    if mask.ndim == 3:
        if not (np.array_equal(mask[..., 0], mask[..., 1]) and np.array_equal(mask[..., 1], mask[..., 2])):
            raise ValueError(f"{path.name} is a colour mask; set data.layout.mask_palette")
        mask = mask[..., 0]
    return mask


def worker_init(_: int) -> None:
    """DataLoader worker setup: one OpenCV thread per worker (4 vCPUs are shared by all workers)."""
    cv2.setNumThreads(1)


class EURSDataset(Dataset):
    """Returns uint8 (degraded, clean) CHW tensors and a uint8 mask at ``size``, augmented (see
    ``augment``) when training; the training step moves them to the GPU and normalises there
    (4x less inter-process traffic than float32).

    The clean image and mask are shared by every degraded version of a scene, so each worker
    decodes them once per scene. With ``kd_cache`` (``macvi.eurs.distill.LogitCache``) a fourth
    item is the cached teacher logits of the sample's ``kd_source`` image ("degraded" or "clean"),
    warped by the same crop/zoom/flip to ``kd_size``.
    """

    def __init__(self, samples: list[Sample], size: tuple[int, int], train: bool = False, aug=None,
                 kd_cache=None, kd_source: str = "degraded", kd_size: tuple[int, int] | None = None):
        from macvi.eurs.augment import AugConfig

        self.samples, self.size, self.train = samples, size, train
        self.aug = aug or AugConfig()
        self.kd_cache, self.kd_source = kd_cache, kd_source
        self.kd_size = kd_size or (max(1, -(-size[0] // 4)), max(1, -(-size[1] // 4)))
        self._shared: dict[str, np.ndarray] = {}

    def __len__(self) -> int:
        return len(self.samples)

    def _scene_arrays(self, s: Sample) -> tuple[np.ndarray, np.ndarray]:
        key = str(s.clean)
        if key not in self._shared:
            self._shared[key] = read_rgb(s.clean)
            self._shared[key + "\0mask"] = read_mask(s.mask)
        return self._shared[key], self._shared[key + "\0mask"]

    def __getitem__(self, i: int):
        from macvi.eurs import augment

        s = self.samples[i]
        h, w = self.size
        deg = read_rgb(s.degraded)
        clean, mask = self._scene_arrays(s)
        src_hw = deg.shape[:2]
        teacher = None
        if self.train:
            rng = np.random.default_rng(int(torch.randint(0, 2**31 - 1, ()).item()))
            deg, clean, mask, geometry = augment.apply(deg, clean, mask, self.size, self.aug, rng, return_geometry=True)
            if self.kd_cache is not None:
                cached = self.kd_cache.load(s.clean if self.kd_source == "clean" else s.degraded)
                teacher = augment.warp_logits(cached, geometry, src_hw, self.kd_size)
        else:
            deg = cv2.resize(deg, (w, h), interpolation=cv2.INTER_AREA)
            clean = cv2.resize(clean, (w, h), interpolation=cv2.INTER_AREA)
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        to_t = lambda a: torch.from_numpy(np.ascontiguousarray(a)).permute(2, 0, 1)  # noqa: E731
        out = (to_t(deg), to_t(clean), torch.from_numpy(np.ascontiguousarray(mask)))
        return out if teacher is None else (*out, torch.from_numpy(teacher.astype(np.float16)))
