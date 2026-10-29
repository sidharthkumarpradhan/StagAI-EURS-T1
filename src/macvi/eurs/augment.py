"""Training augmentation aimed at Track 2's unseen test scenes.

- Joint geometric augmentation applied identically to the degraded input, clean target, mask and
  (when distilling from a cache) the teacher's logits. Two crop modes:
  - ``crop_scale`` (default): a random crop keeping a fraction of each side, resized to ``size``.
  - ``scale_range`` (mmseg-style random resize): the whole image is mapped to ``size`` (exactly
    what the static Track 2 graph sees at test time) and zoomed by a ratio r drawn from the range,
    then cropped back to ``size``. r > 1 zooms in (random crop), r < 1 zooms out; the uncovered
    border is filled by reflection for every array, so image, target, mask and logits stay aligned
    and no ignore label is needed.
- Horizontal flip (no vertical flip: gravity and lighting cues).
- Re-degradation: with probability ``p``, replace the degraded input by a fresh rendering of
  the clean image through the underwater image-formation model I = J*t + B*(1-t)
  (``macvi.eurs.synthetic.degrade``). It runs after the crop and resize (the model is per-pixel
  and its depth field smooth, so this only saves CPU). The target stays the clean image, so the
  model sees many more water types and depths than the release contains.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from macvi.eurs.synthetic import degrade


@dataclass
class AugConfig:
    crop_scale: tuple[float, float] = (0.6, 1.0)  # fraction of the image side kept by the crop
    hflip: bool = True
    redegrade_p: float = 0.0
    scale_range: tuple[float, float] | None = None  # e.g. (0.5, 2.0): random-scale crops instead of crop_scale


@dataclass(frozen=True)
class Geometry:
    """Source window (pixels of the stored image; may extend past its border) and mirror flag."""

    x0: int
    y0: int
    w: int
    h: int
    flip: bool


def sample_geometry(h: int, w: int, cfg: AugConfig, rng: np.random.Generator) -> Geometry:
    if cfg.scale_range is not None:
        r = rng.uniform(*cfg.scale_range)
        bh, bw = max(8, round(h / r)), max(8, round(w / r))
    elif cfg.crop_scale[0] < 1.0:
        s = rng.uniform(*cfg.crop_scale)
        bh, bw = max(8, int(h * s)), max(8, int(w * s))
    else:
        bh, bw = h, w
    # inside the image when the window fits, otherwise a random overhang on both sides
    y0 = int(rng.integers(min(0, h - bh), max(0, h - bh) + 1))
    x0 = int(rng.integers(min(0, w - bw), max(0, w - bw) + 1))
    return Geometry(x0, y0, bw, bh, bool(cfg.hflip and rng.random() < 0.5))


def warp(img: np.ndarray, g: Geometry, size: tuple[int, int], nearest: bool = False) -> np.ndarray:
    """Crop ``g``'s window (reflect-padding any overhang), mirror, resize to ``size`` = (H, W)."""
    h, w = img.shape[:2]
    top, left = max(0, -g.y0), max(0, -g.x0)
    bottom, right = max(0, g.y0 + g.h - h), max(0, g.x0 + g.w - w)
    if top or left or bottom or right:
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_REFLECT_101)
    crop = img[g.y0 + top:g.y0 + top + g.h, g.x0 + left:g.x0 + left + g.w]
    if g.flip:
        crop = crop[:, ::-1]
    oh, ow = size
    interp = cv2.INTER_NEAREST if nearest else (cv2.INTER_AREA if g.w >= ow and g.h >= oh else cv2.INTER_LINEAR)
    return cv2.resize(np.ascontiguousarray(crop), (ow, oh), interpolation=interp)


def warp_logits(logits: np.ndarray, g: Geometry, src_hw: tuple[int, int], size: tuple[int, int]) -> np.ndarray:
    """Apply ``g`` to a low-resolution map (C, hc, wc) covering a (H, W) image; output (C, *size).

    Sub-pixel exact: each output pixel samples the map at the centre of the image region it
    covers (bilinear, reflected past the border), so teacher logits cached once per image line up
    with any crop, zoom and flip of that image.
    """
    (H, W), (oh, ow) = src_hw, size
    hc, wc = logits.shape[-2:]
    ax, ay = g.w / ow * wc / W, g.h / oh * hc / H
    bx = (g.x0 + 0.5 * g.w / ow) * wc / W - 0.5
    by = (g.y0 + 0.5 * g.h / oh) * hc / H - 0.5
    if g.flip:
        ax, bx = -ax, bx + ax * (ow - 1)
    m = np.array([[ax, 0.0, bx], [0.0, ay, by]], np.float64)
    flags = cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP
    return np.stack([cv2.warpAffine(np.ascontiguousarray(c, np.float32), m, (ow, oh), flags=flags,
                                    borderMode=cv2.BORDER_REFLECT_101) for c in logits])


def apply(deg: np.ndarray, clean: np.ndarray, mask: np.ndarray, size: tuple[int, int], cfg: AugConfig,
          rng: np.random.Generator, return_geometry: bool = False):
    """Return augmented (degraded, clean, mask) resized to ``size`` = (H, W) [, Geometry]."""
    g = sample_geometry(*deg.shape[:2], cfg, rng)
    clean_out = warp(clean, g, size)
    deg_out = degrade(clean_out, rng) if rng.random() < cfg.redegrade_p else warp(deg, g, size)
    out = (deg_out, clean_out, warp(mask, g, size, nearest=True))
    return (*out, g) if return_geometry else out
