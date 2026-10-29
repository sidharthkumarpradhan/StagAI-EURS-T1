"""Audit a real EURS download before training: pairing, splits, sizes and label sanity.

Fails loudly on what would silently ruin a run: unpaired images, masks with ids outside
0..C-1, mask/clean size mismatches, or a missing class (Qseg multiplies per-class scores).
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from macvi.eurs.data import Layout, Sample, discover, read_mask

CLASS_NAMES = ("background", "coral", "fish", "elasmobranch", "other")


def audit(root: str | Path, layout: Layout = Layout(), num_classes: int = 5, max_images: int | None = None) -> dict:
    root = Path(root)
    samples = discover(root, layout)
    n_degraded = len(list(root.glob(layout.degraded_glob)))
    scenes = Counter(s.scene for s in samples)
    clean_masks: dict[str, Sample] = {s.scene: s for s in samples}
    pixels = np.zeros(num_classes, np.int64)
    problems, sizes, invalid = [], Counter(), 0
    for i, s in enumerate(clean_masks.values()):
        if max_images and i >= max_images:
            break
        mask = read_mask(s.mask)
        clean = cv2.imread(str(s.clean), cv2.IMREAD_UNCHANGED)
        if clean.shape[:2] != mask.shape[:2]:
            problems.append(f"{s.scene}: clean {clean.shape[:2]} vs mask {mask.shape[:2]}")
        values = np.bincount(mask.ravel(), minlength=256)
        invalid += int(values[num_classes:].sum())
        pixels += values[:num_classes]
        sizes[clean.shape[:2]] += 1
    if n_degraded > len(samples):
        problems.append(f"{n_degraded - len(samples)} degraded images have no clean/mask partner")
    if invalid:
        problems.append(f"{invalid} mask pixels have ids >= {num_classes}")
    missing = [CLASS_NAMES[c] for c in range(num_classes) if pixels[c] == 0]
    if missing:
        problems.append(f"classes never labelled: {missing}")
    freq = pixels / max(pixels.sum(), 1)
    return {
        "samples": len(samples), "scenes": len(scenes),
        "versions_per_scene": [min(scenes.values()), max(scenes.values())],
        "image_sizes": {f"{h}x{w}": n for (h, w), n in sizes.most_common(5)},
        "class_pixel_fraction": {CLASS_NAMES[c]: round(float(freq[c]), 5) for c in range(num_classes)},
        # inverse-sqrt frequency weights: a common, stable choice for imbalanced segmentation
        "suggested_class_weights": [round(float(w), 3) for w in (1 / np.sqrt(np.maximum(freq, 1e-6))) /
                                    (1 / np.sqrt(np.maximum(freq, 1e-6))).mean()],
        "problems": problems,
    }
