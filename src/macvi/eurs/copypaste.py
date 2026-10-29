"""Copy-paste of labelled train-scene objects onto external photos (``data.external.copy_paste``).

Track 1 trains on 20 unique scenes, so a teacher can tie each animal to its scene's background. Simple Copy-Paste
(Ghiasi et al., CVPR 2021) pastes masked objects from one labelled image onto another and gains most in the
low-data regime. Here the donor is a train scene's clean image and GT mask (classes 1..4), the target is an
external photo with its own (pseudo) mask, and the composite is pasted *before* the photo is rendered underwater,
so the pasted animals get the same formation model (blur, attenuation, veiling light) as the rest of the frame.
"""

from __future__ import annotations

import cv2
import numpy as np


def paste(img: np.ndarray, mask: np.ndarray, d_img: np.ndarray, d_mask: np.ndarray, rng: np.random.Generator,
          scale: tuple[float, float] = (0.4, 1.0), hflip: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Paste the foreground (mask 1..254) of donor ``d_img``/``d_mask`` onto ``img``/``mask`` (HxWx3 uint8, HxW).

    The donor is resized so its longer side is ``scale`` x the target's longer side, optionally mirrored, and
    placed at a random offset (it may overhang the border). Returns new arrays; the inputs are not modified."""
    h, w = mask.shape
    r = rng.uniform(*scale) * max(h, w) / max(d_mask.shape)
    dh, dw = max(1, round(d_mask.shape[0] * r)), max(1, round(d_mask.shape[1] * r))
    di = cv2.resize(d_img, (dw, dh), interpolation=cv2.INTER_AREA if r < 1 else cv2.INTER_LINEAR)
    dm = cv2.resize(d_mask, (dw, dh), interpolation=cv2.INTER_NEAREST)
    if hflip and rng.random() < 0.5:
        di, dm = di[:, ::-1], dm[:, ::-1]
    y0, x0 = int(rng.integers(-dh // 2, h - dh // 2 + 1)), int(rng.integers(-dw // 2, w - dw // 2 + 1))
    ty0, tx0, ty1, tx1 = max(0, y0), max(0, x0), min(h, y0 + dh), min(w, x0 + dw)
    out_i, out_m = img.copy(), mask.copy()
    if ty1 <= ty0 or tx1 <= tx0:
        return out_i, out_m
    src_m = dm[ty0 - y0:ty1 - y0, tx0 - x0:tx1 - x0]
    fg = (src_m > 0) & (src_m < 255)
    out_i[ty0:ty1, tx0:tx1][fg] = di[ty0 - y0:ty1 - y0, tx0 - x0:tx1 - x0][fg]
    out_m[ty0:ty1, tx0:tx1][fg] = src_m[fg]
    return out_i, out_m
