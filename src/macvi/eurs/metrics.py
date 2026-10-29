"""Local EURS metrics on numpy arrays.

Exact official implementations are not published, so these follow the standard definitions:
- mIoU / F1: per class over the whole set (accumulated confusion matrix), macro-averaged over
  classes present in ground truth or prediction.
- PSNR on 8-bit RGB; SSIM with the Wang et al. (2004) Gaussian window on luminance.
- sharpness ratio: variance of the Laplacian (restored / clean), clipped to [0, 1].
- UCIQE (Yang & Sowmya, 2015) is computed. UIQM and CCF are not reimplemented: their combined
  weight is ~0.015 composite points per unit, so the scorer uses leaderboard medians for them
  (``NR_DEFAULTS``) and reports that this is an approximation.
"""

from __future__ import annotations

import cv2
import numpy as np

NR_DEFAULTS = {"uiqm": 0.775, "ccf": 0.040}  # medians of the top-10 Track 1 rows, 2026-10-01


def confusion(pred: np.ndarray, gt: np.ndarray, num_classes: int) -> np.ndarray:
    valid = (gt >= 0) & (gt < num_classes)
    idx = num_classes * gt[valid].astype(np.int64) + pred[valid].astype(np.int64)
    return np.bincount(idx, minlength=num_classes**2).reshape(num_classes, num_classes)


# Official coverage penalty ("the same per-class IoU floor as Track 1 -- drives Qseg to exactly 0 if any present
# class is essentially ignored", challenge page, read 6 Oct 2026). The floor's value is not published; our first
# Track 1 entry (0a6952a5c0fb: mIoU 0.56, F1 0.64, class 4 never predicted on starfish1) scored Qseg 0.0.
IOU_FLOOR = 0.05


def seg_scores(conf: np.ndarray, iou_floor: float = IOU_FLOOR) -> dict[str, float]:
    """Macro F1, mIoU and Qseg = F1 * mIoU * coverage, coverage 0 when a class present in the ground truth
    has IoU under ``iou_floor`` (its count in ``ignored``)."""
    tp = np.diag(conf).astype(float)
    fp, fn = conf.sum(0) - tp, conf.sum(1) - tp
    present = (tp + fp + fn) > 0
    iou = tp / np.maximum(tp + fp + fn, 1)
    f1 = 2 * tp / np.maximum(2 * tp + fp + fn, 1)
    miou, mf1 = float(iou[present].mean()), float(f1[present].mean())
    ignored = int(((conf.sum(1) > 0) & (iou < iou_floor)).sum())
    coverage = 0.0 if ignored else 1.0
    return {"miou": miou, "f1": mf1, "q_seg": miou * mf1 * coverage, "q_seg_raw": miou * mf1, "coverage": coverage, "ignored": ignored,
            **{f"iou_{i}": float(v) for i, v in enumerate(iou)}}


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return 100.0 if mse == 0 else float(10 * np.log10(255.0**2 / mse))


def ssim(a: np.ndarray, b: np.ndarray) -> float:
    def lum(x):
        x = x.astype(np.float64)
        return x if x.ndim == 2 else 0.299 * x[..., 0] + 0.587 * x[..., 1] + 0.114 * x[..., 2]

    x, y = lum(a), lum(b)
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    blur = lambda z: cv2.GaussianBlur(z, (11, 11), 1.5)  # noqa: E731
    mx, my = blur(x), blur(y)
    vx, vy, cxy = blur(x * x) - mx**2, blur(y * y) - my**2, blur(x * y) - mx * my
    s = ((2 * mx * my + c1) * (2 * cxy + c2)) / ((mx**2 + my**2 + c1) * (vx + vy + c2))
    return float(s.mean())


def sharpness_ratio(restored: np.ndarray, clean: np.ndarray) -> float:
    lap = lambda z: cv2.Laplacian(cv2.cvtColor(z, cv2.COLOR_RGB2GRAY), cv2.CV_64F).var()  # noqa: E731
    return float(np.clip(lap(restored) / max(lap(clean), 1e-9), 0.0, 1.0))


def uciqe(rgb: np.ndarray) -> float:
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float64)
    l, a, b = lab[..., 0] / 255.0, (lab[..., 1] - 128) / 255.0, (lab[..., 2] - 128) / 255.0
    chroma = np.sqrt(a**2 + b**2)
    lo, hi = np.percentile(l, [1, 99])
    sat = chroma / np.maximum(np.sqrt(chroma**2 + l**2), 1e-9)
    return float(0.4680 * chroma.std() + 0.2745 * (hi - lo) + 0.2576 * sat.mean())


class Accumulator:
    """Accumulates per-image results into dataset-level metrics."""

    def __init__(self, num_classes: int = 5):
        self.conf = np.zeros((num_classes, num_classes), dtype=np.int64)
        self.num_classes = num_classes
        self.rows: list[dict[str, float]] = []

    def add(self, restored: np.ndarray, clean: np.ndarray, pred: np.ndarray, gt: np.ndarray, repeat: int = 1) -> None:
        """Add one prediction; ``repeat`` counts it for that many images (e.g. one fused output
        written for every version of a scene) without recomputing identical metrics."""
        self.conf += confusion(pred, gt, self.num_classes) * repeat
        row = {"psnr": psnr(restored, clean), "ssim": ssim(restored, clean),
               "sharpness": sharpness_ratio(restored, clean), "uciqe": uciqe(restored)}
        self.rows.extend([row] * repeat)

    def result(self) -> dict[str, float]:
        out = seg_scores(self.conf)
        for key in ("psnr", "ssim", "sharpness", "uciqe"):
            out[key] = float(np.mean([r[key] for r in self.rows]))
        return {**out, **NR_DEFAULTS}
