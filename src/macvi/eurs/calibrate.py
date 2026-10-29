"""Per-class logit bias tuned for pooled Qseg, folded into the final classifier (0 params, 0 FLOPs).

Why: Qseg = F1 * mIoU over the whole test set, macro-averaged over classes, and the coverage
penalty drives it to 0 when a present class is essentially ignored. A tiny student trained on
~16 scenes under-predicts the rare classes (elasmobranch, other); a constant bias per class moves
its decision boundaries without retraining (logit adjustment, Menon et al., ICLR 2021).

How: run the model exactly as it will be scored (``evaluate_static``: static size, logits
nearest-resized to the scoring resolution), keep every ``pixel_stride``-th pixel of held-out
scene images, then coordinate-ascend one bias per class (background anchored at 0) on a grid,
maximising pooled F1 * mIoU subject to a coverage floor: every class present in the validation
ground truth keeps predicted / true pixel area >= ``area_floor``. Before the ascent, classes under
the floor have their bias raised until they clear it: on pooled metrics extra false positives
cost far less than a collapse to 0. Moves are ranked by total coverage shortfall first and Qseg
second, so a model that cannot fully clear the floor is never pushed further below it.

Folding: ``b`` is added to the bias of ``EURSNet.final_classifier()``. Everything after that conv
is additions, bilinear resizes (weights sum to 1) and flip-TTA averaging, so every output logit
moves by exactly ``b``: the argmax changes as tuned and ``seg_logits`` stay raw logits.
"""

from __future__ import annotations

from typing import Any, Iterable

import cv2
import numpy as np
import torch

from macvi.eurs.metrics import confusion, seg_scores


@torch.no_grad()
def collect(model, samples: Iterable, size: tuple[int, int], device: str = "cpu", pixel_stride: int = 4,
            max_side: int = 1024) -> tuple[np.ndarray, np.ndarray]:
    """(logits (N, C) float32, labels (N,)) on a pixel grid of the scored static pipeline."""
    from macvi.eurs.evaluate import load_scored

    model.eval()
    h, w = size
    all_logits, all_labels = [], []
    for s in samples:
        deg, _, mask = load_scored(s, max_side)
        x = torch.from_numpy(cv2.resize(deg, (w, h), interpolation=cv2.INTER_AREA)).permute(2, 0, 1)[None].float().div(255)
        _, logits = model(x.to(device))
        hwc = np.ascontiguousarray(logits[0].float().cpu().numpy().transpose(1, 2, 0))
        hwc = cv2.resize(hwc, (deg.shape[1], deg.shape[0]), interpolation=cv2.INTER_NEAREST)
        all_logits.append(hwc[::pixel_stride, ::pixel_stride].reshape(-1, hwc.shape[-1]))
        all_labels.append(mask[::pixel_stride, ::pixel_stride].ravel())
    return np.concatenate(all_logits).astype(np.float32), np.concatenate(all_labels)


def area_ratios(conf: np.ndarray) -> list[float | None]:
    """Predicted / ground-truth pixel area per class (None where the class is absent from GT)."""
    gt, pred = conf.sum(1), conf.sum(0)
    return [float(p / g) if g > 0 else None for p, g in zip(pred, gt)]


def shortfall(conf: np.ndarray, area_floor: float) -> float:
    """Total coverage deficit: sum over GT-present classes of max(0, floor - predicted/true area)."""
    return float(sum(max(0.0, area_floor - r) for r in area_ratios(conf) if r is not None))


def feasible(conf: np.ndarray, area_floor: float) -> bool:
    return shortfall(conf, area_floor) == 0.0


def _conf(logits: np.ndarray, labels: np.ndarray, bias: np.ndarray) -> np.ndarray:
    return confusion(np.argmax(logits + bias, 1), labels, logits.shape[1])


def _sweep(logits: np.ndarray, labels: np.ndarray, bias: np.ndarray, c: int, grid: np.ndarray):
    """Confusion matrix for every value of ``bias[c]`` on ``grid`` (others fixed), computed once
    from the best competing class instead of a full argmax per value."""
    num = logits.shape[1]
    others = [k for k in range(num) if k != c]
    shifted = logits[:, others] + bias[others]
    best_other = np.asarray(others)[shifted.argmax(1)]
    margin = shifted.max(1) - logits[:, c]
    for v in grid:
        pred = np.where(margin < v, c, best_other)  # class c wins where its logit + v beats all others
        yield v, confusion(pred, labels, num)


def fit_bias(logits: np.ndarray, labels: np.ndarray, area_floor: float = 0.5, max_bias: float = 3.0,
             step: float = 0.1, sweeps: int = 3, min_gain: float = 1e-4) -> dict[str, Any]:
    num = logits.shape[1]
    bias = np.zeros(num, np.float32)
    before = _conf(logits, labels, bias)
    # coverage guard first: lift classes under the area floor until they clear it
    for _ in range(int(round(2 * max_bias / step)) * num):
        ratios = area_ratios(_conf(logits, labels, bias))
        low = [c for c, r in enumerate(ratios) if c > 0 and r is not None and r < area_floor and bias[c] < max_bias]
        if not low:
            break
        for c in low:
            bias[c] = min(max_bias, bias[c] + step)
    conf = _conf(logits, labels, bias)
    best_q, best_short = seg_scores(conf, 0.0)["q_seg"], shortfall(conf, area_floor)
    grid = np.round(np.arange(-max_bias, max_bias + step / 2, step), 6).astype(np.float32)
    for _ in range(sweeps):
        moved = False
        for c in range(1, num):
            best_v = None
            for v, cand in _sweep(logits, labels, bias, c, grid):
                q, short = seg_scores(cand, 0.0)["q_seg"], shortfall(cand, area_floor)
                # rank by (coverage shortfall, Qseg): never trade coverage for Qseg
                if short < best_short - 1e-9 or (abs(short - best_short) <= 1e-9 and q > best_q + min_gain):
                    best_q, best_short, best_v = q, short, v
            if best_v is not None:
                bias[c], moved = best_v, True
        if not moved:
            break
    after = _conf(logits, labels, bias)
    s0, s1 = seg_scores(before, 0.0), seg_scores(after, 0.0)
    return {"bias": [float(b) for b in bias], "q_seg_before": s0["q_seg"], "q_seg_after": s1["q_seg"],
            "miou_before": s0["miou"], "miou_after": s1["miou"], "f1_before": s0["f1"], "f1_after": s1["f1"],
            "area_ratio_before": area_ratios(before), "area_ratio_after": area_ratios(after),
            "coverage_ok": feasible(after, area_floor), "pixels": int(labels.size), "area_floor": area_floor}


def fold_bias(model, bias) -> None:
    """Add ``bias`` to the final classifier's bias in place (every output logit shifts by it)."""
    conv = model.final_classifier()
    with torch.no_grad():
        conv.bias.add_(torch.as_tensor(bias, dtype=conv.bias.dtype, device=conv.bias.device))
