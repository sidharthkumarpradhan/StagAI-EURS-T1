"""Training losses aligned with the EURS composite.

- Segmentation: cross-entropy + Lovasz-softmax (Berman et al., CVPR 2018), a convex surrogate
  that optimises the Jaccard index directly; Qseg = F1 * mIoU, so IoU-aware training pays.
- Restoration: L1 + (1 - SSIM), matching the PSNR and SSIM terms of the score.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _lovasz_grad(gt_sorted: torch.Tensor) -> torch.Tensor:
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.cumsum(0)
    union = gts + (1 - gt_sorted).cumsum(0)
    jaccard = 1.0 - intersection / union
    jaccard[1:] = jaccard[1:] - jaccard[:-1]
    return jaccard


def lovasz_softmax(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Mean Lovasz hinge over classes present in the batch. logits (B,C,H,W), labels (B,H,W)."""
    probs = logits.softmax(1).permute(0, 2, 3, 1).reshape(-1, logits.shape[1])
    labels = labels.reshape(-1)
    losses = []
    for c in labels.unique():
        fg = (labels == c).float()
        errors = (fg - probs[:, c]).abs()
        errors_sorted, perm = torch.sort(errors, descending=True)
        losses.append(torch.dot(errors_sorted, _lovasz_grad(fg[perm])))
    return torch.stack(losses).mean()


def _gaussian_window(size: int = 11, sigma: float = 1.5, device=None) -> torch.Tensor:
    x = torch.arange(size, dtype=torch.float32, device=device) - size // 2
    g = torch.exp(-(x**2) / (2 * sigma**2))
    g = g / g.sum()
    return (g[:, None] * g[None, :])[None, None]


def ssim(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Mean SSIM over channels for images in [0, 1]."""
    c = x.shape[1]
    win = _gaussian_window(device=x.device).expand(c, 1, -1, -1)
    blur = lambda z: F.conv2d(z, win, padding=5, groups=c)  # noqa: E731
    mx, my = blur(x), blur(y)
    vx, vy, cxy = blur(x * x) - mx**2, blur(y * y) - my**2, blur(x * y) - mx * my
    c1, c2 = 0.01**2, 0.03**2
    return (((2 * mx * my + c1) * (2 * cxy + c2)) / ((mx**2 + my**2 + c1) * (vx + vy + c2))).mean()


def ohem_cross_entropy(logits: torch.Tensor, labels: torch.Tensor, weight: torch.Tensor | None = None,
                       thresh: float = 0.7, min_kept: float = 1 / 16) -> torch.Tensor:
    """Online hard example mining CE (bootstrapped CE, Wu et al. arXiv:1605.06885; the PP-LiteSeg /
    BiSeNet / DDRNet default): average only over pixels whose true-class probability is below
    ``thresh``, but at least the ``min_kept`` fraction of hardest pixels."""
    loss = F.cross_entropy(logits, labels, weight=weight, reduction="none").flatten()
    with torch.no_grad():
        prob = logits.softmax(1).gather(1, labels.unsqueeze(1)).flatten()
        k = max(1, min(prob.numel(), int(min_kept * prob.numel())))
        keep = (prob <= prob.kthvalue(k).values.clamp(min=thresh)).float()  # stays on the device: no sync
    return (loss * keep).sum() / keep.sum()


def gradient_l1(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """L1 between horizontal and vertical finite differences: penalises blurred or misplaced edges
    (gradient loss; MS-SSIM + gradient took a UIEB restorer from 21.99 to 24.96 dB over L1 alone,
    docs/research.md "Restoration under the Track 2 budget")."""
    dx = lambda z: z[..., :, 1:] - z[..., :, :-1]  # noqa: E731
    dy = lambda z: z[..., 1:, :] - z[..., :-1, :]  # noqa: E731
    return F.l1_loss(dx(x), dx(y)) + F.l1_loss(dy(x), dy(y))


def restore_loss(restored: torch.Tensor, clean: torch.Tensor, w_grad: float = 0.0) -> torch.Tensor:
    """L1 + (1 - SSIM), plus ``w_grad`` x gradient L1."""
    rest = F.l1_loss(restored, clean) + (1.0 - ssim(restored, clean))
    return rest + w_grad * gradient_l1(restored, clean) if w_grad else rest


def total_loss(restored, clean, logits, labels, w_seg=1.0, w_lovasz=0.5, w_restore=1.0, class_weights=None,
               ohem: tuple[float, float] | None = None, w_grad: float = 0.0):
    """``ohem`` = (threshold, min kept fraction) replaces plain CE by ``ohem_cross_entropy``."""
    ce = (ohem_cross_entropy(logits, labels, class_weights, *ohem) if ohem
          else F.cross_entropy(logits, labels, weight=class_weights))
    seg = ce + w_lovasz * lovasz_softmax(logits, labels)
    rest = restore_loss(restored, clean, w_grad)
    parts = {"seg": seg.detach(), "restore": rest.detach()}
    return w_seg * seg + w_restore * rest, parts
