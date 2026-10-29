"""External training samples (public datasets mapped to EURS) mixed into EURS training.

The archive comes from the CPU data kernel (``macvi autopilot --setup-external-data``, see
``external_build``); a GPU job that attaches it sets ``data.external.root``. Configuration
(``data.external``, off unless ``enabled``)::

    enabled: true
    root: /kaggle/tmp/external      # unpacked eurs_external.zip (manifest.csv at its top)
    share: 0.5                      # fraction of training draws that are external images
    sources:                        # omitted: every source in the manifest with the defaults below
      usis16k: {weight: 1.0, ce_weight: 0.5, unlabelled: background}
      coralscapes: {weight: 0.25, ce_weight: 1.0, unlabelled: ignore}
    class_boost: {3: 4.0, 4: 2.0}   # draw images containing elasmobranch / other this much more often
    redegrade_p: 1.0                # probability of a fresh synthetic.degrade rendering per draw
    strength: [0.3, 1.0]            # blend between the external photo and that rendering
    w_restore: 0.0                  # restoration loss weight on external rows (no clean ground truth)

- ``weight``: the source's share of the external draws (images of a source are drawn uniformly,
  times ``class_boost``); ``ce_weight`` scales its segmentation loss (0.5 suits salient-object
  sources whose unannotated pixels are not reliably background); ``unlabelled`` decides what the
  never-annotated pixels (254 in the archive) train as: ``background`` or ``ignore``;
  ``ignore_classes`` lists EURS ids the source labels unreliably, trained as ignore (e.g. ``[2]`` for
  sources that call sharks and rays "fish", which would teach elasmobranch = fish). ``predegraded``
  marks images that are already degraded underwater renderings without labels (e.g. the public EURS Dev
  split): they are never re-rendered, carry no restoration target, and train through the teacher's KD.
- External photos are already underwater; they are re-degraded only partially (``strength``) so
  the input still covers RSUIGM-like casts without stacking two full degradations.
- Validation stays EURS-only: external samples are only ever added to the training dataset.
- Each batch row carries ``meta = [is_external, ce_weight, w_restore]``; ``mixed_loss`` scores
  EURS rows with the normal loss and external rows with an ignore-aware segmentation loss.
"""

from __future__ import annotations

import csv
import dataclasses
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, Dataset

from macvi.eurs.synthetic import degrade

IGNORE, UNLABELLED = 255, 254
# 17 Oct (gap inventory v4, R1 coral convention): with ``data.external.bg_partial`` an external background pixel is
# the partial label {background, coral}: EURS GT draws coral as a coarse reef region that includes rock and sand, which
# the external sources label background, so only the EURS scenes decide coral vs background (Bevandic & Segvic,
# BMVC 2022, arXiv 2207.08445: super-class labels trained as partial labels).
BG_OR_CORAL = 253
DEFAULTS: dict[str, Any] = {"enabled": False, "root": None, "share": 0.5, "sources": None, "class_boost": {},
                            "redegrade_p": 1.0, "strength": [0.3, 1.0], "w_restore": 0.0, "rsuigm": None,
                            "rsuigm_gpu": False, "copy_paste": None, "bg_partial": False}
SOURCE_DEFAULTS: dict[str, Any] = {"weight": 1.0, "ce_weight": 1.0, "unlabelled": "background", "ignore_classes": (),
                                   "predegraded": False}
PREDEGRADED = 2.0  # meta[0] of a predegraded row: external (> 0.5) but never rendered by the training step
CLASS_COLUMNS = ("px_background", "px_coral", "px_fish", "px_elasmobranch", "px_other")


@dataclass(frozen=True)
class ExternalSample:
    source: str
    image: Path
    mask: Path
    ce_weight: float
    unlabelled: int          # what archive value 254 becomes: 0 (background) or 255 (ignore)
    classes: frozenset[int]  # EURS class ids present in the mask
    ignore_classes: tuple[int, ...] = ()  # EURS ids this source labels unreliably: trained as ignore
    predegraded: bool = False  # already a degraded rendering: no re-render, no restoration target


def settings(cfg: dict[str, Any]) -> dict[str, Any] | None:
    """``data.external`` merged with the defaults, or None when external data is off."""
    raw = (cfg.get("data") or {}).get("external") or {}
    s = {**DEFAULTS, **raw}
    if not s["enabled"]:
        return None
    if not s["root"]:
        raise ValueError("data.external.enabled needs data.external.root (the unpacked eurs_external.zip; a GPU "
                         "job sets it when remote.attach_external_data attaches the external data kernel)")
    if not 0.0 < float(s["share"]) < 1.0:
        raise ValueError(f"data.external.share must be in (0, 1), got {s['share']}")
    return s


def load(root: str | Path, s: dict[str, Any]) -> list[ExternalSample]:
    """Samples of the configured sources from ``<root>/manifest.csv`` (every source if none configured)."""
    root = Path(root)
    manifest = root / "manifest.csv"
    if not manifest.exists():
        found = next(iter(sorted(root.rglob("manifest.csv"))), None)  # tolerate a wrapper folder
        if found is None:
            raise FileNotFoundError(f"no manifest.csv under {root}")
        root, manifest = found.parent, found
    with open(manifest, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    wanted = s.get("sources")
    unknown = set(wanted or {}) - {r["source"] for r in rows}
    if unknown:
        raise ValueError(f"data.external.sources {sorted(unknown)} are not in {manifest}")
    out = []
    for r in rows:
        if wanted is not None and r["source"] not in wanted:
            continue
        opts = {**SOURCE_DEFAULTS, **((wanted or {}).get(r["source"]) or {})}
        if opts["unlabelled"] not in ("background", "ignore"):
            raise ValueError(f"data.external.sources.{r['source']}.unlabelled must be background or ignore")
        ign = tuple(int(c) for c in opts["ignore_classes"] or ())
        classes = frozenset(c for c, col in enumerate(CLASS_COLUMNS) if int(r.get(col) or 0) > 0 and c not in ign)
        out.append(ExternalSample(r["source"], root / r["image"], root / r["mask"], float(opts["ce_weight"]),
                                  0 if opts["unlabelled"] == "background" else IGNORE, classes, ign,
                                  bool(opts["predegraded"])))
    if not out:
        raise ValueError(f"no external samples selected from {manifest}")
    return out


def sampling_weights(samples: list[ExternalSample], s: dict[str, Any]) -> list[float]:
    """Each source gets its ``weight`` share of the draws, spread over its images by ``class_boost``."""
    boost = {int(k): float(v) for k, v in (s.get("class_boost") or {}).items()}
    per_image = [max([1.0] + [boost[c] for c in x.classes if c in boost]) for x in samples]
    totals: dict[str, float] = {}
    for x, b in zip(samples, per_image):
        totals[x.source] = totals.get(x.source, 0.0) + b
    wanted = s.get("sources") or {}
    src_w = {name: float({**SOURCE_DEFAULTS, **(wanted.get(name) or {})}["weight"]) for name in totals}
    w = [src_w[x.source] * b / totals[x.source] for x, b in zip(samples, per_image)]
    total = sum(w)
    if total <= 0:
        raise ValueError("data.external.sources weights are all zero")
    return [v / total for v in w]


def mix_weights(eurs: list[float], external: list[float], share: float) -> list[float]:
    """Concatenated sampler weights: EURS draws ``1 - share`` of the time, external ``share``."""
    se, sx = sum(eurs), sum(external)
    return [w * (1 - share) / se for w in eurs] + [w * share / sx for w in external]


def redegrade(img: np.ndarray, rng: np.random.Generator, p: float, strength: tuple[float, float]) -> np.ndarray:
    """With probability ``p``, blend the photo with a fresh ``synthetic.degrade`` rendering."""
    if rng.random() >= p:
        return img
    s = rng.uniform(*strength)
    out = (1 - s) * img.astype(np.float32) + s * degrade(img, rng).astype(np.float32)
    return np.clip(out + 0.5, 0, 255).astype(np.uint8)


def _meta(is_external: float, ce_weight: float, w_restore: float) -> torch.Tensor:
    return torch.tensor([is_external, ce_weight, w_restore], dtype=torch.float32)


class ExternalDataset(Dataset):
    """(degraded, photo, mask, meta) at ``size``; the photo stands in for the missing clean target."""

    def __init__(self, samples: list[ExternalSample], size: tuple[int, int], aug, s: dict[str, Any],
                 kd_shape: tuple[int, int, int] | None = None):
        self.samples, self.size, self.s, self.kd_shape = samples, size, s, kd_shape
        # the dataset re-degrades itself; augmentation keeps only the geometric part
        self.aug = dataclasses.replace(aug, redegrade_p=0.0) if hasattr(aug, "redegrade_p") else aug

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int):
        deg, clean, mask, meta = self.item(i)[:4]
        if self.kd_shape:  # batches with cached teacher logits: a zero slot (the loss skips external rows' KD)
            return deg, clean, mask, torch.zeros(self.kd_shape, dtype=torch.float16), meta
        return deg, clean, mask, meta

    def item(self, i: int):
        """(degraded, photo, mask, tag, key): ``key`` is the (water, depth) rendered when ``rsuigm`` is set (the
        params file of ``eurs.rsuigm``: the measured RSUIGM formation model, at a uniform random key of the release
        grid), else None (``synthetic.degrade`` blend, unknown metadata)."""
        from macvi.eurs import augment
        from macvi.eurs.data import read_mask, read_rgb

        x = self.samples[i]
        img, mask = read_rgb(x.image), read_mask(x.mask).copy()
        mask[mask == UNLABELLED] = x.unlabelled
        if x.ignore_classes:
            mask[np.isin(mask, x.ignore_classes)] = IGNORE
        if self.s.get("bg_partial"):
            mask[mask == 0] = BG_OR_CORAL
        rng = np.random.default_rng(int(torch.randint(0, 2**31 - 1, ()).item()))
        cp = self.s.get("copy_paste") or {}
        if x.predegraded:
            deg, clean, mask = augment.apply(img, img, mask, self.size, self.aug, rng)
            to_t = lambda a: torch.from_numpy(np.ascontiguousarray(a)).permute(2, 0, 1)  # noqa: E731
            return (to_t(deg), to_t(clean), torch.from_numpy(np.ascontiguousarray(mask.astype(np.uint8))),
                    _meta(PREDEGRADED, x.ce_weight, 0.0), None)
        if cp.get("donors") and rng.random() < float(cp.get("p", 0.5)):  # train-scene objects onto this photo
            from macvi.eurs.copypaste import paste
            for _ in range(int(rng.integers(1, int(cp.get("max_objects", 2)) + 1))):
                d_img, d_mask = cp["donors"][int(rng.integers(len(cp["donors"])))]
                img, mask = paste(img, mask, read_rgb(d_img), read_mask(d_mask), rng, tuple(cp.get("scale", (0.4, 1.0))))
        key = None
        if self.s.get("rsuigm") and self.s.get("rsuigm_gpu"):  # the training step renders it (rsuigm.degrade_torch)
            deg = img
        elif self.s.get("rsuigm"):
            from macvi.eurs import rsuigm
            zs = rsuigm.load_params(self.s["rsuigm"])["zs"]
            key = (int(rng.integers(1, 11)), int(zs[rng.integers(len(zs))]))
            deg = rsuigm.degrade(img, key[0], key[1], rng, params_path=self.s["rsuigm"])
        else:
            deg = redegrade(img, rng, float(self.s["redegrade_p"]), tuple(self.s["strength"]))
        deg, clean, mask = augment.apply(deg, img, mask, self.size, self.aug, rng)
        # uint8, like EURSDataset: the training loop scales and casts on the device
        to_t = lambda a: torch.from_numpy(np.ascontiguousarray(a)).permute(2, 0, 1)  # noqa: E731
        return (to_t(deg), to_t(clean), torch.from_numpy(np.ascontiguousarray(mask.astype(np.uint8))),
                _meta(1.0, x.ce_weight, float(self.s["w_restore"])), key)


class Tagged(Dataset):
    """An EURS dataset whose items gain ``meta = [0, 1, 1]`` (normal loss), to batch with external rows."""

    def __init__(self, dataset: Dataset):
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, i: int):
        return (*self.dataset[i], _meta(0.0, 1.0, 1.0))


@dataclass
class Mixture:
    dataset: Dataset
    weights: list[float]
    samples_per_epoch: int
    sources: dict[str, int]


def attach(cfg: dict[str, Any], eurs_dataset: Dataset, eurs_weights: list[float], size: tuple[int, int], aug,
           samples_per_epoch: int | None, kd_shape: tuple[int, int, int] | None = None,
           donors: list[tuple[str, str]] | None = None) -> Mixture | None:
    """The EURS training set mixed with external samples, or None when ``data.external`` is off.

    Without ``train.samples_per_epoch`` an epoch draws as many EURS samples as before plus the external share.
    ``kd_shape`` (C, h, w): the EURS items carry cached teacher logits (``distill.cache``); external items then carry
    zeros of that shape in the same slot, so both batch together. ``donors`` (clean image, mask) pairs of the train
    scenes feed ``data.external.copy_paste``; without them copy-paste is off."""
    s = settings(cfg)
    if s is None:
        return None
    if s.get("copy_paste") and donors:
        s["copy_paste"] = {**s["copy_paste"], "donors": sorted(set(donors))}
    samples = load(s["root"], s)
    share = float(s["share"])
    weights = mix_weights(list(eurs_weights), sampling_weights(samples, s), share)
    n = samples_per_epoch or int(round(len(eurs_dataset) / (1 - share)))
    counts: dict[str, int] = {}
    for x in samples:
        counts[x.source] = counts.get(x.source, 0) + 1
    return Mixture(ConcatDataset([Tagged(eurs_dataset), ExternalDataset(samples, size, aug, s, kd_shape)]), weights, n, counts)


# ------------------------------------------------------------------ losses

def seg_loss(logits: torch.Tensor, labels: torch.Tensor, sample_w: torch.Tensor, class_weights: torch.Tensor | None,
             w_lovasz: float) -> torch.Tensor:
    """Cross-entropy (class weighted, ``IGNORE`` skipped, rows scaled by ``sample_w``) + Lovasz on valid pixels.
    ``BG_OR_CORAL`` pixels score -log(p_background + p_coral) (background's class weight) and are left out of Lovasz."""
    from macvi.eurs.losses import lovasz_softmax

    valid = labels != IGNORE
    if not valid.any():
        return logits.sum() * 0.0
    part = labels == BG_OR_CORAL
    safe = labels.masked_fill(~valid | part, 0)
    pix = F.cross_entropy(logits, safe, weight=class_weights, reduction="none")
    if part.any():
        lse = torch.logsumexp(logits, 1)
        pl = (lse - torch.logsumexp(logits[:, :2], 1)) * (class_weights[0] if class_weights is not None else 1.0)
        pix = torch.where(part, pl, pix)
    cw = class_weights[safe] if class_weights is not None else torch.ones_like(pix)
    norm = (cw * valid).sum()
    ce = (pix * valid * sample_w.view(-1, 1, 1)).sum() / norm.clamp_min(1e-6)
    if not w_lovasz:
        return ce
    c = logits.shape[1]
    exact = valid & ~part
    if not exact.any():
        return ce
    flat = logits.permute(0, 2, 3, 1)[exact]  # (N, C)
    lov = lovasz_softmax(flat.view(-1, c, 1, 1), labels[exact].view(-1, 1, 1))
    return ce + w_lovasz * float(sample_w.mean()) * lov


def mixed_loss(eurs_loss: Callable[..., tuple[torch.Tensor, dict]], restored: torch.Tensor, clean: torch.Tensor,
               logits: torch.Tensor, mask: torch.Tensor, meta: torch.Tensor, w_seg: float, w_lovasz: float,
               w_restore: float, class_weights: torch.Tensor | None, w_grad: float = 0.0) -> tuple[torch.Tensor, dict]:
    """Batch loss with EURS rows scored by ``eurs_loss(restored, clean, logits, mask)`` (the normal
    training loss) and external rows by ``seg_loss`` plus a restoration term only where their
    ``w_restore`` is positive; each part weighted by its share of the batch rows."""
    from macvi.eurs.losses import restore_loss

    ext = meta[:, 0] > 0.5
    n = float(len(mask))
    loss, parts = logits.sum() * 0.0, {}
    if (~ext).any():
        part, parts = eurs_loss(restored[~ext], clean[~ext], logits[~ext], mask[~ext])
        loss = loss + part * float((~ext).sum()) / n
    if ext.any():
        seg = seg_loss(logits[ext], mask[ext], meta[ext, 1], class_weights, w_lovasz)
        part = w_seg * seg
        rw = meta[ext, 2]
        if (rw > 0).any():
            r, c = restored[ext][rw > 0], clean[ext][rw > 0]
            part = part + w_restore * float(rw[rw > 0].mean()) * restore_loss(r, c, w_grad)
        loss = loss + part * float(ext.sum()) / n
        parts = {**parts, "external_seg": float(seg.detach())}
    return loss, parts
