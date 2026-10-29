"""Train the Track 1 teacher v2 (``macvi.eurs.teacher``); same contract as ``macvi.eurs.train.train``.

Writes ``config.json``, ``val_scenes.json``, ``best.pt`` (head only when the backbone is
pretrained) and ``history.json`` into the run dir and returns ``(model, info)`` with the keys the
loop reads, so ``train.train`` dispatches here for ``model.arch: dinov3_vit`` and nothing else in
the loop, the distillation or the packaging changes.

Data (``teacher:`` section of the config, all optional):
- every version is a degraded rendering of its scene's clean image; with ``clean_p`` the clean
  image itself is the input (target = itself, metadata "clean": water type 0, depth 0);
- ``aug.redegrade_p``: a fresh synthetic rendering of the clean image (metadata unknown);
  ``meta_dropout``: the metadata is hidden, so the restorer also works without it (distillation,
  unparseable names);
- scene-balanced sampling, scenes containing ``rare_classes`` (elasmobranch 3, other 4: 2 and 3 of
  the 20 train scenes) drawn ``1 + rare_boost`` times as often; ``water_boost`` ({water type: factor},
  from the file-name metadata) draws e.g. the murky waters 9-10 more often within each scene;
- ``cache: true`` computes the frozen backbone features once per image (``cache_per_scene``
  versions per scene plus the clean image, full frame at ``data.size``) and trains the head on
  them for many cheap epochs (hflip is the only geometric augmentation; flipping ViT features
  approximates the features of the flipped image). Features are fp16 on local disk, deleted after.
- ``train.pasta``: [p, alpha, k, beta] amplitude-spectrum augmentation of the input (``macvi.eurs.train.pasta``);
  not with ``cache``.
- ``data.external`` (``macvi.eurs.external``): public labelled photos mixed into the draws, scored by
  the segmentation loss only. Train has no starfish/turtle ("other") and two elasmobranch scenes, and
  the 5 Oct teacher labelled the Dev starfish scene elasmobranch and never predicted "other".
"""

from __future__ import annotations

import hashlib
import json
import random
import shutil
import tempfile
import time
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader, Dataset

from macvi.common.config import get
from macvi.eurs import augment, external, key_affine, meta as meta_mod, teacher as teacher_mod
from macvi.eurs.augment import AugConfig
from macvi.eurs.data import (Layout, Sample, cap_per_scene, discover, pixel_class_weights, read_mask, read_rgb,
                             split_by_scene)
from macvi.eurs.evaluate import evaluate_torch
from macvi.eurs.losses import lovasz_softmax, ssim
from macvi.eurs.synthetic import degrade


def _to_t(a: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(a)).permute(2, 0, 1).float() / 255.0


def _resize(img: np.ndarray, size: tuple[int, int], nearest: bool = False) -> np.ndarray:
    h, w = size
    return cv2.resize(img, (w, h), interpolation=cv2.INTER_NEAREST if nearest else cv2.INTER_AREA)


# ------------------------------------------------------------------ sampling

def scene_classes(samples: list[Sample], min_fraction: float = 0.001) -> dict[str, set[int]]:
    """Classes covering at least ``min_fraction`` of each scene's mask (one mask read per scene)."""
    out = {}
    for s in {s.scene: s for s in samples}.values():
        counts = np.bincount(read_mask(s.mask).ravel(), minlength=256).astype(np.float64)
        out[s.scene] = {int(c) for c in np.nonzero(counts / counts.sum() >= min_fraction)[0]}
    return out


def sample_weights(samples: list[Sample], rare_classes: list[int], boost: float,
                   water_boost: dict | None = None) -> list[float]:
    """Scene-balanced weights; scenes containing a rare class are drawn ``1 + boost`` times as often.

    ``water_boost`` ({water type: factor}) reweights the versions inside each scene by their water type
    (file-name metadata; unparseable names keep factor 1), keeping every scene's total share unchanged.
    """
    wb = {int(k): float(v) for k, v in (water_boost or {}).items()}
    factor = [wb.get(m.water, 1.0) if (m := meta_mod.parse(s.degraded)) else 1.0 for s in samples]
    total: dict[str, float] = {}
    for s, f in zip(samples, factor):
        total[s.scene] = total.get(s.scene, 0.0) + f
    present = scene_classes(samples) if rare_classes and boost else {}
    rare = set(rare_classes or [])
    return [(1.0 + (boost if present.get(s.scene, set()) & rare else 0.0)) * f / total[s.scene]
            for s, f in zip(samples, factor)]


# ------------------------------------------------------------------ weight averaging

def ema_update(pairs, decay: float, step: int) -> None:
    """EMA of the trainable weights (warm-up decay min(decay, (1+step)/(10+step)), as in timm/tf)."""
    d = min(decay, (1 + step) / (10 + step))
    with torch.no_grad():
        for p, e in pairs:
            e.lerp_(p.detach(), 1 - d)


@contextmanager
def ema_weights(pairs):
    """Temporarily load the EMA weights into the model (for validation and saving)."""
    if not pairs:
        yield
        return
    with torch.no_grad():
        live = [p.detach().clone() for p, _ in pairs]
        for p, e in pairs:
            p.copy_(e)
    try:
        yield
    finally:
        with torch.no_grad():
            for (p, _), v in zip(pairs, live):
                p.copy_(v)


# ------------------------------------------------------------------ datasets

class TeacherDataset(Dataset):
    """(input, clean, mask, meta vector) with the full augmentation (crops, flips, re-degradation)."""

    def __init__(self, samples: list[Sample], size: tuple[int, int], aug: AugConfig, clean_p: float = 0.0,
                 meta_dropout: float = 0.0, input_map: dict | None = None):
        self.samples, self.size, self.clean_p, self.meta_dropout = samples, size, clean_p, meta_dropout
        self.input_map = input_map  # teacher.input_map: train on the keyed colour map the judge feeds at test time
        self.redegrade_p = aug.redegrade_p
        self.aug = replace(aug, redegrade_p=0.0)  # handled here: a re-rendering has unknown metadata

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int):
        s = self.samples[i]
        rng = np.random.default_rng(int(torch.randint(0, 2**31 - 1, ()).item()))
        clean, mask = read_rgb(s.clean), read_mask(s.mask)
        u = rng.random()
        if u < self.clean_p:
            img, m = clean, meta_mod.CLEAN
        elif u < self.clean_p + self.redegrade_p:
            img, m = degrade(clean, rng), None
        else:
            img, m = read_rgb(s.degraded), meta_mod.parse(s.degraded)
            if self.input_map is not None and m is not None:
                img = key_affine.map_input(img, (m.water, m.depth), self.input_map)
        if rng.random() < self.meta_dropout:
            m = None
        img, clean, mask = augment.apply(img, clean, mask, self.size, self.aug, rng)
        return _to_t(img), _to_t(clean), torch.from_numpy(np.ascontiguousarray(mask)).long(), torch.from_numpy(meta_mod.encode(m))


class FeatureCache:
    """Frozen backbone features per image (fp16 .npy, (4, C, H/16, W/16)) at one input size."""

    def __init__(self, root: Path, size: tuple[int, int]):
        self.root, self.size = Path(root), tuple(size)
        self.root.mkdir(parents=True, exist_ok=True)

    def path(self, image: Path) -> Path:
        return self.root / (hashlib.sha1(f"{image}|{self.size}".encode(), usedforsecurity=False).hexdigest()[:20] + ".npy")

    @torch.no_grad()
    def build(self, model: teacher_mod.DinoTeacher, images: list[Path], device: str, batch: int = 4) -> float:
        """Compute missing features; returns seconds spent."""
        start = time.time()
        todo = [p for p in dict.fromkeys(images) if not self.path(p).exists()]
        model.eval()
        for i in range(0, len(todo), batch):
            chunk = todo[i:i + batch]
            x = torch.stack([_to_t(_resize(read_rgb(p), self.size)) for p in chunk]).to(device)
            feats = torch.stack(model.backbone_features(model.normalise(x)), 1).half().cpu().numpy()
            for p, f in zip(chunk, feats):
                np.save(self.path(p), f)
        return time.time() - start

    def load(self, image: Path) -> np.ndarray:
        return np.load(self.path(image))


class CachedDataset(Dataset):
    """(input, clean, mask, meta vector, cached features); hflip only."""

    def __init__(self, samples: list[Sample], size: tuple[int, int], cache: FeatureCache, clean_p: float = 0.0,
                 meta_dropout: float = 0.0, hflip: bool = True):
        self.samples, self.size, self.cache = samples, size, cache
        self.clean_p, self.meta_dropout, self.hflip = clean_p, meta_dropout, hflip

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int):
        s = self.samples[i]
        rng = np.random.default_rng(int(torch.randint(0, 2**31 - 1, ()).item()))
        use_clean = rng.random() < self.clean_p
        src = s.clean if use_clean else s.degraded
        m = meta_mod.CLEAN if use_clean else meta_mod.parse(s.degraded)
        if rng.random() < self.meta_dropout:
            m = None
        img, clean = _resize(read_rgb(src), self.size), _resize(read_rgb(s.clean), self.size)
        mask, feats = _resize(read_mask(s.mask), self.size, nearest=True), self.cache.load(src)
        if self.hflip and rng.random() < 0.5:
            img, clean, mask, feats = img[:, ::-1], clean[:, ::-1], mask[:, ::-1], feats[..., ::-1]
        return (_to_t(img), _to_t(clean), torch.from_numpy(np.ascontiguousarray(mask)).long(),
                torch.from_numpy(meta_mod.encode(m)), torch.from_numpy(np.ascontiguousarray(feats)))


class ExternalTeacherDataset(Dataset):
    """``external.ExternalDataset`` items in the teacher's format: floats in [0, 1]; metadata of the rendered
    (water, depth) key with ``data.external.rsuigm``, else unknown."""

    def __init__(self, inner: external.ExternalDataset, input_map: dict | None = None):
        self.inner, self.input_map = inner, input_map

    def __len__(self) -> int:
        return len(self.inner)

    def __getitem__(self, i: int):
        deg, photo, mask, tag, key = self.inner.item(i)
        if self.input_map is not None and key:
            deg = torch.from_numpy(key_affine.map_input(deg.permute(1, 2, 0).numpy(), key, self.input_map)).permute(2, 0, 1)
        m = meta_mod.Meta("", *key) if key else None
        return deg.float() / 255.0, photo.float() / 255.0, mask.long(), torch.from_numpy(meta_mod.encode(m)), tag


def default_cache_root(out_dir: Path) -> Path:
    base = Path("/kaggle/tmp") if Path("/kaggle/tmp").is_dir() else Path(tempfile.gettempdir())
    return base / "macvi_feature_cache" / out_dir.name


# ------------------------------------------------------------------ loss and loop

def teacher_loss(restored, clean, logits, mask, cfg: dict[str, Any], class_weights=None) -> torch.Tensor:
    seg = F.cross_entropy(logits, mask, weight=class_weights) + get(cfg, "train.w_lovasz", 0.5) * lovasz_softmax(logits, mask)
    rest = F.l1_loss(restored, clean) + (1.0 - ssim(restored, clean))
    return get(cfg, "train.w_seg", 1.0) * seg + get(cfg, "train.w_restore", 1.0) * rest


def _seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def train_teacher(cfg: dict[str, Any], data_root: str | Path, out_dir: str | Path,
                  deadline: float | None = None) -> tuple[teacher_mod.DinoTeacher, dict[str, Any]]:
    from macvi.eurs.train import out_of_time

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(cfg, indent=1, default=str))
    seed = get(cfg, "train.seed", 0)
    _seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.backends.cudnn.benchmark = True
    size = tuple(get(cfg, "data.size", [768, 1024]))

    samples = discover(data_root, Layout(**cfg.get("data", {}).get("layout", {})))
    train_s, val_s = split_by_scene(samples, get(cfg, "data.val_fraction", 0.2), seed, get(cfg, "data.fold", 0))
    val_epoch = cap_per_scene(val_s, get(cfg, "data.val_max_per_scene"), seed)
    val_scenes = sorted({s.scene for s in val_s})
    (out_dir / "val_scenes.json").write_text(json.dumps(val_scenes))

    model = teacher_mod.build(cfg).to(device)
    clean_p, meta_dropout = get(cfg, "teacher.clean_p", 0.15), get(cfg, "teacher.meta_dropout", 0.2)
    use_cache = bool(get(cfg, "teacher.cache", False))
    input_map = key_affine.load_table(p) if (p := get(cfg, "teacher.input_map")) else None
    cache_root, cache_seconds, cached = None, 0.0, []
    if use_cache:
        if input_map is not None:
            raise ValueError("teacher.cache stores features of the raw input; turn teacher.input_map off")
        if model.cfg.lora_rank > 0 or model.cfg.rein_tokens > 0:
            raise ValueError("teacher.cache needs a frozen backbone; turn model.lora_rank and model.rein_tokens off")
        train_s = cap_per_scene(train_s, get(cfg, "teacher.cache_per_scene", 40), seed)
        cache_root = Path(get(cfg, "teacher.cache_dir") or default_cache_root(out_dir))
        cache = FeatureCache(cache_root, size)
        cached = [s.degraded for s in train_s] + sorted({s.clean for s in train_s})
        cache_seconds = cache.build(model, cached, device)
        dataset: Dataset = CachedDataset(train_s, size, cache, clean_p, meta_dropout, get(cfg, "teacher.cache_hflip", True))
    else:
        aug = AugConfig(**{k: tuple(v) if isinstance(v, list) else v for k, v in cfg.get("aug", {}).items()})
        dataset = TeacherDataset(train_s, size, aug, clean_p, meta_dropout, input_map)

    per_epoch = get(cfg, "train.samples_per_epoch") or len(train_s)
    weights = sample_weights(train_s, get(cfg, "teacher.rare_classes", [3, 4]), get(cfg, "teacher.rare_boost", 2.0),
                             get(cfg, "teacher.water_boost"))
    xs, ext_counts = external.settings(cfg), None
    if xs is not None:  # batches gain a tag row: [is_external, ce_weight, w_restore]
        if use_cache:
            raise ValueError("data.external does not combine with teacher.cache: switch one of them off")
        ext = external.load(xs["root"], xs)
        if xs.get("copy_paste"):  # donors: the train scenes' clean image + GT mask (never a held-out scene)
            xs["copy_paste"] = {**xs["copy_paste"], "donors": sorted({(str(s.clean), str(s.mask)) for s in train_s})}
        dataset = ConcatDataset([external.Tagged(dataset), ExternalTeacherDataset(external.ExternalDataset(ext, size, aug, xs), input_map)])
        weights = external.mix_weights(weights, external.sampling_weights(ext, xs), float(xs["share"]))
        per_epoch = get(cfg, "train.samples_per_epoch") or int(round(len(train_s) / (1 - float(xs["share"]))))
        ext_counts = {}
        for x in ext:
            ext_counts[x.source] = ext_counts.get(x.source, 0) + 1
    sampler = torch.utils.data.WeightedRandomSampler(weights, per_epoch, replacement=True)
    batch_size, workers = get(cfg, "train.batch_size", 4), get(cfg, "train.num_workers", 2)
    loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=workers,
                        drop_last=per_epoch >= 2 * batch_size, pin_memory=device == "cuda",
                        persistent_workers=workers > 0)
    if len(loader) == 0:
        raise ValueError(f"no training batches: {per_epoch} samples per epoch, batch size {batch_size}")

    params = teacher_mod.trainable_parameters(model)
    epochs, lr = get(cfg, "train.epochs", 20), get(cfg, "train.lr", 1e-3)
    # backbone adapters (Rein tokens, LoRA) can take a lower rate than the head: train.adapter_lr_mult (default 1)
    is_adapter = {id(p) for n, p in model.named_parameters() if p.requires_grad and (n.startswith("rein.") or ".lora_" in n)}
    mult = float(get(cfg, "train.adapter_lr_mult", 1.0))
    groups = [g for g in ({"params": [p for p in params if id(p) not in is_adapter], "lr": lr},
                          {"params": [p for p in params if id(p) in is_adapter], "lr": lr * mult}) if g["params"]]
    opt = torch.optim.AdamW(groups, lr=lr, weight_decay=get(cfg, "train.weight_decay", 0.05))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[g["lr"] for g in groups], total_steps=epochs * len(loader),
                                                pct_start=0.1)
    grad_clip = float(get(cfg, "train.grad_clip", 0) or 0)
    ema_decay = get(cfg, "train.ema")  # e.g. 0.999: validate and save an EMA of the weights (steadier than one epoch)
    ema = [(p, p.detach().clone()) for p in params] if ema_decay else []
    step = 0
    cw = get(cfg, "train.class_weights")
    if cw == "auto":
        cw = pixel_class_weights(train_s, get(cfg, "model.num_classes", 5))
    class_weights = torch.tensor(cw, dtype=torch.float32, device=device) if cw else None
    use_amp = device == "cuda"
    scaler = torch.amp.GradScaler(enabled=use_amp)
    if use_amp:
        torch.cuda.reset_peak_memory_stats()

    select = get(cfg, "train.select", "composite")  # "q_seg" for segmentation-only Track 1 runs (w_restore 0)
    from macvi.eurs.train import pasta  # lazy: train dispatches here
    pasta_cfg = get(cfg, "train.pasta")
    history, best = [], {"composite": -1.0, select: -1.0}
    epochs_run, stopped_early, started, seen, train_time = 0, False, time.time(), 0, 0.0
    try:
        for epoch in range(epochs):
            if epoch and out_of_time(deadline, (time.time() - started) / epoch, time.time()):
                stopped_early = True
                break
            epochs_run, t0 = epoch + 1, time.time()
            model.train()
            total = torch.zeros((), device=device)
            for batch in loader:
                img, clean, mask, metav = (t.to(device, non_blocking=True) for t in batch[:4])
                feats = list(batch[4].to(device, non_blocking=True).float().unbind(1)) if use_cache else None
                if pasta_cfg and not use_cache:  # train.pasta: amplitude-spectrum augmentation of the input only
                    img = pasta(img, *pasta_cfg)
                with torch.autocast(device_type=device, enabled=use_amp):
                    restored, logits = model(img, meta=metav, feats=feats)
                if xs is not None:  # external rows: segmentation loss only (their photo is no clean target)
                    loss, _ = external.mixed_loss(
                        lambda r, c, lg, m: (teacher_loss(r, c, lg, m, cfg, class_weights), {}), restored.float(), clean,
                        logits.float(), mask, batch[4].to(device), get(cfg, "train.w_seg", 1.0),
                        get(cfg, "train.w_lovasz", 0.5), get(cfg, "train.w_restore", 1.0), class_weights)
                else:
                    loss = teacher_loss(restored.float(), clean, logits.float(), mask, cfg, class_weights)
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                if grad_clip:  # train.grad_clip: global norm cap (15 Oct: Rein lanes diverged on folds 0/1 without it)
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(params, grad_clip)
                scaler.step(opt)
                scaler.update()
                sched.step()
                if ema:
                    ema_update(ema, ema_decay, step)
                step += 1
                total += loss.detach()
                seen += img.shape[0]
            train_time += time.time() - t0
            row = {"epoch": epoch, "loss": float(total) / len(loader), "seconds": time.time() - t0}
            if not val_s:  # all-data fit: same schedule as selection; keep the selected epoch, else the last
                history.append(row)
                best = {"composite": float("nan"), "epoch": epoch}
                with ema_weights(ema):
                    torch.save(model.state_dict(), out_dir / "best.pt")
                if epoch == get(cfg, "train.save_epoch"):
                    break
                continue
            with ema_weights(ema):
                val = evaluate_torch(model, val_epoch, size, device, input_map=input_map)
                history.append({**row, **{k: val[k] for k in ("composite", "q_seg", "psnr", "ssim")}})
                if val[select] > best[select]:
                    best = {**val, "epoch": epoch}
                    torch.save(model.state_dict(), out_dir / "best.pt")
    finally:
        if cache_root is not None and not get(cfg, "teacher.keep_cache", False):
            shutil.rmtree(cache_root, ignore_errors=True)
    model.load_state_dict(torch.load(out_dir / "best.pt", map_location=device, weights_only=True))
    (out_dir / "history.json").write_text(json.dumps(history, indent=1))
    perf = {"train_seconds": train_time, "wall_seconds": time.time() - started, "samples_seen": seen,
            "samples_per_second": seen / max(train_time, 1e-9),
            "cache_seconds": cache_seconds, "cached_images": len(cached),
            "peak_mem_gb": torch.cuda.max_memory_allocated() / 2**30 if use_amp else 0.0}
    return model.cpu(), {"best": best, "history": history, "val_samples": [str(s.degraded) for s in val_s],
                         "class_weights": cw, "external": ext_counts, "epochs_run": epochs_run, "stopped_early": stopped_early, "perf": perf}
