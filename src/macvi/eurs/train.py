"""Train EURSNet from a config dict; keeps the checkpoint with the best validation composite.

Recipe options (all config-driven, off unless set; docs/research.md "Track 2 student v2"):
- ``train.ema_decay``: timm ``ModelEmaV3`` weights; per-epoch selection, the checkpoint and the
  export all use the EMA model.
- ``train.class_boost``: rare-class scene oversampling in the scene-balanced sampler.
- ``train.water_boost``: per-water-type (1..10) oversampling inside each scene, and the same skew for
  external photos rendered on the GPU.
- ``train.pasta``: [p, alpha, k, beta] proportional amplitude-spectrum augmentation of the student's input only
  (PASTA, Chattopadhyay et al., ICCV 2023); the teacher keeps the unperturbed image.
- ``train.ohem``: [threshold, min kept fraction] hard-pixel CE instead of plain CE.
- ``train.w_aux``: weight of the class-weighted CE on ``model.aux_channels``' training-only logits (EURS rows only).
- ``train.w_grad``: gradient L1 added to the restoration loss (``losses.restore_loss``).
- ``distill.w_cwd`` / ``distill.cwd_temperature``: channel-wise distillation next to the logit KD.
- ``distill.w_feat`` / ``distill.feat_layer``: feature distillation from a DINOv3 teacher (GKD, arXiv 2603.02554;
  FitNets-style): a training-only 1x1 projection of the student's deepest encoder map is matched (cosine) to the
  teacher's backbone map ``feat_layer`` on every row, external photos included. The projection is not part of the
  model, so the export is unchanged. Needs an online teacher (``distill.cache`` off).
- ``distill.teacher_input: clean``: the teacher sees the clean twin (privileged information).
- ``distill.cache``: teacher logits computed once per training image (``distill.LogitCache``),
  so student epochs run no teacher forward.
- ``calibrate.enabled``: per-class logit bias tuned on held-out scenes, folded into the model.
- ``train.init_from``: a ``best.pt`` state dict to start from (fine-tuning for a weight soup).
- ``train.run_seed``: reseeds sampling and augmentation after the split, so same-split runs differ.
- ``train.select``: which held-out epoch ``best.pt`` keeps. "composite" (default) is the scored composite;
  "composite_raw" swaps in Qseg without the coverage floor, which a class missing from small held-out
  folds drives to 0 (fold-0: iou_3 = 0, so "composite" selected on restoration alone, 24 Oct 2026);
  "last" keeps the last epoch, as an all-data fit does, so held-out runs judge the epoch that ships.
"""

from __future__ import annotations

import json
import random
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from macvi.common.config import get
from macvi.eurs import calibrate, external, rsuigm
from macvi.eurs.augment import AugConfig
from macvi.eurs.data import (EURSDataset, Layout, cap_per_scene, discover, pixel_class_weights, scene_balanced_weights,
                             split_by_scene, water_boost_weights, worker_init)
from macvi.eurs.distill import TEACHER_INPUTS, LogitCache, cwd_loss, kd_loss, load_teacher, run_teacher, teacher_dirs
from macvi.eurs.evaluate import evaluate_static, evaluate_torch
from macvi.eurs.losses import total_loss
from macvi.eurs.model import EURSNet, ModelConfig
from macvi.eurs.scoring import SEG_W


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_model(cfg: dict[str, Any]) -> EURSNet:
    from macvi.eurs import teacher  # Track 1 teacher v2 (model.arch: dinov3_vit)

    if teacher.is_teacher(cfg):
        return teacher.build(cfg)
    return EURSNet(ModelConfig(**cfg.get("model", {})))


def out_of_time(deadline: float | None, epoch_seconds: float, now: float) -> bool:
    """True when another epoch as long as the last one would end after ``deadline`` (epoch seconds)."""
    return deadline is not None and now + epoch_seconds > deadline


def _check_teacher_split(teacher_dir: str, val_scenes: list[str]) -> None:
    """The teacher must not have trained on our validation scenes."""
    scenes_file = Path(teacher_dir) / "val_scenes.json"
    if not scenes_file.exists():
        raise ValueError(f"teacher {teacher_dir} has no val_scenes.json; retrain it with this version")
    teacher_val = json.loads(scenes_file.read_text())
    if sorted(teacher_val) != val_scenes:
        raise ValueError(f"teacher {teacher_dir} held out {teacher_val}, student holds out {val_scenes}: "
                         "use the same data.fold/val_fraction/seed for both tracks")


def _calibrate(cfg: dict[str, Any], model: EURSNet, val_s: list, size: tuple[int, int], device: str,
               out_dir: Path) -> dict[str, Any] | None:
    """Fit (or take ``calibrate.bias`` from the config) and fold the per-class logit bias."""
    cal = cfg.get("calibrate") or {}
    if not cal.get("enabled"):
        return None
    if cal.get("bias") is not None:  # e.g. an all-scene refit reusing its selection run's bias
        result: dict[str, Any] | None = {"bias": [float(b) for b in cal["bias"]], "source": "config"}
    elif val_s:
        subset = cap_per_scene(val_s, cal.get("max_per_scene", 20), get(cfg, "train.seed", 0))
        logits, labels = calibrate.collect(model.to(device), subset, size, device, cal.get("pixel_stride", 4))
        result = {**calibrate.fit_bias(logits, labels, cal.get("area_floor", 0.5), cal.get("max_bias", 3.0),
                                       cal.get("step", 0.1)), "source": "validation", "images": len(subset)}
    else:
        return None
    calibrate.fold_bias(model, result["bias"])
    (out_dir / "calibration.json").write_text(json.dumps(result, indent=1))
    return result



def select_score(val: dict, select: str) -> float:
    """Checkpoint score under ``train.select`` ("last" scores like "composite"; the caller keeps every epoch)."""
    if select == "composite_raw":
        return val["composite"] + SEG_W * (val["q_seg_raw"] - val["q_seg"])
    return val["composite"]


def pasta(x: torch.Tensor, p: float, alpha: float, k: float, beta: float) -> torch.Tensor:
    """PASTA: scale each Fourier amplitude by N(1, sigma^2), sigma = (2 alpha r)^k + beta, r = radius / diagonal."""
    b, c, h, w = x.shape
    f = torch.fft.fft2(x.float())
    fy = torch.fft.fftfreq(h, device=x.device)[:, None]
    fx = torch.fft.fftfreq(w, device=x.device)[None]
    r = (fy ** 2 + fx ** 2).sqrt() / (2 ** 0.5)  # 0 at DC, 0.5 at the Nyquist corner
    eps = 1 + ((2 * alpha * r) ** k + beta) * torch.randn(b, c, h, w, device=x.device)
    out = torch.fft.ifft2(torch.polar(f.abs() * eps, f.angle())).real.clamp(0, 1)
    keep = (torch.rand(b, 1, 1, 1, device=x.device) >= p).float()
    return keep * x + (1 - keep) * out


def load_init(model: torch.nn.Module, state: dict) -> None:
    """Load a ``train.init_from`` state dict; only a new training-only aux head (``aux.*``) or a frozen pre-restorer
    (``pre.*``, built from its own checkpoint) may be missing."""
    res = model.load_state_dict(state, strict=False)
    bad = [k for k in res.missing_keys if not k.startswith(("aux.", "pre."))] + list(res.unexpected_keys)
    if bad:
        raise RuntimeError(f"init_from does not match the model: {bad[:5]}")

def train(cfg: dict[str, Any], data_root: str | Path, out_dir: str | Path,
          deadline: float | None = None) -> tuple[EURSNet, dict[str, Any]]:
    """``deadline`` (``time.time()`` value, not part of the config) ends training early, keeping the
    best checkpoint so far, so a remote GPU job never overruns its session."""
    from macvi.eurs import teacher

    if teacher.is_teacher(cfg):  # frozen-backbone teacher v2 has its own loop (macvi.eurs.teacher_train)
        from macvi.eurs.teacher_train import train_teacher
        return train_teacher(cfg, data_root, out_dir, deadline)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(cfg, indent=1, default=str))
    seed_everything(get(cfg, "train.seed", 0))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.backends.cudnn.benchmark = True  # fixed training size: autotuning pays off at once
    size = tuple(get(cfg, "data.size", [384, 768]))

    samples = discover(data_root, Layout(**cfg.get("data", {}).get("layout", {})))
    train_s, val_s = split_by_scene(samples, get(cfg, "data.val_fraction", 0.2), get(cfg, "train.seed", 0),
                                    get(cfg, "data.fold", 0))
    # Per-epoch model selection on a fixed subset (RSUIGM-style data has ~200 versions per scene).
    val_epoch = cap_per_scene(val_s, get(cfg, "data.val_max_per_scene"), get(cfg, "train.seed", 0))
    val_scenes = sorted({s.scene for s in val_s})
    (out_dir / "val_scenes.json").write_text(json.dumps(val_scenes))

    teacher_dir = get(cfg, "distill.teacher")
    teacher = load_teacher(teacher_dir, device) if teacher_dir else None
    if teacher is not None and val_s:
        for d in teacher_dirs(teacher_dir):
            _check_teacher_split(d, val_scenes)
    w_kd, w_rkd, temp = get(cfg, "distill.w_kd", 1.0), get(cfg, "distill.w_restore", 0.0), get(cfg, "distill.temperature", 2.0)
    w_cwd, cwd_temp = get(cfg, "distill.w_cwd", 0.0), get(cfg, "distill.cwd_temperature", 4.0)
    w_feat, feat_layer = get(cfg, "distill.w_feat", 0.0), get(cfg, "distill.feat_layer", -1)
    teacher_input = get(cfg, "distill.teacher_input", "degraded")
    if teacher_input not in TEACHER_INPUTS:
        raise ValueError(f"distill.teacher_input must be one of {TEACHER_INPUTS}, got {teacher_input!r}")
    kd_cache = None
    if teacher is not None and get(cfg, "distill.cache", False):
        if w_rkd > 0:
            raise ValueError("distill.cache stores segmentation logits only: set distill.w_restore to 0")
        root = get(cfg, "distill.cache_dir") or LogitCache.default_root(teacher_dir, teacher_input)
        kd_cache = LogitCache(root)
        # The build resizes with OpenCV in this process; with its thread pool alive here, a DataLoader worker forked
        # later deadlocks in worker_init's cv2.setNumThreads (seen 6 Oct 2026 on the V100: 3 h stuck, GPU idle).
        cv2.setNumThreads(1)
        kd_cache.build(teacher, [s.clean if teacher_input == "clean" else s.degraded for s in train_s], device)
        teacher = None  # epochs need no teacher forward
        if device == "cuda":
            torch.cuda.empty_cache()

    if get(cfg, "train.run_seed") is not None:  # same split as train.seed, different draws from here on
        seed_everything(get(cfg, "train.run_seed"))
    aug = AugConfig(**{k: tuple(v) if isinstance(v, list) else v for k, v in cfg.get("aug", {}).items()})
    dataset = EURSDataset(train_s, size, train=True, aug=aug, kd_cache=kd_cache, kd_source=teacher_input)
    per_epoch = get(cfg, "train.samples_per_epoch")
    # Scene-balanced sampling: every scene is seen equally often regardless of its version count
    # (scenes with rare classes more often when ``train.class_boost`` is set).
    sample_w = scene_balanced_weights(train_s, get(cfg, "train.class_boost"))
    water_boost = get(cfg, "train.water_boost")
    if water_boost:
        sample_w = water_boost_weights(train_s, sample_w, water_boost)
    kd_shape = (get(cfg, "model.num_classes", 5), *dataset.kd_size) if kd_cache is not None else None
    donors = [(str(s.clean), str(s.mask)) for s in train_s]  # copy-paste donors: train scenes only, never held-out
    mix = external.attach(cfg, dataset, sample_w, size, aug, per_epoch, kd_shape, donors)  # data.external or None
    if mix:
        dataset, sample_w, per_epoch = mix.dataset, mix.weights, mix.samples_per_epoch
    sampler = (torch.utils.data.WeightedRandomSampler(sample_w, per_epoch, replacement=True) if per_epoch else None)
    batch_size = get(cfg, "train.batch_size", 8)
    n_per_epoch = per_epoch or len(train_s)
    workers = get(cfg, "train.num_workers", 2)
    loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, shuffle=sampler is None, num_workers=workers,
                        drop_last=n_per_epoch >= 2 * batch_size, pin_memory=device == "cuda",
                        persistent_workers=workers > 0, prefetch_factor=4 if workers > 0 else None,
                        worker_init_fn=worker_init)
    if len(loader) == 0:
        raise ValueError(f"no training batches: {n_per_epoch} samples per epoch, batch size {batch_size}")

    model = build_model(cfg).to(device)
    if get(cfg, "train.init_from"):
        load_init(model, torch.load(get(cfg, "train.init_from"), map_location=device, weights_only=True))
    epochs = get(cfg, "train.epochs", 30)
    lr = get(cfg, "train.lr", 2e-3)
    # Pretrained encoders are usually fine-tuned with a smaller LR than freshly initialised heads.
    enc_lr = lr * get(cfg, "train.encoder_lr_mult", 1.0)
    enc_params = list(model.encoder.parameters())
    enc_ids = {id(p) for p in enc_params}
    groups = [{"params": enc_params, "lr": enc_lr}, {"params": [p for p in model.parameters() if id(p) not in enc_ids], "lr": lr}]
    feat_teacher = feat_proj = None
    if w_feat:
        if teacher is None:
            raise ValueError("distill.w_feat needs an online teacher (distill.teacher set, distill.cache off)")
        feat_teacher = teacher.members[0] if hasattr(teacher, "members") else teacher
        feat_proj = torch.nn.Conv2d(model.encoder.feature_info.channels()[model.high_idx],
                                    feat_teacher.backbone.embed_dim, 1).to(device)
        groups.append({"params": list(feat_proj.parameters()), "lr": lr})
    opt = torch.optim.AdamW(groups, weight_decay=get(cfg, "train.weight_decay", 1e-4))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[g["lr"] for g in groups], total_steps=epochs * len(loader))
    ema_decay = get(cfg, "train.ema_decay", 0.0)
    if ema_decay:
        from timm.utils import ModelEmaV3
        ema = ModelEmaV3(model, decay=ema_decay, use_warmup=True)  # warmup: early steps are not averaged in
    else:
        ema = None
    weights = get(cfg, "train.class_weights")
    if weights == "auto":  # long-tailed classes (elasmobranch is rare); Qseg multiplies per-class scores
        weights = pixel_class_weights(train_s, get(cfg, "model.num_classes", 5))
    class_weights = torch.tensor(weights, dtype=torch.float32, device=device) if weights else None
    ohem = tuple(get(cfg, "train.ohem")) if get(cfg, "train.ohem") else None
    use_amp = device == "cuda"
    scaler = torch.amp.GradScaler(enabled=use_amp)

    w_seg, w_lov, w_rest = get(cfg, "train.w_seg", 1.0), get(cfg, "train.w_lovasz", 0.5), get(cfg, "train.w_restore", 1.0)
    w_grad = get(cfg, "train.w_grad", 0.0)
    pasta_cfg = get(cfg, "train.pasta")
    w_aux = get(cfg, "train.w_aux", 0.0)
    eurs_loss = lambda r, c, lg, m: total_loss(r, c, lg, m, w_seg, w_lov, w_rest, class_weights, ohem, w_grad)  # noqa: E731
    history, best = [], {"composite": -1.0}
    xs = external.settings(cfg) if mix else None
    gpu_render = xs["rsuigm"] if xs and xs.get("rsuigm") and xs.get("rsuigm_gpu") else None
    epochs_run, stopped_early, started, step = 0, False, time.time(), 0
    for epoch in range(epochs):
        if epoch and out_of_time(deadline, (time.time() - started) / epoch, time.time()):  # mean epoch time
            stopped_early = True
            break
        epochs_run = epoch + 1
        model.train()
        loss_sum, batches = torch.zeros((), device=device), 0
        for batch in loader:
            deg, clean, mask = (t.to(device, non_blocking=True) for t in batch[:3])
            deg, clean, mask = deg.float().div_(255), clean.float().div_(255), mask.long()  # uint8 from the loader
            if gpu_render:  # external rows arrive clean: render them underwater here (data.external.rsuigm_gpu)
                flag = batch[-1][:, 0].to(device)
                rows = (flag > 0.5) & (flag < 1.5)  # predegraded rows (external.PREDEGRADED) are already rendered
                if rows.any():
                    deg[rows] = rsuigm.degrade_torch(clean[rows], gpu_render, water_p=water_boost)
            t_restored = t_logits = None
            with torch.autocast(device_type=device, enabled=use_amp):
                restored, logits = model(pasta(deg, *pasta_cfg) if pasta_cfg else deg)
                if teacher is not None:
                    t_restored, t_logits = run_teacher(teacher, clean if teacher_input == "clean" else deg)
            if kd_cache is not None:
                t_logits = batch[3].to(device, non_blocking=True)
            if mix:  # external rows: ignore-aware segmentation loss, no restoration target
                loss, _ = external.mixed_loss(eurs_loss, restored.float(), clean, logits.float(), mask,
                                              batch[-1].to(device), w_seg, w_lov, w_rest, class_weights, w_grad)
            else:
                loss, _ = eurs_loss(restored.float(), clean, logits.float(), mask)
            if w_aux and model.aux_logits is not None:
                aux, m = model.aux_logits.float(), mask
                if mix:  # external masks carry IGNORE/BG_OR_CORAL codes: deep supervision on EURS rows only
                    keep = batch[-1][:, 0].to(device) < 0.5
                    aux, m = aux[keep], m[keep]
                if len(m):
                    aux = torch.nn.functional.interpolate(aux, size=m.shape[-2:], mode="bilinear", align_corners=False)
                    loss = loss + w_aux * torch.nn.functional.cross_entropy(aux, m, weight=class_weights)
            if t_logits is not None:
                s_kd, t_kd = logits.float(), t_logits.float()
                if kd_cache is not None and mix:  # external rows have no cached teacher logits
                    keep = batch[-1][:, 0].to(device) < 0.5
                    s_kd, t_kd = s_kd[keep], t_kd[keep]
                if len(t_kd):
                    loss = loss + w_kd * kd_loss(s_kd, t_kd, temp)
                    if w_cwd:
                        loss = loss + w_cwd * cwd_loss(s_kd, t_kd, cwd_temp)
                if w_rkd > 0:
                    loss = loss + w_rkd * torch.nn.functional.l1_loss(restored.float(), t_restored.float())
            if feat_proj is not None:
                t_feat = feat_teacher.last_feats[feat_layer].float()
                s_feat = torch.nn.functional.interpolate(feat_proj(model.kd_feat.float()), size=t_feat.shape[-2:],
                                                         mode="bilinear", align_corners=False, antialias=True)
                loss = loss + w_feat * (1 - torch.nn.functional.cosine_similarity(s_feat, t_feat, dim=1)).mean()
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            step += 1
            if ema is not None:
                ema.update(model, step=step)
            loss_sum, batches = loss_sum + loss.detach(), batches + 1  # one GPU sync per epoch, not per step
        mean_loss = float((loss_sum / max(batches, 1)).item())
        chosen = ema.module if ema is not None else model
        if not val_s:  # all-data fit: keep the epoch chosen during selection (same schedule), else the last
            history.append({"epoch": epoch, "loss": mean_loss})
            best = {"composite": float("nan"), "epoch": epoch}
            torch.save(chosen.state_dict(), out_dir / "best.pt")
            if epoch == get(cfg, "train.save_epoch"):
                break
            continue
        # Select on the pipeline that will be scored: "static" runs the network at data.size and
        # resizes outputs (exactly what the Track 2 ONNX does); "full_res" applies restoration at
        # full resolution (what the Track 1 zip does).
        if get(cfg, "train.select_on", "full_res") == "static":
            val = evaluate_static(chosen, val_epoch, size, device)
        else:
            val = evaluate_torch(chosen, val_epoch, size, device)
        history.append({"epoch": epoch, "loss": mean_loss, **{k: val[k] for k in ("composite", "q_seg", "q_seg_raw", "psnr", "ssim")}})
        val["select"] = select_score(val, get(cfg, "train.select", "composite"))
        if val["select"] > best.get("select", float("-inf")) or get(cfg, "train.select") == "last":
            best = {**val, "epoch": epoch}
            torch.save(chosen.state_dict(), out_dir / "best.pt")
    model.load_state_dict(torch.load(out_dir / "best.pt", map_location=device, weights_only=True))
    (out_dir / "history.json").write_text(json.dumps(history, indent=1))
    calibration = _calibrate(cfg, model, val_s, size, device, out_dir)
    return model.cpu(), {"best": best, "history": history, "val_samples": [str(s.degraded) for s in val_s],
                         "class_weights": weights, "epochs_run": epochs_run, "stopped_early": stopped_early,
                         "ema": ema is not None, "kd_cache": str(kd_cache.root) if kd_cache else None,
                         "calibration": calibration,
                         "external": mix.sources if mix else None}
