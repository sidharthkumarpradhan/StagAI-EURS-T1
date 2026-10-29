"""One-off T4 benchmark of the Track 1 teacher v2 (``python -m macvi.remote.job bench``).

Measures what the Kaggle profile must be sized from, inside a hard time budget (default 50 min of
a 60 min kernel): frozen DINOv3 backbone forward speed and memory per input size, head training
step speed (online and from cached features), data-loader speed on the real train split, then a
short real training run on held-out scenes and its single-image vs per-scene-fused scores.

Everything goes to ``out/job.json`` (also on failure, with the stage that failed).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import traceback
from dataclasses import asdict
from pathlib import Path
from typing import Any

PROFILE = {"data.size": [768, 1024], "train.batch_size": 4, "train.num_workers": 4, "data.val_max_per_scene": 3,
           "train.samples_per_epoch": 400}


def _gpu_info() -> dict[str, Any]:
    import torch

    info = {"torch": torch.__version__, "cuda": torch.version.cuda, "gpus": torch.cuda.device_count(), "cpus": os.cpu_count()}
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        info.update(gpu=p.name, gpu_mem_gb=p.total_memory / 2**30)
    try:
        info["nvidia_smi"] = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True).stdout.strip()
    except OSError:
        pass
    return info


def _device() -> str:
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


def _sync() -> None:
    import torch
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _reset_mem() -> None:
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def _peak_gb() -> float:
    import torch
    return torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else 0.0


def _timed(fn, warmup: int = 2, iters: int = 8) -> float:
    """Seconds per call (CUDA-synchronised)."""
    for _ in range(warmup):
        fn()
    _sync()
    t = time.time()
    for _ in range(iters):
        fn()
    _sync()
    return (time.time() - t) / iters


def backbone_speed(model, sizes: list[list[int]], batches: list[int]) -> list[dict[str, Any]]:
    import torch

    rows = []
    for h, w in sizes:
        for b in batches:
            _reset_mem()
            x = torch.rand(b, 3, h, w, device=_device())
            try:
                with torch.no_grad():
                    s = _timed(lambda: model.backbone_features(model.normalise(x)))
                rows.append({"size": [h, w], "batch": b, "img_per_s": b / s, "peak_gb": _peak_gb()})
            except torch.cuda.OutOfMemoryError:
                rows.append({"size": [h, w], "batch": b, "oom": True})
            print("backbone", rows[-1], flush=True)
    return rows


def train_step_speed(model, size: list[int], batch: int, cached: bool) -> dict[str, Any]:
    """Head training step (fwd + bwd + AdamW) on random tensors; ``cached`` skips the backbone."""
    import torch

    from macvi.eurs import teacher as teacher_mod
    from macvi.eurs.teacher_train import teacher_loss

    _reset_mem()
    h, w, dev = *size, _device()
    x, clean = torch.rand(batch, 3, h, w, device=dev), torch.rand(batch, 3, h, w, device=dev)
    mask = torch.randint(0, 5, (batch, h, w), device=dev)
    with torch.no_grad():
        feats = [f.half().float() for f in model.backbone_features(model.normalise(x))] if cached else None
    opt = torch.optim.AdamW(teacher_mod.trainable_parameters(model), lr=1e-4)
    scaler = torch.amp.GradScaler(enabled=dev == "cuda")
    model.train()

    def step():
        with torch.autocast(dev, enabled=dev == "cuda"):
            restored, logits = model(x, feats=feats)
        loss = teacher_loss(restored.float(), clean, logits.float(), mask, {})
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()

    try:
        s = _timed(step, warmup=2, iters=6)
        out = {"size": size, "batch": batch, "cached": cached, "samples_per_s": batch / s, "peak_gb": _peak_gb()}
    except torch.cuda.OutOfMemoryError:
        out = {"size": size, "batch": batch, "cached": cached, "oom": True}
    opt.zero_grad(set_to_none=True)
    print("train step", out, flush=True)
    return out


def loader_speed(cfg: dict, train_root: Path, n_batches: int = 30) -> dict[str, Any]:
    """Samples per second the configured DataLoader delivers on the real data (no GPU work)."""
    from torch.utils.data import DataLoader

    from macvi.common.config import get
    from macvi.eurs.augment import AugConfig
    from macvi.eurs.data import Layout, discover
    from macvi.eurs.teacher_train import TeacherDataset

    samples = discover(train_root, Layout(**cfg["data"]["layout"]))
    aug = AugConfig(**{k: tuple(v) if isinstance(v, list) else v for k, v in cfg.get("aug", {}).items()})
    ds = TeacherDataset(samples, tuple(get(cfg, "data.size")), aug, 0.15, 0.2)
    loader = DataLoader(ds, batch_size=get(cfg, "train.batch_size"), shuffle=True, num_workers=get(cfg, "train.num_workers"))
    it, t = iter(loader), None
    for i in range(n_batches):
        next(it)
        if i == 2:
            t = time.time()
    rate = (n_batches - 3) * get(cfg, "train.batch_size") / (time.time() - t)
    print("loader samples/s", rate, flush=True)
    return {"samples_per_s": rate, "workers": get(cfg, "train.num_workers")}


def run(params: dict[str, Any], start: float, out: Path = Path("/kaggle/working/out"), tmp: Path = Path("/kaggle/tmp"),
        input_root: Path = Path("/kaggle/input")) -> dict[str, Any]:
    import torch

    from macvi.common import config
    from macvi.eurs import fusion, layout, scoring, teacher
    from macvi.eurs.data import cap_per_scene, discover, split_by_scene
    from macvi.eurs.train import train
    from macvi.remote import job

    out.mkdir(parents=True, exist_ok=True)
    deadline = start + 60 * float(params.get("budget_minutes", 50))
    res: dict[str, Any] = {"status": "failed", "stage": "info", "sha": params.get("sha")}

    def save():
        res["elapsed_hours"] = (time.time() - start) / 3600
        (out / "job.json").write_text(json.dumps(res, indent=1, default=str))

    try:
        res["env"] = _gpu_info()
        if params.get("require_gpu", True) and not torch.cuda.is_available():
            raise RuntimeError("no CUDA GPU visible")
        dev = _device()
        res["stage"] = "data"
        t = time.time()
        train_root = job.get_data(input_root, tmp, job._sh)
        found, summary = layout.infer(train_root)
        res["data"] = {"seconds": time.time() - t, "layout": asdict(found), "summary": summary}
        save()

        base = config.with_overrides(config.load(Path(__file__).resolve().parents[3] / "configs/eurs/track1_vit.yaml"),
                                     {**PROFILE, **params.get("profile", {}), "data.layout": asdict(found)})
        res["stage"] = "weights"
        t = time.time()
        model = teacher.build(base).to(dev).eval()
        res["weights_seconds"] = time.time() - t
        res["params_m"] = {"backbone": sum(p.numel() for p in model.backbone.parameters()) / 1e6,
                           "trainable": sum(p.numel() for p in teacher.trainable_parameters(model)) / 1e6}
        res["stage"] = "backbone"
        res["backbone"] = backbone_speed(model, params.get("sizes", [[512, 704], [640, 864], [768, 1024]]),
                                         params.get("batches", [1, 4]))
        save()
        res["stage"] = "train-step"
        res["train_step"] = [train_step_speed(model, s, b, c) for s, b, c in
                             params.get("steps", [([768, 1024], 4, False), ([768, 1024], 2, False),
                                                  ([768, 1024], 4, True), ([640, 864], 4, False)])]
        save()
        res["stage"] = "loader"
        res["loader"] = loader_speed(base, train_root, int(params.get("loader_batches", 30)))
        del model
        _reset_mem()
        save()

        res["stage"] = "train"
        fits = sorted((r for r in res["train_step"] if not r["cached"] and not r.get("oom")),
                      key=lambda r: (r["size"] == base["data"]["size"], r["batch"]), reverse=True)
        if not fits:
            raise RuntimeError("every online training step ran out of memory")
        online = fits[0]  # the profile size at the largest batch that fits, else the largest that fits
        base = config.with_overrides(base, {"data.size": online["size"], "train.batch_size": online["batch"]})
        rate = min(online["samples_per_s"], res["loader"]["samples_per_s"])
        eval_reserve = 60 * float(params.get("eval_minutes", 9))
        train_budget = max(300.0, deadline - eval_reserve - time.time())
        per_epoch = base["train"]["samples_per_epoch"]
        epochs = max(2, int(train_budget * rate * 0.8 / per_epoch))  # 20% for per-epoch selection and loading
        cfg = config.with_overrides(base, {"train.epochs": epochs})
        res["train_plan"] = {"rate_estimate": rate, "budget_s": train_budget, "epochs": epochs}
        save()
        model, info = train(cfg, train_root, tmp / "bench_run", deadline=deadline - eval_reserve)
        res["train"] = {k: info[k] for k in ("best", "history", "epochs_run", "stopped_early", "perf")}
        save()

        res["stage"] = "eval"
        t = time.time()
        _, val = split_by_scene(discover(train_root, found), cfg["data"]["val_fraction"], cfg["train"]["seed"],
                                cfg["data"]["fold"])
        val = cap_per_scene(val, int(params.get("eval_per_scene", 200)), 0)  # all 200 versions: fusion needs them
        variants = {"consistency": fusion.FusionConfig(), "mean": fusion.FusionConfig(restore="mean"),
                    "top40": fusion.FusionConfig(top_k=40), "seg_confidence": fusion.FusionConfig(seg="confidence")}
        single, fused = fusion.evaluate_variants(model.to(dev), val, tuple(cfg["data"]["size"]), variants, device=dev,
                                                 tta=False)
        keys = ("composite", "q_seg", "miou", "f1", "psnr", "ssim", "sharpness", "uciqe")
        res["eval"] = {"seconds": time.time() - t, "images": len(val), "scenes": sorted({s.scene for s in val}),
                       "single": {k: single[k] for k in keys}, **{n: {k: m[k] for k in keys} for n, m in fused.items()},
                       "components": {n: scoring.components(m) for n, m in {"single": single, **fused}.items()},
                       "per_class_iou_single": {k: v for k, v in single.items() if k.startswith("iou_")}}
        res["status"], res["stage"] = "ok", "done"
    except BaseException as exc:  # noqa: BLE001 - always write job.json
        traceback.print_exc()
        res["error"] = repr(exc)
    finally:
        save()
    print(json.dumps({k: res.get(k) for k in ("status", "stage", "error", "elapsed_hours")}), flush=True)
    return res


if __name__ == "__main__":
    run(json.loads(Path(sys.argv[1]).read_text()), time.time())
