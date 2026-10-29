"""Teacher -> student distillation for Track 2.

The Track 1 teacher (no size cap) is frozen; the student is trained on the ground truth plus:
- KL divergence between temperature-softened class distributions (Hinton et al., 2015),
  per pixel; scaled by T^2 as usual.
- optional channel-wise distillation (CWD, Shu et al., ICCV 2021, arXiv:2011.13256): each class
  map is turned into a distribution over pixels (softmax over space at temperature T) and the
  student matches the teacher's with KL. Every class channel weighs the same, so the rare
  elasmobranch/other maps count as much as background. The paper's best logit setting is T=4.
- optional L1 to the teacher's restoration (off by default: restoration has clean ground truth).

Teacher targets come either from the online teacher or from a ``LogitCache``: the teacher runs
once per training image (full frame, at the teacher's own training size), its logits are stored
at stride 4 in float16, and the dataset warps them with the sample's crop/zoom/flip
(``augment.warp_logits``). Segmentation labels do not depend on the degradation, so a cached
target stays valid under re-degradation. ``distill.teacher_input: clean`` feeds the teacher the
pixel-aligned clean twin of every degraded sample (learning with privileged information,
Lopez-Paz et al., ICLR 2016): its targets do not suffer from the degradation the student sees.

``distill.teacher`` may list several run dirs separated by commas: the teachers' logits (and restored images) are
averaged, ensemble distillation (Hinton et al., 2015, sec. 2; Lin et al., NeurIPS 2020). Every member must hold out the
student's validation scenes.
"""

from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from macvi.common.config import with_overrides

TEACHER_INPUTS = ("degraded", "clean")


def _match(student: torch.Tensor, teacher: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Bring both logit maps to one resolution: pool the student down to a coarser teacher map
    (area average), otherwise resize the teacher bilinearly to the student's."""
    if teacher.shape[-2:] == student.shape[-2:]:
        return student, teacher
    if teacher.shape[-2] <= student.shape[-2] and teacher.shape[-1] <= student.shape[-1]:
        return F.adaptive_avg_pool2d(student, teacher.shape[-2:]), teacher
    return student, F.interpolate(teacher, size=student.shape[-2:], mode="bilinear", align_corners=False)


def kd_loss(student_logits: torch.Tensor, teacher_logits: torch.Tensor, temperature: float = 2.0) -> torch.Tensor:
    student_logits, teacher_logits = _match(student_logits, teacher_logits)
    t = temperature
    log_p = F.log_softmax(student_logits / t, dim=1)
    q = F.softmax(teacher_logits / t, dim=1)
    return F.kl_div(log_p, q, reduction="none").sum(1).mean() * t * t


def cwd_loss(student_logits: torch.Tensor, teacher_logits: torch.Tensor, temperature: float = 4.0) -> torch.Tensor:
    """Channel-wise distillation: KL between per-channel spatial distributions, mean over (B, C), x T^2."""
    student_logits, teacher_logits = _match(student_logits, teacher_logits)
    b, c = student_logits.shape[:2]
    t = temperature
    s = F.log_softmax(student_logits.reshape(b * c, -1) / t, dim=1)
    q = F.softmax(teacher_logits.reshape(b * c, -1) / t, dim=1)
    return F.kl_div(s, q, reduction="sum") / (b * c) * t * t


def teacher_dirs(spec: str | Path) -> list[str]:
    """Run dirs named by a ``distill.teacher`` value (one path, or several separated by commas)."""
    return [d.strip() for d in str(spec).split(",") if d.strip()]


class TeacherEnsemble(torch.nn.Module):
    """Frozen teachers whose outputs ``run_teacher`` averages (logits at the first member's resolution)."""

    def __init__(self, members: list[torch.nn.Module]):
        super().__init__()
        self.members = torch.nn.ModuleList(members)


def load_teacher(run_dir: str | Path, device: str):
    if len(teacher_dirs(run_dir)) > 1:
        return TeacherEnsemble([_load_one(d, device) for d in teacher_dirs(run_dir)])
    return _load_one(teacher_dirs(run_dir)[0], device)


def _load_one(run_dir: str | Path, device: str):
    """Rebuild a trained model from a run directory (``config.json`` + ``best.pt``), frozen.

    ``input_size`` records the (H, W) it was trained at; ``run_teacher`` feeds it that size.

    A teacher v2 checkpoint leaves its pretrained backbone out (``macvi.eurs.teacher``), so that
    backbone is rebuilt from timm's weights instead of being switched to random initialisation."""
    from macvi.eurs import teacher
    from macvi.eurs.train import build_model  # local import avoids a cycle

    run_dir = Path(run_dir)
    cfg = json.loads((run_dir / "config.json").read_text())
    if not teacher.is_teacher(cfg):
        cfg = with_overrides(cfg, {"model.pretrained": False})
    model = build_model(cfg)
    model.load_state_dict(torch.load(run_dir / "best.pt", map_location=device, weights_only=True))
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    model.input_size = tuple(cfg.get("data", {}).get("size", [384, 768]))
    return model


@torch.no_grad()
def run_teacher(teacher, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Teacher (restored, logits) for a batch. Inputs at another size than the teacher was trained
    at are resized to its size first (the teacher then sees what it saw in training); the restored
    image is resized back to the batch size, the logits stay at the teacher's resolution."""
    if isinstance(teacher, TeacherEnsemble):
        outs = [run_teacher(m, x) for m in teacher.members]
        shape = outs[0][1].shape[-2:]
        logits = [o[1] if o[1].shape[-2:] == shape else F.interpolate(o[1], size=shape, mode="bilinear", align_corners=False)
                  for o in outs]
        return torch.stack([o[0] for o in outs]).mean(0), torch.stack(logits).mean(0)
    size = tuple(getattr(teacher, "input_size", x.shape[-2:]))
    if tuple(x.shape[-2:]) == size:
        return teacher(x)
    restored, logits = teacher(F.interpolate(x, size=size, mode="bilinear", align_corners=False, antialias=True))
    return F.interpolate(restored, size=x.shape[-2:], mode="bilinear", align_corners=False), logits


class LogitCache:
    """Teacher logits per source image at stride 4, float16 ``.npy`` files in ``root``.

    ``root`` defaults to a folder under the system temp directory keyed by the teacher checkpoint
    and its input kind, so student runs of one session share it and it never lands in run dirs
    (which are saved to the state dataset).
    """

    def __init__(self, root: Path, stride: int = 4):
        self.root, self.stride = Path(root), stride
        self.root.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def default_root(teacher_dir: str | Path, teacher_input: str) -> Path:
        parts = []
        for d in teacher_dirs(teacher_dir):
            ckpt = Path(d) / "best.pt"
            stat = ckpt.stat()
            parts.append(f"{ckpt.resolve()}|{stat.st_size}|{stat.st_mtime_ns}")
        key = "|".join(parts) + f"|{teacher_input}"
        return Path(tempfile.gettempdir()) / "macvi_kd_cache" / hashlib.sha1(key.encode(), usedforsecurity=False).hexdigest()[:16]

    def path(self, image: str | Path) -> Path:
        return self.root / (hashlib.sha1(str(Path(image)).encode(), usedforsecurity=False).hexdigest()[:20] + ".npy")

    def load(self, image: str | Path) -> np.ndarray:
        return np.load(self.path(image))

    @torch.no_grad()
    def build(self, teacher, images: list[Path], device: str) -> int:
        """Run the teacher once per image not cached yet; returns how many were computed."""
        from macvi.eurs.data import read_rgb

        size = tuple(getattr(teacher, "input_size", (384, 768)))
        done = 0
        for image in sorted(set(map(Path, images))):
            target = self.path(image)
            if target.exists():
                continue
            rgb = read_rgb(image)
            h, w = rgb.shape[:2]
            x = cv2.resize(rgb, (size[1], size[0]), interpolation=cv2.INTER_AREA)
            x = torch.from_numpy(x).permute(2, 0, 1)[None].float().div(255).to(device)
            with torch.autocast(device_type="cuda", enabled=device == "cuda"):
                _, logits = teacher(x)
            cache_hw = (max(1, -(-h // self.stride)), max(1, -(-w // self.stride)))
            logits = F.interpolate(logits.float(), size=cache_hw, mode="bilinear", align_corners=False)
            tmp = target.with_suffix(".tmp.npy")
            np.save(tmp, logits[0].cpu().numpy().astype(np.float16))
            tmp.replace(target)  # atomic: a crash never leaves a truncated cache entry
            done += 1
        return done
