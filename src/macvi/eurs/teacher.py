"""Track 1 teacher v2: frozen DINOv3 ViT + light multi-layer head + metadata-conditioned restorer.

Researched choice (docs/research.md "Track 1 teacher v2"): DINOv3 ViT-L/16 linear-probe ADE20k
mIoU is 54.9 against 44.8 for the ConvNeXt-S the first Kaggle teacher fine-tuned (0.269, Qseg
0.084). A frozen backbone needs no backward pass through 300M parameters, runs in fp16, can be
cached per image, and keeps its degradation-robust features intact on 20 training scenes.

- Backbone: timm ``vit_large_patch16_dinov3.lvd1689m`` (``model.backbone``; ViT-B/16 is the
  fallback), frozen, eval mode, fp16 autocast on CUDA. Four intermediate blocks (default: evenly
  spaced, 6/12/18/24 for ViT-L), layer-normed as in the DINOv2/v3 dense heads. RoPE makes any
  input size a multiple of 16 valid without position-embedding surgery.
- Head (DPT-lite, Ranftl et al. 2021): each block is projected to ``head_channels`` and
  reassembled to strides 4/8/16/32, a stride-4 conv stem on the image adds the fine detail ViT
  tokens lack (fish are ~65-280 px long at 1024), and the semantic FPN head fuses the pyramid.
- Restorer: per-cell 3x4 colour affine (as the student and deepdrivepl's "per-cell affine") on
  the fused + stem features, FiLM-conditioned on the file-name metadata (``macvi.eurs.meta``:
  water type, depth), plus a global-context term, then the NAFNet residual refiner at full
  resolution. Meta input is optional: unknown metadata is the all-zero vector (Track 2 never has it).
- Optional LoRA on the qkv projections of the last ``lora_blocks`` blocks (``lora_rank > 0``; off
  by default): the step after a frozen head plateaus (seg_research.md section 6).
- Optional Rein adapter (``rein_tokens > 0``; Wei et al., CVPR 2024, arXiv 2312.04265): ``rein_tokens`` learnable
  low-rank tokens per block refine each block's output through attention, the backbone stays frozen. In the paper's
  synthetic-to-real setting (DINOv2-L) it beat both the frozen backbone (+3.6 mIoU) and full fine-tuning (+2.2). Its
  output projection starts at zero, so training starts from the frozen model; blocks run with gradient checkpointing.

Checkpoints hold only what is not reproducible from timm: with ``pretrained: true`` the frozen
backbone is left out of ``state_dict`` (ViT-L is 1.2 GB) and comes back from the hub on load.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

from macvi.eurs import meta as meta_mod
from macvi.eurs.model import IMAGENET_MEAN, IMAGENET_STD, AffineRestorer, FPNHead, NAFRefiner, Refiner, conv_bn_relu

ARCH = "dinov3_vit"


def is_teacher(cfg: dict[str, Any]) -> bool:
    """True for a run config of this teacher (``model.arch: dinov3_vit``)."""
    return (cfg.get("model") or {}).get("arch") == ARCH


@dataclass
class TeacherConfig:
    arch: str = ARCH
    backbone: str = "vit_large_patch16_dinov3.lvd1689m"
    pretrained: bool = True
    backbone_kwargs: dict = field(default_factory=dict)  # e.g. {depth: 2, embed_dim: 32, num_heads: 2} in tests
    layers: list[int] | None = None  # 0-based blocks; None = 4 evenly spaced, ending with the last
    num_classes: int = 5
    head_channels: int = 256
    stem_channels: int = 64
    restore_channels: int = 64
    refine_channels: int = 32
    refine_type: str = "naf"
    use_meta: bool = True
    fp16: bool = True  # backbone under fp16 autocast on CUDA
    lora_rank: int = 0
    lora_blocks: int = 8
    lora_alpha: float = 16.0
    rein_tokens: int = 0  # Rein tokens per block (paper: 100); 0 = off
    rein_rank: int = 16
    seg_decoder: str = "fpn"  # "fpn", or "query": FPN logits + a Mask2Former-style query decoder (QueryHead)
    query_layers: int = 6
    queries_per_class: int = 4
    query_heads: int = 8


def default_layers(depth: int, n: int = 4) -> list[int]:
    return [round(depth * (i + 1) / n) - 1 for i in range(n)]


class LoRALinear(nn.Module):
    """y = W x + (alpha / r) * B A x, with B = 0 at start (Hu et al., 2021)."""

    def __init__(self, base: nn.Linear, rank: int, alpha: float):
        super().__init__()
        self.base = base
        self.lora_a = nn.Linear(base.in_features, rank, bias=False)
        self.lora_b = nn.Linear(rank, base.out_features, bias=False)
        nn.init.kaiming_uniform_(self.lora_a.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_b.weight)
        self.scale = alpha / rank

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + self.lora_b(self.lora_a(x)) * self.scale


class Rein(nn.Module):
    """Rein tokens for ``depth`` blocks: T_i = A_i B_i (``tokens`` x C, rank ``rank``); for a block output f,
    S = softmax(f T_i^T / sqrt(C)), f' = f + W_f(S T_i W_t + f) with W_f zero-initialised (identity at start).
    W_t and W_f are shared across blocks, as in the paper."""

    def __init__(self, dim: int, depth: int, tokens: int, rank: int):
        super().__init__()
        va, vb = math.sqrt(6 / (tokens + rank)), math.sqrt(6 / (rank + dim))
        self.a = nn.Parameter(torch.empty(depth, tokens, rank).uniform_(-va, va))
        self.b = nn.Parameter(torch.empty(depth, rank, dim).uniform_(-vb, vb))
        self.w_t = nn.Linear(dim, dim)
        self.w_f = nn.Linear(dim, dim)
        nn.init.zeros_(self.w_f.weight)
        nn.init.zeros_(self.w_f.bias)

    def forward(self, f: torch.Tensor, i: int) -> torch.Tensor:
        # float32 even under autocast: DINOv3 ViT-L block outputs reach ~1.6e5 (massive activations, measured
        # 14 Oct), past the float16 max 65504, so a float16 matmul gives inf and W_f(inf) = 0 * inf = NaN at step 0.
        with torch.autocast(f.device.type, enabled=False):
            f32 = f.float()
            t = self.a[i].float() @ self.b[i].float()
            s = torch.softmax(f32 @ t.t() / math.sqrt(f32.shape[-1]), -1)
            return (f32 + self.w_f(s @ self.w_t(t) + f32)).to(f.dtype)


class ReinBlock(nn.Module):
    """A frozen block followed by its Rein refinement (the shared ``Rein`` is held by the teacher, not here)."""

    def __init__(self, blk: nn.Module, rein: Rein, index: int):
        super().__init__()
        self.blk, self.index = blk, index
        object.__setattr__(self, "_rein", rein)  # not registered: its parameters live once, under ``teacher.rein``

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        return self._rein(self.blk(x, *args, **kwargs), self.index)


class Reassemble(nn.Module):
    """Stride-16 tokens -> ``stride`` (4, 8, 16 or 32): 1x1 projection, then a learned resample."""

    def __init__(self, cin: int, cout: int, stride: int):
        super().__init__()
        self.proj = nn.Conv2d(cin, cout, 1)
        resample = {4: nn.ConvTranspose2d(cout, cout, 4, stride=4), 8: nn.ConvTranspose2d(cout, cout, 2, stride=2),
                    16: nn.Identity(), 32: nn.Conv2d(cout, cout, 3, stride=2, padding=1)}
        self.resample = resample[stride]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.resample(self.proj(x))


class MetaAffineRestorer(nn.Module):
    """Per-cell 3x4 colour affine; FiLM by the metadata vector and a global-context term.

    Both conditioning paths start at zero, so the initial map is the identity (as the student's)."""

    def __init__(self, in_ch: int, mid: int, use_meta: bool):
        super().__init__()
        self.body = nn.Sequential(conv_bn_relu(in_ch, mid, 3), conv_bn_relu(mid, mid, 3))
        self.glob = nn.Conv2d(mid, mid, 1)
        nn.init.zeros_(self.glob.weight)
        nn.init.zeros_(self.glob.bias)
        self.film = None
        if use_meta:
            self.film = nn.Sequential(nn.Linear(meta_mod.DIM, mid), nn.ReLU(inplace=True), nn.Linear(mid, 2 * mid))
            nn.init.zeros_(self.film[-1].weight)
            nn.init.zeros_(self.film[-1].bias)
        self.coeffs = nn.Conv2d(mid, 12, 1)
        nn.init.zeros_(self.coeffs.weight)
        with torch.no_grad():
            self.coeffs.bias.copy_(torch.tensor([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0], dtype=torch.float32))

    def forward(self, feat: torch.Tensor, meta_vec: torch.Tensor) -> torch.Tensor:
        h = self.body(feat)
        h = h + self.glob(h.mean((2, 3), keepdim=True))
        if self.film is not None:
            gamma, beta = self.film(meta_vec.to(h.dtype)).chunk(2, dim=1)
            h = h * (1 + gamma[..., None, None]) + beta[..., None, None]
        return self.coeffs(h)


def sine_pos(h: int, w: int, dim: int, device) -> torch.Tensor:
    """(h*w, dim) 2-D sine position encoding (DETR), half the channels for y, half for x."""
    d = dim // 4
    freq = 1.0 / (10000 ** (torch.arange(d, device=device, dtype=torch.float32) / d))
    y = torch.arange(h, device=device, dtype=torch.float32)[:, None, None] * freq
    x = torch.arange(w, device=device, dtype=torch.float32)[None, :, None] * freq
    y, x = y.expand(h, w, d), x.expand(h, w, d)
    return torch.cat([y.sin(), y.cos(), x.sin(), x.cos()], -1).reshape(h * w, 4 * d)


class QueryHead(nn.Module):
    """Mask2Former-style transformer decoder (Cheng et al., CVPR 2022) over the head's pyramid.

    ``queries_per_class`` learnable queries per class run ``layers`` rounds of masked cross-attention to the
    stride-32/16/8 levels in turn (each query sees only where its previous mask is on), self-attention and an FFN;
    each query's mask is its embedding dotted with the stride-4 pixel features. With queries tied to classes no
    Hungarian matching is needed: a class's logit is the logsumexp of its queries' masks, so the plain
    CE + Lovasz loss trains it, and several queries can cover separate regions (fish, reef patches) of one class."""

    def __init__(self, c: int, num_classes: int, layers: int, per_class: int, heads: int):
        super().__init__()
        self.k, self.per = num_classes, per_class
        nq = num_classes * per_class
        self.query, self.query_pos, self.level_embed = nn.Embedding(nq, c), nn.Embedding(nq, c), nn.Embedding(3, c)
        self.inputs = nn.ModuleList(nn.Conv2d(c, c, 1) for _ in range(3))
        self.cross = nn.ModuleList(nn.MultiheadAttention(c, heads, batch_first=True) for _ in range(layers))
        self.self_attn = nn.ModuleList(nn.MultiheadAttention(c, heads, batch_first=True) for _ in range(layers))
        self.ffn = nn.ModuleList(nn.Sequential(nn.Linear(c, 4 * c), nn.ReLU(inplace=True), nn.Linear(4 * c, c))
                                 for _ in range(layers))
        self.norms = nn.ModuleList(nn.ModuleList(nn.LayerNorm(c) for _ in range(3)) for _ in range(layers))
        self.out_norm = nn.LayerNorm(c)
        self.mask_embed = nn.Sequential(nn.Linear(c, c), nn.ReLU(inplace=True), nn.Linear(c, c), nn.ReLU(inplace=True),
                                        nn.Linear(c, c))
        self.pixel = nn.Conv2d(c, c, 1)
        self.heads = heads

    def _masks(self, q: torch.Tensor, pix: torch.Tensor, size: tuple[int, int]) -> tuple[torch.Tensor, torch.Tensor]:
        m = torch.einsum("bqc,bchw->bqhw", self.mask_embed(self.out_norm(q)), pix)
        a = F.interpolate(m.detach(), size=size, mode="bilinear", align_corners=False).flatten(2) < 0  # True = blocked
        a[a.all(-1)] = False  # a query whose mask is empty attends everywhere (as in Mask2Former)
        return m, a.repeat_interleave(self.heads, 0)

    def forward(self, levels: Sequence[torch.Tensor], fused: torch.Tensor) -> torch.Tensor:
        """Class logits (B, K, H/4, W/4) from ``levels`` [stride 8, 16, 32] and the stride-4 ``fused`` map."""
        b = fused.shape[0]
        srcs = [levels[2], levels[1], levels[0]]  # coarse to fine, as Mask2Former's round robin
        mem, pos = [], []
        for i, (proj, s) in enumerate(zip(self.inputs, srcs)):
            mem.append(proj(s).flatten(2).transpose(1, 2) + self.level_embed.weight[i])
            pos.append(sine_pos(*s.shape[-2:], s.shape[1], s.device)[None])
        pix = self.pixel(fused)
        q = self.query.weight[None].expand(b, -1, -1)
        qpos = self.query_pos.weight[None]
        m, attn = self._masks(q, pix, srcs[0].shape[-2:])
        for i, (cross, sa, ffn, (n1, n2, n3)) in enumerate(zip(self.cross, self.self_attn, self.ffn, self.norms)):
            j = i % 3
            q = n1(q + cross(q + qpos, mem[j] + pos[j], mem[j], attn_mask=attn, need_weights=False)[0])
            q = n2(q + sa(q + qpos, q + qpos, q, need_weights=False)[0])
            q = n3(q + ffn(q))
            m, attn = self._masks(q, pix, srcs[(i + 1) % 3].shape[-2:])
        return torch.logsumexp(m.view(b, self.k, self.per, *m.shape[-2:]), 2)


class DinoTeacher(nn.Module):
    uses_meta = True  # prediction helpers pass file-name metadata (macvi.eurs.evaluate)

    def __init__(self, cfg: TeacherConfig = TeacherConfig()):
        super().__init__()
        self.cfg = cfg
        self.backbone = timm.create_model(cfg.backbone, pretrained=cfg.pretrained, num_classes=0, **cfg.backbone_kwargs)
        self.backbone.requires_grad_(False)
        depth, dim = len(self.backbone.blocks), self.backbone.embed_dim
        self.layers = list(cfg.layers) if cfg.layers else default_layers(depth)
        if len(self.layers) != 4 or not all(0 <= i < depth for i in self.layers):
            raise ValueError(f"model.layers must be 4 block indices in [0, {depth}), got {self.layers}")
        if cfg.lora_rank > 0:
            for blk in self.backbone.blocks[max(0, depth - cfg.lora_blocks):]:
                blk.attn.qkv = LoRALinear(blk.attn.qkv, cfg.lora_rank, cfg.lora_alpha)
        self.rein = Rein(dim, depth, cfg.rein_tokens, cfg.rein_rank) if cfg.rein_tokens > 0 else None
        if self.rein is not None:
            self.backbone.blocks = nn.ModuleList([ReinBlock(b, self.rein, i) for i, b in enumerate(self.backbone.blocks)])
            self.backbone.set_grad_checkpointing(True)
        c, s = cfg.head_channels, cfg.stem_channels
        self.reassemble = nn.ModuleList(Reassemble(dim, c, st) for st in (4, 8, 16, 32))
        self.stem = nn.Sequential(conv_bn_relu(3, s // 2, 3), nn.MaxPool2d(2), conv_bn_relu(s // 2, s, 3), nn.MaxPool2d(2),
                                  conv_bn_relu(s, s, 3))
        self.seg_head = FPNHead([c + s, c, c, c], c, cfg.num_classes)
        if cfg.seg_decoder not in ("fpn", "query"):
            raise ValueError(f"unknown seg_decoder {cfg.seg_decoder!r}")
        self.query_head = (QueryHead(c, cfg.num_classes, cfg.query_layers, cfg.queries_per_class, cfg.query_heads)
                           if cfg.seg_decoder == "query" else None)
        self.restorer = MetaAffineRestorer(c + s, cfg.restore_channels, cfg.use_meta)
        refiners = {"plain": Refiner, "naf": NAFRefiner}
        if cfg.refine_type not in refiners:
            raise ValueError(f"unknown refine_type {cfg.refine_type!r}")
        self.refiner = refiners[cfg.refine_type](cfg.refine_channels) if cfg.refine_channels > 0 else None
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1), persistent=False)

    # ------------------------------------------------------------------ frozen backbone

    def train(self, mode: bool = True):
        super().train(mode)
        self.backbone.eval()  # frozen: no dropout/drop-path, whatever the head does
        return self

    def backbone_features(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Four stride-16 feature maps (B, C, H/16, W/16) of the normalised input ``x``, in float32."""
        h, w = x.shape[-2:]
        hh, ww = max(16, round(h / 16) * 16), max(16, round(w / 16) * 16)
        if (hh, ww) != (h, w):
            x = F.interpolate(x, size=(hh, ww), mode="bilinear", align_corners=False)
        grad = (self.cfg.lora_rank > 0 or self.rein is not None) and self.training and torch.is_grad_enabled()
        amp = x.is_cuda and self.cfg.fp16
        with torch.set_grad_enabled(grad), torch.autocast("cuda", dtype=torch.float16, enabled=amp):
            feats = self.backbone.forward_intermediates(x, indices=self.layers, norm=True, output_fmt="NCHW",
                                                        intermediates_only=True)
        return [f.float() for f in feats]

    def normalise(self, image: torch.Tensor) -> torch.Tensor:
        return (image - self.mean) / self.std

    # ------------------------------------------------------------------ heads

    def meta_vector(self, meta: Any, batch: int, device) -> torch.Tensor:
        """(B, DIM) from None, a tensor, or a list of ``Meta | None``."""
        if isinstance(meta, torch.Tensor):
            return meta.to(device=device, dtype=torch.float32)
        if meta is None or not self.cfg.use_meta:
            return torch.zeros(batch, meta_mod.DIM, device=device)
        return torch.from_numpy(np.stack([meta_mod.encode(m) for m in meta])).to(device)

    def heads(self, x: torch.Tensor, meta: Any = None,
              feats: Sequence[torch.Tensor] | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """(affine coefficients at stride 4, logits at stride 4) for the image ``x`` in [0, 1].

        ``feats`` are cached ``backbone_features`` of this exact input (feature-cache training)."""
        xn = self.normalise(x)
        if feats is None:
            feats = self.backbone_features(xn)
        self.last_feats = feats  # read by the student's feature distillation (train ``distill.w_feat``)
        levels = [r(f) for r, f in zip(self.reassemble, feats)]
        stem = F.interpolate(self.stem(xn), size=levels[0].shape[-2:], mode="bilinear", align_corners=False)
        levels[0] = torch.cat([levels[0], stem], 1)
        logits, fused = self.seg_head(levels)
        if self.query_head is not None:  # the query decoder adds region-level context to the FPN's per-pixel logits
            logits = logits + self.query_head(levels[1:], fused)
        coeffs = self.restorer(torch.cat([fused, stem], 1), self.meta_vector(meta, x.shape[0], x.device))
        return coeffs, logits

    def restore(self, coeffs: torch.Tensor, image: torch.Tensor) -> torch.Tensor:
        affine = AffineRestorer.apply(coeffs, image)
        return self.refiner(image, affine) if self.refiner is not None else affine

    def forward(self, x: torch.Tensor, meta: Any = None,
                feats: Sequence[torch.Tensor] | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        coeffs, logits = self.heads(x, meta, feats)
        restored = self.restore(coeffs, x)
        return restored, F.interpolate(logits, size=x.shape[-2:], mode="bilinear", align_corners=False)

    @torch.no_grad()
    def predict_full_res(self, image: torch.Tensor, size: tuple[int, int], meta: Any = None) -> tuple[torch.Tensor, torch.Tensor]:
        """Heads at the model ``size``; restoration and segmentation at the image's own resolution."""
        small = F.interpolate(image, size=size, mode="bilinear", align_corners=False)
        coeffs, logits = self.heads(small, meta)
        restored = self.restore(coeffs, image)
        return restored, F.interpolate(logits, size=image.shape[-2:], mode="bilinear", align_corners=False)

    # ------------------------------------------------------------------ checkpoints without the frozen backbone

    def _external(self, key: str, prefix: str = "") -> bool:
        """A frozen backbone tensor that timm reproduces exactly (pretrained weights, not LoRA)."""
        return self.cfg.pretrained and key.startswith(prefix + "backbone.") and ".lora_" not in key

    def state_dict(self, *args, **kwargs):
        sd = super().state_dict(*args, **kwargs)
        prefix = kwargs.get("prefix", args[1] if len(args) > 1 else "")
        for k in [k for k in sd if self._external(k, prefix)]:
            del sd[k]
        return sd

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
        if self.cfg.pretrained:  # the backbone keys a checkpoint left out come from the hub weights
            full = {k: v for k, v in super().state_dict().items() if self._external(k)}
            full.update(state_dict)
            state_dict = full
        return super().load_state_dict(state_dict, strict=strict, assign=assign)


def build(cfg: dict[str, Any]) -> DinoTeacher:
    return DinoTeacher(TeacherConfig(**cfg.get("model", {})))


def trainable_parameters(model: nn.Module) -> list[nn.Parameter]:
    return [p for p in model.parameters() if p.requires_grad]
