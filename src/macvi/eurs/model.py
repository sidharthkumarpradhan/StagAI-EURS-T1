"""Compact two-head network for EURS Track 2 (<=1M params, <=12 GFLOPs, static ONNX).

Design (researched choices):
- Encoder: a timm ImageNet-pretrained mobile backbone (``features_only``). MobileNetV3-Small
  0.75 is ~0.57M params; the encoder is a config knob so the loop can search alternatives
  (lcnet_075, mobilenetv4_conv_small_050, tinynet_e, ...).
- Segmentation head: LR-ASPP from "Searching for MobileNetV3" (Howard et al., 2019), the
  same head torchvision uses for its mobile segmentation models.
- Restoration head: per-pixel 3x4 colour affine transform predicted on a coarse grid and
  bilinearly upsampled (in the spirit of HDRNet, Gharbi et al., 2017). Colour cast and haze
  are low-frequency, so a coarse transform suffices; applying it to the input pixels keeps
  every edge (high sharpness ratio). Resize/Mul/ReduceSum are not Conv/Gemm/MatMul, so the
  full-resolution part costs no counted FLOPs.
- Compute headroom: the student uses ~1 of the 12 allowed GFLOPs while the 1M-parameter cap binds,
  so three knobs trade FLOPs for segmentation detail at ~0 parameters: ``output_stride`` (dilated
  encoder, the published LR-ASPP setting is 16), ``fine_channels`` (stride-4 fusion) and
  ``flip_tta`` (mirror averaging inside the exported graph). See docs/research.md.
- Full-resolution I/O (student v2): the graph input is the image at the scoring resolution; the
  heads run on an in-graph downsampled copy (``internal_scale``), the coefficient grid is upsampled
  (bilinear or ``guided``, edge-aware) and applied to the full-resolution pixels, and the stride-4
  logits are upsampled in-graph. Every Track 2 board entry has a sharpness ratio of 0.04-0.28,
  the signature of outputs computed at low resolution and upsampled. ``global_colour`` adds one
  per-image 3x4 colour matrix (a per-image matrix fitted to the ground truth reaches ~28 dB median
  on the train pairs, docs/research.md).

Track 1 (no size cap) uses the same class as a large "teacher": a DINOv3 ConvNeXt encoder
(self-supervised on 1.7B images, robust to degradation), an FPN decoder (``decoder="fpn"``),
and an optional full-resolution residual refiner on top of the affine restorer. The teacher
also supervises the Track 2 student through distillation (``macvi.eurs.distill``).

Input: (B, 3, H, W) RGB in [0, 1]. Outputs: ``restored_image`` (B, 3, H, W) in [0, 1] and
``seg_logits`` (B, num_classes, H, W) raw logits, matching the Track 2 contract.
"""

from __future__ import annotations

from dataclasses import dataclass

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass
class ModelConfig:
    encoder: str = "mobilenetv3_small_075"
    pretrained: bool = True
    num_classes: int = 5
    seg_channels: int = 96
    restore_channels: int = 32
    seg_stride4: bool = True  # extra stride-4 logits for finer boundaries (cheap in params)
    decoder: str = "lraspp"   # "lraspp" (Track 2 student) or "fpn" (Track 1 teacher)
    refine_channels: int = 0  # >0: full-resolution residual refiner after the affine map (Track 1)
    refine_type: str = "plain"  # "plain" (3 convs) or "naf" (compact NAFNet U-Net, Chen et al. 2022)
    # Track 2 compute-headroom knobs (docs/research.md "Track 2 compute headroom"): each spends
    # FLOPs, which the student barely uses, for segmentation detail at (almost) no parameters.
    output_stride: int = 32   # 16/8: dilate the last encoder stages (timm), as LR-ASPP does in MobileNetV3
    fine_channels: int = 0    # >0: 3x3 fusion of stride-4 features with the logits (lraspp only)
    flip_tta: bool = False    # eval/export only: average heads over the image and its mirror (2x FLOPs)
    # Full-resolution I/O (docs/research.md "Track 2 student v2"): the graph takes the image at the
    # scoring resolution, the encoder/decoder run on an in-graph downsampled copy, and restoration
    # and segmentation are applied/upsampled back at full resolution (Resize/pooling are not counted).
    internal_scale: float = 1.0      # heads run at round(H*s) x round(W*s); 1.0 = at the input size
    coeff_upsample: str = "bilinear"  # "bilinear" or "guided" (fast guided filter, edge-aware, 0 params)
    guide_radius: int = 1            # guided: box radius on the coefficient grid
    guide_eps: float = 1e-3          # guided: regulariser; larger is closer to bilinear
    global_colour: bool = False      # per-image 3x4 colour matrix from pooled deep features, added to the grid
    # Training-only deep supervision (PSPNet aux loss, Zhao et al. CVPR 2017; BiSeNetV2 booster, Yu et al. IJCV 2021):
    # a 3x3 conv + classifier on the deepest features, weighted by train.w_aux; eval/export never run it (0 ONNX params).
    aux_channels: int = 0
    # Track 2 input-side levers (docs/research.md "Track 2 arena wave AX"): what the segmentation encoder sees.
    input_norm: str = "imagenet"  # "instance": per-image, per-channel standardisation (IBN-Net-style colour invariance, 0 params)
    pre_restorer: str = ""        # path to a .py whose build() returns a frozen restorer; the encoder sees its output


def downsample(x: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    """Area-like downsampling that exports to TensorRT-friendly, uncounted ops.

    Equal integer factors >= 3 use average pooling (exact area). Factor 2 uses bilinear Resize with
    half-pixel centres, which samples midway between pixel pairs and so equals 2x2 averaging; other
    factors fall back to bilinear Resize.
    """
    h, w = x.shape[-2:]
    oh, ow = int(size[0]), int(size[1])
    if (h, w) == (oh, ow):
        return x
    k = h // oh
    if k >= 3 and h == k * oh and w == k * ow:
        return F.avg_pool2d(x, k)
    return F.interpolate(x, size=(oh, ow), mode="bilinear", align_corners=False)


def internal_size(h: int, w: int, scale: float) -> tuple[int, int]:
    # int(): torch 2.5's ONNX tracer passes shapes as 0-d tensors, which have no __round__ (V100, 6 Oct 2026);
    # the export is static-shape, so freezing them is correct
    return max(1, round(int(h) * scale)), max(1, round(int(w) * scale))


def box_filter(x: torch.Tensor, radius: int) -> torch.Tensor:
    """Mean over a (2r+1)^2 window, border windows averaged over valid pixels only."""
    return F.avg_pool2d(x, 2 * radius + 1, stride=1, padding=radius, count_include_pad=False)


def guided_upsample(coeffs: torch.Tensor, image: torch.Tensor, radius: int = 1, eps: float = 1e-3) -> torch.Tensor:
    """Edge-aware upsampling of a coarse coefficient grid to ``image`` resolution.

    Fast guided filter (He et al., TPAMI 2013; Wu et al., "Fast End-to-End Trainable Guided
    Filter", CVPR 2018) with the image's grey level as guide: at the coarse grid each coefficient
    map is fitted locally as ``a * guide + b``; ``a`` and ``b`` are upsampled bilinearly and applied
    to the full-resolution guide, so coefficient changes follow image edges (depth edges) instead of
    blurring across them. Where a coefficient is locally constant ``a`` is 0 and this is bilinear.
    Guide and coefficients are centred per image first, which keeps the variance terms out of FP16
    cancellation. All ops are pooling, Resize and elementwise: 0 parameters, 0 counted FLOPs.
    """
    grey = image.mean(1, keepdim=True)
    g = downsample(grey, coeffs.shape[-2:])
    g_mean, q_mean = g.mean((2, 3), keepdim=True), coeffs.mean((2, 3), keepdim=True)
    g0, q0 = g - g_mean, coeffs - q_mean
    mg, mq = box_filter(g0, radius), box_filter(q0, radius)
    cov = box_filter(g0 * q0, radius) - mg * mq
    var = box_filter(g0 * g0, radius) - mg * mg
    a = cov / (var + eps)
    b = mq - a * mg
    size = image.shape[-2:]
    a = F.interpolate(box_filter(a, radius), size=size, mode="bilinear", align_corners=False)
    b = F.interpolate(box_filter(b, radius), size=size, mode="bilinear", align_corners=False)
    return a * (grey - g_mean) + b + q_mean


def conv_bn_relu(cin: int, cout: int, k: int = 1) -> nn.Sequential:
    return nn.Sequential(nn.Conv2d(cin, cout, k, padding=k // 2, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


class LRASPP(nn.Module):
    def __init__(self, low_ch: int, high_ch: int, mid: int, num_classes: int):
        super().__init__()
        self.cbr = conv_bn_relu(high_ch, mid)
        self.scale = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(high_ch, mid, 1), nn.Sigmoid())
        self.high_cls = nn.Conv2d(mid, num_classes, 1)
        self.low_cls = nn.Conv2d(low_ch, num_classes, 1)

    def forward(self, low: torch.Tensor, high: torch.Tensor) -> torch.Tensor:
        x = self.cbr(high) * self.scale(high)
        x = F.interpolate(x, size=low.shape[-2:], mode="bilinear", align_corners=False)
        return self.low_cls(low) + self.high_cls(x)


class FPNHead(nn.Module):
    """Semantic FPN (Kirillov et al., 2019, simplified): top-down fusion, merged at the finest level."""

    def __init__(self, in_channels: list[int], mid: int, num_classes: int):
        super().__init__()
        self.lateral = nn.ModuleList(conv_bn_relu(c, mid) for c in in_channels)
        self.smooth = nn.ModuleList(conv_bn_relu(mid, mid, 3) for _ in in_channels)
        self.cls = nn.Conv2d(mid, num_classes, 1)

    def forward(self, feats: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        size = feats[0].shape[-2:]
        x = self.lateral[-1](feats[-1])
        fused = F.interpolate(self.smooth[-1](x), size=size, mode="bilinear", align_corners=False)
        for i in range(len(feats) - 2, -1, -1):
            x = self.lateral[i](feats[i]) + F.interpolate(x, size=feats[i].shape[-2:], mode="bilinear", align_corners=False)
            fused = fused + F.interpolate(self.smooth[i](x), size=size, mode="bilinear", align_corners=False)
        return self.cls(fused), fused


class FineFusion(nn.Module):
    """Stride-4 refinement: a 3x3 conv over [skip features, upsampled logits], added residually.

    The last conv starts at zero, so the head initially equals the plain LR-ASPP output.
    """

    def __init__(self, skip_ch: int, mid: int, num_classes: int):
        super().__init__()
        self.body = conv_bn_relu(skip_ch + num_classes, mid, 3)
        self.cls = nn.Conv2d(mid, num_classes, 1)
        nn.init.zeros_(self.cls.weight)
        nn.init.zeros_(self.cls.bias)

    def forward(self, skip: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
        up = F.interpolate(logits, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return up + self.cls(self.body(torch.cat([skip, up], 1)))


class Refiner(nn.Module):
    """Full-resolution residual correction conditioned on input and affine output; starts at zero."""

    def __init__(self, mid: int):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(6, mid, 3, padding=1), nn.ReLU(inplace=True),
                                 nn.Conv2d(mid, mid, 3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(mid, 3, 3, padding=1))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, image: torch.Tensor, affine: torch.Tensor) -> torch.Tensor:
        return (affine + self.net(torch.cat([image, affine], 1))).clamp(0.0, 1.0)


class LayerNorm2d(nn.Module):
    def __init__(self, c: int, eps: float = 1e-6):
        super().__init__()
        self.weight, self.bias, self.eps = nn.Parameter(torch.ones(c)), nn.Parameter(torch.zeros(c)), eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        return (x - mu) / torch.sqrt(var + self.eps) * self.weight[:, None, None] + self.bias[:, None, None]


class NAFBlock(nn.Module):
    """NAFNet block: LayerNorm, depthwise conv, SimpleGate, simplified channel attention."""

    def __init__(self, c: int):
        super().__init__()
        self.norm1, self.norm2 = LayerNorm2d(c), LayerNorm2d(c)
        self.conv1, self.dw = nn.Conv2d(c, 2 * c, 1), nn.Conv2d(2 * c, 2 * c, 3, padding=1, groups=2 * c)
        self.sca, self.conv2 = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(c, c, 1)), nn.Conv2d(c, c, 1)
        self.ffn1, self.ffn2 = nn.Conv2d(c, 2 * c, 1), nn.Conv2d(c, c, 1)
        self.beta, self.gamma = nn.Parameter(torch.zeros(1, c, 1, 1)), nn.Parameter(torch.zeros(1, c, 1, 1))

    @staticmethod
    def gate(x: torch.Tensor) -> torch.Tensor:
        a, b = x.chunk(2, dim=1)
        return a * b

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        x = self.gate(self.dw(self.conv1(self.norm1(inp))))
        x = self.conv2(x * self.sca(x))
        y = inp + self.beta * x
        return y + self.gamma * self.ffn2(self.gate(self.ffn1(self.norm2(y))))


class NAFRefiner(nn.Module):
    """Compact NAFNet U-Net (3 scales) predicting a residual on top of the affine output."""

    def __init__(self, width: int, enc=(1, 1, 2), middle: int = 2, dec=(1, 1, 1)):
        super().__init__()
        self.intro, self.ending = nn.Conv2d(6, width, 3, padding=1), nn.Conv2d(width, 3, 3, padding=1)
        nn.init.zeros_(self.ending.weight)
        nn.init.zeros_(self.ending.bias)
        self.encoders, self.downs, self.ups, self.decoders = (nn.ModuleList() for _ in range(4))
        c = width
        for n in enc:
            self.encoders.append(nn.Sequential(*[NAFBlock(c) for _ in range(n)]))
            self.downs.append(nn.Conv2d(c, 2 * c, 2, stride=2))
            c *= 2
        self.middle = nn.Sequential(*[NAFBlock(c) for _ in range(middle)])
        for n in dec:
            self.ups.append(nn.Sequential(nn.Conv2d(c, 2 * c, 1, bias=False), nn.PixelShuffle(2)))
            c //= 2
            self.decoders.append(nn.Sequential(*[NAFBlock(c) for _ in range(n)]))
        self.multiple = 2 ** len(enc)

    def forward(self, image: torch.Tensor, affine: torch.Tensor) -> torch.Tensor:
        h, w = image.shape[-2:]
        ph, pw = (-h) % self.multiple, (-w) % self.multiple
        x = F.pad(torch.cat([image, affine], 1), (0, pw, 0, ph), mode="replicate")
        x = self.intro(x)
        skips = []
        for enc, down in zip(self.encoders, self.downs):
            x = enc(x)
            skips.append(x)
            x = down(x)
        x = self.middle(x)
        for up, dec, skip in zip(self.ups, self.decoders, reversed(skips)):
            x = dec(up(x) + skip)
        return (affine + self.ending(x)[..., :h, :w]).clamp(0.0, 1.0)


class AffineRestorer(nn.Module):
    """Predicts 12 affine coefficients per coarse cell; initialised to the identity map."""

    def __init__(self, in_ch: int, mid: int):
        super().__init__()
        self.body = nn.Sequential(conv_bn_relu(in_ch, mid, 3), conv_bn_relu(mid, mid, 3))
        self.coeffs = nn.Conv2d(mid, 12, 1)
        nn.init.zeros_(self.coeffs.weight)
        identity = torch.tensor([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0], dtype=torch.float32)
        with torch.no_grad():
            self.coeffs.bias.copy_(identity)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        return self.coeffs(self.body(feat))

    @staticmethod
    def apply(coeffs: torch.Tensor, image: torch.Tensor) -> torch.Tensor:
        """Upsample coefficients to ``image`` resolution (bilinear, unless they already are at it)
        and apply out_c = A_c . [r, g, b, 1]."""
        b, _, h, w = image.shape
        if coeffs.shape[-2:] != (h, w):
            coeffs = F.interpolate(coeffs, size=(h, w), mode="bilinear", align_corners=False)
        a = coeffs.reshape(b, 3, 4, h, w)
        out = (a[:, :, :3] * image.unsqueeze(1)).sum(dim=2) + a[:, :, 3]
        return out.clamp(0.0, 1.0)


class EURSNet(nn.Module):
    def __init__(self, cfg: ModelConfig = ModelConfig()):
        super().__init__()
        self.cfg = cfg
        dilation = {} if cfg.output_stride == 32 else {"output_stride": cfg.output_stride}
        self.encoder = timm.create_model(cfg.encoder, pretrained=cfg.pretrained, features_only=True, **dilation)
        reductions = self.encoder.feature_info.reduction()
        channels = self.encoder.feature_info.channels()
        self.low_idx = reductions.index(8)
        self.high_idx = len(reductions) - 1
        self.fine_idx = None
        self.fine_cls = None
        self.aux = None
        self.aux_logits = None  # set by the last training-mode forward when aux_channels > 0
        self.kd_feat = None  # deepest encoder map of the last training-mode forward (feature distillation)
        self.fuse_idx = None
        self.fuse = None
        if cfg.fine_channels > 0 and cfg.decoder != "lraspp":
            raise ValueError("fine_channels is an LR-ASPP option; the FPN decoder already fuses stride 4")
        if cfg.aux_channels > 0 and cfg.decoder != "lraspp":
            raise ValueError("aux_channels is an LR-ASPP (Track 2 student) option")
        if cfg.decoder == "fpn":
            self.levels = [i for i, r in enumerate(reductions) if r >= 4]
            self.seg_head = FPNHead([channels[i] for i in self.levels], cfg.seg_channels, cfg.num_classes)
            self.restorer = AffineRestorer(cfg.seg_channels, cfg.restore_channels)
        elif cfg.decoder == "lraspp":
            self.seg_head = LRASPP(channels[self.low_idx], channels[self.high_idx], cfg.seg_channels, cfg.num_classes)
            self.restorer = AffineRestorer(channels[self.low_idx], cfg.restore_channels)
            self.fine_idx = reductions.index(4) if cfg.seg_stride4 and 4 in reductions else None
            self.fine_cls = nn.Conv2d(channels[self.fine_idx], cfg.num_classes, 1) if self.fine_idx is not None else None
            if cfg.fine_channels > 0:
                self.fuse_idx = reductions.index(4)
                self.fuse = FineFusion(channels[self.fuse_idx], cfg.fine_channels, cfg.num_classes)
            if cfg.aux_channels > 0:
                self.aux = nn.Sequential(conv_bn_relu(channels[self.high_idx], cfg.aux_channels, 3),
                                         nn.Conv2d(cfg.aux_channels, cfg.num_classes, 1))
        else:
            raise ValueError(f"unknown decoder {cfg.decoder!r}")
        refiners = {"plain": Refiner, "naf": NAFRefiner}
        if cfg.refine_type not in refiners:
            raise ValueError(f"unknown refine_type {cfg.refine_type!r}")
        if cfg.coeff_upsample not in ("bilinear", "guided"):
            raise ValueError(f"unknown coeff_upsample {cfg.coeff_upsample!r}")
        if not 0.0 < cfg.internal_scale <= 1.0:
            raise ValueError(f"internal_scale must be in (0, 1], got {cfg.internal_scale}")
        self.refiner = refiners[cfg.refine_type](cfg.refine_channels) if cfg.refine_channels > 0 else None
        self.global_coeffs = None
        if cfg.global_colour:  # zero-initialised: starts as the plain per-cell map
            self.global_coeffs = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(channels[-1], 12, 1))
            nn.init.zeros_(self.global_coeffs[1].weight)
            nn.init.zeros_(self.global_coeffs[1].bias)
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1), persistent=False)
        if cfg.input_norm not in ("imagenet", "instance"):
            raise ValueError(f"unknown input_norm {cfg.input_norm!r}")
        self.pre = load_pre_restorer(cfg.pre_restorer) if cfg.pre_restorer else None

    def train(self, mode: bool = True) -> "EURSNet":
        super().train(mode)
        if self.pre is not None:  # frozen: always in eval mode
            self.pre.eval()
        return self

    def normalise(self, x: torch.Tensor) -> torch.Tensor:
        if self.cfg.input_norm == "instance":
            m = x.mean((2, 3), keepdim=True)
            return (x - m) / (((x - m) ** 2).mean((2, 3), keepdim=True) + 1e-6).sqrt()
        return (x - self.mean) / self.std

    def heads(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (affine coefficients on a coarse grid, logits at stride 4 or 8).

        With ``flip_tta`` (inference only) the heads also run on the mirrored image and both
        outputs are averaged: two batch-1 passes, so every FLOP counter sees the true 2x cost.
        """
        coeffs, logits = self._heads(x)
        if self.cfg.flip_tta and not self.training:
            c_flip, l_flip = self._heads(x.flip(-1))
            coeffs, logits = (coeffs + c_flip.flip(-1)) / 2, (logits + l_flip.flip(-1)) / 2
        return coeffs, logits

    def _heads(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feats = self.encoder(self.normalise(x))
        if self.cfg.decoder == "fpn":
            logits, fused = self.seg_head([feats[i] for i in self.levels])
            return self._global(self.restorer(fused), feats), logits
        low, high = feats[self.low_idx], feats[self.high_idx]
        self.kd_feat = high if self.training else None
        logits = self.seg_head(low, high)
        self.aux_logits = self.aux(high) if self.aux is not None and self.training else None
        if self.fine_cls is not None:
            fine = feats[self.fine_idx]
            logits = self.fine_cls(fine) + F.interpolate(logits, size=fine.shape[-2:], mode="bilinear", align_corners=False)
        if self.fuse is not None:
            logits = self.fuse(feats[self.fuse_idx], logits)
        return self._global(self.restorer(low), feats), logits

    def _global(self, coeffs: torch.Tensor, feats: list[torch.Tensor]) -> torch.Tensor:
        """Add the per-image colour matrix (``global_colour``) to every cell of the coefficient grid."""
        return coeffs if self.global_coeffs is None else coeffs + self.global_coeffs(feats[-1])

    def final_classifier(self) -> nn.Conv2d:
        """The last 1x1 conv whose output is added unchanged into the logits (bias folding target).

        A constant added to its bias shifts every output logit by that constant: later steps are
        additions, bilinear resizes (weights sum to 1) and flip-TTA averaging.
        """
        if self.cfg.decoder == "fpn":
            return self.seg_head.cls
        if self.fuse is not None:
            return self.fuse.cls
        if self.fine_cls is not None:
            return self.fine_cls
        return self.seg_head.low_cls

    def restore(self, coeffs: torch.Tensor, image: torch.Tensor) -> torch.Tensor:
        if self.cfg.coeff_upsample == "guided":
            coeffs = guided_upsample(coeffs, image, self.cfg.guide_radius, self.cfg.guide_eps)
        affine = AffineRestorer.apply(coeffs, image)
        return self.refiner(image, affine) if self.refiner is not None else affine

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Heads on an in-graph downsampled copy (``internal_scale``); outputs at the input size."""
        seen = x
        if self.pre is not None:
            with torch.autocast(device_type=x.device.type, enabled=False):
                seen = self.pre(x.float())
        small = downsample(seen, internal_size(*x.shape[-2:], self.cfg.internal_scale))
        coeffs, logits = self.heads(small)
        restored = self.restore(coeffs, x)
        logits = F.interpolate(logits, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return restored, logits

    @torch.no_grad()
    def predict_full_res(self, image: torch.Tensor, size: tuple[int, int]) -> tuple[torch.Tensor, torch.Tensor]:
        """Track 1 inference: run heads at model ``size`` but restore/segment at full resolution."""
        small = F.interpolate(image, size=size, mode="bilinear", align_corners=False)
        coeffs, logits = self.heads(small)
        restored = self.restore(coeffs, image)
        logits = F.interpolate(logits, size=image.shape[-2:], mode="bilinear", align_corners=False)
        return restored, logits


def load_pre_restorer(path: str) -> nn.Module:
    """Frozen restorer from ``build()`` in the file at ``path`` (Track 2 restore-then-segment arm)."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("macvi_pre_restorer", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    net = mod.build().eval()
    for p in net.parameters():
        p.requires_grad_(False)
    return net


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
