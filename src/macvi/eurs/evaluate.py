"""Evaluate a torch model or an exported ONNX file on EURS samples with the calibrated scorer."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import torch

from macvi.eurs import key_affine, scoring
from macvi.eurs.data import Sample, read_mask, read_rgb
from macvi.eurs.metrics import Accumulator


def scoring_size(h: int, w: int, max_side: int = 1024) -> tuple[int, int]:
    """Resolution the Track 1 evaluator scores at: longest side capped, aspect preserved."""
    scale = min(1.0, max_side / max(h, w))
    return max(1, round(h * scale)), max(1, round(w * scale))


def _to_uint8(chw: np.ndarray) -> np.ndarray:
    return (np.clip(chw.transpose(1, 2, 0), 0, 1) * 255 + 0.5).astype(np.uint8)


def _finish(acc: Accumulator) -> dict[str, float]:
    m = acc.result()
    m["composite"] = scoring.composite(m)
    return m


def load_scored(sample: Sample, max_side: int = 1024) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    deg, clean, mask = read_rgb(sample.degraded), read_rgb(sample.clean), read_mask(sample.mask)
    h, w = scoring_size(*deg.shape[:2], max_side)
    size = (w, h)
    return (cv2.resize(deg, size, interpolation=cv2.INTER_AREA), cv2.resize(clean, size, interpolation=cv2.INTER_AREA),
            cv2.resize(mask, size, interpolation=cv2.INTER_NEAREST))


def meta_for(model, path) -> dict:
    """``{"meta": [file-name metadata]}`` for models that use it (Track 1 teacher v2), else ``{}``."""
    if not getattr(model, "uses_meta", False):
        return {}
    from macvi.eurs.meta import parse
    return {"meta": [parse(path) if path is not None else None]}


def _biased(logits: torch.Tensor, logit_bias) -> torch.Tensor:
    """Logits plus a per-class bias (``macvi.eurs.calibrate``) for a model it was not folded into."""
    if logit_bias is None:
        return logits
    return logits + torch.as_tensor(logit_bias, dtype=logits.dtype, device=logits.device).view(1, -1, 1, 1)


@torch.no_grad()
def predict_probs(model, deg: np.ndarray, size: tuple[int, int], device: str = "cpu", tta: bool = False,
                  path=None, logit_bias=None) -> tuple[np.ndarray, np.ndarray]:
    """Full-resolution Track 1 prediction: (restored uint8 HxWx3, class probabilities float32 CxHxW).

    ``tta`` averages the prediction with that of the horizontally flipped image (a standard,
    safe test-time augmentation; Track 1 has no compute limit). ``path`` (the image's file name)
    supplies the degradation metadata to models that use it.
    """
    x = torch.from_numpy(deg).permute(2, 0, 1)[None].float().div(255).to(device)
    kw = meta_for(model, path)
    restored, logits = model.eval().predict_full_res(x, size, **kw)
    probs = _biased(logits.float(), logit_bias).softmax(1)
    if tta:
        r_flip, l_flip = model.predict_full_res(x.flip(-1), size, **kw)
        restored = (restored + r_flip.flip(-1)) / 2
        probs = (probs + _biased(l_flip.float(), logit_bias).flip(-1).softmax(1)) / 2
    return _to_uint8(restored[0].float().cpu().numpy()), probs[0].cpu().numpy()


def predict_torch(model, deg: np.ndarray, size: tuple[int, int], device: str = "cpu",
                  tta: bool = False, path=None, logit_bias=None) -> tuple[np.ndarray, np.ndarray]:
    """Full-resolution Track 1 prediction: (restored uint8 HxWx3, mask uint8 HxW); see ``predict_probs``."""
    restored, probs = predict_probs(model, deg, size, device, tta, path, logit_bias)
    return restored, probs.argmax(0).astype(np.uint8)


def evaluate_torch(model, samples: Iterable[Sample], size: tuple[int, int], device: str = "cpu",
                   max_side: int = 1024, tta: bool = False, logit_bias=None, input_map=None) -> dict[str, float]:
    """``input_map`` (key_affine.load_table): segment and restore each image's keyed colour map, as trained."""
    acc = Accumulator(model.cfg.num_classes)
    for s in samples:
        deg, clean, mask = load_scored(s, max_side)
        if input_map is not None:
            deg = key_affine.map_input(deg, key_affine.key_of(s.degraded), input_map)
        restored, pred = predict_torch(model, deg, size, device, tta, path=s.degraded, logit_bias=logit_bias)
        acc.add(restored, clean, pred, mask)
    return _finish(acc)


@torch.no_grad()
def evaluate_static(model, samples: Iterable[Sample], size: tuple[int, int], device: str = "cpu",
                    max_side: int = 1024, logit_bias=None) -> dict[str, float]:
    """Torch twin of ``evaluate_onnx``: forward at the static ``size``, resize outputs back.

    ``logit_bias`` applies a calibration bias the model does not carry yet; a model with the bias
    folded in (``calibrate.fold_bias``, as exported) needs none.
    """
    model.eval()
    h, w = size
    acc = Accumulator(model.cfg.num_classes)
    for s in samples:
        deg, clean, mask = load_scored(s, max_side)
        x = torch.from_numpy(cv2.resize(deg, (w, h), interpolation=cv2.INTER_AREA)).permute(2, 0, 1)[None].float().div(255)
        restored, logits = model(x.to(device))
        logits = _biased(logits, logit_bias)
        out_size = (deg.shape[1], deg.shape[0])
        restored = cv2.resize(_to_uint8(restored[0].cpu().numpy()), out_size, interpolation=cv2.INTER_LINEAR)
        pred = cv2.resize(logits[0].argmax(0).byte().cpu().numpy(), out_size, interpolation=cv2.INTER_NEAREST)
        acc.add(restored, clean, pred, mask)
    return _finish(acc)


def evaluate_onnx(path: str | Path, samples: Iterable[Sample], num_classes: int = 5, max_side: int = 1024) -> dict[str, float]:
    """Track 2 proxy: run the static ONNX graph, resize outputs back to the scoring resolution."""
    import onnxruntime as ort

    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    _, _, h, w = sess.get_inputs()[0].shape
    acc = Accumulator(num_classes)
    for s in samples:
        deg, clean, mask = load_scored(s, max_side)
        x = cv2.resize(deg, (w, h), interpolation=cv2.INTER_AREA).transpose(2, 0, 1)[None].astype(np.float32) / 255
        restored, logits = sess.run(["restored_image", "seg_logits"], {"input": x})
        size = (deg.shape[1], deg.shape[0])
        restored = cv2.resize(_to_uint8(restored[0]), size, interpolation=cv2.INTER_LINEAR)
        pred = cv2.resize(logits[0].argmax(0).astype(np.uint8), size, interpolation=cv2.INTER_NEAREST)
        acc.add(restored, clean, pred, mask)
    return _finish(acc)
