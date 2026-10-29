"""ONNX export and Track 2 compliance checks.

Track 2 contract: single self-contained .onnx (no external data), static input
(1, 3, H, W) float32 in [0, 1], outputs ``restored_image`` (1, 3, H, W) and ``seg_logits``
(1, 5, H, W); caps of <=1.0M parameters and <=12 GFLOPs counted over Conv/Gemm/MatMul.

The official checker (https://macvi.org/EURS_check_submission_flops.py, read 2 Oct 2026)
counts 2 FLOPs per MAC over Conv/Gemm/MatMul and every initializer as a parameter; our
counter matches that and additionally refuses graphs whose shapes it cannot infer (the
official one silently skips such nodes). ``run_official_checker`` runs the real script and
is the final word: the loop only proposes Track 2 candidates it has not rejected.
"""

from __future__ import annotations

import re
import subprocess
import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnx
import torch
from onnx import numpy_helper, shape_inference

INPUT_NAME, OUTPUT_NAMES = "input", ("restored_image", "seg_logits")
MAX_PARAMS, MAX_GFLOPS = 1_000_000, 12.0
OFFICIAL_CHECKER = "https://macvi.org/EURS_check_submission_flops.py"


def export(model: torch.nn.Module, height: int, width: int, path: str | Path, opset: int = 18) -> Path:
    """Static-shape export with the torch.export-based exporter (legacy TorchScript fallback)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    model = model.eval().cpu()
    dummy = torch.rand(1, 3, height, width)
    names = dict(input_names=[INPUT_NAME], output_names=list(OUTPUT_NAMES), opset_version=opset)
    try:
        torch.onnx.export(model, (dummy,), str(path), dynamo=True, external_data=False, verbose=False, **names)
    except Exception:  # older torch or an op the new exporter cannot handle
        torch.onnx.export(model, (dummy,), str(path), dynamo=False, do_constant_folding=True, **names)
    model = fold_shared_batchnorm(onnx.load(str(path)))
    # Store every intermediate shape: Resize sizes computed by Shape->Slice resolve only with data propagation, and
    # without them the official checker cannot size the Convs after a Resize and counts 0 FLOPs for them.
    model = shape_inference.infer_shapes(model, data_prop=True)
    onnx.save(model, str(path), save_as_external_data=False)  # one file
    return path


def fold_shared_batchnorm(model: onnx.ModelProto) -> onnx.ModelProto:
    """Fold eval-mode BatchNormalization into Conv weights that several Conv nodes share.

    The exporter folds Conv+BN only when a weight has one consumer. A graph that runs the same
    layers twice (``flip_tta``) keeps every BN node, and their scale/bias/mean/var initializers
    count as parameters under the official checker (~15k for the student). Folding once per
    shared weight is lossless and is what TensorRT does at build time anyway. Returns ``model``
    unchanged when there is nothing to fold.
    """
    graph = model.graph
    inits = {t.name: t for t in graph.initializer}
    producer = {o: n for n in graph.node for o in n.output}
    users: dict[str, list] = {}
    for n in graph.node:
        for i in n.input:
            users.setdefault(i, []).append(n)
    groups: dict[tuple, list] = {}
    for idx, bn in enumerate(graph.node):
        if bn.op_type != "BatchNormalization":
            continue
        conv = producer.get(bn.input[0])
        attrs = {a.name: a for a in bn.attribute}
        if (conv is None or conv.op_type != "Conv" or len(users[conv.output[0]]) != 1 or len(bn.output) != 1
                or ("training_mode" in attrs and attrs["training_mode"].i) or conv.input[1] not in inits
                or not all(p in inits for p in [*conv.input[2:], *bn.input[1:5]])):
            continue
        eps = attrs["epsilon"].f if "epsilon" in attrs else 1e-5
        groups.setdefault((tuple(conv.input[1:]), tuple(bn.input[1:5]), eps), []).append((conv, bn, idx))
    folded = set()
    for (conv_in, bn_in, eps), pairs in groups.items():
        if any(len(users[name]) != len(pairs) for name in conv_in):  # weight also used outside this group
            continue
        w = numpy_helper.to_array(inits[conv_in[0]]).astype(np.float32)
        b = numpy_helper.to_array(inits[conv_in[1]]).astype(np.float32) if len(conv_in) > 1 else np.zeros(w.shape[0], np.float32)
        gamma, beta, mean, var = (numpy_helper.to_array(inits[p]).astype(np.float32) for p in bn_in)
        scale = gamma / np.sqrt(var + eps)
        names = (f"{conv_in[0]}__bnfold_w", f"{conv_in[0]}__bnfold_b")
        graph.initializer.extend([numpy_helper.from_array(w * scale[:, None, None, None], names[0]),
                                  numpy_helper.from_array((b - mean) * scale + beta, names[1])])
        for conv, bn, idx in pairs:
            del conv.input[1:]
            conv.input.extend(names)
            conv.output[0] = bn.output[0]
            folded.add(idx)
    if not folded:
        return model
    out = onnx.ModelProto()
    out.CopyFrom(model)
    g = out.graph
    del g.node[:]
    g.node.extend(n for i, n in enumerate(graph.node) if i not in folded)
    used = {i for n in g.node for i in n.input} | {o.name for o in g.output}
    del g.initializer[:]
    g.initializer.extend(t for t in graph.initializer if t.name in used)
    produced = {o for n in g.node for o in n.output}
    del g.value_info[:]
    g.value_info.extend(v for v in graph.value_info if v.name in produced)
    return out


def _dims(value_info) -> list[int]:
    return [d.dim_value if d.HasField("dim_value") else -1 for d in value_info.type.tensor_type.shape.dim]


def _require(shape: list[int] | None, node, what: str) -> list[int]:
    """Refuse to count with unknown shapes: -1 dims would silently cancel real MACs."""
    if not shape or any(d <= 0 for d in shape):
        raise ValueError(f"cannot count FLOPs: unknown {what} shape {shape} at {node.op_type} '{node.name}'")
    return shape


def count_macs(model: onnx.ModelProto) -> int:
    """Multiply-accumulates of Conv/ConvTranspose/Gemm/MatMul nodes (static shape inference).

    Conv: N*Hout*Wout * Cout*(Cin/groups)*kh*kw; ConvTranspose: N*Hin*Win * Cin*(Cout/groups)*kh*kw;
    Gemm/MatMul: prod(output) * K, with K honouring Gemm's transA. Unknown shapes raise.
    """
    model = shape_inference.infer_shapes(model, data_prop=True)
    graph = model.graph
    shapes = {vi.name: _dims(vi) for vi in [*graph.input, *graph.value_info, *graph.output]}
    shapes.update({t.name: list(t.dims) for t in graph.initializer})
    macs = 0
    for node in graph.node:
        if node.op_type in ("Conv", "ConvTranspose"):
            w = _require(shapes.get(node.input[1]), node, "weight")
            ref = node.output[0] if node.op_type == "Conv" else node.input[0]
            ref = _require(shapes.get(ref), node, "activation")
            macs += ref[0] * int(np.prod(ref[2:])) * int(np.prod(w))
        elif node.op_type in ("Gemm", "MatMul"):
            a = _require(shapes.get(node.input[0]), node, "input A")
            out = _require(shapes.get(node.output[0]), node, "output")
            trans_a = next((attr.i for attr in node.attribute if attr.name == "transA"), 0)
            k = a[0] if (node.op_type == "Gemm" and trans_a) else a[-1]
            macs += int(np.prod(out)) * k
    return macs


def count_params(model: onnx.ModelProto) -> int:
    return int(sum(numpy_helper.to_array(t).size for t in model.graph.initializer))


@dataclass
class Report:
    path: str
    input_shape: list[int]
    output_shapes: dict[str, list[int]]
    params: int
    gmacs: float
    gflops: float
    single_file: bool
    problems: list[str]

    @property
    def ok(self) -> bool:
        return not self.problems


def check(path: str | Path, flops_per_mac: int = 2, num_classes: int = 5) -> Report:
    path = Path(path)
    model = onnx.load(str(path))
    onnx.checker.check_model(model)
    problems: list[str] = []
    inputs = [i for i in model.graph.input if i.name not in {t.name for t in model.graph.initializer}]
    in_shape = _dims(inputs[0]) if len(inputs) == 1 else []
    outputs = {o.name: _dims(o) for o in model.graph.output}
    if len(inputs) != 1 or len(in_shape) != 4 or in_shape[0] != 1 or in_shape[1] != 3 or min(in_shape) < 1:
        problems.append(f"input must be static (1,3,H,W), got {in_shape}")
    h, w = (in_shape[2], in_shape[3]) if len(in_shape) == 4 else (-1, -1)
    expected = {OUTPUT_NAMES[0]: [1, 3, h, w], OUTPUT_NAMES[1]: [1, num_classes, h, w]}
    if outputs != expected:
        problems.append(f"outputs must be {expected}, got {outputs}")
    single = not any(t.data_location == onnx.TensorProto.EXTERNAL for t in model.graph.initializer)
    if not single or path.with_suffix(".onnx.data").exists():
        problems.append("weights must be embedded (no external .onnx.data)")
    params = count_params(model)
    try:
        macs = count_macs(model)
    except ValueError as exc:  # an un-countable graph is non-compliant, not a crash
        problems.append(str(exc))
        macs = float("nan")
    gflops = macs * flops_per_mac / 1e9
    if params > MAX_PARAMS:
        problems.append(f"{params:,} params > {MAX_PARAMS:,}")
    if gflops > MAX_GFLOPS:
        problems.append(f"{gflops:.2f} GFLOPs > {MAX_GFLOPS}")
    return Report(str(path), in_shape, outputs, params, macs / 1e9, gflops, single, problems)


def run_official_checker(onnx_path: str | Path, cache_dir: str | Path = "external") -> str | None:
    """Download and run the organisers' checker; returns its output or None if unreachable.

    CLI (read on 2 Oct 2026): ``--onnx PATH`` with the locked caps as defaults; ``--check_ops``
    adds an informational warning for ops outside their baseline's vocabulary.
    """
    script = Path(cache_dir) / "EURS_check_submission_flops.py"
    try:
        if not script.exists():
            script.parent.mkdir(parents=True, exist_ok=True)
            urllib.request.urlretrieve(OFFICIAL_CHECKER, script)
    except OSError:
        return None
    proc = subprocess.run([sys.executable, str(script), "--onnx", str(onnx_path), "--check_ops"],
                          capture_output=True, text=True)
    return proc.stdout + proc.stderr


def official_gflops(output: str | None) -> float | None:
    """GFLOPs from the checker's "Total compute: X GFLOPs" line, None if absent."""
    m = re.search(r"Total compute:\s*([0-9.]+)\s*GFLOPs", output or "")
    return float(m.group(1)) if m else None


def official_verdict(output: str | None) -> float:
    """1.0 if the official checker printed PASSED, 0.0 if it rejected or crashed, NaN if it never ran.

    A PASSED with 0 GFLOPs also counts as a failure: the checker could not size the graph (unknown intermediate
    shapes), so its verdict says nothing about the real compute (seen 7 Oct 2026 on an export without stored shapes).
    """
    if output is None:
        return float("nan")
    passed = "PASSED" in output and "REJECTED" not in output
    return 1.0 if passed and official_gflops(output) != 0.0 else 0.0


def fp16_parity(path: str | Path, inputs: list[np.ndarray]) -> dict[str, float]:
    """How much an FP16 build of the graph changes outputs (TensorRT on the Jetson may run FP16).

    Lesson from MaCVi 2025's embedded track: model stability after reduced-precision conversion
    decided which model to submit. Returns segmentation argmax agreement and max restoration
    difference between the FP32 graph and an FP16 conversion (I/O kept in float32).
    """
    import onnxruntime as ort
    from onnxconverter_common import float16

    model16 = float16.convert_float_to_float16(onnx.load(str(path)), keep_io_types=True)
    s32 = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    s16 = ort.InferenceSession(model16.SerializeToString(), providers=["CPUExecutionProvider"])
    agree, rdiff = [], 0.0
    for x in inputs:
        r32, l32 = s32.run(list(OUTPUT_NAMES), {INPUT_NAME: x})
        r16, l16 = s16.run(list(OUTPUT_NAMES), {INPUT_NAME: x})
        agree.append(float((l32.argmax(1) == l16.argmax(1)).mean()))
        rdiff = max(rdiff, float(np.abs(r32 - r16).max()))
    return {"fp16_seg_agreement": float(np.mean(agree)), "fp16_restore_maxdiff": rdiff}
