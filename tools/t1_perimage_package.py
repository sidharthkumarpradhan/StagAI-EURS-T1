# Track 1 zip under the evaluator's per-input rule (enforced from 6 Oct 2026): "Each restored image and segmentation
# mask must be produced from its own degraded input only. Cross-image or per-scene processing, and reusing the same
# (or a nearly identical) output for several inputs, is not allowed."
# The pipeline of t1_perimage_eval_v100.py (paper Table "public-track pipeline"), applied to every Dev image on its own:
# restored = keyed affine of the image itself (key from its own file name; affines fitted on ALL 20 train scenes,
# never on Dev); mask = the average class probabilities of the teachers on that restored image, flip TTA, median.
# Then the official resize, our zip checks, the 1 GB fit, and a duplicate check of the outputs. No upload.
# Murky images (water 10, high severity) can come out all background, the same mask for many inputs, which the rule
# forbids: below FG_MIN foreground the background probability of that image is scaled down until it is reached.
# Env RUNS (teacher run dirs), TTA (default 1), MEDIAN (default 9), OUT (default /kaggle/tmp/work/t1pi), NAME,
# FG_MIN (default 0.005), REPAIR=1 (redo only the masks below FG_MIN from the saved restored images, then re-zip).
# Round T (13 Oct): RESTNETS="tag:phys:arch,..." makes the restored output the mean of those restorers on the image
# (t1_rest_ens_v100.py, all-scene checkpoints), unsharp UNSHARP (default 0); the masks still come from the teachers on
# the keyed affine (what they were validated on). MASKS_FROM=<pred dir>: reuse that build's per-image masks.
# RESTORED_FROM=<pred dir>: reuse that build's restored images (each still made from its own input only).
# 13 Oct board: masks from the all-scene teachers (f053_all + segD_all) scored q_seg 0.257 (56f7c0e6e99f); against the
# fused 6 Oct masks (board mIoU 0.842) they agree 73 %, fold-0 f053 alone 93 %: keep RUNS on fold-validated teachers.
import os, sys, json, time, zlib, hashlib, zipfile
sys.path.insert(0, "/kaggle/tmp/repo/src")
import numpy as np, cv2
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from macvi.eurs import meta
from macvi.eurs.data import discover, Layout, read_rgb, list_images
from macvi.eurs.evaluate import scoring_size, predict_probs
from macvi.eurs.distill import load_teacher
from macvi.eurs.track1 import verify, official_resize, zip_dir, fit_zip, zip_problems

F053 = "/kaggle/tmp/work/runs/eurs/runs/t1/f053fbd99f50"
SEGD = "/kaggle/tmp/work/runs/seg/segD_ext70_rare4"
RUNS = os.environ.get("RUNS", f"{F053},{SEGD}").split(",")
TTA, MEDIAN = os.environ.get("TTA", "1") == "1", int(os.environ.get("MEDIAN", "9"))
OUT = Path(os.environ.get("OUT", "/kaggle/tmp/work/t1pi"))
NAME = os.environ.get("NAME", "track1_perimage.zip")
# DEGRADED (22 Oct, testing phase): the degraded/<scene>/ folder to predict (default: the Dev split)
DEV = Path(os.environ["DEGRADED"]) if os.environ.get("DEGRADED") else next(
    d for d in Path("/home/jupyter/macvi_run/pkgtmp/eurs").rglob("degraded") if (d / "fish3").is_dir())
RESIZE_DIRS = [Path("/kaggle/tmp/repo/external"), Path("/home/jupyter/macvi_run/external")]
key = lambda p: (meta.parse(p).water, meta.parse(p).depth)
RESTNETS = [m.split(":") for m in os.environ.get("RESTNETS", "").split(",") if m]
UNSHARP, MASKS_FROM = float(os.environ.get("UNSHARP", "0")), os.environ.get("MASKS_FROM")
RESTORED_FROM = os.environ.get("RESTORED_FROM")
# SEG_INPUT=restored (14 Oct, H-SI): masks from this image's own restored output instead of its keyed affine map
# (held-out f053EL: q 0.40 -> 0.63, waters 9-10 0.26 -> 0.41, waters 1-8 unchanged)
SEG_INPUT = os.environ.get("SEG_INPUT", "keyed")
FG_MIN, REPAIR = float(os.environ.get("FG_MIN", "0.005")), os.environ.get("REPAIR") == "1"
# CHUNK=N (17 Oct, 15-teacher builds do not fit one 16 GB GPU): the teachers stay on the CPU and visit the GPU one at a
# time for each block of N images; every image's probabilities are still the mean over the teachers on that image alone.
CHUNK = int(os.environ.get("CHUNK", "0"))
BG_SCALES = (1.0, 0.8, 0.6, 0.45, 0.3, 0.2, 0.1, 0.05)
# BIAS (15 Oct): JSON from t1_bias_cal ({"groups": [[1, 3], ...], "bias": [[b0..b4], ...]}): class probabilities are
# multiplied by exp(bias[group of this image's own file-name water type]) before the argmax (per-input calibration)
BIAS = json.loads(os.environ["BIAS"]) if os.environ.get("BIAS") else None


def biased(prob, water):
    if BIAS is None:
        return prob
    gi = next(i for i, (a, b) in enumerate(BIAS["groups"]) if a <= water <= b)
    return prob * np.exp(np.asarray(BIAS["bias"][gi], np.float32))[:, None, None]


def to_mask(prob):
    """argmax + median; if the image is (almost) all background, scale its background probability down."""
    for s in BG_SCALES:
        q = prob.copy(); q[0] *= s
        mask = q.argmax(0).astype(np.uint8)
        mask = cv2.medianBlur(mask, MEDIAN) if MEDIAN else mask
        if (mask > 0).mean() >= FG_MIN:
            break
    return mask


def scored(path):
    img = read_rgb(path)
    h, w = scoring_size(*img.shape[:2])
    return cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)


def moments(s):
    """Normal equations of one train pair for its key, on 20k pixels (deterministic sample)."""
    deg, clean = scored(s.degraded), scored(s.clean)
    X = np.concatenate([deg.reshape(-1, 3) / 255.0, np.ones((deg.shape[0] * deg.shape[1], 1))], 1)
    Y = clean.reshape(-1, 3) / 255.0
    idx = np.random.default_rng(zlib.crc32(str(s.degraded).encode())).choice(len(X), 20000, replace=False)
    return key(s.degraded), X[idx].T @ X[idx], X[idx].T @ Y[idx]


def restore(deg, A):
    X = deg.reshape(-1, 3).astype(np.float32) / 255
    return np.clip(X @ A[:3].astype(np.float32) + A[3].astype(np.float32), 0, 1).reshape(deg.shape)


if __name__ == "__main__":
    t0 = time.time()
    cfg = json.load(open(F053 + "/config.json"))
    size = tuple(cfg["data"]["size"])
    samples = discover("/kaggle/tmp/eurs/train/train", Layout(**cfg["data"]["layout"]))
    XX, XY = defaultdict(lambda: np.zeros((4, 4))), defaultdict(lambda: np.zeros((4, 3)))
    with ProcessPoolExecutor(16) as ex:
        for k, a, b in ex.map(moments, samples, chunksize=8):
            XX[k] += a; XY[k] += b
    KA = {k: np.linalg.solve(XX[k] + 1e-6 * np.eye(4), XY[k]) for k in XX}
    scenes = sorted({s.scene for s in samples})
    print("fitted", len(KA), "keys on", len(samples), "train images of", len(scenes), "scenes", round(time.time() - t0), "s",
          flush=True)
    images = list_images(DEV)
    pred = OUT / "pred"
    if REPAIR:  # the restored PNGs are lossless, so this equals a full rebuild with to_mask
        low = lambda p: (cv2.imread(str(pred / "seg_masks" / f"{Path(p).stem}.png"), 0) > 0).mean() < FG_MIN
        todo = [p for p in images if low(p)]
    else:
        todo = images
    print("predicting", len(todo), "images", flush=True)
    missing = sorted({key(p) for p in images} - set(KA))
    assert not missing, f"Dev keys without a train affine: {missing[:5]}"
    models = [] if MASKS_FROM else [load_teacher(r, "cpu" if CHUNK else "cuda") for r in RUNS]
    chunk_prob = {}

    def chunk_probs(paths):
        import torch
        xs = [(q, (restore(scored(q), KA[key(q)]) * 255 + 0.5).astype(np.uint8)) for q in paths]
        acc = [0] * len(xs)
        for m in models:
            m.to("cuda")
            for j, (q, x8) in enumerate(xs):
                acc[j] = acc[j] + predict_probs(m, x8, size, "cuda", TTA, path=q)[1]
            m.to("cpu"); torch.cuda.empty_cache()
        return {str(q): a / len(models) for (q, _), a in zip(xs, acc)}

    if RESTNETS:
        sys.path.insert(0, "/home/jupyter/macvi_run")
        import torch, torch.nn.functional as F
        import t1_rest_ens as E
        from t1_rest_ab import unsharp
        gin = E.R.GpuInputs(np.load(E.R.MAPS, allow_pickle=True).item()["maps"]["cub"])
        nets = [E.load(*m) for m in RESTNETS]

        def restnet(deg, k):
            h, w = deg.shape[:2]
            d = torch.from_numpy(deg).cuda().permute(2, 0, 1)[None].float() / 255
            d = F.pad(d, (0, (-w) % 16, 0, (-h) % 16), mode="reflect")
            x = gin(d, [k], torch.from_numpy(E.R.gstats(deg).astype(np.float32))[None].cuda())
            y = torch.stack([E.run(n, ph, x, False)[0, :, :h, :w] for n, ph in nets]).mean(0).clamp(0, 1)
            return unsharp(y.permute(1, 2, 0).cpu().numpy(), UNSHARP)
    (pred / "restored").mkdir(parents=True, exist_ok=True)
    (pred / "seg_masks").mkdir(parents=True, exist_ok=True)
    for i, p in enumerate(todo):  # one image at a time: nothing from any other Dev image is read
        if REPAIR:
            r8 = cv2.cvtColor(cv2.imread(str(pred / "restored" / f"{Path(p).stem}.png")), cv2.COLOR_BGR2RGB)
        else:
            deg = scored(p)
            r8 = (restore(deg, KA[key(p)]) * 255 + 0.5).astype(np.uint8)
            if RESTORED_FROM:
                src = Path(RESTORED_FROM) / "restored" / f"{Path(p).stem}.png"
                assert src.exists(), src
                (pred / "restored" / src.name).write_bytes(src.read_bytes())
            else:
                out8 = (restnet(deg, key(p)) * 255 + 0.5).astype(np.uint8) if RESTNETS else r8
                cv2.imwrite(str(pred / "restored" / f"{Path(p).stem}.png"), cv2.cvtColor(out8, cv2.COLOR_RGB2BGR))
        if MASKS_FROM:
            m = cv2.imread(str(Path(MASKS_FROM) / "seg_masks" / f"{Path(p).stem}.png"), 0)
            assert m is not None, p
            cv2.imwrite(str(pred / "seg_masks" / f"{Path(p).stem}.png"), m)
        else:
            x8 = r8
            if SEG_INPUT == "restored":
                x8 = cv2.cvtColor(cv2.imread(str(pred / "restored" / f"{Path(p).stem}.png")), cv2.COLOR_BGR2RGB)
            if CHUNK:
                assert SEG_INPUT == "keyed"
                if str(p) not in chunk_prob:
                    chunk_prob = chunk_probs(todo[i:i + CHUNK])
                prob = chunk_prob.pop(str(p))
            else:
                prob = sum(predict_probs(m, x8, size, "cuda", TTA, path=p)[1] for m in models) / len(models)
            cv2.imwrite(str(pred / "seg_masks" / f"{Path(p).stem}.png"), to_mask(biased(prob, int(key(p)[0]))))
        if i % 200 == 0:
            print("pred", i, len(todo), round(time.time() - t0), "s", flush=True)
    problems = verify(pred, images)
    assert not problems, problems[:5]
    script_dir = next(d for d in RESIZE_DIRS if (d / "EURS_resize_for_submission.py").exists())
    official, log = official_resize(pred, images, cache_dir=script_dir)
    assert official is not None, log
    problems = verify(official, images)
    assert not problems, problems[:5]
    zp = zip_dir(official, OUT / NAME)
    fit = fit_zip(zp)
    problems = zip_problems(zp, len(images))
    with zipfile.ZipFile(zp) as z:  # the rule also forbids the same (or a nearly identical) output for several inputs
        digests = {sub: [hashlib.sha1(z.read(n)).hexdigest() for n in z.namelist() if n.startswith(sub + "/")]
                   for sub in ("restored", "seg_masks")}
    dup = {sub: len(d) - len(set(d)) for sub, d in digests.items()}
    ins = [hashlib.sha1(Path(p).read_bytes()).hexdigest() for p in images]  # duplicates allowed only for identical inputs
    groups = defaultdict(set)
    for sub in ("restored", "seg_masks"):
        for h_in, h_out in zip(ins, [hashlib.sha1((official / sub / f"{Path(p).stem}.png").read_bytes()).hexdigest() for p in images]):
            groups[(sub, h_out)].add(h_in)
    dup_from_distinct_inputs = sum(len(v) > 1 for v in groups.values())
    low_fg = sum((cv2.imread(str(official / "seg_masks" / f"{Path(p).stem}.png"), 0) > 0).mean() < FG_MIN for p in images)
    hist = np.bincount(np.concatenate([cv2.imread(str(official / "seg_masks" / f"{Path(p).stem}.png"), 0).ravel()
                                       for p in images[::50]]), minlength=5)
    res = {"zip": str(zp), "images": len(images), "runs": [Path(r).name for r in RUNS], "restnets": RESTNETS, "unsharp": UNSHARP, "masks_from": MASKS_FROM, "restored_from": RESTORED_FROM, "seg_input": SEG_INPUT, "tta": TTA, "median": MEDIAN,
           "keys": len(KA), "train_scenes": len(scenes), **fit, "zip_problems": problems, "exact_duplicates": dup, "input_duplicates": len(ins) - len(set(ins)),
           "dup_outputs_from_distinct_inputs": dup_from_distinct_inputs, "repaired": len(todo) if REPAIR else 0, "low_fg_masks": int(low_fg),
           "class_share_sample": [round(float(c) / hist.sum(), 4) for c in hist],
           "sha256": hashlib.sha256(zp.read_bytes()).hexdigest(), "minutes": round((time.time() - t0) / 60, 1)}
    json.dump(res, open(OUT / "result.json", "w"), indent=1)
    if os.environ.get("CLEAN", "1") == "1":  # /home/jupyter is shared and filled up on 13 Oct: keep pred + the zip only
        import shutil
        for d in OUT.iterdir():
            if d.is_dir() and d.name != "pred":
                shutil.rmtree(d)
    print("DONE", json.dumps(res), flush=True)
