# EURS Track 1, team StagAI: rebuilds the reported testing-phase submission from the degraded test images.
# Same computation as the build that produced it (tools/workbench/t1_perimage_package_v100.py, recipe "t1rN"), with every
# path passed explicitly. Each output is made from its own degraded input only (challenge rule):
#   restored  = mean of three restoration networks (rnS4all UNet, rnT2all NAFNet + physics head, rnT5all NAFNet) on the
#               image, its file-name key (water type, depth) and its own colour statistics;
#   seg mask  = argmax of the mean class probabilities of 15 teachers (f053EL, hdEL, nbEL, 5 folds each; frozen DINOv3
#               ViT-L/16 + FPN head) on the image's keyed colour-affine correction, horizontal-flip TTA, 9x9 median.
#               If a mask came out (almost) all background (< 0.5 % foreground), that image's background probability is
#               scaled down step by step until 0.5 % is reached.
# Then the official EURS_resize_for_submission.py, format checks, PNG quantisation of restored/ to fit the 1 GB upload
# limit (largest step that fits, as in the submitted zip: step 3), and the zip. Needs one CUDA GPU (16 GB is enough).
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "code" / "src"))
from macvi.eurs import meta  # noqa: E402
from macvi.eurs.data import read_rgb, list_images  # noqa: E402
from macvi.eurs.distill import load_teacher  # noqa: E402
from macvi.eurs.evaluate import scoring_size, predict_probs  # noqa: E402
from macvi.eurs.track1 import verify, official_resize, zip_dir, fit_zip, zip_problems  # noqa: E402
import restorers  # noqa: E402

TEACHERS = [f"{m}{s}" for m in ("f053EL", "hdEL", "nbEL") for s in ("", "_f1", "_f2", "_f3", "_f4")]
RESTORERS = [("rnS4all", "unet", False), ("rnT2all", "naf", True), ("rnT5all", "naf", False)]
SEG_SIZE = (768, 1024)  # (H, W) the teachers run at (data.size of their configs)
MEDIAN, FG_MIN, CHUNK = 9, 0.005, 100
BG_SCALES = (1.0, 0.8, 0.6, 0.45, 0.3, 0.2, 0.1, 0.05)
key = lambda p: (meta.parse(p).water, meta.parse(p).depth)  # noqa: E731


def scored(path):
    img = read_rgb(path)
    h, w = scoring_size(*img.shape[:2])
    return cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)


def keyed(deg, A):
    """The image's own key affine (4x3, fitted on the 20 train scenes) applied per pixel."""
    X = deg.reshape(-1, 3).astype(np.float32) / 255
    return np.clip(X @ A[:3].astype(np.float32) + A[3].astype(np.float32), 0, 1).reshape(deg.shape)


def to_mask(prob):
    """argmax + median; if the image is (almost) all background, scale its background probability down."""
    for s in BG_SCALES:
        q = prob.copy(); q[0] *= s
        mask = cv2.medianBlur(q.argmax(0).astype(np.uint8), MEDIAN)
        if (mask > 0).mean() >= FG_MIN:
            break
    return mask


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--degraded", required=True, help="folder holding degraded/<scene>/*.png (or the scene folders)")
    ap.add_argument("--weights", required=True, help="weights/ folder of the hand-in")
    ap.add_argument("--out", required=True, help="output folder (pred/ and the zip are written here)")
    ap.add_argument("--resize-script", required=True, help="folder holding the official EURS_resize_for_submission.py")
    ap.add_argument("--name", default="track1_StagAI.zip")
    a = ap.parse_args()
    t0, W, out = time.time(), Path(a.weights), Path(a.out)
    images = list_images(Path(a.degraded))
    ka = np.load(W / "key_affines.npz")
    KA = {(int(w), float(d)): A for w, d, A in zip(ka["water"], ka["depth"], ka["A"])}
    missing = sorted({key(p) for p in images} - set(KA))
    assert not missing, f"images whose key has no train affine: {missing[:5]}"
    print("images", len(images), "keys", len(KA), flush=True)

    teachers = [load_teacher(str(W / "teachers" / t), "cpu") for t in TEACHERS]
    maps = np.load(W / "restorers" / "rest_maps.npy", allow_pickle=True).item()["maps"]["cub"]
    gin = restorers.GpuInputs(maps, str(W / "restorers" / "rsuigm_params.npz"))
    nets = [restorers.load(W / "restorers" / f"{t}.pt", arch, phys) for t, arch, phys in RESTORERS]

    pred = out / "pred"
    (pred / "restored").mkdir(parents=True, exist_ok=True)
    (pred / "seg_masks").mkdir(parents=True, exist_ok=True)
    for c in range(0, len(images), CHUNK):  # 15 teachers do not fit one 16 GB GPU: they visit it one at a time per chunk
        block = images[c:c + CHUNK]
        xs = [(keyed(scored(q), KA[key(q)]) * 255 + 0.5).astype(np.uint8) for q in block]
        acc = [0] * len(block)
        for m in teachers:
            m.to("cuda")
            for j, (q, x8) in enumerate(zip(block, xs)):
                acc[j] = acc[j] + predict_probs(m, x8, SEG_SIZE, "cuda", True, path=q)[1]
            m.to("cpu"); torch.cuda.empty_cache()
        for q, s in zip(block, acc):  # one image at a time: nothing from any other image is read
            stem = Path(q).stem
            cv2.imwrite(str(pred / "seg_masks" / f"{stem}.png"), to_mask(s / len(teachers)))
            r = (restorers.restore_mean(nets, gin, scored(q), key(q)) * 255 + 0.5).astype(np.uint8)
            cv2.imwrite(str(pred / "restored" / f"{stem}.png"), cv2.cvtColor(r, cv2.COLOR_RGB2BGR))
        print("pred", c + len(block), len(images), round(time.time() - t0), "s", flush=True)

    problems = verify(pred, images)
    assert not problems, problems[:5]
    official, log = official_resize(pred, images, cache_dir=Path(a.resize_script))
    assert official is not None, log
    problems = verify(official, images)
    assert not problems, problems[:5]
    zp = zip_dir(official, out / a.name)
    fit = fit_zip(zp)
    res = {"zip": str(zp), "images": len(images), **fit, "zip_problems": zip_problems(zp, len(images)),
           "sha256": hashlib.sha256(zp.read_bytes()).hexdigest(), "minutes": round((time.time() - t0) / 60, 1)}
    json.dump(res, open(out / "result.json", "w"), indent=1)
    print("DONE", json.dumps(res), flush=True)


if __name__ == "__main__":
    main()
