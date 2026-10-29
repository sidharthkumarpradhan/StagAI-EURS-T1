"""Track 1, team StagAI: reproduces testing-phase submission 19062 (6e2b4bf589d1).

The organizers call   main(model_dir, input_dir, output_dir, device).

Same computation as the build that produced the uploaded zip (tools/predict_track1.py, recipe "t1rN"); every output is
made from its own degraded input and its file-name key only:
  restored  = mean of three restoration networks (rnS4all UNet, rnT2all NAFNet + physics head, rnT5all NAFNet) on the
              image, its key (water type, depth: the trailing _<type>_<level> of the name) and its own colour statistics;
  seg mask  = argmax of the mean class probabilities of 15 teachers (f053EL, hdEL, nbEL, 5 folds each; frozen DINOv3
              ViT-L/16 + FPN head) on the image's keyed colour-affine correction, horizontal-flip TTA, 9x9 median filter.
              If a mask is below 0.5 % foreground, that image's background probability is scaled down step by step until
              0.5 % is reached.
Outputs are written at the scoring size (longest side 1024, aspect kept; the official resize step is then a no-op).
The uploaded zip had restored/ quantised to a colour step of 3 to fit the 1 GB upload limit
(macvi.eurs.track1.fit_zip); the same quantisation is applied here so that the files equal the uploaded ones.
The DINOv3 backbone comes from the timm / Hugging Face cache baked into the image by the root Dockerfile (no network
at run time). Needs one CUDA GPU (16 GB is enough) and about 24 GB RAM (the 15 teachers stay in CPU memory and visit
the GPU one at a time per chunk of 100 images).
"""
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "src"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import restorers  # noqa: E402
from macvi.eurs import meta  # noqa: E402
from macvi.eurs.data import read_rgb, list_images  # noqa: E402
from macvi.eurs.distill import load_teacher  # noqa: E402
from macvi.eurs.evaluate import scoring_size, predict_probs  # noqa: E402

TEACHERS = [f"{m}{s}" for m in ("f053EL", "hdEL", "nbEL") for s in ("", "_f1", "_f2", "_f3", "_f4")]
RESTORERS = [("rnS4all", "unet", False), ("rnT2all", "naf", True), ("rnT5all", "naf", False)]
SEG_SIZE = (768, 1024)  # (H, W) the teachers run at
MEDIAN, FG_MIN, CHUNK = 9, 0.005, 100
BG_SCALES = (1.0, 0.8, 0.6, 0.45, 0.3, 0.2, 0.1, 0.05)
QUANT_STEP = 3  # restored/ colour step of the uploaded zip


def key(p):
    """(water type, depth) from the trailing _<type>_<level> of the file name; the scene part is never used."""
    m = meta.parse(p)
    return m.water, m.depth


def scored(path):
    img = read_rgb(path)
    h, w = scoring_size(*img.shape[:2])
    return cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)


def keyed(deg, A):
    """The image's own key affine (4x3, fitted on the 20 train scenes) applied per pixel."""
    X = deg.reshape(-1, 3).astype(np.float32) / 255
    return np.clip(X @ A[:3].astype(np.float32) + A[3].astype(np.float32), 0, 1).reshape(deg.shape)


def to_mask(prob):
    for s in BG_SCALES:
        q = prob.copy(); q[0] *= s
        mask = cv2.medianBlur(q.argmax(0).astype(np.uint8), MEDIAN)
        if (mask > 0).mean() >= FG_MIN:
            break
    return mask


def quantize(rgb):
    """As macvi.eurs.track1.quantize_png: every channel rounded to the nearest multiple of QUANT_STEP."""
    return np.clip(np.round(rgb.astype(np.float32) / QUANT_STEP) * QUANT_STEP, 0, 255).astype(np.uint8)


def main(model_dir, input_dir, output_dir, device):
    torch.manual_seed(0)
    torch.set_grad_enabled(False)
    restorers.DEVICE = device
    W, out = Path(model_dir), Path(output_dir)
    images = list_images(Path(input_dir))
    ka = np.load(W / "key_affines.npz")
    KA = {(int(w), float(d)): A for w, d, A in zip(ka["water"], ka["depth"], ka["A"])}
    missing = sorted({key(p) for p in images} - set(KA))
    assert not missing, f"images whose key has no train affine: {missing[:5]}"

    teachers = [load_teacher(str(W / "teachers" / t), "cpu") for t in TEACHERS]
    maps = np.load(W / "restorers" / "rest_maps.npy", allow_pickle=True).item()["maps"]["cub"]
    gin = restorers.GpuInputs(maps, str(W / "restorers" / "rsuigm_params.npz"))
    nets = [restorers.load(W / "restorers" / f"{t}.pt", arch, phys) for t, arch, phys in RESTORERS]

    (out / "restored").mkdir(parents=True, exist_ok=True)
    (out / "seg_masks").mkdir(parents=True, exist_ok=True)
    for c in range(0, len(images), CHUNK):
        block = images[c:c + CHUNK]
        xs = [(keyed(scored(q), KA[key(q)]) * 255 + 0.5).astype(np.uint8) for q in block]
        acc = [0] * len(block)
        for m in teachers:
            m.to(device)
            for j, (q, x8) in enumerate(zip(block, xs)):
                acc[j] = acc[j] + predict_probs(m, x8, SEG_SIZE, device, True, path=q)[1]
            m.to("cpu")
            if device.startswith("cuda"):
                torch.cuda.empty_cache()
        for q, s in zip(block, acc):  # one image at a time: nothing from any other image is read
            stem = Path(q).stem
            cv2.imwrite(str(out / "seg_masks" / f"{stem}.png"), to_mask(s / len(teachers)))
            r = (restorers.restore_mean(nets, gin, scored(q), key(q)) * 255 + 0.5).astype(np.uint8)
            cv2.imwrite(str(out / "restored" / f"{stem}.png"), cv2.cvtColor(quantize(r), cv2.COLOR_RGB2BGR))
        print("pred", c + len(block), len(images), flush=True)


if __name__ == "__main__":
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4] if len(sys.argv) > 4 else dev)
