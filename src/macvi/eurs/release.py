"""Get the EURS release onto disk: download the official Drive folder, unpack, find each split.

Checked on 2 Oct 2026: the folder linked from the challenge page holds two archives,
``eurs_train.zip`` (scenes with clean ground truth, masks and degraded versions; train on it)
and ``eurs_dev_degraded.zip`` (8.6 GB, degraded Dev-split images only; Track 1 predicts these).

In Colab keep the zips on Drive (downloaded once) and unpack to the local disk each session:
training reads thousands of small images, which is slow over the Drive mount.
"""

from __future__ import annotations

import zipfile
from pathlib import Path
from typing import Any

import cv2
import numpy as np

SPLITS = {"train": "train", "dev": "dev"}  # split -> substring of the archive name


def download(folder_url: str, dest: Path, splits: tuple[str, ...] | None = None) -> list[Path]:
    """Download the archives in the Drive folder into ``dest`` (skips files already there).

    ``splits`` limits it to those splits' archives (a training job needs only ``("train",)``)."""
    import gdown

    dest.mkdir(parents=True, exist_ok=True)
    files = gdown.download_folder(folder_url, skip_download=True, quiet=True)
    out = []
    for f in files:
        target = dest / Path(f.path).name
        if splits is not None and split_of(target) not in splits:
            continue
        if not target.exists():
            gdown.download(id=f.id, output=str(target), quiet=False, resume=True)
        out.append(target)
    return out


def unpack(archive: Path, out_dir: Path) -> Path:
    """Extract once. A marker next to (not inside) ``out_dir`` records completion, so an
    interrupted unzip is redone and the data folder holds nothing but the release's files."""
    done = out_dir.with_name(out_dir.name + ".unpacked")
    if not done.exists():
        out_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(out_dir)
        done.write_text(archive.name)
    return content_root(out_dir)


def content_root(path: Path) -> Path:
    """Skip wrapper folders: descend while a folder holds exactly one sub-folder and no files."""
    while True:
        entries = [p for p in path.iterdir() if not p.name.startswith((".", "__MACOSX"))]
        if len(entries) == 1 and entries[0].is_dir():
            path = entries[0]
        else:
            return path


def is_label_image(img: np.ndarray, max_values: int = 16) -> bool:
    """A mask: one channel (or identical channels) / a handful of distinct colours, not a photo."""
    flat = img.reshape(-1, img.shape[2]) if img.ndim == 3 else img.reshape(-1, 1)
    return len(np.unique(flat[:: max(1, len(flat) // 20000)], axis=0)) <= max_values


def shrink_image(img: np.ndarray, max_side: int = 1024, photo: bool = False) -> np.ndarray:
    """Resize to the evaluator's scoring resolution (longest side ``max_side``, Python round(), no
    upscaling): area interpolation for photos, nearest neighbour for masks (labels stay labels).
    ``photo`` skips the mask test (the Dev split holds degraded photos only)."""
    from macvi.eurs.evaluate import scoring_size

    h, w = scoring_size(img.shape[0], img.shape[1], max_side)
    if (h, w) == img.shape[:2]:
        return img
    interp = cv2.INTER_AREA if photo or not is_label_image(img) else cv2.INTER_NEAREST
    return cv2.resize(img, (w, h), interpolation=interp)


def shrink_archive(archive: Path, out_zip: Path, max_side: int = 1024, photos: bool = False) -> dict[str, Any]:
    """Copy ``archive`` into ``out_zip`` with every image at most ``max_side`` on its longest side.

    Reads members one at a time (no full extraction, so peak disk is the input archive plus the
    output). Images are re-encoded as lossless PNG under the same path and stem; other files are
    copied. The train release (22.1 GB, 2 Oct 2026) does not fit Kaggle's 20 GB kernel output, and
    every stage that uses it works at the scoring resolution anyway (``evaluate.scoring_size``).

    The stats also describe the layout (images per folder, the ten most common original sizes,
    duplicate stems), which is all we know of an archive whose contents are not documented.
    """
    from collections import Counter

    from macvi.eurs.data import IMAGE_EXTS

    stats: dict[str, Any] = {"images": 0, "resized": 0, "other": 0, "in_bytes": archive.stat().st_size}
    folders, sizes, stems = Counter(), Counter(), Counter()
    out_zip.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as src, zipfile.ZipFile(out_zip, "w", zipfile.ZIP_STORED) as dst:
        for info in src.infolist():
            if info.is_dir():
                continue
            name = Path(info.filename)
            data = src.read(info)
            if name.suffix.lower() in IMAGE_EXTS:
                img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_UNCHANGED)
                if img is None:
                    raise ValueError(f"cannot decode {info.filename}")
                small = shrink_image(img, max_side, photo=photos)
                stats["images"] += 1
                stats["resized"] += small.shape[:2] != img.shape[:2]
                folders[name.parent.as_posix()] += 1
                sizes[f"{img.shape[1]}x{img.shape[0]}"] += 1
                stems[name.stem] += 1
                ok, png = cv2.imencode(".png", small)
                if not ok:
                    raise ValueError(f"cannot encode {info.filename}")
                dst.writestr(name.with_suffix(".png").as_posix(), png.tobytes())
            else:
                stats["other"] += 1
                dst.writestr(info.filename, data)
    stats["out_bytes"] = out_zip.stat().st_size
    stats["folders"] = dict(folders.most_common(20))
    stats["sizes_wxh"] = dict(sizes.most_common(10))
    stats["duplicate_stems"] = sum(1 for n in stems.values() if n > 1)
    return stats


def split_of(archive: Path) -> str | None:
    name = archive.name.lower()
    return next((split for split, key in SPLITS.items() if key in name), None)


def prepare(archives: list[Path], out_root: Path, splits: tuple[str, ...] = tuple(SPLITS)) -> dict[str, str]:
    """Unpack each archive of ``splits`` into ``out_root/<split>``; returns {split: data root}."""
    roots = {}
    for archive in archives:
        split = split_of(archive)
        if split not in splits or archive.suffix.lower() != ".zip":
            continue
        roots[split] = str(unpack(archive, out_root / split))
    missing = set(splits) - set(roots)
    if missing:
        raise FileNotFoundError(f"no archive for split(s) {sorted(missing)} among {[a.name for a in archives]}")
    return roots
