"""Build the external training archive: map, shrink, dedupe against EURS, write one zip + manifest.

Runs in the CPU data kernel (``macvi.remote.job.run_external_kernel``); every function takes local
paths, so tests drive it with small fake archives.

- Images are shrunk to longest side 1024 (``release.shrink_image``: area, never upscaled) and stored
  as JPEG (quality 95; ~10x smaller than PNG for photos, and the sources are JPEG already). Masks
  follow with nearest neighbour and are stored as PNG.
- Perceptual hash (DCT pHash of the histogram-equalised grey image, 64 bits; mirrored too) of every
  external image is compared with every image of the EURS train release (clean and degraded) and
  of the Dev split. A match within ``threshold`` bits is dropped and listed: training on a copy of
  a Dev or Test scene would leak the leaderboard. External images also dedupe against each other
  (``cross_threshold``); sources are processed in priority order, so the copy with the most
  useful labels (USIS16K: species-level, including sharks and rays) is the one kept.
- The train release gives a control: each degraded version against its own clean image tells how
  far the hash moves under RSUIGM degradation (per water type), i.e. what the dedupe can catch.

Zip layout: ``images/<source>/<id>.jpg``, ``masks/<source>/<id>.png``, ``manifest.csv`` (one row
per image: source, paths, size, pixel count per value) and ``manifest.json`` (sources with licence,
URL and citation, class mapping, histograms, dedupe hits, controls).
"""

from __future__ import annotations

import csv
import io
import json
import re
import time
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np

from macvi.eurs import external_sources as xs

VALUES = {"background": 0, "coral": 1, "fish": 2, "elasmobranch": 3, "other": 4, "unlabelled": xs.UNLABELLED,
          "ignore": xs.IGNORE}
MANIFEST_FIELDS = ["source", "id", "image", "mask", "height", "width", *[f"px_{k}" for k in VALUES]]
_POPCOUNT = np.array([bin(i).count("1") for i in range(256)], np.uint8)
WATER_TYPE = re.compile(r"_(?P<i>\d+)_(?P<j>\d+)$")  # EURS degraded names: <scene>_<water type>_<depth>


# ------------------------------------------------------------------ perceptual hash

def phash(img: np.ndarray) -> np.uint64:
    """64-bit DCT perceptual hash of the histogram-equalised grey image (colour cast and contrast
    loss, the main RSUIGM effects, are largely normalised away)."""
    grey = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    grey = cv2.equalizeHist(np.ascontiguousarray(grey, np.uint8))
    small = cv2.resize(grey, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
    low = cv2.dct(small)[:8, :8].ravel()
    bits = low > np.median(low[1:])
    return np.uint64(int("".join("1" if b else "0" for b in bits), 2))


def hashes(img: np.ndarray) -> tuple[np.uint64, np.uint64]:
    """Hash of the image and of its mirror (datasets often hold flipped copies)."""
    return phash(img), phash(np.ascontiguousarray(img[:, ::-1]))


def distances(h: np.uint64, refs: np.ndarray) -> np.ndarray:
    """Hamming distance from ``h`` to every hash in ``refs`` (uint64 array)."""
    if refs.size == 0:
        return np.zeros(0, np.int64)
    x = np.bitwise_xor(refs.astype(np.uint64), np.uint64(h))
    return _POPCOUNT[x.view(np.uint8)].reshape(-1, 8).sum(1).astype(np.int64)


class HashIndex:
    """Growing set of named hashes with a nearest-neighbour query (exact, vectorised)."""

    def __init__(self) -> None:
        self.names: list[str] = []
        self._buf = np.zeros(1024, np.uint64)

    def add(self, name: str, h: np.uint64) -> None:
        if len(self.names) == len(self._buf):
            self._buf = np.concatenate([self._buf, np.zeros_like(self._buf)])
        self._buf[len(self.names)] = h
        self.names.append(name)

    def __len__(self) -> int:
        return len(self.names)

    def nearest(self, *query: np.uint64) -> tuple[int, str | None]:
        """(smallest distance from any query hash, that entry's name); (65, None) when empty."""
        if not self.names:
            return 65, None
        arr = self._buf[:len(self.names)]
        d = np.min(np.stack([distances(q, arr) for q in query]), axis=0)
        i = int(np.argmin(d))
        return int(d[i]), self.names[i]


# ------------------------------------------------------------------ EURS references and the control

def _is_mask(name: str, img: np.ndarray) -> bool:
    from macvi.eurs.release import is_label_image

    return "mask" in name.lower() or is_label_image(img)


def eurs_references(archives: Iterable[Path]) -> tuple[HashIndex, dict[str, Any]]:
    """Hash every photo (not mask) in the EURS archives (train: clean + degraded; Dev: degraded).

    Returns the index and the control: per water type, the share of degraded train images whose
    hash lies within each distance of their own scene's clean image (``degraded/<scene>/`` paired
    with the clean image whose stem is ``<scene>``), plus the distances between different clean images.
    """
    index, clean, degraded = HashIndex(), {}, []
    for archive in archives:
        with zipfile.ZipFile(archive) as zf:
            for info in zf.infolist():
                name = info.filename
                if info.is_dir() or not name.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")):
                    continue
                img = cv2.imdecode(np.frombuffer(zf.read(info), np.uint8), cv2.IMREAD_COLOR)
                if img is None or _is_mask(name, img):
                    continue
                h = phash(img)
                index.add(f"{archive.name}:{name}", h)
                parts = Path(name).parts
                if "degraded" in [p.lower() for p in parts]:
                    degraded.append((parts[-2], Path(name).stem, h))
                elif "dev" not in archive.name.lower():
                    clean[Path(name).stem] = h
    return index, _control(clean, degraded)


def _control(clean: dict[str, np.uint64], degraded: list[tuple[str, str, np.uint64]]) -> dict[str, Any]:
    by_type: dict[str, list[int]] = {}
    for scene, stem, h in degraded:
        if scene not in clean:
            continue
        m = WATER_TYPE.search(stem)
        key = m.group("i") if m else "?"
        by_type.setdefault(key, []).append(int(distances(h, np.array([clean[scene]], np.uint64))[0]))
    cl = list(clean.values())
    between = [int(distances(cl[i], np.array(cl[i + 1:], np.uint64)).min()) for i in range(len(cl) - 1)]
    pct = lambda v: {f"p{q}": float(np.percentile(v, q)) for q in (5, 50, 95)} if v else {}  # noqa: E731
    return {"clean_images": len(clean), "degraded_paired": sum(map(len, by_type.values())),
            "degraded_to_own_clean_by_water_type": {k: {"n": len(v), **pct(v),
                                                        **{f"within_{t}": round(float(np.mean(np.array(v) <= t)), 3)
                                                           for t in (6, 10, 14)}}
                                                    for k, v in sorted(by_type.items(), key=lambda kv: kv[0].zfill(3))},
            "clean_to_nearest_other_clean": {"min": min(between) if between else None, **pct(between)}}


# ------------------------------------------------------------------ the archive

def _encode(img: np.ndarray, ext: str, quality: int = 95) -> bytes:
    params = [cv2.IMWRITE_JPEG_QUALITY, quality] if ext == ".jpg" else []
    ok, buf = cv2.imencode(ext, img, params)
    if not ok:
        raise ValueError(f"cannot encode {ext}")
    return buf.tobytes()


def shrink_pair(img: np.ndarray, mask: np.ndarray, max_side: int) -> tuple[np.ndarray, np.ndarray]:
    from macvi.eurs.release import shrink_image

    small = shrink_image(img, max_side, photo=True)
    if small.shape[:2] != mask.shape[:2]:
        mask = cv2.resize(mask, (small.shape[1], small.shape[0]), interpolation=cv2.INTER_NEAREST)
    return small, mask


def histogram(mask: np.ndarray) -> dict[str, int]:
    counts = np.bincount(mask.ravel(), minlength=256)
    return {k: int(counts[v]) for k, v in VALUES.items()}


class ArchiveWriter:
    """Streams mapped items into the output zip and keeps the manifest and dedupe bookkeeping."""

    def __init__(self, out_zip: Path, refs: HashIndex, max_side: int = 1024, threshold: int = 10,
                 cross_threshold: int = 6, jpeg_quality: int = 95) -> None:
        out_zip.parent.mkdir(parents=True, exist_ok=True)
        self.zip = zipfile.ZipFile(out_zip, "w", zipfile.ZIP_STORED)
        self.out_zip, self.refs, self.kept = out_zip, refs, HashIndex()
        self.max_side, self.threshold, self.cross_threshold, self.quality = max_side, threshold, cross_threshold, jpeg_quality
        self.rows: list[dict[str, Any]] = []
        self.sources: dict[str, dict[str, Any]] = {}
        self.eurs_hits: list[dict[str, Any]] = []

    def add_source(self, src: xs.Source, items: Iterable[xs.Item]) -> dict[str, Any]:
        stats: dict[str, Any] = {"title": src.title, "licence": src.licence, "licence_url": src.licence_url,
                                 "citation": src.citation, "files": list(src.files), "notes": src.notes,
                                 "read": 0, "kept": 0, "dropped_eurs_match": 0, "dropped_external_duplicate": 0,
                                 "failed": 0, "pixels": Counter(), "images_with": Counter(), "seconds": 0.0}
        started = time.time()
        it = iter(items)
        while True:
            try:
                item_id, img, mask = next(it)
            except StopIteration:
                break
            except (ValueError, cv2.error, OSError) as exc:  # one bad file must not end a 1-hour build
                stats["failed"] += 1
                stats.setdefault("errors", []).append(repr(exc)[:200])
                continue
            stats["read"] += 1
            self._add(src.name, item_id, img, mask, stats)
        stats["seconds"] = round(time.time() - started, 1)
        stats["pixels"], stats["images_with"] = dict(stats["pixels"]), dict(stats["images_with"])
        stats["errors"] = stats.get("errors", [])[:20]
        self.sources[src.name] = stats
        return stats

    def _add(self, source: str, item_id: str, img: np.ndarray, mask: np.ndarray, stats: dict[str, Any]) -> None:
        img, mask = shrink_pair(img, mask, self.max_side)
        h, h_flip = hashes(img)
        d, ref = self.refs.nearest(h, h_flip)
        if d <= self.threshold:
            stats["dropped_eurs_match"] += 1
            self.eurs_hits.append({"source": source, "id": item_id, "eurs": ref, "distance": d})
            return
        d, other = self.kept.nearest(h, h_flip)
        if d <= self.cross_threshold:
            stats["dropped_external_duplicate"] += 1
            examples = stats.setdefault("duplicate_examples", [])
            if len(examples) < 20:
                examples.append([item_id, other, d])
            return
        image, mask_path = f"images/{source}/{item_id}.jpg", f"masks/{source}/{item_id}.png"
        self.zip.writestr(image, _encode(img, ".jpg", self.quality))
        self.zip.writestr(mask_path, _encode(mask, ".png"))
        self.kept.add(f"{source}:{item_id}", h)
        hist = histogram(mask)
        stats["kept"] += 1
        stats["pixels"].update(hist)
        stats["images_with"].update(k for k, v in hist.items() if v)
        self.rows.append({"source": source, "id": item_id, "image": image, "mask": mask_path, "height": img.shape[0],
                          "width": img.shape[1], **{f"px_{k}": v for k, v in hist.items()}})

    def close(self, extra: dict[str, Any]) -> dict[str, Any]:
        buf = io.StringIO()
        w = csv.DictWriter(buf, fieldnames=MANIFEST_FIELDS)
        w.writeheader()
        w.writerows(self.rows)
        self.zip.writestr("manifest.csv", buf.getvalue())
        total = Counter()
        for s in self.sources.values():
            total.update(s["pixels"])
        labelled = sum(total[k] for k in ("background", "coral", "fish", "elasmobranch", "other"))
        manifest = {"images": len(self.rows), "max_side": self.max_side, "threshold": self.threshold,
                    "cross_threshold": self.cross_threshold, "eurs_reference_images": len(self.refs),
                    "sources": self.sources, "pixels": dict(total),
                    "class_share_of_labelled": {k: round(total[k] / labelled, 5) if labelled else 0.0
                                                for k in ("background", "coral", "fish", "elasmobranch", "other")},
                    "eurs_matches": self.eurs_hits, "mapping": xs.mapping_table(), **extra}
        self.zip.writestr("manifest.json", json.dumps(manifest, indent=1))
        self.zip.close()
        manifest["zip_bytes"] = self.out_zip.stat().st_size
        return manifest
