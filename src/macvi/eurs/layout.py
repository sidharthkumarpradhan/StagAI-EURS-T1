"""Infer the EURS train-split layout from the files themselves, so no glob is typed by hand.

The release (``eurs_train.zip``) holds, per scene, one clean image, one mask and many degraded
versions. Its exact folder names are unknown until download, so roles are decided by content:

- mask folder: images with a handful of distinct values (class ids 0..4) in one channel;
- degraded folder: the largest remaining image group (many versions per scene);
- clean folder: the remaining group whose size matches the mask folder (one per scene).

The scene key is then the simplest candidate that pairs every degraded image with a clean
image and a mask: clean/mask names are keyed by their stem (optionally minus a _mask/_gt
suffix), degraded names by a regex or by their folder. A candidate that gives two clean
images the same key would merge scenes and is never used. ``eurs-audit`` still has the final
say before any training.
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np

from macvi.eurs.data import IMAGE_EXTS, Layout, _scene, discover

MASK_HINTS = ("mask", "label", "seg", "annot")
SCENE_REGEXES = (                    # degraded file name -> scene
    r"^(?P<scene>.+)$",              # same stem in every folder
    r"^(?P<scene>[^_.]+)",           # scene_v3 -> scene
    r"^(?P<scene>.+)_[^_]+$",        # scene_name_003 -> scene_name
    r"^(?P<scene>[^-.]+)",           # scene-v3 -> scene
    r"^(?P<scene>.+)-[^-]+$",        # scene-name-003 -> scene-name
)
CLEAN_REGEXES = (                    # clean / mask file name -> scene (None: same regex as degraded)
    None,
    r"^(?P<scene>.+)$",
    r"^(?P<scene>.+?)(?:[_-](?:mask|masks|gt|label|labels|seg|clean|annotation))?$",
)


def _groups(root: Path) -> dict[str, list[Path]]:
    """Images grouped by folder, plus each top-level folder with all its subfolders (``top/**``)."""
    groups: dict[str, list[Path]] = defaultdict(list)
    for p in sorted(root.rglob("*")):
        if not p.is_file() or p.suffix.lower() not in IMAGE_EXTS:
            continue
        rel = p.relative_to(root)
        if any(part.startswith(".") for part in rel.parts):
            continue
        groups[rel.parent.as_posix()].append(p)
        for depth in range(1, len(rel.parts) - 2 + 1):  # every ancestor folder with files deeper down
            groups["/".join(rel.parts[:depth]) + "/**"].append(p)
    return groups


def looks_like_mask(path: Path, max_values: int = 16) -> bool:
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        return False
    if img.ndim == 3:
        flat = img.reshape(-1, img.shape[2])
        return len(np.unique(flat[:: max(1, len(flat) // 20000)], axis=0)) <= max_values
    return len(np.unique(img)) <= max_values


def _glob(key: str, files: list[Path]) -> str:
    exts = {p.suffix.lower() for p in files}
    ext = exts.pop() if len(exts) == 1 else ".*"
    base = "" if key in (".", "") else key.rstrip("*").rstrip("/") + "/"
    return f"{base}**/*{ext}" if key.endswith("/**") else f"{base}*{ext}"


def roles(root: Path) -> dict[str, str]:
    """Glob for each of degraded / clean / mask."""
    groups = _groups(root)
    if not groups:
        raise FileNotFoundError(f"no images under {root}")
    flat = {k: v for k, v in groups.items() if not k.endswith("/**")}
    masks = [k for k, v in flat.items() if all(looks_like_mask(p) for p in v[:3])]
    if not masks:
        raise ValueError(f"no folder of mask-like images under {root}: {sorted(flat)}")
    mask = max(masks, key=lambda k: (any(h in k.lower() for h in MASK_HINTS), len(flat[k])))
    rest = {k: v for k, v in groups.items() if k != mask and not set(v) & set(flat[mask])}
    if len(rest) < 2:
        raise ValueError(f"need a clean and a degraded folder besides masks '{mask}': {sorted(rest)}")
    degraded = max(rest, key=lambda k: (len(rest[k]), k.endswith("/**"), len(k)))  # ties: deepest folder
    clean_options = {k: v for k, v in rest.items() if not set(v) & set(rest[degraded])}
    if not clean_options:
        raise ValueError(f"no clean folder separate from degraded '{degraded}'")
    clean = min(clean_options, key=lambda k: abs(len(clean_options[k]) - len(flat[mask])))
    return {"degraded_glob": _glob(degraded, rest[degraded]), "clean_glob": _glob(clean, clean_options[clean]),
            "mask_glob": _glob(mask, flat[mask])}


def infer(root: str | Path) -> tuple[Layout, dict]:
    """Best layout for ``root`` and a summary of how it was chosen."""
    root = Path(root)
    globs = roles(root)
    n_degraded = len(list(root.glob(globs["degraded_glob"])))
    per_scene = [sorted(root.glob(globs[k])) for k in ("clean_glob", "mask_glob")]
    tried = []
    for clean_regex in CLEAN_REGEXES:
        for mode, regexes in (("regex", SCENE_REGEXES), ("parent", SCENE_REGEXES[:1])):
            for regex in regexes:
                if mode == "parent" and clean_regex is None:
                    continue  # in parent mode the regex only keys clean/mask files: covered below
                layout = Layout(**globs, scene_regex=regex, degraded_scene=mode, clean_regex=clean_regex)
                pattern = re.compile(clean_regex or regex)
                try:
                    # one clean image and one mask per scene: a key that merges two of them is wrong
                    if any(len({_scene(p, pattern) for p in files}) != len(files) for files in per_scene):
                        continue
                    samples = discover(root, layout)
                except (ValueError, FileNotFoundError):
                    continue
                tried.append((len(samples), len({s.scene for s in samples}), layout))
    if not tried:
        raise ValueError(f"no scene key pairs degraded images with clean/mask files for {globs}")
    paired, scenes, layout = max(tried, key=lambda t: t[:2])  # most pairs, then finest scenes; ties: simplest
    if paired < n_degraded or scenes < 2:
        raise ValueError(f"best layout pairs {paired}/{n_degraded} degraded images over {scenes} scenes: "
                         f"{asdict(layout)}; set data.layout by hand")
    return layout, {"degraded": n_degraded, "paired": paired, "scenes": scenes, "layout": asdict(layout)}
