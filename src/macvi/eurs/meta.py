"""Degradation metadata in EURS file names: ``<scene>_<water type>_<depth>``.

Measured on the release (3 Oct 2026, docs/research.md "Track 1 teacher v2"): every train and Dev
degraded image is named ``<scene>_<i>_<j>`` with i = 1..10 and j = 5, 10, ..., 100, a full 10 x 20
grid per scene. i behaves like the 10 Jerlov water types RSUIGM renders (1-3 clear blue, 6-10
coastal green-brown and dark); j like depth or distance (PSNR falls monotonically with j).

Track 1 may use this ("Meta" flag on the leaderboard; the two leading teams do). Track 2's ONNX
gets the image only, so nothing here may reach the student's graph.

``parse`` returns None for any other naming, so callers degrade gracefully to "unknown".
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

NAME = re.compile(r"^(?P<scene>.+?)_(?P<water>\d{1,2})_(?P<depth>\d{1,4})$")
WATER_TYPES = 10
DEPTH_SCALE = 100.0  # the largest depth in the release
DIM = 1 + (WATER_TYPES + 1) + 1  # known flag, one-hot water type 0..10 (0 = a clean input), depth / 100


@dataclass(frozen=True)
class Meta:
    scene: str
    water: int  # 1..10 Jerlov-like water type; 0 marks a clean (undegraded) input
    depth: int  # 5..100 in the release; 0 for a clean input


CLEAN = Meta("", 0, 0)


def parse(name: str | Path) -> Meta | None:
    """Metadata of a degraded image from its file name, or None if the name does not carry it."""
    m = NAME.match(Path(name).stem)
    if not m:
        return None
    water, depth = int(m.group("water")), int(m.group("depth"))
    if not (1 <= water <= WATER_TYPES and 0 < depth <= 10 * DEPTH_SCALE):
        return None
    return Meta(m.group("scene"), water, depth)


def scene_of(name: str | Path) -> str | None:
    """Scene prefix shared by every degraded version of one clean image (None if unparseable)."""
    m = parse(name)
    return m.scene if m else None


def encode(meta: Meta | None) -> np.ndarray:
    """Fixed-length float vector for a network; all zeros means "unknown"."""
    v = np.zeros(DIM, np.float32)
    if meta is None:
        return v
    v[0] = 1.0
    v[1 + min(max(meta.water, 0), WATER_TYPES)] = 1.0
    v[-1] = min(meta.depth / DEPTH_SCALE, 2.0)
    return v
