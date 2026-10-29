"""EURS composite score, calibrated against the public leaderboards.

Official definition (challenge page):

    composite = 0.6*Qseg + 0.3*(0.45*PSNRnorm + 0.45*SSIM + 0.10*sharpness) + 0.1*NR
    Qseg      = F1 * mIoU * coverage_penalty
    PSNRnorm  = PSNR normalised over a 15-35 dB window
    NR        = mean of UIQM/UCIQE/CCF, each normalised over an undisclosed range

The NR normalisation ranges are not published. Because the normalisation is linear, NR
contributes ``a*UIQM + b*UCIQE + c*CCF + d`` to the composite; the default coefficients
below were least-squares fitted on 63 Track 1 rows (2026-10-01 snapshot) and reproduce the
composite of every Track 1 and Track 2 row to within 0.001 (rounding of displayed values).
``fit_nr_coefficients`` re-fits them from any fresh snapshot so the scorer self-calibrates.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

SEG_W, REST_W, NR_W = 0.6, 0.3, 0.1
PSNR_LO, PSNR_HI = 15.0, 35.0


@dataclass(frozen=True)
class NRCoefficients:
    uiqm: float = 0.0152
    uciqe: float = 0.0770
    ccf: float = 0.1466
    bias: float = 0.0116

    def contribution(self, uiqm: float, uciqe: float, ccf: float) -> float:
        """Composite points contributed by the NR term (already includes the 0.1 weight)."""
        return self.uiqm * uiqm + self.uciqe * uciqe + self.ccf * ccf + self.bias


def psnr_norm(psnr: float) -> float:
    return float(np.clip((psnr - PSNR_LO) / (PSNR_HI - PSNR_LO), 0.0, 1.0))


def components(m: Mapping[str, float], nr: NRCoefficients = NRCoefficients()) -> dict[str, float]:
    """Composite points contributed by segmentation, restoration and NR quality."""
    q_seg = m["q_seg"] if "q_seg" in m else m["f1"] * m["miou"] * m.get("coverage", 1.0)
    restoration = 0.45 * psnr_norm(m["psnr"]) + 0.45 * m["ssim"] + 0.10 * min(m["sharpness"], 1.0)
    return {
        "segmentation": SEG_W * q_seg,
        "restoration": REST_W * restoration,
        "nr_quality": nr.contribution(m["uiqm"], m["uciqe"], m["ccf"]),
    }


def composite(m: Mapping[str, float], nr: NRCoefficients = NRCoefficients()) -> float:
    return float(sum(components(m, nr).values()))


def fit_nr_coefficients(rows: Sequence[Mapping[str, float]]) -> NRCoefficients:
    """Least-squares fit of the NR coefficients from leaderboard rows with all metrics."""
    keys = ("composite", "q_seg", "psnr", "ssim", "sharpness", "uiqm", "uciqe", "ccf")
    rows = [r for r in rows if all(r.get(k) is not None for k in keys)]
    if len(rows) < 8:
        return NRCoefficients()
    zero = NRCoefficients(0.0, 0.0, 0.0, 0.0)
    target = np.array([r["composite"] - sum(components(r, zero).values()) for r in rows])
    x = np.array([[r["uiqm"], r["uciqe"], r["ccf"], 1.0] for r in rows])
    coef, *_ = np.linalg.lstsq(x, target, rcond=None)
    return NRCoefficients(*map(float, coef))
