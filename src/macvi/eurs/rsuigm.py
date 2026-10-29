"""RSUIGM-faithful underwater degrader fitted on the EURS train set (numpy/cv2 only).

Measured formation model (8 Oct 2026: 8 train scenes x 10 waters x 20 depth keys; f053 held-out scenes never fitted;
matches real held-out degraded images at 41-63 dB with their fitted d):

    I_c = e_c(w, z) * (G_0.6 * J)_c * t(x) + Bz_c(w, z) * (1 - t(x)),     t(x) = exp(-beta_w * d(x))

* ``t`` is ONE scalar field shared by the three channels (the colour cast comes from e_c, Bz_c only).
* ``d(x)`` is a smooth range-like field, the same for every (water, z) of a scene; mean ~0.40, std ~0.25,
  right-skewed (p1 0.12, p50 0.30, p99 1.1), mostly larger towards the image bottom.
* ``e_c(w, z)`` is the illumination decay (~exp(-K_c z)), ``Bz_c(w, z)`` the veiling light; both are tabulated per
  (water 1..10, z = 5..100 step 5) in the params file and linearly interpolated in z.
* G_0.6 is a mild Gaussian blur (sigma 0.6 px at the native 1024 px scale) present in every degraded image.
* Output is rounded to 8 bit; no extra sensor noise is added by default (residual of the fit is quantisation-sized).

API: ``degrade(clean_uint8_rgb, water, z, rng, d=None) -> uint8``. With ``d=None`` a plausible field is sampled
(``sample_d``): half real train fields (random flip/crop/scale), half a log-normal field with a vertical gradient.
Parameters come from ``/kaggle/tmp/rsuigm_params.npz`` (built by the fitting scripts); pass ``params_path`` to override.
"""

from __future__ import annotations

import numpy as np
import cv2

DEFAULT_PARAMS = "/kaggle/tmp/rsuigm_params.npz"
_CACHE: dict[str, dict] = {}


def load_params(path: str = DEFAULT_PARAMS) -> dict:
    if path not in _CACHE:
        z = np.load(path, allow_pickle=False)
        _CACHE[path] = {k: z[k] for k in z.files}
        _CACHE[path]["d_bank"] = _CACHE[path]["d_bank"].astype(np.float32)
    return _CACHE[path]


def _grf(h: int, w: int, rng: np.random.Generator) -> np.ndarray:
    """Unit-variance smooth random field: two octaves of cubic-upsampled low-res noise."""
    f = np.zeros((h, w), np.float32)
    for (gh, gw), amp in (((5, 7), 1.0), ((12, 16), 0.45)):
        f += amp * cv2.resize(rng.standard_normal((gh, gw)).astype(np.float32), (w, h), interpolation=cv2.INTER_CUBIC)
    return (f - f.mean()) / (f.std() + 1e-6)


def sample_d(h: int, w: int, rng: np.random.Generator, params: dict | None = None, p_bank: float = 0.5) -> np.ndarray:
    """Plausible range field (h, w) float32 matched to the train statistics of d."""
    p = params or load_params()
    H, W = h, w
    h, w = max(8, h // 4), max(8, w // 4)  # d is smooth: build at 1/4 resolution, upsample linearly at the end
    if rng.random() < p_bank:
        bank = p["d_bank"]
        d = bank[rng.integers(len(bank))]
        if rng.random() < 0.5:
            d = d[:, ::-1]
        if rng.random() < 0.5:  # random crop covering 55-100 % of each side
            hh, ww = d.shape
            ch, cw = int(hh * rng.uniform(0.55, 1)), int(ww * rng.uniform(0.55, 1))
            y0, x0 = rng.integers(0, hh - ch + 1), rng.integers(0, ww - cw + 1)
            d = d[y0:y0 + ch, x0:x0 + cw]
        d = cv2.resize(np.ascontiguousarray(d), (w, h), interpolation=cv2.INTER_LINEAR)
        d = d * np.float32(np.exp(rng.normal(0, 0.12)))  # scene-to-scene level (sd ~15 % of the mean)
    else:
        mu, sd = p["lnd_mean"], p["lnd_std"]  # [ln-level, y-coef, resid sd, ...]
        m, a, s = rng.normal(mu[0], sd[0]), rng.normal(mu[1], sd[1]), abs(rng.normal(mu[2], sd[2]))
        y = np.linspace(-0.5, 0.5, h, dtype=np.float32)[:, None]
        d = np.exp(m + a * y + s * _grf(h, w, rng)).astype(np.float32)
    d = np.clip(d, 0.04, 1.6).astype(np.float32)
    return cv2.resize(d, (W, H), interpolation=cv2.INTER_LINEAR)


def _interp_z(tab: np.ndarray, zs: np.ndarray, z: float) -> np.ndarray:
    """tab: (Z, 3) -> (3,) linear in z, clamped to the table range."""
    z = float(np.clip(z, zs[0], zs[-1]))
    return np.array([np.interp(z, zs, tab[:, c]) for c in range(3)], np.float32)


def degrade(clean: np.ndarray, water: int, z: float, rng: np.random.Generator | None = None, d: np.ndarray | None = None,
            params_path: str = DEFAULT_PARAMS, noise: float = 0.0) -> np.ndarray:
    """Re-degrade a clean uint8 RGB image as RSUIGM would at (water 1..10, depth key z in 5..100).

    ``d`` is an optional range field (any size, resized to the image); None samples one with ``rng``.
    ``noise`` is an optional extra Gaussian sigma in 8-bit units (default 0: the fit residual is quantisation-sized).
    """
    p = load_params(params_path)
    h, w = clean.shape[:2]
    rng = rng or np.random.default_rng()
    w_i = int(np.clip(round(water), 1, 10)) - 1
    e = _interp_z(p["e"][w_i], p["zs"], z)
    bz = _interp_z(p["Bz"][w_i], p["zs"], z)
    if d is None:
        d = sample_d(h, w, rng, p)
    elif d.shape != (h, w):
        d = cv2.resize(d.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
    jb = cv2.GaussianBlur(clean.astype(np.float32), (0, 0), float(p["sigma_blur"]))
    t = np.exp(d.astype(np.float32) * np.float32(-float(p["beta"][w_i])))[..., None]
    out = jb  # in place: out = jb * t * e + (1 - t) * bz  (e and bz broadcast over channels)
    out *= t
    out *= e
    t -= 1.0
    t *= -1.0
    out += t * bz
    if noise > 0:
        out += rng.normal(0, noise, out.shape).astype(np.float32)
    return np.clip(out + 0.5, 0, 255).astype(np.uint8)


# ------------------------------------------------------------------ GPU rendering (training-time, torch)

_TCACHE: dict = {}


def _torch_params(path: str, device):
    import torch
    k = (path, str(device))
    if k not in _TCACHE:
        p = load_params(path)
        _TCACHE[k] = {n: torch.as_tensor(np.asarray(p[n], np.float32), device=device)
                      for n in ("e", "Bz", "beta", "d_bank", "zs")} | {
            "sigma": float(p["sigma_blur"]), "mu": [float(x) for x in p["lnd_mean"][:3]],
            "sd": [float(x) for x in p["lnd_std"][:3]]}
    return _TCACHE[k]


def _gauss_kernel(sigma: float, device):
    import torch
    n = int(round(sigma * 8 + 1)) | 1  # cv2's kernel size for float images: cvRound(sigma*4*2+1)|1
    x = torch.arange(n, device=device, dtype=torch.float32) - n // 2
    g = torch.exp(-x ** 2 / (2 * sigma ** 2))
    return g / g.sum()


def blur_torch(x, sigma: float):
    """Separable Gaussian blur (reflect-101 borders, like cv2.GaussianBlur) of (N, C, H, W)."""
    import torch.nn.functional as F
    g = _gauss_kernel(sigma, x.device)
    n, c = x.shape[:2]
    r = len(g) // 2
    y = F.conv2d(F.pad(x, (r, r, 0, 0), mode="reflect"), g.view(1, 1, 1, -1).expand(c, 1, 1, -1), groups=c)
    return F.conv2d(F.pad(y, (0, 0, r, r), mode="reflect"), g.view(1, 1, -1, 1).expand(c, 1, -1, 1), groups=c)


def sample_d_torch(n: int, h: int, w: int, P: dict, gen=None, p_bank: float = 0.5):
    """``sample_d`` on the GPU for a batch: (n, 1, h, w) range fields (half bank fields, half log-normal)."""
    import torch
    import torch.nn.functional as F
    dev = P["d_bank"].device
    h4, w4 = max(8, h // 4), max(8, w // 4)
    rnd = lambda *s: torch.rand(*s, generator=gen, device=dev)  # noqa: E731
    nrm = lambda *s: torch.randn(*s, generator=gen, device=dev)  # noqa: E731
    out = []
    for _ in range(n):
        if float(rnd(())) < p_bank:
            bank = P["d_bank"]
            d = bank[int(torch.randint(len(bank), (), generator=gen, device=dev))]
            if float(rnd(())) < 0.5:
                d = d.flip(-1)
            if float(rnd(())) < 0.5:
                hh, ww = d.shape
                ch, cw = int(hh * (0.55 + 0.45 * float(rnd(())))), int(ww * (0.55 + 0.45 * float(rnd(()))))
                y0 = int(torch.randint(hh - ch + 1, (), generator=gen, device=dev))
                x0 = int(torch.randint(ww - cw + 1, (), generator=gen, device=dev))
                d = d[y0:y0 + ch, x0:x0 + cw]
            d = F.interpolate(d[None, None], (h4, w4), mode="bilinear", align_corners=False)[0, 0]
            d = d * torch.exp(0.12 * nrm(()))
        else:
            mu, sd = P["mu"], P["sd"]
            m, a = mu[0] + sd[0] * float(nrm(())), mu[1] + sd[1] * float(nrm(()))
            s = abs(mu[2] + sd[2] * float(nrm(())))
            f = torch.zeros(h4, w4, device=dev)
            for (gh, gw), amp in (((5, 7), 1.0), ((12, 16), 0.45)):
                f = f + amp * F.interpolate(nrm(1, 1, gh, gw), (h4, w4), mode="bicubic", align_corners=False)[0, 0]
            f = (f - f.mean()) / (f.std() + 1e-6)
            y = torch.linspace(-0.5, 0.5, h4, device=dev)[:, None]
            d = torch.exp(m + a * y + s * f)
        out.append(d.clamp(0.04, 1.6))
    d = torch.stack(out)[:, None]
    return F.interpolate(d, (h, w), mode="bilinear", align_corners=False)


def degrade_torch(clean, params_path: str = DEFAULT_PARAMS, water=None, zi=None, d=None, gen=None, water_p=None):
    """Batch ``degrade`` on the GPU: clean (N, 3, H, W) float in 0..1 -> degraded, 8-bit quantised, 0..1.

    ``water`` (N,) 1..10 and ``zi`` (N,) indices into the params' depth grid default to uniform random keys, the
    release grid (as ``ExternalDataset`` samples them); ``water_p`` (10 weights) skews the random water types
    (``train.water_boost``); ``d`` (N, 1, H, W) defaults to ``sample_d_torch``."""
    import torch
    P = _torch_params(params_path, clean.device)
    n, _, h, w = clean.shape
    if water is None and water_p is not None:
        p = torch.as_tensor(water_p, dtype=torch.float, device=clean.device)
        water = torch.multinomial(p / p.sum(), n, replacement=True, generator=gen) + 1
    if water is None:
        water = torch.randint(1, 11, (n,), generator=gen, device=clean.device)
    if zi is None:
        zi = torch.randint(len(P["zs"]), (n,), generator=gen, device=clean.device)
    wi = torch.as_tensor(water, device=clean.device).long() - 1
    zi = torch.as_tensor(zi, device=clean.device).long()
    e = P["e"][wi, zi].view(n, 3, 1, 1)
    bz = P["Bz"][wi, zi].view(n, 3, 1, 1) / 255.0
    if d is None:
        d = sample_d_torch(n, h, w, P, gen)
    t = torch.exp(-P["beta"][wi].view(n, 1, 1, 1) * d)
    out = blur_torch(clean.float() * 255.0, P["sigma"]) / 255.0 * t * e + (1 - t) * bz
    return torch.floor(out.clamp(0, 1) * 255.0 + 0.5) / 255.0
