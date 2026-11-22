"""Potential-domain Fourier (DCT) low-pass for ONE monotonic CV branch.

Why DCT and not a plain FFT: a CV branch is not periodic, so an FFT assumes a jump
between its last and first point and rings at the edges. The DCT-II implicitly mirrors
the branch, which removes that jump. With the end-to-end line removed first, edge error
is ~0 for smooth data.

The "frequency" axis is cycles per VOLT (the scan axis), not Hz. For DCT-II of N points
spaced step_v apart, coefficient k sits at  f_k = k / (2 * N * step_v)  cycles/V.
"""
from __future__ import annotations

import numpy as np
from scipy.fft import dct, idct


def dct_spectrum(y: np.ndarray, step_v: float):
    """Return (f_cycles_per_v, coefficients) of a detrended branch (orthonormal DCT)."""
    y = np.asarray(y, dtype=float)
    r = y - np.linspace(y[0], y[-1], len(y))
    c = dct(r, type=2, norm="ortho")
    return np.arange(len(c)) / (2 * len(c) * step_v), c


def estimate_cutoff(
    y: np.ndarray,
    step_v: float,
    power_ratio: float = 5.0,
    smooth: int = 5,
    margin: int = 3,
    min_cycles_per_v: float = 5.0,
) -> tuple[float, float]:
    """Pick the cutoff where signal power meets the noise floor.

    Noise sigma comes from the upper half of the spectrum (signal-free for CV peaks):
    sigma = median(|c|) / 0.6745. The cutoff is the last coefficient (in the lower 60%)
    whose 5-point smoothed power exceeds power_ratio * sigma^2, plus a small margin.
    Returns (cutoff_cycles_per_v, sigma).
    """
    f, c = dct_spectrum(y, step_v)
    n = len(c)
    sigma = float(np.median(np.abs(c[n // 2:])) / 0.6745)
    p = np.convolve(c**2, np.ones(smooth) / smooth, mode="same")
    idx = np.flatnonzero(p[: int(0.6 * n)] > power_ratio * sigma**2)
    k = int(idx.max()) + margin if idx.size else int(min_cycles_per_v * 2 * n * step_v)
    return float(f[min(k, n - 1)]), sigma


def dct_lowpass(
    y: np.ndarray, step_v: float, cutoff_cycles_per_v: float | None = None
) -> np.ndarray:
    """Smooth one branch. cutoff=None -> automatic. Raised-cosine roll-off (no brick wall)."""
    y = np.asarray(y, dtype=float)
    n = len(y)
    if n < 16:
        raise ValueError(f"branch too short for DCT filtering ({n} points)")
    trend = np.linspace(y[0], y[-1], n)
    r = y - trend
    if cutoff_cycles_per_v is None:
        cutoff_cycles_per_v, _ = estimate_cutoff(y, step_v)
    kc = cutoff_cycles_per_v * 2 * n * step_v          # cutoff as a coefficient index
    kt = max(2.0, 0.25 * kc)                           # roll-off width
    k = np.arange(n)
    gain = np.where(k <= kc, 1.0, np.where(k < kc + kt, 0.5 * (1 + np.cos(np.pi * (k - kc) / kt)), 0.0))
    return idct(dct(r, type=2, norm="ortho") * gain, type=2, norm="ortho") + trend
