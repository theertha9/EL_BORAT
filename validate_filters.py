"""Ground-truth test: which filter recovers the NOISE-FREE curve best?
Run:  python validate_filters.py          (uses the synthetic peaks from run_pipeline.py)
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import median_filter
from scipy.signal import savgol_filter

from fft_filter import dct_lowpass, estimate_cutoff

STEP, N_DRAWS, NOISE_UA = 0.005, 300, 0.35
v_f = np.linspace(-0.2, 0.6, 161)
v_r = np.linspace(0.6, -0.2, 161)[1:]
v, K = np.concatenate([v_f, v_r]), 160
truth = np.concatenate([17 * np.exp(-((v_f - 0.25) / 0.065) ** 2), -16.5 * np.exp(-((v_r - 0.16) / 0.065) ** 2)])

METHODS = {
    "raw": lambda y: y,
    "savgol(11,3)": lambda y: savgol_filter(y, 11, 3, mode="nearest"),
    "median(5)": lambda y: median_filter(y, 5, mode="nearest"),
    "dct 15 c/V": lambda y: dct_lowpass(y, STEP, 15.0),
    "dct auto": lambda y: dct_lowpass(y, STEP, None),
}


def per_branch(fn, y):
    out = y.copy()
    out[: K + 1], out[K + 1:] = fn(y[: K + 1]), fn(y[K + 1:])
    return out


def peaks(y):
    a, c = int(np.argmax(y[: K + 1])), int(np.argmin(y[K + 1:])) + K + 1
    return v[a], y[a], v[c], y[c]


def main():
    tp, rng = peaks(truth), np.random.default_rng(1)
    res = {m: [] for m in METHODS}
    cuts = []
    for _ in range(N_DRAWS):
        noisy = truth + rng.normal(0, NOISE_UA, truth.size)
        cuts.append(estimate_cutoff(noisy[: K + 1], STEP)[0])
        for m, fn in METHODS.items():
            y = per_branch(fn, noisy)
            pk = peaks(y)
            res[m].append([np.sqrt(np.mean((y - truth) ** 2)), abs(pk[0] - tp[0]) * 1e3, abs(pk[2] - tp[2]) * 1e3,
                           abs(pk[1] - tp[1]) / abs(tp[1]) * 100, abs(pk[3] - tp[3]) / abs(tp[3]) * 100])
    print(f"{'method':14s} RMSE_uA  dEpa_mV dEpc_mV dIpa_%  dIpc_%   (vs noise-free truth, mean of {N_DRAWS})")
    for m, r in res.items():
        r = np.mean(r, axis=0)
        print(f"{m:14s} {r[0]:7.3f} {r[1]:8.2f} {r[2]:7.2f} {r[3]:7.2f} {r[4]:7.2f}")
    print(f"auto cutoff: median {np.median(cuts):.1f} cycles/V (5-95%: {np.percentile(cuts,5):.1f}-{np.percentile(cuts,95):.1f})")


if __name__ == "__main__":
    main()
