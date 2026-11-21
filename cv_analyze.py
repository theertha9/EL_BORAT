from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scipy.signal import (
    butter, filtfilt, iirnotch, savgol_filter, find_peaks
)
from scipy.ndimage import median_filter


def estimate_noise_metrics(current: np.ndarray, baseline_fraction: float = 0.15) -> dict:
    """
    Estimate baseline noise (std, RMS) from the initial segment and approximate SNR
    using peak-to-peak / peak amplitude over baseline std.
    """
    n = len(current)
    n_base = max(5, int(n * baseline_fraction))
    baseline = current[:n_base]
    diff = np.diff(current)

    if len(diff) == 0:
        noise_std = np.nan
    else:
        median_diff = np.median(diff)
        mad = np.median(np.abs(diff - median_diff))
        noise_std = float(1.4826 * mad / np.sqrt(2))
    signal_peak = float(np.max(np.abs(current)))
    snr_linear = (signal_peak / noise_std) if noise_std > 0 else np.inf
    snr_db = float(20.0 * np.log10(snr_linear)) if np.isfinite(snr_linear) and snr_linear > 0 else np.inf
    return {
        "baseline_std": noise_std,
        "signal_peak_abs": signal_peak,
        "snr_linear": snr_linear,
        "snr_db": snr_db,
    }


def apply_filter(
    current: np.ndarray,
    dt_s: float,
    method: str = "savgol",
    sg_window: int = 11,
    sg_poly: int = 3,
    median_kernel: int = 5,
    lowpass_cutoff_hz: float = 5.0,
    notch_freq_hz: float = 50.0,
) -> np.ndarray:
    """Apply a selectable smoothing/noise filter to a copy of the current array."""
    y = np.asarray(current, dtype=float).copy()
    if method == "raw" or method is None:
        return y

    if method == "savgol":
        win = min(sg_window, len(y) if len(y) % 2 == 1 else len(y) - 1)
        win = max(win, sg_poly + 2 if (sg_poly + 2) % 2 == 1 else sg_poly + 3)
        if win <= len(y):
            return savgol_filter(y, window_length=win, polyorder=sg_poly)
        return y

    if method == "median":
        k = min(median_kernel, len(y) if len(y) % 2 == 1 else len(y) - 1)
        k = max(3, k)
        return medfilt(y, kernel_size=k)

    fs = 1.0 / dt_s if dt_s > 0 else 100.0
    nyq = 0.5 * fs

    if method == "lowpass":
        wn = min(lowpass_cutoff_hz / nyq, 0.99)
        if wn > 0:
            b, a = butter(N=3, Wn=wn, btype="low")
            return filtfilt(b, a, y, padlen=min(15, len(y) - 1))
        return y

    if method == "notch50":
        w0 = notch_freq_hz / nyq
        if 0 < w0 < 1:
            b, a = iirnotch(w0=w0, Q=30.0)
            return filtfilt(b, a, y, padlen=min(15, len(y) - 1))
        return y

    raise ValueError(f"Unknown filter method: {method}")


def extract_peaks(potential_v: np.ndarray, current: np.ndarray) -> dict:
    """
    Split sweep at vertex (index of max potential) into forward (anodic) and
    reverse (cathodic) branches and find max/min current peaks.
    """
    vertex_idx = int(np.argmax(potential_v))
    fwd_e, fwd_i = potential_v[: vertex_idx + 1], current[: vertex_idx + 1]
    rev_e, rev_i = potential_v[vertex_idx:], current[vertex_idx:]

    anodic_idx = int(np.argmax(fwd_i))
    cathodic_idx = int(np.argmin(rev_i))

    return {
        "epa_v": float(fwd_e[anodic_idx]),
        "ipa": float(fwd_i[anodic_idx]),
        "epc_v": float(rev_e[cathodic_idx]),
        "ipc": float(rev_i[cathodic_idx]),
    }


def evaluate_filter_against_raw(
    potential_v: np.ndarray,
    raw_current: np.ndarray,
    filt_current: np.ndarray,
    cfg=CFG,
) -> dict:
    """Check peak potential shift (<10 mV) and peak current change (<5%) vs raw."""
    raw_peaks = extract_peaks(potential_v, raw_current)
    flt_peaks = extract_peaks(potential_v, filt_current)

    shift_pa_v = abs(flt_peaks["epa_v"] - raw_peaks["epa_v"])
    shift_pc_v = abs(flt_peaks["epc_v"] - raw_peaks["epc_v"])
    max_shift_v = max(shift_pa_v, shift_pc_v)

    rel_chg_pa = (
        abs(flt_peaks["ipa"] - raw_peaks["ipa"]) / abs(raw_peaks["ipa"])
        if raw_peaks["ipa"] != 0
        else 0.0
    )
    rel_chg_pc = (
        abs(flt_peaks["ipc"] - raw_peaks["ipc"]) / abs(raw_peaks["ipc"])
        if raw_peaks["ipc"] != 0
        else 0.0
    )
    max_rel_chg_i = max(rel_chg_pa, rel_chg_pc)

    passed = (max_shift_v < cfg.max_peak_shift_v) and (
        max_rel_chg_i < cfg.max_peak_current_rel_change
    )
    return {
        "max_peak_shift_mv": max_shift_v * 1e3,
        "max_peak_current_chg_pct": max_rel_chg_i * 100.0,
        "within_targets": passed,
        "raw_peaks": raw_peaks,
        "filtered_peaks": flt_peaks,
    }


def compute_repeat_rsd(values: list[float]) -> float:
    """Relative Standard Deviation (RSD %) across >=3 repeat measurements."""
    arr = np.asarray(values, dtype=float)
    mean = float(np.mean(arr))
    if mean == 0 or len(arr) < 2:
        return np.nan
    return float((np.std(arr, ddof=1) / abs(mean)) * 100.0)


def analyze_single_csv(
    raw_csv_path: Path,
    filter_methods: tuple[str, ...] = ("savgol", "median", "lowpass", "notch50"),
    save_plot: bool = True,
) -> dict:
    df = pd.read_csv(raw_csv_path)
    pot = df["v_meas_v"].to_numpy(dtype=float)
    curr = df["current_ua"].to_numpy(dtype=float)
    t = df["t_rel_s"].to_numpy(dtype=float)
    dt = float(np.median(np.diff(t))) if len(t) > 1 else 0.01

    raw_noise = estimate_noise_metrics(curr)
    raw_peaks = extract_peaks(pot, curr)

    filter_results = {}
    filtered_traces = {}
    for m in filter_methods:
        y_f = apply_filter(curr, dt_s=dt, method=m)
        filtered_traces[m] = y_f
        f_noise = estimate_noise_metrics(y_f)
        f_val = evaluate_filter_against_raw(pot, curr, y_f)
        filter_results[m] = {**f_noise, **f_val}

    if save_plot:
        fig, ax = plt.subplots(figsize=(8, 4.8))
        ax.plot(pot, curr, color="0.65", lw=1.2, label="Raw (preserved)")
        if "savgol" in filtered_traces:
            ax.plot(pot, filtered_traces["savgol"], lw=1.8, label="Savitzky-Golay")
        if "lowpass" in filtered_traces:
            ax.plot(pot, filtered_traces["lowpass"], lw=1.5, ls="--", label="Low-pass")
        ax.set_xlabel("Potential E (V)")
        ax.set_ylabel("Current (µA)")
        ax.set_title(f"Voltammogram: {raw_csv_path.stem}")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        plot_path = raw_csv_path.with_suffix(".png")
        fig.savefig(plot_path, dpi=200)
        plt.close(fig)

    return {
        "file": raw_csv_path.name,
        "raw_noise": raw_noise,
        "raw_peaks": raw_peaks,
        "filters": filter_results,
    }
