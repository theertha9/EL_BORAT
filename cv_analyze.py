
from __future__ import annotations

from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scipy.signal import (
    butter,
    filtfilt,
    iirnotch,
    savgol_filter,
    find_peaks,
)
from scipy.ndimage import median_filter

from config import CFG
from cv_acquire import build_sweep_potentials


def estimate_noise_metrics(current: np.ndarray) -> dict:
    """
    Estimate noise using the MAD of successive current differences.

    This is an approximate noise estimate and assumes successive
    noise contributions are reasonably independent.
    """
    current = np.asarray(current, dtype=float)

    current = current[np.isfinite(current)]

    if len(current) < 3:
        return {
            "noise_sigma": np.nan,
            "signal_peak_abs": np.nan,
            "snr_linear": np.nan,
            "snr_db": np.nan,
        }
    diff = np.diff(current)
    median_diff = np.median(diff)
    mad = np.median(np.abs(diff - median_diff))
    
    noise_sigma = float(1.4826 * mad / np.sqrt(2.0))
    signal_peak = float(np.max(np.abs(current)))

    if noise_sigma > 0:
        snr_linear = signal_peak / noise_sigma
        snr_db = float(20.0 * np.log10(snr_linear)) if snr_linear > 0 else np.nan
    elif signal_peak > 0:
        snr_linear = np.inf
        snr_db = np.inf
    else:
        snr_linear = np.nan
        snr_db = np.nan

    return {
        "noise_sigma": noise_sigma,
        "signal_peak_abs": signal_peak,
        "snr_linear": float(snr_linear),
        "snr_db": float(snr_db),
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
    """Apply a selectable filter to a copy of the current array."""

    y = np.asarray(current, dtype=float).copy()

    if y.ndim != 1:
        raise ValueError("Current data must be one-dimensional.")

    if len(y) < 3:
        return y

    if method == "raw" or method is None:
        return y

    if method == "savgol":
        if sg_poly < 0:
            raise ValueError("sg_poly cannot be negative.")

        if sg_window < 3:
            raise ValueError("sg_window must be at least 3.")

        win = min(sg_window, len(y))

        if win % 2 == 0:
            win -= 1

        if win <= sg_poly:
            return y

        return savgol_filter(
            y,
            window_length=win,
            polyorder=sg_poly,
            mode="nearest",
        )

    if method == "median":
        if median_kernel < 1:
            raise ValueError("median_kernel must be positive.")

        kernel_samples = min(median_kernel, len(y))

        return median_filter(
            y,
            size=kernel_samples,
            mode="nearest",
        )

    if dt_s <= 0 or not np.isfinite(dt_s):
        raise ValueError("dt_s must be finite and positive.")

    fs = 1.0 / dt_s
    nyq = fs / 2.0

    if method == "lowpass":
        if not 0 < lowpass_cutoff_hz < nyq:
            raise ValueError(
                "Low-pass cutoff must be between 0 and Nyquist frequency."
            )

        b, a = butter(
            N=3,
            Wn=lowpass_cutoff_hz,
            btype="low",
            fs=fs,
        )

        return filtfilt(b, a, y)

    if method == "notch50":
        if not 0 < notch_freq_hz < nyq:
            raise ValueError(
                f"Cannot apply {notch_freq_hz} Hz notch: "
                f"sampling rate is {fs:.2f} Hz."
            )

        b, a = iirnotch(
            w0=notch_freq_hz,
            Q=30.0,
            fs=fs,
        )

        return filtfilt(b, a, y)

    raise ValueError(f"Unknown filter method: {method}")

#filter_methods=("savgol", "median")



def extract_peaks(
    potential_v: np.ndarray,
    current: np.ndarray,
    v_cmd_v: np.ndarray,
    v_start: float,
    v_vertex: float,
    noise_std: float,
) -> dict:
    """
    Split the first CV cycle into forward and reverse branches.

    The commanded potential determines the vertex location and sweep
    direction. Peak potentials are reported using measured potential.
    """
    potential_v = np.asarray(potential_v, dtype=float)
    current = np.asarray(current, dtype=float)
    v_cmd_v = np.asarray(v_cmd_v, dtype=float)

    if not (
        len(potential_v) == len(current) == len(v_cmd_v)
    ):
        raise ValueError("Potential and current arrays must have equal lengths.")

    if len(current) < 5:
        raise ValueError("Not enough samples for peak extraction.")

    if v_vertex > v_start:
        vertex_idx = int(np.argmax(v_cmd_v))
    elif v_vertex < v_start:
        vertex_idx = int(np.argmin(v_cmd_v))
    else:
        raise ValueError("v_vertex and v_start cannot be equal.")

    fwd_e = potential_v[:vertex_idx + 1]
    fwd_i = current[:vertex_idx + 1]

    rev_e = potential_v[vertex_idx:]
    rev_i = current[vertex_idx:]

    prominence = (
        5.0 * noise_std
        if np.isfinite(noise_std) and noise_std > 0
        else 0.0
    )

    anodic_candidates, anodic_props = find_peaks(
        fwd_i,
        prominence=prominence,
    )

    cathodic_candidates, cathodic_props = find_peaks(
        -rev_i,
        prominence=prominence,
    )

    result = {
        "epa_v": np.nan,
        "ipa": np.nan,
        "epc_v": np.nan,
        "ipc": np.nan,
    }

    if len(anodic_candidates) > 0:
        best = int(
            np.argmax(anodic_props["prominences"])
        )
        anodic_idx = int(anodic_candidates[best])

        result["epa_v"] = float(fwd_e[anodic_idx])
        result["ipa"] = float(fwd_i[anodic_idx])

    if len(cathodic_candidates) > 0:
        best = int(
            np.argmax(cathodic_props["prominences"])
        )
        cathodic_idx = int(cathodic_candidates[best])

        result["epc_v"] = float(rev_e[cathodic_idx])
        result["ipc"] = float(rev_i[cathodic_idx])

    return result




def evaluate_filter_against_raw(
    potential_v: np.ndarray,
    raw_current: np.ndarray,
    filt_current: np.ndarray,
    v_cmd_v: np.ndarray,
    v_start: float,
    v_vertex: float,
    noise_std: float,
    cfg=CFG,
) -> dict:
    """
    Compare filtered peak positions and currents against raw peaks.
    """

    raw_peaks = extract_peaks(
        potential_v,
        raw_current,
        v_cmd_v,
        v_start,
        v_vertex,
        noise_std,
    )

    filtered_peaks = extract_peaks(
        potential_v,
        filt_current,
        v_cmd_v,
        v_start,
        v_vertex,
        noise_std,
    )

    keys = ("epa_v", "ipa", "epc_v", "ipc")

    all_valid = all(
        np.isfinite(raw_peaks[k]) and np.isfinite(filtered_peaks[k])
        for k in keys
    )

    raw_currents_valid = (
        np.isfinite(raw_peaks["ipa"])
        and np.isfinite(raw_peaks["ipc"])
        and raw_peaks["ipa"] != 0
        and raw_peaks["ipc"] != 0
    )

    if not all_valid or not raw_currents_valid:
        return {
            "max_peak_shift_mv": np.nan,
            "max_peak_current_chg_pct": np.nan,
            "within_targets": False,
            "raw_peaks": raw_peaks,
            "filtered_peaks": filtered_peaks,
        }

    shift_pa_v = abs(
        filtered_peaks["epa_v"] - raw_peaks["epa_v"]
    )

    shift_pc_v = abs(
        filtered_peaks["epc_v"] - raw_peaks["epc_v"]
    )

    max_shift_v = max(shift_pa_v, shift_pc_v)

    rel_chg_pa = (
        abs(filtered_peaks["ipa"] - raw_peaks["ipa"])
        / abs(raw_peaks["ipa"])
    )

    rel_chg_pc = (
        abs(filtered_peaks["ipc"] - raw_peaks["ipc"])
        / abs(raw_peaks["ipc"])
    )

    max_rel_chg_i = max(rel_chg_pa, rel_chg_pc)

    passed = (
        max_shift_v < cfg.max_peak_shift_v
        and max_rel_chg_i < cfg.max_peak_current_rel_change
    )

    return {
        "max_peak_shift_mv": max_shift_v * 1e3,
        "max_peak_current_chg_pct": max_rel_chg_i * 100.0,
        "within_targets": passed,
        "raw_peaks": raw_peaks,
        "filtered_peaks": filtered_peaks,
    }


def compute_repeat_rsd(values: list[float]) -> float:
    """
    Calculate sample relative standard deviation (%) across
    at least three valid repeat measurements.
    """
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]

    if len(arr) < 3:
        return np.nan

    mean = float(np.mean(arr))

    if mean == 0:
        return np.nan

    sample_std = float(np.std(arr, ddof=1))

    return float((sample_std / abs(mean)) * 100.0)

    


def analyze_single_csv(
    raw_csv_path: Path,
    filter_methods: tuple[str, ...] = ("savgol", "median"),
    save_plot: bool = True,
) -> dict:
    """
    Analyze one complete CV acquisition CSV.

    Uses acquisition metadata to identify the scan parameters and
    isolates the first CV cycle before peak extraction.
    """

    raw_csv_path = Path(raw_csv_path)
    meta_json_path = raw_csv_path.with_suffix(".meta.json")

    if not raw_csv_path.exists():
        raise FileNotFoundError(raw_csv_path)

    if not meta_json_path.exists():
        raise FileNotFoundError(
            f"Metadata file not found: {meta_json_path}"
        )

    with meta_json_path.open("r", encoding="utf-8") as f:
        metadata = json.load(f)

    acquisition = metadata.get("acquisition", {})
    scan_params = metadata.get("scan_params", {})

    if not acquisition.get("complete", False):
        raise ValueError(
            "This acquisition is incomplete. "
            "Partial scans must not be analyzed as complete CV scans."
        )

    required_params = (
        "v_start",
        "v_vertex",
        "v_end",
        "step_v",
        "n_cycles",
    )

    missing_params = [
        key for key in required_params if key not in scan_params
    ]

    if missing_params:
        raise ValueError(
            f"Missing scan metadata: {missing_params}"
        )

    v_start = float(scan_params["v_start"])
    v_vertex = float(scan_params["v_vertex"])
    v_end = float(scan_params["v_end"])
    step_v = float(scan_params["step_v"])
    n_cycles = int(scan_params["n_cycles"])

    df = pd.read_csv(raw_csv_path)

    required_columns = (
        "v_cmd_v",
        "v_meas_v",
        "current_ua",
        "t_rel_s",
    )

    missing_columns = [
        col for col in required_columns if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"CSV is missing required columns: {missing_columns}"
        )

    # Identify the expected first-cycle length using the same
    # sweep-generation function used by acquisition.
    first_cycle = build_sweep_potentials(
        v_start,
        v_vertex,
        v_end,
        step_v,
        n_cycles=1,
    )

    if len(df) < len(first_cycle):
        raise ValueError(
            "CSV contains fewer samples than expected for one complete cycle."
        )

    # Analyze only the first cycle.
    df = df.iloc[:len(first_cycle)].copy()

    pot = df["v_meas_v"].to_numpy(dtype=float)
    curr = df["current_ua"].to_numpy(dtype=float)
    v_cmd = df["v_cmd_v"].to_numpy(dtype=float)
    if not np.allclose(v_cmd, first_cycle, atol=1e-9, rtol=0):
        raise ValueError(
            "CSV commanded potentials do not match the expected first CV cycle."
        )
    t = df["t_rel_s"].to_numpy(dtype=float)

    if not (
        np.all(np.isfinite(pot))
        and np.all(np.isfinite(curr))
        and np.all(np.isfinite(v_cmd))
        and np.all(np.isfinite(t))
    ):
        raise ValueError("CSV contains non-finite measurement values.")

    if len(t) < 3 or np.any(np.diff(t) <= 0):
        raise ValueError(
            "Time values must be strictly increasing."
        )

    dt = float(np.median(np.diff(t)))

    raw_noise = estimate_noise_metrics(curr)
    noise_std = raw_noise["noise_sigma"]

    raw_peaks = extract_peaks(
        pot,
        curr,
        v_cmd,
        v_start,
        v_vertex,
        noise_std,
    )

    # Locate the first-cycle vertex using commanded potential.
    if v_vertex > v_start:
        vertex_idx = int(np.argmax(v_cmd))
    else:
        vertex_idx = int(np.argmin(v_cmd))

    # Split branches at the vertex. The vertex is included in both
    # branches for filtering, then included only once when rejoined.
    fwd_current = curr[:vertex_idx + 1]
    rev_current = curr[vertex_idx:]

    filter_results = {}
    filtered_traces = {}

    for method in filter_methods:
        try:
            filtered_fwd = apply_filter(
                fwd_current,
                dt_s=dt,
                method=method,
            )

            filtered_rev = apply_filter(
                rev_current,
                dt_s=dt,
                method=method,
            )

            filtered_current = np.concatenate(
                [filtered_fwd, filtered_rev[1:]]
            )

            filtered_traces[method] = filtered_current

            filtered_noise = estimate_noise_metrics(
                filtered_current
            )

            filter_validation = evaluate_filter_against_raw(
                pot,
                curr,
                filtered_current,
                v_cmd,
                v_start,
                v_vertex,
                noise_std,
            )

            filter_results[method] = {
                **filtered_noise,
                **filter_validation,
            }

        except (ValueError, RuntimeError) as exc:
            filter_results[method] = {
                "skipped": True,
                "reason": str(exc),
            }

    plot_path = None

    if save_plot:
        fig, ax = plt.subplots(figsize=(8, 4.8))

        ax.plot(
            pot,
            curr,
            color="0.65",
            lw=1.2,
            label="Raw (preserved)",
        )

        if "savgol" in filtered_traces:
            ax.plot(
                pot,
                filtered_traces["savgol"],
                lw=1.8,
                label="Savitzky-Golay",
            )

        if "median" in filtered_traces:
            ax.plot(
                pot,
                filtered_traces["median"],
                lw=1.5,
                label="Median",
            )

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
        "metadata_file": meta_json_path.name,
        "plot_file": plot_path.name if plot_path else None,
        "cycles_analyzed": 1,
        "raw_noise": raw_noise,
        "raw_peaks": raw_peaks,
        "filters": filter_results,
    }
