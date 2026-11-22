
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scipy.fft import dct, idct
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
        snr_db = (
            float(20.0 * np.log10(snr_linear))
            if snr_linear > 0
            else np.nan
        )
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


def dct_spectrum(
    current: np.ndarray,
    step_v: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Calculate the DCT-II power spectrum of one CV branch.

    The frequency axis is expressed in cycles per volt, not Hz.
    The branch is treated independently, avoiding an artificial
    connection between the end of one branch and the start of another.
    """
    y = np.asarray(current, dtype=float)

    if y.ndim != 1:
        raise ValueError("Current data must be one-dimensional.")

    if len(y) < 3:
        raise ValueError("At least three samples are required for a DCT.")

    if not np.all(np.isfinite(y)):
        raise ValueError("Current data contains non-finite values.")

    if not np.isfinite(step_v) or step_v <= 0:
        raise ValueError("step_v must be finite and positive.")

    coefficients = dct(y, type=2, norm="ortho")
    n = len(y)

    frequencies = np.arange(n, dtype=float) / (2.0 * n * step_v)
    power = np.square(np.abs(coefficients))

    upper_half = np.abs(coefficients[n // 2:])
    noise_sigma = float(np.median(upper_half) / 0.6745)

    return frequencies, power, noise_sigma


def estimate_dct_cutoff(
    current: np.ndarray,
    step_v: float,
    margin_bins: int = 2,
) -> dict:
    """
    Estimate a DCT low-pass cutoff from the high-frequency noise floor.

    The power spectrum is smoothed over five bins. The cutoff is placed
    beyond the last bin whose smoothed power exceeds five times the
    estimated noise variance, with an additional bin margin.
    """
    frequencies, power, noise_sigma = dct_spectrum(current, step_v)

    if margin_bins < 0:
        raise ValueError("margin_bins cannot be negative.")

    if not np.isfinite(noise_sigma):
        raise ValueError("Could not estimate the DCT noise floor.")

    noise_power = noise_sigma ** 2

    smoothed_power = np.convolve(
        power,
        np.ones(5, dtype=float) / 5.0,
        mode="same",
    )

    threshold = 5.0 * noise_power

    # Ignore the DC coefficient when looking for signal frequencies.
    significant = np.flatnonzero(
        smoothed_power[1:] > threshold
    ) + 1

    if len(significant) == 0:
        last_significant = 1
    else:
        last_significant = int(significant[-1])

    cutoff_index = min(
        last_significant + margin_bins,
        len(frequencies) - 1,
    )

    return {
        "cutoff_index": cutoff_index,
        "cutoff_cycles_per_v": float(frequencies[cutoff_index]),
        "noise_sigma": noise_sigma,
        "noise_power": noise_power,
        "threshold_power": threshold,
        "frequencies": frequencies,
        "power": power,
        "smoothed_power": smoothed_power,
    }


def dct_lowpass(
    current: np.ndarray,
    step_v: float,
    cutoff_cycles_per_v: float | None = None,
) -> tuple[np.ndarray, dict]:
    """
    Apply a DCT-II low-pass filter to one CV branch.

    If cutoff_cycles_per_v is None, the cutoff is estimated
    automatically. A raised-cosine transition attenuates coefficients
    above the cutoff instead of abruptly removing them.
    """
    y = np.asarray(current, dtype=float)

    if y.ndim != 1:
        raise ValueError("Current data must be one-dimensional.")

    if len(y) < 3:
        raise ValueError("At least three samples are required for DCT filtering.")

    if not np.all(np.isfinite(y)):
        raise ValueError("Current data contains non-finite values.")

    if not np.isfinite(step_v) or step_v <= 0:
        raise ValueError("step_v must be finite and positive.")

    coefficients = dct(y, type=2, norm="ortho")
    n = len(y)

    frequencies = np.arange(n, dtype=float) / (2.0 * n * step_v)

    if cutoff_cycles_per_v is None:
        cutoff_info = estimate_dct_cutoff(y, step_v)
        cutoff_index = cutoff_info["cutoff_index"]
        cutoff = cutoff_info["cutoff_cycles_per_v"]
        noise_sigma = cutoff_info["noise_sigma"]
    else:
        if (
            not np.isfinite(cutoff_cycles_per_v)
            or cutoff_cycles_per_v <= 0
        ):
            raise ValueError("fft_cutoff must be finite and positive.")

        if cutoff_cycles_per_v >= frequencies[-1]:
            raise ValueError(
                "fft_cutoff must be below the maximum DCT frequency."
            )

        cutoff_index = int(
            np.searchsorted(frequencies, cutoff_cycles_per_v)
        )
        cutoff_index = max(1, min(cutoff_index, n - 1))
        cutoff = float(frequencies[cutoff_index])

        upper_half = np.abs(coefficients[n // 2:])
        noise_sigma = float(np.median(upper_half) / 0.6745)

    # The transition starts at the selected cutoff and extends over
    # several bins. This avoids the ringing associated with a brick wall.
    transition_bins = max(2, int(np.ceil(0.10 * cutoff_index)))
    transition_end = min(
        cutoff_index + transition_bins,
        n - 1,
    )

    weights = np.ones(n, dtype=float)

    if transition_end > cutoff_index:
        transition_indices = np.arange(
            cutoff_index,
            transition_end + 1,
        )

        phase = (
            transition_indices - cutoff_index
        ) / (transition_end - cutoff_index)

        weights[transition_indices] = (
            0.5 * (1.0 + np.cos(np.pi * phase))
        )

    weights[transition_end + 1:] = 0.0

    filtered_coefficients = coefficients * weights
    filtered = idct(filtered_coefficients, type=2, norm="ortho")

    diagnostic = {
        "cutoff_cycles_per_v": cutoff,
        "cutoff_index": cutoff_index,
        "noise_sigma": noise_sigma,
        "frequencies": frequencies,
        "power": np.square(np.abs(coefficients)),
        "weights": weights,
    }

    return filtered, diagnostic


def apply_filter(
    current: np.ndarray,
    dt_s: float,
    method: str = "savgol",
    sg_window: int = 11,
    sg_poly: int = 3,
    median_kernel: int = 5,
    lowpass_cutoff_hz: float = 5.0,
    notch_freq_hz: float = 50.0,
    step_v: float | None = None,
    fft_cutoff: float | None = None,
) -> np.ndarray:
    """
    Apply a selectable filter to a copy of the current array.

    The fft method uses a DCT-II low-pass filter. Its frequency axis is
    in cycles per volt, so it requires step_v rather than dt_s.
    """
    y = np.asarray(current, dtype=float).copy()

    if y.ndim != 1:
        raise ValueError("Current data must be one-dimensional.")

    if len(y) < 3:
        return y

    if method == "raw" or method is None:
        return y

    if method == "fft":
        if step_v is None:
            raise ValueError("step_v is required for the fft filter.")

        filtered, _ = dct_lowpass(
            y,
            step_v=step_v,
            cutoff_cycles_per_v=fft_cutoff,
        )
        return filtered

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

    Peak classification follows the scan direction:
    - If the scan starts below its vertex, the forward branch is anodic.
    - If the scan starts above its vertex, the reverse branch is anodic.

    Peak potentials are reported using measured potential.
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

    result = {
        "epa_v": np.nan,
        "ipa": np.nan,
        "epc_v": np.nan,
        "ipc": np.nan,
    }

    if v_vertex > v_start:
        anodic_signal = fwd_i
        cathodic_signal = -rev_i
        anodic_potential = fwd_e
        cathodic_potential = rev_e
        anodic_branch_current = fwd_i
        cathodic_branch_current = rev_i
    else:
        anodic_signal = rev_i
        cathodic_signal = -fwd_i
        anodic_potential = rev_e
        cathodic_potential = fwd_e
        anodic_branch_current = rev_i
        cathodic_branch_current = fwd_i

    anodic_candidates, anodic_props = find_peaks(
        anodic_signal,
        prominence=prominence,
    )

    cathodic_candidates, cathodic_props = find_peaks(
        cathodic_signal,
        prominence=prominence,
    )

    if len(anodic_candidates) > 0:
        best = int(np.argmax(anodic_props["prominences"]))
        anodic_idx = int(anodic_candidates[best])

        result["epa_v"] = float(anodic_potential[anodic_idx])
        result["ipa"] = float(anodic_branch_current[anodic_idx])

    if len(cathodic_candidates) > 0:
        best = int(np.argmax(cathodic_props["prominences"]))
        cathodic_idx = int(cathodic_candidates[best])

        result["epc_v"] = float(cathodic_potential[cathodic_idx])
        result["ipc"] = float(cathodic_branch_current[cathodic_idx])

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
    """Compare filtered peak positions and currents against raw peaks."""

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


def save_spectrum_diagnostic(
    spectrum_path: Path,
    forward_diagnostic: dict,
    reverse_diagnostic: dict,
) -> None:
    """Save DCT power spectra for the forward and reverse branches."""

    fig, ax = plt.subplots(figsize=(10, 6))

    for label, diagnostic in (
        ("Forward branch", forward_diagnostic),
        ("Reverse branch", reverse_diagnostic),
    ):
        frequencies = diagnostic["frequencies"]
        power = diagnostic["power"]
        cutoff = diagnostic["cutoff_cycles_per_v"]
        noise_sigma = diagnostic["noise_sigma"]

        positive = frequencies > 0
        epsilon = np.finfo(float).tiny

        ax.plot(
            frequencies[positive],
            np.log10(np.maximum(power[positive], epsilon)),
            linewidth=1.2,
            label=label,
        )

        if np.isfinite(noise_sigma):
            noise_power = max(noise_sigma ** 2, epsilon)
            ax.axhline(
                np.log10(noise_power),
                linestyle=":",
                linewidth=1.0,
                alpha=0.7,
            )

        ax.axvline(
            cutoff,
            linestyle="--",
            linewidth=1.2,
            label=f"{label} cutoff: {cutoff:.2f} cycles/V",
        )

    ax.set_xlabel("Spatial frequency [cycles/V]")
    ax.set_ylabel("Log10 DCT power")
    ax.set_title("DCT spectrum and selected low-pass cutoffs")
    ax.legend()
    ax.grid(False)

    fig.tight_layout()
    fig.savefig(spectrum_path, dpi=200)
    plt.close(fig)


def analyze_single_csv(
    raw_csv_path: Path,
    filter_methods: tuple[str, ...] = ("savgol", "median", "fft"),
    save_plot: bool = True,
    fft_cutoff: float | None = None,
) -> dict:
    """
    Analyze one complete CV acquisition CSV.

    Uses acquisition metadata to identify scan parameters and isolates
    the first CV cycle before peak extraction.

    The fft filter is a DCT-II low-pass filter applied independently
    to the forward and reverse branches.
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
    program_config = metadata.get("program_config", {})

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

    df = df.iloc[:len(first_cycle)].copy()

    pot = df["v_meas_v"].to_numpy(dtype=float)
    curr = df["current_ua"].to_numpy(dtype=float)
    v_cmd = df["v_cmd_v"].to_numpy(dtype=float)
    t = df["t_rel_s"].to_numpy(dtype=float)

    if not np.allclose(v_cmd, first_cycle, atol=1e-9, rtol=0):
        raise ValueError(
            "CSV commanded potentials do not match the expected first CV cycle."
        )

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

    if v_vertex > v_start:
        vertex_idx = int(np.argmax(v_cmd))
    else:
        vertex_idx = int(np.argmin(v_cmd))

    fwd_current = curr[:vertex_idx + 1]
    rev_current = curr[vertex_idx:]

    filter_results = {}
    filtered_traces = {}
    fft_diagnostics = {}

    for method in filter_methods:
        try:
            if method == "fft":
                filtered_fwd, forward_diagnostic = dct_lowpass(
                    fwd_current,
                    step_v=step_v,
                    cutoff_cycles_per_v=fft_cutoff,
                )

                filtered_rev, reverse_diagnostic = dct_lowpass(
                    rev_current,
                    step_v=step_v,
                    cutoff_cycles_per_v=fft_cutoff,
                )

                fft_diagnostics = {
                    "forward": forward_diagnostic,
                    "reverse": reverse_diagnostic,
                }
            else:
                filtered_fwd = apply_filter(
                    fwd_current,
                    dt_s=dt,
                    method=method,
                    step_v=step_v,
                )

                filtered_rev = apply_filter(
                    rev_current,
                    dt_s=dt,
                    method=method,
                    step_v=step_v,
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

            if method == "fft":
                filter_results[method]["forward_cutoff_cycles_per_v"] = (
                    forward_diagnostic["cutoff_cycles_per_v"]
                )
                filter_results[method]["reverse_cutoff_cycles_per_v"] = (
                    reverse_diagnostic["cutoff_cycles_per_v"]
                )

        except (ValueError, RuntimeError) as exc:
            filter_results[method] = {
                "skipped": True,
                "reason": str(exc),
            }

    spectrum_path = None

    if fft_diagnostics:
        spectrum_path = raw_csv_path.with_name(
            f"{raw_csv_path.stem}_spectrum.png"
        )

        save_spectrum_diagnostic(
            spectrum_path,
            fft_diagnostics["forward"],
            fft_diagnostics["reverse"],
        )

    plot_path = None

    if save_plot:
        fig, ax = plt.subplots(figsize=(10, 7), facecolor="white")
        ax.set_facecolor("white")

        area_cm2 = program_config.get("electrode_area_cm2")
        if area_cm2 is not None:
            area_cm2 = float(area_cm2)

        if area_cm2 is not None and area_cm2 > 0.0:
            plot_curr = curr / (area_cm2 * 1000.0)
            plot_filtered = (
                filtered_traces.get("savgol", curr)
                / (area_cm2 * 1000.0)
            )
            y_label = "Current density [mA cm$^{-2}$]"
        else:
            plot_curr = curr
            plot_filtered = filtered_traces.get("savgol", curr)
            y_label = "Current [µA]"

        fwd_order = np.argsort(pot[:vertex_idx + 1])
        rev_order = np.argsort(pot[vertex_idx:])[::-1]

        fwd_pot = pot[:vertex_idx + 1][fwd_order]
        rev_pot = pot[vertex_idx:][rev_order]
        fwd_curr = plot_curr[:vertex_idx + 1][fwd_order]
        rev_curr = plot_curr[vertex_idx:][rev_order]
        fwd_filtered = plot_filtered[:vertex_idx + 1][fwd_order]
        rev_filtered = plot_filtered[vertex_idx:][rev_order]

        ax.plot(
            fwd_pot,
            fwd_curr,
            color="black",
            linewidth=1.0,
            alpha=0.9,
            zorder=2,
        )
        ax.plot(
            rev_pot,
            rev_curr,
            color="black",
            linewidth=1.0,
            alpha=0.9,
            zorder=2,
        )
        ax.plot(
            fwd_pot,
            fwd_filtered,
            color="red",
            linewidth=1.8,
            alpha=0.95,
            zorder=3,
        )
        ax.plot(
            rev_pot,
            rev_filtered,
            color="red",
            linewidth=1.8,
            alpha=0.95,
            zorder=3,
        )

        ax.set_xlabel("Potential [V]", fontsize=20, fontweight="bold")
        ax.set_ylabel(y_label, fontsize=20, fontweight="bold")
        ax.xaxis.set_label_coords(0.5, -0.10)
        ax.yaxis.set_label_coords(-0.12, 0.5)

        ax.tick_params(
            axis="both",
            which="major",
            labelsize=18,
            width=1.5,
            length=8,
            direction="out",
        )

        for spine in ax.spines.values():
            spine.set_linewidth(1.5)
            spine.set_color("black")

        x_min = min(v_start, v_vertex, v_end)
        x_max = max(v_start, v_vertex, v_end)
        x_span = max(x_max - x_min, step_v)

        x_padding = 0.04 * x_span
        ax.set_xlim(
            x_min - x_padding,
            x_max + x_padding,
        )

        x_ticks = np.linspace(x_min, x_max, 6)
        ax.set_xticks(x_ticks)
        ax.xaxis.set_major_formatter(
            plt.FuncFormatter(lambda value, _: f"{value:.2f}")
        )

        y_min = float(np.min(plot_curr))
        y_max = float(np.max(plot_curr))
        y_span = max(y_max - y_min, 1e-9)
        ax.set_ylim(
            y_min - 0.12 * y_span,
            y_max + 0.12 * y_span,
        )

        ax.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda value, _: f"{value:g}")
        )

        calibrated = program_config.get("calibrated", False)

        if not calibrated:
            ax.text(
                0.02,
                0.98,
                "WARNING: Data is marked as not calibrated",
                transform=ax.transAxes,
                ha="left",
                va="top",
                color="red",
                fontsize=11,
                fontweight="bold",
                bbox={
                    "facecolor": "white",
                    "edgecolor": "red",
                    "alpha": 0.9,
                },
            )

        ax.grid(False)

        epa = raw_peaks["epa_v"]
        ipa = raw_peaks["ipa"]
        epc = raw_peaks["epc_v"]
        ipc = raw_peaks["ipc"]

        if area_cm2 is not None and area_cm2 > 0.0:
            ipa = ipa / (area_cm2 * 1000.0)
            ipc = ipc / (area_cm2 * 1000.0)

        y_low, y_high = ax.get_ylim()

        if np.isfinite(epa) and np.isfinite(ipa):
            ax.vlines(
                epa,
                y_low,
                ipa,
                colors="black",
                linestyles=(0, (7, 5)),
                linewidth=1.6,
            )
            ax.hlines(
                ipa,
                ax.get_xlim()[0],
                epa,
                colors="black",
                linestyles=(0, (7, 5)),
                linewidth=1.6,
            )
            ax.annotate(
                "E$_{PA}$",
                xy=(epa, y_high),
                xytext=(0, -2),
                textcoords="offset points",
                ha="center",
                va="top",
                fontsize=22,
                fontweight="bold",
            )
            ax.annotate(
                "j$_{PA}$",
                xy=(epa, ipa),
                xytext=(12, 0),
                textcoords="offset points",
                ha="left",
                va="center",
                fontsize=20,
                fontweight="bold",
            )

        if np.isfinite(epc) and np.isfinite(ipc):
            ax.vlines(
                epc,
                y_low,
                ipc,
                colors="black",
                linestyles=(0, (7, 5)),
                linewidth=1.6,
            )
            ax.hlines(
                ipc,
                ax.get_xlim()[0],
                epc,
                colors="black",
                linestyles=(0, (7, 5)),
                linewidth=1.6,
            )
            ax.annotate(
                "j$_{PC}$",
                xy=(epc, ipc),
                xytext=(12, -18),
                textcoords="offset points",
                ha="left",
                va="center",
                fontsize=20,
                fontweight="bold",
            )
            ax.annotate(
                "E$_{PC}$",
                xy=(epc, y_low),
                xytext=(0, -14),
                textcoords="offset points",
                ha="center",
                va="top",
                fontsize=22,
                fontweight="bold",
            )

        fig.tight_layout(pad=1.0)

        plot_path = raw_csv_path.with_suffix(".png")
        fig.savefig(plot_path, dpi=200)
        plt.close(fig)

    return {
        "file": raw_csv_path.name,
        "metadata_file": meta_json_path.name,
        "plot_file": plot_path.name if plot_path else None,
        "spectrum_file": spectrum_path.name if spectrum_path else None,
        "simulated": bool(metadata.get("simulated", False)),
        "cycles_analyzed": 1,
        "raw_noise": raw_noise,
        "raw_peaks": raw_peaks,
        "filters": filter_results,
    }







