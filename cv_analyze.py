
from __future__ import annotations

from pathlib import Path
import json
import warnings

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
from fft_filter import dct_spectrum, estimate_cutoff, dct_lowpass


# Illustrative double-layer capacitance.
# This is an assumption for model-based correction, not a measured value.
DEFAULT_CDL_UF_CM2 = 20.0


def estimate_noise_metrics(current: np.ndarray) -> dict:
    """Estimate noise using the MAD of successive current differences."""
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
    mad = np.median(np.abs(diff - np.median(diff)))
    noise_sigma = float(1.4826 * mad / np.sqrt(2.0))
    signal_peak = float(np.max(np.abs(current)))

    if noise_sigma > 0:
        snr_linear = signal_peak / noise_sigma
        snr_db = float(20.0 * np.log10(snr_linear))
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
    dt_s: float | None = None,
    method: str = "savgol",
    sg_window: int = 11,
    sg_poly: int = 3,
    median_kernel: int = 5,
    lowpass_cutoff_hz: float = 5.0,
    notch_freq_hz: float = 50.0,
    step_v: float | None = None,
    fft_cutoff: float | None = None,
) -> np.ndarray:
    """Apply a selectable filter to a copy of the current array."""
    y = np.asarray(current, dtype=float).copy()

    if y.ndim != 1:
        raise ValueError("Current data must be one-dimensional.")

    if not np.all(np.isfinite(y)):
        raise ValueError("Current data contains non-finite values.")

    if len(y) < 3 or method == "raw" or method is None:
        return y

    if method == "savgol":
        if sg_poly < 0 or sg_window < 3:
            raise ValueError("Invalid Savitzky-Golay parameters.")

        win = min(sg_window, len(y))
        if win % 2 == 0:
            win -= 1

        if win <= sg_poly:
            return y

        return savgol_filter(
            y, window_length=win, polyorder=sg_poly, mode="nearest"
        )

    if method == "median":
        if median_kernel < 1:
            raise ValueError("median_kernel must be positive.")

        return median_filter(
            y, size=min(median_kernel, len(y)), mode="nearest"
        )

    if method == "fft":
        if step_v is None or not np.isfinite(step_v) or step_v <= 0:
            raise ValueError("A finite, positive step_v is required.")

        return dct_lowpass(
            y, step_v=step_v, cutoff_cycles_per_v=fft_cutoff
        )

    if dt_s is None or dt_s <= 0 or not np.isfinite(dt_s):
        raise ValueError("dt_s must be finite and positive.")

    fs = 1.0 / dt_s
    nyq = fs / 2.0

    if method == "lowpass":
        if not 0 < lowpass_cutoff_hz < nyq:
            raise ValueError("Low-pass cutoff must be below Nyquist.")

        b, a = butter(
            N=3, Wn=lowpass_cutoff_hz, btype="low", fs=fs
        )
        return filtfilt(b, a, y)

    if method == "notch50":
        if not 0 < notch_freq_hz < nyq:
            raise ValueError(
                f"Cannot apply {notch_freq_hz} Hz notch at {fs:.2f} Hz sampling."
            )

        b, a = iirnotch(w0=notch_freq_hz, Q=30.0, fs=fs)
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
    """Extract anodic and cathodic peaks from the first CV cycle."""
    potential_v = np.asarray(potential_v, dtype=float)
    current = np.asarray(current, dtype=float)
    v_cmd_v = np.asarray(v_cmd_v, dtype=float)

    if not (len(potential_v) == len(current) == len(v_cmd_v)):
        raise ValueError("Potential and current arrays must have equal lengths.")

    if len(current) < 5:
        raise ValueError("Not enough samples for peak extraction.")

    if v_vertex > v_start:
        vertex_idx = int(np.argmax(v_cmd_v))
        anodic_current = current[:vertex_idx + 1]
        anodic_potential = potential_v[:vertex_idx + 1]
        cathodic_current = current[vertex_idx:]
        cathodic_potential = potential_v[vertex_idx:]
    elif v_vertex < v_start:
        vertex_idx = int(np.argmin(v_cmd_v))
        cathodic_current = current[:vertex_idx + 1]
        cathodic_potential = potential_v[:vertex_idx + 1]
        anodic_current = current[vertex_idx:]
        anodic_potential = potential_v[vertex_idx:]
    else:
        raise ValueError("v_vertex and v_start cannot be equal.")

    prominence = (
        5.0 * noise_std
        if np.isfinite(noise_std) and noise_std > 0
        else 0.0
    )

    anodic_candidates, anodic_props = find_peaks(
        anodic_current, prominence=prominence
    )
    cathodic_candidates, cathodic_props = find_peaks(
        -cathodic_current, prominence=prominence
    )

    result = {
        "epa_v": np.nan,
        "ipa": np.nan,
        "epc_v": np.nan,
        "ipc": np.nan,
    }

    if len(anodic_candidates):
        idx = int(anodic_candidates[
            np.argmax(anodic_props["prominences"])
        ])
        result["epa_v"] = float(anodic_potential[idx])
        result["ipa"] = float(anodic_current[idx])

    if len(cathodic_candidates):
        idx = int(cathodic_candidates[
            np.argmax(cathodic_props["prominences"])
        ])
        result["epc_v"] = float(cathodic_potential[idx])
        result["ipc"] = float(cathodic_current[idx])

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
        potential_v, raw_current, v_cmd_v,
        v_start, v_vertex, noise_std
    )
    filtered_peaks = extract_peaks(
        potential_v, filt_current, v_cmd_v,
        v_start, v_vertex, noise_std
    )

    keys = ("epa_v", "ipa", "epc_v", "ipc")
    all_valid = all(
        np.isfinite(raw_peaks[k]) and np.isfinite(filtered_peaks[k])
        for k in keys
    )

    currents_valid = (
        np.isfinite(raw_peaks["ipa"])
        and np.isfinite(raw_peaks["ipc"])
        and raw_peaks["ipa"] != 0
        and raw_peaks["ipc"] != 0
    )

    if not all_valid or not currents_valid:
        return {
            "max_peak_shift_mv": np.nan,
            "max_peak_current_chg_pct": np.nan,
            "within_targets": False,
            "raw_peaks": raw_peaks,
            "filtered_peaks": filtered_peaks,
        }

    shift = max(
        abs(filtered_peaks["epa_v"] - raw_peaks["epa_v"]),
        abs(filtered_peaks["epc_v"] - raw_peaks["epc_v"]),
    )

    current_change = max(
        abs(filtered_peaks["ipa"] - raw_peaks["ipa"])
        / abs(raw_peaks["ipa"]),
        abs(filtered_peaks["ipc"] - raw_peaks["ipc"])
        / abs(raw_peaks["ipc"]),
    )

    return {
        "max_peak_shift_mv": shift * 1e3,
        "max_peak_current_chg_pct": current_change * 100.0,
        "within_targets": (
            shift < cfg.max_peak_shift_v
            and current_change < cfg.max_peak_current_rel_change
        ),
        "raw_peaks": raw_peaks,
        "filtered_peaks": filtered_peaks,
    }


def compute_repeat_rsd(values: list[float]) -> float:
    """Calculate sample relative standard deviation across repeats."""
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]

    if len(arr) < 3:
        return np.nan

    mean = float(np.mean(arr))
    if mean == 0:
        return np.nan

    return float(np.std(arr, ddof=1) / abs(mean) * 100.0)


def save_spectrum_diagnostic(
    forward_current: np.ndarray,
    reverse_current: np.ndarray,
    step_v: float,
    output_path: Path,
    manual_cutoff: float | None = None,
) -> dict:
    """Save forward and reverse DCT spectra in one diagnostic figure."""
    branches = {
        "Forward branch": forward_current,
        "Reverse branch": reverse_current,
    }

    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    diagnostics = {}

    try:
        for ax, (name, branch) in zip(axes, branches.items()):
            frequencies, coefficients = dct_spectrum(branch, step_v)
            estimated_cutoff, sigma = estimate_cutoff(branch, step_v)

            chosen_cutoff = (
                float(manual_cutoff)
                if manual_cutoff is not None
                else float(estimated_cutoff)
            )

            power = coefficients**2
            tiny = np.finfo(float).tiny

            ax.plot(
                frequencies[1:],
                np.log10(np.maximum(power[1:], tiny)),
                color="black",
                linewidth=1.0,
            )
            ax.axhline(
                np.log10(max(sigma**2, tiny)),
                color="red",
                linestyle="--",
                label="Estimated noise power floor",
            )
            ax.axvline(
                chosen_cutoff,
                color="blue",
                linestyle=":",
                label=f"Cutoff = {chosen_cutoff:.2f} cycles/V",
            )

            ax.set_title(name)
            ax.set_xlabel("Spatial frequency [cycles/V]")
            ax.set_ylabel("log10(DCT coefficient power)")
            ax.legend()

            diagnostics[name.lower().replace(" ", "_")] = {
                "noise_sigma": float(sigma),
                "noise_power": float(sigma**2),
                "estimated_cutoff_cycles_per_v": float(estimated_cutoff),
                "chosen_cutoff_cycles_per_v": chosen_cutoff,
            }

        fig.tight_layout()
        fig.savefig(output_path, dpi=200)
    finally:
        plt.close(fig)

    return diagnostics


def estimate_capacitive_background(
    n_samples: int,
    vertex_idx: int,
    v_start: float,
    v_vertex: float,
    scan_rate_v_per_s: float,
    electrode_area_cm2: float,
    cdl_uf_cm2: float = DEFAULT_CDL_UF_CM2,
) -> np.ndarray:
    """
    Estimate ideal double-layer charging current in microamps.

    I = Cdl * A * scan_rate.

    The sign follows the direction of the potential sweep:
    increasing potential gives positive charging current;
    decreasing potential gives negative charging current.
    """
    if electrode_area_cm2 <= 0 or scan_rate_v_per_s <= 0 or cdl_uf_cm2 <= 0:
        raise ValueError("Area, scan rate and capacitance must be positive.")

    forward_sign = float(np.sign(v_vertex - v_start))
    reverse_sign = -forward_sign

    magnitude_ua = (
        cdl_uf_cm2
        * electrode_area_cm2
        * scan_rate_v_per_s
    )

    background = np.empty(n_samples, dtype=float)
    background[:vertex_idx + 1] = forward_sign * magnitude_ua
    background[vertex_idx + 1:] = reverse_sign * magnitude_ua

    return background


def save_comparison_plot(
    potential: np.ndarray,
    before_current: np.ndarray,
    after_current: np.ndarray,
    v_start: float,
    v_vertex: float,
    v_end: float,
    step_v: float,
    output_path: Path,
    title: str,
    y_label: str,
) -> None:
    """Save one CV plot using consistent axes for before/after comparison."""
    fig, ax = plt.subplots(figsize=(10, 7), facecolor="white")
    ax.set_facecolor("white")

    # Split at the turning point, retaining it in both branches.
    vertex_idx = int(np.argmax(potential)) if v_vertex > v_start else int(np.argmin(potential))

    # The actual vertex is identified from the commanded sweep geometry.
    # The caller supplies potential in acquisition order.
    forward_p = potential[:vertex_idx + 1]
    reverse_p = potential[vertex_idx:]
    forward_before = before_current[:vertex_idx + 1]
    reverse_before = before_current[vertex_idx:]
    forward_after = after_current[:vertex_idx + 1]
    reverse_after = after_current[vertex_idx:]

    ax.plot(forward_p, forward_before, color="0.55", linewidth=1.0,
            label="Forward — before")
    ax.plot(reverse_p, reverse_before, color="0.25", linewidth=1.0,
            label="Reverse — before")
    ax.plot(forward_p, forward_after, color="tab:blue", linewidth=1.7,
            label="Forward — after")
    ax.plot(reverse_p, reverse_after, color="tab:red", linewidth=1.7,
            label="Reverse — after")

    x_min = min(v_start, v_vertex, v_end)
    x_max = max(v_start, v_vertex, v_end)
    x_span = max(x_max - x_min, step_v)

    all_y = np.concatenate([before_current, after_current])
    y_min = float(np.min(all_y))
    y_max = float(np.max(all_y))
    y_span = max(y_max - y_min, 1e-9)

    ax.set_xlim(x_min - 0.05 * x_span, x_max + 0.05 * x_span)
    ax.set_ylim(y_min - 0.12 * y_span, y_max + 0.12 * y_span)
    ax.set_xticks(np.linspace(x_min, x_max, 6))

    ax.set_xlabel("Potential [V]", fontsize=15, fontweight="bold")
    ax.set_ylabel(y_label, fontsize=15, fontweight="bold")
    ax.set_title(title, fontsize=14)
    ax.grid(False)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def analyze_single_csv(
    raw_csv_path: Path,
    filter_methods: tuple[str, ...] = ("savgol", "median", "fft"),
    save_plot: bool = True,
    fft_cutoff: float | None = None,
    cdl_uf_cm2: float = DEFAULT_CDL_UF_CM2,
) -> dict:
    """
    Analyze one complete CV acquisition.

    Processing order:
    1. Preserve raw current.
    2. Apply random-noise filtering separately to each branch.
    3. Estimate and subtract capacitive background current.
    4. Save before/after plots and processed data.
    """
    raw_csv_path = Path(raw_csv_path)
    meta_json_path = raw_csv_path.with_suffix(".meta.json")

    if not raw_csv_path.exists():
        raise FileNotFoundError(raw_csv_path)
    if not meta_json_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {meta_json_path}")

    with meta_json_path.open("r", encoding="utf-8") as f:
        metadata = json.load(f)

    acquisition = metadata.get("acquisition", {})
    scan_params = metadata.get("scan_params", {})
    program_config = metadata.get("program_config", {})

    if not acquisition.get("complete", False):
        raise ValueError("Incomplete acquisition cannot be analyzed.")

    required_params = ("v_start", "v_vertex", "v_end", "step_v", "n_cycles")
    missing = [key for key in required_params if key not in scan_params]
    if missing:
        raise ValueError(f"Missing scan metadata: {missing}")

    v_start = float(scan_params["v_start"])
    v_vertex = float(scan_params["v_vertex"])
    v_end = float(scan_params["v_end"])
    step_v = float(scan_params["step_v"])

    if not np.isfinite(step_v) or step_v <= 0:
        raise ValueError("Metadata step_v must be finite and positive.")
    if v_start == v_vertex:
        raise ValueError("v_vertex and v_start cannot be equal.")

    df = pd.read_csv(raw_csv_path)
    required_columns = ("v_cmd_v", "v_meas_v", "current_ua", "t_rel_s")
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}")

    first_cycle = build_sweep_potentials(
        v_start, v_vertex, v_end, step_v, n_cycles=1
    )
    if len(df) < len(first_cycle):
        raise ValueError("CSV contains fewer samples than one complete cycle.")

    df = df.iloc[:len(first_cycle)].copy()

    pot = df["v_meas_v"].to_numpy(dtype=float)
    curr = df["current_ua"].to_numpy(dtype=float)
    v_cmd = df["v_cmd_v"].to_numpy(dtype=float)
    t = df["t_rel_s"].to_numpy(dtype=float)

    if not np.allclose(v_cmd, first_cycle, atol=1e-9, rtol=0):
        raise ValueError("Commanded potentials do not match the expected CV cycle.")
    if not all(np.all(np.isfinite(x)) for x in (pot, curr, v_cmd, t)):
        raise ValueError("CSV contains non-finite measurements.")
    if len(t) < 3 or np.any(np.diff(t) <= 0):
        raise ValueError("Time values must be strictly increasing.")

    dt = float(np.median(np.diff(t)))
    raw_noise = estimate_noise_metrics(curr)
    noise_std = raw_noise["noise_sigma"]

    if v_vertex > v_start:
        vertex_idx = int(np.argmax(v_cmd))
    else:
        vertex_idx = int(np.argmin(v_cmd))

    forward = curr[:vertex_idx + 1]
    reverse = curr[vertex_idx:]

    filter_results = {}
    filtered_traces = {}

    for method in filter_methods:
        try:
            filtered_forward = apply_filter(
                forward, dt_s=dt, method=method,
                step_v=step_v, fft_cutoff=fft_cutoff
            )
            filtered_reverse = apply_filter(
                reverse, dt_s=dt, method=method,
                step_v=step_v, fft_cutoff=fft_cutoff
            )
            filtered = np.concatenate([
                filtered_forward, filtered_reverse[1:]
            ])
            filtered_traces[method] = filtered

            filtered_noise = estimate_noise_metrics(filtered)
            validation = evaluate_filter_against_raw(
                pot, curr, filtered, v_cmd,
                v_start, v_vertex, noise_std
            )
            filter_results[method] = {
                **filtered_noise,
                **validation,
            }
        except (ValueError, RuntimeError) as exc:
            filter_results[method] = {
                "skipped": True,
                "reason": str(exc),
            }

    # Prefer DCT for the requested random-noise reduction.
    if "fft" in filtered_traces:
        noise_reduced = filtered_traces["fft"]
        selected_filter = "fft"
    elif "savgol" in filtered_traces:
        noise_reduced = filtered_traces["savgol"]
        selected_filter = "savgol"
    elif "median" in filtered_traces:
        noise_reduced = filtered_traces["median"]
        selected_filter = "median"
    else:
        noise_reduced = curr.copy()
        selected_filter = "raw"

    # Resolve scan rate from this acquisition's metadata.
    scan_rate = scan_params.get("scan_rate_v_per_s")
    if scan_rate is None:
        scan_rate = acquisition.get("achieved_scan_rate_v_per_s")
    if scan_rate is None:
        scan_rate = acquisition.get("requested_scan_rate_v_per_s")

    area_cm2 = program_config.get("electrode_area_cm2")
    calibrated = bool(program_config.get("calibrated", False))

    if area_cm2 is not None:
        area_cm2 = float(area_cm2)
        if not np.isfinite(area_cm2) or area_cm2 <= 0:
            area_cm2 = None

    background_subtracted = False
    background_reason = None
    background_current = np.zeros_like(curr)

    if area_cm2 is None:
        background_reason = (
            "No valid electrode area in acquisition metadata; "
            "capacitive background subtraction was not performed."
        )
    elif scan_rate is None or not np.isfinite(float(scan_rate)) or float(scan_rate) <= 0:
        background_reason = (
            "No valid scan rate in acquisition metadata; "
            "capacitive background subtraction was not performed."
        )
    else:
        background_current = estimate_capacitive_background(
            n_samples=len(curr),
            vertex_idx=vertex_idx,
            v_start=v_start,
            v_vertex=v_vertex,
            scan_rate_v_per_s=float(scan_rate),
            electrode_area_cm2=area_cm2,
            cdl_uf_cm2=cdl_uf_cm2,
        )
        background_subtracted = True

    corrected_current = noise_reduced - background_current

    corrected_peaks = extract_peaks(
        pot, corrected_current, v_cmd,
        v_start, v_vertex,
        estimate_noise_metrics(corrected_current)["noise_sigma"],
    )

    spectrum_path = None
    spectrum_diagnostics = None

    if "fft" in filter_methods:
        spectrum_path = raw_csv_path.with_name(
            f"{raw_csv_path.stem}_spectrum.png"
        )
        try:
            spectrum_diagnostics = save_spectrum_diagnostic(
                forward, reverse, step_v, spectrum_path,
                manual_cutoff=fft_cutoff,
            )
        except (ValueError, RuntimeError) as exc:
            spectrum_diagnostics = {"skipped": True, "reason": str(exc)}
            spectrum_path = None

    # Current density is calculated only when electrode area is available.
    if area_cm2 is not None:
        before_plot = curr / (area_cm2 * 1000.0)
        after_plot = corrected_current / (area_cm2 * 1000.0)
        y_label = "Current density [mA cm$^{-2}$]"
    else:
        before_plot = curr
        after_plot = corrected_current
        y_label = "Current [µA]"

    before_path = raw_csv_path.with_name(
        f"{raw_csv_path.stem}_before_correction.png"
    )
    after_path = raw_csv_path.with_name(
        f"{raw_csv_path.stem}_after_correction.png"
    )

    if save_plot:
        save_comparison_plot(
            pot, before_plot, before_plot,
            v_start, v_vertex, v_end, step_v,
            before_path,
            "CV before correction (raw data)",
            y_label,
        )
        save_comparison_plot(
            pot, before_plot, after_plot,
            v_start, v_vertex, v_end, step_v,
            after_path,
            (
                f"CV after DCT filtering and capacitive correction "
                f"(Cdl = {cdl_uf_cm2:g} µF/cm²)"
                if background_subtracted
                else "CV after random-noise filtering; background not subtracted"
            ),
            y_label,
        )

    # Save all processing stages so the correction is auditable.
    processed_path = raw_csv_path.with_name(
        f"{raw_csv_path.stem}_processed.csv"
    )
    pd.DataFrame({
        "potential_v": pot,
        "commanded_potential_v": v_cmd,
        "raw_current_ua": curr,
        "noise_reduced_current_ua": noise_reduced,
        "estimated_background_current_ua": background_current,
        "corrected_current_ua": corrected_current,
    }).to_csv(processed_path, index=False)

    calibration_warning = None
    if not calibrated:
        calibration_warning = (
            "Acquisition is marked uncalibrated. "
            "Absolute potential and current values are provisional."
        )
        warnings.warn(calibration_warning, RuntimeWarning)

    raw_peaks = extract_peaks(
        pot, curr, v_cmd, v_start, v_vertex, noise_std
    )

    return {
        "file": raw_csv_path.name,
        "metadata_file": meta_json_path.name,
        "before_plot_file": before_path.name if save_plot else None,
        "after_plot_file": after_path.name if save_plot else None,
        "plot_file": after_path.name if save_plot else None,
        "spectrum_file": spectrum_path.name if spectrum_path else None,
        "spectrum_diagnostics": spectrum_diagnostics,
        "processed_csv_file": processed_path.name,
        "selected_noise_filter": selected_filter,
        "background_subtracted": background_subtracted,
        "background_subtraction_reason": background_reason,
        "assumed_cdl_uf_cm2": float(cdl_uf_cm2),
        "scan_rate_v_per_s": (
            float(scan_rate) if scan_rate is not None else None
        ),
        "estimated_background_current_ua": (
            float(np.max(np.abs(background_current)))
            if background_subtracted else None
        ),
        "cycles_analyzed": 1,
        "simulated": bool(metadata.get("simulated", False)),
        "calibrated": calibrated,
        "calibration_warning": calibration_warning,
        "electrode_area_cm2": area_cm2,
        "raw_noise": raw_noise,
        "raw_peaks": raw_peaks,
        "corrected_peaks": corrected_peaks,
        "filters": filter_results,
    }









