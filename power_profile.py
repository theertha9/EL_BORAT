
from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from config import DATA_DIR


POWER_STATES = ("idle", "sensing", "scan", "transfer")

Workload = Callable[[], None]
MonitorReader = Callable[[], tuple[float, float]]


def _get_monitor_reader() -> MonitorReader:
    """Load the real power-monitor reader."""
    try:
        from power_monitor import read_monitor
    except ImportError as exc:
        raise RuntimeError(
            "power_monitor.py with read_monitor() is required for "
            "power profiling."
        ) from exc

    return read_monitor


def sample_power_monitor(
    monitor_reader: MonitorReader,
) -> tuple[float, float]:
    """Return validated (bus_voltage_v, current_ma) readings."""
    if not callable(monitor_reader):
        raise RuntimeError("A valid power monitor reader is required.")

    reading = monitor_reader()

    if not isinstance(reading, (tuple, list)) or len(reading) != 2:
        raise RuntimeError(
            "read_monitor() must return (voltage_v, current_ma)."
        )

    voltage_v = float(reading[0])
    current_ma = float(reading[1])

    if not np.all(np.isfinite([voltage_v, current_ma])):
        raise RuntimeError("Power monitor returned a non-finite reading.")

    return voltage_v, current_ma


def profile_state(
    state_name: str,
    workload: Workload,
    duration_s: float,
    sample_interval_s: float,
    settle_s: float,
    monitor_reader: MonitorReader,
) -> tuple[pd.DataFrame, dict]:
    """Profile one state while repeatedly executing its actual workload."""
    if state_name not in POWER_STATES:
        raise ValueError(
            f"State must be one of {POWER_STATES}, got '{state_name}'."
        )

    for name, value in (
        ("duration_s", duration_s),
        ("sample_interval_s", sample_interval_s),
    ):
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive.")

    if not np.isfinite(settle_s) or settle_s < 0:
        raise ValueError("settle_s must be finite and non-negative.")

    if settle_s >= duration_s:
        raise ValueError("settle_s must be shorter than duration_s.")

    if not callable(workload):
        raise ValueError(f"A callable workload is required for '{state_name}'.")

    records = []
    t_start = time.perf_counter()

    while True:
        elapsed_before = time.perf_counter() - t_start

        if elapsed_before >= duration_s:
            break

        # Execute actual work for this state before taking its reading.
        workload()

        voltage_v, current_ma = sample_power_monitor(monitor_reader)
        sample_time = time.perf_counter()
        elapsed = sample_time - t_start

        records.append(
            {
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "state": state_name,
                "t_rel_s": elapsed,
                "voltage_v": voltage_v,
                "current_ma": current_ma,
                "power_mw": voltage_v * current_ma,
            }
        )

        remaining = sample_interval_s - (
            time.perf_counter() - sample_time
        )

        if remaining > 0:
            time.sleep(remaining)

    full_df = pd.DataFrame(records)

    # Exclude the settling period from all reported measurements.
    measured_df = full_df.loc[
        full_df["t_rel_s"] >= settle_s
    ].copy()

    if len(measured_df) < 2:
        raise RuntimeError(
            f"Not enough measurements remain for '{state_name}' "
            f"after discarding the {settle_s:.3f} s settling period."
        )

    times = measured_df["t_rel_s"].to_numpy(dtype=float)
    powers = measured_df["power_mw"].to_numpy(dtype=float)

    # mW multiplied by seconds gives millijoules.
    energy_mj = float(
        np.sum(
            (powers[:-1] + powers[1:])
            * 0.5
            * np.diff(times)
        )
    )

    measured_duration_s = float(times[-1] - times[0])

    if measured_duration_s <= 0:
        raise RuntimeError(
            f"Invalid measurement duration for '{state_name}'."
        )

    summary = {
        "state": state_name,
        "measurement_count": int(len(measured_df)),
        "settling_discarded_s": float(settle_s),
        "measurement_duration_s": measured_duration_s,
        "voltage_mean_v": float(measured_df["voltage_v"].mean()),
        "voltage_std_v": float(measured_df["voltage_v"].std()),
        "current_mean_ma": float(measured_df["current_ma"].mean()),
        "current_std_ma": float(measured_df["current_ma"].std()),
        "power_mean_mw": float(measured_df["power_mw"].mean()),
        "power_std_mw": float(measured_df["power_mw"].std()),
        "power_max_mw": float(measured_df["power_mw"].max()),
        "energy_mj": energy_mj,
    }

    return measured_df, summary


def run_four_state_power_profile(
    workloads: dict[str, Workload],
    duration_per_state_s: float = 3.0,
    sample_interval_s: float = 0.05,
    settle_s: float = 0.5,
    monitor_reader: MonitorReader | None = None,
) -> Path:
    """
    Profile idle, sensing, scan, and transfer workloads.

    Each workload is a callable executed repeatedly during its state.
    A real monitor reader is mandatory.
    """
    missing_states = [
        state for state in POWER_STATES
        if state not in workloads
    ]

    if missing_states:
        raise ValueError(
            f"Missing workload callables for: {', '.join(missing_states)}"
        )

    invalid_states = [
        state for state in workloads
        if state not in POWER_STATES
    ]

    if invalid_states:
        raise ValueError(
            f"Unsupported workload states: {', '.join(invalid_states)}"
        )

    if monitor_reader is None:
        monitor_reader = _get_monitor_reader()

    if not callable(monitor_reader):
        raise ValueError("monitor_reader must be callable.")

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    state_frames = []
    state_summaries = []

    for state in POWER_STATES:
        print(f"Profiling power state: {state}")

        measurements, summary = profile_state(
            state_name=state,
            workload=workloads[state],
            duration_s=duration_per_state_s,
            sample_interval_s=sample_interval_s,
            settle_s=settle_s,
            monitor_reader=monitor_reader,
        )

        state_frames.append(measurements)
        state_summaries.append(summary)

    full_df = pd.concat(state_frames, ignore_index=True)
    summary_df = pd.DataFrame(state_summaries)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    out_path = DATA_DIR / f"power_profile_{timestamp}.csv"
    summary_path = out_path.with_name(
        out_path.stem + "_summary.csv"
    )
    meta_path = out_path.with_suffix(".meta.json")

    # Exclusive creation prevents accidental overwriting.
    with out_path.open("x", encoding="utf-8", newline="") as f:
        full_df.to_csv(f, index=False)

    with summary_path.open("x", encoding="utf-8", newline="") as f:
        summary_df.to_csv(f, index=False)

    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "measurement_source": "physical_power_monitor",
        "monitor_reader": "power_monitor.read_monitor",
        "states": list(POWER_STATES),
        "duration_per_state_s": duration_per_state_s,
        "sample_interval_s": sample_interval_s,
        "settling_discarded_s": settle_s,
        "energy_unit": "mJ",
        "measurement_csv": out_path.name,
        "summary_csv": summary_path.name,
        "state_summaries": state_summaries,
    }

    with meta_path.open("x", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, allow_nan=False)

    return out_path


