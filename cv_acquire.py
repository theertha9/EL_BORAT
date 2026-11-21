from __future__ import annotations
import json
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
from config import CFG, DATA_DIR


def build_sweep_potentials(
    v_start: float,
    v_vertex: float,
    v_end: float,
    step_v: float,
    n_cycles: int = 1,
) -> np.ndarray:
    """Build forward (v_start -> v_vertex) and reverse (v_vertex -> v_end) potential steps."""
    step = abs(step_v)
    if n_cycles < 1:
        raise ValueError("n_cycles must be at least 1.")
    
    if not np.all(np.isfinite([v_start, v_vertex, v_end, step_v])):
        raise ValueError("Potential values and step_v must be finite.")
    if step <= 0:
        raise ValueError("step_v must be positive.")

    n_fwd = max(1, int(round(abs(v_vertex - v_start) / step)))
    fwd = np.linspace(v_start, v_vertex, n_fwd + 1)

    n_rev = max(1, int(round(abs(v_end - v_vertex) / step)))
    rev = np.linspace(v_vertex, v_end, n_rev + 1)[1:]  # avoid duplicating vertex point

    single_cycle = np.concatenate([fwd, rev])

    if n_cycles <= 1:
        return single_cycle
    
    cycles = [single_cycle]
    
    for _ in range(1, n_cycles):
        if not np.isclose(v_end, v_start):
            link = np.linspace(
                v_end,
                v_start,
                max(2, int(round(abs(v_start - v_end) / step)) + 1)
            )[1:]
            cycles.append(link)
    
        cycles.append(single_cycle[1:])

    return np.concatenate(cycles)


def voltage_to_current_a(measured_v: float, cfg=CFG) -> float:
    """Convert shunt or TIA output voltage (V) to cell current (A)."""
    if cfg.current_mode == "shunt":
        if cfg.shunt_ohm == 0:
            raise ValueError("shunt_ohm cannot be zero.")
        raw_i = measured_v / cfg.shunt_ohm
    elif cfg.current_mode == "tia":
        if cfg.tia_gain_v_per_a == 0:
            raise ValueError("tia_gain_v_per_a cannot be zero.")
        raw_i = measured_v / cfg.tia_gain_v_per_a
    else:
        raise ValueError(f"Unsupported current_mode: {cfg.current_mode}")
    return float(raw_i * cfg.current_sign)

    
def read_temperature_c(p=None, fallback_c: float | None = None) -> float | None:
    """
    Temperature sensor placeholder.

    Until a real sensor is connected, return the manually supplied
    fallback temperature, or None if no value was supplied.
    """
    return fallback_c


def _write_cv_report(
    report_path: Path,
    metadata: dict,
    rows: list[dict],
) -> None:
    """Write experiment settings and per-step readings to a text report."""
    with report_path.open("x", encoding="utf-8") as report:
        report.write("CYCLIC VOLTAMMETRY EXPERIMENT REPORT\n")
        report.write("=" * 38 + "\n\n")
        report.write(f"Created (UTC): {metadata['created_utc']}\n")
        report.write(f"Run label: {metadata['run_label']}\n")
        report.write(f"Repeat index: {metadata['repeat_idx']}\n")
        report.write(f"Raw CSV: {metadata['raw_csv']}\n")
        report.write(f"Report status: {'complete' if metadata['acquisition']['complete'] else 'incomplete'}\n\n")

        report.write("EXPERIMENT AND HARDWARE SPECIFICATIONS\n")
        report.write("-------------------------------------\n")
        for name, value in metadata["program_config"].items():
            report.write(f"CFG.{name}: {value}\n")
        report.write("\n")

        for section_name in (
            "scan_params",
            "acquisition",
            "front_end",
            "auxiliary_context",
        ):
            report.write(f"{section_name.upper().replace('_', ' ')}\n")
            report.write("-" * len(section_name) + "\n")
            for name, value in metadata[section_name].items():
                report.write(f"{name}: {value}\n")
            report.write("\n")

        setup_audit = metadata.get("setup_audit")
        if setup_audit is not None:
            report.write("PRE-SCAN SETUP AUDIT READINGS\n")
            report.write("-----------------------------\n")
            for name, value in setup_audit.items():
                report.write(f"{name}: {value}\n")
            report.write("\n")

        report.write("READINGS BEFORE AND AFTER EACH POTENTIAL COMMAND\n")
        report.write("-----------------------------------------------\n")
        report.write(
            "The first pre-command values are measured before the initial PV1 "
            "setpoint. Each later pre-command value is the preceding step's "
            "averaged reading (no additional samples or sweep delays are added).\n"
            "Post-command values are averages over the configured averaging "
            "window. Individual readings from each window follow each step.\n\n"
        )

        if not rows:
            report.write("No potential-step readings were collected.\n")

        for row in rows:
            report.write(
                f"Step {row['sample_idx']} | t={row['t_rel_s']:.6f} s | "
                f"commanded PV1={row['v_cmd_v']:.9g} V\n"
            )
            report.write(
                f"  Before: potential={row['v_pre_v']:.9g} V, "
                f"current-sense={row['v_curr_sense_pre_v']:.9g} V\n"
            )
            report.write(
                f"  After (averaged): potential={row['v_meas_v']:.9g} V, "
                f"current-sense={row['v_curr_sense_v']:.9g} V, "
                f"current={row['current_a']:.9g} A "
                f"({row['current_ua']:.9g} uA), "
                f"temperature={row['temp_c']}\n"
            )
            report.write(
                f"  Step elapsed: {row['step_elapsed_s']:.6f} s\n"
            )
            report.write("  Individual readings during averaging window:\n")

            samples = json.loads(row["sample_readings_json"])
            if not samples:
                report.write("    No individual readings recorded.\n")
            else:
                report.write(
                    "    time_rel_s | potential_v | current_sense_v | "
                    "current_a | current_ua\n"
                )
                for sample in samples:
                    report.write(
                        f"    {sample['t_rel_s']:.6f} | "
                        f"{sample['potential_v']:.9g} | "
                        f"{sample['current_sense_v']:.9g} | "
                        f"{sample['current_a']:.9g} | "
                        f"{sample['current_ua']:.9g}\n"
                    )
            report.write("\n")

        report.write("READINGS AFTER THE SCAN / PV1 RESET\n")
        report.write("----------------------------------\n")
        for name, value in metadata["acquisition"]["post_reset_readings"].items():
            report.write(f"{name}: {value}\n")

        for key in ("acquisition_error", "reset_error"):
            error = metadata["acquisition"].get(key)
            if error is not None:
                report.write(f"{key}: {error}\n")



def acquire_cv_scan(
    p,
    run_label: str = "scan",
    repeat_idx: int = 1,
    ph_measured_between_scans: float | None = None,
    temp_c_override: float | None = None,
    setup_audit: dict | None = None,
    cfg=CFG,
) -> tuple[Path, Path]:
    """
    Execute a cyclic voltammetry sweep using PV1 and log raw data to CSV + JSON metadata.
    Never overwrites or mutates raw data after writing.
    """
    
    if not cfg.calibrated:
        print(
            "WARNING: FRONT END IS NOT CALIBRATED. "
            "Current measurements are not validated."
        )

   
    
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Use microseconds to reduce filename collisions.
    ts_slug = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    base_name = f"cv_raw_{run_label}_rep{repeat_idx}_{ts_slug}"

    raw_csv_path = DATA_DIR / f"{base_name}.csv"
    meta_json_path = DATA_DIR / f"{base_name}.meta.json"

    # Build the complete commanded potential sequence.
    potentials = build_sweep_potentials(
        cfg.v_start,
        cfg.v_vertex,
        cfg.v_end,
        cfg.step_v,
        cfg.n_cycles,
    )

    # Validate acquisition timing.
    if cfg.scan_rate_v_per_s <= 0:
        raise ValueError("scan_rate_v_per_s must be positive.")

    if cfg.avg_window_s <= 0:
        raise ValueError("avg_window_s must be positive.")

    if cfg.equilibration_s < 0:
        raise ValueError("equilibration_s cannot be negative.")

    # Average over complete 20 ms mains periods.
    mains_period_s = 0.020
    n_periods = round(cfg.avg_window_s / mains_period_s)

    if n_periods < 1 or not np.isclose(
        cfg.avg_window_s,
        n_periods * mains_period_s,
        atol=1e-6,
    ):
        raise ValueError(
            "avg_window_s must be a positive whole multiple of 0.020 seconds."
        )

    # Calculate the actual time interval between commanded points.
    # This accounts for linspace rounding and cycle-link ramps.
    step_distances = np.abs(np.diff(potentials))
    step_intervals = step_distances / cfg.scan_rate_v_per_s

    positive_intervals = step_intervals[step_intervals > 0]

    if len(positive_intervals) and (
        cfg.avg_window_s > float(np.min(positive_intervals))
    ):
        raise ValueError(
            "Averaging window is longer than the shortest potential-step interval."
        )

    # Cumulative deadlines for each potential point.
    step_offsets = np.concatenate(
        ([0.0], np.cumsum(step_intervals))
    )

    rows = []
    command_times = []

    acquisition_error = None
    reset_error = None
    complete = False
    post_reset_readings = {}

    start_utc = datetime.now(timezone.utc).isoformat()
    t0 = time.perf_counter()

    # Temperature source is explicitly recorded.
    temperature_source = (
        "manual" if temp_c_override is not None else "stub"
    )

    try:
        # Record the baseline immediately before applying the first sweep
        # potential; later pre-command values reuse the preceding step average.
        previous_potential = float(p.get_voltage(cfg.potential_channel))
        previous_current_sense = float(p.get_voltage(cfg.current_channel))

        # Move to initial potential and allow the electrode to equilibrate.
        p.set_pv1(float(potentials[0]))
        time.sleep(cfg.equilibration_s)

        sweep_t0 = time.perf_counter()

        for idx, v_cmd in enumerate(potentials):

            # Use absolute deadlines rather than repeatedly sleeping
            # for a fixed interval. This limits accumulated timing drift.
            deadline = sweep_t0 + step_offsets[idx]

            remaining = deadline - time.perf_counter()

            if remaining > 0:
                time.sleep(remaining)

            # The first potential was already set before equilibration.
            if idx > 0:
                p.set_pv1(float(v_cmd))

            pre_potential = previous_potential
            pre_current_sense = previous_current_sense
            command_time = time.perf_counter()
            command_times.append(command_time)

            step_start = command_time
            window_start = time.perf_counter()

            potential_samples = []
            current_samples = []
            sample_readings = []

            # Collect readings throughout the averaging window.
            while (
                time.perf_counter() - window_start
                < cfg.avg_window_s
            ):
                potential_sample = float(
                    p.get_voltage(cfg.potential_channel)
                )
                current_sample = float(
                    p.get_voltage(cfg.current_channel)
                )
                sample_time = time.perf_counter()
                sample_current_a = voltage_to_current_a(
                    current_sample,
                    cfg,
                )
                potential_samples.append(potential_sample)
                current_samples.append(current_sample)
                sample_readings.append(
                    {
                        "t_rel_s": sample_time - t0,
                        "potential_v": potential_sample,
                        "current_sense_v": current_sample,
                        "current_a": sample_current_a,
                        "current_ua": sample_current_a * 1e6,
                    }
                )

            window_end = time.perf_counter()

            if not potential_samples or not current_samples:
                raise RuntimeError(
                    f"No readings collected at sample {idx}."
                )

            # Average readings without modifying the raw instrument values.
            v_meas = float(np.mean(potential_samples))
            v_curr_node = float(np.mean(current_samples))

            i_a = voltage_to_current_a(v_curr_node, cfg)

            temp_c = read_temperature_c(
                p,
                fallback_c=temp_c_override,
            )

            # Use the midpoint of the averaging window as its timestamp.
            t_rel = (
                (window_start + window_end) / 2.0
            ) - t0

            rows.append(
                {
                    "sample_idx": idx,
                    "t_rel_s": t_rel,
                    "v_cmd_v": float(v_cmd),
                    "v_pre_v": pre_potential,
                    "v_curr_sense_pre_v": pre_current_sense,
                    "v_meas_v": v_meas,
                    "v_curr_sense_v": v_curr_node,
                    "current_a": i_a,
                    "current_ua": i_a * 1e6,
                    "temp_c": temp_c,
                    "step_elapsed_s": window_end - step_start,
                    "sample_readings_json": json.dumps(sample_readings),
                }
            )
            previous_potential = v_meas
            previous_current_sense = v_curr_node

        complete = True

    except Exception as exc:
        acquisition_error = repr(exc)

    finally:
        # Always attempt to return PV1 to zero, including after failures.
        try:
            p.set_pv1(0.0)
        except Exception as exc:
            reset_error = repr(exc)
        else:
            for name, channel in (
                ("potential_v", cfg.potential_channel),
                ("current_sense_v", cfg.current_channel),
            ):
                try:
                    post_reset_readings[name] = float(
                        p.get_voltage(channel)
                    )
                except Exception as exc:
                    post_reset_readings[name] = f"Read error: {exc!r}"

    # Calculate achieved scan rate from actual commanded path distance
    # and actual time between the first and last commands.
    achieved_rate = None
    rate_error_pct = None

    if len(command_times) >= 2:
        actual_duration = command_times[-1] - command_times[0]
        total_path_distance = float(np.sum(step_distances))

        if actual_duration > 0:
            achieved_rate = total_path_distance / actual_duration

            rate_error_pct = (
                abs(achieved_rate - cfg.scan_rate_v_per_s)
                / cfg.scan_rate_v_per_s
                * 100.0
            )

    rate_within_tolerance = (
        None
        if rate_error_pct is None
        else rate_error_pct <= cfg.max_rate_error_pct
    )

    if rate_error_pct is not None and not rate_within_tolerance:
        print(
            f"WARNING: Scan-rate error is {rate_error_pct:.2f}%, "
            f"exceeding the allowed {cfg.max_rate_error_pct:.2f}%."
        )

    # Use fixed columns so even an interrupted scan produces a readable CSV.
    csv_columns = [
        "sample_idx",
        "t_rel_s",
        "v_cmd_v",
        "v_meas_v",
        "v_curr_sense_v",
        "current_a",
        "current_ua",
        "temp_c",
        "step_elapsed_s",
        "v_pre_v",
        "v_curr_sense_pre_v",
        "sample_readings_json",
    ]

    df = pd.DataFrame(rows, columns=csv_columns)

    # Exclusive creation: do not overwrite an existing raw data file.
    with raw_csv_path.open("x", encoding="utf-8", newline="") as f:
        df.to_csv(f, index=False)

    metadata = {
        "created_utc": start_utc,
        "run_label": run_label,
        "repeat_idx": repeat_idx,
        "raw_csv": raw_csv_path.name,

        "scan_params": {
            "v_start": cfg.v_start,
            "v_vertex": cfg.v_vertex,
            "v_end": cfg.v_end,
            "scan_rate_v_per_s": cfg.scan_rate_v_per_s,
            "step_v": cfg.step_v,
            "n_cycles": cfg.n_cycles,
            "calibrated": cfg.calibrated,
        },

        "acquisition": {
            "equilibration_s": cfg.equilibration_s,
            "avg_window_s": cfg.avg_window_s,
            "requested_scan_rate_v_per_s": cfg.scan_rate_v_per_s,
            "achieved_scan_rate_v_per_s": achieved_rate,
            "rate_error_pct": rate_error_pct,
            "max_rate_error_pct": cfg.max_rate_error_pct,
            "rate_within_tolerance": rate_within_tolerance,
            "n_samples_saved": len(rows),
            "complete": complete,
            "acquisition_error": acquisition_error,
            "reset_error": reset_error,
            "post_reset_readings": post_reset_readings,
        },

        "front_end": {
            "potential_channel": cfg.potential_channel,
            "current_channel": cfg.current_channel,
            "current_mode": cfg.current_mode,
            "shunt_ohm": cfg.shunt_ohm,
            "tia_gain_v_per_a": cfg.tia_gain_v_per_a,
            "current_sign": cfg.current_sign,
            "potential_range_v": cfg.range_potential_v,
            "current_range_v": cfg.range_current_v,
            "calibrated": cfg.calibrated,
            "temperature": {
                "value_c": temp_c_override,
                "source": temperature_source,
            },
        },

        "auxiliary_context": {
            "ph_measured_between_scans": ph_measured_between_scans,
            "notes": (
                "pH logged between scans to avoid electrode coupling "
                "per Phase-I methodology."
            ),
        },
        "setup_audit": setup_audit,
        "program_config": asdict(cfg),
    }

    report_path = raw_csv_path.with_suffix(".report.txt")
    metadata["report_file"] = report_path.name

    with meta_json_path.open("x", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, allow_nan=False)

    _write_cv_report(report_path, metadata, rows)

    # The CSV and metadata are saved before an acquisition/reset error
    # is raised, preserving partial data for diagnosis.
    if acquisition_error is not None:
        raise RuntimeError(
            f"CV acquisition failed. Partial data saved to "
            f"{raw_csv_path}. Error: {acquisition_error}"
        )

    if reset_error is not None:
        raise RuntimeError(
            f"CV data was acquired, but PV1 reset failed: {reset_error}. "
            f"Data saved to {raw_csv_path}."
        )

    return raw_csv_path, meta_json_path

    

    


    

    
