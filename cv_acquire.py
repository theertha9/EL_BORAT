import json
import time
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
    Hook for DS18B20 (1-Wire), thermistor on analog channel, or microcontroller bridge.
    Replace with concrete sensor read once audited on the lab hardware.
    """
    return fallback_c


def acquire_cv_scan(
    p,
    run_label: str = "scan",
    repeat_idx: int = 1,
    ph_measured_between_scans: float | None = None,
    temp_c_override: float | None = None,
    cfg=CFG,
    if not cfg.calibrated:
        print(
            "WARNING: FRONT END IS NOT CALIBRATED. "
            "Current measurements are not validated."
    )
) -> tuple[Path, Path]:
    """
    Execute a cyclic voltammetry sweep using PV1 and log raw data to CSV + JSON metadata.
    Never overwrites or mutates raw data after writing.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    ts_slug = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = f"cv_raw_{run_label}_rep{repeat_idx}_{ts_slug}"
    raw_csv_path = DATA_DIR / f"{base_name}.csv"
    meta_json_path = DATA_DIR / f"{base_name}.meta.json"

    potentials = build_sweep_potentials(
        cfg.v_start, cfg.v_vertex, cfg.v_end, cfg.step_v, cfg.n_cycles
    )
    dt_step = cfg.step_v / cfg.scan_rate_v_per_s

    rows = []
    t0 = time.perf_counter()
    start_utc = datetime.now(timezone.utc).isoformat()

    # Move smoothly to initial potential
    p.set_pv1(float(potentials[0]))
    time.sleep(cfg.equilibration_s)
    sweep_t0 = time.perf_counter()
    for idx, v_cmd in enumerate(potentials):
        deadline = sweep_t0 + idx * dt_step
    
        p.set_pv1(float(v_cmd))
    
        remaining = deadline - time.perf_counter()
        if remaining > 0:
            time.sleep(remaining)

        t_rel = time.perf_counter() - t0
        window_start = time.perf_counter()
        potential_samples = []
        current_samples = []
        
        while time.perf_counter() - window_start < cfg.avg_window_s:
            potential_samples.append(
                float(p.get_voltage(cfg.potential_channel))
            )
            current_samples.append(
                float(p.get_voltage(cfg.current_channel))
            )
        
        v_meas = float(np.mean(potential_samples))
        v_curr_node = float(np.mean(current_samples))
        i_a = voltage_to_current_a(v_curr_node, cfg)
        temp_c = read_temperature_c(p, fallback_c=temp_c_override)
        period_s = 0.020
        elapsed = rows[-1]["t_rel_s"] - rows[0]["t_rel_s"]

    if elapsed > 0:
        achieved_rate = (
            abs(rows[-1]["v_cmd_v"] - rows[0]["v_cmd_v"]) / elapsed
        )
    else:
        achieved_rate = np.nan
    
    rate_error_pct = (
        abs(achieved_rate - cfg.scan_rate_v_per_s)
        / cfg.scan_rate_v_per_s * 100
    )

        rows.append(
            {
                "sample_idx": idx,
                "t_rel_s": t_rel,
                "v_cmd_v": float(v_cmd),
                "v_meas_v": v_meas,
                "v_curr_sense_v": v_curr_node,
                "current_a": i_a,
                "current_ua": i_a * 1e6,
                "temp_c": temp_c,
                "step_elapsed_s": time.perf_counter() - step_start,
            }
        )
        

    # Return PV1 safely to 0 V after sweep
    rows = []
    error_message = None
    complete = False
    
    try:
        # Equilibration and acquisition loop here
        complete = True
    
    except Exception as exc:
        error_message = repr(exc)
        raise
    
    finally:
        try:
            p.set_pv1(0.0)
        except Exception as reset_exc:
            error_message = (
                f"{error_message}; PV1 reset failed: {reset_exc}"
                if error_message
                else f"PV1 reset failed: {reset_exc}"
            )

    # Save partial data and metadata here

    df = pd.DataFrame(rows)
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
        "front_end": {
            "potential_channel": cfg.potential_channel,
            "current_channel": cfg.current_channel,
            "current_mode": cfg.current_mode,
            "shunt_ohm": cfg.shunt_ohm,
            "tia_gain_v_per_a": cfg.tia_gain_v_per_a,
            "current_sign": cfg.current_sign,
        },
        "auxiliary_context": {
            "ph_measured_between_scans": ph_measured_between_scans,
            "notes": "pH logged between scans to avoid electrode coupling per Phase-I methodology.",
        },
    }
    with meta_json_path.open("x", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    return raw_csv_path, meta_json_path
