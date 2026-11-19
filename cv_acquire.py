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
    return np.tile(single_cycle, n_cycles)


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
    time.sleep(0.05)

    for idx, v_cmd in enumerate(potentials):
        step_start = time.perf_counter()
        p.set_pv1(float(v_cmd))
        if dt_step > 0:
            time.sleep(dt_step)

        t_rel = time.perf_counter() - t0
        v_meas = float(p.get_voltage(cfg.potential_channel))
        v_curr_node = float(p.get_voltage(cfg.current_channel))
        i_a = voltage_to_current_a(v_curr_node, cfg)
        temp_c = read_temperature_c(p, fallback_c=temp_c_override)

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
    p.set_pv1(0.0)

    df = pd.DataFrame(rows)
    df.to_csv(raw_csv_path, index=False)

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
    meta_json_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return raw_csv_path, meta_json_path
