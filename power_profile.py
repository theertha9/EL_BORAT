import time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
from config import DATA_DIR
from typing import Tuple

POWER_STATES = ("idle", "sensing", "scan", "transfer")


def sample_power_monitor(monitor_reader=None) -> Tuple[float, float]:
    """
    Return (bus_voltage_v, current_ma).
    Connect `monitor_reader` (e.g., INA219/ESP32 serial or I2C helper) once wired.
    """
    if monitor_reader is None:
        raise RuntimeError(
            "No power monitor attached. Cannot collect real power measurements."
        )

    return monitor_reader()


def profile_state(
        
    state_name: str,
    duration_s: float = 3.0,
    sample_interval_s: float = 0.05,
    monitor_reader=None,
) -> pd.DataFrame:
    if state_name not in POWER_STATES:
        raise ValueError(f"State must be one of {POWER_STATES}, got '{state_name}'.")
    if not np.isfinite(duration_s) or duration_s <= 0:
        raise ValueError("duration_s must be a finite positive number.")
    
    if not np.isfinite(sample_interval_s) or sample_interval_s <= 0:
        raise ValueError("sample_interval_s must be a finite positive number.")

    records = []
    t_start = time.perf_counter()
    while (time.perf_counter() - t_start) < duration_s:
        v_bus, i_ma = sample_power_monitor(monitor_reader)
        p_mw = v_bus * i_ma
        records.append(
            {
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "state": state_name,
                "t_rel_s": time.perf_counter() - t_start,
                "voltage_v": v_bus,
                "current_ma": i_ma,
                "power_mw": p_mw,
            }
        )
        time.sleep(sample_interval_s)
    return pd.DataFrame(records)


def run_four_state_power_profile(duration_per_state_s: float = 2.0, monitor_reader=None) -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    dfs = [
        profile_state(st, duration_s=duration_per_state_s, monitor_reader=monitor_reader)
        for st in POWER_STATES
    ]
    full_df = pd.concat(dfs, ignore_index=True)

    summary = (
        full_df.groupby("state", sort=False)[["voltage_v", "current_ma", "power_mw"]]
        .agg(["mean", "std", "max"])
    )
    out_path = DATA_DIR / f"power_profile_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.csv"
    full_df.to_csv(out_path, index=False)
    summary_path = out_path.with_name(out_path.stem + "_summary.csv")
    summary.to_csv(summary_path)
    return out_path

