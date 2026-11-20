from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

BASE_DIR = Path(__file__).resolve().parent

_data_dir_env = os.environ.get("DATA_DIR")
DATA_DIR = Path(_data_dir_env).expanduser() if _data_dir_env else BASE_DIR / "data"


@dataclass(frozen=True)
class HardwareConfig:
    # Potential sweep source on ExpEYES-17 / SEELab3 (PV1 is typically -5V to +5V)
    v_start: float = -0.20          # Volts (vs reference)
    v_vertex: float = 0.60          # Volts (forward turn-around potential)
    v_end: float = -0.20            # Volts (return potential)
    scan_rate_v_per_s: float = 0.05 # 50 mV/s (proposal range: 0.02 to 0.20 V/s)
    step_v: float = 0.005           # 5 mV per step (adjust to balance resolution vs speed)
    n_cycles: int = 1

    # Channel mapping (VERIFY during lab audit!)
    # Example: A1 reads working/cell potential or TIA output; A2 reads current-sense node
    potential_channel: str = "A1"
    current_channel: str = "A2"

    # Current conversion:
    # - If measuring voltage across a shunt resistor R_shunt: I = V / R_shunt
    # - If using a Transimpedance Amplifier (TIA) with feedback gain R_f (V/A): I = V_out / R_f
    # Set current_sign = -1.0 if TIA/shunt inverts polarity
    current_mode: str = "shunt"     # "shunt" or "tia"
    shunt_ohm: float = 1000.0       # Ohms (placeholder until audited)
    tia_gain_v_per_a: float = 1e5   # V/A (e.g., 100 kOhm -> 1e5 V/A; placeholder)
    current_sign: float = 1.0

    # Filter validation targets from Phase-I proposal
    max_peak_shift_v: float = 0.010 # < 10 mV
    max_peak_current_rel_change: float = 0.05  # < 5%

    # Acquisition timing
    equilibration_s: float = 2.0
    avg_window_s: float = 0.10  # 5 periods at 50 Hz
    max_rate_error_pct: float = 5.0

    # Input voltage ranges (configure after verifying hardware support)
    range_potential_v: Optional[float] = None
    range_current_v: Optional[float] = None

    # Calibration and electrode information
    calibrated: bool = False
    electrode_area_cm2: Optional[float] = None
    target_rsd_pct: float = 5.0     # <= 5% repeatability across >=3 repeats
    def __post_init__(self) -> None:

        # Validate potential limits
        potentials = (self.v_start, self.v_vertex, self.v_end)

        if any(
            not math.isfinite(v) or not -5.0 <= v <= 5.0
            for v in potentials
        ):
            raise ValueError("All potentials must be between -5 V and +5 V")

        # Validate step size
        if not math.isfinite(self.step_v) or self.step_v <= 0:
            raise ValueError("step_v must be greater than zero")

        # Validate scan rate
        if (
            not math.isfinite(self.scan_rate_v_per_s)
            or not 0.02 <= self.scan_rate_v_per_s <= 0.20
        ):
            raise ValueError("Scan rate must be between 0.02 and 0.20 V/s")

        # Validate current measurement mode
        if self.current_mode not in {"shunt", "tia"}:
            raise ValueError("current_mode must be 'shunt' or 'tia'")

        if self.current_mode == "shunt":
            if not math.isfinite(self.shunt_ohm) or self.shunt_ohm <= 0:
                raise ValueError("shunt_ohm must be positive")

        if self.current_mode == "tia":
            if (
                not math.isfinite(self.tia_gain_v_per_a)
                or self.tia_gain_v_per_a <= 0
            ):
                raise ValueError("tia_gain_v_per_a must be positive")

        # Validate current polarity
        if self.current_sign not in (-1.0, 1.0):
            raise ValueError("current_sign must be either +1 or -1")

        # Validate channel mapping
        p_channel = self.potential_channel.strip().upper()
        i_channel = self.current_channel.strip().upper()

        if not p_channel or not i_channel:
            raise ValueError("Channel names cannot be empty")

        if p_channel == i_channel:
            raise ValueError("Potential and current channels must be distinct")

        # Validate equilibration time
        if not math.isfinite(self.equilibration_s) or self.equilibration_s < 0:
            raise ValueError("equilibration_s cannot be negative")

        # Validate averaging window (whole 20 ms periods at 50 Hz)
        if not math.isfinite(self.avg_window_s) or self.avg_window_s <= 0:
            raise ValueError("avg_window_s must be positive")

        periods = round(self.avg_window_s / 0.020)

        if periods < 1 or not math.isclose(
            self.avg_window_s,
            periods * 0.020,
            rel_tol=0.0,
            abs_tol=1e-9
        ):
            raise ValueError(
                "avg_window_s must be a whole multiple of 20 ms"
            )

        # Validate scan-rate tolerance
        if (
            not math.isfinite(self.max_rate_error_pct)
            or self.max_rate_error_pct <= 0
        ):
            raise ValueError("max_rate_error_pct must be positive")

        # Validate optional input ranges
        for name, value in (
            ("range_potential_v", self.range_potential_v),
            ("range_current_v", self.range_current_v),
        ):
            if value is not None and (
                not math.isfinite(value) or value <= 0
            ):
                raise ValueError(f"{name} must be positive when specified")

        # Validate optional electrode area
        if self.electrode_area_cm2 is not None:
            if (
                not math.isfinite(self.electrode_area_cm2)
                or self.electrode_area_cm2 <= 0
            ):
                raise ValueError("electrode_area_cm2 must be positive")

CFG = HardwareConfig()



