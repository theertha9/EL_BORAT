from __future__ import annotations

import math
import os
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Optional

BASE_DIR = Path(__file__).resolve().parent

_data_dir_env = os.environ.get("DATA_DIR")
DATA_DIR = Path(_data_dir_env).expanduser() if _data_dir_env else BASE_DIR / "data"
CONFIG_FILE = BASE_DIR / "hardware_config.txt"


@dataclass(frozen=True)
class HardwareConfig:
    # Potential sweep source on ExpEYES-17 / SEELab3 (PV1 is typically -5V to +5V)
    v_start: float = -0.20          # Volts (vs reference)
    v_vertex: float = 0.60          # Volts (forward turn-around potential)
    v_end: float = -0.20            # Volts (return potential)
    scan_rate_v_per_s: float = 0.05 # 50 mV/s (proposal range: 0.02 to 0.20 V/s)
    step_v: float = 0.005           # 5 mV per step (adjust to balance resolution vs speed)
    n_cycles: int = 1
    n_repeats: int = 3

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
    avg_window_s: float = 0.08  # 5 periods at 50 Hz
    max_rate_error_pct: float = 5.0

    # Input voltage ranges (configure after verifying hardware support)
    range_potential_v: Optional[float] = None
    range_current_v: Optional[float] = None

    # Calibration and electrode information
    calibrated: bool = False
    electrode_area_cm2: Optional[float] = None
    target_rsd_pct: float = 5.0     # <= 5% repeatability across >=3 repeats
    def __post_init__(self) -> None:

        # Validate cycle count
        if isinstance(self.n_cycles, bool) or not isinstance(self.n_cycles, int) or self.n_cycles <= 0:
            raise ValueError("n_cycles must be a positive integer")

        if isinstance(self.n_repeats, bool) or not isinstance(self.n_repeats, int) or self.n_repeats <= 0:
            raise ValueError("n_repeats must be a positive integer")

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
        
        object.__setattr__(self, "potential_channel", p_channel)
        object.__setattr__(self, "current_channel", i_channel)

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

def load_hardware_config(path: Path = CONFIG_FILE) -> HardwareConfig:
    """Load every HardwareConfig value from a name:value text file."""
    config_fields = {item.name: item.default for item in fields(HardwareConfig)}
    values = {}

    with path.open("r", encoding="utf-8") as config_file:
        for line_number, line in enumerate(config_file, start=1):
            content = line.partition("#")[0].strip()
            if not content:
                continue

            if ":" not in content:
                raise ValueError(
                    f"{path}:{line_number}: expected 'name:value'."
                )

            name, raw_value = (part.strip() for part in content.split(":", 1))
            if name not in config_fields:
                raise ValueError(
                    f"{path}:{line_number}: unknown configuration variable "
                    f"'{name}'."
                )
            if name in values:
                raise ValueError(
                    f"{path}:{line_number}: duplicate configuration variable "
                    f"'{name}'."
                )
            if not raw_value:
                raise ValueError(
                    f"{path}:{line_number}: value for '{name}' cannot be empty."
                )

            default = config_fields[name]
            try:
                if isinstance(default, bool):
                    normalized = raw_value.lower()
                    if normalized not in {"true", "false"}:
                        raise ValueError("expected true or false")
                    values[name] = normalized == "true"
                elif isinstance(default, int):
                    values[name] = int(raw_value)
                elif isinstance(default, float):
                    values[name] = float(raw_value)
                elif default is None:
                    values[name] = (
                        None if raw_value.lower() == "none"
                        else float(raw_value)
                    )
                else:
                    values[name] = raw_value
            except ValueError as exc:
                raise ValueError(
                    f"{path}:{line_number}: invalid value for '{name}': "
                    f"{raw_value!r}."
                ) from exc

    missing = sorted(set(config_fields) - set(values))
    if missing:
        raise ValueError(
            f"{path}: missing configuration variables: {', '.join(missing)}."
        )

    return HardwareConfig(**values)


CFG = load_hardware_config()



















