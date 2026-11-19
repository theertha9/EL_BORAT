from dataclasses import dataclass
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"


@dataclass
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
    target_rsd_pct: float = 5.0     # <= 5% repeatability across >=3 repeats


CFG = HardwareConfig()
