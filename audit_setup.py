import platform
from datetime import datetime, timezone
from eyes17 import eyes
from config import CFG, DATA_DIR


def run_audit() -> dict:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": platform.python_version(),
        "os": platform.platform(),
        "connected": False,
        "device_repr": None,
        "baseline_voltages": {},
        "notes": [],
    }

    p = eyes.open()
    if p is None or not getattr(p, "connected", False):
        report["notes"].append(
            "ExpEYES/SEELab3 not detected. Check USB cable, udev/driver permissions, or device power."
        )
        return report

    report["connected"] = True
    report["device_repr"] = repr(p)

    # Set PV1 to 0 V and sample baseline channels
    p.set_pv1(0.0)
    for ch in ("A1", "A2", "A3"):
        try:
            report["baseline_voltages"][ch] = float(p.get_voltage(ch))
        except Exception as exc:
            report["baseline_voltages"][ch] = f"Error: {exc}"

    report["notes"].append(
        f"Configured channels: potential={CFG.potential_channel}, current={CFG.current_channel}, "
        f"mode={CFG.current_mode}. Verify actual input ranges (-16V/+16V vs -3.3V/+3.3V) and TIA/shunt values."
    )
    return report


if __name__ == "__main__":
    res = run_audit()
    for k, v in res.items():
        print(f"{k}: {v}")
