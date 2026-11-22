
import platform
from datetime import datetime, timezone

from eyes17 import eyes

from config import CFG, DATA_DIR


def run_audit() -> tuple[dict, object | None]:
    """Open the device once, audit its state, and return the same handle."""
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

    # This is the only eyes.open() call in the hardware workflow.
    try:
        p = eyes.open()
    except Exception as exc:
        report["notes"].append(f"Device connection failed: {exc}")
        return report, None

    if p is None or not getattr(p, "connected", False):
        report["notes"].append(
            "ExpEYES/SEELab3 not detected. Check USB cable, "
            "udev/driver permissions, or device power."
        )
        return report, None

    report["connected"] = True
    report["device_repr"] = repr(p)

    # Set PV1 to 0 V and sample baseline channels.
    try:
        p.set_pv1(0.0)
    except Exception as exc:
        report["notes"].append(f"Failed to set PV1 to 0 V: {exc}")

    for channel in ("A1", "A2", "A3"):
        try:
            report["baseline_voltages"][channel] = float(
                p.get_voltage(channel)
            )
        except Exception as exc:
            report["baseline_voltages"][channel] = f"Error: {exc}"

    report["notes"].append(
        f"Configured channels: potential={CFG.potential_channel}, "
        f"current={CFG.current_channel}, mode={CFG.current_mode}. "
        "Verify actual input ranges (-16 V/+16 V vs -3.3 V/+3.3 V) "
        "and TIA/shunt values."
    )

    return report, p


if __name__ == "__main__":
    audit, device = run_audit()

    for key, value in audit.items():
        print(f"{key}: {value}")

    if device is None:
        raise SystemExit(1)

