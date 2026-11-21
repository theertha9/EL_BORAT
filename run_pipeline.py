import argparse #xyz
from pathlib import Path
import numpy as np
import pandas as pd
from config import CFG, DATA_DIR
from cv_analyze import analyze_single_csv, compute_repeat_rsd
from power_profile import run_four_state_power_profile
from typing import List


def make_synthetic_cv_csv(repeat_idx: int, scan_rate_v_per_s: float = 0.05) -> Path:
    """Create a synthetic reversible-like CV trace for offline testing before hardware wiring."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    v_fwd = np.linspace(-0.2, 0.6, 160)
    v_rev = np.linspace(0.6, -0.2, 160)[1:]
    v = np.concatenate([v_fwd, v_rev])

    # Simulated anodic peak near +0.25 V and cathodic peak near +0.16 V
    rng = np.random.default_rng(seed=100 + repeat_idx)
    scale = np.sqrt(scan_rate_v_per_s / 0.05)
    i_anodic = 17.0 * scale * np.exp(-((v - 0.25) / 0.065) ** 2)
    i_cathodic = -16.5 * scale * np.exp(-((v - 0.16) / 0.065) ** 2)
    noise = rng.normal(0.0, 0.35, size=len(v))
    i_ua = i_anodic + i_cathodic + noise

    total_time = abs(0.6 - (-0.2)) * 2 / scan_rate_v_per_s
    t = np.linspace(0.0, total_time, len(v))

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    out_csv = DATA_DIR / f"cv_raw_sim_{run_id}_rep{repeat_idx}.csv"
    pd.DataFrame(
        {
            "sample_idx": np.arange(len(v)),
            "t_rel_s": t,
            "v_cmd_v": v,
            "v_meas_v": v + rng.normal(0.0, 0.0005, size=len(v)),
            "v_curr_sense_v": i_ua * 1e-6 * CFG.shunt_ohm,
            "current_a": i_ua * 1e-6,
            "current_ua": i_ua,
            "temp_c": 25.0 + 0.05 * repeat_idx,
            "step_elapsed_s": np.gradient(t),
        }
    ).to_csv(out_csv, index=False)
    return out_csv


def main() -> None:
    parser = argparse.ArgumentParser(description="BORAT CV Enhancement Pipeline (ExpEYES/SEELab3)")
    parser.add_argument("--simulate", action="store_true", help="Run without physical ExpEYES hardware")
    parser.add_argument("--repeats", type=int, default=3, help="Number of repeat scans (>=3 recommended)")
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be at least 1.")

    csv_files: List[Path] = []

    if args.simulate:
        print("Running in --simulate mode (synthetic data)...")
        for r in range(1, args.repeats + 1):
            csv_files.append(make_synthetic_cv_csv(r, CFG.scan_rate_v_per_s))
    else:
        from eyes17 import eyes
        from audit_setup import run_audit
        from cv_acquire import acquire_cv_scan
        audit = run_audit()
        print("Audit summary:", audit["connected"], audit["notes"])
        if not audit["connected"]:
            return
        p = eyes.open()
        for r in range(1, args.repeats + 1):
            raw_path, meta_path = acquire_cv_scan(p, run_label="ferricyanide", repeat_idx=r)
            print(f"Saved: {raw_path.name}, {meta_path.name}")
            csv_files.append(raw_path)

    analyses = [analyze_single_csv(f, save_plot=True) for f in csv_files]
    ipa_vals = [a["raw_peaks"]["ipa"] for a in analyses]
    ipc_vals = [a["raw_peaks"]["ipc"] for a in analyses]

    print(f"Repeat Ipa RSD (%): {compute_repeat_rsd(ipa_vals):.2f}% (target <= {CFG.target_rsd_pct}%)")
    print(f"Repeat Ipc RSD (%): {compute_repeat_rsd(ipc_vals):.2f}% (target <= {CFG.target_rsd_pct}%)")

    for a in analyses:
        print(f"\nFile: {a['file']} | Raw baseline std: {a['raw_noise']['baseline_std']:.4f} µA")
        for fname, fres in a["filters"].items():
            print(
                f"  [{fname}] shift={fres['max_peak_shift_mv']:.2f} mV, "
                f"dI={fres['max_peak_current_chg_pct']:.2f}%, within_targets={fres['within_targets']}"
            )

if not args.simulate:
    try:
        pwr_csv = run_four_state_power_profile(duration_per_state_s=2.0)
        print(f"\nPower profile saved to: {pwr_csv.name}")
    except RuntimeError as exc:
        print(f"\nPower profiling skipped: {exc}")
else:
    print("\nPower profiling skipped in simulation mode.")


if __name__ == "__main__":
    main()
