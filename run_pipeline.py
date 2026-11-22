
import argparse
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

from config import BASE_DIR, CFG, DATA_DIR
from cv_analyze import analyze_single_csv, compute_repeat_rsd


def write_synthetic_report(
    report_path: Path,
    csv_path: Path,
    measurements: pd.DataFrame,
) -> None:
    """Write the sample_dataset settings and generated synthetic readings."""
    with report_path.open("x", encoding="utf-8") as report:
        report.write("CYCLIC VOLTAMMETRY EXPERIMENT REPORT\n")
        report.write("=" * 38 + "\n\n")
        report.write("Data source: SYNTHETIC (not physical instrument readings)\n")
        report.write(f"Configuration source: {BASE_DIR / 'sample_dataset.txt'}\n")
        report.write(f"Raw CSV: {csv_path.name}\n")
        report.write(f"Report: {report_path.name}\n\n")

        report.write("EXPERIMENT AND HARDWARE CONFIGURATION\n")
        report.write("-------------------------------------\n")
        for name, value in asdict(CFG).items():
            report.write(f"CFG.{name}: {value}\n")
        report.write("\n")

        report.write("READINGS AT EACH GENERATED POTENTIAL STEP\n")
        report.write("-----------------------------------------\n")
        report.write(
            "The first pre-step reading is unavailable because simulation "
            "does not sample a physical baseline. Later pre-step values are "
            "the preceding generated point. Each after-step value is the "
            "synthetic measurement generated for that commanded potential; "
            "individual instrument samples are not available in simulation.\n\n"
        )

        previous = None
        for row in measurements.itertuples(index=False):
            report.write(
                f"Step {row.sample_idx} | t={row.t_rel_s:.6f} s | "
                f"commanded potential={row.v_cmd_v:.9g} V\n"
            )

            if previous is None:
                report.write(
                    "  Before: no physical baseline reading (simulation)\n"
                )
            else:
                report.write(
                    f"  Before (previous generated point): "
                    f"potential={previous.v_meas_v:.9g} V, "
                    f"current-sense={previous.v_curr_sense_v:.9g} V, "
                    f"current={previous.current_a:.9g} A\n"
                )

            report.write(
                f"  After (synthetic): potential={row.v_meas_v:.9g} V, "
                f"current-sense={row.v_curr_sense_v:.9g} V, "
                f"current={row.current_a:.9g} A "
                f"({row.current_ua:.9g} uA), "
                f"temperature={row.temp_c}\n\n"
            )
            previous = row


def make_synthetic_cv_csv(
    repeat_idx: int,
    scan_rate_v_per_s: float = CFG.scan_rate_v_per_s,
) -> Path:
    """Create a synthetic CV trace for offline testing."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    v_fwd = np.linspace(
        CFG.v_start,
        CFG.v_vertex,
        int(round(abs(CFG.v_vertex - CFG.v_start) / CFG.step_v)) + 1,
    )
    v_rev = np.linspace(
        CFG.v_vertex,
        CFG.v_end,
        int(round(abs(CFG.v_end - CFG.v_vertex) / CFG.step_v)) + 1,
    )[1:]
    v = np.concatenate([v_fwd, v_rev])

    # Simulated branch currents: oxidation on the forward sweep
    # and reduction on the reverse sweep.
    rng = np.random.default_rng(seed=100 + repeat_idx)
    scale = np.sqrt(scan_rate_v_per_s / 0.05)
    noise = rng.normal(0.0, 0.35, size=len(v))

    forward_mask = np.arange(len(v)) < len(v_fwd)
    i_ua = np.zeros(len(v), dtype=float)

    i_ua[forward_mask] = (
        17.0
        * scale
        * np.exp(-((v[forward_mask] - 0.25) / 0.065) ** 2)
    )
    i_ua[~forward_mask] = (
        -16.5
        * scale
        * np.exp(-((v[~forward_mask] - 0.16) / 0.065) ** 2)
    )
    i_ua += noise

    total_sweep_distance = (
        abs(CFG.v_vertex - CFG.v_start)
        + abs(CFG.v_end - CFG.v_vertex)
    )
    total_time = total_sweep_distance / scan_rate_v_per_s
    t = np.linspace(0.0, total_time, len(v))

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    out_csv = DATA_DIR / f"cv_raw_sim_{run_id}_rep{repeat_idx}.csv"

    measurements = pd.DataFrame(
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
    )

    measurements.to_csv(out_csv, index=False)

    report_path = out_csv.with_suffix(".report.txt")

    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "run_label": "synthetic",
        "simulated": True,
        "repeat_idx": repeat_idx,
        "raw_csv": out_csv.name,
        "scan_params": {
            "v_start": CFG.v_start,
            "v_vertex": CFG.v_vertex,
            "v_end": CFG.v_end,
            "scan_rate_v_per_s": scan_rate_v_per_s,
            "step_v": CFG.step_v,
            "n_cycles": CFG.n_cycles,
            "calibrated": CFG.calibrated,
        },
        "acquisition": {
            "complete": True,
            "n_samples_saved": len(v),
            "achieved_scan_rate_v_per_s": scan_rate_v_per_s,
            "requested_scan_rate_v_per_s": scan_rate_v_per_s,
        },
        "program_config": asdict(CFG),
        "report_file": report_path.name,
    }

    with out_csv.with_suffix(".meta.json").open(
        "w", encoding="utf-8"
    ) as f:
        json.dump(metadata, f, indent=2)

    write_synthetic_report(report_path, out_csv, measurements)

    return out_csv


def main() -> int:
    parser = argparse.ArgumentParser(
        description="BORAT CV Enhancement Pipeline (ExpEYES/SEELab3)"
    )

    parser.add_argument(
        "--simulate",
        action="store_true",
        help="Run without physical ExpEYES hardware",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=CFG.n_repeats,
        help="Number of repeat scans (>=3 recommended)",
    )
    parser.add_argument(
        "--power",
        action="store_true",
        help="Request power profiling (currently unavailable)",
    )

    args = parser.parse_args()

    if args.repeats < 1:
        parser.error("--repeats must be at least 1.")

    if args.power:
        parser.error(
            "Power profiling is not operational yet: "
            "power_monitor.py and verified workload callbacks are not available."
        )

    csv_files: List[Path] = []

    if args.simulate:
        print("Running in --simulate mode (synthetic data)...")

        for r in range(1, args.repeats + 1):
            csv_files.append(
                make_synthetic_cv_csv(r, CFG.scan_rate_v_per_s)
            )

    else:
        from audit_setup import run_audit
        from cv_acquire import acquire_cv_scan

        # run_audit opens the device once and returns that same handle.
        audit, p = run_audit()

        print("Audit summary:", audit["connected"], audit["notes"])

        if not audit["connected"] or p is None:
            print("Hardware audit failed. Acquisition cancelled.")
            return 1

        for r in range(1, args.repeats + 1):
            raw_path, meta_path = acquire_cv_scan(
                p,
                run_label="ferricyanide",
                repeat_idx=r,
                setup_audit=audit,
            )

            print(f"Saved: {raw_path.name}, {meta_path.name}")
            csv_files.append(raw_path)

    if not csv_files:
        print("No CSV files were generated or acquired.")
        return 1

    analyses = [
        analyze_single_csv(csv_file, save_plot=True)
        for csv_file in csv_files
    ]

    ipa_vals = [a["raw_peaks"]["ipa"] for a in analyses]
    ipc_vals = [a["raw_peaks"]["ipc"] for a in analyses]

    print(
        f"Repeat Ipa RSD (%): {compute_repeat_rsd(ipa_vals):.2f}% "
        f"(target <= {CFG.target_rsd_pct}%)"
    )
    print(
        f"Repeat Ipc RSD (%): {compute_repeat_rsd(ipc_vals):.2f}% "
        f"(target <= {CFG.target_rsd_pct}%)"
    )

    for analysis in analyses:
        print(
            f"\nFile: {analysis['file']} | "
            f"Raw noise sigma: "
            f"{analysis['raw_noise']['noise_sigma']:.4f} µA"
        )

        for filter_name, filter_result in analysis["filters"].items():
            if filter_result.get("skipped"):
                print(
                    f"  [{filter_name}] SKIPPED: "
                    f"{filter_result['reason']}"
                )
                continue

            print(
                f"  [{filter_name}] "
                f"shift={filter_result['max_peak_shift_mv']:.2f} mV, "
                f"dI={filter_result['max_peak_current_chg_pct']:.2f}%, "
                f"within_targets={filter_result['within_targets']}"
            )

    if args.simulate:
        print("\nPower profiling skipped in simulation mode.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

