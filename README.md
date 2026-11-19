# Enhancement of an Existing Cyclic Voltammetry Setup (ExpEYES-17 / SEELab3)

**Team BORAT** | Department of Computer Science and Engineering, RV College of Engineering (RVCE)  
**Mentor:** Dr. Rajesh BM  
**Theme:** Electrochemical Instrumentation — Cyclic Voltammetry Enhancement via Auxiliary Sensing, Python-Based Signal Processing, and Power Optimization

---

## Overview

Commercial potentiostats are expensive, whereas teaching and research laboratories frequently already possess low-cost hardware such as **ExpEYES-17** or **SEELab3**. Rather than replacing existing lab equipment, this project provides a modular, budget-conscious ($\le ₹10,000$ add-on cap) Python acquisition, analysis, and profiling workflow designed to extract richer, cleaner, and better-documented cyclic voltammetry (CV) data.

### Core Capabilities
1. **Hardware & Interface Audit (`audit_setup.py`):** Verifies USB connection via `eyes17lib`, records baseline channel voltages (`A1`, `A2`, `A3`), and assists in documenting hardware range and front-end limits.
2. **CV Acquisition & Auxiliary Logging (`cv_acquire.py`):** Sweeps `PV1` linearly forward and backward across user-defined potential windows and scan rates ($20\text{–}200\text{ mV/s}$), logs cell potential, current (shunt or TIA front end), and temperature metadata, and preserves an **immutable raw CSV** plus sidecar `.meta.json` for every scan.
3. **Signal Processing & Validation (`cv_analyze.py`):** Computes baseline noise (`std`, `RMS`) and approximate SNR, evaluates smoothing/noise filters (Savitzky–Golay, median, low-pass Butterworth, and optional $50\text{ Hz}$ notch) against raw data, verifies filter distortion limits, extracts anodic/cathodic peaks ($E_{pa}, i_{pa}, E_{pc}, i_{pc}$), calculates repeat Relative Standard Deviation ($\text{RSD}\%$), and generates raw-vs.-processed comparison plots.
4. **Four-State Power Profiling (`power_profile.py`):** Profiles voltage, current, and power across **idle**, **sensing**, **scan**, and **transfer** states for battery-powered auxiliary subsystems and exports raw and summary CSVs.
5. **End-to-End & Offline Simulation Runner (`run_pipeline.py`):** Orchestrates multi-repeat runs on physical hardware or in `--simulate` mode using synthetic dummy-cell/CV data.

---

## Project Structure

```text
borat_cv_expeyes/
├── config.py           # Scan parameters, channel mapping, shunt/TIA scaling, and validation targets
├── audit_setup.py      # USB connection check and baseline channel voltage audit
├── cv_acquire.py       # PV1 staircase sweep, auxiliary sensor hooks, raw CSV + JSON metadata writer
├── cv_analyze.py       # Noise metrics, filtering, peak extraction, RSD calculation, and plotting
├── power_profile.py    # Four-state power logger (idle, sensing, scan, transfer)
├── run_pipeline.py     # Main CLI entry point (supports live hardware and --simulate mode)
├── README.md           # Project documentation
└── data/               # Auto-generated directory for raw CSVs, metadata JSONs, PNG plots, and reports
```

---

## Prerequisites & Setup

### 1. Hardware & OS Setup
* **Hardware:** ExpEYES-17 or SEELab3 connected over USB (close the desktop ExpEYES GUI before running Python scripts so the serial port is free).
* **Linux (`udev` / serial permissions):** Ensure your user belongs to the `dialout` group (or run the ExpEYES `postinst.sh` permissions script) and re-login:
  ```bash
  sudo usermod -aG dialout $USER
  ```
* **Windows:** Verify that the USB-to-serial driver is installed and a COM port is assigned in Device Manager.

### 2. Python Environment & VS Code Setup
1. Open the `borat_cv_expeyes/` folder in **VS Code** (`File > Open Folder...`).
2. Install required Python dependencies in your target environment:
   ```bash
   python3 -m pip install --upgrade pip
   python3 -m pip install eyes17lib numpy scipy matplotlib pandas
   ```
3. In VS Code, open the Command Palette (`Ctrl+Shift+P` / `Cmd+Shift+P`), run **`Python: Select Interpreter`**, and select the same Python interpreter where `eyes17lib` was installed.

---

## Usage

### Step 1: Audit Connected Hardware
Verify that `eyes17lib` detects your board and inspect baseline channel readings:
```bash
python3 audit_setup.py
```

### Step 2: Configure Calibration & Scan Parameters
Edit `config.py` after auditing your lab hardware:
* Set `v_start`, `v_vertex`, `v_end`, `scan_rate_v_per_s`, and `step_v`.
* Verify `potential_channel` (`A1`) and `current_channel` (`A2`).
* Set `current_mode` (`"shunt"` or `"tia"`) and update `shunt_ohm` or `tia_gain_v_per_a` and `current_sign` to match your front-end circuit.

### Step 3: Run Offline Simulation (Before Wiring Hardware)
Test the full acquisition, filtering, peak extraction, plotting, and power-profiling pipeline using synthetic data:
```bash
python3 run_pipeline.py --simulate --repeats 3
```

### Step 4: Run Live CV Experiments
With your dummy cell or three-electrode cell (WE, RE, CE) and audited front end connected:
```bash
python3 run_pipeline.py --repeats 3
```

All outputs are saved inside `data/`:
* `cv_raw_<label>_rep<n>_<timestamp>.csv` — untouched raw time, potential, current, and temperature log
* `cv_raw_<label>_rep<n>_<timestamp>.meta.json` — scan parameters, front-end scaling, and between-scan pH notes
* `cv_raw_<label>_rep<n>_<timestamp>.png` — comparison plot of raw vs. filtered voltammograms
* `power_profile_<timestamp>.csv` & `*_summary.csv` — four-state power profiling logs

---

## Experimental & Validation Targets

| Metric / Rule | Target / Guideline |
| :--- | :--- |
| **Raw Data Preservation** | Always saved untouched; filters are applied only to copies during analysis |
| **Peak Potential Shift ($\Delta E_p$)** | $< 10\text{ mV}$ compared to raw trace |
| **Peak Current Distortion ($\Delta i_p / i_{p,\text{raw}}$)** | $< 5\%$ compared to raw trace |
| **Repeatability ($\text{RSD}$)** | $\le 5\%$ across $\ge 3$ repeat scans |
| **pH Measurement Protocol** | Logged **between** scans in metadata rather than continuously during sweeps to avoid electrode coupling |
| **Power Optimization Scope** | Measured across *idle*, *sensing*, *scan*, and *transfer* states on battery-powered add-ons (software cannot reduce native USB-powered ExpEYES draw) |

---

## Ethics & Safety Notes
* Follow standard laboratory safety procedures when handling electrochemical reagents (e.g., potassium ferricyanide and $\text{KCl}$ supporting electrolyte).
* Never overwrite or fabricate raw measurement files; report hardware limits, sampling constraints, and filter trade-offs transparently.
