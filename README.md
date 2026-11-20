# Enhancement of an Existing Cyclic .Voltammetry Setup (ExpEYES-17 / SEELab3)

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
