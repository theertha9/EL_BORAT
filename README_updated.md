# EL_BORAT — Cyclic Voltammetry Enhancement

A Python-based acquisition, analysis, hardware-audit, and power-profiling project for an **ExpEYES-17 / SEELab3** cyclic voltammetry (CV) setup.

The project aims to improve the quality and traceability of CV measurements using a configurable voltage sweep, current sensing through a shunt or transimpedance amplifier (TIA), signal-processing checks, and repeatability analysis.

> **Measurement note:** This project is an add-on workflow, not automatically a complete commercial potentiostat. In the present wiring concept, **PV1 drives the cell relative to ExpEYES ground**. Unless a proper potentiostat feedback loop is built and verified, treat the arrangement as a **two-electrode setup**. Do not describe the commanded PV1 voltage as a working-electrode potential versus a reference electrode unless the circuit actually measures and controls that potential.

## Project modules

| File | Responsibility |
|---|---|
| `config.py` | Sweep settings, channel mapping, current conversion settings, acquisition timing, and validation rules. |
| `audit_setup.py` | Checks device availability and records hardware/channel audit information. |
| `cv_acquire.py` | Commands the PV1 sweep and records potential, current, timing, and metadata. |
| `cv_analyze.py` | Estimates noise, applies signal-processing filters, extracts CV peaks, and evaluates repeatability. |
| `power_profile.py` | Records power-monitor measurements for defined operating states. Requires a real power monitor for experimental results. |
| `run_pipeline.py` | Command-line entry point for coordinating the workflow and offline testing. |
| `requirements.txt` | Python package dependencies. |
| `data/` | Local output directory for raw measurements, metadata, plots, audit records, and reports. Created when needed. |

## Requirements

- Python **3.10 or newer is recommended** for the current checked-in source.
- Python 3.8+ is a project compatibility goal, but it is not yet guaranteed by every module: some current type annotations use syntax that requires Python 3.10 unless postponed annotations are enabled. Use Python 3.10+ until that compatibility cleanup has been completed and tested.
- `pip` and the standard-library `venv` module.
- An ExpEYES-17 or SEELab3 device and its USB driver/permissions for real hardware acquisition.
- A compatible, externally connected power monitor for experimental power profiling. The project does not treat placeholder readings as measured data.

### ExpEYES package name versus import name

Install the library-only distribution named **`eyes17lib`**. The Python import used by this project is:

```python
from eyes17 import eyes
```

The PyPI distribution name and Python import name are different. The GUI-oriented `eyes17` distribution is not needed just to import the communication library used by this project.

## Installation

### 1. Get the repository

For the main repository:

```bash
git clone https://github.com/theertha9/EL_BORAT.git
cd EL_BORAT
```

For contribution work, clone your fork instead, then create a feature branch before editing:

```bash
git switch -c feature/your-change
```

### 2. Create a virtual environment

A virtual environment keeps this project's packages separate from the system Python installation.

#### Windows — PowerShell

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation for this terminal session, run:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

#### Windows — Command Prompt

```cmd
py -3 -m venv .venv
.venv\Scripts\activate.bat
```

#### Linux / macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
```

After activation, the terminal prompt normally shows `(.venv)`. Use `python` and `python -m pip` while the environment is active so commands target the virtual environment.

### 3. Upgrade pip and install dependencies

Run from the repository root with the virtual environment activated:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 4. Verify the installation

Check that the required modules import successfully:

```bash
python -c "import numpy, pandas, scipy, matplotlib; from eyes17 import eyes; print('Imports OK')"
```

This is an **import check only**. It does not connect to, test, or calibrate an ExpEYES device.

You can also check the command-line arguments without starting an acquisition:

```bash
python run_pipeline.py --help
```

## Quick setup and test commands (Windows PowerShell)

Run these commands from the repository root, in order:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -c "import numpy, pandas, scipy, matplotlib; from eyes17 import eyes; print('Imports OK')"
python validate_filters.py
python run_pipeline.py --simulate --repeats 1
```

The import command checks that the required Python packages and `eyes17` import are available. `validate_filters.py` runs the filter-validation checks. The final command runs one simulated pipeline repeat; simulated results are software-test data and must not be treated as experimental measurements.

If PowerShell blocks virtual-environment activation, run this for the current terminal session and retry activation:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

## ExpEYES-17 / SEELab3 hardware setup

1. Install the USB communication driver for the board on Windows. On Linux, configure the required USB/serial-device permissions for your distribution.
2. Connect the board and confirm that the operating system detects it before running a hardware audit.
3. Verify the actual analog-input ranges, PV1 behavior, channel wiring, current-sense circuit, and shunt/TIA values on the physical setup. Do not assume that PV1's output range is also the analog-input range.
4. Confirm the current-sense polarity and conversion factor before interpreting current signs or magnitudes.
5. Run the hardware audit and inspect its report before attempting a CV scan.

See the [ExpEYES installation manual](https://eyes17lib.readthedocs.io/en/latest/installation/) for platform-specific installation and USB-access notes.

## Configuration

The default configuration is defined in `config.py` through `HardwareConfig` and `CFG`.

Important settings include:

- `v_start`, `v_vertex`, `v_end`: commanded sweep potentials in volts.
- `scan_rate_v_per_s`: scan rate in volts per second.
- `step_v`: commanded voltage increment in volts.
- `n_cycles`: number of cycles.
- `potential_channel`, `current_channel`: analog input channels used by the acquisition code.
- `current_mode`: `shunt` or `tia`.
- `shunt_ohm`, `tia_gain_v_per_a`, `current_sign`: current conversion settings.
- `equilibration_s`, `avg_window_s`: settling and averaging settings.
- `calibrated`: remains `False` until the measurement chain has actually been calibrated.
- `electrode_area_cm2`: leave unset until the active electrode area is measured and recorded.

Values marked as placeholders must be replaced with values verified for the actual circuit. A configuration file cannot calibrate the hardware by itself.

## Temperature sensing status

Temperature acquisition is currently a **stub**, not a verified sensor measurement. A manually supplied temperature may be recorded when the acquisition interface supports it, but it must be identified as a manual value. Do not present a stub or fallback value as a measured temperature. A real temperature sensor, its wiring, calibration, and data source must be implemented and checked before reporting experimental temperature data.

## Data integrity

- Raw acquisition files are source records. Do not edit, replace, or fabricate their contents.
- Keep the associated `.meta.json` file with each raw CSV; it records the settings and context needed to interpret the run.
- Simulated data is for software testing only. It must be clearly labelled as simulated and must never be mixed with real experimental measurements.
- Do not use power-profile values as experimental results unless they came from a connected, validated power monitor.
- The default output directory is `data/`. The `DATA_DIR` environment variable can override it.

Examples:

**Windows PowerShell**

```powershell
$env:DATA_DIR = "D:\EL_BORAT\data"
```

**Windows Command Prompt**

```cmd
set DATA_DIR=D:\EL_BORAT\data
```

**Linux / macOS**

```bash
export DATA_DIR="$HOME/EL_BORAT-data"
```

## Running and current development status

The command-line interface can be inspected safely with:

```bash
python run_pipeline.py --help
```

The offline simulation path is being updated so that it exercises the real acquisition routine through a simulated ExpEYES device (`FakeEyes`) and a virtual clock. Until that change is complete, distinguish the existing synthetic CSV path from a hardware-level simulation: synthetic traces are software test fixtures, not measurements.

Power profiling is also being updated to require a real monitor for non-simulated runs and to label any explicitly requested simulation. Do not report placeholder power values as measurements. The final invocation syntax for optional power profiling should be documented after the corresponding `run_pipeline.py` changes are merged.

## Leaving the virtual environment

When finished, deactivate the environment:

```bash
deactivate
```

To reactivate it later:

- Windows PowerShell: `.\.venv\Scripts\Activate.ps1`
- Windows Command Prompt: `.venv\Scripts\activate.bat`
- Linux / macOS: `source .venv/bin/activate`

## References

- [ExpEYES-17 / SEELab3 installation manual](https://eyes17lib.readthedocs.io/en/latest/installation/)
- [ExpEYES-17 programming introduction](https://eyes17lib.readthedocs.io/en/latest/programming/intro/)
- [Python Packaging User Guide — installing packages with pip and virtual environments](https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/)
