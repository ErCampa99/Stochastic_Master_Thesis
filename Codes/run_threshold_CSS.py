"""
Runner script: executes threshold_counts_N2.ipynb for three CSS initial states.
Each run uses a different theta_css and saves outputs to its own subfolder.
"""

import json
import subprocess
import sys
from pathlib import Path

NOTEBOOK = Path("threshold_counts_N2.ipynb")
KERNEL   = sys.executable  # use same venv python

CONFIGS = [
    dict(theta_expr="np.pi/6", label="CSS_theta_pi6"),
    dict(theta_expr="np.pi/4", label="CSS_theta_pi4"),
    dict(theta_expr="np.pi/3", label="CSS_theta_pi3"),
]

# To re-run only specific configs, override here, e.g.:
# CONFIGS = [CONFIGS[1], CONFIGS[2]]

# Cell id of the parameters cell
PARAMS_CELL_ID = "c8a19b17"

def params_source(theta_expr: str, label: str) -> str:
    return f"""\
# ── Parameters ─────────────────────────────────────────────────────────────
from pathlib import Path   # ensure Path is always available
# === Run configuration (modify here for each run) ===
theta_css   = {theta_expr}   # CSS polar angle
phi_css     = 0.0              # CSS azimuthal angle
state_label = "{label}"        # label used in folder and file names
# =====================================================

N_SPIN     = 2
gamma      = 1.0
phi1       = 0.0
phi2       = 0.0
T_END      = 5.0
dt         = 0.005
ntraj      = 200

assert dt <= 0.005, f"dt={{dt}} exceeds maximum allowed value of 0.005"
N_TIMES = int(round(T_END / dt)) + 1

# eta_1 is fixed (C_+ channel fully observed)
# eta_2 varies (C_- channel efficiency)
eta_1      = 1.0
eta_values = [1.0, 0.8, 0.5, 0.0]   # eta_2 values

THRESH_XI   = 0.98
THRESH_CHI2 = 0.98

times = np.linspace(0.0, T_END, N_TIMES)
t1    = 1.0 / gamma          # T1 = 1/gamma  (natural time unit)
x     = times / t1           # normalised x-axis: t/T1

# Output directory
out_dir = Path(f"Graphs/QFI/ntraj{{ntraj}}_{{state_label}}")
out_dir.mkdir(parents=True, exist_ok=True)

print(f'times: {{N_TIMES}} pt,  dt = {{times[1]-times[0]:.4f}},  T1 = {{t1}}')
print(f'ntraj = {{ntraj}},  eta_1 = {{eta_1}},  eta_2 values = {{eta_values}}')
print(f'Initial state: CSS(theta={{theta_css/np.pi:.4f}}*pi, phi={{phi_css:.4f}}),  label = {{state_label}}')
print(f'Output dir: {{out_dir}}')
"""

def run_config(cfg: dict):
    label = cfg["label"]
    theta_expr = cfg["theta_expr"]
    print(f"\n{'='*60}")
    print(f"  Running: {label}  (theta = {theta_expr})")
    print(f"{'='*60}")

    # Load notebook
    with open(NOTEBOOK, encoding="utf-8") as f:
        nb = json.load(f)

    # Patch parameters cell
    for cell in nb["cells"]:
        if cell.get("id") == PARAMS_CELL_ID:
            cell["source"] = params_source(theta_expr, label)
            cell["outputs"] = []
            cell["execution_count"] = None
            break
    else:
        raise RuntimeError(f"Parameters cell {PARAMS_CELL_ID} not found!")

    # Write temp notebook
    tmp = Path(f"_tmp_{label}.ipynb")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)

    # Execute
    cmd = [
        "jupyter", "nbconvert",
        "--to", "notebook",
        "--execute",
        "--inplace",
        f"--ExecutePreprocessor.timeout=7200",
        str(tmp),
    ]
    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        print(f"  ERROR: nbconvert failed for {label} (exit {result.returncode})")
    else:
        print(f"  Done: {label}")
        tmp.unlink(missing_ok=True)  # clean up temp file


if __name__ == "__main__":
    for cfg in CONFIGS:
        run_config(cfg)
    print("\nAll runs complete.")
