"""
Homodyne dephasing simulation for N=2 — phi_2 scan.
====================================================
Simulates the 4 quantities (Conc, Xi2_KU, Norm_J, Xi2_WIN) with phi_1=0 fixed
and phi_2 varying over 4 values in [0, pi/2].
Efficiencies are fixed per run: two canonical cases
    - ideal:    eta_1=1, eta_2=1
    - one-side: eta_1=1, eta_2=0
are selected via --eta1 / --eta2 from the command line.

Each output figure contains 4 curves (one per phi_2 value).
"""

import ast
import argparse
import json
import os
import time
from fractions import Fraction
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from joblib import Parallel, delayed
except ImportError:
    Parallel = None
    delayed = None

from qutip import (
    basis,
    expect,
    ket2dm,
    qeye,
    sigmax,
    sigmay,
    sigmaz,
    smesolve,
    tensor,
    variance,
)

# ── Basis & states ─────────────────────────────────────────────────────────────
gnd = basis(2, 0)
exc = basis(2, 1)
plus_state = (gnd + exc).unit()
PlusPlus   = tensor(plus_state, plus_state)
ee         = tensor(exc, exc)

def css_2(theta: float, phi: float):
    single_qubit = (
        np.sin(theta / 2.0) * basis(2, 0)
        + np.exp(1j * phi) * np.cos(theta / 2.0) * basis(2, 1)
    )
    return tensor(single_qubit, single_qubit)

# ── Collective spin operators ──────────────────────────────────────────────────
I_2 = qeye(2)
J_x = 0.5 * (tensor(sigmax(), I_2) + tensor(I_2, sigmax()))
J_y = 0.5 * (tensor(sigmay(), I_2) + tensor(I_2, sigmay()))
J_z = 0.5 * (tensor(sigmaz(), I_2) + tensor(I_2, sigmaz()))
H_free = 0.0 * J_z

# Dephasing channels (same as mina_SLURM_thesis.py)
Cp = np.sqrt(0.5) * (tensor(sigmaz(), I_2) + tensor(I_2, sigmaz()))
Cm = np.sqrt(0.5) * (tensor(sigmaz(), I_2) - tensor(I_2, sigmaz()))

def collapsing_operators(gamma, phi_1, phi_2, eta_1, eta_2):
    return [
        np.sqrt(gamma * eta_1) * np.exp(1j * phi_1) * Cp,
        np.sqrt(gamma * eta_2) * np.exp(1j * phi_2) * Cm,
    ]

# ── Observables ────────────────────────────────────────────────────────────────
def concurrence_pure_state(state):
    c = state.full().flatten()
    return 2.0 * np.abs(c[0] * c[3] - c[1] * c[2])

def concurrence_for_solver_general(t, state):
    if state.isket:
        conc = concurrence_pure_state(state)
    else:
        sysy = tensor(sigmay(), sigmay())
        rho_tilde = (state * sysy) * (state.conj() * sysy)
        evals = rho_tilde.eigenenergies()
        evals = abs(np.sort(np.real(evals)))
        sqrt_evals = np.sqrt(evals)
        lsum = sqrt_evals[3] - sqrt_evals[2] - sqrt_evals[1] - sqrt_evals[0]
        conc = np.maximum(0.0, lsum)
    return float(np.real_if_close(conc))

def eval_spin_components(state):
    return expect(J_x, state), expect(J_y, state), expect(J_z, state)

def xi_KU_solver(t, rho, N=None, tol=1e-12):
    if N is None:
        N = int(round(np.log2(rho.shape[0])))
    Jx_exp, Jy_exp, Jz_exp = eval_spin_components(rho)
    Jmean = np.array([float(np.real_if_close(v)) for v in [Jx_exp, Jy_exp, Jz_exp]])
    m  = np.linalg.norm(Jmean)
    Js = [J_x, J_y, J_z]
    C  = np.zeros((3, 3), dtype=float)
    for i, Ji in enumerate(Js):
        for j, Jj in enumerate(Js):
            sm = 0.5 * expect(Ji * Jj + Jj * Ji, rho)
            C[i, j] = float(np.real_if_close(sm - Jmean[i] * Jmean[j]))
    if m > tol:
        u  = Jmean / m
        a  = np.array([1., 0., 0.]) if abs(u[0]) < 0.9 else np.array([0., 1., 0.])
        e1 = a - u * np.dot(u, a); e1 /= np.linalg.norm(e1)
        e2 = np.cross(u, e1)
        C2 = np.array([[e1@C@e1, e1@C@e2], [e2@C@e1, e2@C@e2]], dtype=float)
        lam_min = np.linalg.eigvalsh(C2)[0]
    else:
        lam_min = np.linalg.eigvalsh(C)[0]
    lam_min = max(float(np.real_if_close(lam_min)), 0.0)
    return float((4.0 / N) * lam_min)

def norm_J(t, rho):
    return np.linalg.norm(eval_spin_components(rho))

def xi_WIN_solver(t, rho, N=None, tol=1e-12):
    if N is None:
        N = int(round(np.log2(rho.shape[0])))
    norm = np.linalg.norm(eval_spin_components(rho))
    return xi_KU_solver(t, rho) * ((N / 2) / norm) ** 2 if norm > tol else xi_KU_solver(t, rho)

# ── Angle helpers ──────────────────────────────────────────────────────────────
def angle_to_path(phi, max_den=12, tol=1e-10):
    twopi = 2 * np.pi
    x = (phi % twopi + twopi) % twopi
    if abs(x) < tol:
        return "0"
    q = Fraction(x / np.pi).limit_denominator(max_den)
    n, d = q.numerator, q.denominator
    if n == 0: return "0"
    if d == 1: return "pi" if n == 1 else f"{n}pi"
    return ("pi_" + str(d)) if n == 1 else f"{n}pi_{d}"

def angle_to_tex(phi, max_den=12, tol=1e-10):
    twopi = 2 * np.pi
    x = (phi % twopi + twopi) % twopi
    if abs(x) < tol:
        return "0"
    q = Fraction(x / np.pi).limit_denominator(max_den)
    n, d = q.numerator, q.denominator
    if n == 0: return "0"
    if d == 1: return r"\pi" if n == 1 else fr"{n}\pi"
    num = "" if abs(n) == 1 else str(abs(n))
    s = r"\pi / %d" % d if abs(n) == 1 else r"%s\pi / %d" % (num, d)
    return s if n > 0 else "-" + s

# ── Simulation bookkeeping ─────────────────────────────────────────────────────
DEFAULT_EOPS = [
    ("Conc",   concurrence_for_solver_general),
    ("Xi2_KU", xi_KU_solver),
    ("Norm_J", norm_J),
]

LABELS = {
    "Conc":    r"$\overline{\mathcal{C}}$",
    "Xi2_KU":  r"$\overline{\xi^2_{KU}}$",
    "Norm_J":  r"$\overline{|\langle \mathbf{J} \rangle|}$",
    "Xi2_WIN": r"$\overline{\xi^2_{WIN}}$",
}

# 4 phi_2 values from 0 to pi/2 inclusive
PHI2_VALUES = [0.0, np.pi / 6, np.pi / 3, np.pi / 2]

# Colors for the 4 phi_2 curves
COLORS_PHI2 = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

def build_initial_state(kind: str, theta: float, phi_state: float):
    if kind == "plusplus": return PlusPlus, np.pi / 2.0
    if kind == "ee":       return ee,       np.pi
    if kind == "css":      return css_2(theta, phi_state), theta
    raise ValueError(f"Unknown state kind: {kind}")

# ── Core simulation (one phi_2 value) ─────────────────────────────────────────
def run_single_phi2(rho0, times, gamma, phi1, phi2, eta_1, eta_2, e_ops, ntraj, num_cpus):
    solver_cpus = max(1, int(num_cpus))
    options = {
        "keep_runs_results": False,
        "num_cpus": solver_cpus,
        "map": "parallel" if solver_cpus > 1 else "serial",
        "method": "milstein",
    }
    sol = smesolve(
        H_free, rho0, times,
        c_ops  = collapsing_operators(gamma, phi1, phi2, 1.0 - eta_1, 1.0 - eta_2),
        sc_ops = collapsing_operators(gamma, phi1, phi2, eta_1, eta_2),
        heterodyne=False,
        e_ops=e_ops,
        ntraj=ntraj,
        options=options,
    )
    return sol.expect   # list of arrays, one per e_op

# ── Argument parsing ───────────────────────────────────────────────────────────
def parse_angle_arg(raw: str) -> float:
    text = raw.strip()
    if not text:
        raise argparse.ArgumentTypeError("Angle value cannot be empty.")
    try:
        return float(text)
    except ValueError:
        pass
    try:
        node = ast.parse(text, mode="eval")
    except SyntaxError as exc:
        raise argparse.ArgumentTypeError(f"Invalid angle expression '{raw}'.") from exc

    def _eval(expr):
        if isinstance(expr, ast.Expression):  return _eval(expr.body)
        if isinstance(expr, ast.Constant) and isinstance(expr.value, (int, float)):
            return float(expr.value)
        if isinstance(expr, ast.Name) and expr.id == "pi":
            return float(np.pi)
        if (isinstance(expr, ast.Attribute)
                and isinstance(expr.value, ast.Name)
                and expr.value.id in {"np", "numpy"}
                and expr.attr == "pi"):
            return float(np.pi)
        if isinstance(expr, ast.UnaryOp) and isinstance(expr.op, (ast.UAdd, ast.USub)):
            val = _eval(expr.operand)
            return val if isinstance(expr.op, ast.UAdd) else -val
        if isinstance(expr, ast.BinOp) and isinstance(expr.op, (ast.Add, ast.Sub, ast.Mult, ast.Div)):
            l, r = _eval(expr.left), _eval(expr.right)
            if isinstance(expr.op, ast.Add):  return l + r
            if isinstance(expr.op, ast.Sub):  return l - r
            if isinstance(expr.op, ast.Mult): return l * r
            return l / r
        raise argparse.ArgumentTypeError(f"Unsupported expression '{raw}'.")

    return float(_eval(node))

def make_parser():
    p = argparse.ArgumentParser(description="N=2 homodyne — phi_2 scan at fixed eta")
    p.add_argument("--state",     choices=["plusplus", "ee", "css"], default="plusplus")
    p.add_argument("--theta",     type=float, default=np.pi / 2.0)
    p.add_argument("--phi-state", type=float, default=0.0)
    p.add_argument("--gamma",     type=float, default=1.0)
    # phi_1 is fixed to 0; phi_2 is scanned internally
    p.add_argument("--phi1",      type=parse_angle_arg, default=0.0,
                   help="Fixed phi_1 (default 0)")
    # eta: two canonical cases via CLI
    p.add_argument("--eta1",      type=float, default=1.0,
                   help="Detection efficiency channel 1 (default 1)")
    p.add_argument("--eta2",      type=float, default=1.0,
                   help="Detection efficiency channel 2 (default 1)")
    p.add_argument("--ntraj",     type=int,   default=20)
    p.add_argument("--t-end",     type=float, default=5.0,
                   help="Final time in units of T1")
    p.add_argument("--dt",        type=float, default=0.005,
                   help="Time step in units of T1 (max 0.005 with Milstein)")
    p.add_argument("--num-cpus",  type=int,   default=max(1, os.cpu_count() - 1))
    p.add_argument("--out-root",  type=str,   default=r"./Graphs/Phi2Scan_N2")
    p.add_argument("--usetex",    action="store_true")
    return p

# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    t0   = time.perf_counter()
    print("PY_START", flush=True)
    args = make_parser().parse_args()

    assert args.dt <= 0.005, f"dt={args.dt} exceeds max 0.005 (Milstein stability)"

    psi0, _ = build_initial_state(args.state, args.theta, args.phi_state)
    rho0    = ket2dm(psi0)
    gamma   = float(args.gamma)
    t1      = 1.0 / gamma
    times   = np.arange(0.0, args.t_end * t1, args.dt * t1)
    x       = times / t1

    e_ops   = [op for _, op in DEFAULT_EOPS]
    columns = [name for name, _ in DEFAULT_EOPS]

    eta_1 = float(args.eta1)
    eta_2 = float(args.eta2)

    # Output directory encodes eta and phi1
    phi1_str = angle_to_path(float(args.phi1))
    eta_str  = f"eta1={eta_1:g}_eta2={eta_2:g}"
    out_dir  = Path(args.out_root) / f"ntraj={args.ntraj}__{eta_str}__phi1={phi1_str}"
    out_dir.mkdir(parents=True, exist_ok=True)

    plt.rcParams["text.usetex"] = bool(args.usetex)
    plt.rcParams.update({
        "mathtext.fontset":   "cm",
        "font.family":        "serif",
        "font.size":          14,
        "axes.unicode_minus": False,
    })

    print(f"eta_1={eta_1}, eta_2={eta_2}, phi_1={args.phi1:.4f}")
    print(f"phi_2 values: {[f'{v/np.pi:.3f}π' for v in PHI2_VALUES]}")
    print(f"ntraj={args.ntraj}, T_end={args.t_end}, dt={args.dt}")
    print(f"out_dir: {out_dir}\n")

    # ── Run simulation for each phi_2 ─────────────────────────────────────────
    results = {}    # phi2 -> pd.DataFrame with columns
    for phi2 in PHI2_VALUES:
        phi2_tex = angle_to_tex(phi2)
        print(f"  Simulating phi_2 = {phi2_tex} ...", flush=True)
        avg_expect = run_single_phi2(
            rho0=rho0, times=times, gamma=gamma,
            phi1=float(args.phi1), phi2=phi2,
            eta_1=eta_1, eta_2=eta_2,
            e_ops=e_ops, ntraj=args.ntraj, num_cpus=args.num_cpus,
        )
        df = pd.DataFrame(np.transpose(avg_expect), columns=columns)

        # Post-process Xi2_WIN from Xi2_KU and Norm_J
        ku_vals   = df["Xi2_KU"].to_numpy(dtype=float)
        norm_vals = df["Norm_J"].to_numpy(dtype=float)
        xi_win    = ku_vals.copy()
        mask      = norm_vals > 1e-12
        xi_win[mask] = ku_vals[mask] * ((2.0 / 2.0) / norm_vals[mask]) ** 2
        df["Xi2_WIN"] = xi_win

        results[phi2] = df
        print(f"    done.", flush=True)

    all_columns = columns + ["Xi2_WIN"]

    # ── Plots — one figure per observable, 4 curves for phi_2 ─────────────────
    for col in all_columns:
        plt.figure(figsize=(12, 8))

        for phi2, col_color in zip(PHI2_VALUES, COLORS_PHI2):
            phi2_tex = angle_to_tex(phi2)
            y = results[phi2][col].to_numpy()
            plt.plot(x, y, color=col_color, linewidth=2.0,
                     label=rf"$\phi_2 = {phi2_tex}$")

        plt.xlim(0.0, float(args.t_end))

        if col == "Xi2_WIN":
            plt.axhline(1.0, linestyle="--", linewidth=2, color="black", alpha=0.7)
            plt.ylim(0.0, 5.0)
        if col in ("Xi2_KU", "Conc"):
            plt.axhline(1.0, linestyle="--", linewidth=2, color="black", alpha=0.7)
            plt.ylim(0.0, 1.05)

        phi1_tex = angle_to_tex(float(args.phi1))
        plt.xlabel(r"$t/T_1$")
        plt.ylabel(LABELS[col])
        plt.title(
            r"Homodyne $J_z$ (N=2): "
            + LABELS[col]
            + rf"$,\ \phi_1={phi1_tex}$"
            + rf"$,\ \eta_1={eta_1:g},\ \eta_2={eta_2:g}$"
        )
        plt.grid(True, linestyle=":", alpha=0.6)
        plt.legend(loc="upper right", fontsize=12)
        plt.savefig(out_dir / f"Phi2scan_{col}.pdf", bbox_inches="tight")
        plt.close()
        print(f"  Saved: Phi2scan_{col}.pdf")

    # ── Metadata ───────────────────────────────────────────────────────────────
    meta = {
        "script":       "main_SLURM_phi2scan.py",
        "measurement":  "homodyne_dephasing",
        "N":            2,
        "state":        args.state,
        "gamma":        gamma,
        "phi1":         float(args.phi1),
        "phi2_values":  PHI2_VALUES,
        "eta_1":        eta_1,
        "eta_2":        eta_2,
        "ntraj":        args.ntraj,
        "dt_T1":        float(args.dt),
        "t_end_T1":     float(args.t_end),
        "columns":      all_columns,
        "integrator":   "milstein",
    }
    (out_dir / "run_config.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8")

    print(f"\nOUT_DIR={out_dir}")
    print(f"PY_TOTAL_SECONDS={time.perf_counter() - t0:.3f}", flush=True)


if __name__ == "__main__":
    main()
