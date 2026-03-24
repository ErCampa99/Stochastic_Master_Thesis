"""
QFI analysis for N=2 homodyne dephasing — cluster-ready script.

Reproduces the QFI_pure_unitary_N2_Jy notebook step by step and saves:
  - ideal_spaghetti.pdf          per-trajectory plot (eta=1)
  - ideal_ensemble.pdf           ensemble-average plot (eta=1)
  - ideal_final_histograms.pdf   final-time distributions (eta=1)
  - phase_scan.pdf               QFI/xi^2 vs time for several phi1 values
  - efficiency_scan.pdf          QFI/xi^2/purity vs time for several eta values
  - efficiency_meanstate.pdf     trajectory mean vs mean state per eta
  - scatter_qfi.pdf              F_Q^max vs xi^2_W and vs purity
  - run_config.json              full parameter log

SLURM array usage (efficiency scan):
  sbatch --array=0-3 submit_qfi.sh
  # Inside the script pass --use-slurm-array; each task runs one eta value.
"""

import ast
import argparse
import json
import os
import time
from fractions import Fraction
from pathlib import Path

# ── Thread control (must be before any numpy/scipy import) ────────────────────
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from qutip import basis, expect, ket2dm, qeye, sigmax, sigmay, sigmaz, smesolve, tensor, variance

# =============================================================================
# ── Utility: angle parsing (identical to main_SLURM.py) ─────────────────────
# =============================================================================

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
        raise argparse.ArgumentTypeError(
            f"Invalid angle expression '{raw}'. Use float or expressions like pi/2, np.pi/2."
        ) from exc

    def _eval(expr):
        if isinstance(expr, ast.Expression):
            return _eval(expr.body)
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
        if isinstance(expr, ast.BinOp) and isinstance(
                expr.op, (ast.Add, ast.Sub, ast.Mult, ast.Div)):
            L, R = _eval(expr.left), _eval(expr.right)
            if isinstance(expr.op, ast.Add):  return L + R
            if isinstance(expr.op, ast.Sub):  return L - R
            if isinstance(expr.op, ast.Mult): return L * R
            return L / R
        raise argparse.ArgumentTypeError(
            f"Unsupported expression '{raw}'. Allowed: numbers, pi, np.pi, +-*/ and parentheses."
        )

    return float(_eval(node))


def parse_angle_list(raw: str) -> list[float]:
    """Parse a comma-separated list of angle expressions, e.g. '0,pi/4,pi/2,pi'."""
    return [parse_angle_arg(tok.strip()) for tok in raw.split(",") if tok.strip()]


def angle_to_path(phi, max_den=12, tol=1e-10) -> str:
    twopi = 2 * np.pi
    x = (phi % twopi + twopi) % twopi
    if abs(x) < tol:
        return "0"
    q = Fraction(x / np.pi).limit_denominator(max_den)
    n, d = q.numerator, q.denominator
    if n == 0: return "0"
    if d == 1: return "pi" if n == 1 else f"{n}pi"
    return ("pi_" + str(d)) if n == 1 else f"{n}pi_{d}"


def angle_to_tex(phi, max_den=12, tol=1e-10) -> str:
    twopi = 2 * np.pi
    x = (phi % twopi + twopi) % twopi
    if abs(x) < tol:
        return "0"
    q = Fraction(x / np.pi).limit_denominator(max_den)
    n, d = q.numerator, q.denominator
    if n == 0: return "0"
    if d == 1: return r"\pi" if n == 1 else rf"{n}\pi"
    num = "" if abs(n) == 1 else str(abs(n))
    s = r"\pi / %d" % d if abs(n) == 1 else r"%s\pi / %d" % (num, d)
    return s if n > 0 else "-" + s


# =============================================================================
# ── Operators and initial state ───────────────────────────────────────────────
# =============================================================================

N_SPIN = 2

gnd  = basis(2, 0)
exc  = basis(2, 1)
I_2  = qeye(2)

plus_state = (gnd + exc).unit()
PlusPlus   = tensor(plus_state, plus_state)

J_x  = 0.5 * (tensor(sigmax(), I_2) + tensor(I_2, sigmax()))
J_y  = 0.5 * (tensor(sigmay(), I_2) + tensor(I_2, sigmay()))
J_z  = 0.5 * (tensor(sigmaz(), I_2) + tensor(I_2, sigmaz()))
Pi_z = tensor(sigmaz(), sigmaz())

SPIN_OPS    = [J_x, J_y, J_z]
H_free      = 0 * J_z

Cp = np.sqrt(0.5) * (tensor(sigmaz(), I_2) + tensor(I_2, sigmaz()))
Cm = np.sqrt(0.5) * (tensor(sigmaz(), I_2) - tensor(I_2, sigmaz()))

OBSERVABLE_COLUMNS = [
    "Norm_J", "Xi2_KU", "Xi2_WIN", "QFI_max", "chi_square", "Var_Jz", "Parity_Pi_z",
]
AXIS_COLUMNS = ["axis_x", "axis_y", "axis_z"]


def collapsing_operators(gamma, phi_1, phi_2, eta_1, eta_2):
    return [
        np.sqrt(gamma * eta_1) * np.exp(1j * phi_1) * Cp,
        np.sqrt(gamma * eta_2) * np.exp(1j * phi_2) * Cm,
    ]


# =============================================================================
# ── Spin-squeezing helpers ────────────────────────────────────────────────────
# =============================================================================

def eval_spin_components(rho):
    return expect(J_x, rho), expect(J_y, rho), expect(J_z, rho)


def norm_J(rho):
    return float(np.linalg.norm(eval_spin_components(rho)))


def xi_KU_solver(rho, tol=1e-12):
    Jmean = np.array([float(np.real_if_close(v)) for v in eval_spin_components(rho)])
    m = np.linalg.norm(Jmean)
    C = np.zeros((3, 3))
    for i, Ji in enumerate(SPIN_OPS):
        for j, Jj in enumerate(SPIN_OPS):
            sym = 0.5 * expect(Ji * Jj + Jj * Ji, rho)
            C[i, j] = float(np.real_if_close(sym - Jmean[i] * Jmean[j]))
    if m > tol:
        u = Jmean / m
        a = np.array([1.0, 0.0, 0.0]) if abs(u[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        e1 = a - u * np.dot(u, a); e1 /= np.linalg.norm(e1)
        e2 = np.cross(u, e1)
        C2 = np.array([[e1 @ C @ e1, e1 @ C @ e2],
                       [e2 @ C @ e1, e2 @ C @ e2]])
        lam_min = np.linalg.eigvalsh(C2)[0]
    else:
        lam_min = np.linalg.eigvalsh(C)[0]
    return float((4.0 / N_SPIN) * max(float(np.real_if_close(lam_min)), 0.0))


def xi_WIN_solver(rho, tol=1e-12):
    n = norm_J(rho)
    xi_ku = xi_KU_solver(rho)
    return xi_ku * ((N_SPIN / 2.0) / n) ** 2 if n > tol else xi_ku


# =============================================================================
# ── QFI helpers ───────────────────────────────────────────────────────────────
# =============================================================================

def normalize_axis(axis):
    a = np.asarray(axis, dtype=float)
    return a / np.linalg.norm(a)


def state_as_dm(state):
    return ket2dm(state) if state.isket else state


def mixed_qfi_matrix(rho, tol=1e-12):
    rho = state_as_dm(rho)
    evals, ekets = rho.eigenstates()
    F = np.zeros((3, 3))
    for i, p_i in enumerate(evals):
        p_i = float(np.real_if_close(p_i))
        for j, p_j in enumerate(evals):
            p_j = float(np.real_if_close(p_j))
            denom = p_i + p_j
            if denom <= tol:
                continue
            coeff = 2.0 * (p_i - p_j) ** 2 / denom
            for a, Ja in enumerate(SPIN_OPS):
                for b, Jb in enumerate(SPIN_OPS):
                    F[a, b] += coeff * np.real(
                        complex(ekets[i].dag() * Ja * ekets[j]) *
                        complex(ekets[j].dag() * Jb * ekets[i])
                    )
    return 0.5 * (F + F.T)


def optimal_qfi(rho):
    """Returns (F_Q_max, optimal_axis, F_matrix)."""
    F = mixed_qfi_matrix(state_as_dm(rho))
    eigvals, eigvecs = np.linalg.eigh(F)
    idx = int(np.argmax(eigvals))
    axis = eigvecs[:, idx]
    if axis[int(np.argmax(np.abs(axis)))] < 0:
        axis = -axis
    return float(eigvals[idx]), normalize_axis(axis), F


def purity(rho):
    dm = state_as_dm(rho)
    return float(np.real_if_close((dm * dm).tr()))


def chi_square_from_qfi(qfi_max, tol=1e-12):
    return np.inf if qfi_max <= tol else float(N_SPIN / qfi_max)


# =============================================================================
# ── DataFrame builders ────────────────────────────────────────────────────────
# =============================================================================

def observables_row(traj_idx, time_idx, t, state, label, eta_1, eta_2):
    rho = state_as_dm(state)
    qfi_max, axis_opt, _ = optimal_qfi(rho)
    qfi_max = max(float(np.real_if_close(qfi_max)), 0.0)
    return {
        "monitoring": label,
        "traj":        int(traj_idx),
        "time_idx":    int(time_idx),
        "time":        float(t),
        "eta_1":       float(eta_1),
        "eta_2":       float(eta_2),
        "Norm_J":      norm_J(rho),
        "Xi2_KU":      xi_KU_solver(rho),
        "Xi2_WIN":     xi_WIN_solver(rho),
        "QFI_max":     qfi_max,
        "chi_square":  chi_square_from_qfi(qfi_max),
        "Var_Jz":      float(np.real_if_close(variance(J_z, rho))),
        "Parity_Pi_z": float(np.real_if_close(expect(Pi_z, rho))),
        "Purity":      purity(rho),
        "axis_x":      float(axis_opt[0]),
        "axis_y":      float(axis_opt[1]),
        "axis_z":      float(axis_opt[2]),
    }


def build_traj_df(sol, times, label, eta_1, eta_2):
    rows = []
    for r, traj_states in enumerate(sol.runs_states):
        for t_idx, (t, state) in enumerate(zip(times, traj_states)):
            rows.append(observables_row(r, t_idx, t, state, label, eta_1, eta_2))
    cols = (["monitoring", "traj", "time_idx", "time", "eta_1", "eta_2"]
            + OBSERVABLE_COLUMNS + ["Purity"] + AXIS_COLUMNS)
    return pd.DataFrame(rows, columns=cols)


def build_ensemble_df(traj_df):
    agg_cols = OBSERVABLE_COLUMNS + ["Purity"]
    summary = traj_df.groupby("time", as_index=False).agg(
        monitoring=("monitoring", "first"),
        eta_1=("eta_1", "first"),
        eta_2=("eta_2", "first"),
        **{c: (c, "mean") for c in agg_cols},
        **{c + "_std": (c, "std") for c in agg_cols},
        **{c: (c, "mean") for c in AXIS_COLUMNS},
    )
    summary["state_kind"] = "trajectory_mean"
    return summary


def build_mean_state_df(sol, times, label, eta_1, eta_2):
    ntraj = len(sol.runs_states)
    rows = []
    for t_idx, t in enumerate(times):
        rho_mean = sum(sol.runs_states[r][t_idx] for r in range(ntraj)) / ntraj
        row = observables_row(-1, t_idx, t, rho_mean, label, eta_1, eta_2)
        row["state_kind"] = "mean_state"
        rows.append(row)
    return pd.DataFrame(rows)


# =============================================================================
# ── argparse ─────────────────────────────────────────────────────────────────
# =============================================================================

def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="QFI analysis for N=2 homodyne dephasing (cluster-ready)"
    )

    # ── Physical parameters ───────────────────────────────────────────────────
    parser.add_argument("--gamma",  type=float, default=1.0,
                        help="Decay rate (default: 1.0)")
    parser.add_argument("--phi1",   type=parse_angle_arg, default=0.0,
                        help="Base measurement phase phi_1 (e.g. 0, pi/2)")
    parser.add_argument("--phi2",   type=parse_angle_arg, default=0.0,
                        help="Base measurement phase phi_2 (e.g. 0, pi/2)")

    # ── Simulation parameters ─────────────────────────────────────────────────
    parser.add_argument("--t-end",       type=float, default=5.0,
                        help="Final time in units of 1/gamma")
    parser.add_argument("--dt",          type=float, default=0.001,
                        help="Time step in units of 1/gamma")
    parser.add_argument("--ntraj-ideal", type=int,   default=30,
                        help="Trajectories for ideal (eta=1) run")
    parser.add_argument("--ntraj-scan",  type=int,   default=15,
                        help="Trajectories per phi1 value in phase scan")
    parser.add_argument("--ntraj-eta",   type=int,   default=20,
                        help="Trajectories per eta value in efficiency scan")
    parser.add_argument("--num-cpus",    type=int,   default=max(1, os.cpu_count() - 1),
                        help="CPUs passed to smesolve")
    parser.add_argument("--seed",        type=int,   default=12345)

    # ── Phase scan ────────────────────────────────────────────────────────────
    parser.add_argument("--phi1-scan",   type=str,
                        default="0,pi/4,pi/2,3*pi/4,pi",
                        help="Comma-separated phi1 values for phase scan")
    parser.add_argument("--no-phase-scan", action="store_true",
                        help="Skip the phase scan section")

    # ── Efficiency scan (SLURM-array-compatible) ──────────────────────────────
    parser.add_argument("--etas",        type=str, default="1.0,0.8,0.6,0.4",
                        help="Comma-separated eta values for efficiency scan "
                             "(eta_1 = eta_2 for each entry)")
    parser.add_argument("--eta-index",   type=int, default=None,
                        help="Run only one eta by index in --etas "
                             "(for manual SLURM array dispatch)")
    parser.add_argument("--use-slurm-array", action="store_true",
                        help="Read eta index from $SLURM_ARRAY_TASK_ID")
    parser.add_argument("--no-eta-scan", action="store_true",
                        help="Skip the efficiency scan section")

    # ── Output ────────────────────────────────────────────────────────────────
    parser.add_argument("--out-root", type=str,
                        default="./Graphs/QFI_pure_unitary_N2_Jy",
                        help="Root directory for output")
    parser.add_argument("--usetex", action="store_true",
                        help="Enable LaTeX rendering in plots (requires LaTeX)")
    parser.add_argument("--no-scatter", action="store_true",
                        help="Skip scatter plot (requires full efficiency scan)")

    return parser


# =============================================================================
# ── Plot helpers ──────────────────────────────────────────────────────────────
# =============================================================================

OBS_LABELS = {
    "Norm_J":      r"$|\langle\mathbf{J}\rangle|$",
    "Xi2_KU":      r"$\xi^2_{KU}$",
    "Xi2_WIN":     r"$\xi^2_W$",
    "QFI_max":     r"$F_Q^{\max}$",
    "chi_square":  r"$\chi^2$",
    "Var_Jz":      r"$\mathrm{Var}(J_z)$",
    "Parity_Pi_z": r"$\langle\Pi_z\rangle$",
    "Purity":      r"$\mathrm{Tr}(\rho_c^2)$",
}

SQL = float(N_SPIN)
HL  = float(N_SPIN ** 2)


def savefig(pdf_dir: Path, jpg_dir: Path, name: str, **kwargs):
    """Save the current figure as PDF (in pdf_dir) and JPEG (in jpg_dir)."""
    kw = {"bbox_inches": "tight", **kwargs}
    plt.savefig(pdf_dir / f"{name}.pdf",  **kw)
    plt.savefig(jpg_dir / f"{name}.jpeg", dpi=150, **kw)


def add_sql_hl(ax, obs):
    if obs == "QFI_max":
        ax.axhline(SQL, color="steelblue", linestyle=":", linewidth=1.2,
                   label=rf"SQL $= {int(SQL)}$")
        ax.axhline(HL,  color="tomato",    linestyle=":", linewidth=1.2,
                   label=rf"HL $= {int(HL)}$")
    elif obs in {"Xi2_WIN", "chi_square"}:
        ax.axhline(1.0, color="black", linestyle="--", linewidth=1.2,
                   alpha=0.6, label="SQL")


# =============================================================================
# ── Main ─────────────────────────────────────────────────────────────────────
# =============================================================================

def main() -> None:
    t0 = time.perf_counter()
    print("PY_START", flush=True)

    args = make_parser().parse_args()

    # ── SLURM array: resolve eta index ────────────────────────────────────────
    eta_index = args.eta_index
    if eta_index is None and args.use_slurm_array:
        slurm_id = os.getenv("SLURM_ARRAY_TASK_ID", "").strip()
        if not slurm_id:
            raise ValueError("--use-slurm-array set but $SLURM_ARRAY_TASK_ID is missing")
        eta_index = int(slurm_id)

    # ── Parse eta list ────────────────────────────────────────────────────────
    all_etas = [float(x.strip()) for x in args.etas.split(",") if x.strip()]
    if eta_index is not None:
        if not (0 <= eta_index < len(all_etas)):
            raise ValueError(f"eta_index={eta_index} out of range [0, {len(all_etas)-1}]")
        run_etas = [all_etas[eta_index]]
        print(f"[SLURM array] task {eta_index} -> eta={run_etas[0]:.2f}", flush=True)
    else:
        run_etas = all_etas

    gamma  = float(args.gamma)
    phi1   = float(args.phi1)
    phi2   = float(args.phi2)
    times  = np.arange(0.0, args.t_end / gamma + 0.5 * args.dt, args.dt / gamma)

    # ── Output directory ──────────────────────────────────────────────────────
    phi1_dir = angle_to_path(phi1)
    phi2_dir = angle_to_path(phi2)
    run_tag  = f"ntraj={args.ntraj_ideal}__phi_1={phi1_dir}__phi_2={phi2_dir}"
    if eta_index is not None:
        run_tag += f"__eta_idx={eta_index}"
    out_dir = Path(args.out_root) / run_tag
    out_dir.mkdir(parents=True, exist_ok=True)
    pdf_dir = out_dir / "pdf"
    jpg_dir = out_dir / "jpg"
    pdf_dir.mkdir(exist_ok=True)
    jpg_dir.mkdir(exist_ok=True)
    print(f"Output: {out_dir}", flush=True)

    # ── Matplotlib style (identical to main_SLURM.py) ─────────────────────────
    plt.rcParams["text.usetex"] = bool(args.usetex)
    plt.rcParams.update({
        "mathtext.fontset": "cm",
        "font.family":      "serif",
        "font.size":        14,
        "axes.unicode_minus": False,
    })

    phi1_tex = angle_to_tex(phi1)
    phi2_tex = angle_to_tex(phi2)

    t1 = 1.0 / gamma                          # T_1 in same units as times
    x_plot = times * gamma                    # dimensionless time t/T_1

    rho0 = ket2dm(PlusPlus)

    def sme_opts(ntraj, num_cpus=1):
        return {"keep_runs_results": True, "num_cpus": num_cpus,
                "map": "serial" if num_cpus == 1 else "parallel",
                "store_measurement": False}

    # =========================================================================
    # Section 1 — Ideal trajectories (eta_1 = eta_2 = 1)
    # =========================================================================
    print("\n[1/4] Ideal trajectories ...", flush=True)

    sol_ideal = smesolve(
        H_free, rho0, times,
        c_ops  = collapsing_operators(gamma, phi1, phi2, 0.0, 0.0),
        sc_ops = collapsing_operators(gamma, phi1, phi2, 1.0, 1.0),
        heterodyne=False,
        ntraj=args.ntraj_ideal,
        options=sme_opts(args.ntraj_ideal, args.num_cpus),
    )

    ideal_traj_df = build_traj_df(sol_ideal, times,
                                   label="ideal", eta_1=1.0, eta_2=1.0)
    ideal_ens_df  = build_ensemble_df(ideal_traj_df)

    # ── Spaghetti plots (one figure per observable) ───────────────────────────
    spaghetti_obs = ["QFI_max", "Xi2_WIN", "Norm_J", "Parity_Pi_z"]
    cmap = plt.get_cmap("tab20", args.ntraj_ideal)

    for obs in spaghetti_obs:
        fig, ax = plt.subplots(figsize=(10, 6))
        for r in range(args.ntraj_ideal):
            sub = ideal_traj_df[ideal_traj_df.traj == r]
            ax.plot(sub["time"] * gamma, sub[obs], color=cmap(r), alpha=0.35, linewidth=0.8)
        ax.plot(ideal_ens_df["time"] * gamma, ideal_ens_df[obs],
                color="black", linewidth=2.2, label="ensemble mean")
        add_sql_hl(ax, obs)
        ax.set_xlabel(r"$t / T_1$")
        ax.set_xlim(0.0, args.t_end)
        ax.set_ylabel(OBS_LABELS[obs])
        ax.set_title(
            rf"Ideal trajectories — {OBS_LABELS[obs]}"
            rf"  ($\eta=1$, ntraj$={args.ntraj_ideal}$,"
            rf" $\phi_1={phi1_tex}$, $\phi_2={phi2_tex}$)",
            fontsize=12
        )
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend(loc="upper left", bbox_to_anchor=(0, 0.95), fontsize=12)
        plt.tight_layout()
        savefig(pdf_dir, jpg_dir, f"ideal_spaghetti_{obs}")
        plt.close()

    # ── Ensemble-average plots (one figure per observable) ────────────────────
    ens_obs = ["Norm_J", "Parity_Pi_z", "Xi2_KU", "Xi2_WIN",
               "QFI_max", "chi_square", "Var_Jz"]

    for obs in ens_obs:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(ideal_ens_df["time"] * gamma, ideal_ens_df[obs],
                linewidth=2, color="steelblue", label=OBS_LABELS[obs])
        add_sql_hl(ax, obs)
        ax.set_xlabel(r"$t / T_1$")
        ax.set_xlim(0.0, args.t_end)
        ax.set_ylabel(OBS_LABELS[obs])
        ax.set_title(
            rf"Ideal monitoring — {OBS_LABELS[obs]}"
            rf"  (ntraj$={args.ntraj_ideal}$, $\eta=1$)",
            fontsize=12
        )
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend(loc="upper left", bbox_to_anchor=(0, 0.95), fontsize=12)
        plt.tight_layout()
        savefig(pdf_dir, jpg_dir, f"ideal_ensemble_{obs}")
        plt.close()

    # ── Final-time histograms (one figure per observable) ─────────────────────
    t_final  = times[-1]
    final_df = ideal_traj_df[np.isclose(ideal_traj_df["time"], t_final)]

    hist_obs = ["QFI_max", "Xi2_WIN", "chi_square", "Parity_Pi_z"]
    for obs in hist_obs:
        fig, ax = plt.subplots(figsize=(7, 5))
        vals = final_df[obs].values
        ax.hist(vals, bins=12, color="steelblue", edgecolor="white", alpha=0.85)
        ax.axvline(vals.mean(), color="black", linewidth=2, linestyle="--",
                   label=rf"mean $= {vals.mean():.2f}$")
        ax.set_xlabel(rf"value at $t / T_1 = {t_final * gamma:.1f}$")
        ax.set_ylabel("count")
        ax.set_title(
            rf"Final-time distribution — {OBS_LABELS[obs]}"
            rf"  (ntraj$={args.ntraj_ideal}$)",
            fontsize=12
        )
        ax.legend(loc="upper left", bbox_to_anchor=(0, 0.95), fontsize=12)
        ax.grid(True, linestyle=":", alpha=0.6)
        plt.tight_layout()
        savefig(pdf_dir, jpg_dir, f"ideal_hist_{obs}")
        plt.close()

    print(f"  [1/4] done ({time.perf_counter()-t0:.1f}s)", flush=True)

    # =========================================================================
    # Section 2 — Phase scan
    # =========================================================================
    scan_results = {}   # phi1_tex_label -> ensemble_df

    if not args.no_phase_scan:
        print("\n[2/4] Phase scan ...", flush=True)

        phi1_values = parse_angle_list(args.phi1_scan)
        phi1_labels = [rf"$\phi_1 = {angle_to_tex(v)}$" for v in phi1_values]

        for phi_val, phi_label in zip(phi1_values, phi1_labels):
            sol_scan = smesolve(
                H_free, rho0, times,
                c_ops  = collapsing_operators(gamma, phi_val, phi2, 0.0, 0.0),
                sc_ops = collapsing_operators(gamma, phi_val, phi2, 1.0, 1.0),
                heterodyne=False,
                ntraj=args.ntraj_scan,
                options=sme_opts(args.ntraj_scan, args.num_cpus),
            )
            tdf = build_traj_df(sol_scan, times,
                                label=phi_label, eta_1=1.0, eta_2=1.0)
            scan_results[phi_label] = build_ensemble_df(tdf)
            print(f"    {phi_label}  done", flush=True)

        colors_scan   = plt.cm.plasma(np.linspace(0.1, 0.9, len(phi1_labels)))
        scan_plot_obs = ["QFI_max", "Xi2_WIN", "chi_square"]

        for obs in scan_plot_obs:
            fig, ax = plt.subplots(figsize=(10, 6))
            for (lbl, edf), col in zip(scan_results.items(), colors_scan):
                ax.plot(edf["time"] * gamma, edf[obs], linewidth=2, color=col, label=lbl)
            add_sql_hl(ax, obs)
            ax.set_xlabel(r"$t / T_1$")
            ax.set_xlim(0.0, args.t_end)
            ax.set_ylabel(OBS_LABELS[obs])
            ax.set_title(
                rf"Phase scan — {OBS_LABELS[obs]}"
                rf"  ($\phi_2={phi2_tex}$, $\eta=1$, ntraj$={args.ntraj_scan}$)",
                fontsize=12
            )
            ax.grid(True, linestyle=":", alpha=0.6)
            ax.legend(loc="upper left", bbox_to_anchor=(0, 0.95), fontsize=12)
            plt.tight_layout()
            savefig(pdf_dir, jpg_dir, f"phase_scan_{obs}")
            plt.close()

        # Summary table
        rows = []
        for lbl, edf in scan_results.items():
            last = edf.iloc[-1]
            rows.append({"phi1": lbl, "QFI_max": last["QFI_max"],
                         "Xi2_WIN": last["Xi2_WIN"], "chi^2": last["chi_square"]})

        print(f"  [2/4] done ({time.perf_counter()-t0:.1f}s)", flush=True)
    else:
        print("[2/4] Phase scan skipped.", flush=True)

    # =========================================================================
    # Section 3 — Efficiency scan
    # =========================================================================
    eta_traj_dfs     = {}
    eta_ensemble_dfs = {}
    eta_meanstate_dfs= {}

    if not args.no_eta_scan:
        print(f"\n[3/4] Efficiency scan  eta={run_etas} ...", flush=True)

        for eta in run_etas:
            sol_eta = smesolve(
                H_free, rho0, times,
                c_ops  = collapsing_operators(gamma, phi1, phi2, 1.0-eta, 1.0-eta),
                sc_ops = collapsing_operators(gamma, phi1, phi2, eta,     eta),
                heterodyne=False,
                ntraj=args.ntraj_eta,
                options=sme_opts(args.ntraj_eta, args.num_cpus),
            )
            tdf  = build_traj_df(sol_eta, times,
                                  label=f"eta={eta:.2f}", eta_1=eta, eta_2=eta)
            edf  = build_ensemble_df(tdf)
            mdf  = build_mean_state_df(sol_eta, times,
                                        label=f"eta={eta:.2f}", eta_1=eta, eta_2=eta)

            eta_traj_dfs[eta]     = tdf
            eta_ensemble_dfs[eta] = edf
            eta_meanstate_dfs[eta]= mdf
            print(f"    eta={eta:.2f}  done", flush=True)

        colors_eta   = plt.cm.viridis(np.linspace(0.15, 0.85, len(run_etas)))
        eta_plot_obs = ["QFI_max", "Xi2_WIN", "Purity"]

        # ── Trajectory-average plots (one figure per observable) ──────────────
        for obs in eta_plot_obs:
            fig, ax = plt.subplots(figsize=(10, 6))
            for eta, col in zip(run_etas, colors_eta):
                edf = eta_ensemble_dfs[eta]
                ax.plot(edf["time"] * gamma, edf[obs], linewidth=2, color=col,
                        label=rf"$\eta = {eta:.1f}$")
            add_sql_hl(ax, obs)
            ax.set_xlabel(r"$t / T_1$")
            ax.set_xlim(0.0, args.t_end)
            ax.set_ylabel(OBS_LABELS[obs])
            ax.set_title(
                rf"Efficiency scan — {OBS_LABELS[obs]}"
                rf"  (ntraj$={args.ntraj_eta}$, $\phi_1={phi1_tex}$, $\phi_2={phi2_tex}$)",
                fontsize=12
            )
            ax.grid(True, linestyle=":", alpha=0.6)
            ax.legend(loc="upper left", bbox_to_anchor=(0, 0.95), fontsize=12)
            plt.tight_layout()
            savefig(pdf_dir, jpg_dir, f"efficiency_scan_{obs}")
            plt.close()

        # ── Trajectory mean vs mean state (one figure per eta) ────────────────
        for eta in run_etas:
            eta_str = f"{eta:.2f}".replace(".", "p")
            fig, ax = plt.subplots(figsize=(10, 6))
            edf = eta_ensemble_dfs[eta]
            mdf = eta_meanstate_dfs[eta]
            ax.plot(edf["time"] * gamma, edf["QFI_max"], linewidth=2,
                    label="traj mean", color="steelblue")
            ax.plot(mdf["time"] * gamma, mdf["QFI_max"], linewidth=2, linestyle="--",
                    label=r"mean state $\bar{\rho}$", color="tomato")
            ax.axhline(SQL, color="grey",  linestyle=":", linewidth=1,
                       label=rf"SQL $= {int(SQL)}$")
            ax.axhline(HL,  color="black", linestyle=":", linewidth=1,
                       label=rf"HL $= {int(HL)}$")
            ax.set_xlabel(r"$t / T_1$")
            ax.set_xlim(0.0, args.t_end)
            ax.set_ylabel(OBS_LABELS["QFI_max"])
            ax.set_title(
                rf"$F_Q^{{\max}}$: traj mean vs mean state — $\eta = {eta:.1f}$",
                fontsize=12
            )
            ax.grid(True, linestyle=":", alpha=0.6)
            ax.legend(loc="upper left", bbox_to_anchor=(0, 0.95), fontsize=12)
            plt.tight_layout()
            savefig(pdf_dir, jpg_dir, f"efficiency_meanstate_eta{eta_str}")
            plt.close()

        print(f"  [3/4] done ({time.perf_counter()-t0:.1f}s)", flush=True)
    else:
        print("[3/4] Efficiency scan skipped.", flush=True)

    # =========================================================================
    # Section 4 — Scatter: F_Q^max vs xi^2_W and vs purity
    # =========================================================================
    if not args.no_scatter and eta_traj_dfs:
        print("\n[4/4] Scatter plot ...", flush=True)

        all_dfs = []
        for eta, tdf in eta_traj_dfs.items():
            sub = tdf[["time", "QFI_max", "Xi2_WIN", "Purity"]].copy()
            sub["eta"] = eta
            all_dfs.append(sub)
        # Always include the ideal run
        sub_ideal = ideal_traj_df[["time", "QFI_max", "Xi2_WIN", "Purity"]].copy()
        sub_ideal["eta"] = 1.0
        all_dfs.append(sub_ideal)
        scatter_df = pd.concat(all_dfs, ignore_index=True)

        plot_etas = run_etas if run_etas else all_etas
        colors_sc = plt.cm.viridis(np.linspace(0.1, 0.9, len(plot_etas)))

        # ── F_Q^max vs xi^2_W ─────────────────────────────────────────────────
        fig, ax = plt.subplots(figsize=(8, 6))
        for eta, col in zip(plot_etas, colors_sc):
            sub = scatter_df[scatter_df["eta"] == eta]
            ax.scatter(sub["Xi2_WIN"], sub["QFI_max"],
                       s=3, alpha=0.4, color=col, label=rf"$\eta={eta:.1f}$")
        ax.axhline(SQL, color="steelblue", linestyle="--", linewidth=1.2,
                   label=rf"SQL $= {int(SQL)}$")
        ax.axhline(HL,  color="tomato",    linestyle="--", linewidth=1.2,
                   label=rf"HL $= {int(HL)}$")
        ax.axvline(1.0, color="black", linestyle=":", linewidth=1.2, alpha=0.6)
        ax.set_xlabel(r"$\xi^2_W$")
        ax.set_ylabel(r"$F_Q^{\max}$")
        ax.set_title(r"$F_Q^{\max}$ vs $\xi^2_W$ — all trajectories", fontsize=12)
        ax.legend(loc="upper left", bbox_to_anchor=(0, 0.95), fontsize=12, markerscale=4)
        ax.grid(True, linestyle=":", alpha=0.6)
        plt.tight_layout()
        savefig(pdf_dir, jpg_dir, "scatter_FQ_vs_xi2W")
        plt.close()

        # ── F_Q^max vs purity ─────────────────────────────────────────────────
        fig, ax = plt.subplots(figsize=(8, 6))
        for eta, col in zip(plot_etas, colors_sc):
            sub = scatter_df[scatter_df["eta"] == eta]
            ax.scatter(sub["Purity"], sub["QFI_max"],
                       s=3, alpha=0.4, color=col, label=rf"$\eta={eta:.1f}$")
        ax.axhline(SQL, color="steelblue", linestyle="--", linewidth=1.2,
                   label=rf"SQL $= {int(SQL)}$")
        ax.axhline(HL,  color="tomato",    linestyle="--", linewidth=1.2,
                   label=rf"HL $= {int(HL)}$")
        ax.set_xlabel(r"$\mathrm{Tr}(\rho_c^2)$")
        ax.set_ylabel(r"$F_Q^{\max}$")
        ax.set_title(r"$F_Q^{\max}$ vs purity — all trajectories", fontsize=12)
        ax.legend(loc="upper left", bbox_to_anchor=(0, 0.95), fontsize=12, markerscale=4)
        ax.grid(True, linestyle=":", alpha=0.6)
        plt.tight_layout()
        savefig(pdf_dir, jpg_dir, "scatter_FQ_vs_purity")
        plt.close()
        print(f"  [4/4] done ({time.perf_counter()-t0:.1f}s)", flush=True)
    else:
        print("[4/4] Scatter skipped (no eta scan data or --no-scatter).", flush=True)

    # =========================================================================
    # Save run config
    # =========================================================================
    elapsed = time.perf_counter() - t0
    config = {
        "script":          "QFI_pure_unitary_N2_Jy_SLURM.py",
        "N":               N_SPIN,
        "gamma":           gamma,
        "phi1":            phi1,
        "phi2":            phi2,
        "t_end":           args.t_end,
        "dt":              args.dt,
        "ntraj_ideal":     args.ntraj_ideal,
        "ntraj_scan":      args.ntraj_scan,
        "ntraj_eta":       args.ntraj_eta,
        "num_cpus":        args.num_cpus,
        "seed":            args.seed,
        "phi1_scan":       args.phi1_scan,
        "etas":            args.etas,
        "eta_index":       eta_index,
        "use_slurm_array": args.use_slurm_array,
        "no_phase_scan":   args.no_phase_scan,
        "no_eta_scan":     args.no_eta_scan,
        "no_scatter":      args.no_scatter,
        "out_dir":         str(out_dir),
        "elapsed_s":       round(elapsed, 2),
    }
    with open(out_dir / "run_config.json", "w") as f:
        json.dump(config, f, indent=2)

    print(f"\nPY_END  elapsed={elapsed:.1f}s  out={out_dir}", flush=True)


if __name__ == "__main__":
    main()
