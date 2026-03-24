"""
Cluster-ready script version of QFI_pure_unitary_N2_Jy.ipynb.

It reproduces the notebook step by step for N=2 and saves:
- sanity-check outputs from the pure and mixed QFI sections
- trajectory and ensemble DataFrames as CSV files
- the Step 3 and Step 5 figures as both PNG and PDF
- a JSON run summary with the main parameters and output paths
"""

import argparse
import ast
import json
import os
import sys
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
from qutip import basis, expect, ket2dm, qeye, sigmax, sigmay, sigmaz, smesolve, tensor, variance


SCRIPT_DIR = Path(__file__).resolve().parent
for candidate in [SCRIPT_DIR, SCRIPT_DIR.parent, Path.cwd(), Path.cwd() / "Codes"]:
    if (candidate / "main_SLURM.py").exists():
        CODE_DIR = candidate
        break
else:
    raise FileNotFoundError("Could not locate main_SLURM.py from this script.")

if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from main_SLURM import norm_J, xi_KU_solver, xi_WIN_solver


N_SPIN = 2
DT = 0.001
T_END = 5.0

gnd = basis(2, 0)
exc = basis(2, 1)

# Convention
plus_state = (gnd + exc).unit()
# State |++>
psi0 = tensor(plus_state, plus_state)
rho0 = ket2dm(psi0)

# Identity, collective-spin operators and helper lists
I_2 = qeye(2)
J_x = 0.5 * (tensor(sigmax(), I_2) + tensor(I_2, sigmax()))
J_y = 0.5 * (tensor(sigmay(), I_2) + tensor(I_2, sigmay()))
J_z = 0.5 * (tensor(sigmaz(), I_2) + tensor(I_2, sigmaz()))
Pi_z = tensor(sigmaz(), sigmaz())

SPIN_OPS = [J_x, J_y, J_z]
SPIN_LABELS = ["J_x", "J_y", "J_z"]
OBSERVABLE_COLUMNS = [
    "Norm_J",
    "Xi2_KU",
    "Xi2_WIN",
    "QFI_max",
    "chi_square",
    "Var_Jz",
    "Parity_Pi_z",
]
AXIS_COLUMNS = ["axis_x", "axis_y", "axis_z"]

# Homodyne channels
Cp = np.sqrt(0.5) * (tensor(sigmaz(), I_2) + tensor(I_2, sigmaz()))
Cm = np.sqrt(0.5) * (tensor(sigmaz(), I_2) - tensor(I_2, sigmaz()))

# We keep the free Hamiltonian to zero, as in the notebook.
H_free = 0 * J_z


def collapsing_operators(gamma, phi_1, phi_2, eta_1, eta_2):
    return [
        np.sqrt(gamma * eta_1) * np.exp(1j * phi_1) * Cp,
        np.sqrt(gamma * eta_2) * np.exp(1j * phi_2) * Cm,
    ]


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
        if (
            isinstance(expr, ast.Attribute)
            and isinstance(expr.value, ast.Name)
            and expr.value.id in {"np", "numpy"}
            and expr.attr == "pi"
        ):
            return float(np.pi)
        if isinstance(expr, ast.UnaryOp) and isinstance(expr.op, (ast.UAdd, ast.USub)):
            val = _eval(expr.operand)
            return val if isinstance(expr.op, ast.UAdd) else -val
        if isinstance(expr, ast.BinOp) and isinstance(expr.op, (ast.Add, ast.Sub, ast.Mult, ast.Div)):
            left = _eval(expr.left)
            right = _eval(expr.right)
            if isinstance(expr.op, ast.Add):
                return left + right
            if isinstance(expr.op, ast.Sub):
                return left - right
            if isinstance(expr.op, ast.Mult):
                return left * right
            return left / right
        raise argparse.ArgumentTypeError(
            f"Unsupported angle expression '{raw}'. Allowed: numbers, pi, np.pi, + - * / and parentheses."
        )

    return float(_eval(node))


def angle_to_path(phi, max_den=12, tol=1e-10):
    twopi = 2 * np.pi
    x = (phi % twopi + twopi) % twopi
    if abs(x) < tol:
        return "0"
    q = Fraction(x / np.pi).limit_denominator(max_den)
    n, d = q.numerator, q.denominator
    if n == 0:
        return "0"
    if d == 1:
        return "pi" if n == 1 else f"{n}pi"
    return ("pi_" + str(d)) if n == 1 else f"{n}pi_{d}"


def angle_to_tex(phi, max_den=12, tol=1e-10):
    twopi = 2 * np.pi
    x = (phi % twopi + twopi) % twopi
    if abs(x) < tol:
        return "0"
    q = Fraction(x / np.pi).limit_denominator(max_den)
    n, d = q.numerator, q.denominator
    if n == 0:
        return "0"
    if d == 1:
        return r"\pi" if n == 1 else fr"{n}\pi"
    num = "" if abs(n) == 1 else str(abs(n))
    s = r"\pi / %d" % d if abs(n) == 1 else r"%s\pi / %d" % (num, d)
    return s if n > 0 else "-" + s


def configure_matplotlib(usetex: bool) -> None:
    plt.rcParams["text.usetex"] = bool(usetex)
    plt.rcParams.update(
        {
            "mathtext.fontset": "cm",
            "font.family": "serif",
            "font.size": 13,
            "axes.unicode_minus": False,
        }
    )


# QFI helpers

def normalize_axis(axis):
    axis = np.asarray(axis, dtype=float)
    norm = np.linalg.norm(axis)
    if norm <= 0:
        raise ValueError("The axis must be non-zero.")
    return axis / norm


def collective_generator_from_axis(axis):
    n = normalize_axis(axis)
    return n[0] * J_x + n[1] * J_y + n[2] * J_z


def pure_state_qfi(psi, G):
    return float(np.real_if_close(4.0 * variance(G, psi)))


def pure_qfi_along_axis(psi, axis):
    return pure_state_qfi(psi, collective_generator_from_axis(axis))


def pure_qfi_matrix(psi):
    means = np.array([expect(op, psi) for op in SPIN_OPS], dtype=complex)
    cov = np.zeros((3, 3), dtype=float)
    for a, Ja in enumerate(SPIN_OPS):
        for b, Jb in enumerate(SPIN_OPS):
            sym = 0.5 * expect(Ja * Jb + Jb * Ja, psi)
            cov[a, b] = float(np.real_if_close(sym - means[a] * means[b]))
    return 4.0 * cov


def pure_optimal_axis_qfi(psi):
    F = pure_qfi_matrix(psi)
    eigvals, eigvecs = np.linalg.eigh(F)
    idx = int(np.argmax(eigvals))
    axis = eigvecs[:, idx]
    if axis[0] < 0:
        axis = -axis
    return float(eigvals[idx]), normalize_axis(axis), F


def state_as_dm(state):
    return ket2dm(state) if state.isket else state


def mixed_state_qfi(rho, G, tol=1e-12):
    rho = state_as_dm(rho)
    evals, ekets = rho.eigenstates()
    qfi = 0.0
    for i, p_i in enumerate(evals):
        p_i = float(np.real_if_close(p_i))
        for j, p_j in enumerate(evals):
            p_j = float(np.real_if_close(p_j))
            denom = p_i + p_j
            if denom <= tol:
                continue
            G_ij = complex(ekets[i].dag() * G * ekets[j])
            qfi += 2.0 * ((p_i - p_j) ** 2 / denom) * abs(G_ij) ** 2
    return float(np.real_if_close(qfi))


def mixed_qfi_along_axis(rho, axis):
    return mixed_state_qfi(rho, collective_generator_from_axis(axis))


def mixed_qfi_matrix(rho, tol=1e-12):
    rho = state_as_dm(rho)
    evals, ekets = rho.eigenstates()
    F = np.zeros((3, 3), dtype=float)
    for i, p_i in enumerate(evals):
        p_i = float(np.real_if_close(p_i))
        for j, p_j in enumerate(evals):
            p_j = float(np.real_if_close(p_j))
            denom = p_i + p_j
            if denom <= tol:
                continue
            coeff = 2.0 * ((p_i - p_j) ** 2 / denom)
            for a, Ja in enumerate(SPIN_OPS):
                for b, Jb in enumerate(SPIN_OPS):
                    term_a = complex(ekets[i].dag() * Ja * ekets[j])
                    term_b = complex(ekets[j].dag() * Jb * ekets[i])
                    F[a, b] += coeff * np.real(term_a * term_b)
    return 0.5 * (F + F.T)


def mixed_optimal_axis_qfi(rho):
    F = mixed_qfi_matrix(rho)
    eigvals, eigvecs = np.linalg.eigh(F)
    idx = int(np.argmax(eigvals))
    axis = eigvecs[:, idx]
    if axis[0] < 0:
        axis = -axis
    return float(eigvals[idx]), normalize_axis(axis), F


def state_qfi_along_axis(state, axis):
    return mixed_qfi_along_axis(state_as_dm(state), axis)


def state_optimal_axis_qfi(state):
    return mixed_optimal_axis_qfi(state_as_dm(state))


def chi_square_from_qfi(qfi_max, n_spin=N_SPIN, tol=1e-12):
    qfi_max = float(np.real_if_close(qfi_max))
    return np.inf if qfi_max <= tol else float(n_spin / qfi_max)


def conditional_observables_row(traj, time_idx, time_value, state, label, eta_1, eta_2, n_spin=N_SPIN):
    if n_spin != 2:
        raise NotImplementedError("This script currently supports only N=2.")

    rho = state_as_dm(state)
    qfi_max, axis_opt, _ = state_optimal_axis_qfi(rho)

    return {
        "monitoring": label,
        "traj": int(traj),
        "time_idx": int(time_idx),
        "time": float(time_value),
        "eta_1": float(eta_1),
        "eta_2": float(eta_2),
        "Norm_J": float(norm_J(time_value, rho)),
        "Xi2_KU": float(xi_KU_solver(time_value, rho, N=n_spin)),
        "Xi2_WIN": float(xi_WIN_solver(time_value, rho, N=n_spin)),
        "QFI_max": max(float(np.real_if_close(qfi_max)), 0.0),
        "chi_square": chi_square_from_qfi(qfi_max, n_spin=n_spin),
        "Var_Jz": float(np.real_if_close(variance(J_z, rho))),
        "Parity_Pi_z": float(np.real_if_close(expect(Pi_z, rho))),
        "axis_x": float(axis_opt[0]),
        "axis_y": float(axis_opt[1]),
        "axis_z": float(axis_opt[2]),
    }


def iter_conditional_observables(solution, times, label, eta_1, eta_2, n_spin=N_SPIN):
    if n_spin != 2:
        raise NotImplementedError("This script currently supports only N=2.")

    for traj_idx, traj_states in enumerate(solution.runs_states):
        for time_idx, (time_value, state) in enumerate(zip(times, traj_states)):
            yield conditional_observables_row(
                traj=traj_idx,
                time_idx=time_idx,
                time_value=time_value,
                state=state,
                label=label,
                eta_1=eta_1,
                eta_2=eta_2,
                n_spin=n_spin,
            )


def build_conditional_trajectory_df(solution, times, label, eta_1, eta_2, n_spin=N_SPIN):
    columns = ["monitoring", "traj", "time_idx", "time", "eta_1", "eta_2"] + OBSERVABLE_COLUMNS + AXIS_COLUMNS
    return pd.DataFrame.from_records(
        iter_conditional_observables(solution, times, label, eta_1, eta_2, n_spin=n_spin),
        columns=columns,
    )


def build_conditional_ensemble_df(traj_df):
    summary = (
        traj_df.groupby("time", as_index=False)
        .agg(
            monitoring=("monitoring", "first"),
            eta_1=("eta_1", "first"),
            eta_2=("eta_2", "first"),
            Norm_J=("Norm_J", "mean"),
            Xi2_KU=("Xi2_KU", "mean"),
            Xi2_WIN=("Xi2_WIN", "mean"),
            QFI_max=("QFI_max", "mean"),
            chi_square=("chi_square", "mean"),
            Var_Jz=("Var_Jz", "mean"),
            Parity_Pi_z=("Parity_Pi_z", "mean"),
            axis_x=("axis_x", "mean"),
            axis_y=("axis_y", "mean"),
            axis_z=("axis_z", "mean"),
        )
    )
    summary["state_kind"] = "trajectory_mean"
    return summary[["monitoring", "state_kind", "time", "eta_1", "eta_2"] + OBSERVABLE_COLUMNS + AXIS_COLUMNS]


def iter_mean_state_observables(solution, times, label, eta_1, eta_2, n_spin=N_SPIN):
    if n_spin != 2:
        raise NotImplementedError("This script currently supports only N=2.")

    ntraj = len(solution.runs_states)
    for time_idx, time_value in enumerate(times):
        states_t = [traj_states[time_idx] for traj_states in solution.runs_states]
        rho_mean_t = sum(states_t) / ntraj
        row = conditional_observables_row(
            traj=-1,
            time_idx=time_idx,
            time_value=time_value,
            state=rho_mean_t,
            label=label,
            eta_1=eta_1,
            eta_2=eta_2,
            n_spin=n_spin,
        )
        row["state_kind"] = "mean_state"
        yield row


def build_mean_state_observables_df(solution, times, label, eta_1, eta_2, n_spin=N_SPIN):
    columns = ["monitoring", "state_kind", "time_idx", "time", "eta_1", "eta_2"] + OBSERVABLE_COLUMNS + AXIS_COLUMNS
    df = pd.DataFrame.from_records(iter_mean_state_observables(solution, times, label, eta_1, eta_2, n_spin=n_spin))
    return df[columns]


def save_dataframe(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def to_builtin(value):
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return value.resolve().as_posix()
    if isinstance(value, dict):
        return {k: to_builtin(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_builtin(v) for v in value]
    return value


def save_json(data, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_builtin(data), indent=2), encoding="utf-8")


def save_figure(fig, base_path: Path) -> None:
    fig.savefig(base_path.with_suffix(".png"), dpi=200, bbox_inches="tight")
    fig.savefig(base_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def save_preview_text(df: pd.DataFrame, path: Path, nrows: int = 5) -> None:
    path.write_text(df.head(nrows).to_string(index=False), encoding="utf-8")


def build_times() -> np.ndarray:
    return np.arange(0.0, T_END + 0.5 * DT, DT)


def run_conditional_solver(rho0_dm, times, gamma, phi1, phi2, eta_1, eta_2, ntraj, num_cpus):
    solver_cpus = max(1, int(num_cpus))
    options = {
        "keep_runs_results": True,
        "num_cpus": solver_cpus,
        "map": "parallel" if solver_cpus > 1 else "serial",
        "store_measurement": False,
    }
    return smesolve(
        H_free,
        rho0_dm,
        times,
        c_ops=collapsing_operators(gamma, phi1, phi2, 1.0 - eta_1, 1.0 - eta_2),
        sc_ops=collapsing_operators(gamma, phi1, phi2, eta_1, eta_2),
        heterodyne=False,
        ntraj=ntraj,
        options=options,
    )


def step2_pure_sanity():
    pure_qfi_xyz = {
        "J_x": pure_qfi_along_axis(psi0, [1, 0, 0]),
        "J_y": pure_qfi_along_axis(psi0, [0, 1, 0]),
        "J_z": pure_qfi_along_axis(psi0, [0, 0, 1]),
    }
    pure_qfi_opt, pure_axis_opt, pure_F_matrix = pure_optimal_axis_qfi(psi0)
    return {
        "pure_qfi_xyz": pure_qfi_xyz,
        "pure_qfi_opt": pure_qfi_opt,
        "pure_axis_opt": pure_axis_opt,
        "pure_F_matrix": pure_F_matrix,
    }


def plot_step3_ideal(pure_plot_df: pd.DataFrame):
    plot_labels = {
        "Norm_J": r"$\overline{|\langle \mathbf{J} \rangle|}$",
        "Xi2_KU": r"$\overline{\xi^2_{KU}}$",
        "Xi2_WIN": r"$\overline{\xi^2_{WIN}}$",
        "QFI_max": r"$\overline{F_Q^{\max}}$",
        "chi_square": r"$\overline{\chi^2}$",
        "Var_Jz": r"$\overline{\mathrm{Var}(J_z)}$",
        "Parity_Pi_z": r"$\overline{\langle \Pi_z \rangle}$",
    }
    plot_groups = [
        ("Norm and parity", ["Norm_J", "Parity_Pi_z"]),
        ("Spin squeezing", ["Xi2_KU", "Xi2_WIN"]),
        ("Metrological gain", ["QFI_max", "chi_square"]),
        ("Variance on J_z", ["Var_Jz"]),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=True)

    for ax, (title, columns) in zip(axes.flat, plot_groups):
        for col in columns:
            df_obs = pure_plot_df[pure_plot_df["observable"] == col]
            ax.plot(df_obs["time"], df_obs["value"], linewidth=2, label=plot_labels[col])
        if any(col in {"Xi2_KU", "Xi2_WIN", "chi_square"} for col in columns):
            ax.axhline(1.0, color="black", linestyle="--", linewidth=1, alpha=0.7)
        ax.set_title(f"Ideal trajectories: {title}")
        ax.set_xlabel("t")
        ax.grid(alpha=0.3)
        ax.legend()

    axes[0, 0].set_ylabel("ensemble average")
    axes[1, 0].set_ylabel("ensemble average")
    fig.tight_layout()
    return fig


def step3_ideal(args, times, out_dir: Path):
    print("Step 3. Ideal trajectories: running conditional simulation.")
    sol_pure = run_conditional_solver(
        rho0_dm=rho0,
        times=times,
        gamma=args.gamma,
        phi1=args.phi1,
        phi2=args.phi2,
        eta_1=args.eta1_ideal,
        eta_2=args.eta2_ideal,
        ntraj=args.ntraj_ideal,
        num_cpus=args.num_cpus,
    )

    pure_traj_df = build_conditional_trajectory_df(
        sol_pure,
        times,
        label="ideal",
        eta_1=args.eta1_ideal,
        eta_2=args.eta2_ideal,
        n_spin=N_SPIN,
    )
    pure_ensemble_df = build_conditional_ensemble_df(pure_traj_df)
    pure_plot_df = pure_ensemble_df.melt(
        id_vars=["time", "state_kind"],
        value_vars=OBSERVABLE_COLUMNS,
        var_name="observable",
        value_name="value",
    )

    save_dataframe(pure_traj_df, out_dir / "step3_pure_traj_df.csv")
    save_dataframe(pure_ensemble_df, out_dir / "step3_pure_ensemble_df.csv")
    save_dataframe(pure_plot_df, out_dir / "step3_pure_plot_df.csv")
    save_preview_text(
        pure_traj_df[["traj", "time", "Norm_J", "Xi2_KU", "Xi2_WIN", "QFI_max", "chi_square", "Var_Jz", "Parity_Pi_z"]],
        out_dir / "step3_pure_traj_preview.txt",
    )
    save_preview_text(
        pure_ensemble_df[["time"] + OBSERVABLE_COLUMNS],
        out_dir / "step3_pure_ensemble_preview.txt",
    )

    fig = plot_step3_ideal(pure_plot_df)
    save_figure(fig, out_dir / "step3_ideal_summary")

    print("Step 3 completed: conditional-state DataFrames ready.")
    print("pure_traj_df shape:", pure_traj_df.shape)
    print("pure_ensemble_df shape:", pure_ensemble_df.shape)
    print("\nTrajectory-level preview:")
    print(
        pure_traj_df[["traj", "time", "Norm_J", "Xi2_KU", "Xi2_WIN", "QFI_max", "chi_square", "Var_Jz", "Parity_Pi_z"]]
        .head()
        .to_string(index=False)
    )
    print("\nEnsemble-average preview:")
    print(pure_ensemble_df[["time"] + OBSERVABLE_COLUMNS].head().to_string(index=False))

    return sol_pure, pure_traj_df, pure_ensemble_df, pure_plot_df


def step4_mixed_sanity():
    mixed_qfi_xyz = {
        "J_x": mixed_qfi_along_axis(rho0, [1, 0, 0]),
        "J_y": mixed_qfi_along_axis(rho0, [0, 1, 0]),
        "J_z": mixed_qfi_along_axis(rho0, [0, 0, 1]),
    }
    mixed_qfi_opt, mixed_axis_opt, mixed_F_matrix = mixed_optimal_axis_qfi(rho0)
    return {
        "mixed_qfi_xyz": mixed_qfi_xyz,
        "mixed_qfi_opt": mixed_qfi_opt,
        "mixed_axis_opt": mixed_axis_opt,
        "mixed_F_matrix": mixed_F_matrix,
    }


def plot_step5_main(mixed_compare_plot_df: pd.DataFrame):
    compare_labels = {
        "trajectory_mean": "mean over conditional trajectories",
        "mean_state": "mean state",
    }
    observable_labels = {
        "Norm_J": r"$|\langle \mathbf{J} \rangle|$",
        "Xi2_KU": r"$\xi^2_{KU}$",
        "Xi2_WIN": r"$\xi^2_{WIN}$",
        "QFI_max": r"$F_Q^{\max}$",
        "chi_square": r"$\chi^2$",
        "Var_Jz": r"$\mathrm{Var}(J_z)$",
        "Parity_Pi_z": r"$\langle \Pi_z \rangle$",
    }
    compare_observables = ["QFI_max", "chi_square", "Xi2_WIN", "Var_Jz"]

    fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=True)

    for ax, obs in zip(axes.flat, compare_observables):
        df_obs = mixed_compare_plot_df[mixed_compare_plot_df["observable"] == obs]
        for state_kind, df_kind in df_obs.groupby("state_kind", sort=False):
            linestyle = "-" if state_kind == "trajectory_mean" else "--"
            ax.plot(
                df_kind["time"],
                df_kind["value"],
                linewidth=2,
                linestyle=linestyle,
                label=compare_labels[state_kind],
            )
        if obs in {"Xi2_WIN", "chi_square"}:
            ax.axhline(1.0, color="black", linestyle=":", linewidth=1, alpha=0.7)
        ax.set_title(observable_labels[obs])
        ax.set_xlabel("t")
        ax.grid(alpha=0.3)
        ax.legend()

    axes[0, 0].set_ylabel("value")
    axes[1, 0].set_ylabel("value")
    fig.tight_layout()
    return fig


def plot_step5_aux(mixed_compare_plot_df: pd.DataFrame, mixed_mean_state_df: pd.DataFrame):
    compare_labels = {
        "trajectory_mean": "mean over conditional trajectories",
        "mean_state": "mean state",
    }
    observable_labels = {
        "Norm_J": r"$|\langle \mathbf{J} \rangle|$",
        "Parity_Pi_z": r"$\langle \Pi_z \rangle$",
    }

    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5), sharex=True)

    for obs in ["Norm_J", "Parity_Pi_z"]:
        df_obs = mixed_compare_plot_df[mixed_compare_plot_df["observable"] == obs]
        for state_kind, df_kind in df_obs.groupby("state_kind", sort=False):
            linestyle = "-" if state_kind == "trajectory_mean" else "--"
            axes[0].plot(
                df_kind["time"],
                df_kind["value"],
                linewidth=2,
                linestyle=linestyle,
                label=f"{observable_labels[obs]} ({compare_labels[state_kind]})",
            )

    axes[0].set_xlabel("t")
    axes[0].set_ylabel("value")
    axes[0].set_title("Mixed monitoring: norm and parity")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    for axis_col, label in zip(AXIS_COLUMNS, ["n_x", "n_y", "n_z"]):
        axes[1].plot(mixed_mean_state_df["time"], mixed_mean_state_df[axis_col], linewidth=2, label=label)

    axes[1].set_xlabel("t")
    axes[1].set_ylabel("axis component")
    axes[1].set_title("Mean state: optimal-axis components")
    axes[1].grid(alpha=0.3)
    axes[1].legend()
    fig.tight_layout()
    return fig


def step5_mixed(args, times, out_dir: Path):
    print("Step 5. Inefficient monitoring: running conditional simulation.")
    sol_mixed = run_conditional_solver(
        rho0_dm=rho0,
        times=times,
        gamma=args.gamma,
        phi1=args.phi1,
        phi2=args.phi2,
        eta_1=args.eta1_mixed,
        eta_2=args.eta2_mixed,
        ntraj=args.ntraj_mixed,
        num_cpus=args.num_cpus,
    )

    mixed_traj_df = build_conditional_trajectory_df(
        sol_mixed,
        times,
        label="inefficient",
        eta_1=args.eta1_mixed,
        eta_2=args.eta2_mixed,
        n_spin=N_SPIN,
    )
    mixed_ensemble_df = build_conditional_ensemble_df(mixed_traj_df)
    mixed_mean_state_df = build_mean_state_observables_df(
        sol_mixed,
        times,
        label="inefficient",
        eta_1=args.eta1_mixed,
        eta_2=args.eta2_mixed,
        n_spin=N_SPIN,
    )
    mixed_compare_df = pd.concat([mixed_ensemble_df, mixed_mean_state_df], ignore_index=True)
    mixed_compare_plot_df = mixed_compare_df.melt(
        id_vars=["time", "state_kind"],
        value_vars=OBSERVABLE_COLUMNS,
        var_name="observable",
        value_name="value",
    )

    save_dataframe(mixed_traj_df, out_dir / "step5_mixed_traj_df.csv")
    save_dataframe(mixed_ensemble_df, out_dir / "step5_mixed_ensemble_df.csv")
    save_dataframe(mixed_mean_state_df, out_dir / "step5_mixed_mean_state_df.csv")
    save_dataframe(mixed_compare_df, out_dir / "step5_mixed_compare_df.csv")
    save_dataframe(mixed_compare_plot_df, out_dir / "step5_mixed_compare_plot_df.csv")
    save_preview_text(
        mixed_traj_df[["traj", "time", "Norm_J", "Xi2_KU", "Xi2_WIN", "QFI_max", "chi_square", "Var_Jz", "Parity_Pi_z"]],
        out_dir / "step5_mixed_traj_preview.txt",
    )
    save_preview_text(
        mixed_ensemble_df[["time"] + OBSERVABLE_COLUMNS],
        out_dir / "step5_mixed_ensemble_preview.txt",
    )
    save_preview_text(
        mixed_mean_state_df[["time"] + OBSERVABLE_COLUMNS],
        out_dir / "step5_mixed_mean_state_preview.txt",
    )

    fig_main = plot_step5_main(mixed_compare_plot_df)
    save_figure(fig_main, out_dir / "step5_mixed_main_compare")

    fig_aux = plot_step5_aux(mixed_compare_plot_df, mixed_mean_state_df)
    save_figure(fig_aux, out_dir / "step5_mixed_aux_compare")

    print("Step 5 completed: mixed conditional-state DataFrames ready.")
    print("mixed_traj_df shape:", mixed_traj_df.shape)
    print("mixed_ensemble_df shape:", mixed_ensemble_df.shape)
    print("mixed_mean_state_df shape:", mixed_mean_state_df.shape)
    print("\nTrajectory-level preview:")
    print(
        mixed_traj_df[["traj", "time", "Norm_J", "Xi2_KU", "Xi2_WIN", "QFI_max", "chi_square", "Var_Jz", "Parity_Pi_z"]]
        .head()
        .to_string(index=False)
    )
    print("\nTrajectory-average preview:")
    print(mixed_ensemble_df[["time"] + OBSERVABLE_COLUMNS].head().to_string(index=False))
    print("\nMean-state preview:")
    print(mixed_mean_state_df[["time"] + OBSERVABLE_COLUMNS].head().to_string(index=False))

    return sol_mixed, mixed_traj_df, mixed_ensemble_df, mixed_mean_state_df, mixed_compare_df, mixed_compare_plot_df


def qfi_collective(state, axis):
    return state_qfi_along_axis(state, axis)


def optimal_collective_qfi(state):
    return state_optimal_axis_qfi(state)


def step6_examples(mixed_traj_df: pd.DataFrame, mixed_mean_state_df: pd.DataFrame):
    return {
        "example_qfi_along_Jy_on_rho0": qfi_collective(rho0, [0, 1, 0]),
        "example_optimized_qfi_first_mixed_state": mixed_traj_df.loc[
            (mixed_traj_df["traj"] == 0) & (mixed_traj_df["time_idx"] == 0), "QFI_max"
        ].iloc[0],
        "example_chi_square_final_mixed_mean_state": mixed_mean_state_df["chi_square"].iloc[-1],
        "example_optimal_mixed_mean_state_axis_final": mixed_mean_state_df[AXIS_COLUMNS].iloc[-1].to_numpy(),
    }


def make_parser() -> argparse.ArgumentParser:
    default_out_root = SCRIPT_DIR / "Graphs" / "QFI_pure_unitary_N2_Jy"

    parser = argparse.ArgumentParser(
        description="Cluster-ready Python reproduction of QFI_pure_unitary_N2_Jy.ipynb for N=2."
    )
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--phi1", type=parse_angle_arg, default=0.0, help="Angle (e.g. 0, pi/2, np.pi/2)")
    parser.add_argument("--phi2", type=parse_angle_arg, default=0.0, help="Angle (e.g. 0, pi/2, np.pi/2)")
    parser.add_argument("--eta1-ideal", type=float, default=1.0)
    parser.add_argument("--eta2-ideal", type=float, default=1.0)
    parser.add_argument("--eta1-mixed", type=float, default=0.5)
    parser.add_argument("--eta2-mixed", type=float, default=0.5)
    parser.add_argument("--ntraj-ideal", type=int, default=5)
    parser.add_argument("--ntraj-mixed", type=int, default=6)
    parser.add_argument("--num-cpus", type=int, default=1)
    parser.add_argument("--seed", type=int, default=None, help="Optional numpy seed for reproducibility.")
    parser.add_argument("--out-root", type=str, default=str(default_out_root))
    parser.add_argument("--usetex", action="store_true", help="Enable LaTeX rendering in saved plots.")
    return parser


def main() -> None:
    args = make_parser().parse_args()

    if args.seed is not None:
        np.random.seed(args.seed)

    configure_matplotlib(args.usetex)
    times = build_times()

    phi1_dir = angle_to_path(float(args.phi1))
    phi2_dir = angle_to_path(float(args.phi2))
    out_dir = (
        Path(args.out_root)
        / f"phi_1={phi1_dir}__phi_2={phi2_dir}"
        / f"ntraj_ideal={int(args.ntraj_ideal)}__ntraj_mixed={int(args.ntraj_mixed)}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    print(f"OUT_DIR={out_dir.resolve().as_posix()}")
    print(f"Fixed time grid from notebook: dt = {DT}, T_end = {T_END}, n_times = {len(times)}")

    pure_sanity = step2_pure_sanity()
    print("Step 2. Pure-state QFI on |++>:")
    for label, value in pure_sanity["pure_qfi_xyz"].items():
        print(f"  {label}: {value:.6f}")
    print(f"  optimal QFI: {pure_sanity['pure_qfi_opt']:.6f}")
    print(f"  optimal axis: {np.round(pure_sanity['pure_axis_opt'], 6)}")
    save_json(pure_sanity, out_dir / "step2_pure_sanity.json")

    if args.seed is not None:
        np.random.seed(args.seed)
    _, pure_traj_df, pure_ensemble_df, pure_plot_df = step3_ideal(args, times, out_dir)
    save_dataframe(pure_ensemble_df, out_dir / "step3_pure_ensemble_df_display.csv")

    mixed_sanity = step4_mixed_sanity()
    print("Step 4. Mixed-state formula evaluated on rho0 = |++><++|:")
    for label, value in mixed_sanity["mixed_qfi_xyz"].items():
        print(f"  {label}: {value:.6f}")
    print(f"  optimal QFI: {mixed_sanity['mixed_qfi_opt']:.6f}")
    print(f"  optimal axis: {np.round(mixed_sanity['mixed_axis_opt'], 6)}")
    save_json(mixed_sanity, out_dir / "step4_mixed_sanity.json")

    if args.seed is not None:
        np.random.seed(args.seed + 1)
    _, mixed_traj_df, mixed_ensemble_df, mixed_mean_state_df, mixed_compare_df, mixed_compare_plot_df = step5_mixed(
        args, times, out_dir
    )

    examples = step6_examples(mixed_traj_df, mixed_mean_state_df)
    print("Step 6. Minimal reusable helpers:")
    print("  Example QFI along J_y on |++><++|:", examples["example_qfi_along_Jy_on_rho0"])
    print("  Example optimized QFI on the first mixed trajectory state:", examples["example_optimized_qfi_first_mixed_state"])
    print("  Example chi^2 at final mixed mean state:", examples["example_chi_square_final_mixed_mean_state"])
    print(
        "  Example optimal mixed mean-state axis at final time:",
        np.round(examples["example_optimal_mixed_mean_state_axis_final"], 6),
    )
    save_json(examples, out_dir / "step6_examples.json")

    elapsed = time.perf_counter() - t0
    run_summary = {
        "script": Path(__file__).resolve(),
        "code_dir": CODE_DIR.resolve(),
        "N": N_SPIN,
        "dt": DT,
        "t_end": T_END,
        "n_times": int(len(times)),
        "gamma": float(args.gamma),
        "phi1": float(args.phi1),
        "phi2": float(args.phi2),
        "eta1_ideal": float(args.eta1_ideal),
        "eta2_ideal": float(args.eta2_ideal),
        "eta1_mixed": float(args.eta1_mixed),
        "eta2_mixed": float(args.eta2_mixed),
        "ntraj_ideal": int(args.ntraj_ideal),
        "ntraj_mixed": int(args.ntraj_mixed),
        "num_cpus": int(args.num_cpus),
        "seed": args.seed,
        "usetex": bool(args.usetex),
        "out_dir": out_dir.resolve(),
        "pure_traj_shape": list(pure_traj_df.shape),
        "pure_ensemble_shape": list(pure_ensemble_df.shape),
        "pure_plot_shape": list(pure_plot_df.shape),
        "mixed_traj_shape": list(mixed_traj_df.shape),
        "mixed_ensemble_shape": list(mixed_ensemble_df.shape),
        "mixed_mean_state_shape": list(mixed_mean_state_df.shape),
        "mixed_compare_shape": list(mixed_compare_df.shape),
        "mixed_compare_plot_shape": list(mixed_compare_plot_df.shape),
        "step2_pure_sanity_path": out_dir / "step2_pure_sanity.json",
        "step3_pure_traj_path": out_dir / "step3_pure_traj_df.csv",
        "step3_pure_ensemble_path": out_dir / "step3_pure_ensemble_df.csv",
        "step3_plot_path": out_dir / "step3_ideal_summary.png",
        "step4_mixed_sanity_path": out_dir / "step4_mixed_sanity.json",
        "step5_mixed_traj_path": out_dir / "step5_mixed_traj_df.csv",
        "step5_mixed_ensemble_path": out_dir / "step5_mixed_ensemble_df.csv",
        "step5_mixed_mean_state_path": out_dir / "step5_mixed_mean_state_df.csv",
        "step5_main_plot_path": out_dir / "step5_mixed_main_compare.png",
        "step5_aux_plot_path": out_dir / "step5_mixed_aux_compare.png",
        "step6_examples_path": out_dir / "step6_examples.json",
        "runtime_seconds": elapsed,
    }
    save_json(run_summary, out_dir / "run_summary.json")

    print(f"Done. Results saved in: {out_dir.resolve().as_posix()}")
    print(f"Runtime: {elapsed:.2f} s")


if __name__ == "__main__":
    main()
