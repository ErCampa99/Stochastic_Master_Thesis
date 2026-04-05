"""
Utilities for tracking individual N=2 homodyne trajectories.

The dynamical setup mirrors ``main_SLURM.py``:
- two qubits (N=2)
- homodyne monitoring
- collective plus/minus channels built from sigma_z
- density-matrix stochastic evolution via ``qutip.smesolve``

This module is notebook-oriented: it stores the full state at every time step
for a small number of trajectories and computes non-averaged observables for
each trajectory separately.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from qutip import Qobj, basis, expect, fidelity, ket2dm, qeye, sigmax, sigmay, sigmaz, smesolve, tensor


# Basis states. Convention matches Codes/main_SLURM.py:
# |g> = |0>, |e> = |1>.
gnd = basis(2, 0)
exc = basis(2, 1)

plus_state = (gnd + exc).unit()
minus_state = (gnd - exc).unit()
PlusPlus = tensor(plus_state, plus_state)
ee = tensor(exc, exc)

I_2 = qeye(2)
J_x = 0.5 * (tensor(sigmax(), I_2) + tensor(I_2, sigmax()))
J_y = 0.5 * (tensor(sigmay(), I_2) + tensor(I_2, sigmay()))
J_z = 0.5 * (tensor(sigmaz(), I_2) + tensor(I_2, sigmaz()))

OMEGA = 0.0
H_free = 0.5 * OMEGA * (tensor(sigmaz(), I_2) + tensor(I_2, sigmaz()))

Cp = np.sqrt(0.5) * (tensor(sigmaz(), I_2) + tensor(I_2, sigmaz()))
Cm = np.sqrt(0.5) * (tensor(sigmaz(), I_2) - tensor(I_2, sigmaz()))


FIDELITY_TARGETS = {
    "gg": tensor(gnd, gnd),
    "ge": tensor(gnd, exc),
    "eg": tensor(exc, gnd),
    "ee": tensor(exc, exc),
    "plusplus": tensor(plus_state, plus_state),
    "plusminus": tensor(plus_state, minus_state),
    "minusplus": tensor(minus_state, plus_state),
    "minusminus": tensor(minus_state, minus_state),
    "phi_plus": (tensor(gnd, gnd) + tensor(exc, exc)).unit(),
    "phi_minus": (tensor(gnd, gnd) - tensor(exc, exc)).unit(),
    "psi_plus": (tensor(gnd, exc) + tensor(exc, gnd)).unit(),
    "psi_minus": (tensor(gnd, exc) - tensor(exc, gnd)).unit(),
}

COMPUTATIONAL_FIDELITY_COLUMNS = ["fid_gg", "fid_ge", "fid_eg", "fid_ee"]
PLUS_MINUS_FIDELITY_COLUMNS = ["fid_plusplus", "fid_plusminus", "fid_minusplus", "fid_minusminus"]
BELL_FIDELITY_COLUMNS = ["fid_phi_plus", "fid_phi_minus", "fid_psi_plus", "fid_psi_minus"]
SUMMARY_FIDELITY_COLUMNS = COMPUTATIONAL_FIDELITY_COLUMNS + BELL_FIDELITY_COLUMNS
ALL_FIDELITY_COLUMNS = SUMMARY_FIDELITY_COLUMNS + PLUS_MINUS_FIDELITY_COLUMNS
UNDETERMINED_STATE_LABEL = "undetermined"
STEADY_STATE_ORDER = ["gg", "ge", "eg", "ee", "phi_plus", "phi_minus", "psi_plus", "psi_minus", UNDETERMINED_STATE_LABEL]

FIDELITY_LABELS = {
    "fid_gg": r"$F_{|gg\rangle}$",
    "fid_ge": r"$F_{|ge\rangle}$",
    "fid_eg": r"$F_{|eg\rangle}$",
    "fid_ee": r"$F_{|ee\rangle}$",
    "fid_plusplus": r"$F_{|++\rangle}$",
    "fid_plusminus": r"$F_{|+-\rangle}$",
    "fid_minusplus": r"$F_{|-+\rangle}$",
    "fid_minusminus": r"$F_{|--\rangle}$",
    "fid_phi_plus": r"$F_{|\phi^+\rangle}$",
    "fid_phi_minus": r"$F_{|\phi^-\rangle}$",
    "fid_psi_plus": r"$F_{|\psi^+\rangle}$",
    "fid_psi_minus": r"$F_{|\psi^-\rangle}$",
}

STATE_DISPLAY_LABELS = {
    "gg": r"$|gg\rangle$",
    "ge": r"$|ge\rangle$",
    "eg": r"$|eg\rangle$",
    "ee": r"$|ee\rangle$",
    "phi_plus": r"$|\phi^+\rangle$",
    "phi_minus": r"$|\phi^-\rangle$",
    "psi_plus": r"$|\psi^+\rangle$",
    "psi_minus": r"$|\psi^-\rangle$",
    UNDETERMINED_STATE_LABEL: "undetermined",
}


@dataclass(frozen=True)
class TrackingConfig:
    state: str = "plusplus"
    theta: float = np.pi / 2.0
    phi_state: float = 0.0
    gamma: float = 1.0
    phi1: float = 0.0
    phi2: float = 0.0
    eta_1: float = 1.0
    eta_2: float = 0.0
    ntraj: int = 10
    t_end: float = 5.0
    dt: float = 0.001
    num_cpus: int = 1
    seed: int | None = 12345
    out_root: str | Path = "./Codes/Graphs/Jz_2_Homodyne_N2_single_trajectory"


@dataclass
class TrackingResults:
    config: TrackingConfig
    times: np.ndarray
    times_t1: np.ndarray
    states: np.ndarray
    trajectory_dataframes: dict[int, pd.DataFrame]
    final_state_summary: pd.DataFrame

    def trajectory_frame(self, trajectory: int) -> pd.DataFrame:
        return self.trajectory_dataframes[int(trajectory)].copy()

    def trajectory_frames(self) -> dict[int, pd.DataFrame]:
        return {trajectory: self.trajectory_frame(trajectory) for trajectory in self.trajectory_ids}

    @property
    def trajectory_ids(self) -> list[int]:
        return sorted(self.trajectory_dataframes)

    @property
    def nsteps(self) -> int:
        return int(self.states.shape[1])

    @property
    def dims(self) -> list[list[int]]:
        return [[2, 2], [2, 2]]


def get_trajectory_dataframes(results: TrackingResults | object) -> dict[int, pd.DataFrame]:
    """Return one dataframe per trajectory, also supporting older result objects.

    This keeps notebooks working even if the kernel still holds a ``results``
    object built with a previous version of the module.
    """
    if hasattr(results, "trajectory_frames"):
        return results.trajectory_frames()

    if hasattr(results, "trajectory_dataframes"):
        data = getattr(results, "trajectory_dataframes")
        return {int(trajectory): df.copy() for trajectory, df in data.items()}

    if hasattr(results, "observables"):
        observables = getattr(results, "observables")
        if "trajectory" not in observables.columns:
            raise AttributeError("Legacy results.observables exists but has no 'trajectory' column.")
        return {
            int(trajectory): group.drop(columns=["trajectory"]).reset_index(drop=True)
            for trajectory, group in observables.groupby("trajectory", sort=True)
        }

    raise AttributeError("Unsupported results object: cannot extract per-trajectory dataframes.")


def css_2(theta: float, phi: float) -> Qobj:
    single_qubit = (
        np.sin(theta / 2.0) * basis(2, 0)
        + np.exp(1j * phi) * np.cos(theta / 2.0) * basis(2, 1)
    )
    return tensor(single_qubit, single_qubit)


def build_initial_state(kind: str, theta: float, phi_state: float) -> Qobj:
    if kind == "plusplus":
        return PlusPlus
    if kind == "ee":
        return ee
    if kind == "css":
        return css_2(theta, phi_state)
    raise ValueError(f"Unknown initial state kind: {kind}")


def collapsing_operators(gamma: float, phi_1: float, phi_2: float, eta_1: float, eta_2: float) -> list[Qobj]:
    return [
        np.sqrt(gamma * eta_1) * np.exp(1j * phi_1) * Cp,
        np.sqrt(gamma * eta_2) * np.exp(1j * phi_2) * Cm,
    ]


def runs_states_to_array(runs_states: list[list[Qobj]]) -> np.ndarray:
    ntraj = len(runs_states)
    nsteps = len(runs_states[0]) if ntraj > 0 else 0
    if ntraj == 0 or nsteps == 0:
        return np.empty((ntraj, nsteps, 0, 0), dtype=np.complex128)

    dim0, dim1 = runs_states[0][0].shape
    out = np.empty((ntraj, nsteps, dim0, dim1), dtype=np.complex128)
    for itr, run in enumerate(runs_states):
        if len(run) != nsteps:
            raise ValueError(f"Inconsistent time grid in trajectory {itr}: {len(run)} != {nsteps}")
        for istep, state in enumerate(run):
            out[itr, istep] = state.full()
    return out


def state_from_array(state_matrix: np.ndarray) -> Qobj:
    return Qobj(state_matrix, dims=[[2, 2], [2, 2]])


def concurrence_pure_state(state: Qobj) -> float:
    c = state.full().flatten()
    return float(2.0 * np.abs(c[0] * c[3] - c[1] * c[2]))


def concurrence_for_state(state: Qobj) -> float:
    if state.isket:
        conc = concurrence_pure_state(state)
    else:
        sysy = tensor(sigmay(), sigmay())
        rho_tilde = (state * sysy) * (state.conj() * sysy)
        evals = rho_tilde.eigenenergies()
        evals = abs(np.sort(np.real(evals)))
        sqrt_evals = np.sqrt(evals)
        conc = np.maximum(0.0, sqrt_evals[3] - sqrt_evals[2] - sqrt_evals[1] - sqrt_evals[0])
    return float(np.real_if_close(conc))


def eval_spin_components(state: Qobj) -> tuple[complex, complex, complex]:
    return expect(J_x, state), expect(J_y, state), expect(J_z, state)


def xi_ku_for_state(state: Qobj, tol: float = 1e-12) -> float:
    jx_exp, jy_exp, jz_exp = eval_spin_components(state)
    jmean = np.array(
        [
            float(np.real_if_close(jx_exp)),
            float(np.real_if_close(jy_exp)),
            float(np.real_if_close(jz_exp)),
        ]
    )
    norm = np.linalg.norm(jmean)

    cov = np.zeros((3, 3), dtype=float)
    js = [J_x, J_y, J_z]
    for i, op_i in enumerate(js):
        for j, op_j in enumerate(js):
            second_moment = 0.5 * expect(op_i * op_j + op_j * op_i, state)
            cov[i, j] = float(np.real_if_close(second_moment - jmean[i] * jmean[j]))

    if norm > tol:
        u = jmean / norm
        a = np.array([1.0, 0.0, 0.0]) if abs(u[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        e1 = a - u * np.dot(u, a)
        e1 /= np.linalg.norm(e1)
        e2 = np.cross(u, e1)
        cov2 = np.array(
            [
                [e1 @ cov @ e1, e1 @ cov @ e2],
                [e2 @ cov @ e1, e2 @ cov @ e2],
            ],
            dtype=float,
        )
        lam_min = np.linalg.eigvalsh(cov2)[0]
    else:
        lam_min = np.linalg.eigvalsh(cov)[0]

    lam_min = max(float(np.real_if_close(lam_min)), 0.0)
    return float((4.0 / 2.0) * lam_min)


def norm_j_for_state(state: Qobj) -> float:
    return float(np.linalg.norm(eval_spin_components(state)))


def xi_win_for_state(state: Qobj, tol: float = 1e-12) -> float:
    xi2_ku = xi_ku_for_state(state, tol=tol)
    norm = norm_j_for_state(state)
    return float(xi2_ku * ((2.0 / 2.0) / norm) ** 2) if norm > tol else float(xi2_ku)


def times_from_config(config: TrackingConfig) -> tuple[np.ndarray, np.ndarray]:
    gamma = float(config.gamma)
    if gamma <= 0.0:
        raise ValueError("gamma must be strictly positive.")

    t1 = 1.0 / gamma
    times = np.arange(0.0, float(config.t_end) * t1, float(config.dt) * t1)
    times_t1 = times / t1
    return times, times_t1


def compute_observables_by_trajectory(
    times: np.ndarray,
    times_t1: np.ndarray,
    states: np.ndarray,
) -> dict[int, pd.DataFrame]:
    trajectory_dataframes: dict[int, pd.DataFrame] = {}

    for itr in range(states.shape[0]):
        rows: list[dict[str, float | int]] = []
        for istep in range(states.shape[1]):
            rho = state_from_array(states[itr, istep])
            row: dict[str, float | int] = {
                "step": int(istep),
                "t": float(times[istep]),
                "t_over_T1": float(times_t1[istep]),
                "concurrence": concurrence_for_state(rho),
                "xi2_ku": xi_ku_for_state(rho),
                "xi2_win": xi_win_for_state(rho),
                "norm_j": norm_j_for_state(rho),
            }
            for name, target in FIDELITY_TARGETS.items():
                row[f"fid_{name}"] = float(np.real_if_close(fidelity(rho, target)))
            rows.append(row)

        trajectory_dataframes[int(itr)] = pd.DataFrame(rows)

    return trajectory_dataframes


def summarize_final_states(
    trajectory_dataframes: dict[int, pd.DataFrame],
    close_to_one_tol: float = 5e-2,
) -> tuple[dict[int, pd.DataFrame], pd.DataFrame]:
    annotated_dataframes: dict[int, pd.DataFrame] = {}
    summary_rows: list[dict[str, object]] = []

    for trajectory in sorted(trajectory_dataframes):
        df = trajectory_dataframes[trajectory].copy()
        final_row = df.iloc[-1]

        final_fidelities = {col: float(final_row[col]) for col in SUMMARY_FIDELITY_COLUMNS}
        best_column = max(final_fidelities, key=final_fidelities.get)
        best_state = best_column.removeprefix("fid_")
        best_fidelity = float(final_fidelities[best_column])
        close_to_one = abs(1.0 - best_fidelity) <= float(close_to_one_tol)
        classified_state = best_state if close_to_one else UNDETERMINED_STATE_LABEL

        df["best_match_label"] = best_state
        df["steady_state_label"] = classified_state
        df["steady_state_fidelity"] = best_fidelity
        df["steady_state_close_to_one"] = close_to_one
        annotated_dataframes[trajectory] = df

        summary_rows.append(
            {
                "trajectory": int(trajectory),
                "best_match_label": best_state,
                "best_match_display": STATE_DISPLAY_LABELS[best_state],
                "steady_state_label": classified_state,
                "steady_state_display": STATE_DISPLAY_LABELS[classified_state],
                "steady_state_fidelity": best_fidelity,
                "steady_state_close_to_one": close_to_one,
                **{f"final_{col}": val for col, val in final_fidelities.items()},
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    return annotated_dataframes, summary_df


def run_tracking_experiment(config: TrackingConfig) -> TrackingResults:
    if config.ntraj <= 0:
        raise ValueError("ntraj must be positive.")

    if config.seed is not None:
        np.random.seed(int(config.seed))

    psi0 = build_initial_state(config.state, config.theta, config.phi_state)
    rho0 = ket2dm(psi0)
    times, times_t1 = times_from_config(config)

    num_cpus = max(1, int(config.num_cpus))
    options = {
        "keep_runs_results": True,
        "store_states": True,
        "num_cpus": num_cpus,
        "map": "parallel" if num_cpus > 1 else "serial",
    }

    sol = smesolve(
        H_free,
        rho0,
        times,
        c_ops=collapsing_operators(config.gamma, config.phi1, config.phi2, 1.0 - config.eta_1, 1.0 - config.eta_2),
        sc_ops=collapsing_operators(config.gamma, config.phi1, config.phi2, config.eta_1, config.eta_2),
        heterodyne=False,
        ntraj=int(config.ntraj),
        options=options,
    )

    states = runs_states_to_array(sol.runs_states)
    trajectory_dataframes = compute_observables_by_trajectory(times, times_t1, states)
    trajectory_dataframes, final_state_summary = summarize_final_states(trajectory_dataframes)

    return TrackingResults(
        config=config,
        times=times,
        times_t1=times_t1,
        states=states,
        trajectory_dataframes=trajectory_dataframes,
        final_state_summary=final_state_summary,
    )


def angle_to_path(phi: float, max_den: int = 12, tol: float = 1e-10) -> str:
    twopi = 2 * np.pi
    x = (phi % twopi + twopi) % twopi
    if abs(x) < tol:
        return "0"

    ratio = x / np.pi
    for den in range(1, max_den + 1):
        num = round(ratio * den)
        if abs(ratio - num / den) < tol:
            if num == 0:
                return "0"
            if den == 1:
                return "pi" if num == 1 else f"{num}pi"
            return f"pi_{den}" if num == 1 else f"{num}pi_{den}"

    return f"{x:.6g}".replace(".", "p")


def make_output_dir(config: TrackingConfig) -> Path:
    root = Path(config.out_root)
    state_tag = config.state
    eta_tag = f"eta1={config.eta_1:g}_eta2={config.eta_2:g}"
    phi_tag = f"phi1={angle_to_path(config.phi1)}_phi2={angle_to_path(config.phi2)}"
    traj_tag = f"ntraj={config.ntraj}"
    out_dir = root / f"state={state_tag}_{eta_tag}_{phi_tag}_{traj_tag}"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def save_tracking_results(results: TrackingResults, out_dir: Path | None = None) -> Path:
    out_dir = make_output_dir(results.config) if out_dir is None else Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        out_dir / "trajectory_states.npz",
        times=np.asarray(results.times, dtype=float),
        times_t1=np.asarray(results.times_t1, dtype=float),
        states=np.asarray(results.states, dtype=np.complex128),
    )
    trajectory_csv_files: list[str] = []
    for trajectory in results.trajectory_ids:
        df = results.trajectory_frame(trajectory)
        csv_name = f"trajectory_{trajectory + 1:02d}_observables.csv"
        df.to_csv(out_dir / csv_name, index=False)
        trajectory_csv_files.append(csv_name)
    results.final_state_summary.to_csv(out_dir / "final_state_summary.csv", index=False)

    meta = asdict(results.config)
    meta["out_root"] = str(meta["out_root"])
    meta.update(
        {
            "nsteps": int(results.nsteps),
            "state_shape": list(results.states.shape),
            "trajectory_csv_files": trajectory_csv_files,
            "final_state_summary_file": "final_state_summary.csv",
            "computational_fidelities": COMPUTATIONAL_FIDELITY_COLUMNS,
            "plus_minus_fidelities": PLUS_MINUS_FIDELITY_COLUMNS,
            "bell_fidelities": BELL_FIDELITY_COLUMNS,
        }
    )
    (out_dir / "run_config.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return out_dir


def _plot_by_trajectory(
    results: TrackingResults,
    value_col: str,
    ylabel: str,
    title: str,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    ax = ax or plt.gca()
    for itr in results.trajectory_ids:
        group = results.trajectory_frame(itr)
        ax.plot(group["t_over_T1"], group[value_col], linewidth=1.8, label=f"traj {int(itr) + 1}")
    ax.set_xlabel(r"$t / T_1$")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    return ax


def plot_spin_squeezing(results: TrackingResults, figsize: tuple[float, float] = (12, 8)) -> tuple[plt.Figure, np.ndarray]:
    fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True, constrained_layout=True)
    _plot_by_trajectory(results, "xi2_ku", r"$\xi^2_{KU}$", "Spin Squeezing KU", ax=axes[0])
    _plot_by_trajectory(results, "xi2_win", r"$\xi^2_{WIN}$", "Spin Squeezing Wineland", ax=axes[1])
    axes[0].axhline(1.0, linestyle="--", color="black", linewidth=1.2, alpha=0.8)
    axes[1].axhline(1.0, linestyle="--", color="black", linewidth=1.2, alpha=0.8)
    axes[1].set_ylim(0.0, 10.0)
    axes[0].legend(ncol=min(5, results.config.ntraj), frameon=False)
    return fig, axes


def plot_concurrence(results: TrackingResults, figsize: tuple[float, float] = (12, 4.5)) -> tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    _plot_by_trajectory(results, "concurrence", r"$\mathcal{C}$", "Concurrence", ax=ax)
    ax.legend(ncol=min(5, results.config.ntraj), frameon=False)
    return fig, ax


def plot_fidelity_grid(
    results: TrackingResults,
    columns: list[str],
    suptitle: str,
    figsize: tuple[float, float] = (12, 8),
) -> tuple[plt.Figure, np.ndarray]:
    fig, axes = plt.subplots(2, 2, figsize=figsize, sharex=True, sharey=True, constrained_layout=True)
    axes = np.asarray(axes)

    for ax, col in zip(axes.flat, columns):
        _plot_by_trajectory(results, col, FIDELITY_LABELS[col], FIDELITY_LABELS[col], ax=ax)
        ax.axhline(1.0, linestyle="--", color="black", linewidth=1.2, alpha=0.8)
        ax.axhline(1.0 / np.sqrt(2.0), linestyle="--", color="gray", linewidth=1.2, alpha=0.8)
        ax.set_ylim(-0.02, 1.02)

    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=min(5, results.config.ntraj), frameon=False)
    fig.suptitle(suptitle, y=1.02)
    return fig, axes


def plot_final_state_histogram(
    results: TrackingResults,
    figsize: tuple[float, float] = (10, 4.8),
) -> tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    counts = (
        results.final_state_summary["steady_state_label"]
        .value_counts()
        .reindex(STEADY_STATE_ORDER, fill_value=0)
    )
    labels = [STATE_DISPLAY_LABELS[state] for state in STEADY_STATE_ORDER]
    ax.bar(labels, counts.to_numpy(dtype=float), color="#4C72B0", edgecolor="black", alpha=0.85)
    ax.set_xlabel("Steady state finale")
    ax.set_ylabel("Conteggio traiettorie")
    ax.set_title("Istogramma finale dei final state")
    ax.grid(True, axis="y", alpha=0.3)
    return fig, ax


def save_figures(figures: dict[str, plt.Figure], out_dir: Path) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for stem, fig in figures.items():
        fig.savefig(out_dir / f"{stem}.png", dpi=220, bbox_inches="tight")
        fig.savefig(out_dir / f"{stem}.pdf", bbox_inches="tight")
