"""
Homodyne Jz (N=2) with eta_minus scan and final-state tracking.

Key features:
- phi1 and phi2 fixed a priori
- eta_plus fixed to 1
- eta_minus scanned over user list (default 5 points from 0 to 1)
- observables: pair concurrence (N=2 concurrence) and xi^2_KU
- saves final states per trajectory and fidelity histograms on a fixed state basis
"""

import argparse
import json
import os
import time
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
from qutip import fidelity, ket2dm, qsave, smesolve

from quantum_function2 import (
    H_free,
    PlusPlus,
    Sz_1,
    Sz_2,
    angle_to_path,
    concurrence_for_solver_general,
    css_2,
    ee,
    eg,
    ge,
    gg,
    phi_minus,
    phi_plus,
    psi_minus,
    psi_plus,
    save_ineff_df_npz,
    xi_KU_solver,
)


DEFAULT_EOPS = [
    ("Conc", concurrence_for_solver_general),
    ("Xi2_KU", xi_KU_solver),
]

LABELS = {
    "Conc": r"$\overline{\mathcal{C}}$",
    "Xi2_KU": r"$\overline{\xi^2_{KU}}$",
}


def parse_float_list(raw: str) -> list[float]:
    vals = [float(x.strip()) for x in raw.split(",") if x.strip()]
    if not vals:
        raise ValueError("List is empty.")
    return vals


def fmt_eta(x: float) -> str:
    return f"{x:.3g}"


def build_initial_state(kind: str, theta: float, phi_state: float):
    if kind == "plusplus":
        return PlusPlus
    if kind == "ee":
        return ee
    if kind == "css":
        return css_2(theta, phi_state)
    raise ValueError(f"Unknown state kind: {kind}")


def build_homodyne_ops_eta_minus(gamma: float, phi1: float, phi2: float, eta_minus: float):
    if eta_minus < 0.0 or eta_minus > 1.0:
        raise ValueError(f"eta_minus must be in [0,1], got {eta_minus}")

    l_plus = np.sqrt(gamma / 2.0) * np.exp(1j * phi1) * (Sz_1 + Sz_2)
    l_minus = np.sqrt(gamma / 2.0) * np.exp(1j * phi2) * (Sz_1 - Sz_2)

    # Observed channels (homodyne currents)
    sc_ops = [l_plus]
    if eta_minus > 0.0:
        sc_ops.append(np.sqrt(eta_minus) * l_minus)

    # Unobserved channels (loss)
    c_ops = []
    if eta_minus < 1.0:
        c_ops.append(np.sqrt(1.0 - eta_minus) * l_minus)

    return c_ops, sc_ops


def _chunk_mean_expect(expect_chunk: np.ndarray, n_eops: int, n_times: int):
    arr = np.asarray(expect_chunk)
    arr = np.real_if_close(arr)
    arr = np.asarray(arr, dtype=float)

    # With keep_runs_results=True this is typically (n_eops, ntraj_chunk, n_times)
    if arr.ndim == 3 and arr.shape[0] == n_eops and arr.shape[2] == n_times:
        return arr.mean(axis=1)

    # Already averaged: (n_eops, n_times)
    if arr.ndim == 2 and arr.shape == (n_eops, n_times):
        return arr

    if n_eops == 1 and arr.ndim == 1 and arr.shape[0] == n_times:
        return arr.reshape(1, -1)

    raise ValueError(f"Unexpected expect shape {arr.shape}")


def run_homodyne_avg_eta_minus_with_finals(
    rho0,
    times: np.ndarray,
    gamma: float,
    phi1: float,
    phi2: float,
    eta_minus: float,
    e_ops: list,
    ntraj: int,
    chunk_size: int,
    num_cpus: int,
    seed: int,
):
    options = {
        "keep_runs_results": True,
        "store_states": False,
        "store_final_state": True,
        "num_cpus": max(1, num_cpus - 1),
        "map": "parallel" if num_cpus > 1 else "serial",
    }

    n_eops = len(e_ops)
    n_times = len(times)
    weighted = None
    done = 0
    chunk_id = 0
    rng = np.random.default_rng(seed)

    final_states_all = []

    while done < ntraj:
        chunk_id += 1
        n_this = min(chunk_size, ntraj - done)
        np.random.seed(int(rng.integers(0, 2**31 - 1)))

        c_ops, sc_ops = build_homodyne_ops_eta_minus(
            gamma=gamma,
            phi1=phi1,
            phi2=phi2,
            eta_minus=eta_minus,
        )

        sol = smesolve(
            H_free,
            rho0,
            times,
            c_ops=c_ops,
            sc_ops=sc_ops,
            heterodyne=False,
            e_ops=e_ops,
            ntraj=n_this,
            options=options,
        )

        mean_chunk = _chunk_mean_expect(sol.expect, n_eops=n_eops, n_times=n_times)

        if weighted is None:
            weighted = mean_chunk * n_this
        else:
            weighted += mean_chunk * n_this

        if hasattr(sol, "runs_final_states") and sol.runs_final_states is not None:
            final_states_all.extend(sol.runs_final_states)
        elif hasattr(sol, "runs_states") and sol.runs_states is not None:
            final_states_all.extend([traj[-1] for traj in sol.runs_states])

        done += n_this
        print(f"[eta_minus={eta_minus:.3f}] chunk {chunk_id}: {done}/{ntraj} trajectories completed")

    mean_expect = weighted / float(ntraj)
    return mean_expect, final_states_all


def build_candidate_library():
    return {
        "phi_plus": phi_plus,
        "phi_minus": phi_minus,
        "psi_plus": psi_plus,
        "psi_minus": psi_minus,
        "plusplus": PlusPlus,
        "ee": ee,
        "eg": eg,
        "ge": ge,
        "gg": gg,
    }


def fidelity_table(final_states, library: dict):
    labels = list(library.keys())
    if len(final_states) == 0:
        cols = ["traj"] + [f"F_{lab}" for lab in labels]
        return pd.DataFrame(columns=cols)

    rows = []
    for i, rho_f in enumerate(final_states, start=1):
        row = {"traj": i}
        for lab in labels:
            row[f"F_{lab}"] = float(np.real_if_close(fidelity(rho_f, library[lab])))
        rows.append(row)
    return pd.DataFrame(rows)


def average_final_state(final_states):
    if len(final_states) == 0:
        return None
    rho = 0 * final_states[0]
    for r in final_states:
        rho = rho + r
    return rho / len(final_states)


def plot_fidelity_histograms(ftab: pd.DataFrame, state_labels: list[str], eta_minus: float, out_path: Path):
    n = len(state_labels)
    ncols = 3
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.0 * ncols, 3.6 * nrows), sharex=True)
    axes = np.asarray(axes).reshape(-1)

    for i, lab in enumerate(state_labels):
        ax = axes[i]
        col = f"F_{lab}"
        vals = ftab[col].to_numpy(dtype=float) if col in ftab.columns else np.array([], dtype=float)
        if vals.size > 0:
            ax.hist(vals, bins=np.linspace(0.0, 1.0, 26), color="#4472C4", alpha=0.85, edgecolor="white")
        ax.set_xlim(0.0, 1.0)
        ax.set_title(lab)
        ax.grid(True, linestyle=":", alpha=0.5)
        ax.set_ylabel("count")
        ax.set_xlabel("fidelity")

    for j in range(n, len(axes)):
        axes[j].axis("off")

    fig.suptitle(rf"Final-state fidelity histograms, $\eta_-={eta_minus:.2f}$", y=1.01)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="N=2 homodyne eta_minus scan with final-state tracking")
    parser.add_argument("--state", choices=["plusplus", "ee", "css"], default="plusplus")
    parser.add_argument("--theta", type=float, default=np.pi / 2.0, help="Used when --state css")
    parser.add_argument("--phi-state", type=float, default=0.0, help="Used when --state css")
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--phi1", type=float, default=0.0)
    parser.add_argument("--phi2", type=float, default=0.0)
    parser.add_argument("--eta-minus-list", type=str, default="0,0.25,0.5,0.75,1")
    parser.add_argument("--ntraj", type=int, default=1000)
    parser.add_argument("--chunk-size", type=int, default=100)
    parser.add_argument("--t-end", type=float, default=10.0, help="Final time in units of T1")
    parser.add_argument("--dt", type=float, default=0.01, help="Time step in units of T1")
    parser.add_argument("--num-cpus", type=int, default=max(1, os.cpu_count() - 1))
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--usetex", action="store_true", help="Enable matplotlib usetex")
    parser.add_argument("--tag", type=str, default="", help="Optional tag appended to output folder")
    parser.add_argument(
        "--out-root",
        type=str,
        default=r".\Codes\Graphs\Jz_2_Homodyne_N2_eta_minus_scan_final_state",
    )
    return parser


def main() -> None:
    t0 = time.perf_counter()
    args = make_parser().parse_args()

    eta_plus = 1.0
    eta_minus_list = parse_float_list(args.eta_minus_list)

    psi0 = build_initial_state(args.state, args.theta, args.phi_state)
    rho0 = ket2dm(psi0)

    gamma = float(args.gamma)
    t1 = 1.0 / gamma
    times = np.arange(0.0, args.t_end * t1, args.dt * t1)

    columns = [name for name, _ in DEFAULT_EOPS]
    e_ops = [op for _, op in DEFAULT_EOPS]

    ineff_df = {}
    final_states_dict = {}

    for eta_minus in eta_minus_list:
        key = fmt_eta(eta_minus)
        mean_expect, final_states = run_homodyne_avg_eta_minus_with_finals(
            rho0=rho0,
            times=times,
            gamma=gamma,
            phi1=float(args.phi1),
            phi2=float(args.phi2),
            eta_minus=float(eta_minus),
            e_ops=e_ops,
            ntraj=int(args.ntraj),
            chunk_size=int(args.chunk_size),
            num_cpus=int(args.num_cpus),
            seed=int(args.seed + 1000 * eta_minus),
        )

        df = pd.DataFrame(mean_expect.T, columns=columns)
        df.insert(0, "step", np.arange(len(times), dtype=int))
        ineff_df[key] = df
        final_states_dict[key] = final_states

    # Output path
    phi1_dir = angle_to_path(float(args.phi1))
    phi2_dir = angle_to_path(float(args.phi2))
    base_name = f"state={args.state}_phi_1={phi1_dir}_phi_2={phi2_dir}"
    if args.tag:
        base_name = f"{base_name}_{args.tag}"
    out_dir = Path(args.out_root) / base_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save mean curves
    npz_path = out_dir / f"homodyne_eta_minus_scan_phi1={float(args.phi1):.6g}_phi2={float(args.phi2):.6g}.npz"
    save_ineff_df_npz(ineff_df, npz_path, meta=None, step_col="step")

    # Save final states
    qsave(final_states_dict, out_dir / "final_states_dict")

    # Fidelity analysis
    library = build_candidate_library()
    fidelity_tables = {}
    avg_state_fidelities = {}
    state_labels = list(library.keys())

    for eta_minus in eta_minus_list:
        key = fmt_eta(eta_minus)
        ftab = fidelity_table(final_states_dict[key], library)
        fidelity_tables[key] = ftab

        rho_avg = average_final_state(final_states_dict[key])
        if rho_avg is not None:
            avg_state_fidelities[key] = {
                lab: float(np.real_if_close(fidelity(rho_avg, st))) for lab, st in library.items()
            }
        else:
            avg_state_fidelities[key] = {}

        ftab.to_csv(out_dir / f"final_state_fidelities_eta_minus={key}.csv", index=False)
        plot_fidelity_histograms(
            ftab=ftab,
            state_labels=state_labels,
            eta_minus=float(eta_minus),
            out_path=out_dir / f"final_state_fidelity_hist_eta_minus={key}.pdf",
        )

    # Plot
    plt.rcParams["text.usetex"] = bool(args.usetex)
    plt.rcParams.update(
        {
            "mathtext.fontset": "cm",
            "font.family": "serif",
            "font.size": 14,
            "axes.unicode_minus": False,
        }
    )
    x = times / t1
    for col in columns:
        plt.figure(figsize=(12, 8))
        for eta_minus in eta_minus_list:
            key = fmt_eta(eta_minus)
            y = ineff_df[key][col].to_numpy()
            plt.plot(x, y, label=rf"$\eta_- = {eta_minus:.2f}$")

        plt.xlim(0.0, float(args.t_end))
        plt.xlabel(r"$t/T_1$")
        plt.ylabel(LABELS[col])
        plt.title(
            r"Homodyne $J_z$ (N=2), $\eta_+=1$: "
            + LABELS[col]
            + rf" (n$_\mathrm{{traj}}$={args.ntraj})"
        )
        plt.grid(True, linestyle=":", alpha=0.6)
        plt.legend(loc="upper left", bbox_to_anchor=(0, 0.95), fontsize=12)
        plt.savefig(out_dir / f"Homodyne_eta_minus_scan_{col}.pdf", bbox_inches="tight")
        plt.close()

    runtime_sec = float(time.perf_counter() - t0)
    meta = {
        "measurement": "homodyne",
        "N": 2,
        "state": args.state,
        "theta": float(args.theta),
        "phi_state": float(args.phi_state),
        "gamma": gamma,
        "phi1": float(args.phi1),
        "phi2": float(args.phi2),
        "eta_plus_fixed": eta_plus,
        "eta_minus_list": [float(x) for x in eta_minus_list],
        "ntraj": int(args.ntraj),
        "chunk_size": int(args.chunk_size),
        "dt_T1": float(args.dt),
        "t_end_T1": float(args.t_end),
        "num_cpus": int(args.num_cpus),
        "seed": int(args.seed),
        "columns": columns,
        "runtime_sec": runtime_sec,
        "out_dir": str(out_dir),
    }
    run_cfg = out_dir / "run_config.json"
    run_cfg.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    avg_fid_json = out_dir / "avg_state_fidelities.json"
    avg_fid_json.write_text(json.dumps(avg_state_fidelities, indent=2), encoding="utf-8")

    print(f"Done. Results saved in: {out_dir}")
    print(f"OUT_DIR={out_dir.resolve()}")
    print(f"RUN_CONFIG_PATH={run_cfg.resolve()}")


if __name__ == "__main__":
    main()
