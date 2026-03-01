"""
Homodyne fluorescence simulation for N=2 (Jz setup).

This script is intentionally standalone and does not modify existing notebooks/files.
It reuses the operators/utilities already defined in `quantum_function2.py`.
"""

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from qutip import ket2dm, smesolve

from quantum_function2 import (
    H_free,
    PlusPlus,
    Variance_z,
    angle_to_path,
    angle_to_tex,
    collapsing_operators,
    concurrence_for_solver_general,
    css_2,
    ee,
    save_ineff_df_npz,
    xi_KU_solver,
)


DEFAULT_EOPS = [
    ("Conc", concurrence_for_solver_general),
    ("Xi2_KU", xi_KU_solver),
    ("Variance_z", Variance_z),
]

LABELS = {
    "Conc": r"$\overline{\mathcal{C}}$",
    "Xi2_KU": r"$\overline{\xi^2_{KU}}$",
    "Variance_z": r"$\overline{\mathrm{Var}(J_z)}$",
}


def parse_etas(raw: str) -> list[float]:
    etas = [float(x.strip()) for x in raw.split(",") if x.strip()]
    if not etas:
        raise ValueError("Eta list is empty. Use something like: --etas 1,0.9,0.7")
    for eta in etas:
        if eta < 0.0 or eta > 1.0:
            raise ValueError(f"Invalid eta={eta}. Each eta must be in [0, 1].")
    return etas


def build_initial_state(kind: str, theta: float, phi_state: float):
    if kind == "plusplus":
        return PlusPlus, np.pi / 2.0
    if kind == "ee":
        return ee, np.pi
    if kind == "css":
        return css_2(theta, phi_state), theta
    raise ValueError(f"Unknown state kind: {kind}")


def run_homodyne_avg(
    rho0,
    times: np.ndarray,
    gamma: float,
    phi1: float,
    phi2: float,
    eta: float,
    e_ops: list,
    ntraj: int,
    chunk_size: int,
    num_cpus: int,
    seed: int,
) -> np.ndarray:
    if ntraj <= 0:
        raise ValueError("ntraj must be > 0")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")

    options = {
        "keep_runs_results": False,
        "num_cpus": max(1, num_cpus),
        "map": "parallel" if num_cpus > 1 else "serial",
    }

    weighted_sum = None
    done = 0
    chunk_id = 0
    rng = np.random.default_rng(seed)

    while done < ntraj:
        chunk_id += 1
        n_this = min(chunk_size, ntraj - done)
        np.random.seed(int(rng.integers(0, 2**31 - 1)))

        sol = smesolve(
            H_free,
            rho0,
            times,
            c_ops=collapsing_operators(gamma, phi1, phi2, 1.0 - eta),
            sc_ops=collapsing_operators(gamma, phi1, phi2, eta),
            heterodyne=False,
            e_ops=e_ops,
            ntraj=n_this,
            options=options,
        )

        expect_chunk = np.asarray(sol.expect)
        expect_chunk = np.real_if_close(expect_chunk)
        expect_chunk = np.asarray(expect_chunk, dtype=float)

        if weighted_sum is None:
            weighted_sum = expect_chunk * n_this
        else:
            weighted_sum += expect_chunk * n_this

        done += n_this
        print(f"[eta={eta:.3f}] chunk {chunk_id}: {done}/{ntraj} trajectories completed")

    return weighted_sum / float(ntraj)


def theoretical_concurrence_curve(
    times: np.ndarray,
    gamma: float,
    eta: float,
    theta: float,
) -> np.ndarray:
    return (np.sin(theta) ** 2) * np.exp(-(1.0 - eta) * times) * (
        1.0 - np.exp(-gamma * eta * times / 2.0)
    )


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="N=2 homodyne simulation (Jz)")
    parser.add_argument("--state", choices=["plusplus", "ee", "css"], default="plusplus")
    parser.add_argument("--theta", type=float, default=np.pi / 2.0, help="Used when --state css")
    parser.add_argument("--phi-state", type=float, default=0.0, help="Used when --state css")
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--phi1", type=float, default=0.0)
    parser.add_argument("--phi2", type=float, default=0.0)
    parser.add_argument("--etas", type=str, default="1,0.9,0.7,0.5,0.3")
    parser.add_argument("--ntraj", type=int, default=4000)
    parser.add_argument("--chunk-size", type=int, default=500)
    parser.add_argument("--t-end", type=float, default=10.0, help="Final time in units of T1")
    parser.add_argument("--dt", type=float, default=0.01, help="Time step in units of T1")
    parser.add_argument("--num-cpus", type=int, default=max(1, os.cpu_count() - 1))
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument(
        "--out-root",
        type=str,
        default=r".\Codes\Graphs\Jz_2_Homodyne_N2",
    )
    parser.add_argument(
        "--no-theory",
        action="store_true",
        help="Disable theoretical concurrence overlay",
    )
    return parser


def main() -> None:
    args = make_parser().parse_args()
    etas = parse_etas(args.etas)

    psi0, theta_for_theory = build_initial_state(args.state, args.theta, args.phi_state)
    rho0 = ket2dm(psi0)

    gamma = float(args.gamma)
    t1 = 1.0 / gamma
    times = np.arange(0.0, args.t_end * t1, args.dt * t1)

    columns = [name for name, _ in DEFAULT_EOPS]
    e_ops = [op for _, op in DEFAULT_EOPS]

    ineff_df: dict[str, pd.DataFrame] = {}
    for eta in etas:
        avg_expect = run_homodyne_avg(
            rho0=rho0,
            times=times,
            gamma=gamma,
            phi1=float(args.phi1),
            phi2=float(args.phi2),
            eta=float(eta),
            e_ops=e_ops,
            ntraj=int(args.ntraj),
            chunk_size=int(args.chunk_size),
            num_cpus=int(args.num_cpus),
            seed=int(args.seed + 1000 * eta),
        )
        ineff_df[f"{eta:g}"] = pd.DataFrame(avg_expect.T, columns=columns)

    phi1_dir = angle_to_path(float(args.phi1))
    phi2_dir = angle_to_path(float(args.phi2))
    out_dir = Path(args.out_root) / f"phi_1={phi1_dir}_phi_2={phi2_dir}"
    out_dir.mkdir(parents=True, exist_ok=True)

    meta = {
        "measurement": "homodyne",
        "N": 2,
        "state": args.state,
        "theta": float(args.theta),
        "phi_state": float(args.phi_state),
        "gamma": gamma,
        "phi1": float(args.phi1),
        "phi2": float(args.phi2),
        "etas": etas,
        "ntraj": int(args.ntraj),
        "chunk_size": int(args.chunk_size),
        "dt_T1": float(args.dt),
        "t_end_T1": float(args.t_end),
        "columns": columns,
    }

    npz_path = out_dir / f"homodyne_phi1={args.phi1:.6g}_phi2={args.phi2:.6g}.npz"
    save_ineff_df_npz(ineff_df, npz_path, meta=meta, step_col="step")

    plt.rcParams["text.usetex"] = True
    plt.rcParams.update(
        {
            "mathtext.fontset": "cm",
            "font.family": "serif",
            "font.size": 14,
            "axes.unicode_minus": False,
        }
    )

    phi1_tex = angle_to_tex(float(args.phi1))
    phi2_tex = angle_to_tex(float(args.phi2))

    x = times / t1
    for col in columns:
        plt.figure(figsize=(12, 8))

        for eta in etas:
            key = f"{eta:g}"
            y = ineff_df[key][col].to_numpy()
            line, = plt.plot(x, y, label=rf"$\eta = {eta:g}$")

            if col == "Conc" and not args.no_theory:
                y_th = theoretical_concurrence_curve(
                    times=times,
                    gamma=gamma,
                    eta=eta,
                    theta=theta_for_theory,
                )
                plt.plot(
                    x,
                    y_th,
                    "--",
                    color=line.get_color(),
                    alpha=0.55,
                    linewidth=2,
                )

        plt.xlim(0.0, float(args.t_end))
        plt.xlabel(r"$t/T_1$")
        plt.ylabel(LABELS[col])
        plt.title(
            r"Homodyne $J_z$ (N=2): "
            + LABELS[col]
            + rf"$,\ \phi_1={phi1_tex}\ \phi_2={phi2_tex}$"
            + rf"  (n$_\mathrm{{traj}}$ = {args.ntraj})"
        )
        plt.grid(True, linestyle=":", alpha=0.6)
        plt.legend(loc="upper left", bbox_to_anchor=(0, 0.95), fontsize=12)
        plt.savefig(out_dir / f"Homodyne_{col}.pdf", bbox_inches="tight")
        plt.close()

    (out_dir / "run_config.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Done. Results saved in: {out_dir}")


if __name__ == "__main__":
    main()
