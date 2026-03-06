"""
N=3 homodyne scan with spin-squeezing observables only (KU and Wineland).

Setup:
- Initial state: |+++>
- Interferometer unitary: collective_diff_3 by default (first output mode = spin-sum channel)
- Efficiencies: eta1 fixed, eta2 = eta3 scanned over user list
- Output-mode phases: phi1, phi2, phi3 (applied to channels 1,2,3)
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
from qutip import expect, ket2dm, sigmaz

from Jz_N_Homodyne_Generic import (
    build_collapsing_ops_sigma_z,
    choose_unitary,
    collective_spin_ops,
    run_homodyne_avg_n,
    single_site_op,
    state_plus_n,
    unitary_error,
    xi_ku_general,
    xi_w_general,
)
from quantum_function2 import angle_to_path, save_ineff_df_npz


LABELS = {
    "Xi2_KU": r"$\overline{\xi^2_{KU}}$",
    "Xi2_W": r"$\overline{\xi^2_{W}}$",
}


def parse_float_list(raw: str):
    vals = [float(x.strip()) for x in raw.split(",") if x.strip()]
    if not vals:
        raise ValueError("Empty eta23 list.")
    if any((x < 0.0 or x > 1.0) for x in vals):
        raise ValueError("All eta23 values must be in [0,1].")
    return vals


def fmt_eta(x: float):
    return f"{x:.3g}"


def apply_output_phases(collapse_ops: list, phases: list[float]):
    if len(collapse_ops) != len(phases):
        raise ValueError("collapse_ops and phases must have same length.")
    return [np.exp(1j * float(phases[k])) * collapse_ops[k] for k in range(len(collapse_ops))]


def make_spin_only_observables(n_atoms: int):
    jx, jy, jz = collective_spin_ops(n_atoms)

    def ku_eop(_t, state):
        return xi_ku_general(state, jx, jy, jz, n_atoms)

    def w_eop(_t, state):
        return xi_w_general(state, jx, jy, jz, n_atoms)

    return [ku_eop, w_eop], ["Xi2_KU", "Xi2_W"]


def make_parser():
    parser = argparse.ArgumentParser(description="N=3 homodyne eta2=eta3 scan with KU/Wineland only")
    parser.add_argument("--unitary-mode", choices=["collective_diff_3", "tritter", "dft"], default="collective_diff_3")
    parser.add_argument("--eta1", type=float, default=1.0, help="Efficiency of first (spin-sum) channel")
    parser.add_argument("--eta23-list", type=str, default="0,0.3,0.5,0.9,1")
    parser.add_argument("--phi1", type=float, default=0.0, help="Phase on output channel 1")
    parser.add_argument("--phi2", type=float, default=0.0, help="Phase on output channel 2")
    parser.add_argument("--phi3", type=float, default=0.0, help="Phase on output channel 3")
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--omega", type=float, default=0.0)
    parser.add_argument("--ntraj", type=int, default=100)
    parser.add_argument("--chunk-size", type=int, default=100)
    parser.add_argument("--t-end", type=float, default=10.0, help="Final time in units of T1")
    parser.add_argument("--dt", type=float, default=0.02, help="Step in units of T1")
    parser.add_argument("--num-cpus", type=int, default=max(1, os.cpu_count() - 1))
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--usetex", action="store_true")
    parser.add_argument("--tag", type=str, default="")
    parser.add_argument(
        "--out-root",
        type=str,
        default=r".\Codes\Graphs\Jz_N3_Homodyne_eta23_scan_spin_only",
    )
    return parser


def main():
    t0 = time.perf_counter()
    args = make_parser().parse_args()

    n_atoms = 3
    eta1 = float(args.eta1)
    if eta1 < 0.0 or eta1 > 1.0:
        raise ValueError("eta1 must be in [0,1].")

    eta23_list = parse_float_list(args.eta23_list)

    gamma = float(args.gamma)
    t1 = 1.0 / gamma
    times = np.arange(0.0, float(args.t_end) * t1, float(args.dt) * t1)

    psi0 = state_plus_n(n_atoms)
    rho0 = ket2dm(psi0)

    u = choose_unitary(n_atoms, args.unitary_mode)
    err = unitary_error(u)
    collapse_ops = build_collapsing_ops_sigma_z(n_atoms=n_atoms, gamma=gamma, u=u)
    collapse_ops = apply_output_phases(collapse_ops, [float(args.phi1), float(args.phi2), float(args.phi3)])

    hamiltonian = 0
    for i in range(n_atoms):
        hamiltonian += 0.5 * float(args.omega) * single_site_op(sigmaz(), i, n_atoms)

    e_ops, columns = make_spin_only_observables(n_atoms)

    results = {}
    for idx, eta23 in enumerate(eta23_list):
        eta_vec = [eta1, float(eta23), float(eta23)]
        mean_arr, eta_used = run_homodyne_avg_n(
            rho0=rho0,
            times=times,
            hamiltonian=hamiltonian,
            collapse_ops=collapse_ops,
            eta=eta_vec,
            e_ops=e_ops,
            ntraj=int(args.ntraj),
            chunk_size=int(args.chunk_size),
            num_cpus=int(args.num_cpus),
            seed=int(args.seed) + 1000 * idx,
        )

        key = fmt_eta(eta23)
        df = pd.DataFrame(mean_arr.T, columns=columns)
        df.insert(0, "step", np.arange(len(times), dtype=int))
        df.insert(1, "t", times)
        df.insert(2, "t_T1", times / t1)
        results[key] = df
        print(f"eta23={eta23:.3f} completed with eta vector {eta_used}")

    phi1_tag = angle_to_path(float(args.phi1))
    phi2_tag = angle_to_path(float(args.phi2))
    phi3_tag = angle_to_path(float(args.phi3))
    out_name = (
        f"N=3_U={args.unitary_mode}_eta1={eta1:.3g}_eta23_scan"
        f"_phi1={phi1_tag}_phi2={phi2_tag}_phi3={phi3_tag}"
    )
    if args.tag:
        out_name = f"{out_name}_{args.tag}"
    out_dir = Path(args.out_root) / out_name
    out_dir.mkdir(parents=True, exist_ok=True)

    for k, df in results.items():
        df.to_csv(out_dir / f"mean_spin_squeezing_eta23={k}.csv", index=False)

    save_ineff_df_npz(results, out_dir / "mean_spin_squeezing_all_eta23.npz", meta=None, step_col="step")

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
        for eta23 in eta23_list:
            key = fmt_eta(eta23)
            y = results[key][col].to_numpy()
            plt.plot(x, y, linewidth=2.0, label=rf"$\eta_2=\eta_3={eta23:.2f}$")

        if col in {"Xi2_KU", "Xi2_W"}:
            plt.axhline(1.0, color="k", linestyle="--", linewidth=1.2, alpha=0.6)

        plt.xlim(0.0, float(args.t_end))
        plt.xlabel(r"$t/T_1$")
        plt.ylabel(LABELS[col])
        plt.title(
            rf"N=3 homodyne $|+\!+\!+\rangle$, "
            rf"$\eta_1={eta1:.2f}$, $U={args.unitary_mode}$, "
            rf"$\phi_1={float(args.phi1):.2f},\phi_2={float(args.phi2):.2f},\phi_3={float(args.phi3):.2f}$, "
            rf"n$_\mathrm{{traj}}$={args.ntraj}"
        )
        plt.grid(True, linestyle=":", alpha=0.6)
        plt.legend(loc="best")
        plt.savefig(out_dir / f"{col}_eta23_scan.pdf", bbox_inches="tight")
        plt.close()

    runtime_sec = float(time.perf_counter() - t0)
    run_cfg = out_dir / "run_config.json"
    meta = {
        "measurement": "homodyne",
        "N": 3,
        "state": "plusN",
        "unitary_mode": args.unitary_mode,
        "unitary_error": float(err),
        "gamma": gamma,
        "omega": float(args.omega),
        "eta1": eta1,
        "eta23_list": [float(x) for x in eta23_list],
        "phi1": float(args.phi1),
        "phi2": float(args.phi2),
        "phi3": float(args.phi3),
        "ntraj": int(args.ntraj),
        "chunk_size": int(args.chunk_size),
        "dt_T1": float(args.dt),
        "t_end_T1": float(args.t_end),
        "num_cpus": int(args.num_cpus),
        "seed": int(args.seed),
        "columns": columns,
        "U_real": u.real.tolist(),
        "U_imag": u.imag.tolist(),
        "runtime_sec": runtime_sec,
        "out_dir": str(out_dir),
    }
    run_cfg.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"Done. Results saved in: {out_dir}")
    print(f"OUT_DIR={out_dir.resolve()}")
    print(f"RUN_CONFIG_PATH={run_cfg.resolve()}")


if __name__ == "__main__":
    main()
