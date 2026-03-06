"""
N=3 homodyne simulation for SLURM with spin-squeezing observables only (KU and Wineland).
Default initial state is |+++>, with optional GHZ.
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

import numpy as np
import pandas as pd
from qutip import expect, ket2dm, sigmaz

from N_Generic.Jz_N_Homodyne_Generic import (
    build_collapsing_ops_sigma_z,
    choose_unitary,
    collective_spin_ops,
    run_homodyne_avg_n,
    single_site_op,
    state_ghz_n,
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


def choose_initial_state(state_label: str, n_atoms: int):
    if state_label == "plusplus":
        return state_plus_n(n_atoms)
    if state_label == "ghz":
        return state_ghz_n(n_atoms)
    raise ValueError(f"Unknown state: {state_label}")


def make_parser():
    parser = argparse.ArgumentParser(description="N=3 homodyne SLURM run with KU/Wineland only")
    parser.add_argument("--state", choices=["plusplus", "ghz"], default="plusplus")
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
    parser.add_argument(
        "--eta-index",
        type=int,
        default=None,
        help="Run only one eta23 by index in --eta23-list (useful for Slurm job arrays)",
    )
    parser.add_argument(
        "--use-slurm-array",
        action="store_true",
        help="Read eta index from SLURM_ARRAY_TASK_ID if --eta-index is not set",
    )
    parser.add_argument("--tag", type=str, default="")
    parser.add_argument(
        "--out-root",
        type=str,
        default=r"./Graphs/Jz_N3_Homodyne_SLURM_spin_only",
    )
    return parser


def main():
    t0 = time.perf_counter()
    print("PY_START", flush=True)
    args = make_parser().parse_args()

    n_atoms = 3
    eta1 = float(args.eta1)
    if eta1 < 0.0 or eta1 > 1.0:
        raise ValueError("eta1 must be in [0,1].")

    eta23_all = parse_float_list(args.eta23_list)
    eta_index = args.eta_index
    if eta_index is None and args.use_slurm_array:
        slurm_idx = os.getenv("SLURM_ARRAY_TASK_ID", "").strip()
        if slurm_idx == "":
            raise ValueError("--use-slurm-array set but SLURM_ARRAY_TASK_ID is missing")
        eta_index = int(slurm_idx)

    if eta_index is not None:
        if eta_index < 0 or eta_index >= len(eta23_all):
            raise ValueError(
                f"Invalid eta index {eta_index}. Valid range: [0, {len(eta23_all) - 1}]"
            )
        eta23_list = [eta23_all[eta_index]]
        print(f"[array] Running eta index {eta_index} -> eta23={eta23_list[0]:g}")
    else:
        eta23_list = eta23_all

    gamma = float(args.gamma)
    t1 = 1.0 / gamma
    times = np.arange(0.0, float(args.t_end) * t1, float(args.dt) * t1)

    psi0 = choose_initial_state(args.state, n_atoms)
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
        f"N=3_state={args.state}_U={args.unitary_mode}_eta1={eta1:.3g}_eta23_scan"
        f"_phi1={phi1_tag}_phi2={phi2_tag}_phi3={phi3_tag}"
    )
    if eta_index is not None:
        out_name = f"{out_name}_eta_idx={eta_index}_eta23={eta23_list[0]:.3g}"
    if args.tag:
        out_name = f"{out_name}_{args.tag}"
    out_dir = Path(args.out_root) / out_name
    out_dir.mkdir(parents=True, exist_ok=True)

    for k, df in results.items():
        df.to_csv(out_dir / f"mean_spin_squeezing_eta23={k}.csv", index=False)

    save_ineff_df_npz(results, out_dir / "mean_spin_squeezing_all_eta23.npz", meta=None, step_col="step")

    runtime_sec = float(time.perf_counter() - t0)
    run_cfg = out_dir / "run_config.json"
    meta = {
        "measurement": "homodyne",
        "N": 3,
        "state": args.state,
        "unitary_mode": args.unitary_mode,
        "unitary_error": float(err),
        "gamma": gamma,
        "omega": float(args.omega),
        "eta1": eta1,
        "eta23_all": [float(x) for x in eta23_all],
        "eta23_run": [float(x) for x in eta23_list],
        "eta_index": eta_index,
        "use_slurm_array": bool(args.use_slurm_array),
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
        "labels_saved": list(results.keys()),
        "U_real": u.real.tolist(),
        "U_imag": u.imag.tolist(),
        "runtime_sec": runtime_sec,
        "out_dir": str(out_dir),
    }
    run_cfg.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"Done. Results saved in: {out_dir}")
    print(f"OUT_DIR={out_dir.resolve()}")
    print(f"RUN_CONFIG_PATH={run_cfg.resolve()}")
    print(f"PY_TOTAL_SECONDS={runtime_sec:.3f}", flush=True)


if __name__ == "__main__":
    main()

