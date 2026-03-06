"""
Homodyne spin-squeezing scaling versus N in collective (symmetric) spin-J representation.

Model used to reach large N:
- Symmetric subspace, dimension = N + 1
- Initial state: spin coherent state along +x (equivalent to |+>^{⊗N} in symmetric subspace)
- Homodyne measurement of collective sum channel only (phase = 0)
- Efficiency fixed to 1

Outputs:
- Mean trajectories Xi2_KU(t), Xi2_W(t), |<J>|(t) for each N
- Summary table with minima versus N
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
from qutip import expect, jmat, ket2dm, smesolve, spin_coherent

from Jz_N_Homodyne_Generic import xi_ku_general, xi_w_general


def default_n_values():
    vals = np.linspace(2, 100, 10)
    out = sorted(set(int(round(x)) for x in vals))
    if out[0] != 2:
        out[0] = 2
    if out[-1] != 100:
        out[-1] = 100
    return out


def parse_n_list(raw: str):
    if not raw.strip():
        return default_n_values()
    vals = [int(x.strip()) for x in raw.split(",") if x.strip()]
    if not vals:
        raise ValueError("Empty N list.")
    if any(n < 2 for n in vals):
        raise ValueError("All N must be >= 2.")
    return sorted(set(vals))


def j_mean_norm(state, jx, jy, jz):
    mx = float(np.real_if_close(expect(jx, state)))
    my = float(np.real_if_close(expect(jy, state)))
    mz = float(np.real_if_close(expect(jz, state)))
    return float(np.linalg.norm([mx, my, mz]))


def run_single_n(
    n_atoms: int,
    times: np.ndarray,
    gamma: float,
    ntraj: int,
    num_cpus: int,
    seed: int,
):
    j = n_atoms / 2.0
    jx = jmat(j, "x")
    jy = jmat(j, "y")
    jz = jmat(j, "z")

    psi0 = spin_coherent(j, np.pi / 2.0, 0.0)
    rho0 = ket2dm(psi0)

    # Same normalization as "sum mode" built from local sigmaz:
    # c_sum = sqrt(gamma/N) * sum_i sigma_z^(i) = 2*sqrt(gamma/N) * Jz
    c_sum = 2.0 * np.sqrt(gamma / n_atoms) * jz

    def ku_eop(_t, state):
        return xi_ku_general(state, jx, jy, jz, n_atoms)

    def w_eop(_t, state):
        return xi_w_general(state, jx, jy, jz, n_atoms)

    def jnorm_eop(_t, state):
        return j_mean_norm(state, jx, jy, jz)

    options = {
        "keep_runs_results": False,
        "map": "parallel" if num_cpus > 1 else "serial",
        "num_cpus": max(1, num_cpus - 1),
    }

    np.random.seed(seed)
    sol = smesolve(
        0 * jz,
        rho0,
        times,
        c_ops=[],
        sc_ops=[c_sum],
        heterodyne=False,
        e_ops=[ku_eop, w_eop, jnorm_eop],
        ntraj=int(ntraj),
        options=options,
    )

    arr = np.asarray(sol.expect, dtype=float)
    if arr.ndim != 2 or arr.shape[0] != 3:
        raise ValueError(f"Unexpected expect shape for N={n_atoms}: {arr.shape}")

    df = pd.DataFrame(
        {
            "step": np.arange(len(times), dtype=int),
            "t": times,
            "Xi2_KU": arr[0],
            "Xi2_W": arr[1],
            "J_norm": arr[2],
        }
    )
    return df


def make_parser():
    parser = argparse.ArgumentParser(description="Homodyne spin-squeezing scaling vs N (collective model)")
    parser.add_argument(
        "--n-list",
        type=str,
        default="",
        help="Comma-separated N list. Empty -> 10 points from 2 to 100.",
    )
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--ntraj", type=int, default=50)
    parser.add_argument("--t-end", type=float, default=10.0, help="in units of T1")
    parser.add_argument("--dt", type=float, default=0.02, help="in units of T1")
    parser.add_argument("--num-cpus", type=int, default=1)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--usetex", action="store_true")
    parser.add_argument("--tag", type=str, default="")
    parser.add_argument(
        "--out-root",
        type=str,
        default=r".\Codes\Graphs\Jz_N_Homodyne_scaling_spin_only",
    )
    return parser


def main():
    t0 = time.perf_counter()
    args = make_parser().parse_args()

    n_values = parse_n_list(args.n_list)
    gamma = float(args.gamma)
    t1 = 1.0 / gamma
    times = np.arange(0.0, float(args.t_end) * t1, float(args.dt) * t1)

    out_name = f"Nscan_{n_values[0]}_{n_values[-1]}_count={len(n_values)}_ntraj={int(args.ntraj)}"
    if args.tag:
        out_name = f"{out_name}_{args.tag}"
    out_dir = Path(args.out_root) / out_name
    out_dir.mkdir(parents=True, exist_ok=True)

    curves = {}
    summary_rows = []
    for i, n_atoms in enumerate(n_values, start=1):
        print(f"[{i}/{len(n_values)}] Running N={n_atoms} ...")
        df = run_single_n(
            n_atoms=n_atoms,
            times=times,
            gamma=gamma,
            ntraj=int(args.ntraj),
            num_cpus=int(args.num_cpus),
            seed=int(args.seed + 1000 * i),
        )
        df["t_T1"] = df["t"] / t1
        curves[n_atoms] = df
        df.to_csv(out_dir / f"mean_spin_squeezing_N={n_atoms}.csv", index=False)

        i_ku = int(np.nanargmin(df["Xi2_KU"].to_numpy(dtype=float)))
        ku_min = float(df.loc[i_ku, "Xi2_KU"])
        t_ku = float(df.loc[i_ku, "t_T1"])

        w_arr = df["Xi2_W"].to_numpy(dtype=float)
        finite_mask = np.isfinite(w_arr)
        if np.any(finite_mask):
            idx_candidates = np.where(finite_mask)[0]
            i_w_local = int(np.argmin(w_arr[finite_mask]))
            i_w = int(idx_candidates[i_w_local])
            w_min = float(df.loc[i_w, "Xi2_W"])
            t_w = float(df.loc[i_w, "t_T1"])
        else:
            w_min = float("nan")
            t_w = float("nan")

        summary_rows.append(
            {
                "N": int(n_atoms),
                "Xi2_KU_min": ku_min,
                "t_T1_at_Xi2_KU_min": t_ku,
                "Xi2_W_min": w_min,
                "t_T1_at_Xi2_W_min": t_w,
                "J_norm_at_KU_min": float(df.loc[i_ku, "J_norm"]),
                "J_norm_final": float(df["J_norm"].iloc[-1]),
            }
        )

    summary = pd.DataFrame(summary_rows).sort_values("N").reset_index(drop=True)
    summary.to_csv(out_dir / "spin_squeezing_vs_N_summary.csv", index=False)

    plt.rcParams["text.usetex"] = bool(args.usetex)
    plt.rcParams.update(
        {
            "mathtext.fontset": "cm",
            "font.family": "serif",
            "font.size": 14,
            "axes.unicode_minus": False,
        }
    )

    plt.figure(figsize=(12, 8))
    plt.plot(summary["N"], summary["Xi2_KU_min"], "o-", linewidth=2.0, label=r"$\min_t\,\overline{\xi^2_{KU}}$")
    plt.plot(summary["N"], summary["Xi2_W_min"], "s-", linewidth=2.0, label=r"$\min_t\,\overline{\xi^2_{W}}$")
    plt.axhline(1.0, color="k", linestyle="--", linewidth=1.2, alpha=0.6)
    plt.xlabel(r"$N$")
    plt.ylabel(r"Squeezing")
    plt.title(rf"Homodyne spin-squeezing scaling vs N (n$_\mathrm{{traj}}$={args.ntraj})")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(loc="best")
    plt.savefig(out_dir / "spin_squeezing_min_vs_N.pdf", bbox_inches="tight")
    plt.close()

    runtime_sec = float(time.perf_counter() - t0)
    meta = {
        "measurement": "homodyne",
        "model": "collective_symmetric_subspace",
        "state": "plusN",
        "gamma": gamma,
        "eta": 1.0,
        "phi": 0.0,
        "n_values": [int(x) for x in n_values],
        "ntraj": int(args.ntraj),
        "dt_T1": float(args.dt),
        "t_end_T1": float(args.t_end),
        "num_cpus": int(args.num_cpus),
        "seed": int(args.seed),
        "runtime_sec": runtime_sec,
        "out_dir": str(out_dir),
    }
    run_cfg = out_dir / "run_config.json"
    run_cfg.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"Done. Results saved in: {out_dir}")
    print(f"OUT_DIR={out_dir.resolve()}")
    print(f"RUN_CONFIG_PATH={run_cfg.resolve()}")


if __name__ == "__main__":
    main()
