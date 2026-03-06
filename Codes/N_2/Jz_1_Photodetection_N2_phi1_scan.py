"""
N=2 homodyne scan over phi1 (Cp phase), with phi2 fixed.

Goal:
- Initial state: |++>
- Collapse channels: Cp and Cm
- Efficiencies eta_plus = eta_minus = 1
- Scan phi1 in [0, pi/2] with >= 10 points
- Find the angle maximizing mean concurrence
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
from qutip import ket2dm, smesolve

from quantum_function2 import (
    Cm,
    Cp,
    H_free,
    PlusPlus,
    angle_to_path,
    angle_to_tex,
    concurrence_for_solver_general,
    save_ineff_df_npz,
)


def build_homodyne_channels(
    gamma: float,
    phi1: float,
    phi2: float,
    eta_plus: float,
    eta_minus: float,
):
    if not (0.0 <= eta_plus <= 1.0 and 0.0 <= eta_minus <= 1.0):
        raise ValueError("eta_plus and eta_minus must be in [0,1].")

    lp = np.sqrt(gamma) * np.exp(1j * phi1) * Cp
    lm = np.sqrt(gamma) * np.exp(1j * phi2) * Cm

    c_ops = []
    sc_ops = []

    if eta_plus < 1.0:
        c_ops.append(np.sqrt(1.0 - eta_plus) * lp)
    if eta_minus < 1.0:
        c_ops.append(np.sqrt(1.0 - eta_minus) * lm)

    if eta_plus > 0.0:
        sc_ops.append(np.sqrt(eta_plus) * lp)
    if eta_minus > 0.0:
        sc_ops.append(np.sqrt(eta_minus) * lm)

    return c_ops, sc_ops


def normalize_expect(expect_data, n_times: int):
    arr = np.asarray(expect_data)
    arr = np.real_if_close(arr)
    arr = np.asarray(arr, dtype=float)

    if arr.ndim == 1 and arr.shape[0] == n_times:
        return arr
    if arr.ndim == 2 and arr.shape == (1, n_times):
        return arr[0]

    raise ValueError(f"Unexpected expectation shape: {arr.shape}")


def run_phi_scan(
    times: np.ndarray,
    phi1_values: np.ndarray,
    gamma: float,
    phi2: float,
    eta_plus: float,
    eta_minus: float,
    ntraj: int,
    num_cpus: int,
    seed: int,
):
    options = {
        "keep_runs_results": False,
        "map": "parallel" if num_cpus > 1 else "serial",
        "num_cpus": max(1, num_cpus - 1),
    }

    curves = {}
    rng = np.random.default_rng(seed)
    rho0 = ket2dm(PlusPlus)

    for idx, phi1 in enumerate(phi1_values):
        np.random.seed(int(rng.integers(0, 2**31 - 1)))
        c_ops, sc_ops = build_homodyne_channels(
            gamma=float(gamma),
            phi1=float(phi1),
            phi2=float(phi2),
            eta_plus=float(eta_plus),
            eta_minus=float(eta_minus),
        )

        sol = smesolve(
            H_free,
            rho0,
            times,
            c_ops=c_ops,
            sc_ops=sc_ops,
            heterodyne=False,
            e_ops=[concurrence_for_solver_general],
            ntraj=int(ntraj),
            options=options,
        )

        conc = normalize_expect(sol.expect[0], n_times=len(times))
        key = f"{phi1:.10g}"
        curves[key] = pd.DataFrame(
            {
                "step": np.arange(len(times), dtype=int),
                "t": times,
                "Conc": conc,
            }
        )
        print(f"[{idx + 1}/{len(phi1_values)}] phi1={phi1:.6f} done")

    return curves


def summarize_phi_scan(curves: dict[str, pd.DataFrame], t1: float):
    rows = []
    for key, df in curves.items():
        phi1 = float(key)
        y = df["Conc"].to_numpy(dtype=float)
        t = df["t"].to_numpy(dtype=float)
        i_max = int(np.argmax(y))
        rows.append(
            {
                "phi1_rad": phi1,
                "phi1_over_pi": phi1 / np.pi,
                "C_max": float(y[i_max]),
                "t_at_C_max": float(t[i_max]),
                "t_at_C_max_over_T1": float(t[i_max] / t1),
                "C_final": float(y[-1]),
                "C_auc_over_tT1": float(np.trapezoid(y, x=t / t1)),
            }
        )

    summary = pd.DataFrame(rows).sort_values("phi1_rad", kind="stable").reset_index(drop=True)
    best_idx = int(summary["C_max"].idxmax())
    best = summary.loc[best_idx].to_dict()
    return summary, best


def make_parser():
    parser = argparse.ArgumentParser(description="N=2 homodyne scan over phi1 with phi2 fixed")
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--phi2", type=float, default=np.pi / 2.0, help="Fixed phase for Cm")
    parser.add_argument("--phi1-start", type=float, default=0.0)
    parser.add_argument("--phi1-stop", type=float, default=np.pi / 2.0)
    parser.add_argument("--n-phi1", type=int, default=11, help="Number of scan points (>=10 recommended)")
    parser.add_argument("--eta-plus", type=float, default=1.0)
    parser.add_argument("--eta-minus", type=float, default=1.0)
    parser.add_argument("--ntraj", type=int, default=1000)
    parser.add_argument("--t-end", type=float, default=10.0, help="Final time in units of T1")
    parser.add_argument("--dt", type=float, default=0.01, help="Step in units of T1")
    parser.add_argument("--num-cpus", type=int, default=max(1, os.cpu_count() - 1))
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--usetex", action="store_true")
    parser.add_argument("--tag", type=str, default="")
    parser.add_argument(
        "--out-root",
        type=str,
        default=r".\Codes\Graphs\Jz_2_Homodyne_N2_phi1_scan",
    )
    return parser


def main():
    t0 = time.perf_counter()
    args = make_parser().parse_args()

    if int(args.n_phi1) < 10:
        raise ValueError("--n-phi1 must be >= 10 for your requested scan density.")

    gamma = float(args.gamma)
    t1 = 1.0 / gamma
    times = np.arange(0.0, float(args.t_end) * t1, float(args.dt) * t1)
    phi1_values = np.linspace(float(args.phi1_start), float(args.phi1_stop), int(args.n_phi1))

    curves = run_phi_scan(
        times=times,
        phi1_values=phi1_values,
        gamma=gamma,
        phi2=float(args.phi2),
        eta_plus=float(args.eta_plus),
        eta_minus=float(args.eta_minus),
        ntraj=int(args.ntraj),
        num_cpus=int(args.num_cpus),
        seed=int(args.seed),
    )

    summary, best = summarize_phi_scan(curves=curves, t1=t1)

    phi2_dir = angle_to_path(float(args.phi2))
    out_name = f"state=plusplus_phi2={phi2_dir}_nphi={int(args.n_phi1)}"
    if args.tag:
        out_name = f"{out_name}_{args.tag}"
    out_dir = Path(args.out_root) / out_name
    out_dir.mkdir(parents=True, exist_ok=True)

    npz_path = out_dir / "phi1_scan_mean_concurrence.npz"
    save_ineff_df_npz(curves, npz_path, meta=None, step_col="step")

    summary_path = out_dir / "phi1_scan_summary.csv"
    summary.to_csv(summary_path, index=False)

    plt.rcParams["text.usetex"] = bool(args.usetex)
    plt.rcParams.update(
        {
            "mathtext.fontset": "cm",
            "font.family": "serif",
            "font.size": 14,
            "axes.unicode_minus": False,
        }
    )

    # Plot 1: concurrence(t) for all phi1
    x = times / t1
    plt.figure(figsize=(12, 8))
    for phi1 in phi1_values:
        key = f"{phi1:.10g}"
        y = curves[key]["Conc"].to_numpy()
        plt.plot(x, y, linewidth=1.8, label=rf"$\phi_1={phi1/np.pi:.3f}\pi$")

    plt.xlim(0.0, float(args.t_end))
    plt.ylim(-0.01, 1.02)
    plt.xlabel(r"$t/T_1$")
    plt.ylabel(r"$\overline{\mathcal{C}}$")
    plt.title(
        r"Homodyne $J_z$ (N=2, $|+\!+\rangle$): scan of $\phi_1$"
        + rf", $\phi_2={angle_to_tex(float(args.phi2))}$"
        + rf", $\eta_+={float(args.eta_plus):.2f},\eta_-={float(args.eta_minus):.2f}$"
        + rf"  (n$_\mathrm{{traj}}$={args.ntraj})"
    )
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(loc="upper left", bbox_to_anchor=(0, 0.95), fontsize=10, ncol=2)
    plt.savefig(out_dir / "Concurrence_vs_time_phi1_scan.pdf", bbox_inches="tight")
    plt.close()

    # Plot 2: C_max vs phi1
    plt.figure(figsize=(10, 6))
    plt.plot(summary["phi1_rad"], summary["C_max"], "o-", linewidth=2.0)
    plt.scatter([best["phi1_rad"]], [best["C_max"]], s=100, marker="*", zorder=5)
    plt.xlabel(r"$\phi_1$ [rad]")
    plt.ylabel(r"$\max_t \overline{\mathcal{C}}(t)$")
    plt.title(r"Best concurrence vs scanned $\phi_1$")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.savefig(out_dir / "Cmax_vs_phi1.pdf", bbox_inches="tight")
    plt.close()

    runtime_sec = float(time.perf_counter() - t0)
    meta = {
        "measurement": "homodyne",
        "N": 2,
        "state": "plusplus",
        "gamma": gamma,
        "phi2_fixed": float(args.phi2),
        "phi1_start": float(args.phi1_start),
        "phi1_stop": float(args.phi1_stop),
        "n_phi1": int(args.n_phi1),
        "phi1_values": [float(x) for x in phi1_values],
        "eta_plus": float(args.eta_plus),
        "eta_minus": float(args.eta_minus),
        "ntraj": int(args.ntraj),
        "dt_T1": float(args.dt),
        "t_end_T1": float(args.t_end),
        "num_cpus": int(args.num_cpus),
        "seed": int(args.seed),
        "best_by_C_max": best,
        "runtime_sec": runtime_sec,
        "out_dir": str(out_dir),
    }
    run_cfg = out_dir / "run_config.json"
    run_cfg.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print("")
    print("Best phi1 (by max mean concurrence):")
    print(f"  phi1 = {best['phi1_rad']:.8f} rad = {best['phi1_over_pi']:.6f} * pi")
    print(f"  C_max = {best['C_max']:.6f} at t/T1 = {best['t_at_C_max_over_T1']:.6f}")
    print("")
    print(f"Done. Results saved in: {out_dir}")
    print(f"OUT_DIR={out_dir.resolve()}")
    print(f"RUN_CONFIG_PATH={run_cfg.resolve()}")


if __name__ == "__main__":
    main()
