"""
Generic N-atom homodyne simulation with interferometric mixing of local sigma_z channels.

Features:
- Initial states for generic N: plusN, css, ghz.
- Collapse operators built from an NxN unitary applied to local sigma_z operators.
- Built-in unitaries: DFT(N), tritter (N=3), collective+differences (N=3).
- Observables: pairwise mean concurrence and Kitagawa-Ueda spin squeezing xi^2_KU.
- Chunked SME trajectories for better memory control.
"""

import argparse
import json
import os
import time
from itertools import combinations
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
from qutip import (
    basis,
    concurrence,
    expect,
    ket2dm,
    qeye,
    sigmax,
    sigmay,
    sigmaz,
    smesolve,
    tensor,
)

from quantum_function2 import save_ineff_df_npz

DEFAULT_EOPS = [
    ("Conc_pair_mean", "pairwise_mean_concurrence"),
    ("Xi2_KU", "xi_ku"),
    ("Xi2_W", "xi_wineland"),
]

LABELS = {
    "Conc_pair_mean": r"$\overline{\mathcal{C}}_{\mathrm{pair}}$",
    "Xi2_KU": r"$\overline{\xi^2_{KU}}$",
    "Xi2_W": r"$\overline{\xi^2_{W}}$",
}


def single_site_op(op, site: int, n_atoms: int):
    ops = [qeye(2) for _ in range(n_atoms)]
    ops[site] = op
    return tensor(ops)


def local_sigma_z_ops(n_atoms: int):
    return [single_site_op(sigmaz(), i, n_atoms) for i in range(n_atoms)]


def collective_spin_ops(n_atoms: int):
    jx = 0
    jy = 0
    jz = 0
    for i in range(n_atoms):
        jx += 0.5 * single_site_op(sigmax(), i, n_atoms)
        jy += 0.5 * single_site_op(sigmay(), i, n_atoms)
        jz += 0.5 * single_site_op(sigmaz(), i, n_atoms)
    return jx, jy, jz


def state_plus_n(n_atoms: int):
    g = basis(2, 0)
    e = basis(2, 1)
    plus = (g + e).unit()
    return tensor([plus for _ in range(n_atoms)])


def state_css_n(n_atoms: int, theta: float, phi: float):
    single = (
        np.sin(theta / 2.0) * basis(2, 0)
        + np.exp(1j * phi) * np.cos(theta / 2.0) * basis(2, 1)
    ).unit()
    return tensor([single for _ in range(n_atoms)])


def state_ghz_n(n_atoms: int):
    g = basis(2, 0)
    e = basis(2, 1)
    ket_g = tensor([g for _ in range(n_atoms)])
    ket_e = tensor([e for _ in range(n_atoms)])
    return (ket_g + ket_e).unit()


def dft_unitary(n_atoms: int):
    j = np.arange(n_atoms)
    k = np.arange(n_atoms)
    jj, kk = np.meshgrid(j, k, indexing="ij")
    return np.exp(2j * np.pi * jj * kk / n_atoms) / np.sqrt(n_atoms)


def tritter_unitary_3():
    return dft_unitary(3)


def collective_diff_unitary_3():
    return np.array(
        [
            [1 / np.sqrt(3), 1 / np.sqrt(3), 1 / np.sqrt(3)],
            [1 / np.sqrt(2), -1 / np.sqrt(2), 0],
            [1 / np.sqrt(6), 1 / np.sqrt(6), -2 / np.sqrt(6)],
        ],
        dtype=complex,
    )


def choose_unitary(n_atoms: int, mode: str):
    if mode == "dft":
        return dft_unitary(n_atoms)
    if mode == "tritter":
        if n_atoms != 3:
            raise ValueError("'tritter' is defined only for N=3")
        return tritter_unitary_3()
    if mode == "collective_diff_3":
        if n_atoms != 3:
            raise ValueError("'collective_diff_3' is defined only for N=3")
        return collective_diff_unitary_3()
    raise ValueError(f"Unknown unitary mode: {mode}")


def unitary_error(u: np.ndarray):
    n_atoms = u.shape[0]
    return float(np.linalg.norm(u.conj().T @ u - np.eye(n_atoms)))


def build_collapsing_ops_sigma_z(n_atoms: int, gamma: float, u: np.ndarray):
    if u.shape != (n_atoms, n_atoms):
        raise ValueError(f"U must be {n_atoms}x{n_atoms}, got {u.shape}")

    local_ops = local_sigma_z_ops(n_atoms)
    c_ops = []
    for k in range(n_atoms):
        ck = 0
        for j in range(n_atoms):
            ck += u[k, j] * local_ops[j]
        c_ops.append(np.sqrt(gamma) * ck)
    return c_ops


def split_observed_unobserved(collapse_ops: list, eta_raw):
    n_atoms = len(collapse_ops)
    eta_vec = np.asarray(eta_raw, dtype=float)
    if eta_vec.ndim == 0:
        eta_vec = np.full(n_atoms, float(eta_vec))
    if eta_vec.shape[0] != n_atoms:
        raise ValueError(
            f"eta must be scalar or length {n_atoms}, got shape {eta_vec.shape}"
        )
    if np.any(eta_vec < 0.0) or np.any(eta_vec > 1.0):
        raise ValueError("All eta values must be in [0,1]")

    c_unobs = [np.sqrt(1.0 - eta_vec[k]) * collapse_ops[k] for k in range(n_atoms)]
    c_obs = [np.sqrt(eta_vec[k]) * collapse_ops[k] for k in range(n_atoms)]
    return c_unobs, c_obs, eta_vec


def concurrence_pairwise_mean(state, n_atoms: int):
    if n_atoms < 2:
        return 0.0

    rho = ket2dm(state) if state.isket else state
    vals = []
    for i, j in combinations(range(n_atoms), 2):
        rho_ij = rho.ptrace([i, j])
        vals.append(float(np.real_if_close(concurrence(rho_ij))))
    return float(np.mean(vals))


def xi_ku_general(state, jx, jy, jz, n_atoms: int, tol: float = 1e-12):
    jx_exp = float(np.real_if_close(expect(jx, state)))
    jy_exp = float(np.real_if_close(expect(jy, state)))
    jz_exp = float(np.real_if_close(expect(jz, state)))
    jmean = np.array([jx_exp, jy_exp, jz_exp], dtype=float)
    m = np.linalg.norm(jmean)

    js = [jx, jy, jz]
    cov = np.zeros((3, 3), dtype=float)
    for i, ji in enumerate(js):
        for j, jj in enumerate(js):
            cov[i, j] = float(
                np.real_if_close(
                    0.5 * expect(ji * jj + jj * ji, state) - jmean[i] * jmean[j]
                )
            )

    if m > tol:
        uvec = jmean / m
        a = np.array([1.0, 0.0, 0.0]) if abs(uvec[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        e1 = a - uvec * np.dot(uvec, a)
        e1 /= np.linalg.norm(e1)
        e2 = np.cross(uvec, e1)

        cov2 = np.array(
            [
                [e1 @ cov @ e1, e1 @ cov @ e2],
                [e2 @ cov @ e1, e2 @ cov @ e2],
            ],
            dtype=float,
        )
        lam_min = float(np.linalg.eigvalsh(cov2)[0])
    else:
        lam_min = float(np.linalg.eigvalsh(cov)[0])

    lam_min = max(lam_min, 0.0)
    return float((4.0 / n_atoms) * lam_min)


def j_mean_norm(state, jx, jy, jz):
    jx_exp = float(np.real_if_close(expect(jx, state)))
    jy_exp = float(np.real_if_close(expect(jy, state)))
    jz_exp = float(np.real_if_close(expect(jz, state)))
    return float(np.linalg.norm([jx_exp, jy_exp, jz_exp]))


def xi_w_general(state, jx, jy, jz, n_atoms: int, tol: float = 1e-12):
    xi2_ku = xi_ku_general(state, jx, jy, jz, n_atoms=n_atoms, tol=tol)
    jnorm = j_mean_norm(state, jx, jy, jz)
    if jnorm <= tol:
        return float("nan")
    return float(xi2_ku * (n_atoms / (2.0 * jnorm)) ** 2)


def make_observables(n_atoms: int):
    jx, jy, jz = collective_spin_ops(n_atoms)

    def conc_eop(_t, state):
        return concurrence_pairwise_mean(state, n_atoms)

    def xi_eop(_t, state):
        return xi_ku_general(state, jx, jy, jz, n_atoms)

    def xiw_eop(_t, state):
        return xi_w_general(state, jx, jy, jz, n_atoms)

    return [conc_eop, xi_eop, xiw_eop]


def normalize_expect_chunk(sol_expect, n_eops: int, n_times: int):
    arr = np.asarray(sol_expect)
    arr = np.real_if_close(arr)
    arr = np.asarray(arr, dtype=float)

    if arr.ndim == 2 and arr.shape == (n_eops, n_times):
        return arr

    if arr.ndim == 3:
        if arr.shape[0] == n_eops and arr.shape[2] == n_times:
            return arr.mean(axis=1)
        if arr.shape[1] == n_eops and arr.shape[2] == n_times:
            return arr.mean(axis=0)

    if n_eops == 1 and arr.ndim == 1 and arr.shape[0] == n_times:
        return arr.reshape(1, -1)

    raise ValueError(f"Unexpected expect shape {arr.shape}")


def run_homodyne_avg_n(
    rho0,
    times,
    hamiltonian,
    collapse_ops,
    eta,
    e_ops,
    ntraj: int,
    chunk_size: int,
    num_cpus: int,
    seed: int,
):
    options = {
        "keep_runs_results": False,
        "num_cpus": max(1, num_cpus - 1),
        "map": "parallel" if num_cpus > 1 else "serial",
    }

    n_eops = len(e_ops)
    n_times = len(times)
    weighted = None
    done = 0
    chunk_id = 0
    rng = np.random.default_rng(seed)

    c_unobs, c_obs, eta_vec = split_observed_unobserved(collapse_ops, eta)

    while done < ntraj:
        chunk_id += 1
        n_this = min(chunk_size, ntraj - done)
        np.random.seed(int(rng.integers(0, 2**31 - 1)))

        sol = smesolve(
            hamiltonian,
            rho0,
            times,
            c_ops=c_unobs,
            sc_ops=c_obs,
            heterodyne=False,
            e_ops=e_ops,
            ntraj=n_this,
            options=options,
        )

        mean_chunk = normalize_expect_chunk(sol.expect, n_eops=n_eops, n_times=n_times)
        if weighted is None:
            weighted = mean_chunk * n_this
        else:
            weighted += mean_chunk * n_this

        done += n_this
        print(f"chunk {chunk_id}: {done}/{ntraj} trajectories")

    return weighted / float(ntraj), eta_vec


def parse_eta(raw: str, n_atoms: int):
    if "," not in raw:
        return float(raw)
    vals = np.array([float(x.strip()) for x in raw.split(",") if x.strip()], dtype=float)
    if len(vals) != n_atoms:
        raise ValueError(f"eta list must have exactly N={n_atoms} values")
    return vals


def parse_states(raw: str):
    labels = [x.strip() for x in raw.split(",") if x.strip()]
    if not labels:
        raise ValueError("states list cannot be empty")
    return labels


def build_initial_state(label: str, n_atoms: int, theta: float, phi_state: float):
    if label == "plusN":
        return state_plus_n(n_atoms), {"kind": "plusN"}
    if label == "css_pi_3":
        return state_css_n(n_atoms, theta=np.pi / 3.0, phi=0.0), {
            "kind": "css",
            "theta": float(np.pi / 3.0),
            "phi": 0.0,
        }
    if label == "css":
        return state_css_n(n_atoms, theta=theta, phi=phi_state), {
            "kind": "css",
            "theta": float(theta),
            "phi": float(phi_state),
        }
    if label == "ghz":
        return state_ghz_n(n_atoms), {"kind": "ghz"}
    raise ValueError(f"Unknown state label: {label}")


def eta_to_tag(eta_vec: np.ndarray):
    return "_".join(f"{x:.3g}" for x in eta_vec)


def make_parser():
    parser = argparse.ArgumentParser(description="Generic N homodyne simulation with interferometer unitary")
    parser.add_argument("--N", type=int, default=3)
    parser.add_argument(
        "--unitary-mode",
        choices=["dft", "tritter", "collective_diff_3"],
        default="collective_diff_3",
    )
    parser.add_argument(
        "--states",
        type=str,
        default="plusN,css_pi_3",
        help="Comma-separated among: plusN,css_pi_3,css,ghz",
    )
    parser.add_argument("--theta", type=float, default=np.pi / 3.0, help="Used when 'css' is in --states")
    parser.add_argument("--phi-state", type=float, default=0.0, help="Used when 'css' is in --states")
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--omega", type=float, default=0.0)
    parser.add_argument(
        "--eta",
        type=str,
        default="1.0",
        help="Either scalar (e.g. 1.0) or comma-list of N values (e.g. 1,0.8,0.6)",
    )
    parser.add_argument("--ntraj", type=int, default=400)
    parser.add_argument("--chunk-size", type=int, default=100)
    parser.add_argument("--t-end", type=float, default=10.0, help="Final time in units of T1")
    parser.add_argument("--dt", type=float, default=0.02, help="Time step in units of T1")
    parser.add_argument("--num-cpus", type=int, default=max(1, os.cpu_count() - 1))
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--usetex", action="store_true")
    parser.add_argument("--tag", type=str, default="")
    parser.add_argument("--skip-plots", action="store_true")
    parser.add_argument(
        "--out-root",
        type=str,
        default=r".\Codes\Graphs\Jz_N_Homodyne_Generic",
    )
    return parser


def main():
    t0 = time.perf_counter()
    args = make_parser().parse_args()

    n_atoms = int(args.N)
    if n_atoms < 2:
        raise ValueError("N must be >= 2 for concurrence.")

    gamma = float(args.gamma)
    t1 = 1.0 / gamma
    times = np.arange(0.0, float(args.t_end) * t1, float(args.dt) * t1)

    eta_input = parse_eta(args.eta, n_atoms=n_atoms)
    state_labels = parse_states(args.states)

    u = choose_unitary(n_atoms, args.unitary_mode)
    err = unitary_error(u)
    collapse_ops = build_collapsing_ops_sigma_z(n_atoms=n_atoms, gamma=gamma, u=u)

    hamiltonian = 0
    for i in range(n_atoms):
        hamiltonian += 0.5 * float(args.omega) * single_site_op(sigmaz(), i, n_atoms)

    e_ops = make_observables(n_atoms)
    columns = [name for name, _ in DEFAULT_EOPS]

    results = {}
    state_meta = {}
    eta_used = None
    for idx, label in enumerate(state_labels):
        psi0, meta_s = build_initial_state(
            label=label,
            n_atoms=n_atoms,
            theta=float(args.theta),
            phi_state=float(args.phi_state),
        )
        rho0 = ket2dm(psi0)

        mean_arr, eta_vec = run_homodyne_avg_n(
            rho0=rho0,
            times=times,
            hamiltonian=hamiltonian,
            collapse_ops=collapse_ops,
            eta=eta_input,
            e_ops=e_ops,
            ntraj=int(args.ntraj),
            chunk_size=int(args.chunk_size),
            num_cpus=int(args.num_cpus),
            seed=int(args.seed) + 1000 * idx,
        )

        df = pd.DataFrame(mean_arr.T, columns=columns)
        df.insert(0, "step", np.arange(len(times), dtype=int))
        df.insert(1, "t", times)
        df.insert(2, "t_T1", times / t1)
        results[label] = df
        state_meta[label] = meta_s
        eta_used = eta_vec

    if eta_used is None:
        raise RuntimeError("No simulations executed.")

    eta_tag = eta_to_tag(np.asarray(eta_used, dtype=float))
    base_name = f"N={n_atoms}_U={args.unitary_mode}_eta={eta_tag}"
    if args.tag:
        base_name = f"{base_name}_{args.tag}"
    out_dir = Path(args.out_root) / base_name
    out_dir.mkdir(parents=True, exist_ok=True)

    for label, df in results.items():
        df.to_csv(out_dir / f"mean_observables_{label}.csv", index=False)

    npz_path = out_dir / "mean_observables_all_states.npz"
    save_ineff_df_npz(results, npz_path, meta=None, step_col="step")

    if not args.skip_plots:
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
            for label, df in results.items():
                plt.plot(x, df[col].to_numpy(), linewidth=2.0, label=label)

            if col == "Xi2_KU":
                plt.axhline(1.0, color="k", linestyle="--", linewidth=1.2, alpha=0.6)

            plt.xlim(0.0, float(args.t_end))
            plt.xlabel(r"$t/T_1$")
            plt.ylabel(LABELS[col])
            plt.title(
                rf"Homodyne $J_z$ generic N={n_atoms}, U={args.unitary_mode}, "
                rf"n$_\mathrm{{traj}}$={args.ntraj}"
            )
            plt.grid(True, linestyle=":", alpha=0.6)
            plt.legend(loc="best")
            plt.savefig(out_dir / f"{col}.pdf", bbox_inches="tight")
            plt.close()

    runtime_sec = float(time.perf_counter() - t0)
    run_cfg = out_dir / "run_config.json"
    meta = {
        "measurement": "homodyne",
        "N": n_atoms,
        "unitary_mode": args.unitary_mode,
        "unitary_error": float(err),
        "gamma": gamma,
        "omega": float(args.omega),
        "eta_vector": [float(x) for x in np.asarray(eta_used, dtype=float)],
        "ntraj": int(args.ntraj),
        "chunk_size": int(args.chunk_size),
        "dt_T1": float(args.dt),
        "t_end_T1": float(args.t_end),
        "num_cpus": int(args.num_cpus),
        "seed": int(args.seed),
        "states": state_meta,
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
