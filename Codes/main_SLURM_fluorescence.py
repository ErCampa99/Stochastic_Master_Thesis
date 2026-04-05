"""
Homodyne fluorescence simulation for N=2 atoms.
This script simulates homodyne detection of collective spontaneous emission for two atoms (N=2).
Convention: |0> = excited state, |1> = ground state.
The collective lowering operators (fluorescence channels) are mixed interferometrically
into C_plus and C_minus channels, exactly as in the dephasing case but with sigma_minus
replacing sigma_z.
Computes: Concurrence, xi^2_KU, |<J>|, xi^2_WIN (post-processed from KU and norm).
"""

import ast
import argparse
import json
import os
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

try:
    from joblib import Parallel, delayed
except ImportError:
    Parallel = None
    delayed = None

from qutip import (
    basis,
    expect,
    ket2dm,
    qeye,
    sigmax,
    sigmay,
    sigmaz,
    sigmam,
    smesolve,
    tensor,
    variance,
)


# ── Basis states ──────────────────────────────────────────────────────────────
# Convention for fluorescence: |0> = excited, |1> = ground state.
exc = basis(2, 0)   # |e> = |0>
gnd = basis(2, 1)   # |g> = |1>

# Superposition state (|e> + |g>) / sqrt(2)
plus_state = (exc + gnd).unit()
PlusPlus   = tensor(plus_state, plus_state)   # |++>
ee         = tensor(exc, exc)                 # |ee> — both excited (natural for fluorescence)


# Function to create a coherent spin state (CSS) for N=2 atoms.
def css_2(theta: float, phi: float):
    single_qubit = (
        np.sin(theta / 2.0) * exc
        + np.exp(1j * phi) * np.cos(theta / 2.0) * gnd
    )
    return tensor(single_qubit, single_qubit)


# ── Collective spin operators for N=2 ────────────────────────────────────────
I_2 = qeye(2)
J_x = 0.5 * (tensor(sigmax(), I_2) + tensor(I_2, sigmax()))
J_y = 0.5 * (tensor(sigmay(), I_2) + tensor(I_2, sigmay()))
J_z = 0.5 * (tensor(sigmaz(), I_2) + tensor(I_2, sigmaz()))

# Free Hamiltonian — zero (rotating frame, no driving)
H_free = 0.0 * J_z

# ── Fluorescence channels — interferometric mix of sigma_minus ────────────────
# sigma_minus = sigmam() takes |0>(excited) -> |1>(ground).
# C_plus  = (sigma_minus_1 ⊗ I  +  I ⊗ sigma_minus_2) / sqrt(2)
# C_minus = (sigma_minus_1 ⊗ I  -  I ⊗ sigma_minus_2) / sqrt(2)
Cp = np.sqrt(0.5) * (tensor(sigmam(), I_2) + tensor(I_2, sigmam()))
Cm = np.sqrt(0.5) * (tensor(sigmam(), I_2) - tensor(I_2, sigmam()))


def collapsing_operators(gamma: float, phi_1: float, phi_2: float, eta_1: float, eta_2: float):
    """
    Returns [c_plus, c_minus] collapse operators for smesolve.
    eta_i is the detection efficiency for channel i.
    The total decay rate per atom is gamma (shared between + and - channels).
    """
    return [
        np.sqrt(gamma * eta_1) * np.exp(1j * phi_1) * Cp,
        np.sqrt(gamma * eta_2) * np.exp(1j * phi_2) * Cm,
    ]


# ── Observables ───────────────────────────────────────────────────────────────
def Variance_z(t, state):
    return variance(J_z, state)


def concurrence_pure_state(state):
    c = state.full().flatten()
    return 2.0 * np.abs(c[0] * c[3] - c[1] * c[2])


def concurrence_for_solver_general(t, state):
    if state.isket:
        conc = concurrence_pure_state(state)
    else:
        sysy = tensor(sigmay(), sigmay())
        rho_tilde = (state * sysy) * (state.conj() * sysy)
        evals = rho_tilde.eigenenergies()
        evals = abs(np.sort(np.real(evals)))
        sqrt_evals = np.sqrt(evals)
        lsum = sqrt_evals[3] - sqrt_evals[2] - sqrt_evals[1] - sqrt_evals[0]
        conc = np.maximum(0.0, lsum)
    return float(np.real_if_close(conc))


def eval_spin_components(state):
    return expect(J_x, state), expect(J_y, state), expect(J_z, state)


def xi_KU_solver(t, rho, N=None, tol=1e-12):
    if N is None:
        dim = rho.shape[0]
        N = int(round(np.log2(dim)))

    Jx_exp, Jy_exp, Jz_exp = eval_spin_components(rho)
    Jmean = np.array(
        [
            float(np.real_if_close(Jx_exp)),
            float(np.real_if_close(Jy_exp)),
            float(np.real_if_close(Jz_exp)),
        ]
    )
    m = np.linalg.norm(Jmean)

    Js = [J_x, J_y, J_z]
    C = np.zeros((3, 3), dtype=float)
    for i, Ji in enumerate(Js):
        for j, Jj in enumerate(Js):
            second_moment = 0.5 * expect(Ji * Jj + Jj * Ji, rho)
            C[i, j] = float(np.real_if_close(second_moment - Jmean[i] * Jmean[j]))

    if m > tol:
        u  = Jmean / m
        a  = np.array([1.0, 0.0, 0.0]) if abs(u[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        e1 = a - u * np.dot(u, a)
        e1 /= np.linalg.norm(e1)
        e2 = np.cross(u, e1)
        C2 = np.array(
            [
                [e1 @ C @ e1, e1 @ C @ e2],
                [e2 @ C @ e1, e2 @ C @ e2],
            ],
            dtype=float,
        )
        lam_min = np.linalg.eigvalsh(C2)[0]
    else:
        lam_min = np.linalg.eigvalsh(C)[0]

    lam_min = max(float(np.real_if_close(lam_min)), 0.0)
    xi2 = (4.0 / N) * lam_min
    return float(xi2)


def norm_J(t, rho):
    return np.linalg.norm(eval_spin_components(rho))


def xi_WIN_solver(t, rho, N=None, tol=1e-12):
    if N is None:
        dim = rho.shape[0]
        N = int(round(np.log2(dim)))
    Jmean = np.array(eval_spin_components(rho))
    norm  = np.linalg.norm(Jmean)
    return xi_KU_solver(t, rho) * ((N / 2) / norm) ** 2 if norm > tol else xi_KU_solver(t, rho)


# ── Angle helpers ─────────────────────────────────────────────────────────────
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


# ── Simulation bookkeeping ────────────────────────────────────────────────────
DEFAULT_EOPS = [
    ("Conc",   concurrence_for_solver_general),
    ("Xi2_KU", xi_KU_solver),
    ("Norm_J", norm_J),
]

LABELS = {
    "Conc":    r"$\overline{\mathcal{C}}$",
    "Xi2_KU":  r"$\overline{\xi^2_{KU}}$",
    "Norm_J":  r"$\overline{|\langle \mathbf{J} \rangle|}$",
    "Xi2_WIN": r"$\overline{\xi^2_{WIN}}$",
}


def eta_pair_key(eta_1: float, eta_2: float) -> str:
    return f"{eta_1:g}_{eta_2:g}"


def parse_etas(raw_1: str, raw_2: str) -> list[tuple[float, float]]:
    etas_1 = [float(x.strip()) for x in raw_1.split(",") if x.strip()]
    etas_2 = [float(x.strip()) for x in raw_2.split(",") if x.strip()]
    if not etas_1 or not etas_2:
        raise ValueError("Eta list is empty. Use --etas_1 and --etas_2 with comma-separated values.")
    if len(etas_1) == 1 and len(etas_2) > 1:
        etas_1 = etas_1 * len(etas_2)
    elif len(etas_2) == 1 and len(etas_1) > 1:
        etas_2 = etas_2 * len(etas_1)
    elif len(etas_1) != len(etas_2):
        raise ValueError(
            f"Length mismatch: len(etas_1)={len(etas_1)} and len(etas_2)={len(etas_2)}. "
            "Use equal lengths, or one list with a single value to broadcast."
        )
    for eta_1, eta_2 in zip(etas_1, etas_2):
        if eta_1 < 0.0 or eta_1 > 1.0 or eta_2 < 0.0 or eta_2 > 1.0:
            raise ValueError(f"Invalid etas=({eta_1}, {eta_2}). Each eta must be in [0, 1].")
    return list(zip(etas_1, etas_2))


def build_initial_state(kind: str, theta: float, phi_state: float):
    # Default for fluorescence is "ee": both atoms in the excited state |0>.
    if kind == "ee":
        return ee, np.pi
    if kind == "plusplus":
        return PlusPlus, np.pi / 2.0
    if kind == "css":
        return css_2(theta, phi_state), theta
    raise ValueError(f"Unknown state kind: {kind}")


# ── Core simulation ───────────────────────────────────────────────────────────
def run_homodyne(
    rho0,
    times: np.ndarray,
    gamma: float,
    phi1: float,
    phi2: float,
    eta_1: float,
    eta_2: float,
    e_ops: list,
    ntraj: int,
    num_cpus: int,
) -> np.ndarray:
    solver_cpus = max(1, int(num_cpus))
    print(solver_cpus)
    options = {
        "keep_runs_results": False,
        "num_cpus": solver_cpus,
        "map": "parallel" if solver_cpus > 1 else "serial",
        "method": "milstein",
    }

    sol = smesolve(
        H_free,
        rho0,
        times,
        c_ops  = collapsing_operators(gamma, phi1, phi2, 1.0 - eta_1, 1.0 - eta_2),
        sc_ops = collapsing_operators(gamma, phi1, phi2, eta_1, eta_2),
        heterodyne=False,
        e_ops=e_ops,
        ntraj=ntraj,
        options=options,
    )
    return sol.expect


def simulate_single_eta(
    eta_1: float,
    eta_2: float,
    rho0,
    times: np.ndarray,
    gamma: float,
    phi1: float,
    phi2: float,
    e_ops: list,
    columns: list[str],
    ntraj: int,
    num_cpus: int,
) -> tuple[str, pd.DataFrame]:
    avg_expect = run_homodyne(
        rho0=rho0,
        times=times,
        gamma=gamma,
        phi1=phi1,
        phi2=phi2,
        eta_1=float(eta_1),
        eta_2=float(eta_2),
        e_ops=e_ops,
        ntraj=ntraj,
        num_cpus=num_cpus,
    )
    key = eta_pair_key(float(eta_1), float(eta_2))
    return key, pd.DataFrame(np.transpose(avg_expect), columns=columns)


# ── CLI argument parsing ──────────────────────────────────────────────────────
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
            left  = _eval(expr.left)
            right = _eval(expr.right)
            if isinstance(expr.op, ast.Add):  return left + right
            if isinstance(expr.op, ast.Sub):  return left - right
            if isinstance(expr.op, ast.Mult): return left * right
            return left / right
        raise argparse.ArgumentTypeError(
            f"Unsupported angle expression '{raw}'. Allowed: numbers, pi, np.pi, + - * / and parentheses."
        )

    return float(_eval(node))


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="N=2 homodyne fluorescence simulation")
    parser.add_argument("--state", choices=["ee", "plusplus", "css"], default="ee",
                        help="Initial state. 'ee': both excited (default for fluorescence).")
    parser.add_argument("--theta",     type=float, default=np.pi / 2.0, help="Used when --state css")
    parser.add_argument("--phi-state", type=float, default=0.0,         help="Used when --state css")
    parser.add_argument("--gamma",     type=float, default=1.0)
    parser.add_argument("--phi1", type=parse_angle_arg, default=0.0,
                        help="Homodyne phase C_plus channel (e.g. 0, pi/2, np.pi/2)")
    parser.add_argument("--phi2", type=parse_angle_arg, default=0.0,
                        help="Homodyne phase C_minus channel")
    parser.add_argument("--etas_1", type=str, default="1")
    parser.add_argument("--etas_2", type=str, default="1,0.8,0.5,0")
    parser.add_argument("--ntraj",    type=int,   default=20)
    parser.add_argument("--t-end",    type=float, default=10.0,
                        help="Final time in units of T1 = 1/gamma")
    parser.add_argument("--dt",       type=float, default=0.005,
                        help="Time step in units of T1 (max 0.005 with Milstein)")
    parser.add_argument("--num-cpus", type=int, default=max(1, os.cpu_count() - 1))
    parser.add_argument("--eta-jobs", type=int, default=1,
                        help="Number of parallel joblib workers over eta values")
    parser.add_argument("--joblib-backend",
                        choices=["loky", "threading", "multiprocessing"], default="loky")
    parser.add_argument("--joblib-verbose", type=int, default=0)
    parser.add_argument("--eta-index", type=int, default=None,
                        help="Run only one eta by index (useful for Slurm job arrays)")
    parser.add_argument("--use-slurm-array", action="store_true")
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--out-root", type=str, default=r"./Graphs/Thesis_Fluorescence_N2")
    parser.add_argument("--no-theory", action="store_true",
                        help="Disable theoretical overlay (reserved for future use)")
    parser.add_argument("--usetex", action="store_true",
                        help="Enable LaTeX rendering in plots (requires a LaTeX installation).")
    return parser


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    t0 = time.perf_counter()
    print("PY_START", flush=True)

    args    = make_parser().parse_args()
    etas_all = parse_etas(args.etas_1, args.etas_2)
    eta_index = args.eta_index
    etas      = etas_all

    psi0, theta_for_theory = build_initial_state(args.state, args.theta, args.phi_state)
    rho0   = ket2dm(psi0)
    gamma  = float(args.gamma)
    t1     = 1.0 / gamma
    times  = np.arange(0.0, args.t_end * t1, args.dt * t1)
    columns = [name for name, _ in DEFAULT_EOPS]
    e_ops   = [op  for _, op  in DEFAULT_EOPS]

    eta_jobs = max(1, int(args.eta_jobs))
    if eta_jobs > len(etas):
        eta_jobs = len(etas)

    num_cpus_eff = int(args.num_cpus)
    if eta_jobs > 1 and num_cpus_eff > 1:
        print(
            f"[joblib] eta-jobs={eta_jobs}: forcing --num-cpus to 1 per eta "
            f"(requested {num_cpus_eff}) to avoid nested parallelism."
        )
        num_cpus_eff = 1

    if eta_jobs > 1 and Parallel is None:
        raise ImportError("joblib is required for --eta-jobs > 1. Install with: pip install joblib")

    def run_serial():
        return [
            simulate_single_eta(
                eta_1=float(eta_1),
                eta_2=float(eta_2),
                rho0=rho0,
                times=times,
                gamma=gamma,
                phi1=float(args.phi1),
                phi2=float(args.phi2),
                e_ops=e_ops,
                columns=columns,
                ntraj=int(args.ntraj),
                num_cpus=num_cpus_eff,
            )
            for eta_1, eta_2 in etas
        ]

    eta_jobs_used      = 1
    joblib_backend_used = "serial"

    if eta_jobs == 1:
        results = run_serial()
    else:
        eta_jobs_used       = eta_jobs
        joblib_backend_used = args.joblib_backend

        def run_parallel(backend_name: str):
            return Parallel(
                n_jobs=eta_jobs,
                backend=backend_name,
                verbose=int(args.joblib_verbose),
            )(
                delayed(simulate_single_eta)(
                    eta_1=float(eta_1),
                    eta_2=float(eta_2),
                    rho0=rho0,
                    times=times,
                    gamma=gamma,
                    phi1=float(args.phi1),
                    phi2=float(args.phi2),
                    e_ops=e_ops,
                    columns=columns,
                    ntraj=int(args.ntraj),
                    num_cpus=num_cpus_eff,
                )
                for eta_1, eta_2 in etas
            )

        try:
            results = run_parallel(args.joblib_backend)
        except PermissionError:
            if args.joblib_backend == "threading":
                print("[joblib] backend='threading' failed. Falling back to serial.")
                eta_jobs_used       = 1
                joblib_backend_used = "serial_fallback"
                results = run_serial()
            else:
                print(f"[joblib] backend='{args.joblib_backend}' failed. Retrying with 'threading'.")
                try:
                    results = run_parallel("threading")
                    joblib_backend_used = "threading"
                except PermissionError:
                    print("[joblib] 'threading' also failed. Falling back to serial.")
                    eta_jobs_used       = 1
                    joblib_backend_used = "serial_fallback"
                    results = run_serial()

    # ── Organize results ──────────────────────────────────────────────────────
    ineff_df_map = {key: df for key, df in results}
    ineff_df: dict[str, pd.DataFrame] = {}
    for eta_1, eta_2 in etas:
        key = eta_pair_key(float(eta_1), float(eta_2))
        ineff_df[key] = ineff_df_map[key]

    # Post-process Xi2_WIN from Xi2_KU and Norm_J
    N_spin   = 2.0
    tol_norm = 1e-12
    for key in ineff_df:
        ku_vals   = ineff_df[key]["Xi2_KU"].to_numpy(dtype=float)
        norm_vals = ineff_df[key]["Norm_J"].to_numpy(dtype=float)
        xi_win    = ku_vals.copy()
        mask      = norm_vals > tol_norm
        xi_win[mask] = ku_vals[mask] * ((N_spin / 2.0) / norm_vals[mask]) ** 2
        ineff_df[key]["Xi2_WIN"] = xi_win
    if "Xi2_WIN" not in columns:
        columns = columns + ["Xi2_WIN"]

    # ── Output directory ──────────────────────────────────────────────────────
    phi1_dir = angle_to_path(float(args.phi1))
    phi2_dir = angle_to_path(float(args.phi2))
    out_dir  = Path(args.out_root) / f"ntraj={int(args.ntraj)}__phi_1={phi1_dir}__phi_2={phi2_dir}"
    if eta_index is not None:
        eta_key = eta_pair_key(float(etas[0][0]), float(etas[0][1]))
        out_dir = out_dir / f"eta_idx={eta_index}__eta={eta_key}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Metadata ──────────────────────────────────────────────────────────────
    meta = {
        "measurement":        "homodyne_fluorescence",
        "N":                  2,
        "convention":         "|0>=excited, |1>=ground",
        "channels":           "C_plus = (sm_1 + sm_2)/sqrt(2), C_minus = (sm_1 - sm_2)/sqrt(2)",
        "state":              args.state,
        "theta":              float(args.theta),
        "phi_state":          float(args.phi_state),
        "gamma":              gamma,
        "phi1":               float(args.phi1),
        "phi2":               float(args.phi2),
        "etas_all":           etas_all,
        "etas_1":             [float(e1) for e1, _ in etas_all],
        "etas_2":             [float(e2) for _, e2 in etas_all],
        "eta_index":          eta_index,
        "use_slurm_array":    bool(args.use_slurm_array),
        "ntraj":              int(args.ntraj),
        "num_cpus_requested": int(args.num_cpus),
        "num_cpus_per_eta":   int(num_cpus_eff),
        "eta_jobs":           int(eta_jobs_used),
        "joblib_backend":     args.joblib_backend,
        "joblib_backend_used": joblib_backend_used,
        "dt_T1":              float(args.dt),
        "t_end_T1":           float(args.t_end),
        "columns":            columns,
    }

    # ── Plots ─────────────────────────────────────────────────────────────────
    plt.rcParams["text.usetex"] = bool(args.usetex)
    plt.rcParams.update(
        {
            "mathtext.fontset":   "cm",
            "font.family":        "serif",
            "font.size":          14,
            "axes.unicode_minus": False,
        }
    )

    phi1_tex = angle_to_tex(float(args.phi1))
    phi2_tex = angle_to_tex(float(args.phi2))
    x = times / t1

    for col in columns:
        plt.figure(figsize=(12, 8))

        for eta_1, eta_2 in etas:
            key = eta_pair_key(float(eta_1), float(eta_2))
            y   = ineff_df[key][col].to_numpy()
            plt.plot(
                x, y,
                label=rf"$\eta_1 = {eta_1:g},\ \eta_2 = {eta_2:g}$",
            )

        plt.xlim(0.0, float(args.t_end))

        if col == "Xi2_WIN":
            plt.axhline(1.0, linestyle="--", linewidth=2, color="black", alpha=0.7)
            plt.ylim(0.0, 5.0)

        if col in ("Xi2_KU", "Conc"):
            plt.axhline(1.0, linestyle="--", linewidth=2, color="black", alpha=0.7)
            plt.ylim(0.0, 1.05)

        plt.xlabel(r"$t/T_1$")
        plt.ylabel(LABELS[col])
        plt.title(
            r"Homodyne fluorescence (N=2): "
            + LABELS[col]
            + rf"$,\ \phi_1={phi1_tex}\, \phi_2={phi2_tex}$"
        )
        plt.grid(True, linestyle=":", alpha=0.6)
        plt.legend(loc="upper left", bbox_to_anchor=(0, 0.95), fontsize=12)
        plt.savefig(out_dir / f"Fluorescence_{col}.pdf", bbox_inches="tight")
        plt.close()

    run_config_path = out_dir / "run_config.json"
    run_config_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"OUT_DIR={out_dir}")
    print(f"RUN_CONFIG_PATH={run_config_path}")
    print(f"Done. Results saved in: {out_dir}")

    dt = time.perf_counter() - t0
    print(f"PY_TOTAL_SECONDS={dt:.3f}", flush=True)


if __name__ == "__main__":
    main()
