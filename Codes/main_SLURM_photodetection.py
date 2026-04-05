"""
Photodetection (quantum jump) simulation for N=2 atoms, with Jz measurement channels.
=======================================================================================
Identical structure to mina_SLURM_thesis.py (homodyne), but uses the quantum jump
unraveling (photon counting) instead of homodyne (diffusive).

Physical model
--------------
Same dephasing channels C_+ and C_- built from sigma_z.
For each channel i with efficiency eta_i:
  - Detected   photons: sqrt(gamma * eta_i)     * exp(i*phi_i) * C_i  -> c_ops (known jump)
  - Undetected photons: sqrt(gamma * (1-eta_i)) * exp(i*phi_i) * C_i  -> c_ops (averaged)
All operators go into mcsolve c_ops. mcsolve conditions on every jump,
so eta=1 gives the fully conditional quantum-jump trajectory;
eta=0 gives the Lindblad (unconditional) average.

Computes: Conc, Xi2_KU, Norm_J, Xi2_WIN (post-processed).
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
    mcsolve,          # <-- quantum jump unraveling
    tensor,
    variance,
)


# ── Basis states (same convention as mina_SLURM_thesis.py) ───────────────────
gnd = basis(2, 0)
exc = basis(2, 1)

plus_state = (gnd + exc).unit()
PlusPlus   = tensor(plus_state, plus_state)
ee         = tensor(exc, exc)

def css_2(theta: float, phi: float):
    single_qubit = (
        np.sin(theta / 2.0) * basis(2, 0)
        + np.exp(1j * phi) * np.cos(theta / 2.0) * basis(2, 1)
    )
    return tensor(single_qubit, single_qubit)


# ── Collective spin operators ─────────────────────────────────────────────────
I_2 = qeye(2)
J_x = 0.5 * (tensor(sigmax(), I_2) + tensor(I_2, sigmax()))
J_y = 0.5 * (tensor(sigmay(), I_2) + tensor(I_2, sigmay()))
J_z = 0.5 * (tensor(sigmaz(), I_2) + tensor(I_2, sigmaz()))
H_free = 0.0 * J_z

# Dephasing channels (same as mina_SLURM_thesis.py)
Cp = np.sqrt(0.5) * (tensor(sigmaz(), I_2) + tensor(I_2, sigmaz()))
Cm = np.sqrt(0.5) * (tensor(sigmaz(), I_2) - tensor(I_2, sigmaz()))


def collapsing_operators_photodetection(
    gamma: float,
    phi_1: float,
    phi_2: float,
    eta_1: float,
    eta_2: float,
) -> list:
    """
    Returns c_ops for mcsolve containing ONLY the detected (conditioned) channels.
    mcsolve conditions on every operator in c_ops; undetected photons must NOT
    be included here — they are handled via the non-Hermitian effective Hamiltonian.
      detected : sqrt(gamma * eta) * exp(i*phi) * C  -> c_ops (known jump)
    """
    ops = []
    for eta, phi, C in [(eta_1, phi_1, Cp), (eta_2, phi_2, Cm)]:
        if eta > 0.0:
            ops.append(np.sqrt(gamma * eta) * np.exp(1j * phi) * C)
    return ops


def build_H_eff(gamma: float, eta_1: float, eta_2: float):
    """
    Non-Hermitian effective Hamiltonian that accounts for undetected photon loss.
    mcsolve internally adds  -i/2 * C†C  for each detected c_op; here we add the
    remaining  -i*gamma*(1-eta)/2 * C†C  for the undetected fraction so that the
    total non-Hermitian decay rate is always gamma, independent of eta.
      H_eff = H_free - i*gamma*(1-eta_1)/2 * Cp†Cp - i*gamma*(1-eta_2)/2 * Cm†Cm
    """
    H = H_free
    for eta, C in [(eta_1, Cp), (eta_2, Cm)]:
        if eta < 1.0:
            H = H - 0.5j * gamma * (1.0 - eta) * C.dag() * C
    return H


# ── Observables (identical to mina_SLURM_thesis.py) ──────────────────────────
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
        N = int(round(np.log2(rho.shape[0])))
    Jx_exp, Jy_exp, Jz_exp = eval_spin_components(rho)
    Jmean = np.array([float(np.real_if_close(v)) for v in [Jx_exp, Jy_exp, Jz_exp]])
    m  = np.linalg.norm(Jmean)
    Js = [J_x, J_y, J_z]
    C  = np.zeros((3, 3), dtype=float)
    for i, Ji in enumerate(Js):
        for j, Jj in enumerate(Js):
            sm = 0.5 * expect(Ji * Jj + Jj * Ji, rho)
            C[i, j] = float(np.real_if_close(sm - Jmean[i] * Jmean[j]))
    if m > tol:
        u  = Jmean / m
        a  = np.array([1., 0., 0.]) if abs(u[0]) < 0.9 else np.array([0., 1., 0.])
        e1 = a - u * np.dot(u, a); e1 /= np.linalg.norm(e1)
        e2 = np.cross(u, e1)
        C2 = np.array([[e1@C@e1, e1@C@e2], [e2@C@e1, e2@C@e2]], dtype=float)
        lam_min = np.linalg.eigvalsh(C2)[0]
    else:
        lam_min = np.linalg.eigvalsh(C)[0]
    lam_min = max(float(np.real_if_close(lam_min)), 0.0)
    return float((4.0 / N) * lam_min)

def norm_J(t, rho):
    return np.linalg.norm(eval_spin_components(rho))

def xi_WIN_solver(t, rho, N=None, tol=1e-12):
    if N is None:
        N = int(round(np.log2(rho.shape[0])))
    norm = np.linalg.norm(eval_spin_components(rho))
    return xi_KU_solver(t, rho) * ((N / 2) / norm) ** 2 if norm > tol else xi_KU_solver(t, rho)


# ── Angle helpers ─────────────────────────────────────────────────────────────
def angle_to_path(phi, max_den=12, tol=1e-10):
    twopi = 2 * np.pi
    x = (phi % twopi + twopi) % twopi
    if abs(x) < tol: return "0"
    q = Fraction(x / np.pi).limit_denominator(max_den)
    n, d = q.numerator, q.denominator
    if n == 0: return "0"
    if d == 1: return "pi" if n == 1 else f"{n}pi"
    return ("pi_" + str(d)) if n == 1 else f"{n}pi_{d}"

def angle_to_tex(phi, max_den=12, tol=1e-10):
    twopi = 2 * np.pi
    x = (phi % twopi + twopi) % twopi
    if abs(x) < tol: return "0"
    q = Fraction(x / np.pi).limit_denominator(max_den)
    n, d = q.numerator, q.denominator
    if n == 0: return "0"
    if d == 1: return r"\pi" if n == 1 else fr"{n}\pi"
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
        raise ValueError("Eta list is empty.")
    if len(etas_1) == 1 and len(etas_2) > 1:
        etas_1 = etas_1 * len(etas_2)
    elif len(etas_2) == 1 and len(etas_1) > 1:
        etas_2 = etas_2 * len(etas_1)
    elif len(etas_1) != len(etas_2):
        raise ValueError(
            f"Length mismatch: len(etas_1)={len(etas_1)} and len(etas_2)={len(etas_2)}."
        )
    for e1, e2 in zip(etas_1, etas_2):
        if not (0.0 <= e1 <= 1.0 and 0.0 <= e2 <= 1.0):
            raise ValueError(f"Invalid etas=({e1}, {e2}). Each eta must be in [0, 1].")
    return list(zip(etas_1, etas_2))

def build_initial_state(kind: str, theta: float, phi_state: float):
    if kind == "plusplus": return PlusPlus, np.pi / 2.0
    if kind == "ee":       return ee,       np.pi
    if kind == "css":      return css_2(theta, phi_state), theta
    raise ValueError(f"Unknown state kind: {kind}")


# ── Core simulation — quantum jump / photodetection ───────────────────────────
def run_photodetection(
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
    }

    H_eff = build_H_eff(gamma, eta_1, eta_2)
    c_ops = collapsing_operators_photodetection(gamma, phi1, phi2, eta_1, eta_2)

    sol = mcsolve(
        H_eff,
        rho0,
        times,
        c_ops=c_ops,
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
    avg_expect = run_photodetection(
        rho0=rho0, times=times, gamma=gamma,
        phi1=phi1, phi2=phi2,
        eta_1=float(eta_1), eta_2=float(eta_2),
        e_ops=e_ops, ntraj=ntraj, num_cpus=num_cpus,
    )
    key = eta_pair_key(float(eta_1), float(eta_2))
    return key, pd.DataFrame(np.transpose(avg_expect), columns=columns)


# ── CLI ────────────────────────────────────────────────────────────────────────
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
        raise argparse.ArgumentTypeError(f"Invalid angle expression '{raw}'.") from exc

    def _eval(expr):
        if isinstance(expr, ast.Expression):  return _eval(expr.body)
        if isinstance(expr, ast.Constant) and isinstance(expr.value, (int, float)):
            return float(expr.value)
        if isinstance(expr, ast.Name) and expr.id == "pi":
            return float(np.pi)
        if (isinstance(expr, ast.Attribute)
                and isinstance(expr.value, ast.Name)
                and expr.value.id in {"np", "numpy"}
                and expr.attr == "pi"):
            return float(np.pi)
        if isinstance(expr, ast.UnaryOp) and isinstance(expr.op, (ast.UAdd, ast.USub)):
            val = _eval(expr.operand)
            return val if isinstance(expr.op, ast.UAdd) else -val
        if isinstance(expr, ast.BinOp) and isinstance(expr.op, (ast.Add, ast.Sub, ast.Mult, ast.Div)):
            l, r = _eval(expr.left), _eval(expr.right)
            if isinstance(expr.op, ast.Add):  return l + r
            if isinstance(expr.op, ast.Sub):  return l - r
            if isinstance(expr.op, ast.Mult): return l * r
            return l / r
        raise argparse.ArgumentTypeError(f"Unsupported expression '{raw}'.")

    return float(_eval(node))


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="N=2 photodetection (quantum jump) simulation — Jz channels"
    )
    parser.add_argument("--state",     choices=["plusplus", "ee", "css"], default="plusplus")
    parser.add_argument("--theta",     type=float, default=np.pi / 2.0)
    parser.add_argument("--phi-state", type=float, default=0.0)
    parser.add_argument("--gamma",     type=float, default=1.0)
    parser.add_argument("--phi1",      type=parse_angle_arg, default=0.0)
    parser.add_argument("--phi2",      type=parse_angle_arg, default=0.0)
    parser.add_argument("--etas_1",    type=str,   default="1")
    parser.add_argument("--etas_2",    type=str,   default="1,0.8,0.5,0")
    parser.add_argument("--ntraj",     type=int,   default=20)
    parser.add_argument("--t-end",     type=float, default=10.0,
                        help="Final time in units of T1")
    parser.add_argument("--dt",        type=float, default=0.005,
                        help="Time step in units of T1")
    parser.add_argument("--num-cpus",  type=int,   default=max(1, os.cpu_count() - 1))
    parser.add_argument("--eta-jobs",  type=int,   default=1)
    parser.add_argument("--joblib-backend",
                        choices=["loky", "threading", "multiprocessing"], default="loky")
    parser.add_argument("--joblib-verbose", type=int, default=0)
    parser.add_argument("--eta-index", type=int,   default=None)
    parser.add_argument("--use-slurm-array", action="store_true")
    parser.add_argument("--seed",      type=int,   default=12345)
    parser.add_argument("--out-root",  type=str,   default=r"./Graphs/Photodetection_N2")
    parser.add_argument("--usetex",    action="store_true")
    return parser


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    t0 = time.perf_counter()
    print("PY_START", flush=True)

    args      = make_parser().parse_args()
    etas_all  = parse_etas(args.etas_1, args.etas_2)
    eta_index = args.eta_index
    etas      = etas_all

    psi0, theta_for_theory = build_initial_state(args.state, args.theta, args.phi_state)
    rho0   = psi0  # mcsolve requires ket, not density matrix
    gamma  = float(args.gamma)
    t1     = 1.0 / gamma
    times  = np.arange(0.0, args.t_end * t1, args.dt * t1)
    columns = [name for name, _ in DEFAULT_EOPS]
    e_ops   = [op   for _, op   in DEFAULT_EOPS]

    eta_jobs     = max(1, int(args.eta_jobs))
    if eta_jobs > len(etas):
        eta_jobs = len(etas)

    num_cpus_eff = int(args.num_cpus)
    if eta_jobs > 1 and num_cpus_eff > 1:
        print(f"[joblib] eta-jobs={eta_jobs}: forcing --num-cpus to 1 per eta.")
        num_cpus_eff = 1

    if eta_jobs > 1 and Parallel is None:
        raise ImportError("joblib required for --eta-jobs > 1.")

    def run_serial():
        return [
            simulate_single_eta(
                eta_1=float(e1), eta_2=float(e2),
                rho0=rho0, times=times, gamma=gamma,
                phi1=float(args.phi1), phi2=float(args.phi2),
                e_ops=e_ops, columns=columns,
                ntraj=int(args.ntraj), num_cpus=num_cpus_eff,
            )
            for e1, e2 in etas
        ]

    eta_jobs_used       = 1
    joblib_backend_used = "serial"

    if eta_jobs == 1:
        results = run_serial()
    else:
        eta_jobs_used       = eta_jobs
        joblib_backend_used = args.joblib_backend

        def run_parallel(backend_name: str):
            return Parallel(
                n_jobs=eta_jobs, backend=backend_name,
                verbose=int(args.joblib_verbose),
            )(
                delayed(simulate_single_eta)(
                    eta_1=float(e1), eta_2=float(e2),
                    rho0=rho0, times=times, gamma=gamma,
                    phi1=float(args.phi1), phi2=float(args.phi2),
                    e_ops=e_ops, columns=columns,
                    ntraj=int(args.ntraj), num_cpus=num_cpus_eff,
                )
                for e1, e2 in etas
            )

        try:
            results = run_parallel(args.joblib_backend)
        except PermissionError:
            if args.joblib_backend == "threading":
                print("[joblib] threading failed. Falling back to serial.")
                eta_jobs_used = 1; joblib_backend_used = "serial_fallback"
                results = run_serial()
            else:
                print(f"[joblib] {args.joblib_backend} failed. Retrying with threading.")
                try:
                    results = run_parallel("threading")
                    joblib_backend_used = "threading"
                except PermissionError:
                    print("[joblib] threading also failed. Falling back to serial.")
                    eta_jobs_used = 1; joblib_backend_used = "serial_fallback"
                    results = run_serial()

    # ── Organize & post-process ───────────────────────────────────────────────
    ineff_df_map = {key: df for key, df in results}
    ineff_df: dict[str, pd.DataFrame] = {}
    for e1, e2 in etas:
        key = eta_pair_key(float(e1), float(e2))
        ineff_df[key] = ineff_df_map[key]

    # Xi2_WIN from Xi2_KU and Norm_J
    N_spin = 2.0; tol_norm = 1e-12
    for key in ineff_df:
        ku   = ineff_df[key]["Xi2_KU"].to_numpy(dtype=float)
        norm = ineff_df[key]["Norm_J"].to_numpy(dtype=float)
        win  = ku.copy()
        mask = norm > tol_norm
        win[mask] = ku[mask] * ((N_spin / 2.0) / norm[mask]) ** 2
        ineff_df[key]["Xi2_WIN"] = win
    if "Xi2_WIN" not in columns:
        columns = columns + ["Xi2_WIN"]

    # ── Output directory ──────────────────────────────────────────────────────
    phi1_dir = angle_to_path(float(args.phi1))
    phi2_dir = angle_to_path(float(args.phi2))
    out_dir  = (
        Path(args.out_root)
        / f"ntraj={int(args.ntraj)}__phi_1={phi1_dir}__phi_2={phi2_dir}"
    )
    if eta_index is not None:
        eta_key = eta_pair_key(float(etas[0][0]), float(etas[0][1]))
        out_dir = out_dir / f"eta_idx={eta_index}__eta={eta_key}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Metadata ──────────────────────────────────────────────────────────────
    meta = {
        "measurement":        "photodetection_quantum_jump",
        "N":                  2,
        "channels":           "C_plus/C_minus from sigma_z (dephasing)",
        "unraveling":         "quantum jump (mcsolve)",
        "state":              args.state,
        "gamma":              gamma,
        "phi1":               float(args.phi1),
        "phi2":               float(args.phi2),
        "etas_all":           etas_all,
        "etas_1":             [float(e1) for e1, _ in etas_all],
        "etas_2":             [float(e2) for _, e2 in etas_all],
        "eta_index":          eta_index,
        "ntraj":              int(args.ntraj),
        "num_cpus_requested": int(args.num_cpus),
        "num_cpus_per_eta":   int(num_cpus_eff),
        "eta_jobs":           int(eta_jobs_used),
        "joblib_backend_used": joblib_backend_used,
        "dt_T1":              float(args.dt),
        "t_end_T1":           float(args.t_end),
        "columns":            columns,
    }

    # ── Plots ─────────────────────────────────────────────────────────────────
    plt.rcParams["text.usetex"] = bool(args.usetex)
    plt.rcParams.update({
        "mathtext.fontset":   "cm",
        "font.family":        "serif",
        "font.size":          14,
        "axes.unicode_minus": False,
    })

    phi1_tex = angle_to_tex(float(args.phi1))
    phi2_tex = angle_to_tex(float(args.phi2))
    x = times / t1

    for col in columns:
        plt.figure(figsize=(12, 8))
        for e1, e2 in etas:
            key = eta_pair_key(float(e1), float(e2))
            y   = ineff_df[key][col].to_numpy()
            plt.plot(x, y, label=rf"$\eta_1 = {e1:g},\ \eta_2 = {e2:g}$")

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
            r"Photodetection $J_z$ (N=2): "
            + LABELS[col]
            + rf"$,\ \phi_1={phi1_tex}\, \phi_2={phi2_tex}$"
        )
        plt.grid(True, linestyle=":", alpha=0.6)
        plt.legend(loc="upper left", bbox_to_anchor=(0, 0.95), fontsize=12)
        plt.savefig(out_dir / f"Photodetection_{col}.pdf", bbox_inches="tight")
        plt.close()

    (out_dir / "run_config.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8")
    print(f"OUT_DIR={out_dir}")
    print(f"PY_TOTAL_SECONDS={time.perf_counter() - t0:.3f}", flush=True)


if __name__ == "__main__":
    main()
