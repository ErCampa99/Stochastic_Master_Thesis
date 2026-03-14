"""
Homodyne single dephasing simulation for N=2 atoms, with Jz measurement as plus channel.
This script simulates the homodyne detection of a collective dephasing process for two atoms (N=2) using the QuTiP library. The measurement is performed on the Jz operator, which corresponds to the collective spin along the z-axis. 
The script allows for varying the measurement efficiency (eta) and other parameters, and it computes expectation values such as concurrence, xi^2_KU, and variance of Jz over time.
"""

#Import standard libraries and set environment variables to limit thread usage
import ast
import argparse
import json
import os
import time
from fractions import Fraction
from pathlib import Path

# Set environment variables to limit the number of threads used by various libraries
# This is important to prevent oversubscription when using multiprocessing in QuTiP.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

#Import libraries  
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Try to import joblib for parallel processing. If it's not available, set Parallel and delayed to None.
try:
    from joblib import Parallel, delayed
except ImportError:
    Parallel = None
    delayed = None

# Import necessary functions and classes from QuTiP for quantum simulations.
from qutip import (
    basis,
    expect,
    ket2dm,
    qeye,
    sigmax,
    sigmay,
    sigmaz,
    smesolve,
    tensor,
    variance,
)


# Basis states, convention is |g> = |0>, |e> = |1>
gnd = basis(2, 0)
exc = basis(2, 1)

#Define plus state 
plus_state = (gnd + exc).unit()
PlusPlus = tensor(plus_state, plus_state)
ee = tensor(exc, exc)

# Function to create a coherent spin state (CSS) for N=2 atoms based on the given angles theta and phi.
def css_2(theta: float, phi: float):
    single_qubit = (
        np.sin(theta / 2.0) * basis(2, 0)
        + np.exp(1j * phi) * np.cos(theta / 2.0) * basis(2, 1)
    )
    return tensor(single_qubit, single_qubit)


# Collective spin operators for N=2
I_2 = qeye(2)
J_x = 0.5 * (tensor(sigmax(), I_2) + tensor(I_2, sigmax()))
J_y = 0.5 * (tensor(sigmay(), I_2) + tensor(I_2, sigmay()))
J_z = 0.5 * (tensor(sigmaz(), I_2) + tensor(I_2, sigmaz()))

# Free Hamiltonian, Omega = 0
OMEGA = 0.0
H_free = 0.5 * OMEGA * (tensor(sigmaz(), I_2) + tensor(I_2, sigmaz()))

# Homodyne channels for N=2, C_plus & C_minus 
Cp = np.sqrt(0.5) * (tensor(sigmaz(), I_2) + tensor(I_2, sigmaz()))
Cm = np.sqrt(0.5) * (tensor(sigmaz(), I_2) - tensor(I_2, sigmaz()))

#To create the collapsing operators for N=2 based on the measurement efficiency (eta) and the phases (phi_1 and phi_2) of the homodyne detection. 
def collapsing_operators(gamma: float, phi_1: float, phi_2: float, eta_1: float, eta_2: float):
    return [
        np.sqrt(gamma*eta_1) * np.exp(1j * phi_1) * Cp,
        np.sqrt(gamma*eta_2) * np.exp(1j * phi_2) * Cm,
    ]

#Variance for the solver
def Variance_z(t, state):
    return variance(J_z, state)

#Concurrence for pure states, used in the solver when the state is a ket. 
#For mixed states, the concurrence is computed using the standard formula involving the eigenvalues of the rho_tilde matrix.
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

#Function to evaluate the expectation values of the collective spin components J_x, J_y, and J_z for a given state. 
# This is used to compute the mean spin vector and the covariance matrix needed for the xi^2_KU calculation.
def eval_spin_components(state):
    return expect(J_x, state), expect(J_y, state), expect(J_z, state)

#To compute the spin-squeezing parameter xi^2_KU based on the method described by Kitagawa and Ueda.
#It uses MSD to compute the mean spin vector and the covariance matrix of the spin components, and then calculates the minimum variance in the plane perpendicular to the mean spin direction. 
#The result is then normalized by N/4 to give xi^2_KU.
def xi_KU_solver(t, rho, N=None, tol=1e-12):

    # If N is not provided, infer it from the dimension of the state. For N=2, the dimension is 4, so N=log2(4)=2.
    if N is None:
        dim = rho.shape[0]
        # For a system of N qubits, the dimension of the Hilbert space is 2^N. Therefore, we can infer N by taking the logarithm base 2 of the dimension.
        N = int(round(np.log2(dim)))

    #Calculate the mean spin vector and the covariance matrix of the spin components. 
    Jx_exp, Jy_exp, Jz_exp = eval_spin_components(rho)
    Jmean = np.array(
        [
            float(np.real_if_close(Jx_exp)),
            float(np.real_if_close(Jy_exp)),
            float(np.real_if_close(Jz_exp)),
        ]
    )
    #The norm of the MSD
    m = np.linalg.norm(Jmean)

    Js = [J_x, J_y, J_z]
    C = np.zeros((3, 3), dtype=float)

    #Compute the covariance matrix C of the spin components. The covariance matrix is defined as C[i, j] = 0.5 * <Ji Jj + Jj Ji> - <Ji><Jj>, where Ji and Jj are the collective spin operators.
    for i, Ji in enumerate(Js):
        for j, Jj in enumerate(Js):
            second_moment = 0.5 * expect(Ji * Jj + Jj * Ji, rho)
            C[i, j] = float(np.real_if_close(second_moment - Jmean[i] * Jmean[j]))

    #If MSD is non-zero, we need to find the minimum variance in the plane perpendicular to the mean spin direction. 
    #This involves projecting the covariance matrix onto the plane orthogonal to the mean spin vector and finding the minimum eigenvalue of this projected matrix. 
    #If MSD is zero, we can simply take the minimum eigenvalue of the original covariance matrix.
    if m > tol:

        #Unit vector in the direction of the mean spin vector
        u = Jmean / m

        #To find the plane perpendicular to the mean spin vector, we can choose any two orthogonal vectors that are also orthogonal to u.
        a = np.array([1.0, 0.0, 0.0]) if abs(u[0]) < 0.9 else np.array([0.0, 1.0, 0.0])

        #We can use the Gram-Schmidt process to find two orthogonal vectors e1 and e2 that are perpendicular to u.
        #First, we project a onto the plane perpendicular to u to get e1, and then we normalize it.
        e1 = a - u * np.dot(u, a)
        e1 /= np.linalg.norm(e1)

        #Then, we can find e2 by taking the cross product of u and e1, which will also be perpendicular to u and orthogonal to e1.
        e2 = np.cross(u, e1)

        #Finally, we project the covariance matrix C onto the plane spanned by e1 and e2 to get a 2x2 matrix C2, 
        #and we find the minimum eigenvalue of C2, which corresponds to the minimum variance in the plane perpendicular to the mean spin direction.
        C2 = np.array(
            [
                #La chiocciola serve a fare le moltiplicazioni vettoriali e ottenere i componenti scalari necessari per costruire la matrice di covarianza proiettata.
                [e1 @ C @ e1, e1 @ C @ e2],
                [e2 @ C @ e1, e2 @ C @ e2],
            ],
            dtype=float,
        )
        #Take the 0-th eigenvalue of C2, which is the minimum eigenvalue since C2 is a 2x2 matrix. 
        #This gives us the minimum variance in the plane perpendicular to the mean spin direction.
        lam_min = np.linalg.eigvalsh(C2)[0]
    else:
        #If MSD=0, we can simply take the minimum eigenvalue of the original covariance matrix C, since there is no preferred direction for the mean spin vector.
        lam_min = np.linalg.eigvalsh(C)[0]

    #To ensure that the minimum variance is non-negative (as it should be for a physical state), we take the maximum of lam_min and 0.0.
    lam_min = max(float(np.real_if_close(lam_min)), 0.0)

    #Finally, we normalize the minimum variance by N/4 to get xi^2_KU. 
    #The factor of 4 comes from the fact that for a coherent spin state (CSS), the variance in the plane perpendicular to the mean spin direction is N/4, so this normalization allows us to compare the squeezing relative to a CSS.
    xi2 = (4.0 / N) * lam_min
    return float(xi2)

def norm_J(t, rho):
    return np.linalg.norm(eval_spin_components(rho))

def xi_WIN_solver(t, rho, N=None, tol=1e-12):
    
    #This function computes the Wineland spin-squeezing parameter xi^2_WIN.
    #It is similar to the Kitagawa-Ueda parameter but includes an additional normalization by the mean spin length.

    if N is None:
        dim = rho.shape[0]
        N = int(round(np.log2(dim)))
    
    #Evaluate the norm of the mean spin vector (MSD)
    Jmean = np.array(eval_spin_components(rho))
    norm = np.linalg.norm(Jmean)

    #Compute xi^2_KU using the previously defined function. 
    #If the norm of the mean spin vector is greater than a small tolerance, we normalize xi^2_KU by (N/2)^2 / norm^2 to get xi^2_WIN.
    return xi_KU_solver(t, rho)*((N/2)/norm)**2 if norm > tol else xi_KU_solver(t, rho)


def angle_to_path(phi, max_den=12, tol=1e-10):

    #Convert an angle phi to a string representation suitable for file paths. 
    # he function normalizes the angle to the range [0, 2*pi) and then expresses it as a rational multiple of pi if possible, using the fractions module to find a simple fraction representation.
    twopi = 2 * np.pi
    x = (phi % twopi + twopi) % twopi
    if abs(x) < tol:
        return "0"
    
    #Convert the angle to a fraction of pi and limit the denominator to max_den to get a simple representation.
    q = Fraction(x / np.pi).limit_denominator(max_den)

    #Extract the numerator and denominator from the fraction. 
    #If the numerator is zero, return "0". If the denominator is one, return "pi" or "npi" depending on the numerator. Otherwise, return a string in the format "pi_d" or "npi_d" where d is the denominator.
    n, d = q.numerator, q.denominator

    #If n==0, the angle is effectively zero, so we return "0". 
    #If d==1, the angle is an integer multiple of pi, so we return "pi" or "npi". For other cases, we return a string that indicates the fraction of pi, such as "pi_3" for pi/3 or "2pi_5" for 2*pi/5.
    if n == 0:
        return "0"
    if d == 1:
        return "pi" if n == 1 else f"{n}pi"
    
    #For other cases, we return a string that indicates the fraction of pi, such as "pi_3" for pi/3 or "2pi_5" for 2*pi/5. If n is negative, we include the negative sign in the output.
    return ("pi_" + str(d)) if n == 1 else f"{n}pi_{d}"


def angle_to_tex(phi, max_den=12, tol=1e-10):

    #Convert an angle phi to a LaTeX string representation.
    #This function is similar to angle_to_path but formats the output for LaTeX.
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

    #Here's the magic
    s = r"\pi / %d" % d if abs(n) == 1 else r"%s\pi / %d" % (num, d)
    return s if n > 0 else "-" + s

#Utility function to save the inefficiency DataFrames into a compressed .npz file.
# def save_ineff_df_npz(
#     ineff_df: dict[str, pd.DataFrame],
#     out_npz: str | Path,
#     meta: dict | None = None,
#     step_col: str = "step",
# ):
#     out_npz = Path(out_npz)
#     out_npz.parent.mkdir(parents=True, exist_ok=True)

#     labels = list(ineff_df.keys())
#     dfs = [ineff_df[l] for l in labels]

#     if step_col in dfs[0].columns:
#         step = dfs[0][step_col].to_numpy()
#     else:
#         step = np.arange(len(dfs[0]), dtype=int)

#     n_steps = len(step)
#     for i, df in enumerate(dfs):
#         this_step = (
#             df[step_col].to_numpy()
#             if step_col in df.columns
#             else np.arange(len(df), dtype=int)
#         )
#         if len(this_step) != n_steps or not np.all(this_step == step):
#             raise ValueError(
#                 f"Step grid non consistente per label='{labels[i]}': "
#                 f"atteso {n_steps} step uguali al primo DF."
#             )

#     cols = [c for c in dfs[0].columns if c != step_col]
#     all_cols = set(cols)
#     for df in dfs[1:]:
#         all_cols |= set([c for c in df.columns if c != step_col])
#     cols = sorted(all_cols)

#     arrays = {}
#     for c in cols:
#         mat = np.full((len(labels), n_steps), np.nan, dtype=float)
#         for i, df in enumerate(dfs):
#             if c in df.columns:
#                 v = df[c].to_numpy()
#                 if np.iscomplexobj(v):
#                     v = np.real(v)
#                 mat[i, :] = v.astype(float, copy=False)
#         arrays[c] = mat

#     np.savez_compressed(
#         out_npz,
#         step=step.astype(int, copy=False),
#         labels=np.array(labels, dtype=str),
#         **arrays,
#     )

#     if meta is not None:
#         out_npz.with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


#===================================================================================================================
#===================================================================================================================
#===================================================================================================================

# Define the default expectation value operators to compute during the simulation.
DEFAULT_EOPS = [
    ("Conc", concurrence_for_solver_general),
    ("Xi2_KU", xi_KU_solver),
    ("Norm_J", norm_J),
    #("Variance_z", Variance_z),
]

# Define LaTeX labels for the quantities being plotted.
LABELS = {
    "Conc": r"$\overline{\mathcal{C}}$",
    "Xi2_KU": r"$\overline{\xi^2_{KU}}$",
    "Norm_J": r"$\overline{|\langle \mathbf{J} \rangle|}$",
    "Xi2_WIN": r"$\overline{\xi^2_{WIN}}$",
    #"Variance_z": r"$\overline{\mathrm{Var}(J_z)}$",
}

def eta_pair_key(eta_1: float, eta_2: float) -> str:
    return f"{eta_1:g}_{eta_2:g}"

# Utility function to parse the list of etas from a comma-separated string.
def parse_etas(raw_1: str, raw_2: str) -> list[tuple[float, float]]:
    #The function takes a comma-separated string of eta values, splits it, and converts each value to a float. 
    #It also checks that the list is not empty and that each eta value is between 0 and 1.

    #Parse the comma-separated strings for etas_1 and etas_2, convert them to lists of floats, and validate the values.#
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

# Utility function to build the initial state based on the specified kind and parameters.
def build_initial_state(kind: str, theta: float, phi_state: float):
    if kind == "plusplus":
        return PlusPlus, np.pi / 2.0
    if kind == "ee":
        return ee, np.pi
    if kind == "css":
        return css_2(theta, phi_state), theta
    raise ValueError(f"Unknown state kind: {kind}")


# Function to run the homodyne simulation in chunks and average the results.
def run_homodyne(
        
    #List of all parameters needed for the simulation
    rho0,
    times: np.ndarray,
    gamma: float,
    phi1: float,
    phi2: float,
    eta_1: float,
    eta_2: float,
    e_ops: list,
    ntraj: int,
    #columns: list[str],
    #chunk_size: int,
    num_cpus: int,
    #seed: int,
) -> np.ndarray:
    #Set options for the smesolve function, including parallelization settings based on the number of CPUs available.
    solver_cpus = max(1, int(num_cpus))
    print(solver_cpus)
    options = {
        "keep_runs_results": False,
        "num_cpus": solver_cpus,
        "map": "parallel" if solver_cpus > 1 else "serial",
    }

    #Setting up variables for the chunked simulation. 
    #We will accumulate a weighted sum of the expectation values across chunks, and keep track of how many trajectories have been simulated so far.
    # weighted_sum = None
    # done = 0
    # chunk_id = 0
    # rng = np.random.default_rng(seed)

    #Run the simulation in chunks to manage memory usage and allow for progress tracking. Each chunk will simulate a portion of the total trajectories, and the results will be averaged together at the end.
    #Finché non abbiamo simulato tutte le traiettorie (done < ntraj), continuiamo a eseguire i chunk. In ogni iterazione, calcoliamo quante traiettorie simulare in questo chunk (n_this)
    # while done < ntraj:
    #     chunk_id += 1
    #     n_this = min(chunk_size, ntraj - done)
    #     np.random.seed(int(rng.integers(0, 2**31 - 1)))

    sol = smesolve(
        H_free,
        rho0,
        times,
        c_ops=collapsing_operators(gamma, phi1, phi2, 1.0 - eta_1, 1.0- eta_2),
        sc_ops=collapsing_operators(gamma, phi1, phi2, eta_1, eta_2),
        heterodyne=False,
        e_ops=e_ops,
        ntraj=ntraj,
        options=options,
    )

        #Extract the expectation values from the solution, convert them to real numbers if they are close to real, and accumulate a weighted sum for averaging.
        # expect_chunk = np.asarray(sol.expect)
        # expect_chunk = np.real_if_close(expect_chunk)
        # expect_chunk = np.asarray(expect_chunk, dtype=float)

        # if weighted_sum is None:
        #     weighted_sum = expect_chunk * n_this
        # else:
        #     weighted_sum += expect_chunk * n_this

        # done += n_this
        #print(f"[eta={eta:.3f}] chunk {chunk_id}: {done}/{ntraj} trajectories completed")

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
    #chunk_size: int,
    num_cpus: int,
    #seed: int,
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
        #chunk_size=chunk_size,
        num_cpus=num_cpus,
        #seed=seed,
    )
    key = eta_pair_key(float(eta_1), float(eta_2))
    return key, pd.DataFrame(np.transpose(avg_expect), columns=columns)

def theoretical_concurrence_curve(
    times: np.ndarray,
    gamma: float,
    eta: float,
    theta: float,
) -> np.ndarray:
    return (np.sin(theta) ** 2) * np.exp(-(1.0 - eta) * times) * (
        1.0 - np.exp(-gamma * eta * times / 2.0)
    )


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
            left = _eval(expr.left)
            right = _eval(expr.right)
            if isinstance(expr.op, ast.Add):
                return left + right
            if isinstance(expr.op, ast.Sub):
                return left - right
            if isinstance(expr.op, ast.Mult):
                return left * right
            return left / right
        raise argparse.ArgumentTypeError(
            f"Unsupported angle expression '{raw}'. Allowed: numbers, pi, np.pi, + - * / and parentheses."
        )

    return float(_eval(node))


# Main function to parse arguments, run the simulation, and save results.
def make_parser() -> argparse.ArgumentParser:

    #Create an argument parser for the script, allowing the user to specify various parameters such as the initial state, measurement efficiency, number of trajectories, and output options.
    parser = argparse.ArgumentParser(description="N=2 homodyne simulation (Jz)")

    #The user can choose the initial state from predefined options (plusplus, ee, css) and specify parameters for the CSS state if chosen. 
    #They can also set the measurement efficiency (eta), the phases of the homodyne detection, and various options for the simulation and output.
    parser.add_argument("--state", choices=["plusplus", "ee", "css"], default="plusplus")
    parser.add_argument("--theta", type=float, default=np.pi / 2.0, help="Used when --state css")
    parser.add_argument("--phi-state", type=float, default=0.0, help="Used when --state css")
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--phi1", type=parse_angle_arg, default=0.0, help="Angle (e.g. 0, pi/2, np.pi/2)")
    parser.add_argument("--phi2", type=parse_angle_arg, default=0.0, help="Angle (e.g. 0, pi/2, np.pi/2)")

    #The user can specify a list of measurement efficiencies (etas) for both channels
    parser.add_argument("--etas_1", type=str, default="1")
    parser.add_argument("--etas_2", type=str, default="1,0.9,0.7,0.5,0.3")
    parser.add_argument("--ntraj", type=int, default=20)
    #parser.add_argument("--chunk-size", type=int, default=500)
    parser.add_argument("--t-end", type=float, default=5.0, help="Final time in units of T1")
    parser.add_argument("--dt", type=float, default=0.001, help="Time step in units of T1")
    parser.add_argument("--num-cpus", type=int, default=max(1, os.cpu_count() - 1))
    parser.add_argument(
        "--eta-jobs",
        type=int,
        default=1,
        help="Number of parallel joblib workers over eta values",
    )
    parser.add_argument(
        "--joblib-backend",
        choices=["loky", "threading", "multiprocessing"],
        default="loky",
        help="joblib backend for eta-level parallelism",
    )
    parser.add_argument(
        "--joblib-verbose",
        type=int,
        default=0,
        help="joblib verbosity level (0 disables)",
    )
    parser.add_argument(
        "--eta-index",
        type=int,
        default=None,
        help="Run only one eta by index in --etas (useful for Slurm job arrays)",
    )
    parser.add_argument(
        "--use-slurm-array",
        action="store_true",
        help="Read eta index from SLURM_ARRAY_TASK_ID if --eta-index is not set",
    )
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument(
        "--out-root",
        type=str,
        default=r"./Graphs/Jz_2_Homodyne_N2",
    )
    parser.add_argument(
        "--no-theory",
        action="store_true",
        help="Disable theoretical concurrence overlay",
    )
    parser.add_argument(
        "--usetex",
        action="store_true",
        help="Enable LaTeX rendering in plots (requires a LaTeX installation).",
    )
    return parser


#Dentro il main creo lo script principale che esegue la simulazione.
def main() -> None:

    t0 = time.perf_counter()
    print("PY_START", flush=True)

    #Prendo gli argomenti dalla riga di comando, analizzo la lista di etas e gestisco l'indice eta se specificato o se si sta usando un array di Slurm.
    args = make_parser().parse_args()
    etas_all = parse_etas(args.etas_1, args.etas_2)
    eta_index = args.eta_index

    #Tutta roba per gestire l'esecuzione di un singolo eta quando si usa un array di Slurm. 
    #Se --eta-index non è specificato ma --use-slurm-array è attivo, cerchiamo di leggere l'indice eta da SLURM_ARRAY_TASK_ID.
    # if eta_index is None and args.use_slurm_array:
    #     slurm_idx = os.getenv("SLURM_ARRAY_TASK_ID", "").strip()
    #     if slurm_idx == "":
    #         raise ValueError("--use-slurm-array set but SLURM_ARRAY_TASK_ID is missing")
    #     eta_index = int(slurm_idx)

    # #
    # if eta_index is not None:
    #     if eta_index < 0 or eta_index >= len(etas_all):
    #         raise ValueError(
    #             f"Invalid eta index {eta_index}. Valid range: [0, {len(etas_all) - 1}]"
    #         )
    #     etas = [etas_all[eta_index]]
    #     print(f"[array] Running eta index {eta_index} -> eta={etas[0]:g}")
    # else:
    etas = etas_all

    #Build the initial state based on the specified kind and parameters, and convert it to a density matrix for the simulation.
    psi0, theta_for_theory = build_initial_state(args.state, args.theta, args.phi_state)
    rho0 = ket2dm(psi0)
    gamma = float(args.gamma)
    t1 = 1.0 / gamma
    times = np.arange(0.0, args.t_end * t1, args.dt * t1)
    columns = [name for name, _ in DEFAULT_EOPS]
    e_ops = [op for _, op in DEFAULT_EOPS]

    #Determine the number of parallel jobs to run over the eta values. 
    #If eta_jobs is greater than 1, we will use joblib to parallelize the simulations across different eta values. We also need to ensure that we don't use more CPUs than available, and if we're parallelizing over eta, we should set num_cpus to 1 for each individual simulation to avoid nested parallelism.
    eta_jobs = max(1, int(args.eta_jobs))
    if eta_jobs > len(etas):
        eta_jobs = len(etas)

    #Calculate the effective number of CPUs to use for each simulation. 
    #If we're parallelizing over eta, we should set num_cpus to 1 for each simulation to avoid nested parallelism, since each simulation may already be using multiple CPUs internally.
    num_cpus_eff = int(args.num_cpus)
    if eta_jobs > 1 and num_cpus_eff > 1:
        print(
            f"[joblib] eta-jobs={eta_jobs}: forcing --num-cpus to 1 per eta "
            f"(requested {num_cpus_eff}) to avoid nested parallelism."
        )
        num_cpus_eff = 1

    if eta_jobs > 1 and Parallel is None:
        raise ImportError(
            "joblib is required for --eta-jobs > 1. Install with: pip install joblib"
        )

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

    eta_jobs_used = 1
    joblib_backend_used = "serial"

    if eta_jobs == 1:
        results = run_serial()
    else:
        eta_jobs_used = eta_jobs
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
                print(
                    "[joblib] backend='threading' failed with PermissionError. "
                    "Falling back to serial eta loop."
                )
                eta_jobs_used = 1
                joblib_backend_used = "serial_fallback"
                results = run_serial()
            else:
                print(
                    f"[joblib] backend='{args.joblib_backend}' failed with PermissionError. "
                    "Retrying with backend='threading'."
                )
                try:
                    results = run_parallel("threading")
                    joblib_backend_used = "threading"
                except PermissionError:
                    print("[joblib] backend='threading' also failed. Falling back to serial eta loop.")
                    eta_jobs_used = 1
                    joblib_backend_used = "serial_fallback"
                    results = run_serial()

    #Organize the results into a dictionary mapping eta values to DataFrames, and prepare the output directory and metadata for saving the results.
    ineff_df_map = {key: df for key, df in results}
    ineff_df: dict[str, pd.DataFrame] = {}

    #We create a new dictionary ineff_df that will contain only the eta values specified in the etas list, mapping each eta to its corresponding DataFrame from the results. This allows us to easily access the results for each eta when plotting and saving.
    for eta_1, eta_2 in etas:
        key = eta_pair_key(float(eta_1), float(eta_2))
        ineff_df[key] = ineff_df_map[key]

    #Compute xi^2_WIN from already computed xi^2_KU and |<J>| to avoid re-evaluating xi^2_KU in the solver.
    N_spin = 2.0
    tol_norm = 1e-12
    for key in ineff_df:
        ku_vals = ineff_df[key]["Xi2_KU"].to_numpy(dtype=float)
        norm_vals = ineff_df[key]["Norm_J"].to_numpy(dtype=float)
        xi_win = ku_vals.copy()
        mask = norm_vals > tol_norm
        xi_win[mask] = ku_vals[mask] * ((N_spin / 2.0) / norm_vals[mask]) ** 2
        ineff_df[key]["Xi2_WIN"] = xi_win
    if "Xi2_WIN" not in columns:
        columns = columns + ["Xi2_WIN"]

    #Create the output directory based on the parameters of the simulation, including the phases phi1 and phi2, and optionally the eta index if specified.
    phi1_dir = angle_to_path(float(args.phi1))
    phi2_dir = angle_to_path(float(args.phi2))
    out_dir = Path(args.out_root) / f"ntraj={int(args.ntraj)}__phi_1={phi1_dir}__phi_2={phi2_dir}"
    if eta_index is not None:
        eta_key = eta_pair_key(float(etas[0][0]), float(etas[0][1]))
        out_dir = out_dir / f"eta_idx={eta_index}__eta={eta_key}"
    out_dir.mkdir(parents=True, exist_ok=True)

    #Prepare metadata about the simulation run, including the parameters used and the structure of the results, to be saved alongside the data for reference.
    meta = {
        "measurement": "homodyne",
        "N": 2,
        "state": args.state,
        "theta": float(args.theta),
        "phi_state": float(args.phi_state),
        "gamma": gamma,
        "phi1": float(args.phi1),
        "phi2": float(args.phi2),
        "etas_all": etas_all,
        "etas_1": [float(eta_1) for eta_1, _ in etas_all],
        "etas_2": [float(eta_2) for _, eta_2 in etas_all],
        "eta_index": eta_index,
        "use_slurm_array": bool(args.use_slurm_array),
        "ntraj": int(args.ntraj),
        "num_cpus_requested": int(args.num_cpus),
        "num_cpus_per_eta": int(num_cpus_eff),
        "eta_jobs": int(eta_jobs_used),
        "joblib_backend": args.joblib_backend,
        "joblib_backend_used": joblib_backend_used,
        "dt_T1": float(args.dt),
        "t_end_T1": float(args.t_end),
        "columns": columns,
    }

    #npz_path = out_dir / f"homodyne_phi1={args.phi1:.6g}_phi2={args.phi2:.6g}.npz"
    #save_ineff_df_npz(ineff_df, npz_path, meta=meta, step_col="step")

    plt.rcParams["text.usetex"] = bool(args.usetex)
    plt.rcParams.update(
        {
            "mathtext.fontset": "cm",
            "font.family": "serif",
            "font.size": 14,
            "axes.unicode_minus": False,
        }
    )

    #Convert the angles phi1 and phi2 to LaTeX strings for use in the plot titles, using the angle_to_tex function defined earlier.
    phi1_tex = angle_to_tex(float(args.phi1))
    phi2_tex = angle_to_tex(float(args.phi2))

    #Plot the results for each quantity in the columns list, creating a separate figure for each quantity and overlaying the theoretical concurrence curve if applicable and not disabled by the user.
    x = times / t1

    #Plot the results for each quantity in the columns list, creating a separate figure for each quantity and overlaying the theoretical concurrence curve if applicable and not disabled by the user.
    for col in columns:
        plt.figure(figsize=(12, 8))

        #Every eta gets its own curve in the plot, and we label it accordingly. 
        #If the column being plotted is "Conc" and the user has not disabled the theoretical overlay
        for eta_1, eta_2 in etas:
            key = eta_pair_key(float(eta_1), float(eta_2))
            y = ineff_df[key][col].to_numpy()
            line, = plt.plot(
                x,
                y,
                label=rf"$\eta_1 = {eta_1:g},\ \eta_2 = {eta_2:g}$",
            )

            # if col == "Conc" and not args.no_theory and np.isclose(eta_1, eta_2):
            #     y_th = theoretical_concurrence_curve(
            #         times=times,
            #         gamma=gamma,
            #         eta=float(eta_1),
            #         theta=theta_for_theory,
            #     )
            #     plt.plot(
            #         x,
            #         y_th,
            #         "--",
            #         color=line.get_color(),
            #         alpha=0.55,
            #         linewidth=2,
            #     )

        #Set the x and y limits, labels, title, grid, and legend for the plot, and save it to a PDF file in the output directory.
        plt.xlim(0.0, float(args.t_end))
        #plt.ylim(0.0, 1.01)
        if col == "Xi2_WIN":
            plt.axhline(1.0, linestyle="--", linewidth=2, color="black", alpha=0.7)
            plt.ylim(top=10.0)

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

    run_config_path = out_dir / "run_config.json"
    run_config_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"OUT_DIR={out_dir}")
    print(f"RUN_CONFIG_PATH={run_config_path}")
    print(f"Done. Results saved in: {out_dir}")

    dt = time.perf_counter() - t0
    print(f"PY_TOTAL_SECONDS={dt:.3f}", flush=True)


if __name__ == "__main__":
    main()
