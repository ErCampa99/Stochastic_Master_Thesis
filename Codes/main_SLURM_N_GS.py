"""
Homodyne single dephasing simulation for N atoms, with Jz measurement as plus channel.
This script simulates the homodyne detection of a collective dephasing process for N atoms using the QuTiP library. The measurement is performed on the Jz operator, which corresponds to the collective spin along the z-axis. 
The script allows for varying the measurement efficiency (eta) and other parameters, and it computes expectation values such as xi^2_KU and variance of Jz over time.
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
I_2 = qeye(2)

# Function to create a coherent spin state (CSS) for N atoms based on the given angles theta and phi.
def css_n(theta: float, phi: float, N: int):
    single_qubit = (
        np.sin(theta / 2.0) * basis(2, 0)
        + np.exp(1j * phi) * np.cos(theta / 2.0) * basis(2, 1)
    )
    return tensor([single_qubit for _ in range(N)])


# Function to build the product state |+,...,+> for N atoms.
def plus_state_n(N: int):
    return tensor([plus_state for _ in range(N)])

# Free Hamiltonian, Omega = 0
OMEGA = 0.0
N_ATOMS = 3

# Collective spin operators and channels for N atoms.
J_x = None
J_y = None
J_z = None
H_free = None
U_GS = None
Z_local = []
C_channels = []


def local_operator(single_op, site: int, N: int):
    ops = [I_2 for _ in range(N)]
    ops[site] = single_op
    return tensor(ops)


def collective_spin_operator(single_op, N: int):
    total = local_operator(single_op, 0, N)
    for site in range(1, N):
        total = total + local_operator(single_op, site, N)
    return 0.5 * total

#Want to set first row to all 1's, then the next N rows to the identity matrix, and then apply Gram-Schmidt to get an orthonormal basis.
def gram_schmidt_unitary(N: int, tol: float = 1e-12):
    collective_row = np.ones(N, dtype=complex) / np.sqrt(N)
    rows = [collective_row]
    seeds = [np.eye(N, dtype=complex)[k] for k in range(N)]
    for vec in seeds:
        w = vec.astype(complex).copy()
        for u in rows:
            w = w - np.vdot(u, w) * u
        norm = np.linalg.norm(w)
        if norm > tol:
            rows.append(w / norm)
        if len(rows) == N:
            break
    if len(rows) != N:
        raise RuntimeError(f"Unable to build Gram-Schmidt unitary for N={N}.")
    return np.asarray(rows, dtype=complex)

def check_gs_unitarity_and_collective_row(U: np.ndarray):
    N = U.shape[0]
    I = np.eye(N, dtype=complex)
    UUdag = U @ U.conj().T
    unitarity_error = np.linalg.norm(UUdag - I)

    collective_row = np.ones(N, dtype=complex) / np.sqrt(N)
    first_row_error = np.linalg.norm(U[0, :] - collective_row)
    return unitarity_error, first_row_error, collective_row


def setup_system_operators(N: int):
    global N_ATOMS, J_x, J_y, J_z, H_free, U_GS, Z_local, C_channels

    N_ATOMS = int(N)
    J_x = collective_spin_operator(sigmax(), N_ATOMS)
    J_y = collective_spin_operator(sigmay(), N_ATOMS)
    J_z = collective_spin_operator(sigmaz(), N_ATOMS)
    H_free = OMEGA * J_z

    #To construct the collective measurement operators for N atoms based on the local Z operators and the Gram-Schmidt unitary transformation (U_GS).
    Z_local = [local_operator(sigmaz(), site, N_ATOMS) for site in range(N_ATOMS)]
    U_GS = gram_schmidt_unitary(N_ATOMS)

    #The collective measurement operators are obtained by applying the Gram-Schmidt unitary transformation to the local Z operators.
    C_channels = []
    for i in range(N_ATOMS):
        Ci = U_GS[i, 0] * Z_local[0]
        for j in range(1, N_ATOMS):
            Ci = Ci + U_GS[i, j] * Z_local[j]
        C_channels.append(Ci)


#To create the collapsing operators for N atoms based on the measurement efficiency (eta) and the phases (phi) of the homodyne detection.
def collapsing_operators(gamma: float, phis: list[float], eta_1: float, eta_other_channels: float):

    #Check that the number of phase values provided matches the number of atoms, since each atom has its own phase for the homodyne detection.
    if len(phis) != N_ATOMS:
        raise ValueError(f"Expected {N_ATOMS} phase values, got {len(phis)}.")
    
    #Combining efficiency and phase information to construct the collapsing operators for each atom. 
    #The first atom uses eta_1, while the others use eta_other_channels.
    etas = [eta_1] + [eta_other_channels] * (N_ATOMS - 1)
    return [
        #
        np.sqrt(gamma * etas[k]) * np.exp(1j * phis[k]) * C_channels[k]
        for k in range(N_ATOMS)
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

#Cache collective spin operators by N to avoid rebuilding them inside solver callbacks.
_SPIN_OP_CACHE = {}

def get_collective_spin_ops_cached(N: int):
    N = int(N)
    if N not in _SPIN_OP_CACHE:
        _SPIN_OP_CACHE[N] = (
            collective_spin_operator(sigmax(), N),
            collective_spin_operator(sigmay(), N),
            collective_spin_operator(sigmaz(), N),
        )
    return _SPIN_OP_CACHE[N]

#Function to evaluate the expectation values of the collective spin components J_x, J_y, and J_z for a given state. 
# This is used to compute the mean spin vector and the covariance matrix needed for the xi^2_KU calculation.
def eval_spin_components(state):
    dim = state.shape[0]
    N = int(round(np.log2(dim)))
    Jx_op, Jy_op, Jz_op = get_collective_spin_ops_cached(N)
    return expect(Jx_op, state), expect(Jy_op, state), expect(Jz_op, state)

#To compute the spin-squeezing parameter xi^2_KU based on the method described by Kitagawa and Ueda.
#It uses MSD to compute the mean spin vector and the covariance matrix of the spin components, and then calculates the minimum variance in the plane perpendicular to the mean spin direction. 
#The result is then normalized by N/4 to give xi^2_KU.
def xi_KU_solver(t, rho, N=None, tol=1e-12):

    # If N is not provided, infer it from the dimension of the state. For N=2, the dimension is 4, so N=log2(4)=2.
    if N is None:
        dim = rho.shape[0]
        # For a system of N qubits, the dimension of the Hilbert space is 2^N. Therefore, we can infer N by taking the logarithm base 2 of the dimension.
        N = int(round(np.log2(dim)))

    Jx_op, Jy_op, Jz_op = get_collective_spin_ops_cached(N)

    #Calculate the mean spin vector and the covariance matrix of the spin components. 
    Jx_exp, Jy_exp, Jz_exp = expect(Jx_op, rho), expect(Jy_op, rho), expect(Jz_op, rho)
    Jmean = np.array(
        [
            float(np.real_if_close(Jx_exp)),
            float(np.real_if_close(Jy_exp)),
            float(np.real_if_close(Jz_exp)),
        ]
    )
    #The norm of the MSD
    m = np.linalg.norm(Jmean)

    Js = [Jx_op, Jy_op, Jz_op]
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
    return s if n > 0 else "-"

#===================================================================================================================
#===================================================================================================================
#===================================================================================================================

# Define the default expectation value operators to compute during the simulation.
DEFAULT_EOPS = [
    ("Xi2_KU", xi_KU_solver),
    ("Norm_J", norm_J),
]

# Define LaTeX labels for the quantities being plotted.
LABELS = {
    "Xi2_KU": r"$\overline{\xi^2_{KU}}$",
    "Norm_J": r"$\overline{|\langle \mathbf{J} \rangle|}$",
    "Xi2_WIN": r"$\overline{\xi^2_{WIN}}$",
}

def eta_triplet_key(eta_1: float, eta_other_channels: float) -> str:
    return f"{eta_1:g}_{eta_other_channels:g}"

# Utility function to parse the list of etas from a comma-separated string.
def parse_etas(raw_1: str, raw_other_channels: str) -> list[tuple[float, float]]:
    #The function takes a comma-separated string of eta values, splits it, and converts each value to a float. 
    #It also checks that the list is not empty and that each eta value is between 0 and 1.

    #Parse the comma-separated strings for etas_1 and etas_other_channels, convert them to lists of floats, and validate the values.#
    etas_1 = [float(x.strip()) for x in raw_1.split(",") if x.strip()]
    etas_other_channels = [float(x.strip()) for x in raw_other_channels.split(",") if x.strip()]
    if not etas_1 or not etas_other_channels:
        raise ValueError("Eta list is empty. Use --etas_1 and --etas_other_channels with comma-separated values.")

    if len(etas_1) == 1 and len(etas_other_channels) > 1:
        etas_1 = etas_1 * len(etas_other_channels)
    elif len(etas_other_channels) == 1 and len(etas_1) > 1:
        etas_other_channels = etas_other_channels * len(etas_1)
    elif len(etas_1) != len(etas_other_channels):
        raise ValueError(
            f"Length mismatch: len(etas_1)={len(etas_1)} and len(etas_other_channels)={len(etas_other_channels)}. "
            "Use equal lengths, or one list with a single value to broadcast."
        )

    for eta_1, eta_other_channels in zip(etas_1, etas_other_channels):
        if eta_1 < 0.0 or eta_1 > 1.0 or eta_other_channels < 0.0 or eta_other_channels > 1.0:
            raise ValueError(f"Invalid etas=({eta_1}, {eta_other_channels}). Each eta must be in [0, 1].")
    return list(zip(etas_1, etas_other_channels))

# Utility function to build the initial state based on the specified kind and parameters.
def build_initial_state(kind: str, theta: float, phi_state: float, N: int):
    if kind == "PlusStateN":
        return plus_state_n(N), np.pi / 2.0
    raise ValueError(f"Unknown state kind: {kind}")


# Function to run the homodyne simulation in chunks and average the results.
def run_homodyne(
        
    #List of all parameters needed for the simulation
    rho0,
    times: np.ndarray,
    gamma: float,
    phi1: float,
    phi_other_channels: float,
    eta_1: float,
    eta_other_channels: float,
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


    phis = [float(phi1)] + [float(phi_other_channels)] * (N_ATOMS - 1)
    c_ops_unobs = collapsing_operators(gamma, phis, 1.0 - eta_1, 1.0 - eta_other_channels)
    c_ops_obs = collapsing_operators(gamma, phis, eta_1, eta_other_channels)

    try:
        sol = smesolve(
            H_free,
            rho0,
            times,
            c_ops=c_ops_unobs,
            sc_ops=c_ops_obs,
            heterodyne=False,
            e_ops=e_ops,
            ntraj=ntraj,
            options=options,
        )
    except (TypeError, PermissionError) as exc:
        if options["map"] == "parallel":
            print(
                f"[smesolve] map='parallel' failed ({type(exc).__name__}: {exc}). "
                "Retrying with map='serial' and num_cpus=1."
            )
            options["map"] = "serial"
            options["num_cpus"] = 1
            sol = smesolve(
                H_free,
                rho0,
                times,
                c_ops=c_ops_unobs,
                sc_ops=c_ops_obs,
                heterodyne=False,
                e_ops=e_ops,
                ntraj=ntraj,
                options=options,
            )
        else:
            raise

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
    eta_other_channels: float,
    num_atoms: int,
    rho0,
    times: np.ndarray,
    gamma: float,
    phi1: float,
    phi_other_channels: float,
    e_ops: list,
    columns: list[str],
    ntraj: int,
    #chunk_size: int,
    num_cpus: int,
    #seed: int,
) -> tuple[str, pd.DataFrame]:
    if H_free is None or N_ATOMS != int(num_atoms):
        setup_system_operators(int(num_atoms))

    avg_expect = run_homodyne(
        rho0=rho0,
        times=times,
        gamma=gamma,
        phi1=phi1,
        phi_other_channels=phi_other_channels,
        eta_1=float(eta_1),
        eta_other_channels=float(eta_other_channels),

        e_ops=e_ops,
        ntraj=ntraj,
        num_cpus=num_cpus,
    )
    key = eta_triplet_key(float(eta_1), float(eta_other_channels))
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
    parser = argparse.ArgumentParser(description="N-atoms homodyne simulation (Jz, Gram-Schmidt unitary)")

    #The user can choose the initial state and set the measurement efficiency (eta), the phases of the homodyne detection, and various options for the simulation and output.
    #They can also set the measurement efficiency (eta), the phases of the homodyne detection, and various options for the simulation and output.
    parser.add_argument("--N", type=int, default=3)
    parser.add_argument("--state", choices=["PlusStateN"], default="PlusStateN")
    parser.add_argument("--theta", type=float, default=np.pi / 2.0, help="Used when --state css")
    parser.add_argument("--phi-state", type=float, default=0.0, help="Used when --state css")
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--phi1", type=parse_angle_arg, default=0.0, help="Angle (e.g. 0, pi/2, np.pi/2)")
    parser.add_argument("--phi-other-channels", type=parse_angle_arg, default=0.0, help="Angle for channels 2..N (e.g. 0, pi/2, np.pi/2)")

    #The user can specify a list of measurement efficiencies (etas) for both channels
    parser.add_argument("--etas_1", type=str, default="1")
    parser.add_argument("--etas_other_channels", type=str, default="1,0.8,0.5,0.3,0")
    parser.add_argument("--ntraj", type=int, default=20)
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
        default=r"./Graphs/Jz_Homodyne_N_GS",
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
    etas_all = parse_etas(args.etas_1, args.etas_other_channels)
    eta_index = args.eta_index
    num_atoms = int(args.N)
    if num_atoms < 2:
        raise ValueError("N must be >= 2.")
    setup_system_operators(num_atoms)
    unitarity_error, first_row_error, collective_row = check_gs_unitarity_and_collective_row(U_GS)
    print(f"GS_CHECK_N={num_atoms}")
    print(f"GS_UNITARITY_ERR={unitarity_error:.3e}")
    print(f"GS_COLLECTIVE_ROW_ERR={first_row_error:.3e}")
    print("GS_FIRST_ROW=" + np.array2string(U_GS[0, :], precision=6, suppress_small=True))
    print("GS_COLLECTIVE=" + np.array2string(collective_row, precision=6, suppress_small=True))

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
    psi0, theta_for_theory = build_initial_state(args.state, args.theta, args.phi_state, num_atoms)
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
                eta_other_channels=float(eta_other_channels),
                num_atoms=num_atoms,
                rho0=rho0,
                times=times,
                gamma=gamma,
                phi1=float(args.phi1),
                phi_other_channels=float(args.phi_other_channels),
                e_ops=e_ops,
                columns=columns,
                ntraj=int(args.ntraj),
                num_cpus=num_cpus_eff,
            )
            for eta_1, eta_other_channels in etas
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
                    eta_other_channels=float(eta_other_channels),
                    num_atoms=num_atoms,
                    rho0=rho0,
                    times=times,
                    gamma=gamma,
                    phi1=float(args.phi1),
                    phi_other_channels=float(args.phi_other_channels),
                    e_ops=e_ops,
                    columns=columns,
                    ntraj=int(args.ntraj),
                    num_cpus=num_cpus_eff,
                )
                for eta_1, eta_other_channels in etas
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
    for eta_1, eta_other_channels in etas:
        key = eta_triplet_key(float(eta_1), float(eta_other_channels))
        ineff_df[key] = ineff_df_map[key]

    #Compute xi^2_WIN from already computed xi^2_KU and |<J>| to avoid re-evaluating xi^2_KU in the solver.
    N_spin = float(num_atoms)
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

    #Create the output directory based on the parameters of the simulation, including the phases phi1 and phi_other, and optionally the eta index if specified.
    phi1_dir = angle_to_path(float(args.phi1))
    phi_other_dir = angle_to_path(float(args.phi_other_channels))
    out_dir = Path(args.out_root) / f"N={num_atoms}__ntraj={int(args.ntraj)}__phi_1={phi1_dir}__phi_other={phi_other_dir}"
    if eta_index is not None:
        eta_key = eta_triplet_key(float(etas[0][0]), float(etas[0][1]))
        out_dir = out_dir / f"eta_idx={eta_index}__eta={eta_key}"
    out_dir.mkdir(parents=True, exist_ok=True)

    #Prepare metadata about the simulation run, including the parameters used and the structure of the results, to be saved alongside the data for reference.
    meta = {
        "measurement": "homodyne",
        "N": num_atoms,
        "state": args.state,
        "gs_unitarity_error": float(unitarity_error),
        "gs_collective_row_error": float(first_row_error),
        "gs_first_row_real": [float(np.real(x)) for x in U_GS[0, :]],
        "gs_first_row_imag": [float(np.imag(x)) for x in U_GS[0, :]],
        "gs_collective_row_real": [float(np.real(x)) for x in collective_row],
        "gs_collective_row_imag": [float(np.imag(x)) for x in collective_row],
        "theta": float(args.theta),
        "phi_state": float(args.phi_state),
        "gamma": gamma,
        "phi1": float(args.phi1),
        "phi_other_channels": float(args.phi_other_channels),
        "etas_all": etas_all,
        "etas_1": [float(eta_1) for eta_1, _ in etas_all],
        "etas_other_channels": [float(eta_other_channels) for _, eta_other_channels in etas_all],
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

    plt.rcParams["text.usetex"] = bool(args.usetex)
    plt.rcParams.update(
        {
            "mathtext.fontset": "cm",
            "font.family": "serif",
            "font.size": 14,
            "axes.unicode_minus": False,
        }
    )

    #Convert the angles phi1 and phi_other to LaTeX strings for use in the plot titles, using the angle_to_tex function defined earlier.
    phi1_tex = angle_to_tex(float(args.phi1))
    phi_other_tex = angle_to_tex(float(args.phi_other_channels))

    #Plot the results for each quantity in the columns list, creating a separate figure for each quantity and overlaying the theoretical concurrence curve if applicable and not disabled by the user.
    x = times / t1

    #Plot the results for each quantity in the columns list, creating a separate figure for each quantity and overlaying the theoretical concurrence curve if applicable and not disabled by the user.
    for col in columns:
        plt.figure(figsize=(12, 8))

        #Every eta gets its own curve in the plot, and we label it accordingly. 
        #If the column being plotted is "Conc" and the user has not disabled the theoretical overlay
        for eta_1, eta_other_channels in etas:
            key = eta_triplet_key(float(eta_1), float(eta_other_channels))
            y = ineff_df[key][col].to_numpy()
            line, = plt.plot(
                x,
                y,
                label=rf"$\eta_1 = {eta_1:g},\ \eta_{{other}} = {eta_other_channels:g}$",
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
            rf"Homodyne $J_z$ (N={num_atoms}): "
            + LABELS[col]
            + rf"$,\ \phi_1={phi1_tex}\ \phi_{{other}}={phi_other_tex}$"
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
