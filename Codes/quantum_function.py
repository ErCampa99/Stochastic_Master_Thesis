import numpy as np
import matplotlib.pyplot as plt
from qutip import *
import pandas as pd
import os
from pathlib import Path

def flatten_to_qudit(state):
    """
    Prende un ket bipartito (Qobj con dims=[[2,2],[1,1]])
    e lo trasforma in un ket di un sistema a 4 livelli
    (Qobj con dims=[[4],[1]]).
    """
    return Qobj(state.full(), dims=[[4], [1]]).unit()

def css_2(theta, phi):
    single_qubit = (np.sin(theta/2) * basis(2,0) + np.exp(1j*phi) * np.cos(theta/2) * basis(2,1))
    return tensor(single_qubit, single_qubit)

#=================================================================================================================================
# //-- UTILITIES STATI --// ======================================================================================================
#=================================================================================================================================

# Stati base_1-Qubit
exc = qutip.basis(2, 0)  # ground
gnd = qutip.basis(2, 1)  # excited

plus_state = (exc + gnd).unit()  # |+>
minus_state = (exc - gnd).unit()  # |->

PlusPlus = tensor(plus_state, plus_state)  # |++>

#Stati base 2-Qubit
ee = tensor(exc, exc)  # |ee>   
eg = tensor(exc, gnd)  # |eg>
ge = tensor(gnd, exc)  # |ge>
gg = tensor(gnd, gnd)  # |gg>

#Stati di Bell
psi_plus = (tensor(gnd, exc) + tensor(exc, gnd)).unit()
psi_minus = (tensor(exc, gnd) - tensor(gnd, exc)).unit()
phi_plus = (tensor(gnd, gnd) + tensor(exc, exc)).unit()
phi_minus = (tensor(gnd, gnd) - tensor(exc, exc)).unit()

#Stati base 2-Qubit --- FLATTENED ---
ee_f = flatten_to_qudit(tensor(exc, exc))  # |ee>
eg_f = flatten_to_qudit(tensor(exc, gnd))  # |eg>
ge_f = flatten_to_qudit(tensor(gnd, exc))  # |ge>
gg_f = flatten_to_qudit(tensor(gnd, gnd))  # |gg>

#Stati di Bell --- FLATTENED ---
psi_plus_f = flatten_to_qudit(tensor(gnd, exc) + tensor(exc, gnd))
psi_minus_f = flatten_to_qudit(tensor(exc, gnd) - tensor(gnd, exc))
phi_plus_f = flatten_to_qudit(tensor(gnd, gnd) + tensor(exc, exc))
phi_minus_f = flatten_to_qudit(tensor(gnd, gnd) - tensor(exc, exc)) 

# Computational basis
comp_basis = [ee, eg, ge, gg]
comp_basis_f = [ee_f, eg_f, ge_f, gg_f]

# Bell basis
bell_states = [[phi_plus, psi_plus], [phi_minus, psi_minus] ]
bell_states_f = [[phi_plus_f, psi_plus_f], [phi_minus_f, psi_minus_f] ]

#=================================================================================================================================
# //-- OPERATORS --// ============================================================================================================  
#=================================================================================================================================

OMEGA = 0
I_2 = qeye(2)  # identity operator for 2-level system
sm = sigmam()  # lowering operator for 2-level system (atom)

Sz = 0.5*sigmaz()  # Pauli Z operator for 2-level system (atom)

Sz_1 = tensor(Sz, qeye(2))
Sz_2 = tensor(qeye(2), Sz)


sigma_minus_1 = tensor(sm, I_2) # lowering operator for atom 1
sigma_minus_2 = tensor(I_2, sm) # lowering operator for atom 2
sigma_minus_12 = tensor(sigma_minus_1, sigma_minus_2) # combined lowering operator for both atoms    

sigma_plus_1 = sigma_minus_1.dag() # raising operator for atom 1
sigma_plus_2 = sigma_minus_2.dag() # raising operator for atom       
sigma_plus_12 = tensor(sigma_plus_1, sigma_plus_2) # raising operator for both atoms

sigma_plus_list = [sigma_plus_1, sigma_plus_2] # list of raising operators for both atoms

H_free_atom_1 = 0.5 * OMEGA * tensor(sigmaz(), I_2) # Free Hamiltonian atom 1
H_free_atom_2 = 0.5 * OMEGA * tensor(I_2, sigmaz()) # Free Hamiltonian atom 2
H_free = H_free_atom_1 + H_free_atom_2 # Free Hamiltonian for both atoms        


#=================================================================================================================================
# //-- FUNZIONI PER SOLVER --// ==================================================================================================
#=================================================================================================================================


def pop_excited(state):
    return abs(state[0]**2)

def pop_excited_for_solver(t, state):
    return abs(state[0]**2)

def theo_concurrence(t, gamma):
    return 2*np.exp(-gamma*t)*(1-np.exp(-gamma*t)) 

def concurrence_pure_state(state): 
    ###Compute the concurrence for a two-qubit pure state
    return 2*np.abs(state[0]*state[3]-state[1]*state[2])

def concurrence_for_solver_pure(t, state):
    # psi is a Qobj, we need the data
    # .full() returns a 2D column vector, .flatten() makes it a 1D array
    c = state.full().flatten()
    return 2 * np.abs(c[0] * c[3] - c[1] * c[2])

def concurrence_for_solver(t, state):
    return concurrence(state)

def compute_concurrences(solution):
    ntraj = len(solution.runs_states)
    n_times = len(solution.runs_states[0])  # si assume che tutte le traiettorie abbiano stesso numero di time step

    conc_array = np.zeros((ntraj, n_times))

    for traj_idx in range(ntraj):
        for t_idx, state in enumerate(solution.runs_states[traj_idx]):

            if state.isket:
                conc_array[traj_idx, t_idx] = concurrence_pure_state(state) #Use fast method for pure states
            else:               
                conc_array[traj_idx, t_idx] = concurrence(state)

    return conc_array


def concurrence_for_solver_general(t, state):

    if state.isket:
        conc = concurrence_pure_state(state)

    else:
        sysy = tensor(sigmay(), sigmay())

        rho_tilde = (state * sysy) * (state.conj() * sysy)

        evals = rho_tilde.eigenenergies()

        # abs to avoid problems with sqrt for very small negative numbers
        evals = abs(np.sort(np.real(evals)))

        sqrt_evals = np.sqrt(evals)
        lsum = sqrt_evals[3] - sqrt_evals[2] - sqrt_evals[1] - sqrt_evals[0]
        conc = np.maximum(0, lsum)

    return float(conc)

def energy_solver(t, state):
    """Calcola l'energia media dell'hamiltoniana libera."""
    return expect(Sz_1+Sz_2, state)


#=================================================================================================================================
# //-- KRAUS OPERATORS --// ======================================================================================================
#=================================================================================================================================


def kraus_probabilities(rho, kraus_ops):
    """
    Calcola le probabilità associate a una lista di Kraus operators su uno stato rho.

    Parametri:
    - rho : qutip.Qobj, stato (ket o matrice densità)
    - kraus_ops : lista di qutip.Qobj, operatori di Kraus
    
    Ritorna:
    - probs : lista di float, probabilità di ogni risultato
    """
    probs = []
    for M in kraus_ops:
        if isket(rho):
            #print("Ket")
            # Caso stato puro |psi>
            p = (M * rho).norm()**2
        else:
            print("Density matrix")
            # Caso matrice densità
            p = abs((M.dag() * M * rho).tr())
        probs.append(p)

    #Normalize for safety (sum = 1)
    probs = np.array(probs)
    probs = probs / probs.sum()

    return probs


def sample_outcome(probs):
    """
    Estrae un outcome di misura a partire da una lista di probabilità.

    Parametri:
    - probs : lista di float, probabilità associate a ciascun outcome

    Ritorna:
    - outcome : int, indice dell'operatore misurato
    """
    # normalizza per sicurezza (somma = 1)
    probs = np.array(probs)
    probs = probs / probs.sum()
    
    # estrae un indice basato sulla distribuzione
    outcome = np.random.choice(len(probs), p=probs)
    if outcome is None:
        raise ValueError("Nessun outcome estratto. Controlla le probabilità.")
    
    #if outcome==1:
        #print("Emissione fotone da atomo 1 - psi plus")
    
    #if outcome==2:
        #print("Emissione fotone da atomo 2 - psi minus")    
    
    return outcome

def measure_and_update(rho, kraus_ops):
    """
    Esegue una misura tramite Kraus operators:
    - calcola le probabilità
    - campiona un outcome
    - aggiorna lo stato post-misura
    """
    # calcola probabilità
    probs = kraus_probabilities(rho, kraus_ops)
    # estrai outcome
    outcome = sample_outcome(probs)

    # aggiorna stato
    M = kraus_ops[outcome]
    
    if rho.isket:      
        rho_post = (M * rho).unit()
    else:
        rho_post = (M * rho * M.dag()) / probs[outcome]

    return rho_post


def mean_over_trajectories(trajs, key):
    arr = np.array([[row[key].item() for row in traj] for traj in trajs])
    return arr.mean(axis=0)   # media sulle traiettorie, step per step


def simulate_trajectory(state, steps, kraus_ops, funcs=None):
    """Simula una singola traiettoria quantistica."""

    if funcs is None:
        funcs = []

    # Contenitore risultati per ogni funzione
    results = []

    for step in range(steps):
        #infilo state in measure and update poi 
        psi_post = measure_and_update(state, kraus_ops)

        #aggiorno state in psi_post
        state = psi_post

        # Eventuali calcoli sullo stato post-misura
        row = {"step": step}
        for f in funcs:
            row[f.__name__] = f(state)

        results.append(row)  
    #print("Fine traiettoria")  

    return results


#=================================================================================================================================
# //-- SPIN SQUEEZING --// =======================================================================================================
#=================================================================================================================================

J_x = 0.5 * (tensor(sigmax(), I_2) + tensor(I_2, sigmax()))
J_y = 0.5 * (tensor(sigmay(), I_2) + tensor(I_2, sigmay()))
J_z = 0.5 * (tensor(sigmaz(), I_2) + tensor(I_2, sigmaz())) 

def eval_J_norm(state):
    return np.sqrt(expect(J_x, state)**2 + expect(J_y, state)**2 + expect(J_z, state)**2)

def compute_spin_squeezing_angles(rho):
    """Calcola θ e φ a partire dai valori medi di J"""
    
    # valori medi
    Jx_exp, Jy_exp, Jz_exp = [expect(J, rho) for J in (J_x, J_y, J_z)]

    J_norm = eval_J_norm(rho)

    if J_norm < 1e-12:
        raise ValueError("Il vettore di spin medio è nullo, θ e φ non sono ben definiti.")
    
    # theta
    theta = np.arccos(Jz_exp / J_norm)
    
    # phi
    if abs(np.sin(theta)) < 1e-12:
        phi = 0.0   # convenzione arbitraria sull'asse z
    else:
        cosphi = Jx_exp / (J_norm * np.sin(theta))
        cosphi = np.clip(cosphi, -1.0, 1.0)  # stabilità numerica
        if Jy_exp > 0:
            phi = np.arccos(cosphi)
        else:
            phi = 2*np.pi - np.arccos(cosphi)
    
    return theta, phi


#=================================================================================================================================
# //-- SPIN SQUEEZING --// =======================================================================================================
#=================================================================================================================================

J_x = 0.5 * (tensor(sigmax(), I_2) + tensor(I_2, sigmax()))
J_y = 0.5 * (tensor(sigmay(), I_2) + tensor(I_2, sigmay()))
J_z = 0.5 * (tensor(sigmaz(), I_2) + tensor(I_2, sigmaz())) 

val_sus = 0

def css_2(theta, phi):
    single_qubit = (np.sin(theta/2) * basis(2,0) + np.exp(1j*phi) * np.cos(theta/2) * basis(2,1))
    return tensor(single_qubit, single_qubit)

def round_if_close(val, target=1.0, atol=1e-12):
    """
    Se `val` è numericamente vicino a `target` entro `atol`, restituisce `target`.
    Altrimenti restituisce `val` invariato.
    """
    
    if np.isclose(val, target, atol=atol):
        return target
    return val

def eval_spin_components(state):
    return expect(J_x, state), expect(J_y, state), expect(J_z, state)

#Using the rounding to 1
def eval_variances(state):
    return round_if_close(variance(J_x, state)), round_if_close(variance(J_y, state)), round_if_close(variance(J_z, state))

def eval_J_norm(state):
    return np.sqrt(np.abs(expect(J_x, state))**2 + np.abs(expect(J_y, state))**2 + np.abs(expect(J_z, state))**2)


def compute_spin_squeezing_angles(rho):
    """Calcola θ e φ a partire dai valori medi di J"""
    
    # valori medi
    Jx_exp, Jy_exp, Jz_exp = eval_spin_components(rho)
    J_norm = eval_J_norm(rho)  
    
    #If J_norm==0, as Bell states
    if np.abs(J_norm)==0:
        return val_sus, val_sus

    else:
        theta = np.arccos(Jz_exp / J_norm)

        if np.sin(theta)==0:
            return theta, 0

        cosphi = Jx_exp / (J_norm * np.sin(theta))
        cosphi = np.clip(cosphi, -1.0, 1.0)  # stabilità numerica
        if Jy_exp > 0:
            phi = np.arccos(cosphi)
        else:
            phi = 2*np.pi - np.arccos(cosphi)
    
    return theta, phi



def Jn1_Jn_2(rho, theta, phi):
    """Calcola J_n1 e J_n2 a partire da θ e φ"""
    
    # versori
    n1 = np.array([-np.sin(phi), np.cos(phi), 0.0])
    n2 = np.array([np.cos(theta)*np.cos(phi), np.cos(theta)*np.sin(phi), -np.sin(theta)])
    
    # operatori
    Jn1 = n1[0]*J_x + n1[1]*J_y + n1[2]*J_z
    Jn2 = n2[0]*J_x + n2[1]*J_y + n2[2]*J_z
    
    return Jn1, Jn2



def spin_squeezing_KU(rho):
    """Calcola il parametro di spin squeezing secondo Kitagawa-Ueda"""
    
    # calcolo θ e φ
    theta, phi = compute_spin_squeezing_angles(rho)

    if theta==val_sus and phi==val_sus:
        return val_sus
    
    # calcolo J_n1 e J_n2
    Jn1, Jn2 = Jn1_Jn_2(rho, theta, phi)
    
    #xi_sq = (2/N)*(expect(Jn1**2+Jn2**2, rho)-np.sqrt((expect(Jn1**2-Jn2**2, rho))**2 + (expect(Jn1*Jn2+Jn2*Jn1, rho))**2))/eval_J_norm(rho) by autocompletion
    xi_sq = (expect(Jn1**2+Jn2**2, rho)-np.sqrt((expect(Jn1**2-Jn2**2, rho))**2 + (expect(Jn1*Jn2+Jn2*Jn1, rho))**2))
    return np.clip(xi_sq,0,1) # clip to avoid numerical issues

def spin_squeezing_KU_solver(t, rho):
    """Calcola il parametro di spin squeezing secondo Kitagawa-Ueda"""
    
    # calcolo θ e φ
    theta, phi = compute_spin_squeezing_angles(rho)
    
    if theta==val_sus and phi==val_sus:
        return val_sus
    
    # calcolo J_n1 e J_n2
    Jn1, Jn2 = Jn1_Jn_2(rho, theta, phi)
    
    #xi_sq = (2/N)*(expect(Jn1**2+Jn2**2, rho)-np.sqrt((expect(Jn1**2-Jn2**2, rho))**2 + (expect(Jn1*Jn2+Jn2*Jn1, rho))**2))/eval_J_norm(rho) by autocompletion
    xi_sq = (expect(Jn1**2+Jn2**2, rho)-np.sqrt((expect(Jn1**2-Jn2**2, rho))**2 + (expect(Jn1*Jn2+Jn2*Jn1, rho))**2))
    return np.clip(xi_sq,0,1) # clip to avoid numerical issues

def peres_horodecki_test(rho, tol=1e-10):
    """
    Implementazione del test di Peres–Horodecki per 2 qubit.
    Restituisce (bool, autovalori).
    - True: lo stato è entangled
    - False: lo stato è separabile
    """

    if rho.isket==True:
        rho = ket2dm(rho)
        
    rho_pt = qutip.partial_transpose(rho, mask=[0,1])  # trasposizione parziale su secondo qubit
    eigvals = rho_pt.eigenenergies()
    entangled = np.any(eigvals < -tol)
    return int(not entangled)


def peres_horodecki_test_solver(t, rho):
    """
    Implementazione del test di Peres–Horodecki per 2 qubit.
    Restituisce (bool, autovalori).
    - True: lo stato è entangled
    - False: lo stato è separabile
    """

    if rho.isket==True:
        rho = ket2dm(rho)
        
    rho_pt = qutip.partial_transpose(rho, mask=[0,1])  # trasposizione parziale su secondo qubit
    eigvals = rho_pt.eigenenergies()
    entangled = np.any(eigvals < -1e-10)
    return int(not entangled)


def spin_squeezing_W(rho, N):
    return spin_squeezing_KU(rho)*(N/(2*eval_J_norm(rho)))**2

def spin_squeezing_W_solver(t, rho):
    return spin_squeezing_KU(rho)*(2/(2*eval_J_norm(rho)))**2

def xi_KU_solver(t, rho, N=None, tol=1e-12):
    """
    Compute the Kitagawa–Ueda spin-squeezing parameter ξ^2_KU for an N-qubit state `rho`.

    Definition used:
        ξ^2_KU = (4 / N) * min_{n ⟂ <J>} Var(J_n)    if ||<J>|| > 0
        ξ^2_KU = (4 / N) * min_{n ∈ R^3, ||n||=1} Var(J_n)   if ||<J>|| = 0

    where J = (J_x, J_y, J_z) are collective spin operators for N spin-1/2 particles:
        J_k = (1/2) * sum_{i=1}^N σ_k^(i),  k ∈ {x, y, z}
    and Var(J_n) = n^T C n with the 3×3 covariance matrix
        C_ij = 1/2 <J_i J_j + J_j J_i> - <J_i><J_j>.

    Notes:
    - For coherent spin states (CSS), ξ^2_KU = 1 (shot-noise level).
    - ξ^2_KU < 1 is a sufficient condition for spin squeezing (and implies entanglement).
    - When <J> = 0 (e.g., some Bell states), the “perpendicular plane” is undefined;
      we then minimize variance over all directions, i.e. take the smallest eigenvalue of C.

    Args:
        rho: qutip.Qobj density matrix (or state) on a (2^N)-dimensional Hilbert space.
        N (int, optional): number of qubits. If None, inferred from dim(rho) = 2^N.
        tol (float): numerical threshold to decide whether ||<J>|| is effectively zero.

    Returns:
        float: ξ^2_KU in [0, +∞). For well-behaved states it should be ≥ 0 and
               equals 1 for CSS. Values slightly below 0 can occur from round-off
               and are truncated to 0 for robustness.
    """

    # 1) Infer N from Hilbert-space dimension if not provided.
    #    For N qubits, dim = 2^N.
    if N is None:
        dim = rho.shape[0]  # qutip.Qobj supports .shape
        N = int(round(np.log2(dim)))

    # 2) Mean spin vector <J> = ( <J_x>, <J_y>, <J_z> ).
    #    `eval_spin_components(rho)` is assumed to return those three expectations.
    Jx_exp, Jy_exp, Jz_exp = eval_spin_components(rho)
    Jmean = np.array([Jx_exp, Jy_exp, Jz_exp], dtype=float)
    m = np.linalg.norm(Jmean)  # ||<J>||

    # 3) Build the 3×3 covariance matrix C:
    #    C_ij = 1/2 <J_i J_j + J_j J_i> - <J_i><J_j>.
    #    The operators J_x, J_y, J_z must exist in scope (qutip Qobj operators).
    Js = [J_x, J_y, J_z]
    C = np.zeros((3, 3), dtype=float)

    # Double loop is tiny (3×3), explicit is fine and readable.
    for i, Ji in enumerate(Js):
        for j, Jj in enumerate(Js):
            # Symmetrized second moment minus product of means.
            C[i, j] = 0.5 * expect(Ji * Jj + Jj * Ji, rho) - Jmean[i] * Jmean[j]

    # 4) If ||<J>|| > tol: minimize Var(J_n) within the plane orthogonal to <J>.
    #    If ||<J>|| ≤ tol: minimize over all directions (smallest eigenvalue of C).
    if m > tol:
        # Unit vector u = <J> / ||<J>|| sets the "polar axis".
        u = Jmean / m

        # Construct an orthonormal basis {e1, e2} spanning the plane perpendicular to u.
        # Start from a helper vector 'a' not parallel to u to avoid degeneracy.
        a = np.array([1.0, 0.0, 0.0]) if abs(u[0]) < 0.9 else np.array([0.0, 1.0, 0.0])

        # Project 'a' orthogonally to u to form e1, then normalize.
        e1 = a - u * np.dot(u, a)
        e1 /= np.linalg.norm(e1)

        # e2 completes the right-handed frame: e2 = u × e1.
        e2 = np.cross(u, e1)

        # Restrict C to the perpendicular plane via the 2×2 block:
        # C2_{αβ} = e_α^T C e_β with α,β ∈ {1,2}.
        C2 = np.array([
            [e1 @ C @ e1, e1 @ C @ e2],
            [e2 @ C @ e1, e2 @ C @ e2]
        ], dtype=float)

        # For a real symmetric 2×2 matrix, the minimal variance in that plane is
        # the smallest eigenvalue. eigvalsh exploits symmetry and returns sorted vals.
        lam_min = np.linalg.eigvalsh(C2)[0]

    else:
        # <J> is essentially zero: the “perpendicular plane” is undefined (Bell states)
        # The KU definition collapses to minimizing over all unit directions,
        # which is exactly the smallest eigenvalue of the full 3×3 C.
        lam_min = np.linalg.eigvalsh(C)[0]

    # 5) Numerical hygiene: covariance is PSD, but rounding can yield tiny negatives.
    lam_min = max(lam_min, 0.0)

    # 6) Final KU normalization: ξ^2_KU = (4/N) * λ_min.
    #    For a CSS, Var(J_⊥) = N/4 ⇒ ξ^2_KU = 1.
    xi2 = (4.0 / N) * lam_min

    return float(xi2)

#=================================================================================================================================
# //-- FIDELITY --// =======================================================================================================
#=================================================================================================================================

def fidelity_phi_plus(rho):
    return fidelity(rho, phi_plus)

def fidelity_phi_plus_solver(t, rho):
    return fidelity(rho, phi_plus)

def fidelity_phi_minus_solver(t, rho):
    return fidelity(rho, phi_minus)

def fidelity_psi_plus_solver(t, rho):
    return fidelity(rho, psi_plus)

def fidelity_psi_minus_solver(t, rho):
    return fidelity(rho, psi_minus)

fidelity_ops = [
    fidelity_phi_plus_solver,
    fidelity_phi_minus_solver,
    fidelity_psi_plus_solver,
    fidelity_psi_minus_solver]


#=================================================================================================================================
# //-- UTILITIES GRAFICI --// ====================================================================================================
#=================================================================================================================================

def label_states_sim(name: str) -> str:
    r"""
    Costruisce la label per la legenda:
    - 'ee'                -> $\overline{\mathcal{C}}$ (|ee>)
    - '++'                -> $\overline{\mathcal{C}}$ (|++>)
    - 'css_theta=pi/6'    -> $\overline{\mathcal{C}}$ (Css(theta=\pi/6))

    - '0'                 -> $\overline{\mathcal{C}}$ (lambda=0)
    - 'pi/6'              -> $\overline{\mathcal{C}}$ (lambda=\pi/6)
    """
    if name == "ee":
        return r"$\overline{\mathcal{C}}\ (|ee\rangle)$"
    if name == "++":
        # ++ più piccolo e compattato
        return r"$\overline{\mathcal{C}}\ (|{\scriptstyle +\!+}\rangle)$"
    if name.startswith("css_theta="):
        theta = name.split("=", 1)[1]            # es. 'pi/6'
        theta_tex = theta.replace("pi", r"\pi")  # -> '\pi/6'
        return rf"$\overline{{\mathcal{{C}}}}\ (\mathrm{{Css}}(\theta={theta_tex}))$"
    if name == "0":
        return r"$\overline{\mathcal{C}}\ (\lambda=0)$"
    if name.startswith("pi/"):
        angle = name.replace("pi", r"\pi")  # -> '\pi/6'
        return rf"$\overline{{\mathcal{{C}}}}\ (\lambda={angle})$"
    
    return rf"$\overline{{\mathcal{{C}}}}\ (\mathrm{{{name}}})$"



def label_states_ineff(name: str) -> str:
    r"""
    Costruisce la label per la legenda:
    - 'ee'                -> $\overline{\mathcal{C}}$ (|ee>)
    - '++'                -> $\overline{\mathcal{C}}$ (|++>)
    - 'css_theta=pi/6'    -> $\overline{\mathcal{C}}$ (Css(theta=\pi/6))

    - '0'                 -> $\overline{\mathcal{C}}$ (lambda=0)
    - 'pi/6'              -> $\overline{\mathcal{C}}$ (lambda=\pi/6)
    """
    if name == "ee":
        return r"$\overline{\mathcal{C}}\ (|ee\rangle)$"
    if name == "++":
        # ++ più piccolo e compattato
        return r"$\overline{\mathcal{C}}\ (|{\scriptstyle +\!+}\rangle)$"
    if name.startswith("css_theta="):
        theta = name.split("=", 1)[1]            # es. 'pi/6'
        theta_tex = theta.replace("pi", r"\pi")  # -> '\pi/6'
        return rf"$\overline{{\mathcal{{C}}}}\ (\mathrm{{Css}}(\theta={theta_tex}))$"
    if name == "0":
        return r"$\overline{\mathcal{C}}\ (\lambda=0)$"
    if name.startswith("pi/"):
        angle = name.replace("pi", r"\pi")  # -> '\pi/6'
        return rf"$\overline{{\mathcal{{C}}}}\ (\lambda={angle})$"
    
    return rf"$\overline{{\mathcal{{C}}}}\ (\mathrm{{{name}}})$"









def add_state_columns(
    fidelity_df,
    bell_order=("phi_plus", "phi_minus", "psi_plus", "psi_minus"),
    thr_pure=0.9999,           # soglia “~1”
    css_target=0.707107,       # per |++>: usa 0.5 se la tua fidelity è |⟨⋅|⋅⟩|^2
    css_tol=5e-3               # tolleranza intorno al target CSS
):
    """
    Aggiunge per ogni Trajectory una colonna (Trajectory, 'state') subito dopo 'psi_minus'.
    Mappatura: +2 φ+, +1 φ−, 0 CSS (++), -1 ψ−, -2 ψ+.
    Richiede colonne MultiIndex con livelli ['Trajectory','Bell state'].
    """
    # controllo livelli
    assert fidelity_df.columns.names == ["Trajectory", "Bell state"] or \
           fidelity_df.columns.names == ["Trajectory", "Bell state"], "MultiIndex non conforme"

    trajs = fidelity_df.columns.get_level_values("Trajectory").unique()
    out = fidelity_df.copy()

    for tr in trajs:
        sub = out.xs(tr, axis=1, level="Trajectory")[list(bell_order)]

        F_phi_plus = sub["phi_plus"]
        F_phi_minus = sub["phi_minus"]
        F_psi_plus = sub["psi_plus"]
        F_psi_minus = sub["psi_minus"]

        # inizializza a NaN e assegna in ordine di priorità
        state = np.full(len(out), np.nan, dtype=float)

        # 1) stati “puri” vicini a 1
        state = np.where(F_phi_plus  >= thr_pure,  +2, state)
        state = np.where((np.isnan(state)) & (F_phi_minus >= thr_pure), +1, state)
        state = np.where((np.isnan(state)) & (F_psi_minus >= thr_pure),  -1, state)
        state = np.where((np.isnan(state)) & (F_psi_plus >= thr_pure),  -2, state)

        # 2) firma CSS ++: φ+ ≈ css_target e ψ+ ≈ css_target
        css_mask = (np.abs(F_phi_plus - css_target) <= css_tol) & \
                   (np.abs(F_psi_plus - css_target) <= css_tol)
        state = np.where((np.isnan(state)) & css_mask, 0, state)

        # aggiungi la colonna (traj, 'state')
        out[(tr, "state")] = state

    # riordina: per ogni Traj: φ+, φ−, ψ+, ψ−, state
    new_cols = []
    for tr in trajs:
        for b in bell_order:
            new_cols.append((tr, b))
        new_cols.append((tr, "state"))

    out = out.reindex(columns=pd.MultiIndex.from_tuples(new_cols, names=out.columns.names))

    return out
