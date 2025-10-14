import numpy as np
import matplotlib.pyplot as plt
from qutip import *
import pandas as pd

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