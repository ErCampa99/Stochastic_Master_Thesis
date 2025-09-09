import numpy as np
import matplotlib.pyplot as plt
from qutip import *

# Stati base
exc = qutip.basis(2, 0)  # ground
gnd = qutip.basis(2, 1)  # excited

def theo_concurrence(t, gamma):
    return 2*np.exp(-gamma*t)*(1-np.exp(-gamma*t)) 

def concurrence_pure_state(state): 
    ###Compute the concurrence for a pure state
    return 2*np.abs(state[0]*state[3]-state[1]*state[2])

def concurrence_for_solver_pure(t, state):
    # psi is a Qobj, we need the data
    # .full() returns a 2D column vector, .flatten() makes it a 1D array
    c = state.full().flatten()
    return 2 * np.abs(c[0] * c[3] - c[1] * c[2])

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

def compute_mean_concurrence(conc_array):
    mean_conc = np.mean(np.array(conc_array), axis=0)
    return mean_conc

def flatten_to_qudit(state):
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
            # Caso stato puro |psi>
            p = (rho.dag() * M.dag() * M * rho)[0, 0].real
        else:
            # Caso matrice densità
            p = (M.dag() * M * rho).tr().real
        probs.append(p)
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