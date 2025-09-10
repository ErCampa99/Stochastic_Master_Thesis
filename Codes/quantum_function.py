import numpy as np
import matplotlib.pyplot as plt
from qutip import *

def flatten_to_qudit(state):
    """
    Prende un ket bipartito (Qobj con dims=[[2,2],[1,1]])
    e lo trasforma in un ket di un sistema a 4 livelli
    (Qobj con dims=[[4],[1]]).
    """
    return Qobj(state.full(), dims=[[4], [1]])

# Stati base_1-Qubit
exc = qutip.basis(2, 0)  # ground
gnd = qutip.basis(2, 1)  # excited

#Stati base 2-Qubit 
ee = tensor(exc, exc)  # |ee>
eg = tensor(exc, gnd)  # |eg>
ge = tensor(gnd, exc)  # |ge>   
gg = tensor(gnd, gnd)  # |gg>            


#Stati di Bell
psi_plus = flatten_to_qudit(tensor(gnd, exc) + tensor(exc, gnd))
psi_minus = flatten_to_qudit(tensor(exc, gnd) - tensor(gnd, exc))
phi_plus = flatten_to_qudit(tensor(gnd, gnd) + tensor(exc, exc))
phi_minus = flatten_to_qudit(tensor(gnd, gnd) - tensor(exc, exc)) 

# B_(x,y)
bell_states = [[phi_plus, psi_plus], [phi_minus, psi_minus] ]

def pop_excited(state):
    return abs(state[0]**2)

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
        #print("Emissione fotone da atomo  - psi plus")
    
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


def simulate_trajectory(state, steps, kraus_ops, funcs=None):
    """Simula una singola traiettoria quantistica."""

    # Contenitore risultati per ogni funzione
    results = {f.__name__: [] for f in funcs}

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

    return results