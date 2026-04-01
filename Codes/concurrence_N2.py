"""
concurrence_N2.py
=================
Computes the stochastic Wootters concurrence for a two-qubit system (N=2)
evolving under homodyne-monitored Lindblad dynamics.

Physical setup
--------------
- Initial state : CSS (Coherent Spin State) |θ,φ⟩^⊗2
- Lindblad channels:
    C_+ = √(γ/2) (σz⊗I + I⊗σz)   -- symmetric collective dephasing
    C_- = √(γ/2) (σz⊗I - I⊗σz)   -- antisymmetric collective dephasing
- Each channel is split into a measured part (efficiency η) and an
  unmeasured part (efficiency 1-η), giving:
    Lindblad (unmeasured): √(γ(1-η_k)) C_k
    stochastic  (measured): √(γ η_k) e^{iφ_k} C_k

Sweep parameters
----------------
- phi_1, phi_2 : homodyne LO phases for C_+ and C_- channels.
  All 4 combinations from {0, π/2}² are simulated.
- eta_2 : detection efficiency of the C_- channel.
  eta_1 is kept fixed at 1 (C_+ fully measured).
  Looped over the list given by --eta2.

Implementation note
-------------------
The concurrence is passed as an e_ops callable to smesolve so that it is
evaluated on-the-fly during integration.  This avoids storing the full
density-matrix history (runs_states) and eliminates the post-hoc double
loop, significantly reducing memory usage and wall time.

Output (per eta_2 value)
------------------------
- *_spaghetti.pdf : 2×2 grid of trajectory plots, one panel per (φ1,φ2) pair
- *_avg.pdf       : ensemble-average ⟨C(t)⟩ for all 4 (φ1,φ2) pairs overlaid
"""

import argparse
import math
import os
import time
from pathlib import Path

# Disable multithreading in BLAS/OpenMP to avoid oversubscription when
# parallelism is handled at the trajectory level (or when running on a
# single-core SLURM allocation).
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import matplotlib
matplotlib.use("Agg")   # non-interactive backend, required on headless clusters
import matplotlib.pyplot as plt
import numpy as np
from qutip import (
    basis, ket2dm, qeye, sigmay, sigmaz, smesolve, tensor
)

# Try to use QuTiP's built-in concurrence (available in qutip >= 4.x).
# It handles both kets and density matrices, and uses the same Wootters
# formula internally.  A manual fallback is provided just in case.
try:
    from qutip import concurrence as _qtp_concurrence
    def conc_eop(t, state):
        """
        e_ops callable passed to smesolve.
        QuTiP evaluates this at every time step for every trajectory,
        storing the results in sol.runs_expect without keeping the full state.
        Signature (t, state) is mandatory for callable e_ops in QuTiP 5.
        """
        return float(_qtp_concurrence(state))
except ImportError:
    # Fallback: manual Wootters formula
    _SYSY = tensor(sigmay(), sigmay())
    def conc_eop(t, state):
        """Fallback Wootters concurrence with (t, state) signature."""
        if state.isket:
            c = state.full().flatten()
            return float(2.0 * np.abs(c[0] * c[3] - c[1] * c[2]))
        rho_tilde = (state * _SYSY) * (state.conj() * _SYSY)
        evals = np.sort(np.abs(np.real(rho_tilde.eigenenergies())))
        sqrt_ev = np.sqrt(evals)
        return float(max(0.0, sqrt_ev[3] - sqrt_ev[2] - sqrt_ev[1] - sqrt_ev[0]))

# ── Style ───────────────────────────────────────────────────────────────────
def apply_style(usetex: bool = False):
    """
    Set global matplotlib style.
    usetex=True requires a working LaTeX installation (not always available
    on clusters); the default uses mathtext with Computer Modern fonts instead,
    which is visually identical and cluster-safe.
    """
    plt.rcParams["text.usetex"] = usetex
    plt.rcParams.update({
        "mathtext.fontset":   "cm",      # Computer Modern, matches LaTeX look
        "font.family":        "serif",
        "font.size":          14,
        "axes.labelsize":     13,
        "axes.titlesize":     13,
        "legend.fontsize":    10,
        "xtick.labelsize":    11,
        "ytick.labelsize":    11,
        "axes.unicode_minus": False,     # avoid encoding issues on Windows
        "figure.dpi":         150,
        "savefig.dpi":        150,
        "savefig.bbox":       "tight",
    })

# ── Argparse ─────────────────────────────────────────────────────────────────
def parse_angle(s: str) -> float:
    """
    Parse an angle string that may be a plain float or a Python expression
    such as 'pi/4' or 'np.pi/3'. Evaluation is sandboxed (no builtins).
    """
    try:
        return float(s)
    except ValueError:
        try:
            return float(eval(s, {"__builtins__": {}}, {"pi": math.pi, "np": np}))
        except Exception:
            raise argparse.ArgumentTypeError(f"Cannot parse angle: '{s}'")

def parse_args():
    p = argparse.ArgumentParser(description="Stochastic concurrence for N=2")
    p.add_argument("--theta_css",   type=float, default=math.pi / 2,
                   help="CSS polar angle θ (default π/2 → |+⟩⊗|+⟩)")
    p.add_argument("--phi_css",     type=float, default=0.0,
                   help="CSS azimuthal angle φ (default 0)")
    p.add_argument("--state_label", type=str,   default="PlusPlus",
                   help="Short label used in folder/file names")
    p.add_argument("--ntraj",       type=int,   default=1000,
                   help="Number of stochastic trajectories per run")
    p.add_argument("--gamma",       type=float, default=1.0,
                   help="Decay rate γ (sets T1 = 1/γ as time unit)")
    p.add_argument("--T_END",       type=float, default=5.0,
                   help="Total simulation time in units of T1")
    p.add_argument("--dt",          type=float, default=0.005,
                   help="Time step (must be ≤ 0.005 for numerical stability)")
    p.add_argument("--eta1",        type=float, default=1.0,
                   help="Detection efficiency of C_+ channel (fixed at 1)")
    p.add_argument("--eta2",        type=str,   default="1.0,0.8,0.5,0.0",
                   help="Comma-separated list of η_2 values (C_- efficiency)")
    p.add_argument("--out_base",    type=str,   default="Graphs/QFI",
                   help="Base output directory (subfolder created automatically)")
    p.add_argument("--usetex",      action="store_true",
                   help="Use system LaTeX for plot rendering (requires texlive)")
    return p.parse_args()

# ── Operators (N=2) ──────────────────────────────────────────────────────────
# Single-qubit identity
I2 = qeye(2)

# Collective jump operators (Hilbert space: qubit_1 ⊗ qubit_2)
# C_+ = (1/√2)(σz⊗I + I⊗σz)  -- symmetric combination, generates entanglement
# C_- = (1/√2)(σz⊗I - I⊗σz)  -- antisymmetric combination
Cp = np.sqrt(0.5) * (tensor(sigmaz(), I2) + tensor(I2, sigmaz()))
Cm = np.sqrt(0.5) * (tensor(sigmaz(), I2) - tensor(I2, sigmaz()))

# Free Hamiltonian: set to zero (pure monitoring, no coherent evolution)
H_free = 0.0 * tensor(sigmaz(), I2)

def css_2(theta: float, phi: float = 0.0):
    """
    Build the density matrix ρ = |ψ⟩⟨ψ| for the two-qubit CSS
        |ψ⟩ = |θ,φ⟩ ⊗ |θ,φ⟩
    where the single-spin CSS is
        |θ,φ⟩ = sin(θ/2)|g⟩ + e^{iφ} cos(θ/2)|e⟩.
    Here |g⟩ = basis(2,0) is the ground state and |e⟩ = basis(2,1)
    the excited state in QuTiP's convention.
    """
    single = (np.sin(theta / 2.0) * basis(2, 0)
              + np.exp(1j * phi) * np.cos(theta / 2.0) * basis(2, 1)).unit()
    return ket2dm(tensor(single, single))

# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    apply_style(args.usetex)

    # Time grid: dt is the primary resolution parameter; assert safety bound
    assert args.dt <= 0.005, f"dt={args.dt} exceeds max 0.005"
    N_TIMES = int(round(args.T_END / args.dt)) + 1
    times   = np.linspace(0.0, args.T_END, N_TIMES)
    t1      = 1.0 / args.gamma          # T1 = 1/γ, natural time unit
    x       = times / t1                # normalised x-axis for plots

    # Initial CSS density matrix
    rho0 = css_2(args.theta_css, args.phi_css)

    # Output directory: one subfolder per (ntraj, state) combination
    out_dir = Path(args.out_base) / f"ntraj{args.ntraj}_{args.state_label}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Parse comma-separated η_2 values
    eta2_values = [float(e) for e in args.eta2.split(",")]

    # All 4 homodyne LO phase pairs (φ1, φ2) ∈ {0, π/2}²
    # These correspond to homodyne detection along the real or imaginary quadrature
    phi_vals   = [0.0, math.pi / 2]
    phi_pairs  = [(p1, p2) for p1 in phi_vals for p2 in phi_vals]   # 4 combinations
    phi_labels = [
        r"$\phi_1=0,\;\phi_2=0$",
        r"$\phi_1=0,\;\phi_2=\pi/2$",
        r"$\phi_1=\pi/2,\;\phi_2=0$",
        r"$\phi_1=\pi/2,\;\phi_2=\pi/2$",
    ]
    colors_phi = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    print(f"N_TIMES={N_TIMES}, dt={args.dt}, ntraj={args.ntraj}")
    print(f"eta1={args.eta1}, eta2 values={eta2_values}, state={args.state_label}")

    # ── Outer loop: η_2 values ───────────────────────────────────────────────
    for eta2 in eta2_values:
        print(f"\n{'='*60}")
        print(f"  eta_2 = {eta2}")
        print(f"{'='*60}")

        # CONC[i_pair, i_traj, i_t] = C(t) for a given (φ1,φ2) pair and trajectory.
        # Populated from sol.runs_expect[0], which stores the conc_eop output
        # for every trajectory without ever keeping the full state in memory.
        CONC = np.zeros((4, args.ntraj, N_TIMES))

        # ── Inner loop: (φ1, φ2) pairs ──────────────────────────────────────
        for i_pair, (phi1, phi2) in enumerate(phi_pairs):
            print(f"\n  --- phi1={phi1/math.pi:.3f}*pi, phi2={phi2/math.pi:.3f}*pi ---")
            t0 = time.time()

            # Lindblad operators for unmeasured (unmonitored) fraction of each channel:
            #   L_k = √(γ (1-η_k)) C_k   → information lost to environment
            c_ops  = [np.sqrt(args.gamma * (1 - args.eta1)) * Cp,
                      np.sqrt(args.gamma * (1 - eta2))       * Cm]

            # Stochastic (measured) operators for homodyne detection:
            #   M_k = √(γ η_k) e^{iφ_k} C_k   → back-action on state + photocurrent
            # The LO phase φ_k rotates the measured quadrature.
            sc_ops = [np.sqrt(args.gamma * args.eta1) * np.exp(1j * phi1) * Cp,
                      np.sqrt(args.gamma * eta2)       * np.exp(1j * phi2) * Cm]

            # Integrate the stochastic master equation (Itô, homodyne):
            #   dρ = -i[H,ρ]dt + D[c_ops]ρ dt + H[sc_ops]ρ dW
            #
            # e_ops=[conc_eop]: QuTiP calls conc_eop(t, ρ_t) at every time step
            # during integration, storing only the scalar result.  This avoids
            # keeping the full ρ(t) history (runs_states) in memory.
            # keep_runs_results=True is needed to access sol.runs_expect
            # (per-trajectory values), not just sol.expect (the average).
            sol = smesolve(
                H_free, rho0, times,
                c_ops=c_ops, sc_ops=sc_ops,
                heterodyne=False,
                ntraj=args.ntraj,
                e_ops=[conc_eop],
                options={"keep_runs_results": True, "map": "serial",
                         "store_measurement": False},
            )
            print(f"    SME done in {time.time()-t0:.1f}s.")

            # sol.runs_expect[i_eop][i_traj] -> 1D array of shape (N_TIMES,)
            # np.array(...) stacks all trajectories → shape (ntraj, N_TIMES)
            CONC[i_pair] = np.real(np.array(sol.runs_expect[0]))
            del sol   # free memory before next run
            print(f"    pair {i_pair+1}/4 done.")

        # File-name tag: "eta2_1p0", "eta2_0p8", etc. (dots → 'p' for safe filenames)
        eta2_tag = f"eta2_{eta2:.1f}".replace(".", "p")

        # ── Figure 1: spaghetti plot (2×2 grid, one panel per (φ1,φ2) pair) ─
        # Each thin line = one trajectory; thick black = ensemble average ⟨C(t)⟩
        fig, axes = plt.subplots(2, 2, figsize=(12, 9), sharex=True, sharey=True)

        for ax, (i_pair, lbl, col) in zip(axes.flat,
                [(i, phi_labels[i], colors_phi[i]) for i in range(4)]):
            for i_traj in range(args.ntraj):
                ax.plot(x, CONC[i_pair, i_traj], color=col, alpha=0.12, lw=0.5)
            ax.plot(x, CONC[i_pair].mean(axis=0),
                    color="black", lw=2.2, zorder=5, label="ensemble average")
            ax.set_title(lbl, fontsize=12)
            ax.set_ylim(-0.02, 1.05)
            ax.axhline(1.0, color="gray", ls="--", lw=1.0, alpha=0.5)  # C=1 reference
            ax.legend(fontsize=9, loc="upper right")
            ax.grid(True, linestyle=":", alpha=0.45)

        for ax in axes[1]:
            ax.set_xlabel(r"$t/T_1$", fontsize=12)
        for ax in axes[:, 0]:
            ax.set_ylabel(r"Concurrence $\mathcal{C}$", fontsize=12)

        fig.suptitle(
            rf"{args.state_label}, $\eta_1={args.eta1}$, $\eta_2={eta2}$,"
            rf" ntraj$={args.ntraj}$",
            fontsize=13, y=1.01
        )
        plt.tight_layout()
        fname1 = out_dir / f"concurrence_{args.state_label}_ntraj{args.ntraj}_{eta2_tag}_spaghetti.pdf"
        plt.savefig(fname1, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {fname1}")

        # ── Figure 2: ensemble-average concurrence, all 4 (φ1,φ2) overlaid ──
        # Useful to directly compare which homodyne phase generates more entanglement
        fig2, ax2 = plt.subplots(figsize=(9, 5))
        for i_pair, (lbl, col) in enumerate(zip(phi_labels, colors_phi)):
            ax2.plot(x, CONC[i_pair].mean(axis=0), color=col, lw=2.0, label=lbl)
        ax2.set_xlabel(r"$t/T_1$", fontsize=12)
        ax2.set_ylabel(r"$\langle\mathcal{C}\rangle$", fontsize=12)
        ax2.set_ylim(-0.02, 1.05)
        ax2.axhline(1.0, color="gray", ls="--", lw=1.0, alpha=0.5)
        ax2.legend(fontsize=10, loc="upper right")
        ax2.grid(True, linestyle=":", alpha=0.45)
        ax2.set_title(
            rf"Ensemble-average concurrence -- {args.state_label},"
            rf" $\eta_1={args.eta1}$, $\eta_2={eta2}$, ntraj$={args.ntraj}$",
            fontsize=12
        )
        plt.tight_layout()
        fname2 = out_dir / f"concurrence_{args.state_label}_ntraj{args.ntraj}_{eta2_tag}_avg.pdf"
        plt.savefig(fname2, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {fname2}")

    print("\nAll eta_2 values done.")


if __name__ == "__main__":
    main()
