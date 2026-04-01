"""
concurrence_phi2_sweep_N2.py
For N=2, CSS initial state, phi_1=0 fixed:
sweep phi_2 in [0, pi/2] and track max_t <C(t)> for each eta_2 value.
Produces one plot: phi_2 vs peak ensemble-average concurrence, one curve per eta_2.
"""

import argparse
import math
import os
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from qutip import (
    basis, ket2dm, qeye, sigmay, sigmaz, smesolve, tensor
)

# Use QuTiP's built-in concurrence as e_ops callable (signature: t, state)
try:
    from qutip import concurrence as _qtp_concurrence
    def conc_eop(t, state):
        return float(_qtp_concurrence(state))
except ImportError:
    _SYSY = tensor(sigmay(), sigmay())
    def conc_eop(t, state):
        if state.isket:
            c = state.full().flatten()
            return float(2.0 * np.abs(c[0] * c[3] - c[1] * c[2]))
        rho_tilde = (state * _SYSY) * (state.conj() * _SYSY)
        evals = np.sort(np.abs(np.real(rho_tilde.eigenenergies())))
        sqrt_ev = np.sqrt(evals)
        return float(max(0.0, sqrt_ev[3] - sqrt_ev[2] - sqrt_ev[1] - sqrt_ev[0]))

# -- Style ------------------------------------------------------------------
def apply_style(usetex: bool = False):
    plt.rcParams["text.usetex"] = usetex
    plt.rcParams.update({
        "mathtext.fontset":   "cm",
        "font.family":        "serif",
        "font.size":          14,
        "axes.labelsize":     13,
        "axes.titlesize":     13,
        "legend.fontsize":    11,
        "xtick.labelsize":    11,
        "ytick.labelsize":    11,
        "axes.unicode_minus": False,
        "figure.dpi":         150,
        "savefig.dpi":        150,
        "savefig.bbox":       "tight",
    })

# -- Argparse ---------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(
        description="Concurrence vs phi_2 sweep for N=2 (phi_1=0 fixed)"
    )
    p.add_argument("--theta_css",   type=float, default=math.pi / 2,
                   help="CSS polar angle (default pi/2 -> |++>)")
    p.add_argument("--phi_css",     type=float, default=0.0)
    p.add_argument("--state_label", type=str,   default="PlusPlus")
    p.add_argument("--ntraj",       type=int,   default=200)
    p.add_argument("--gamma",       type=float, default=1.0)
    p.add_argument("--T_END",       type=float, default=5.0)
    p.add_argument("--dt",          type=float, default=0.005,
                   help="Time step (max 0.005)")
    p.add_argument("--eta1",        type=float, default=1.0)
    p.add_argument("--eta2",        type=str,   default="1.0,0.8,0.5,0.0",
                   help="Comma-separated eta_2 values")
    p.add_argument("--n_phi",       type=int,   default=15,
                   help="Number of phi_2 points in [0, pi/2]")
    p.add_argument("--out_base",    type=str,   default="Graphs/QFI")
    p.add_argument("--usetex",      action="store_true")
    return p.parse_args()

# -- Operators (N=2) --------------------------------------------------------
I2  = qeye(2)
Cp  = np.sqrt(0.5) * (tensor(sigmaz(), I2) + tensor(I2, sigmaz()))
Cm  = np.sqrt(0.5) * (tensor(sigmaz(), I2) - tensor(I2, sigmaz()))
H_free = 0.0 * tensor(sigmaz(), I2)
SYSY   = tensor(sigmay(), sigmay())

def css_2(theta: float, phi: float = 0.0):
    single = (np.sin(theta / 2.0) * basis(2, 0)
              + np.exp(1j * phi) * np.cos(theta / 2.0) * basis(2, 1)).unit()
    from qutip import ket2dm
    return ket2dm(tensor(single, single))

# -- Concurrence (Wootters) -------------------------------------------------
def concurrence(state) -> float:
    if state.isket:
        c = state.full().flatten()
        return float(2.0 * np.abs(c[0] * c[3] - c[1] * c[2]))
    rho_tilde = (state * SYSY) * (state.conj() * SYSY)
    evals = np.sort(np.abs(np.real(rho_tilde.eigenenergies())))
    sqrt_ev = np.sqrt(evals)
    return float(max(0.0, sqrt_ev[3] - sqrt_ev[2] - sqrt_ev[1] - sqrt_ev[0]))

# -- Main -------------------------------------------------------------------
def main():
    args = parse_args()
    apply_style(args.usetex)

    assert args.dt <= 0.005, f"dt={args.dt} exceeds max 0.005"
    N_TIMES = int(round(args.T_END / args.dt)) + 1
    times   = np.linspace(0.0, args.T_END, N_TIMES)
    t1      = 1.0 / args.gamma

    rho0 = css_2(args.theta_css, args.phi_css)

    out_dir = Path(args.out_base) / f"ntraj{args.ntraj}_{args.state_label}"
    out_dir.mkdir(parents=True, exist_ok=True)

    eta2_values  = [float(e) for e in args.eta2.split(",")]
    phi2_values  = np.linspace(0.0, math.pi / 2, args.n_phi)
    phi1         = 0.0   # fixed

    # colors for eta_2 curves
    eta_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    if len(eta2_values) > len(eta_colors):
        eta_colors = plt.cm.viridis(np.linspace(0, 0.9, len(eta2_values)))

    print(f"N_TIMES={N_TIMES}, dt={args.dt}, ntraj={args.ntraj}")
    print(f"phi_1=0 (fixed), phi_2: {args.n_phi} points in [0, pi/2]")
    print(f"eta1={args.eta1}, eta2={eta2_values}, state={args.state_label}")

    # max_C[i_eta, i_phi] = max_t <C(t)>
    max_C     = np.zeros((len(eta2_values), args.n_phi))
    # t_peak[i_eta, i_phi] = time index of the peak
    t_peak    = np.zeros((len(eta2_values), args.n_phi), dtype=int)

    total_runs = len(eta2_values) * args.n_phi
    run_idx    = 0

    for i_eta, eta2 in enumerate(eta2_values):
        print(f"\n{'='*60}")
        print(f"  eta_2 = {eta2}")
        print(f"{'='*60}")

        c_ops_base  = [np.sqrt(args.gamma * (1 - args.eta1)) * Cp,
                       np.sqrt(args.gamma * (1 - eta2))       * Cm]

        for i_phi, phi2 in enumerate(phi2_values):
            run_idx += 1
            print(f"  [{run_idx}/{total_runs}] phi_2 = {phi2/math.pi:.4f}*pi", end="", flush=True)
            t0 = time.time()

            sc_ops = [np.sqrt(args.gamma * args.eta1) * np.exp(1j * phi1) * Cp,
                      np.sqrt(args.gamma * eta2)       * np.exp(1j * phi2) * Cm]

            # e_ops=[conc_eop]: concurrence computed on-the-fly, no state history kept.
            # sol.expect[0] is the average over trajectories, shape (N_TIMES,).
            # keep_runs_results not needed here since we only need the average.
            sol = smesolve(
                H_free, rho0, times,
                c_ops=c_ops_base, sc_ops=sc_ops,
                heterodyne=False, ntraj=args.ntraj,
                e_ops=[conc_eop],
                options={"map": "serial", "store_measurement": False},
            )

            avg_C = np.real(np.array(sol.expect[0]))
            del sol

            idx_peak       = int(np.argmax(avg_C))
            max_C[i_eta, i_phi]  = avg_C[idx_peak]
            t_peak[i_eta, i_phi] = idx_peak

            print(f"  ->  max<C> = {max_C[i_eta, i_phi]:.4f}"
                  f"  at t/T1 = {times[idx_peak]/t1:.3f}"
                  f"  ({time.time()-t0:.1f}s)")

    # -- Save raw data -------------------------------------------------------
    np.save(out_dir / f"phi2_sweep_{args.state_label}_ntraj{args.ntraj}_maxC.npy", max_C)
    np.save(out_dir / f"phi2_sweep_{args.state_label}_ntraj{args.ntraj}_tpeak.npy", t_peak)
    print("\nRaw data saved.")

    # -- Plot 1: max<C> vs phi_2 --------------------------------------------
    phi2_deg = phi2_values / math.pi  # in units of pi

    fig, ax = plt.subplots(figsize=(8, 5))
    for i_eta, (eta2, col) in enumerate(zip(eta2_values, eta_colors)):
        ax.plot(phi2_deg, max_C[i_eta],
                color=col, lw=2.2, marker="o", ms=5,
                label=rf"$\eta_2={eta2}$")

    ax.set_xlabel(r"$\phi_2\;[\times\pi]$", fontsize=13)
    ax.set_ylabel(r"$\max_t\,\langle\mathcal{C}\rangle$", fontsize=13)
    ax.set_xlim(-0.02, 0.52)
    ax.set_ylim(bottom=0.0)
    ax.set_xticks([0, 0.125, 0.25, 0.375, 0.5])
    ax.set_xticklabels([r"$0$", r"$\pi/8$", r"$\pi/4$", r"$3\pi/8$", r"$\pi/2$"])
    ax.axvline(0.5, color="gray", ls="--", lw=1.0, alpha=0.5)
    ax.legend(fontsize=11, loc="best")
    ax.grid(True, linestyle=":", alpha=0.45)
    ax.set_title(
        rf"Peak concurrence vs $\phi_2$ ($\phi_1=0$) -- {args.state_label},"
        rf" $\eta_1={args.eta1}$, ntraj$={args.ntraj}$",
        fontsize=12
    )
    plt.tight_layout()
    fname1 = out_dir / f"phi2_sweep_{args.state_label}_ntraj{args.ntraj}_maxC.pdf"
    plt.savefig(fname1, bbox_inches="tight")
    plt.close()
    print(f"Saved: {fname1}")

    # -- Plot 2: t_peak/T1 vs phi_2 -----------------------------------------
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    for i_eta, (eta2, col) in enumerate(zip(eta2_values, eta_colors)):
        ax2.plot(phi2_deg, times[t_peak[i_eta]] / t1,
                 color=col, lw=2.2, marker="o", ms=5,
                 label=rf"$\eta_2={eta2}$")

    ax2.set_xlabel(r"$\phi_2\;[\times\pi]$", fontsize=13)
    ax2.set_ylabel(r"$t_\mathrm{peak}/T_1$", fontsize=13)
    ax2.set_xlim(-0.02, 0.52)
    ax2.set_xticks([0, 0.125, 0.25, 0.375, 0.5])
    ax2.set_xticklabels([r"$0$", r"$\pi/8$", r"$\pi/4$", r"$3\pi/8$", r"$\pi/2$"])
    ax2.legend(fontsize=11, loc="best")
    ax2.grid(True, linestyle=":", alpha=0.45)
    ax2.set_title(
        rf"Time of peak concurrence vs $\phi_2$ ($\phi_1=0$) -- {args.state_label},"
        rf" $\eta_1={args.eta1}$, ntraj$={args.ntraj}$",
        fontsize=12
    )
    plt.tight_layout()
    fname2 = out_dir / f"phi2_sweep_{args.state_label}_ntraj{args.ntraj}_tpeak.pdf"
    plt.savefig(fname2, bbox_inches="tight")
    plt.close()
    print(f"Saved: {fname2}")

    print("\nAll done.")


if __name__ == "__main__":
    main()
