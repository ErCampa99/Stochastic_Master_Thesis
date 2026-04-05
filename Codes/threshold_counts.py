"""
threshold_counts.py  —  generalised to N atoms
================================================
Simulates N-atom homodyne SME trajectories with Gram-Schmidt channels,
computes xi^2_KU, xi^2_WIN, QFI, chi^2 per trajectory and saves plots.

Channel convention (same as main_SLURM_N_GS.py):
  C_1  = collective mode (Gram-Schmidt row 0,  all-equal weights)
  C_2 … C_N = Gram-Schmidt orthogonal modes (rows 1 … N-1)

  eta1 / phi1        → C_1
  eta_other / phi_other → C_2 … C_N  (same value for all)

Default initial state: |+>^⊗N  (CSS with theta=pi/2, phi=0)

Usage
-----
python threshold_counts.py                          # N=2 defaults
python threshold_counts.py --N 3 --ntraj 500
python threshold_counts.py --phi1 pi/2 --phi_other pi/2 --N 4
sbatch submit_threshold.sh
"""

import argparse
import math
import os
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from qutip import (
    basis, ket2dm, qeye,
    sigmax, sigmay, sigmaz,
    smesolve, tensor,
)

# ── Thread limits ─────────────────────────────────────────────────────────
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS",
           "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

# ── Style (applied after arg parsing, see apply_style()) ──────────────────
def apply_style(usetex: bool = False):
    plt.rcParams["text.usetex"] = usetex
    plt.rcParams.update({
        "mathtext.fontset":   "cm",
        "font.family":        "serif",
        "font.size":          14,
        "axes.labelsize":     13,
        "axes.titlesize":     13,
        "legend.fontsize":    10,
        "xtick.labelsize":    11,
        "ytick.labelsize":    11,
        "axes.unicode_minus": False,
        "figure.dpi":         150,
        "savefig.dpi":        150,
        "savefig.bbox":       "tight",
    })

# ──────────────────────────────────────────────────────────────────────────
# Angle parser
# ──────────────────────────────────────────────────────────────────────────
def angle(s: str) -> float:
    """Accept plain floats OR expressions like 'pi/2', 'np.pi/4', '3*pi/4'."""
    try:
        return float(s)
    except ValueError:
        expr = s.strip()
        try:
            return float(eval(expr, {"__builtins__": {}},
                               {"pi": math.pi, "np": np}))
        except Exception:
            raise argparse.ArgumentTypeError(
                f"Cannot parse '{s}' as an angle. "
                "Use a float or an expression like 'pi/2', 'np.pi/4'.")


# ──────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(
        description="Threshold-count / trajectory plots for N-atom SME")
    # --- system ---
    p.add_argument("--N",           type=int,   default=2,
                   help="Number of atoms (default 2)")
    # --- initial state ---
    p.add_argument("--theta_css",   type=angle, default=math.pi/2,
                   help="CSS polar angle (default pi/2 → |+>^N)")
    p.add_argument("--phi_css",     type=angle, default=0.0)
    p.add_argument("--state_label", type=str,   default="PlusN")
    # --- simulation ---
    p.add_argument("--ntraj",       type=int,   default=200)
    p.add_argument("--gamma",       type=float, default=1.0)
    p.add_argument("--T_END",       type=float, default=5.0)
    p.add_argument("--dt",          type=float, default=0.005,
                   help="Time step in units of T1 (must be <= 0.005)")
    # --- channel efficiencies ---
    p.add_argument("--eta1",        type=float, default=1.0,
                   help="Efficiency for C_1 (collective channel, fixed)")
    p.add_argument("--eta_other",   type=str,   default="1.0,0.8,0.5,0.0",
                   help="Comma-separated eta values for C_2…C_N")
    # --- homodyne phases ---
    p.add_argument("--phi1",        type=angle, default=0.0,
                   help="Homodyne phase for C_1, e.g. 'pi/2'")
    p.add_argument("--phi_other",   type=angle, default=0.0,
                   help="Homodyne phase for C_2…C_N (same for all)")
    # --- output ---
    p.add_argument("--out_base",    type=str,   default="Graphs/QFI")
    p.add_argument("--usetex",      action="store_true",
                   help="Enable LaTeX rendering (requires LaTeX on PATH)")
    return p.parse_args()


# ──────────────────────────────────────────────────────────────────────────
# System setup  (Gram-Schmidt channels, spin operators)
# ──────────────────────────────────────────────────────────────────────────
I2 = qeye(2)

def local_op(single_op, site: int, N: int):
    ops = [I2] * N
    ops[site] = single_op
    return tensor(ops)

def collective_spin(single_op, N: int):
    return 0.5 * sum(local_op(single_op, k, N) for k in range(N))

def gram_schmidt_unitary(N: int, tol: float = 1e-12) -> np.ndarray:
    """Unitary whose first row is the all-equal collective vector 1/√N."""
    rows = [np.ones(N, dtype=complex) / np.sqrt(N)]
    for vec in np.eye(N, dtype=complex):
        w = vec.copy()
        for u in rows:
            w -= np.vdot(u, w) * u
        norm = np.linalg.norm(w)
        if norm > tol:
            rows.append(w / norm)
        if len(rows) == N:
            break
    if len(rows) != N:
        raise RuntimeError(f"Gram-Schmidt failed for N={N}")
    return np.asarray(rows, dtype=complex)

# Global operator cache (rebuilt whenever setup_operators is called)
_G = {}   # keys: Jx, Jy, Jz, H_free, C_channels, SPIN_NP, AC_NP, N

def setup_operators(N: int, gamma: float = 1.0):
    """Build and cache all operators for N atoms."""
    Jx = collective_spin(sigmax(), N)
    Jy = collective_spin(sigmay(), N)
    Jz = collective_spin(sigmaz(), N)

    Z_loc = [local_op(sigmaz(), k, N) for k in range(N)]
    U_GS  = gram_schmidt_unitary(N)
    C_channels = []
    for i in range(N):
        Ci = sum(U_GS[i, j] * Z_loc[j] for j in range(N))
        C_channels.append(Ci)

    dim = 2**N
    SPIN_NP = [Jx.full(), Jy.full(), Jz.full()]
    AC_NP   = np.zeros((3, 3, dim, dim), dtype=complex)
    for i, Ji in enumerate([Jx, Jy, Jz]):
        for j, Jj in enumerate([Jx, Jy, Jz]):
            AC_NP[i, j] = (Ji * Jj + Jj * Ji).full()

    _G.update(dict(N=N, Jx=Jx, Jy=Jy, Jz=Jz,
                   H_free=0.0 * Jz,
                   C_channels=C_channels,
                   SPIN_NP=SPIN_NP, AC_NP=AC_NP))
    return _G


# ──────────────────────────────────────────────────────────────────────────
# Initial state
# ──────────────────────────────────────────────────────────────────────────
def css_n(theta: float, phi: float, N: int):
    """CSS |theta,phi>^⊗N as a density matrix."""
    single = (np.sin(theta / 2.0) * basis(2, 0)
              + np.exp(1j * phi) * np.cos(theta / 2.0) * basis(2, 1)).unit()
    return ket2dm(tensor([single] * N))


# ──────────────────────────────────────────────────────────────────────────
# Observable computation
# ──────────────────────────────────────────────────────────────────────────
def compute_all_obs(rho_np: np.ndarray) -> tuple:
    """Return (xi_ku, xi_win, qfi, chi2).  Uses global _G operators."""
    N       = _G["N"]
    SPIN_NP = _G["SPIN_NP"]
    AC_NP   = _G["AC_NP"]
    tol     = 1e-12

    means = np.array([np.real(np.trace(Ji @ rho_np)) for Ji in SPIN_NP])
    m     = np.linalg.norm(means)

    C = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            C[i, j] = np.real(0.5 * np.trace(AC_NP[i, j] @ rho_np)
                               - means[i] * means[j])

    if m > tol:
        u  = means / m
        a  = np.array([1., 0., 0.]) if abs(u[0]) < 0.9 else np.array([0., 1., 0.])
        e1 = a - u * np.dot(u, a); e1 /= np.linalg.norm(e1)
        e2 = np.cross(u, e1)
        C2 = np.array([[e1@C@e1, e1@C@e2], [e2@C@e1, e2@C@e2]])
        lam_min = max(np.linalg.eigvalsh(C2)[0], 0.0)
    else:
        lam_min = max(np.linalg.eigvalsh(C)[0], 0.0)

    xi_ku  = float(4.0 / N * lam_min)
    xi_win = xi_ku * ((N / 2.0) / m) ** 2 if m > tol else xi_ku

    # QFI via SLD  (exact for mixed states)
    evals, evecs = np.linalg.eigh(rho_np)
    evals = np.maximum(evals, 0.0)
    J_eig = [evecs.conj().T @ Ji @ evecs for Ji in SPIN_NP]
    Mf    = np.zeros((3, 3))
    for k in range(len(evals)):
        for l in range(len(evals)):
            denom = evals[k] + evals[l]
            if denom < tol:
                continue
            pf  = 2.0 * (evals[k] - evals[l]) ** 2 / denom
            Jkl = np.array([J_eig[i][k, l] for i in range(3)])
            Mf  += pf * np.real(np.outer(Jkl, Jkl.conj()))
    qfi  = max(float(np.linalg.eigvalsh(Mf).max()), 0.0)
    chi2 = float(N / qfi) if qfi > tol else np.inf

    return xi_ku, xi_win, qfi, chi2


# ──────────────────────────────────────────────────────────────────────────
# SME collapse operators
# ──────────────────────────────────────────────────────────────────────────
def build_c_ops(gamma, eta1, eta_other, phi1, phi_other):
    N = _G["N"]
    C = _G["C_channels"]
    etas = [eta1]    + [eta_other]  * (N - 1)
    phis = [phi1]    + [phi_other]  * (N - 1)
    c_ops, sc_ops = [], []
    for k in range(N):
        eta, phi = etas[k], phis[k]
        op = np.exp(1j * phi) * C[k]
        if eta < 1.0:
            c_ops.append(np.sqrt(gamma * (1.0 - eta)) * op)
        sc_ops.append(np.sqrt(gamma * eta) * op)
    # drop zero sc_ops (eta=0 channels produce no signal)
    sc_ops = [o for o, e in zip(sc_ops, etas) if e > 0.0]
    return c_ops, sc_ops


# ──────────────────────────────────────────────────────────────────────────
# Plots
# ──────────────────────────────────────────────────────────────────────────
COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
          '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
LW = 1.8


def _fig_info(N, state_label, eta1, ntraj):
    return rf'$N={N}$, {state_label}, $\eta_1={eta1}$, ntraj$={ntraj}$'


def plot_threshold_by_condition(fracs, eta_values, x, N,
                                 state_label, eta1, ntraj, out_dir):
    frac_xi_ku, frac_xi_win, frac_qfi, frac_chi2 = fracs
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharex=True)
    panels = [
        (axes[0, 0], frac_xi_ku,
         r'$\xi^2_{\mathrm{KU}} < 1$',
         r'Fraction with $\xi^2_{\mathrm{KU}} < 1$'),
        (axes[0, 1], frac_xi_win,
         r'$\xi^2_{\mathrm{WIN}} < 1$',
         r'Fraction with $\xi^2_{\mathrm{WIN}} < 1$'),
        (axes[1, 0], frac_qfi,
         rf'$F_Q > {N}$ (above SQL)',
         rf'Fraction with $F_Q > N={N}$'),
        (axes[1, 1], frac_chi2,
         r'$\chi^2 > 1$',
         r'Fraction with $\chi^2 = N/F_Q > 1$'),
    ]
    for ax, frac, title, ylabel in panels:
        for i_eta, (eta_2, col) in enumerate(zip(eta_values, COLORS)):
            ax.plot(x, frac[i_eta], color=col, lw=LW,
                    label=rf'$\eta_\mathrm{{other}} = {eta_2}$')
        ax.set_title(title); ax.set_ylabel(ylabel)
        ax.set_ylim(-0.02, 1.05)
        ax.axhline(1.0, color='k', ls='--', lw=0.8, alpha=0.35)
        ax.legend(loc='best'); ax.grid(True, linestyle=':', alpha=0.5)
    for ax in axes[1]:
        ax.set_xlabel(r'$t/T_1$')
    fig.suptitle(_fig_info(N, state_label, eta1, ntraj), y=1.01)
    plt.tight_layout()
    fname = out_dir / f'threshold_counts_{state_label}_ntraj{ntraj}.pdf'
    fig.savefig(fname); plt.close(fig); print(f'  Saved: {fname}')


def plot_threshold_by_eta(fracs, eta_values, x, N,
                           state_label, ntraj, out_dir):
    frac_xi_ku, frac_xi_win, frac_qfi, frac_chi2 = fracs
    n_eta  = len(eta_values)
    n_cols = 3 if n_eta > 4 else 2
    n_rows = math.ceil(n_eta / n_cols)
    cond_specs = [
        (frac_xi_ku,  r'$\xi^2_{\mathrm{KU}}<1$',  'tab:blue',   '-'),
        (frac_xi_win, r'$\xi^2_{\mathrm{WIN}}<1$', 'tab:orange',  '--'),
        (frac_qfi,    rf'$F_Q>{N}$',               'tab:green',   '-.'),
        (frac_chi2,   r'$\chi^2>1$',               'tab:red',     ':'),
    ]
    fig, axes = plt.subplots(n_rows, n_cols,
                              figsize=(6*n_cols, 4.5*n_rows),
                              sharex=True, sharey=True)
    axes_flat = axes.flatten() if n_eta > 1 else [axes]
    for i_eta, (eta_2, ax) in enumerate(zip(eta_values, axes_flat)):
        for frac, label, col, ls in cond_specs:
            ax.plot(x, frac[i_eta], color=col, ls=ls, lw=LW, label=label)
        ax.set_title(rf'$\eta_\mathrm{{other}} = {eta_2}$')
        ax.set_ylim(-0.02, 1.05)
        ax.axhline(0.5, color='gray', ls=':', lw=0.7, alpha=0.5)
        ax.axhline(1.0, color='k',    ls='--', lw=0.8, alpha=0.35)
        ax.legend(loc='best'); ax.grid(True, linestyle=':', alpha=0.5)
        if i_eta % n_cols == 0:
            ax.set_ylabel(r'Fraction of trajectories')
        if i_eta >= (n_rows - 1) * n_cols:
            ax.set_xlabel(r'$t/T_1$')
    for ax in axes_flat[n_eta:]:
        ax.set_visible(False)
    fig.suptitle(
        rf'Threshold conditions — $N={N}$, {state_label}, ntraj$={ntraj}$',
        y=1.01)
    plt.tight_layout()
    fname = out_dir / f'threshold_counts_{state_label}_ntraj{ntraj}_by_eta.pdf'
    fig.savefig(fname); plt.close(fig); print(f'  Saved: {fname}')


def plot_histograms(XI_KU, XI_WIN, QFI, CHI2,
                    eta_values, N, state_label, ntraj, out_dir):
    hist_specs = [
        (XI_KU [:, :, -1], r'$\xi^2_{\mathrm{KU}}$',  1.0, r'$\xi^2=1$',
         (0.0, max(4.0, N)), 30),
        (XI_WIN[:, :, -1], r'$\xi^2_{\mathrm{WIN}}$', 1.0, r'$\xi^2=1$',
         (0.0, max(8.0, N)), 30),
        (QFI   [:, :, -1], r'$F_Q$',  float(N), rf'SQL $F_Q=N={N}$',
         (0.0, max(5.0, N**2)), 30),
        (CHI2  [:, :, -1], r'$\chi^2 = N/F_Q$', 1.0, r'$\chi^2=1$',
         (0.0, 5.0), 30),
    ]
    for i_eta, (eta_2, col) in enumerate(zip(eta_values, COLORS)):
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        for ax, (data, xlabel, vline_val, vline_lbl, (xlo, xhi), bins) \
                in zip(axes.flat, hist_specs):
            d = np.clip(data[i_eta], xlo, xhi)
            ax.hist(d, bins=bins, range=(xlo, xhi),
                    color=col, alpha=0.7, density=True,
                    edgecolor='white', linewidth=0.4)
            ax.axvline(vline_val, color='k', ls='--', lw=1.5, label=vline_lbl)
            ax.set_xlabel(xlabel + r'  at $t=T_{\mathrm{end}}$')
            ax.set_ylabel(r'Probability density')
            ax.set_xlim(xlo, xhi)
            ax.legend(loc='upper right')
            ax.grid(True, linestyle=':', alpha=0.5)
        fig.suptitle(
            rf'Final-state distributions — $N={N}$, {state_label},'
            rf' $\eta_\mathrm{{other}}={eta_2}$, ntraj$={ntraj}$',
            y=1.01)
        plt.tight_layout()
        fname = out_dir / \
            f'histograms_final_{state_label}_ntraj{ntraj}_eta_other_{eta_2}.pdf'
        fig.savefig(fname); plt.close(fig); print(f'  Saved: {fname}')


def plot_trajectories(XI_KU, XI_WIN, QFI, CHI2,
                      eta_values, x, N, ntraj, state_label, out_dir):
    CLIP = {
        'xi_ku' : (0.0, max(4.0, N)),
        'xi_win': (0.0, max(8.0, N)),
        'qfi'   : (0.0, float(N**2) + 1.0),
        'chi2'  : (0.0, 5.0),
    }
    obs_specs = [
        (XI_KU,  r'$\xi^2_{\mathrm{KU}}$',  'xi_ku',
         [(1.0, 'gray', '--', r'$\xi^2=1$')]),
        (XI_WIN, r'$\xi^2_{\mathrm{WIN}}$', 'xi_win',
         [(1.0, 'gray', '--', r'$\xi^2=1$')]),
        (QFI,    r'$F_Q$',                  'qfi',
         [(float(N),    'gray',      '--', rf'SQL $F_Q=N={N}$'),
          (float(N**2), 'steelblue', ':',  rf'HL $F_Q=N^2={N**2}$')]),
        (CHI2,   r'$\chi^2 = N/F_Q$',       'chi2',
         [(1.0, 'gray', '--', r'$\chi^2=1$')]),
    ]
    for i_eta, (eta_2, traj_col) in enumerate(zip(eta_values, COLORS)):
        fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
        for ax, (data, ylabel, key, hlines) in zip(axes.flat, obs_specs):
            ylo, yhi = CLIP[key]
            d = np.clip(data[i_eta], ylo, yhi)
            for i_traj in range(ntraj):
                ax.plot(x, d[i_traj], color=traj_col, alpha=0.18, lw=0.6)
            ax.plot(x, d.mean(axis=0), color='black', lw=2.2,
                    zorder=5, label=r'ensemble average')
            for yval, col, ls, lbl in hlines:
                ax.axhline(yval, color=col, ls=ls, lw=1.5, label=lbl)
            ax.set_ylabel(ylabel)
            ax.set_ylim(ylo, yhi)
            ax.legend(fontsize=8, loc='upper right')
            ax.grid(True, linestyle=':', alpha=0.5)
        for ax in axes[1]:
            ax.set_xlabel(r'$t/T_1$')
        fig.suptitle(
            rf'Trajectories — $N={N}$, {state_label},'
            rf' $\eta_1={1.0}$, $\eta_\mathrm{{other}}={eta_2}$, ntraj$={ntraj}$',
            y=1.01)
        plt.tight_layout()
        fname = out_dir / \
            f'trajectories_{state_label}_ntraj{ntraj}_eta_other_{eta_2}.pdf'
        fig.savefig(fname); plt.close(fig); print(f'  Saved: {fname}')


# ──────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    N    = args.N

    apply_style(args.usetex)

    # ── build operators ───────────────────────────────────────────────────
    setup_operators(N, args.gamma)

    eta_values = [float(e) for e in args.eta_other.split(',')]
    assert args.dt <= 0.005, f"dt={args.dt} exceeds maximum allowed value of 0.005"
    N_TIMES = int(round(args.T_END / args.dt)) + 1
    times  = np.linspace(0.0, args.T_END, N_TIMES)
    t1     = 1.0 / args.gamma
    x      = times / t1

    out_dir = Path(args.out_base) / f"N{N}_ntraj{args.ntraj}_{args.state_label}"
    out_dir.mkdir(parents=True, exist_ok=True)

    rho0 = css_n(args.theta_css, args.phi_css, N)

    print(f"=== threshold_counts.py ===")
    print(f"  N           : {N}")
    print(f"  state_label : {args.state_label}")
    print(f"  theta_css   : {args.theta_css/math.pi:.4f} * pi")
    print(f"  ntraj       : {args.ntraj}")
    print(f"  eta1        : {args.eta1}   phi1 : {args.phi1/math.pi:.4f}*pi")
    print(f"  eta_other   : {eta_values}  phi_other : {args.phi_other/math.pi:.4f}*pi")
    print(f"  T_END={args.T_END}  N_TIMES={N_TIMES}  dt={times[1]-times[0]:.5f}")
    print(f"  out_dir     : {out_dir}")

    # ── simulation ────────────────────────────────────────────────────────
    n_eta  = len(eta_values)
    XI_KU  = np.zeros((n_eta, args.ntraj, N_TIMES))
    XI_WIN = np.zeros((n_eta, args.ntraj, N_TIMES))
    QFI    = np.zeros((n_eta, args.ntraj, N_TIMES))
    CHI2   = np.zeros((n_eta, args.ntraj, N_TIMES))

    t0 = time.time()
    for i_eta, eta_other in enumerate(eta_values):
        print(f"\n--- eta1={args.eta1}, eta_other={eta_other} ---")
        c_ops, sc_ops = build_c_ops(
            args.gamma, args.eta1, eta_other, args.phi1, args.phi_other)
        sol = smesolve(
            _G["H_free"], rho0, times,
            c_ops=c_ops, sc_ops=sc_ops,
            heterodyne=False, ntraj=args.ntraj,
            options={"keep_runs_results": True,
                     "map": "serial",
                     "store_measurement": False,
                     "method": "milstein"},
        )
        print("  SME done. Computing observables...")
        for i_traj in range(args.ntraj):
            if i_traj % 50 == 0:
                print(f"    traj {i_traj}/{args.ntraj}")
            for i_t, state in enumerate(sol.runs_states[i_traj]):
                rho_np = np.asarray(state.full(), dtype=complex)
                xk, xw, qf, c2 = compute_all_obs(rho_np)
                XI_KU [i_eta, i_traj, i_t] = xk
                XI_WIN[i_eta, i_traj, i_t] = xw
                QFI   [i_eta, i_traj, i_t] = qf
                CHI2  [i_eta, i_traj, i_t] = c2
        del sol
        print(f"  eta_other={eta_other} done.  ({time.time()-t0:.1f} s total)")

    print(f"\nSimulation complete in {time.time()-t0:.1f} s.")

    # ── threshold fractions ───────────────────────────────────────────────
    fracs = (
        (XI_KU  < 1.0).sum(axis=1) / args.ntraj,
        (XI_WIN < 1.0).sum(axis=1) / args.ntraj,
        (QFI    > float(N)).sum(axis=1) / args.ntraj,   # SQL = N
        (CHI2   > 1.0).sum(axis=1) / args.ntraj,
    )

    # ── plots ─────────────────────────────────────────────────────────────
    print("\nSaving plots...")
    plot_threshold_by_condition(fracs, eta_values, x, N,
                                 args.state_label, args.eta1, args.ntraj, out_dir)
    plot_threshold_by_eta(fracs, eta_values, x, N,
                           args.state_label, args.ntraj, out_dir)
    plot_histograms(XI_KU, XI_WIN, QFI, CHI2,
                    eta_values, N, args.state_label, args.ntraj, out_dir)
    plot_trajectories(XI_KU, XI_WIN, QFI, CHI2,
                      eta_values, x, N, args.ntraj, args.state_label, out_dir)
    print(f"\nAll plots saved to {out_dir}")


if __name__ == "__main__":
    main()
