"""
energy.py
=========
Energy conservation check for wave_solver.cu (PUMPS+AI 2026).

Computes the discrete energy E^n = 0.5*||v^n||²_M + 0.5*(u^T K u)
at each interior snapshot of the homogeneous nx=256 run, plots E vs t,
and reports the relative drift in the source-free region (t > 0.15s).

Kinetic term:
    v^n = (u^{n+1} - u^{n-1}) / (2*dt)   [time-centered]
    ||v||²_M = sum(v² * hx * hy)           [interior nodes only]

Potential term:
    u^T K u = c² * sum( (du/dx)² + (du/dy)² ) * hx * hy
    gradients by central differences on interior nodes.

Energy is computable only at interior snapshots (need n-1 and n+1),
so snapshots 2..10 (0-indexed 1..9) are used.

Usage (from scripts/):
    python3 energy.py

Output:
    ../figures/energy.png
    Printed energy table and drift to stdout.
"""

import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import os

# ── Configuration ─────────────────────────────────────────────────────────────

SNAP_DIR  = "../results/conv/nx256"
FIG_PATH  = "../figures/energy.png"

NX        = 256
NNX       = NX + 1          # 257 nodes per side
HX        = 1.0 / NX        # km
HY        = 1.0 / NX
DT        = 3.324468e-04    # s  (matches solver output)
C         = 1.5             # km/s, homogeneous

# Ricker wavelet is negligible after t ~ 3*t0 = 3/f0 = 0.15 s
T_SOURCE_OFF = 0.15         # s

COLOR_E   = '#4400aa'
COLOR_ANN = '#cc44ff'

# ── Helpers ───────────────────────────────────────────────────────────────────

def load_snapshot(path):
    """Load a single snapshot, reshape to (nny, nnx) = u[iy, ix]."""
    u = np.fromfile(path, dtype=np.float64)
    assert u.size == NNX * NNX, f"Unexpected size {u.size} in {path}"
    # node_id(ix,iy) = ix*nny + iy  =>  reshape (nnx, nny) then .T
    return u.reshape(NNX, NNX).T    # shape (nny, nnx) = u[iy, ix]


def kinetic_energy(u_prev, u_next):
    """
    Kinetic energy from time-centered velocity estimate.
    v^n = (u^{n+1} - u^{n-1}) / (2*dt)
    KE = 0.5 * sum_interior( v² * hx * hy )
    """
    v = (u_next - u_prev) / (2.0 * DT)
    # Interior nodes only: indices [1:-1, 1:-1]
    v_int = v[1:-1, 1:-1]
    return 0.5 * np.sum(v_int ** 2) * HX * HY


def potential_energy(u):
    """
    Potential energy via FEM stiffness action, evaluated with
    central-difference gradients on interior nodes.
    PE = 0.5 * c² * sum_interior( (du/dx)² + (du/dy)² ) * hx * hy
    """
    u_int = u[1:-1, 1:-1]   # shape (NX-1, NX-1) — interior only

    # Central differences (interior nodes have neighbours on both sides)
    # du/dx: axis=1 (x is column index in u[iy, ix])
    dudx = (u[1:-1, 2:] - u[1:-1, :-2]) / (2.0 * HX)
    # du/dy: axis=0
    dudy = (u[2:, 1:-1] - u[:-2, 1:-1]) / (2.0 * HY)

    return 0.5 * C**2 * np.sum(dudx**2 + dudy**2) * HX * HY


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    os.makedirs("../figures", exist_ok=True)

    # Load all snapshots in order
    files = sorted(glob(os.path.join(SNAP_DIR, "snap_*.bin")))
    if len(files) < 3:
        raise RuntimeError(f"Need at least 3 snapshots, found {len(files)} in {SNAP_DIR}")

    print(f"Found {len(files)} snapshots in {SNAP_DIR}")

    # Extract timestep numbers from filenames to compute snapshot times
    step_numbers = []
    for f in files:
        base = os.path.basename(f)          # snap_000075.bin
        step = int(base.split('_')[1].split('.')[0])
        step_numbers.append(step)

    snapshots = [load_snapshot(f) for f in files]
    print(f"Loaded {len(snapshots)} snapshots, each {NNX}x{NNX} nodes")

    # Compute energy at interior snapshots (index 1 to len-2)
    times  = []
    E_vals = []
    KE_vals = []
    PE_vals = []

    print(f"\n{'snap':>6}  {'t (s)':>8}  {'KE':>12}  {'PE':>12}  {'E_total':>12}")
    print("-" * 58)

    print(f"\nGradient check at snap 5:")
    u = snapshots[5]
    dudx = (u[1:-1, 2:] - u[1:-1, :-2]) / (2.0 * HX)
    dudy = (u[2:, 1:-1] - u[:-2, 1:-1]) / (2.0 * HY)
    print(f"  max |du/dx| = {np.abs(dudx).max():.4e}")
    print(f"  max |du/dy| = {np.abs(dudy).max():.4e}")
    print(f"  max |u|     = {np.abs(u).max():.4e}")

    for i in range(1, len(snapshots) - 1):
        t   = step_numbers[i] * DT
        KE  = kinetic_energy(snapshots[i-1], snapshots[i+1])
        PE  = potential_energy(snapshots[i])
        E   = KE + PE

        times.append(t)
        E_vals.append(E)
        KE_vals.append(KE)
        PE_vals.append(PE)

        print(f"{step_numbers[i]:>6}  {t:>8.4f}  {KE:>12.4e}  {PE:>12.4e}  {E:>12.4e}")

    times   = np.array(times)
    E_vals  = np.array(E_vals)

    # Relative drift in source-free region
    mask_free = times > T_SOURCE_OFF
    if mask_free.sum() >= 2:
        E_free = E_vals[mask_free]
        E_mean = E_free.mean()
        drift  = (E_free.max() - E_free.min()) / E_mean
        print(f"\nSource-free region (t > {T_SOURCE_OFF}s): {mask_free.sum()} snapshots")
        print(f"  Mean energy : {E_mean:.4e}")
        print(f"  Relative drift (max-min)/mean : {drift:.2e}  ({drift*100:.3f}%)")
    else:
        print("\nNot enough source-free snapshots to compute drift.")
        drift = None
        E_mean = None

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 4.5))

    ax.plot(times, E_vals, 'o-', color=COLOR_E, lw=2, ms=7, zorder=3,
            label='Total energy $E^n$')

    # Shade source-active region
    ax.axvspan(0, T_SOURCE_OFF, alpha=0.08, color=COLOR_ANN,
            label=f'Source active ($t < {T_SOURCE_OFF}$ s)')

    # Mark source-off boundary
    ax.axvline(T_SOURCE_OFF, color=COLOR_ANN, lw=1.2, ls='--', zorder=2)

    # Horizontal reference line at mean energy in source-free region
    if E_mean is not None:
        ax.axhline(E_mean, color=COLOR_ANN, lw=1.0, ls=':',
                label=f'Mean (source-free) = {E_mean:.3e}')
        # Annotate drift
        ax.text(T_SOURCE_OFF + 0.005, E_mean * 1.05,
                f'drift = {drift*100:.2f}%',
                fontsize=9, color=COLOR_ANN)

    ax.set_xlabel('Time $t$ (s)', fontsize=12)
    ax.set_ylabel('Discrete energy $E^n$', fontsize=12)
    ax.set_title(
        'Energy conservation — homogeneous medium\n'
        r'$c = 1.5\,\mathrm{km/s}$,  $n_x = 256$,  leapfrog explicit',
        fontsize=11, pad=10
    )
    ax.legend(fontsize=9, framealpha=0.9)
    ax.grid(True, alpha=0.25, linestyle=':')

    plt.tight_layout()
    plt.savefig(FIG_PATH, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {FIG_PATH}")


if __name__ == "__main__":
    main()

