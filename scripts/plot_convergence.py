"""
plot_convergence.py
===================
Produces a publication-quality convergence figure for the PUMPS+AI 2026 poster.

Loads the final snapshot from each homogeneous run (nx=64, 128, 256),
interpolates coarser solutions onto the nx=256 grid, computes L2 error
relative to the nx=256 reference, and plots the convergence curve.

Usage (from scripts/):
    python3 plot_convergence.py

Output:
    ../figures/convergence.png
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.interpolate import RegularGridInterpolator
from glob import glob
import os

# ── Configuration ─────────────────────────────────────────────────────────────

RESOLUTIONS = [64, 128, 256]
CONV_DIR    = "../results/conv"
FIG_PATH    = "../figures/convergence.png"

# Palette consistent with banner.py
COLOR_LINE   = '#4400aa'
COLOR_SLOPE  = '#cc44ff'
COLOR_POINTS = '#4400aa'

# ── Helpers (identical to validate.py) ────────────────────────────────────────

def load_last_snapshot(nx):
    nnx     = nx + 1
    pattern = os.path.join(CONV_DIR, f"nx{nx:03d}", "snap_*.bin")
    files   = sorted(glob(pattern))
    if not files:
        raise FileNotFoundError(f"No snapshots found: {pattern}")
    u = np.fromfile(files[-1], dtype=np.float64)
    return u.reshape(nnx, nnx).T   # shape (nny, nnx)

def node_coords(nx):
    return np.linspace(0.0, 1.0, nx + 1)

def l2_error(u_coarse, nx_coarse, u_ref, nx_ref):
    x_c = node_coords(nx_coarse)
    y_c = node_coords(nx_coarse)
    interp = RegularGridInterpolator(
        (y_c, x_c), u_coarse,
        method='linear', bounds_error=False, fill_value=0.0
    )
    x_f = node_coords(nx_ref)
    y_f = node_coords(nx_ref)
    YY, XX = np.meshgrid(y_f, x_f, indexing='ij')
    pts      = np.stack([YY.ravel(), XX.ravel()], axis=-1)
    u_interp = interp(pts).reshape(nx_ref + 1, nx_ref + 1)
    h        = 1.0 / nx_ref
    return np.sqrt(h * h * np.sum((u_interp - u_ref) ** 2))

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    os.makedirs("../figures", exist_ok=True)

    snapshots = {nx: load_last_snapshot(nx) for nx in RESOLUTIONS}
    nx_ref    = RESOLUTIONS[-1]
    u_ref     = snapshots[nx_ref]

    h_vals, e_vals = [], []
    for nx in RESOLUTIONS[:-1]:
        h_vals.append(1.0 / nx)
        e_vals.append(l2_error(snapshots[nx], nx, u_ref, nx_ref))

    h_vals = np.array(h_vals)
    e_vals = np.array(e_vals)

    slope, intercept = np.polyfit(np.log10(h_vals), np.log10(e_vals), 1)

# ── Figure ────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(5.5, 4.5))

    # Data points and connecting line
    ax.loglog(h_vals, e_vals,
            'o-', color=COLOR_LINE, lw=2, ms=9, zorder=3,
            label=r'FEM $L^2$ error  (ref: $n_x = 256$)')

    # Reference slope-2 line anchored at the coarse point, data range only
    h_mid   = h_vals[0]
    e_mid   = e_vals[0]
    h_range = np.array([h_vals[0] * 0.7, h_vals[-1] * 1.0])
    e_range = e_mid * (h_range / h_mid) ** 2
    ax.loglog(h_range, e_range,
            '--', color=COLOR_SLOPE, lw=1.8, zorder=2,
            label=r'Slope $= 2$ (reference)')

    # Simple text annotation, no arrow, placed near the fine point
    ax.text(h_vals[1] * 1.05, e_vals[1] * 1.4,
            f'order = {slope:.2f}',
            fontsize=10, color=COLOR_LINE)

    ax.set_xlabel(r'Grid spacing $h = 1/n_x$', fontsize=12)
    ax.set_ylabel(r'$L^2$ error  (relative to $n_x = 256$)', fontsize=12)
    ax.set_title(
        'Spatial convergence — homogeneous medium\n'
        r'$c = 1.5\,\mathrm{km/s}$,  $t = 0.250\,\mathrm{s}$',
        fontsize=11, pad=10
    )
    ax.legend(fontsize=10, framealpha=0.9)
    ax.grid(True, which='both', alpha=0.25, linestyle=':')

    plt.tight_layout()
    plt.savefig(FIG_PATH, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved: {FIG_PATH}")
    print(f"Fitted convergence order: {slope:.3f}")

if __name__ == "__main__":
    main()
