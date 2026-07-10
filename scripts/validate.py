"""
validate.py
===========
Convergence study for wave_solver.cu (PUMPS+AI 2026).

Loads the final snapshot from each homogeneous run (nx=64, 128, 256),
interpolates the coarser solutions onto the nx=256 grid, computes the
L2 error relative to the nx=256 reference solution, and fits a log-log
slope to confirm second-order convergence in h.

Usage (from scripts/):
    python validate.py

Output:
    ../figures/convergence.png
    Printed convergence table to stdout.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator
from glob import glob
import os

# ── Configuration ─────────────────────────────────────────────────────────────

RESOLUTIONS = [64, 128, 256]
CONV_DIR    = "../results/conv"
FIG_PATH    = "../figures/convergence.png"

# ── Helpers ───────────────────────────────────────────────────────────────────

def load_last_snapshot(nx):
    """Load the final snapshot for a given resolution.
    Uses glob + sort so the filename doesn't need to be hardcoded.
    Nodes per side: nx+1. Layout: flat double, row-major, x slow index.
    Reshape as (nnx, nny) then transpose for (x, y) indexing.
    """
    nnx = nx + 1
    pattern = os.path.join(CONV_DIR, f"nx{nx:03d}", "snap_*.bin")
    files = sorted(glob(pattern))
    if not files:
        raise FileNotFoundError(f"No snapshots found matching: {pattern}")
    last = files[-1]
    print(f"  nx={nx:3d}: loading {os.path.basename(last)}")
    u = np.fromfile(last, dtype=np.float64)
    expected = nnx * nnx
    if u.size != expected:
        raise ValueError(f"nx={nx}: expected {expected} values, got {u.size}")
    # Reshape: solver stores as (nnx, nny) row-major, x is slow index
    # Transpose gives u[ix, iy] with ix=x-index, iy=y-index
    return u.reshape(nnx, nnx).T   # shape: (nny, nnx) = (nnx, nnx) for square grid


def node_coords(nx):
    """Return 1D node coordinate arrays for a square grid on [0,1]."""
    return np.linspace(0.0, 1.0, nx + 1)


def l2_error(u_coarse, nx_coarse, u_ref, nx_ref):
    """
    Interpolate u_coarse (on nx_coarse grid) onto the nx_ref grid,
    then compute the discrete L2 error norm relative to u_ref.

    L2 error = sqrt( hx*hy * sum( (u_interp - u_ref)^2 ) )
    where hx=hy=1/nx_ref on the fine grid.
    """
    x_coarse = node_coords(nx_coarse)
    y_coarse = node_coords(nx_coarse)

    # RegularGridInterpolator expects (row, col) = (y, x) if u[iy, ix]
    # Our u is shaped (nny, nnx) = u[iy, ix], so axes are (y_coarse, x_coarse)
    interp = RegularGridInterpolator(
        (y_coarse, x_coarse), u_coarse,
        method='linear', bounds_error=False, fill_value=0.0
    )

    x_fine = node_coords(nx_ref)
    y_fine = node_coords(nx_ref)
    YY, XX = np.meshgrid(y_fine, x_fine, indexing='ij')   # shape (nny_ref, nnx_ref)
    pts = np.stack([YY.ravel(), XX.ravel()], axis=-1)

    u_interp = interp(pts).reshape(nx_ref + 1, nx_ref + 1)

    h = 1.0 / nx_ref
    diff = u_interp - u_ref
    return np.sqrt(h * h * np.sum(diff**2))


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    os.makedirs("../figures", exist_ok=True)

    print("Loading snapshots...")
    snapshots = {}
    for nx in RESOLUTIONS:
        snapshots[nx] = load_last_snapshot(nx)

    nx_ref = RESOLUTIONS[-1]   # 256
    u_ref  = snapshots[nx_ref]

    print(f"\nReference: nx={nx_ref}, t=0.250s")
    print(f"{'nx':>6}  {'h':>10}  {'L2 error':>12}  {'ratio':>8}  {'order':>7}")
    print("-" * 52)

    h_vals  = []
    e_vals  = []
    prev_e  = None

    for nx in RESOLUTIONS[:-1]:   # 64, 128 — compare against 256
        h = 1.0 / nx
        e = l2_error(snapshots[nx], nx, u_ref, nx_ref)
        h_vals.append(h)
        e_vals.append(e)

        ratio = prev_e / e if prev_e is not None else float('nan')
        order = np.log2(ratio) if not np.isnan(ratio) else float('nan')
        print(f"{nx:>6}  {h:>10.6f}  {e:>12.4e}  {ratio:>8.3f}  {order:>7.3f}")
        prev_e = e

    # Fit log-log slope across the two coarse points
    log_h = np.log10(h_vals)
    log_e = np.log10(e_vals)
    slope, intercept = np.polyfit(log_h, log_e, 1)
    print(f"\nFitted convergence order: {slope:.3f}  (expected ~2.0 for bilinear Q1 FEM)")

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(6, 5))

    ax.loglog(h_vals, e_vals, 'o-', color='#4400aa', lw=2, ms=8,
              label='FEM L2 error (vs nx=256 reference)')

    # Reference slope-2 line anchored at the coarsest point
    h_line = np.array([h_vals[0] * 1.2, h_vals[-1] * 0.8])
    e_line = e_vals[0] * (h_line / h_vals[0]) ** 2
    ax.loglog(h_line, e_line, '--', color='#cc44ff', lw=1.5,
              label='Slope = 2 (reference)')

    ax.set_xlabel("Grid spacing  h = 1/nx", fontsize=12)
    ax.set_ylabel("L2 error  (relative to nx=256)", fontsize=12)
    ax.set_title(
        f"Spatial convergence — homogeneous medium, c=1.5 km/s, t=0.250 s\n"
        f"Fitted order: {slope:.2f}",
        fontsize=11
    )
    ax.legend(fontsize=10)
    ax.grid(True, which='both', alpha=0.3)

    plt.tight_layout()
    plt.savefig(FIG_PATH, dpi=150)
    plt.close()
    print(f"\nSaved: {FIG_PATH}")


if __name__ == "__main__":
    main()
