"""
lic_render.py
=============
Line Integral Convolution visualization of the wavefield gradient.
Applied to FEM snapshots directly — no PINN required.

Output: figures/lic/lic_step{N}.png for each selected snapshot.

Author: María García Penalva
Project: PUMPS+AI 2026 poster
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter

SNAP_DIR  = "../results/dense/nx256"
FIG_DIR   = "../figures/lic"
N_NODES   = 257
DT        = 3.32e-4

SNAP_STEPS = [300, 450, 600, 750]

LIC_KERNEL_LEN = 25
LIC_NOISE_SEED = 42

os.makedirs(FIG_DIR, exist_ok=True)

def load_snap(step):
    path = os.path.join(SNAP_DIR, f"snap_{step:06d}.bin")
    u = np.fromfile(path, dtype=np.float64).reshape(N_NODES, N_NODES)
    return u

def gradient_field(u):
    h = 1.0 / (N_NODES - 1)
    gx = np.gradient(u, h, axis=0)
    gy = np.gradient(u, h, axis=1)
    return gx, gy

def normalize_field(gx, gy):
    mag = np.sqrt(gx**2 + gy**2) + 1e-10
    return gx / mag, gy / mag

def lic(vx, vy, length=25, seed=42):
    N = vx.shape[0]
    rng = np.random.default_rng(seed)
    noise = rng.random((N, N)).astype(np.float32)

    result = np.zeros((N, N), dtype=np.float32)
    weight = np.zeros((N, N), dtype=np.float32)

    coords = np.linspace(0, N - 1, N)
    IX, IY = np.meshgrid(coords, coords, indexing='ij')

    px = IX.copy()
    py = IY.copy()
    px_b = IX.copy()
    py_b = IY.copy()

    for step in range(length):
        ix = np.clip(px.astype(int), 0, N - 1)
        iy = np.clip(py.astype(int), 0, N - 1)
        ix_b = np.clip(px_b.astype(int), 0, N - 1)
        iy_b = np.clip(py_b.astype(int), 0, N - 1)

        result += noise[ix, iy]
        result += noise[ix_b, iy_b]
        weight += 2.0

        px = np.clip(px + vx[ix, iy], 0, N - 1)
        py = np.clip(py + vy[ix, iy], 0, N - 1)
        px_b = np.clip(px_b - vx[ix_b, iy_b], 0, N - 1)
        py_b = np.clip(py_b - vy[ix_b, iy_b], 0, N - 1)

    return result / weight

def render_snap(step):
    u = load_snap(step)
    t = step * DT

    gx, gy = gradient_field(u)
    vx, vy = normalize_field(gx, gy)
    texture = lic(vx, vy, length=LIC_KERNEL_LEN, seed=LIC_NOISE_SEED)
    texture = uniform_filter(texture, size=2)
    texture = (texture - texture.min()) / (texture.max() - texture.min() + 1e-10)

    u_norm = u / (np.abs(u).max() + 1e-10)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.patch.set_facecolor('#0d0d0d')
    for ax in axes:
        ax.set_facecolor('#0d0d0d')
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')
        ax.tick_params(colors='white')
        ax.title.set_color('white')
        ax.spines['bottom'].set_color('white')
        ax.spines['left'].set_color('white')
        ax.spines['top'].set_color('#0d0d0d')
        ax.spines['right'].set_color('#0d0d0d')

    # Panel 1: raw wavefield
    vmax = np.abs(u).max()
    im0 = axes[0].imshow(u.T, origin="lower", extent=[0,1,0,1],
                          cmap="coolwarm", vmin=-vmax, vmax=vmax, aspect="auto")
    axes[0].set_facecolor('#0d0d0d')
    axes[0].set_title(f"Wavefield u(x,y,t={t:.3f}s)", color='white')
    axes[0].set_xlabel("x"); axes[0].set_ylabel("y")
    cb0 = plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
    cb0.ax.yaxis.set_tick_params(color='white')
    plt.setp(cb0.ax.yaxis.get_ticklabels(), color='white')

    # Panel 2: LIC texture — magma on dark background
    axes[1].imshow(texture.T, origin="lower", extent=[0,1,0,1],
                   cmap="magma", aspect="auto")
    axes[1].set_title(f"LIC — wavefield gradient (t={t:.3f}s)", color='white')
    axes[1].set_xlabel("x"); axes[1].set_ylabel("y")

    # Panel 3: LIC overlaid with wavefield amplitude
    axes[2].imshow(texture.T, origin="lower", extent=[0,1,0,1],
                   cmap="magma", aspect="auto", alpha=0.75)
    axes[2].imshow(u_norm.T, origin="lower", extent=[0,1,0,1],
                   cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto", alpha=0.3)
    axes[2].set_title(f"LIC + wavefield overlay (t={t:.3f}s)", color='white')
    axes[2].set_xlabel("x"); axes[2].set_ylabel("y")

    plt.suptitle(f"Line Integral Convolution — FEM snapshot step {step}",
                 color='white', y=1.01)
    plt.tight_layout()
    out = os.path.join(FIG_DIR, f"lic_step{step:06d}.png")
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor='#0d0d0d')
    plt.close(fig)
    print(f"Saved: {out}")

for step in SNAP_STEPS:
    render_snap(step)

print("LIC rendering complete.")
