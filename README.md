# PUMPS+AI 2026 — Seismic Wavefield Reconstruction

**Can PINNs Reconstruct Seismic Wavefields?**  
GPU-Accelerated FEM Validation, Gradient Pathology, and a SIREN Surrogate

María García Penalva · UNED · PUMPS+AI Summer School 2026

## Contents

- `src/wave_solver.cu` — Assembly-free CUDA FEM solver for the 2D acoustic wave equation on Marmousi2
- `scripts/pinn_forward.py` — PINN training experiments (v1–v7), gradient pathology documentation
- `scripts/validate.py` — Spatial convergence study (order 2.27)
- `scripts/lic_render.py` — Line Integral Convolution visualization of wavefield gradients
- `scripts/energy.py` — Discrete energy conservation check

## Key results

- FEM solver: second-order convergence (fitted order 2.27), CFL-stable over 752 timesteps
- Forward PINN: fails due to PDE gradient pathology (10⁴× data gradient) — reproduces Wang et al. 2021
- SIREN surrogate (v5): ~1% relative error, trained in 13.8s on Tesla T4

## Requirements

CUDA 12+, PyTorch, NumPy, SciPy, matplotlib
