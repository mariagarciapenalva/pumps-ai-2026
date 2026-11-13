# PUMPS+AI 2026 — Seismic Wavefield Reconstruction

**Can PINNs Reconstruct Seismic Wavefields?**  
GPU-Accelerated FEM Validation, Gradient Pathology, and a SIREN Surrogate

María García Penalva · UNED · PUMPS+AI Summer School 2026
Poster: [poster/pumps26_seismic_poster_final.pdf](poster/pumps26_seismic_poster_final.pdf)


## Summary
Here is an assembly-free approach of a CUDA finite element method (FEM) solver for the 2D acoustic wave equation on the Marmousi2 velocity model, used as baseline to ask whether a neural network is able to reproduce the mesh as the wavefield representation.
The main result is a reproduction of a known pathology (Wang, Teng & Perdikaris 2021) showing that a forward PINN fails because the PDE-residual gradients are much more larger than the data gradients. A data-supervised SIREN surrogate reaches about 1% of relative error.  

## Contents

- `src/wave_solver.cu` — Assembly-free CUDA FEM solver for the 2D acoustic wave equation on Marmousi2
- `scripts/pinn_forward.py` — PINN training experiments (v1–v7), gradient pathology documentation
- `scripts/validate.py` — Spatial convergence study (order 2.27)
- `scripts/lic_render.py` — Line Integral Convolution visualization of wavefield gradients
- `scripts/energy.py` — Discrete energy conservation check

## Key results

- FEM solver: second-order convergence (fitted order 2.27), CFL-stable over 752 timesteps
- Forward PINN: fails due to PDE gradient pathology (10⁴× data gradient) — reproduces Wang et al. 2021
- SIREN surrogate (v5): ~1% relative error, trained in 13.8s on Tesla T4.
- 
## Requirements

CUDA 12+, PyTorch, NumPy, SciPy, matplotlib
