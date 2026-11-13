# PUMPS+AI 2026 — Seismic Wavefield Reconstruction

**Can PINNs Reconstruct Seismic Wavefields?**  
GPU-Accelerated FEM Validation, Gradient Pathology, and a SIREN Surrogate

María García Penalva · UNED · PUMPS+AI Summer School 2026


Poster: [poster/pumps26_seismic_poster_final.pdf](poster/pumps26_seismic_poster_final.pdf)

> **Status:** This repository reflects the work presented at PUMPS+AI 2026 (July 2026).
> 
> An extended version is in progress.

## Summary
Here is an assembly-free approach of a CUDA finite element method (FEM) solver for the 2D acoustic wave equation on the Marmousi2 velocity model, used as ground truth to ask whether a neural network is able to replace the mesh as the wavefield representation.
The main result is a reproduction of a known pathology (Wang, Teng & Perdikaris 2021) showing that a forward PINN fails because the PDE-residual gradients are about 10^4 times larger than the data gradients. A data-supervised SIREN surrogate (Sitzmann et al. 2020) reaches about 1% of relative error.  

## Contents

- `src/wave_solver.cu`: Assembly-free CUDA FEM solver for the 2D acoustic wave equation on Marmousi2
- `scripts/pinn_forward.py`: PINN training experiments (v1–v7), gradient pathology documentation
- `scripts/validate.py`: Spatial convergence study (order 2.27)
- `scripts/lic_render.py`: Line Integral Convolution visualization of wavefield gradients
- `scripts/energy.py`: Discrete energy conservation check
- `poster/`: Final poster (PDF) and logos.

## Key results

- FEM solver: second-order convergence (fitted order 2.27), CFL-stable over 752 timesteps.
- Forward PINN: fails due to PDE gradient pathology (10^4 x data gradient) — reproduces Wang, Teng & Perdikaris 2021.
- SIREN surrogate (v5): ~1% relative error, trained in 13.8s on Tesla T4 (Google Colab).
  
## Requirements

CUDA 12+, PyTorch, NumPy, SciPy, matplotlib

## Future work
- NTK-based adaptive loss weighting (Wang, Yu & Perdikaris 2022).
- Balanced residual decay rate (BRDR) weighting (Chen, Howard & Stinis 2024).
- Physics-constrained surrogate for well-posed inversion, following the SIREN full-waveform inversion setup (Sitzmann et al. 2020).
- Weak-form PDE residual and perfectly matched layer boundaries.

## References

- Sitzmann, Martel, Bergman, Lindell & Wetzstein (2020), *Implicit Neural Representations with Periodic Activation Functions*, NeurIPS
- Wang, Teng & Perdikaris (2021), *Understanding and mitigating gradient flow pathologies in physics-informed neural networks*, SIAM J. Sci. Comput. 43(5), A3055–A3081, arXiv:2001.04536v1
- Wang, Yu & Perdikaris (2022), *When and why PINNs fail to train: A neural tangent kernel perspective*, J. Comput. Phys. 449, 110768 ([code](https://github.com/PredictiveIntelligenceLab/PINNsNTK))
- Chen, Howard & Stinis (2024), *Self-adaptive weights based on balanced residual decay rate for physics-informed neural networks and deep operator networks*, arXiv:2407.01613
- Martin, Wiley & Marfurt (2006), *Marmousi2: An elastic upgrade for Marmousi*, The Leading Edge

## Citation

If this work is useful to you, you can reference the poster:

García Penalva, M. (2026). *Can Physics-Informed Neural Networks Reconstruct Seismic Wavefields?* Poster, PUMPS+AI Summer School 2026.

