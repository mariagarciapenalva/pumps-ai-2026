import numpy as np

sigma = 0.05
sx, sy = 0.5, 0.15

for nx in [64, 128, 256]:
    h = 1.0 / nx
    coords = np.linspace(0, 1, nx+1)
    X, Y = np.meshgrid(coords, coords)
    r2 = (X - sx)**2 + (Y - sy)**2
    f = np.exp(-r2 / (2*sigma**2))
    integral = f.sum() * h * h
    print(f"nx={nx}: source integral = {integral:.6f},  max_f = {f.max():.6f}")

f0 = 20.0
t0 = 1.0 / f0
t_end = 0.25

print()
for nx in [64, 128, 256]:
    h = 1.0 / nx
    dt = 0.4 * h / 4.7
    n_steps = int(np.ceil(t_end / dt))

    active = 0
    for step in range(n_steps):
        t = step * dt
        tau = t - t0
        u = np.pi * f0 * tau
        ricker = abs((1 - 2*u**2) * np.exp(-u**2))
        if ricker > 1e-6:
            active += 1
    print(f"nx={nx}: n_steps={n_steps}, active source steps={active}, dt={dt:.4e}")
