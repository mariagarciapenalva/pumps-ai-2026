import numpy as np
import matplotlib.pyplot as plt
import os

snaps = {
    "t=0.025s": "../results/snap_000075.bin",
    "t=0.075s": "../results/snap_000225.bin",
    "t=0.125s": "../results/snap_000375.bin",
    "t=0.250s": "../results/snap_000752.bin",
}

os.makedirs("../figures/snapshots", exist_ok=True)

for label, fname in snaps.items():
    u = np.fromfile(fname, dtype=np.float64).reshape(257, 257).T
    vmax = np.abs(u).max()
    print(f"{label}: min={u.min():.4e}  max={u.max():.4e}  any_nan={np.isnan(u).any()}")
    plt.figure(figsize=(6, 6))
    plt.imshow(u, origin="lower", cmap="seismic",
            vmin=-vmax, vmax=vmax, aspect="equal")
    plt.colorbar(label="u (displacement)")
    plt.title(label)
    plt.xlabel("x (node index)")
    plt.ylabel("y (node index)")
    plt.tight_layout()
    plt.savefig(f"../figures/snapshots/{label}.png", dpi=150)
    plt.close()

print("Saved to figures/snapshots/")
