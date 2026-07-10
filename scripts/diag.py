import numpy as np
from glob import glob
import os

for nx in [64, 128, 256]:
    files = sorted(glob(f"../results/conv/nx{nx:03d}/snap_*.bin"))
    u = np.fromfile(files[-1], dtype=np.float64)
    print(f"nx={nx}: n_values={u.size}, max={np.abs(u).max():.4e}, rms={np.sqrt(np.mean(u**2)):.4e}")
