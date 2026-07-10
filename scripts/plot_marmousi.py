import numpy as np
import matplotlib.pyplot as plt

vp = np.fromfile("../data/marmousi2_256x256.bin", dtype=np.float32).reshape(256, 256)

plt.figure(figsize=(10, 4))
# cmap = "gray_r" changes to cmap = "plasma"
plt.imshow(vp, cmap="plasma", aspect="auto",
        extent=[0, 17.0, 3.5, 0],
        vmin=1.0, vmax=4.8)
plt.colorbar(label="Vp (km/s)")
plt.xlabel("Distance (km)")
plt.ylabel("Depth (km)")
plt.title("Marmousi2 P-wave velocity model (256×256)")
plt.tight_layout()
plt.savefig("../figures/marmousi2_vp.png", dpi=150)
print("Saved figures/marmousi2_vp.png")
