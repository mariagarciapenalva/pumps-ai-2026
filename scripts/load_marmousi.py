import numpy as np
import segyio
import os
from scipy.ndimage import gaussian_filter
from PIL import Image

segy_path = os.path.join(os.path.dirname(__file__),
    "../data/elastic-marmousi-model/model/MODEL_P-WAVE_VELOCITY_1.25m.segy")
out_path = os.path.join(os.path.dirname(__file__),
    "../data/marmousi2_256x256.bin")

print("Reading SEG-Y...")
with segyio.open(segy_path, ignore_geometry=True) as f:
    traces = segyio.collect(f.trace[:])

print(f"Raw shape: {traces.shape}")

# Actual dimensions: 2801 lateral traces x 13601 depth samples at 1.25m
NX = 2801
NY = traces.shape[1]  # 13601
vp = traces.T.astype(np.float32)  # (depth=13601, lateral=2801)

print(f"Vp raw range: {vp.min():.1f} - {vp.max():.1f} m/s")

vp /= 1000.0

img = Image.fromarray(vp).resize((256, 256), Image.BILINEAR)
vp_small = np.array(img, dtype=np.float32)

vp_small = gaussian_filter(vp_small, sigma=0.5).astype(np.float32)

vp_small.tofile(out_path)
print(f"Vp range (km/s): {vp_small.min():.3f} - {vp_small.max():.3f}")
print(f"Saved: {out_path}")
