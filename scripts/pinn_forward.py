"""
pinn_forward.py
===============
Physics-Informed Neural Network (PINN) for the 2D acoustic wave equation
in a fixed heterogeneous medium (Marmousi2 benchmark).

PDE:  d²u/dt² = c(x,y)² * Δu   in [0,1]² × [0, 0.25]
BCs:  u = 0 on boundary (Dirichlet)
ICs:  u(x,y,0) = 0,  du/dt(x,y,0) = 0

Network: SIREN (Sitzmann et al., NeurIPS 2020)
  - Inputs:  (x, y, t)  →  Output: u
  - 4 hidden layers, 64 units each, sine activations
  - ω₀ = 30 (first layer frequency scale)

Loss:
  L = λ_pde * L_pde + λ_data * L_data + λ_bc * L_bc + λ_ic * L_ic

FEM ground truth: wave_solver.cu snapshots
  - 11 snapshots in results/conv/nx256/snap_NNNNNN.bin
  - float64, shape (257*257,) = 66049 doubles per file
  - time recovered as step * dt, dt = 3.32e-4 s

Author: María García Penalva
Project: PUMPS+AI 2026 poster
"""

import os
import glob
import math
import time
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ============================================================
# 0. Configuration — edit here, nowhere else
# ============================================================
CFG = {
    # Paths (relative to this script's location)
    "snap_dir"   : "../results/conv/nx256",
    "fig_dir"    : "../figures",
    "out_dir"    : "../results/pinn",

    # Grid (must match wave_solver.cu)
    "nx"         : 256,           # elements
    "ny"         : 256,
    "nnx"        : 257,           # nodes = nx+1
    "nny"        : 257,
    "dt_fem"     : 3.32e-4,       # FEM timestep [s]
    "t_end"      : 0.25,          # simulation end time [s]

    # Source (must match wave_solver.cu)
    "sx"         : 0.5,           # source x position
    "sy"         : 0.15,          # source y position
    "f0"         : 20.0,          # Ricker dominant frequency [Hz]
    "sigma_src"  : 0.05,          # Gaussian spatial envelope [m]

    # Marmousi2 velocity field
    "vp_path"    : "../data/marmousi2_256x256.bin",
    "vp_nx"      : 256,
    "vp_ny"      : 256,
    "c_min"      : 1.05,          # [km/s]
    "c_max"      : 4.70,          # [km/s]

    # SIREN architecture
    "hidden_layers" : 4,
    "hidden_units"  : 64,
    "omega0"        : 30.0,

    # Training
    "epochs"        : 5000,
    "lr"            : 1e-4,
    "lr_decay_step" : 2000,       # halve LR every this many epochs
    "lr_decay_gamma": 0.5,
    "n_colloc"      : 8192,       # PDE collocation points per epoch
    "n_bc"          : 512,        # BC points per epoch (per boundary edge)
    "n_ic"          : 1024,       # IC points per epoch

    # Loss weights
    "lambda_pde"    : 1.0,
    "lambda_data"   : 0.1,
    "lambda_bc"     : 10.0,
    "lambda_ic"     : 10.0,

    # Amplitude normalisation (divide snapshots by this before training)
    # Set to None to auto-compute from data
    "u_scale"       : None,

    # Logging
    "print_every"   : 100,
    "save_every"    : 1000,
}

# ============================================================
# 1. Device
# ============================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")
if DEVICE.type == "cuda":
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# ============================================================
# 2. Load Marmousi2 velocity field
# ============================================================
def load_vp(cfg):
    """
    Load Marmousi2 Vp model from binary file.
    Returns a float32 tensor of shape (vp_ny, vp_nx) on DEVICE.
    Values in km/s.
    """
    path = os.path.join(os.path.dirname(__file__), cfg["vp_path"])
    n = cfg["vp_nx"] * cfg["vp_ny"]
    vp = np.fromfile(path, dtype=np.float32, count=n)
    vp = vp.reshape(cfg["vp_ny"], cfg["vp_nx"])
    return torch.from_numpy(vp).to(DEVICE)

def wavespeed_at(x, y, vp_tensor, cfg):
    """
    Bilinear interpolation of Vp at physical coordinates (x, y) ∈ [0,1]².
    x, y: tensors of shape (N,)
    Returns: tensor of shape (N,) in km/s
    """
    nx, ny = cfg["vp_nx"], cfg["vp_ny"]
    fx = x * (nx - 1)
    fy = y * (ny - 1)

    ix = torch.clamp(fx.long(), 0, nx - 2)
    iy = torch.clamp(fy.long(), 0, ny - 2)

    tx = fx - ix.float()
    ty = fy - iy.float()

    # Flatten vp for indexing
    vp_flat = vp_tensor.reshape(-1)
    c00 = vp_flat[iy * nx + ix]
    c10 = vp_flat[iy * nx + ix + 1]
    c01 = vp_flat[(iy + 1) * nx + ix]
    c11 = vp_flat[(iy + 1) * nx + ix + 1]

    return (c00 * (1.0 - tx) * (1.0 - ty)
          + c10 * tx          * (1.0 - ty)
          + c01 * (1.0 - tx) * ty
          + c11 * tx          * ty)

# ============================================================
# 3. Load FEM snapshots
# ============================================================
def load_snapshots(cfg):
    """
    Load all FEM snapshots from snap_dir.
    Returns:
        coords : float32 tensor (N_total, 3) — (x, y, t) for every node × snapshot
        u_vals : float32 tensor (N_total,)   — normalised u values
        u_scale: float scalar used for normalisation
    """
    snap_dir = os.path.join(os.path.dirname(__file__), cfg["snap_dir"])
    files = sorted(glob.glob(os.path.join(snap_dir, "snap_*.bin")))
    if len(files) == 0:
        raise FileNotFoundError(f"No snapshots found in {snap_dir}")
    print(f"Found {len(files)} snapshots in {snap_dir}")

    nnx, nny = cfg["nnx"], cfg["nny"]
    n_nodes = nnx * nny

    # Build node coordinate arrays (same for every snapshot)
    ix_arr = np.arange(nnx, dtype=np.float32)
    iy_arr = np.arange(nny, dtype=np.float32)
    # x varies along first index, y along second (matches node_id = ix*nny + iy)
    IX, IY = np.meshgrid(ix_arr, iy_arr, indexing="ij")  # (nnx, nny)
    x_nodes = (IX / cfg["nx"]).reshape(-1).astype(np.float32)  # [0,1]
    y_nodes = (IY / cfg["ny"]).reshape(-1).astype(np.float32)

    all_x, all_y, all_t, all_u = [], [], [], []

    for fpath in files:
        # Extract step number from filename
        bname = os.path.basename(fpath)                 # snap_000075.bin
        step  = int(bname.replace("snap_","").replace(".bin",""))
        t_val = step * cfg["dt_fem"]

        u_raw = np.fromfile(fpath, dtype=np.float64).astype(np.float32)
        if u_raw.size != n_nodes:
            raise ValueError(f"{fpath}: expected {n_nodes} values, got {u_raw.size}")

        all_x.append(x_nodes)
        all_y.append(y_nodes)
        all_t.append(np.full(n_nodes, t_val, dtype=np.float32))
        all_u.append(u_raw)

    all_x = np.concatenate(all_x)
    all_y = np.concatenate(all_y)
    all_t = np.concatenate(all_t)
    all_u = np.concatenate(all_u)

    # Normalise amplitude so loss values are O(1)
    u_scale = cfg["u_scale"]
    if u_scale is None:
        u_scale = float(np.abs(all_u).max())
        if u_scale < 1e-30:
            u_scale = 1.0
    print(f"Amplitude scale (u_scale): {u_scale:.6e}")
    all_u /= u_scale

    coords = np.stack([all_x, all_y, all_t], axis=1)   # (N, 3)
    coords_t = torch.from_numpy(coords).to(DEVICE)
    u_t      = torch.from_numpy(all_u).to(DEVICE)

    return coords_t, u_t, u_scale

# ============================================================
# 4. SIREN network
# ============================================================
class SirenLayer(nn.Module):
    def __init__(self, in_f, out_f, omega0=30.0, is_first=False):
        super().__init__()
        self.omega0   = omega0
        self.is_first = is_first
        self.linear   = nn.Linear(in_f, out_f)
        self._init_weights(in_f)

    def _init_weights(self, in_f):
        with torch.no_grad():
            if self.is_first:
                # Sitzmann et al. eq. (1): uniform(-1/fan_in, 1/fan_in)
                bound = 1.0 / in_f
            else:
                # Sitzmann et al. eq. (2): uniform(-√(6/fan_in)/ω₀, √(6/fan_in)/ω₀)
                bound = math.sqrt(6.0 / in_f) / self.omega0
            self.linear.weight.uniform_(-bound, bound)
            self.linear.bias.uniform_(-bound, bound)

    def forward(self, x):
        return torch.sin(self.omega0 * self.linear(x))


class SirenNet(nn.Module):
    """
    SIREN network: (x, y, t) → u
    Architecture: input(3) → [hidden_layers × hidden_units] → output(1)
    Final layer is linear (no sine activation).
    """
    def __init__(self, hidden_layers=4, hidden_units=64, omega0=30.0):
        super().__init__()
        layers = []

        # First layer
        layers.append(SirenLayer(3, hidden_units, omega0=omega0, is_first=True))

        # Hidden layers
        for _ in range(hidden_layers - 1):
            layers.append(SirenLayer(hidden_units, hidden_units, omega0=omega0))

        self.net = nn.Sequential(*layers)

        # Output layer — linear, no activation
        self.out = nn.Linear(hidden_units, 1)
        with torch.no_grad():
            bound = math.sqrt(6.0 / hidden_units) / omega0
            self.out.weight.uniform_(-bound, bound)
            self.out.bias.uniform_(-bound, bound)

    def forward(self, xyt):
        """
        xyt: tensor of shape (N, 3) — columns are (x, y, t)
        Returns: tensor of shape (N,)
        """
        return self.out(self.net(xyt)).squeeze(-1)

# ============================================================
# 5. PDE residual (via autograd)
# ============================================================
def pde_residual(model, xyt, vp_tensor, cfg):
    """
    Computes the PDE residual:
        R = d²u/dt² - c(x,y)² * (d²u/dx² + d²u/dy²)

    All second derivatives via torch.autograd.grad with create_graph=True.
    xyt: (N, 3) with requires_grad=True
    Returns: residual tensor of shape (N,)
    """
    xyt = xyt.requires_grad_(True)
    u   = model(xyt)                               # (N,)

    # First derivatives
    grads = torch.autograd.grad(
        u, xyt,
        grad_outputs=torch.ones_like(u),
        create_graph=True
    )[0]                                           # (N, 3)

    u_x = grads[:, 0]
    u_y = grads[:, 1]
    u_t = grads[:, 2]

    # Second derivatives
    u_xx = torch.autograd.grad(u_x, xyt,
        grad_outputs=torch.ones_like(u_x),
        create_graph=True)[0][:, 0]

    u_yy = torch.autograd.grad(u_y, xyt,
        grad_outputs=torch.ones_like(u_y),
        create_graph=True)[0][:, 1]

    u_tt = torch.autograd.grad(u_t, xyt,
        grad_outputs=torch.ones_like(u_t),
        create_graph=True)[0][:, 2]

    # Wave speed at collocation points
    x = xyt[:, 0].detach()
    y = xyt[:, 1].detach()
    c = wavespeed_at(x, y, vp_tensor, cfg)        # (N,) in km/s

    residual = u_tt - c * c * (u_xx + u_yy)
    return residual

# ============================================================
# 6. Source term (Ricker wavelet) — for IC velocity condition
# ============================================================
def ricker_source(x, y, t, cfg):
    """
    Ricker wavelet point source — matches wave_solver.cu source_at().
    x, y, t: tensors of shape (N,)
    Returns: tensor of shape (N,)
    """
    f0    = cfg["f0"]
    t0    = 1.0 / f0
    sx    = cfg["sx"]
    sy    = cfg["sy"]
    sigma = cfg["sigma_src"]

    r2      = (x - sx)**2 + (y - sy)**2
    spatial = torch.exp(-r2 / (2.0 * sigma**2))

    tau = t - t0
    u   = math.pi * f0 * tau
    ricker = (1.0 - 2.0 * u**2) * torch.exp(-u**2)

    return spatial * ricker

# ============================================================
# 7. Sampling utilities
# ============================================================
def sample_collocation(n, t_end, device):
    """Random (x,y,t) points in (0,1)² × (0, t_end), interior only."""
    pts = torch.rand(n, 3, device=device)
    pts[:, 2] *= t_end
    # push slightly off boundary to avoid BC overlap
    pts[:, :2] = pts[:, :2] * 0.98 + 0.01
    return pts

def sample_bc(n_per_edge, t_end, device):
    """
    Points on the 4 boundary edges × random t.
    Returns tensor of shape (4*n_per_edge, 3).
    """
    t  = torch.rand(4 * n_per_edge, device=device) * t_end
    s  = torch.rand(4 * n_per_edge, device=device)  # parameter along edge

    x = torch.zeros(4 * n_per_edge, device=device)
    y = torch.zeros(4 * n_per_edge, device=device)

    n = n_per_edge
    # Bottom: y=0, x in [0,1]
    x[:n]      = s[:n];       y[:n]      = 0.0
    # Top:    y=1, x in [0,1]
    x[n:2*n]   = s[n:2*n];    y[n:2*n]   = 1.0
    # Left:   x=0, y in [0,1]
    x[2*n:3*n] = 0.0;         y[2*n:3*n] = s[2*n:3*n]
    # Right:  x=1, y in [0,1]
    x[3*n:]    = 1.0;         y[3*n:]    = s[3*n:]

    return torch.stack([x, y, t], dim=1)

def sample_ic(n, device):
    """
    Points in (0,1)² at t=0 (interior).
    Returns tensor of shape (n, 3) with t column = 0.
    """
    pts = torch.rand(n, 2, device=device) * 0.98 + 0.01
    t   = torch.zeros(n, 1, device=device)
    return torch.cat([pts, t], dim=1)

def sample_data_batch(coords_data, u_data, batch_size):
    """Random minibatch from FEM supervision data."""
    idx = torch.randint(0, coords_data.shape[0], (batch_size,), device=coords_data.device)
    return coords_data[idx], u_data[idx]

# ============================================================
# 8. Training
# ============================================================
def train(cfg):
    os.makedirs(os.path.join(os.path.dirname(__file__), cfg["fig_dir"]), exist_ok=True)
    os.makedirs(os.path.join(os.path.dirname(__file__), cfg["out_dir"]), exist_ok=True)

    # Data
    print("\n--- Loading data ---")
    vp_tensor                    = load_vp(cfg)
    coords_data, u_data, u_scale = load_snapshots(cfg)
    cfg["u_scale"]               = u_scale   # store for evaluation

    # Data batch size: use all nodes from 2 random snapshots per step
    n_nodes       = cfg["nnx"] * cfg["nny"]
    data_batch_sz = 2 * n_nodes   # ~132k points — subsample if OOM
    data_batch_sz = min(data_batch_sz, 32768)   # cap for memory safety

    # Model
    print("\n--- Building model ---")
    model = SirenNet(
        hidden_layers = cfg["hidden_layers"],
        hidden_units  = cfg["hidden_units"],
        omega0        = cfg["omega0"],
    ).to(DEVICE)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"SIREN: {cfg['hidden_layers']} hidden layers × {cfg['hidden_units']} units")
    print(f"Parameters: {n_params:,}")

    # Optimiser + scheduler
    optimiser = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimiser,
        step_size = cfg["lr_decay_step"],
        gamma     = cfg["lr_decay_gamma"]
    )

    # Loss history
    history = {k: [] for k in ["total", "pde", "data", "bc", "ic"]}

    print(f"\n--- Training for {cfg['epochs']} epochs on {DEVICE} ---")
    t_start = time.time()

    for epoch in range(1, cfg["epochs"] + 1):

        model.train()
        optimiser.zero_grad()

        # ---- PDE residual loss ----------------------------------------
        xyt_col = sample_collocation(cfg["n_colloc"], cfg["t_end"], DEVICE)
        xyt_col.requires_grad_(True)
        res = pde_residual(model, xyt_col, vp_tensor, cfg)
        loss_pde = (res ** 2).mean()

        # ---- BC loss --------------------------------------------------
        xyt_bc  = sample_bc(cfg["n_bc"], cfg["t_end"], DEVICE)
        u_bc    = model(xyt_bc)
        loss_bc = (u_bc ** 2).mean()

        # ---- IC loss (displacement = 0 at t=0) ------------------------
        xyt_ic  = sample_ic(cfg["n_ic"], DEVICE)
        u_ic    = model(xyt_ic)
        loss_ic_u = (u_ic ** 2).mean()

        # IC velocity: du/dt(x,y,0) = 0 (quiescent initial state)
        xyt_ic_grad = xyt_ic.clone().requires_grad_(True)
        u_ic2       = model(xyt_ic_grad)
        u_ic_t      = torch.autograd.grad(
            u_ic2, xyt_ic_grad,
            grad_outputs=torch.ones_like(u_ic2),
            create_graph=True
        )[0][:, 2]
        loss_ic_v = (u_ic_t ** 2).mean()

        loss_ic = loss_ic_u + loss_ic_v

        # ---- Data supervision loss ------------------------------------
        xyt_b, u_b = sample_data_batch(coords_data, u_data, data_batch_sz)
        u_pred      = model(xyt_b)
        loss_data   = ((u_pred - u_b) ** 2).mean()

        # ---- Total loss -----------------------------------------------
        loss = (cfg["lambda_pde"]  * loss_pde
              + cfg["lambda_data"] * loss_data
              + cfg["lambda_bc"]   * loss_bc
              + cfg["lambda_ic"]   * loss_ic)

        loss.backward()
        # Gradient clipping — important for SIREN stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimiser.step()
        scheduler.step()

        # ---- Logging --------------------------------------------------
        history["total"].append(loss.item())
        history["pde"].append(loss_pde.item())
        history["data"].append(loss_data.item())
        history["bc"].append(loss_bc.item())
        history["ic"].append(loss_ic.item())

        if epoch % cfg["print_every"] == 0 or epoch == 1:
            elapsed = time.time() - t_start
            lr_now  = scheduler.get_last_lr()[0]
            print(f"Epoch {epoch:5d}/{cfg['epochs']} | "
                  f"loss={loss.item():.4e}  "
                  f"pde={loss_pde.item():.4e}  "
                  f"data={loss_data.item():.4e}  "
                  f"bc={loss_bc.item():.4e}  "
                  f"ic={loss_ic.item():.4e}  "
                  f"lr={lr_now:.2e}  "
                  f"t={elapsed:.1f}s")

        if epoch % cfg["save_every"] == 0 or epoch == cfg["epochs"]:
            ckpt_path = os.path.join(
                os.path.dirname(__file__), cfg["out_dir"],
                f"pinn_forward_epoch{epoch:05d}.pt"
            )
            torch.save({
                "epoch"   : epoch,
                "model"   : model.state_dict(),
                "cfg"     : cfg,
                "history" : history,
            }, ckpt_path)
            print(f"  Checkpoint saved: {ckpt_path}")

    total_time = time.time() - t_start
    print(f"\nTraining complete in {total_time:.1f}s ({total_time/60:.1f} min)")

    return model, history, cfg

# ============================================================
# 9. Evaluation: dense grid at arbitrary t
# ============================================================
def evaluate_on_grid(model, t_val, cfg, u_scale=None):
    """
    Evaluate PINN on the full 257×257 node grid at time t_val.
    Returns numpy array of shape (257, 257) in physical units (not normalised).
    """
    model.eval()
    nnx, nny = cfg["nnx"], cfg["nny"]

    ix = np.linspace(0, 1, nnx, dtype=np.float32)
    iy = np.linspace(0, 1, nny, dtype=np.float32)
    IX, IY = np.meshgrid(ix, iy, indexing="ij")     # (nnx, nny)

    x_flat = torch.from_numpy(IX.reshape(-1)).to(DEVICE)
    y_flat = torch.from_numpy(IY.reshape(-1)).to(DEVICE)
    t_flat = torch.full_like(x_flat, t_val)

    xyt = torch.stack([x_flat, y_flat, t_flat], dim=1)

    with torch.no_grad():
        # Process in chunks to avoid OOM on large grids
        chunk = 16384
        u_chunks = []
        for i in range(0, xyt.shape[0], chunk):
            u_chunks.append(model(xyt[i:i+chunk]))
        u_flat = torch.cat(u_chunks).cpu().numpy()

    u_grid = u_flat.reshape(nnx, nny)

    # Denormalise
    if u_scale is not None:
        u_grid = u_grid * u_scale

    return u_grid

# ============================================================
# 10. Post-processing plots
# ============================================================
def plot_loss(history, cfg):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    epochs = np.arange(1, len(history["total"]) + 1)

    ax = axes[0]
    for key, label, ls in [
        ("total", "Total",  "-"),
        ("pde",   "PDE",    "--"),
        ("data",  "Data",   "-."),
        ("bc",    "BC",     ":"),
        ("ic",    "IC",     (0,(3,1,1,1))),
    ]:
        ax.semilogy(epochs, history[key], ls, label=label, lw=1.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training loss (log scale)")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)

    ax = axes[1]
    ax.semilogy(epochs, history["pde"],  "--", label="PDE",  lw=1.5)
    ax.semilogy(epochs, history["data"], "-.", label="Data", lw=1.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("PDE vs Data loss")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)

    fig.tight_layout()
    path = os.path.join(os.path.dirname(__file__), cfg["fig_dir"], "pinn_forward_loss.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Loss plot saved: {path}")

def plot_snapshot_comparison(model, cfg, u_scale, snap_step=300):
    """
    Side-by-side comparison: FEM snapshot vs PINN prediction vs difference.
    snap_step: FEM step number to compare against (default 300 → t=0.0997s)
    """
    t_val    = snap_step * cfg["dt_fem"]
    snap_dir = os.path.join(os.path.dirname(__file__), cfg["snap_dir"])
    snap_path = os.path.join(snap_dir, f"snap_{snap_step:06d}.bin")

    if not os.path.exists(snap_path):
        print(f"Snapshot {snap_path} not found, skipping comparison plot.")
        return

    n_nodes = cfg["nnx"] * cfg["nny"]
    u_fem   = np.fromfile(snap_path, dtype=np.float64).reshape(cfg["nnx"], cfg["nny"])
    u_pinn  = evaluate_on_grid(model, t_val, cfg, u_scale=u_scale)
    diff    = u_pinn - u_fem

    vmax = max(np.abs(u_fem).max(), np.abs(u_pinn).max())
    vmax = vmax if vmax > 0 else 1.0

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    titles = [f"FEM  (t={t_val:.4f}s)", f"PINN (t={t_val:.4f}s)", "PINN − FEM"]
    data   = [u_fem, u_pinn, diff]
    vmaxs  = [vmax, vmax, np.abs(diff).max()]

    for ax, d, title, vm in zip(axes, data, titles, vmaxs):
        vm = vm if vm > 0 else 1.0
        im = ax.imshow(d.T, origin="lower", extent=[0,1,0,1],
                       cmap="seismic", vmin=-vm, vmax=vm, aspect="auto")
        ax.set_title(title)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.tight_layout()
    path = os.path.join(os.path.dirname(__file__), cfg["fig_dir"],
                        f"pinn_forward_comparison_step{snap_step}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Comparison plot saved: {path}")

# ============================================================
# 11. Entry point
# ============================================================
if __name__ == "__main__":
    model, history, cfg = train(CFG)

    print("\n--- Post-processing ---")
    plot_loss(history, cfg)

    u_scale = cfg["u_scale"]
    # Compare against a mid-simulation snapshot
    plot_snapshot_comparison(model, cfg, u_scale, snap_step=300)
    plot_snapshot_comparison(model, cfg, u_scale, snap_step=450)

    # Save final model with metadata
    final_path = os.path.join(
        os.path.dirname(__file__), cfg["out_dir"], "pinn_forward_final.pt"
    )
    torch.save({
        "model"  : model.state_dict(),
        "cfg"    : cfg,
        "history": history,
    }, final_path)
    print(f"Final model saved: {final_path}")
    print("\nDone.")

