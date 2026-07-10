// ===============================================================
// wave_solver.cu
// ===============================================================
//
// This is an Assembly-Free solver for the 2D acoustic wave equation
// in a fixed heterogeneous medium.
// There is explicit leapfrog time stepping (central-difference).
// No iterative solve is required (provide lumped mass matrix).
//
// Project: PUMPS + AI 2026, poster. Extension of assembly-free architecture
// from heat_solver.cu (MNHACK25, 7th MN5 Hackathon) to a hyperbolic (wave) PDE.
//
//
// PDE at hand:
// d^2u/dt^2 = c(x,y)^2 * Laplacian(u) in Omega = [0,1]^2
// u = 0 on boundary (Dirichlet BC)
//
// Discretization: bilinear quadrilateral FEM on uniform grid.
// Time integration: explicit leapfrog (central difference).
// =================================================================

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>

// ===============================================================================================
// 0. Grid Structure
// Actual Grid object gets created and configured in main().
struct Grid {
    int nx, ny;     //positions of cells in x-axis and y-axis
    double hx, hy;  //size of cells (hx-width, hy-height)
    // number of nodes in each dimension, x and y. (nnx = nx + 1 and nny = ny +1)
    int nnx, nny;
    // number of nodes and number of elements (cells)
    long n_nodes, n_cells;
};

// Helpers - not kernels, not physics. utility units callable by GPU and CPU.
// -----------------------------------------------------------------------------------------------
// Given a node's grid position (ix, iy) return its index in the flat 1Darray stored
// by GPU: ix * nny + iy
__host__ __device__ __forceinline__ long node_id(int ix, int iy, const Grid& g){
    return (long)ix * g.nny + iy;
}
// Given a node's grid position (ix, iy), return its physical coordinates in m (x,y)
__host__ __device__ __forceinline__ void node_coords(int ix, int iy, const Grid& g, double& x, double& y){
    x = (double)ix * g.hx;
    y = (double)iy * g.hy;
}
// Given (ix, iy), returns true if the node is on the edge of the domain. Enforces Dirichlet BC.
__host__ __device__ __forceinline__ bool is_boundary (int ix, int iy, const Grid& g){
    return (ix == 0 || iy == 0 || ix == g.nx || iy == g.ny);
}

//--------------------------------------------------------------------------------------------------
// Physics Input: these are device-only functions (called from GPU)
// Marmousi2 Vp model on device (loaded in main, read-only during simulation)
__device__ float* d_vp = nullptr;
__device__ int d_vp_nx = 256;
__device__ int d_vp_ny = 256;

// Homogeneous medium - flags
__device__ bool d_homogeneous = false;
__device__ float d_c_homogeneous = 1.5f;

// Wave speed field c (x, y) - real Marmousi2 Vp model
// (x, y) in [0,1]^2: x=0 left, x=1 right, y=0 surface, y=1 max depth.
// Returns velocity in km/s via bilinear interpolation
__device__ __forceinline__ double wavespeed_at(double x, double y){
    // homogeneous early return
    if (d_homogeneous) return (double)d_c_homogeneous;
    double fx = x * (d_vp_nx - 1);
    double fy = y * (d_vp_ny - 1);

    int ix = (int)fx;
    int iy = (int)fy;

    ix = max(0, min(ix, d_vp_nx - 2));
    iy = max(0, min(iy, d_vp_ny - 2));

    double tx = fx - ix;
    double ty = fy - iy;

    double c00 = (double)d_vp[iy * d_vp_nx + ix];
    double c10 = (double)d_vp[iy * d_vp_nx + ix + 1];
    double c01 = (double)d_vp[(iy + 1) * d_vp_nx + ix];
    double c11 = (double)d_vp[(iy + 1) * d_vp_nx + ix + 1];

    return c00 * (1.0 - tx) * (1.0 - ty)
    + c10 * tx * (1.0 - ty)
    + c01 * (1.0 - tx) * ty
    + c11 * tx * ty;
}

// Source term: Ricker wavelet point source at domain centre
// Returns the source value at physical position (x, y) and time t
// f0 is the dominant frequency (Hz)
// t0 is the time delay (s)
__device__ __forceinline__ double source_at(double x, double y, double t){
    // f0 = 20 Hz: satisfies >= 8 pts/wavelength at c_min = 1.05 km/s on a 256-node grid.
    // Standard benchmark frequency for Marmousi2 (Martin et al. 2006).
    const double f0 = 20.0;
    const double t0 = 1.0 / f0;
    const double sx = 0.5, sy = 0.15; //source position: centre-top of [0,1]^2 domain

    // Gaussian spatial envelope with fixed physical width sigma=0.02 km.
    // Resolution-independent: total injected energy is the same on all grids.
    const double sigma = 0.05;
    double r2 = (x - sx)*(x - sx) + (y - sy)*(y - sy);
    double spatial = exp(-r2 / (2.0 * sigma * sigma));

    // Ricker wavelet: second derivative of Gaussian
    double tau = t - t0;
    double u = M_PI * f0 * tau;
    return spatial * (1.0 - 2.0 * u*u) * exp(-u*u);
}
// ===============================================================================================
// 1. Shape functions - element indexing
//
// Each element (cell) is a bilinear quadrilateral with 4 corner nodes.
// Element (ex, ey) occupies x in [ex*hx, (ex+1)*hx], y in [ey*hy, (ey+1)*hy].
// Node numbering within the element (local indices 0-3):
//
//   3 --- 2        (ix,   iy+1) --- (ix+1, iy+1)
//   |     |    =        |                  |
//   0 --- 1        (ix,   iy  ) --- (ix+1, iy  )
//
// Local node 0 = bottom-left, 1 = bottom-right, 2 = top-right, 3 = top-left.
// This is standard counter-clockwise ordering.
//
// Reference element: xi in [-1,1], eta in [-1,1].
// Bilinear shape functions:
//   N0 = (1-xi)(1-eta)/4
//   N1 = (1+xi)(1-eta)/4
//   N2 = (1+xi)(1+eta)/4
//   N3 = (1-xi)(1+eta)/4

// Given element (ex, ey) and local node index (0-3),
// return the global node indices (ix, iy) of that corner.
__host__ __device__ __forceinline__
void elem_node_coords(int ex, int ey, int local,
                    int& ix, int& iy)
{
    // local: 0=BL, 1=BR, 2=TR, 3=TL
    ix = ex + (local == 1 || local == 2 ? 1 : 0);
    iy = ey + (local == 2 || local == 3 ? 1 : 0);
}

// Given element (ex, ey) and local node index (0-3),
// return the flat global node ID consistent with node_id().
__host__ __device__ __forceinline__
long elem_node_id(int ex, int ey, int local, const Grid& g)
{
    int ix, iy;
    elem_node_coords(ex, ey, local, ix, iy);
    return node_id(ix, iy, g);
}

// Bilinear shape function values at reference coords (xi, eta).
// N[0..3] must be a float/double array of length 4.
__device__ __forceinline__
void shape_vals(double xi, double eta, double N[4])
{
    N[0] = 0.25 * (1.0 - xi) * (1.0 - eta);
    N[1] = 0.25 * (1.0 + xi) * (1.0 - eta);
    N[2] = 0.25 * (1.0 + xi) * (1.0 + eta);
    N[3] = 0.25 * (1.0 - xi) * (1.0 + eta);
}

// Bilinear shape function gradients at reference coords (xi, eta).
// dNdxi[i] and dNdeta[i] are d(Ni)/d(xi) and d(Ni)/d(eta).
__device__ __forceinline__
void shape_grads(double xi, double eta,
                double dNdxi[4], double dNdeta[4])
{
    dNdxi[0] = -0.25 * (1.0 - eta);
    dNdxi[1] =  0.25 * (1.0 - eta);
    dNdxi[2] =  0.25 * (1.0 + eta);
    dNdxi[3] = -0.25 * (1.0 + eta);

    dNdeta[0] = -0.25 * (1.0 - xi);
    dNdeta[1] = -0.25 * (1.0 + xi);
    dNdeta[2] =  0.25 * (1.0 + xi);
    dNdeta[3] =  0.25 * (1.0 - xi);
}

// Jacobian of the mapping from reference to physical element.
// For a uniform rectangular grid this is always diagonal:
//   J = [[hx/2, 0], [0, hy/2]]
// detJ = hx*hy/4
// We precompute detJ and the inverse Jacobian factors once per kernel launch.
__device__ __forceinline__
void jacobian(const Grid& g, double& detJ,
            double& inv_jxx, double& inv_jyy)
{
    detJ     = 0.25 * g.hx * g.hy;
    inv_jxx  = 2.0 / g.hx;   // d(xi)/d(x)
    inv_jyy  = 2.0 / g.hy;   // d(eta)/d(y)
}
// ===============================================================================================
// 2. Kernels
//
// Two kernels per timestep:
//   k_apply  : computes K*u (stiffness-vector product), one thread per element
//   leapfrog : advances u using the lumped mass matrix, one thread per node
//
// Gauss quadrature points and weights for 2x2 rule on [-1,1]^2:
//   points:  +/- 1/sqrt(3)
//   weights: 1.0 each (product rule, total weight = 4 for the reference element)

// -----------------------------------------------------------------------------------------------
// k_apply: computes out[i] += sum over elements sharing node i of
//          integral( c^2 * grad(Ni) . grad(u) ) dV
// This is the assembly-free stiffness-vector product K*u.
// out[] must be zeroed before this kernel is called each timestep.
// -----------------------------------------------------------------------------------------------
// Kernel 1.
__global__ void k_apply(const double* __restrict__ u,
                            double* __restrict__ out,
                        const Grid g)
{
    // One thread per element
    int ex = blockIdx.x * blockDim.x + threadIdx.x;
    int ey = blockIdx.y * blockDim.y + threadIdx.y;
    if (ex >= g.nx || ey >= g.ny) return;

    // Jacobian (same for all elements on uniform grid)
    double detJ, inv_jxx, inv_jyy;
    jacobian(g, detJ, inv_jxx, inv_jyy);

    // Global node IDs and values for this element's 4 corners
    long gid[4];
    double u_loc[4];
    for (int l = 0; l < 4; l++) {
        gid[l]   = elem_node_id(ex, ey, l, g);
        u_loc[l] = u[gid[l]];
    }

    // 2x2 Gauss quadrature
    const double gp = 0.577350269189626;  // 1/sqrt(3)
    const double xi_pts[2]  = {-gp,  gp};
    const double eta_pts[2] = {-gp,  gp};
    // weight = 1.0 for each point; weight*weight = 1.0 per pair

    // Accumulate local residual for each corner node
    double r_loc[4] = {0.0, 0.0, 0.0, 0.0};

    for (int qi = 0; qi < 2; qi++) {
        for (int qj = 0; qj < 2; qj++) {
            double xi  = xi_pts[qi];
            double eta = eta_pts[qj];

            // Physical centre of element (for wavespeed sampling)
            double xc = ((double)ex + 0.5 * (1.0 + xi))  * g.hx;  // correct Gauss point
            double yc = ((double)ey + 0.5 * (1.0 + eta)) * g.hy;  // correct Gauss point
            double c  = wavespeed_at(xc, yc);
            double c2 = c * c;
            double fac = c2 * detJ;

            double dNdxi[4], dNdeta[4];
            shape_grads(xi, eta, dNdxi, dNdeta);

            // Map gradients to physical space
            // dN/dx = dN/dxi * dxi/dx = dN/dxi * inv_jxx
            // dN/dy = dN/deta * deta/dy = dN/deta * inv_jyy
            double dNdx[4], dNdy[4];
            for (int l = 0; l < 4; l++) {
                dNdx[l] = dNdxi[l]  * inv_jxx;
                dNdy[l] = dNdeta[l] * inv_jyy;
            }

            // grad(u) at this quadrature point
            double dudx = 0.0, dudy = 0.0;
            for (int l = 0; l < 4; l++) {
                dudx += dNdx[l] * u_loc[l];
                dudy += dNdy[l] * u_loc[l];
            }

            // Local stiffness contribution: c^2 * grad(Ni) . grad(u) * detJ * w
            // w = 1*1 = 1 for 2x2 Gauss rule
            for (int l = 0; l < 4; l++) {
                r_loc[l] += fac * (dNdx[l] * dudx + dNdy[l] * dudy);
            }
        }
    }

    // Scatter local contributions to global array (race-safe)
    for (int l = 0; l < 4; l++) {
        atomicAdd(&out[gid[l]], r_loc[l]);
    }
}
// -----------------------------------------------------------------------------------------------
// Kernel 2.
// leapfrog: advances the wavefield one timestep using the explicit leapfrog scheme.
//   u_new[i] = 2*u[i] - u_old[i] - dt^2 * m_inv[i] * (Ku[i] - f[i])
//
// One thread per node.
// Boundary nodes are held at zero (Dirichlet BC).
// m_inv= 1 / (hx * hy) - constant on uniform grid, no array needed
// Ku[] = output of k_apply (already computed this timestep).
// f[i] = source term evaluated at node i and current time t.
// -----------------------------------------------------------------------------------------------
__global__ void leapfrog(const double* __restrict__ u,
                        const double* __restrict__ u_old,
                            double* __restrict__ u_new,
                        const double* __restrict__ Ku,
                        const Grid g,
                        double dt,
                        double t)
{
    // One thread per node
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    if (ix >= g.nnx || iy >= g.nny) return;

    long nid = node_id(ix, iy, g);

    // Enforce Dirichlet BC: boundary nodes stay zero
    if (is_boundary(ix, iy, g)) {
        u_new[nid] = 0.0;
        return;
    }

    // Physical coords of this node
    double x, y;
    node_coords(ix, iy, g, x, y);

    // Source injection
    double f = source_at(x, y, t) * g.hx * g.hy;

    // Leapfrog update
    double m_inv = 1.0 / (g.hx * g.hy);
    u_new[nid] = 2.0 * u[nid]
            - u_old[nid]
               + dt * dt * m_inv * (f - Ku[nid]);
}
// ===============================================================================================
// 3. main()
int main(int argc, char* argv[])
{
    // Defaults
    int nx_arg = 256;
    bool homogeneous = false;
    float c_hom = 1.5f;

    // Parse arguments: --nx N, --homogeneous, --c C
    for (int i=1; i<argc; ++i){
        if(strcmp(argv[i], "--homogeneous") == 0){
            homogeneous = true;
        } else if (strcmp(argv[i], "--nx") == 0 && i+1 <argc) {
            nx_arg = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--c") == 0 && i+1 < argc) {
            c_hom = (float)atof(argv[++i]);
        }
    }

    // 1. Grid ---------------------------------------------------------------------------------
    Grid g;
    g.nx      = nx_arg;
    g.ny      = nx_arg;
    g.hx      = 1.0 / g.nx;
    g.hy      = 1.0 / g.ny;
    g.nnx     = g.nx + 1;
    g.nny     = g.ny + 1;
    g.n_nodes = (long)g.nnx * g.nny;
    g.n_cells = (long)g.nx  * g.ny;

    printf("Grid: %d x %d elements | %ld nodes | hx=%.6f hy=%.6f km\n",
    g.nx, g.ny, g.n_nodes, g.hx, g.hy);

    // Push homogeneous flags to GPU
    cudaMemcpyToSymbol(d_homogeneous, &homogeneous, sizeof(bool));
    cudaMemcpyToSymbol(d_c_homogeneous, &c_hom, sizeof(float));

    // 2. CFL-stable dt --------------------------------------------------------------------------
    const double c_max = 4.7;
    const double h_min = fmin(g.hx, g.hy);
    const double cfl   = 0.4;
    const double dt    = cfl * h_min / c_max;
    printf("dt = %.6e s  (CFL=%.2f, c_max=%.1f km/s)\n", dt, cfl, c_max);

    //3. Simulation parameters --------------------------------------------------------------------------
    const double t_end      = 0.25;
    const int    n_steps    = (int)ceil(t_end / dt);
    const int    snap_every = 10;
    const char*  data_path  = "../data/marmousi2_256x256.bin";

    // results_dir branches on homogeneous flag
    char results_dir[512];
    if (homogeneous)
        snprintf(results_dir, sizeof(results_dir), "../results/conv/nx%03d", nx_arg);
    else
        snprintf(results_dir, sizeof(results_dir), "../results/dense/nx256");
    // Create output directory (no-op if it already exists)
    {
        char cmd[600];
        snprintf(cmd, sizeof(cmd), "mkdir -p %s", results_dir);
        system(cmd);
    }

    printf("n_steps=%d  snap_every=%d  t_end=%.3f s\n",
        n_steps, snap_every, t_end);
    printf("results_dir: %s\n", results_dir);

    // 4. Load Marmousi2 into d_vp -------------------------------------------------------------
    if (!homogeneous) {
        FILE* fp = fopen(data_path, "rb");
        if (!fp) {
            fprintf(stderr, "Cannot open %s\n", data_path);
            return EXIT_FAILURE;
        }
        const int vp_n = 256 * 256;
        float* h_vp = (float*)malloc(vp_n * sizeof(float));
        if (!h_vp) { fprintf(stderr, "malloc failed for Vp\n"); return EXIT_FAILURE; }

        size_t nread = fread(h_vp, sizeof(float), vp_n, fp);
        fclose(fp);
        if ((int)nread != vp_n) {
            fprintf(stderr, "Vp: expected %d floats, got %zu\n", vp_n, nread);
            free(h_vp); return EXIT_FAILURE;
        }

        float* d_vp_tmp;
        cudaMalloc(&d_vp_tmp, vp_n * sizeof(float));
        cudaMemcpy(d_vp_tmp, h_vp, vp_n * sizeof(float), cudaMemcpyHostToDevice);
        free(h_vp);

        // Write device pointer into the __device__ symbol
        cudaMemcpyToSymbol(d_vp,    &d_vp_tmp, sizeof(float*));
        int vp_nx = 256, vp_ny = 256;
        cudaMemcpyToSymbol(d_vp_nx, &vp_nx,    sizeof(int));
        cudaMemcpyToSymbol(d_vp_ny, &vp_ny,    sizeof(int));

        printf("Marmousi2 loaded: 256x256 floats\n");
        // d_vp_tmp intentionally not freed here — d_vp holds the live pointer
    } else {
        printf("Homogeneous medium: c=%.2f km/s\n", c_hom);
    }

    // 5. Allocate wavefield arrays -------------------------------------------------------------
    const size_t field_bytes = (size_t)g.n_nodes * sizeof(double);

    double *d_u, *d_u_old, *d_u_new, *d_Ku;
    cudaMalloc(&d_u,     field_bytes);
    cudaMalloc(&d_u_old, field_bytes);
    cudaMalloc(&d_u_new, field_bytes);
    cudaMalloc(&d_Ku,    field_bytes);

    // 6. Initial conditions -------------------------------------------------------------
    cudaMemset(d_u,     0, field_bytes);
    cudaMemset(d_u_old, 0, field_bytes);
    cudaMemset(d_u_new, 0, field_bytes);
    cudaMemset(d_Ku,    0, field_bytes);

    // 7. Kernel launch configs -------------------------------------------------------------
    dim3 block_elem(16, 16);
    dim3 grid_elem((g.nx + 15) / 16, (g.ny + 15) / 16);

    dim3 block_node(16, 16);
    dim3 grid_node((g.nnx + 15) / 16, (g.nny + 15) / 16);

    // 8. Time loop -------------------------------------------------------------
    printf("Starting time loop...\n");
    double t = 0.0;

    for (int step = 0; step < n_steps; ++step)
    {
        cudaMemset(d_Ku, 0, field_bytes);

        k_apply<<<grid_elem, block_elem>>>(d_u, d_Ku, g);

        leapfrog<<<grid_node, block_node>>>(d_u, d_u_old, d_u_new, d_Ku, g, dt, t);

        // Pointer rotation — no data copy
        double* tmp = d_u_old;
        d_u_old = d_u;
        d_u     = d_u_new;
        d_u_new = tmp;

        t += dt;

        if ((step + 1) % snap_every == 0 || step == n_steps - 1)
        {
            cudaDeviceSynchronize();
            printf("step %d / %d  t=%.4f s\n", step + 1, n_steps, t);

            // Copy to host and write snapshot
            double* h_u = (double*)malloc(field_bytes);
            cudaMemcpy(h_u, d_u, field_bytes, cudaMemcpyDeviceToHost);

            char path[512];
            snprintf(path, sizeof(path), "%s/snap_%06d.bin", results_dir, step + 1);
            FILE* fp = fopen(path, "wb");
            if (fp) {
                fwrite(h_u, sizeof(double), g.n_nodes, fp);
                fclose(fp);
                printf("  snapshot -> %s\n", path);
            } else {
                fprintf(stderr, "  Cannot open %s\n", path);
            }
            free(h_u);
        }
    }

    cudaDeviceSynchronize();
    printf("Done. t_final = %.6f s\n", t);

    // 9. Free memory -------------------------------------------------------------
    if (!homogeneous) {
        float* d_vp_ptr = nullptr;
        cudaMemcpyFromSymbol(&d_vp_ptr, d_vp, sizeof(float*));
        if (d_vp_ptr) cudaFree(d_vp_ptr);
    }

    cudaFree(d_u);
    cudaFree(d_u_old);
    cudaFree(d_u_new);
    cudaFree(d_Ku);

    printf("Clean exit.\n");
    return 0;
}
// ===============================================================================================

