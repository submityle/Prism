// water_flip_mac_project.wesl
//
// FLIP/APIC MAC pressure-projection stage 3: subtract (here add) the compact
// pressure gradient from the staggered MAC face velocities so the field
// becomes discretely divergence-free.
//
// Given the solved pressure p of the Poisson system A p = b (A the crate's
// negative-Laplacian from pressure_multigrid, b the stage-1 divergence), the
// projected velocity is u' = u + grad(p). With A = (6 p_c - neighbours)/dx^2
// and b taken as the raw +divergence (flip::cell_divergence), adding the
// gradient drives the interior divergence to the solver residual b - A p ~ 0:
//
//   u'[face i] = u[face i] + (p[cell i] - p[cell i-1]) / dx
//
// per staggered family. Only interior faces (those separating two fluid cells)
// are updated; the domain-boundary faces are the fixed solid/free-surface wall
// and are copied through unchanged, matching the Dirichlet p = 0 boundary of
// the multigrid solve. A non-positive dx is a no-op.
//
// ABI (WaterKernel::FlipMacProject shared descriptor: 4 storage + 1 uniform,
// @workgroup_size(64, 1, 1), Faces domain). The MAC passes share the bind
// group: faces (read_write, updated here), divergence (unused), the pressure
// pair (pressure_a read as the solved field, pressure_b unused), plus params.
//
// Packed face layout (identical to mac_divergence): a single f32 buffer holds
// the three staggered families back to back, u then v then w, each i-fastest
// row-major over its own face lattice:
//   u (x faces): (nx+1) * ny     * nz,     u_idx = i + (nx+1)*(j + ny*k)
//   v (y faces): nx     * (ny+1) * nz,     v_idx = i + nx    *(j + (ny+1)*k)
//   w (z faces): nx     * ny     * (nz+1), w_idx = i + nx    *(j + ny*k)
// with offsets u_off = 0, v_off = u_count, w_off = u_count + v_count.
// Cells (and the co-located pressure field) are i-fastest row-major:
//   cell_idx = i + nx*(j + ny*k).

const MAC_EPS: f32 = 1.0e-6;

struct MacProjectParams {
    grid_nx: u32,
    grid_ny: u32,
    grid_nz: u32,
    pad_dims: u32,
    dx: f32,
    pad0: f32,
    pad1: f32,
    pad2: f32,
};

@group(0) @binding(0) var<storage, read_write> faces: array<f32>;
@group(0) @binding(1) var<storage, read_write> divergence: array<f32>;
@group(0) @binding(2) var<storage, read> pressure_a: array<f32>;
@group(0) @binding(3) var<storage, read_write> pressure_b: array<f32>;
@group(0) @binding(4) var<uniform> proj_params: MacProjectParams;

fn u_count(nx: u32, ny: u32, nz: u32) -> u32 {
    return (nx + 1u) * ny * nz;
}

fn v_count(nx: u32, ny: u32, nz: u32) -> u32 {
    return nx * (ny + 1u) * nz;
}

fn w_count(nx: u32, ny: u32, nz: u32) -> u32 {
    return nx * ny * (nz + 1u);
}

fn cell_index(i: u32, j: u32, k: u32, nx: u32, ny: u32) -> u32 {
    return i + nx * (j + ny * k);
}

@compute @workgroup_size(64, 1, 1)
fn mac_project(@builtin(global_invocation_id) gid: vec3<u32>) {
    let nx = proj_params.grid_nx;
    let ny = proj_params.grid_ny;
    let nz = proj_params.grid_nz;
    let dx = proj_params.dx;

    let fid = gid.x;
    let uc = u_count(nx, ny, nz);
    let vc = v_count(nx, ny, nz);
    let wc = w_count(nx, ny, nz);
    let total = uc + vc + wc;

    // Faces outside the packed buffer, or a degenerate spacing: no-op.
    if (fid >= total || dx <= MAC_EPS) {
        return;
    }

    let inv_dx = 1.0 / dx;

    if (fid < uc) {
        // u (x) face: decode (i, j, k) with i in 0..=nx.
        let stride_j = nx + 1u;
        let stride_k = stride_j * ny;
        let k = fid / stride_k;
        let rem = fid - k * stride_k;
        let j = rem / stride_j;
        let i = rem - j * stride_j;
        // Interior x-faces (1..=nx-1) separate two cells; walls stay fixed.
        if (i >= 1u && i + 1u <= nx) {
            let p_hi = pressure_a[cell_index(i, j, k, nx, ny)];
            let p_lo = pressure_a[cell_index(i - 1u, j, k, nx, ny)];
            faces[fid] = faces[fid] + (p_hi - p_lo) * inv_dx;
        }
        return;
    }

    if (fid < uc + vc) {
        // v (y) face.
        let local = fid - uc;
        let stride_j = nx;
        let stride_k = stride_j * (ny + 1u);
        let k = local / stride_k;
        let rem = local - k * stride_k;
        let j = rem / stride_j;
        let i = rem - j * stride_j;
        if (j >= 1u && j + 1u <= ny) {
            let p_hi = pressure_a[cell_index(i, j, k, nx, ny)];
            let p_lo = pressure_a[cell_index(i, j - 1u, k, nx, ny)];
            faces[fid] = faces[fid] + (p_hi - p_lo) * inv_dx;
        }
        return;
    }

    // w (z) face.
    let local = fid - uc - vc;
    let stride_j = nx;
    let stride_k = stride_j * ny;
    let k = local / stride_k;
    let rem = local - k * stride_k;
    let j = rem / stride_j;
    let i = rem - j * stride_j;
    if (k >= 1u && k + 1u <= nz) {
        let p_hi = pressure_a[cell_index(i, j, k, nx, ny)];
        let p_lo = pressure_a[cell_index(i, j, k - 1u, nx, ny)];
        faces[fid] = faces[fid] + (p_hi - p_lo) * inv_dx;
    }
}
