// water_flip_mac_divergence.wesl
//
// FLIP/APIC MAC pressure-projection stage 1: per-cell velocity divergence.
//
// The incompressible pressure projection drives the discrete divergence of the
// staggered MAC velocity field to zero. This kernel computes that divergence
// right-hand side, one scalar per fluid cell, exactly mirroring the crate's
// golden flip::cell_divergence:
//
//   div = ((u[i+1] - u[i]) + (v[j+1] - v[j]) + (w[k+1] - w[k])) / dx
//
// A non-positive dx yields 0 (degenerate guard, matching the CPU golden).
//
// ABI (WaterKernel::FlipMacDivergence shared descriptor: 4 storage + 1 uniform,
// @workgroup_size(4, 4, 4), Grid3d domain). The MAC divergence and Jacobi
// pressure passes share the same bind group — faces (read), divergence
// (written here), and the ping-pong pressure pair (unused by this stage but
// bound so the pressure pass can reuse the group) plus mac_params.
//
// Packed face layout: a single f32 storage buffer holds the three staggered
// face families back to back, u then v then w, each in i-fastest row-major
// order over its own face lattice:
//   u (x faces): (nx+1) * ny     * nz,     u_idx = i + (nx+1)*(j + ny*k)
//   v (y faces): nx     * (ny+1) * nz,     v_idx = i + nx    *(j + (ny+1)*k)
//   w (z faces): nx     * ny     * (nz+1), w_idx = i + nx    *(j + ny*k)
// with offsets u_off = 0, v_off = u_count, w_off = u_count + v_count.
// Cells are i-fastest row-major: cell_idx = i + nx*(j + ny*k).

const MAC_EPS: f32 = 1.0e-6;

struct MacParams {
    grid_nx: u32,
    grid_ny: u32,
    grid_nz: u32,
    pad_dims: u32,
    dx: f32,
    pad0: f32,
    pad1: f32,
    pad2: f32,
};

@group(0) @binding(0) var<storage, read> faces: array<f32>;
@group(0) @binding(1) var<storage, read_write> divergence: array<f32>;
@group(0) @binding(2) var<storage, read_write> pressure_a: array<f32>;
@group(0) @binding(3) var<storage, read_write> pressure_b: array<f32>;
@group(0) @binding(4) var<uniform> mac_params: MacParams;

fn u_count(nx: u32, ny: u32, nz: u32) -> u32 {
    return (nx + 1u) * ny * nz;
}

fn v_count(nx: u32, ny: u32, nz: u32) -> u32 {
    return nx * (ny + 1u) * nz;
}

fn u_index(i: u32, j: u32, k: u32, nx: u32, ny: u32) -> u32 {
    return i + (nx + 1u) * (j + ny * k);
}

fn v_index(i: u32, j: u32, k: u32, nx: u32, ny: u32) -> u32 {
    return i + nx * (j + (ny + 1u) * k);
}

fn w_index(i: u32, j: u32, k: u32, nx: u32, ny: u32) -> u32 {
    return i + nx * (j + ny * k);
}

fn cell_index(i: u32, j: u32, k: u32, nx: u32, ny: u32) -> u32 {
    return i + nx * (j + ny * k);
}

@compute @workgroup_size(4, 4, 4)
fn mac_divergence(@builtin(global_invocation_id) gid: vec3<u32>) {
    let nx = mac_params.grid_nx;
    let ny = mac_params.grid_ny;
    let nz = mac_params.grid_nz;

    let i = gid.x;
    let j = gid.y;
    let k = gid.z;

    // Guard the Grid3d brick padding: skip invocations outside the cell domain.
    if (i >= nx || j >= ny || k >= nz) {
        return;
    }

    let cell = cell_index(i, j, k, nx, ny);

    let u_off = 0u;
    let v_off = u_count(nx, ny, nz);
    let w_off = v_off + v_count(nx, ny, nz);

    let x_neg = faces[u_off + u_index(i, j, k, nx, ny)];
    let x_pos = faces[u_off + u_index(i + 1u, j, k, nx, ny)];
    let y_neg = faces[v_off + v_index(i, j, k, nx, ny)];
    let y_pos = faces[v_off + v_index(i, j + 1u, k, nx, ny)];
    let z_neg = faces[w_off + w_index(i, j, k, nx, ny)];
    let z_pos = faces[w_off + w_index(i, j, k + 1u, nx, ny)];

    let dx = mac_params.dx;
    var div = 0.0;
    if (dx > MAC_EPS) {
        div = ((x_pos - x_neg) + (y_pos - y_neg) + (z_pos - z_neg)) / dx;
    }

    divergence[cell] = div;
}
