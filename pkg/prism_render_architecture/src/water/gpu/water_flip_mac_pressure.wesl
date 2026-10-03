// water_flip_mac_pressure.wesl
//
// FLIP/APIC MAC pressure-projection stage 2: one damped-Jacobi relaxation
// sweep of the pressure Poisson system A p = b.
//
// The incompressible projection solves the negative-Laplacian pressure
// equation whose right-hand side b is the per-cell divergence from stage 1
// (mac_divergence). This kernel performs a single out-of-place damped-Jacobi
// sweep, exactly mirroring the crate's golden pressure_multigrid smoother:
//
//   inv_h2 = 1 / (h*h)
//   ap   = (6 p_c - p_xm - p_xp - p_ym - p_yp - p_zm - p_zp) * inv_h2   (A p)
//   r    = b_c - ap                                                     (residual)
//   p'   = p_c + (omega * h * h / 6) * r                                (update)
//
// over an n^3 node grid in x-fastest order idx = (z*n + y)*n + x. Boundary
// nodes form a fixed Dirichlet layer: they are copied through unchanged so the
// ping-pong preserves the boundary condition across sweeps, matching the golden
// smoother that only updates interior nodes 1..n-1. A degenerate grid (n < 3)
// copies the field through untouched.
//
// ABI (WaterKernel::FlipMacPressure shared descriptor: 4 storage + 1 uniform,
// @workgroup_size(4, 4, 4), Grid3d). The divergence and Jacobi pressure passes
// share the bind group: faces (unused here), divergence (read as b),
// pressure_in / pressure_out ping-pong, plus mac_params. Only + - * / appear.

struct MacPressureParams {
    grid_n: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
    h: f32,
    omega: f32,
    pad3: f32,
    pad4: f32,
};

@group(0) @binding(0) var<storage, read> faces: array<f32>;
@group(0) @binding(1) var<storage, read> divergence: array<f32>;
@group(0) @binding(2) var<storage, read> pressure_in: array<f32>;
@group(0) @binding(3) var<storage, read_write> pressure_out: array<f32>;
@group(0) @binding(4) var<uniform> mac_params: MacPressureParams;

fn node_index(x: u32, y: u32, z: u32, n: u32) -> u32 {
    return (z * n + y) * n + x;
}

@compute @workgroup_size(4, 4, 4)
fn mac_pressure(@builtin(global_invocation_id) gid: vec3<u32>) {
    let n = mac_params.grid_n;

    let x = gid.x;
    let y = gid.y;
    let z = gid.z;

    // Skip invocations outside the node domain (Grid3d brick padding).
    if (x >= n || y >= n || z >= n) {
        return;
    }

    let c = node_index(x, y, z, n);

    // Degenerate grid: nothing to relax, copy through.
    if (n < 3u) {
        pressure_out[c] = pressure_in[c];
        return;
    }

    // Fixed Dirichlet boundary layer: copy through unchanged.
    if (x == 0u || y == 0u || z == 0u || x == n - 1u || y == n - 1u || z == n - 1u) {
        pressure_out[c] = pressure_in[c];
        return;
    }

    let h = mac_params.h;
    let omega = mac_params.omega;
    let inv_h2 = 1.0 / (h * h);

    let p_c = pressure_in[c];
    let p_xm = pressure_in[node_index(x - 1u, y, z, n)];
    let p_xp = pressure_in[node_index(x + 1u, y, z, n)];
    let p_ym = pressure_in[node_index(x, y - 1u, z, n)];
    let p_yp = pressure_in[node_index(x, y + 1u, z, n)];
    let p_zm = pressure_in[node_index(x, y, z - 1u, n)];
    let p_zp = pressure_in[node_index(x, y, z + 1u, n)];

    let s = 6.0 * p_c - p_xm - p_xp - p_ym - p_yp - p_zm - p_zp;
    let ap = s * inv_h2;
    let r = divergence[c] - ap;
    let factor = omega * h * h / 6.0;

    pressure_out[c] = p_c + factor * r;
}
