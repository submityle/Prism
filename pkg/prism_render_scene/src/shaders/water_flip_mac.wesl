//! Staggered-`MAC` incompressible pressure projection kernels — the
//! unconditionally-stable replacement for the collocated cell-centered
//! projection folded into `water_flip.wesl`.
//!
//! ## Why a separate staggered stage
//! The `water_flip.wesl` pressure stage samples one velocity vector per cell
//! centre and measures divergence with a *wide* central difference
//! `(v[i+1] - v[i-1])·0.5/dx` while relaxing pressure with a *compact* 7-point
//! `Laplacian`. Those two operators disagree: `div(grad p)` is not the same
//! stencil as the relaxation `Laplacian`, so `v - ∇p` is not a true orthogonal
//! projection and the discrete operator carries the classic odd/even
//! (checkerboard) null space. A divergence living in that null space is
//! invisible to the solve, so repeated projection injects energy — exactly the
//! instability the collocated golden documents.
//!
//! This module fixes that at the discretization level, the way `Houdini`,
//! `UE5` `Niagara`, and `Bridson`-style solvers do: velocities live on the six
//! `MAC` cell faces, divergence is the *compact* one-sided face difference
//! `(u[i+1] - u[i])/dx`, and the projection subtracts the *matching* compact
//! face gradient `(p[i] - p[i-1])/dx`. Then `div(grad)` is byte-for-byte the
//! same compact 7-point `Laplacian` the `Jacobi` sweep relaxes, the projection
//! is a genuine orthogonal removal of the divergent component, and the
//! checkerboard null space is gone — projection can only remove kinetic energy,
//! never inject it.
//!
//! ## Face storage layout
//! A single `faces` buffer packs the three staggered velocity components as
//! contiguous blocks `[u | v | w]` (each a plain `f32`, no atomics — the
//! projection is a gather, every face written by exactly one invocation):
//! * `u` (x-velocity) on x-faces: `(nx+1)·ny·nz`, index `(k·ny + j)·(nx+1) + i`,
//!   `i ∈ 0..=nx`;
//! * `v` (y-velocity) on y-faces: `nx·(ny+1)·nz`, index `(k·(ny+1) + j)·nx + i`,
//!   `j ∈ 0..=ny`;
//! * `w` (z-velocity) on z-faces: `nx·ny·(nz+1)`, index `(k·ny + j)·nx + i`,
//!   `k ∈ 0..=nz`.
//! Faces on the domain boundary (`i = 0`/`i = nx` for `u`, and likewise for
//! `v`/`w`) are solid walls: they are seeded to zero and never projected, giving
//! the homogeneous `Neumann` pressure boundary a full fluid tank assumes.
//!
//! ## Determinism & races
//! `mac_divergence` writes one scalar per cell, `mac_pressure` ping-pongs the
//! pressure buffer (host swaps `pressure_in`/`pressure_out` between sweeps) and
//! writes one scalar per cell, and `mac_project` writes one face per invocation
//! while reading only the frozen converged pressure. No cell or face is written
//! by two invocations, so every kernel is race-free and order-independent.

/// Squared/absolute epsilon guarding the `Neumann` neighbour count divide.
const MAC_EPS: f32 = 1.0e-6;

/// Per-projection grid scalars mirrored from the host `MacParams`.
struct MacParams {
    /// Grid resolution in cells along x, y, z (`w` = total cell count).
    dim: vec4<u32>,
    /// Reciprocal cell edge length `1 / dx`.
    inv_dx: f32,
    /// `MAC` cell edge length (`> 0`).
    dx: f32,
    /// Damped-`Jacobi` relaxation factor `ω` in `(0, 1]`.
    jacobi_omega: f32,
    /// Padding to a 16-byte uniform row.
    pad: f32,
}

// Packed staggered face velocities `[u | v | w]`. `mac_divergence` reads it,
// `mac_project` subtracts the compact pressure gradient in place.
@group(0) @binding(0) var<storage, read_write> faces: array<f32>;
// Per-cell velocity divergence (`mac_divergence` writes, `mac_pressure` reads).
@group(0) @binding(1) var<storage, read_write> divergence: array<f32>;
// Current pressure estimate `pⁿ` (read side of the ping-pong; the converged
// field `mac_project` reads for the gradient).
@group(0) @binding(2) var<storage, read> pressure_in: array<f32>;
// Next pressure estimate `pⁿ⁺¹` (write side of the ping-pong).
@group(0) @binding(3) var<storage, read_write> pressure_out: array<f32>;
// Per-projection grid scalars.
@group(0) @binding(4) var<uniform> mac_params: MacParams;

/// `u`-face element count `(nx+1)·ny·nz`.
fn mac_u_count() -> u32 {
    return (mac_params.dim.x + 1u) * mac_params.dim.y * mac_params.dim.z;
}

/// `v`-face element count `nx·(ny+1)·nz`.
fn mac_v_count() -> u32 {
    return mac_params.dim.x * (mac_params.dim.y + 1u) * mac_params.dim.z;
}

/// Linear cell index `(k·ny + j)·nx + i` (x fastest).
fn mac_cell_index(i: u32, j: u32, k: u32) -> u32 {
    return (k * mac_params.dim.y + j) * mac_params.dim.x + i;
}

/// Linear `u`-face index for `i ∈ 0..=nx`.
fn mac_u_index(i: u32, j: u32, k: u32) -> u32 {
    let nxp1 = mac_params.dim.x + 1u;
    return (k * mac_params.dim.y + j) * nxp1 + i;
}

/// Linear `v`-face index for `j ∈ 0..=ny` (offset past the `u` block).
fn mac_v_index(i: u32, j: u32, k: u32) -> u32 {
    let nyp1 = mac_params.dim.y + 1u;
    return mac_u_count() + (k * nyp1 + j) * mac_params.dim.x + i;
}

/// Linear `w`-face index for `k ∈ 0..=nz` (offset past the `u` + `v` blocks).
fn mac_w_index(i: u32, j: u32, k: u32) -> u32 {
    return mac_u_count() + mac_v_count()
        + (k * mac_params.dim.y + j) * mac_params.dim.x + i;
}

/// Whether cell `(i, j, k)` lies inside the grid.
fn mac_in_bounds(i: i32, j: i32, k: i32) -> bool {
    return i >= 0 && j >= 0 && k >= 0
        && i < i32(mac_params.dim.x)
        && j < i32(mac_params.dim.y)
        && k < i32(mac_params.dim.z);
}

/// Stage 1 (`mac_divergence`): compact one-sided face divergence per cell.
/// `div = ((u[i+1]-u[i]) + (v[j+1]-v[j]) + (w[k+1]-w[k])) / dx`. One invocation
/// per cell. This is the right-hand side the pressure solve drives to zero.
@compute @workgroup_size(4, 4, 4)
fn mac_divergence(@builtin(global_invocation_id) gid: vec3<u32>) {
    if gid.x >= mac_params.dim.x || gid.y >= mac_params.dim.y || gid.z >= mac_params.dim.z {
        return;
    }
    let i = gid.x;
    let j = gid.y;
    let k = gid.z;
    let du = faces[mac_u_index(i + 1u, j, k)] - faces[mac_u_index(i, j, k)];
    let dv = faces[mac_v_index(i, j + 1u, k)] - faces[mac_v_index(i, j, k)];
    let dw = faces[mac_w_index(i, j, k + 1u)] - faces[mac_w_index(i, j, k)];
    divergence[mac_cell_index(i, j, k)] = (du + dv + dw) * mac_params.inv_dx;
}

/// Stage 2 (`mac_pressure`): one damped-`Jacobi` sweep of `∇²p = div` with the
/// compact 7-point `Laplacian` and homogeneous `Neumann` walls (out-of-grid
/// neighbours drop from both the sum and the count). One invocation per cell.
/// The host swaps `pressure_in`/`pressure_out` between sweeps; the residual
/// decreases monotonically toward the divergence-free pressure.
@compute @workgroup_size(4, 4, 4)
fn mac_pressure(@builtin(global_invocation_id) gid: vec3<u32>) {
    if gid.x >= mac_params.dim.x || gid.y >= mac_params.dim.y || gid.z >= mac_params.dim.z {
        return;
    }
    let i = i32(gid.x);
    let j = i32(gid.y);
    let k = i32(gid.z);
    let cell = mac_cell_index(gid.x, gid.y, gid.z);
    var p_sum = 0.0;
    var count = 0.0;
    let offsets = array<vec3<i32>, 6>(
        vec3<i32>(1, 0, 0), vec3<i32>(-1, 0, 0),
        vec3<i32>(0, 1, 0), vec3<i32>(0, -1, 0),
        vec3<i32>(0, 0, 1), vec3<i32>(0, 0, -1),
    );
    for (var n = 0u; n < 6u; n = n + 1u) {
        let o = offsets[n];
        let ni = i + o.x;
        let nj = j + o.y;
        let nk = k + o.z;
        if !mac_in_bounds(ni, nj, nk) {
            continue;
        }
        p_sum = p_sum + pressure_in[mac_cell_index(u32(ni), u32(nj), u32(nk))];
        count = count + 1.0;
    }
    if count <= MAC_EPS {
        pressure_out[cell] = 0.0;
        return;
    }
    let dx2 = mac_params.dx * mac_params.dx;
    let relaxed = (p_sum - dx2 * divergence[cell]) / count;
    let omega = clamp(mac_params.jacobi_omega, 0.0, 1.0);
    pressure_out[cell] = mix(pressure_in[cell], relaxed, omega);
}

/// Stage 3 (`mac_project`): subtract the compact face pressure gradient
/// `(p[cell⁺] - p[cell⁻]) / dx` from each interior face. One invocation per
/// face (flat index over the `[u | v | w]` blocks). Boundary faces are solid
/// walls and keep their seeded zero. Because this gradient is the exact adjoint
/// of the `mac_divergence` operator, the post-projection field is
/// (`Jacobi`-tolerance) divergence-free with no checkerboard residue.
@compute @workgroup_size(64)
fn mac_project(@builtin(global_invocation_id) gid: vec3<u32>) {
    let f = gid.x;
    let u_count = mac_u_count();
    let v_count = mac_v_count();
    let w_count = mac_params.dim.x * mac_params.dim.y * (mac_params.dim.z + 1u);
    if f >= u_count + v_count + w_count {
        return;
    }
    if f < u_count {
        let nxp1 = mac_params.dim.x + 1u;
        let i = f % nxp1;
        let rem = f / nxp1;
        let j = rem % mac_params.dim.y;
        let k = rem / mac_params.dim.y;
        // Solid domain walls: no correction.
        if i == 0u || i == mac_params.dim.x {
            return;
        }
        let grad = (pressure_in[mac_cell_index(i, j, k)]
            - pressure_in[mac_cell_index(i - 1u, j, k)]) * mac_params.inv_dx;
        faces[f] = faces[f] - grad;
    } else if f < u_count + v_count {
        let g = f - u_count;
        let nyp1 = mac_params.dim.y + 1u;
        let i = g % mac_params.dim.x;
        let rem = g / mac_params.dim.x;
        let j = rem % nyp1;
        let k = rem / nyp1;
        if j == 0u || j == mac_params.dim.y {
            return;
        }
        let grad = (pressure_in[mac_cell_index(i, j, k)]
            - pressure_in[mac_cell_index(i, j - 1u, k)]) * mac_params.inv_dx;
        faces[f] = faces[f] - grad;
    } else {
        let g = f - u_count - v_count;
        let i = g % mac_params.dim.x;
        let rem = g / mac_params.dim.x;
        let j = rem % mac_params.dim.y;
        let k = rem / mac_params.dim.y;
        if k == 0u || k == mac_params.dim.z {
            return;
        }
        let grad = (pressure_in[mac_cell_index(i, j, k)]
            - pressure_in[mac_cell_index(i, j, k - 1u)]) * mac_params.inv_dx;
        faces[f] = faces[f] - grad;
    }
}
