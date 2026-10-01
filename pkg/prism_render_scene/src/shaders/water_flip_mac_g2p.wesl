//! Face-centered (staggered `MAC`) `G2P` transfer — the gather half that closes
//! the `FLIP`/`APIC` loop against the unconditionally-stable staggered pressure
//! projection in `water_flip_mac.wesl`.
//!
//! ## Why a face-centered gather
//! The matching scatter (`water_flip_mac_p2g.wesl`) splats each particle's
//! velocity *component* onto the cell face it is normal to, the projection in
//! `water_flip_mac.wesl` removes the divergent part of those face velocities
//! with a compact one-sided gradient, and this kernel gathers the projected
//! face field back to the particles. Storing each component on its own
//! staggered lattice (as `Houdini`, `UE5` `Niagara`, and `Bridson`-style solvers
//! do) is what lets the compact divergence/gradient share the exact 7-point
//! `Laplacian` the `Jacobi` sweep relaxes, so the gather here samples each
//! component from its own face family with the matching half-cell shift.
//!
//! ## `FLIP`/`PIC` blend and the `FLIP` delta
//! Classic `FLIP` carries the *change* the grid applied this step, not the grid
//! velocity itself. On the staggered grid that change is read directly as
//! `projected_face − preprojection_face` for every sampled face — no pressure
//! gradient is re-derived on the particle side, so the delta is exact even where
//! the one-sided boundary stencil clamps the gradient. `PIC` gathers the
//! projected face velocity. The two are mixed as `(1 − α)·PIC + α·FLIP` with the
//! same clamp as the CPU `blend_flip_pic`, so `α = 0` is pure dissipative `PIC`
//! and `α = 1` is pure `FLIP`.
//!
//! ## `APIC` affine reconstruction
//! When the affine field is carried, each affine row is rebuilt from the
//! projected face velocities of its own component family,
//! `C_rowₐ = (3 / dx²) · Σ w · v_proj · (xface − xp)`, exactly mirroring the CPU
//! `apic_velocity` inverse so a linear velocity field is reproduced bit-for-bit.
//!
//! ## Face storage layout (identical to `water_flip_mac.wesl` / `_p2g`)
//! * `u` (x-velocity) on x-faces: `(nx+1)·ny·nz`, index `(k·ny + j)·(nx+1) + i`,
//!   `i ∈ 0..=nx`, face centre `origin + (i, j+0.5, k+0.5)·dx`;
//! * `v` (y-velocity) on y-faces: `nx·(ny+1)·nz`, index
//!   `u_count + (k·(ny+1) + j)·nx + i`, `j ∈ 0..=ny`,
//!   face centre `origin + (i+0.5, j, k+0.5)·dx`;
//! * `w` (z-velocity) on z-faces: `nx·ny·(nz+1)`, index
//!   `u_count + v_count + (k·ny + j)·nx + i`, `k ∈ 0..=nz`,
//!   face centre `origin + (i+0.5, j+0.5, k)·dx`.
//!
//! ## Determinism & races
//! `water_flip_mac_g2p` runs one invocation per particle and only ever *reads*
//! the two face buffers while *writing* its own particle slot, so the gather is
//! race-free and order-independent. Every sampled face index is bounds-checked
//! against its staggered lattice, so corners outside the domain are skipped and
//! never read out of range.

/// Absolute epsilon guarding degenerate comparisons (mirrors the CPU `EPS`).
const FLIP_EPS: f32 = 1.0e-6;

/// `APIC` affine inertia inverse `D⁻¹ = 3 / dx²` for multilinear transfer,
/// identical to `water_flip.wesl`.
const FLIP_APIC_INV_D: f32 = 3.0;

/// One fluid particle. Byte-identical to `FlipParticle` in `water_flip.wesl`.
struct FlipParticle {
    /// World position (`xyz`) and active flag (`w`).
    pos: vec4<f32>,
    /// Particle velocity (`xyz`).
    vel: vec4<f32>,
    /// `APIC` affine matrix row 0 (`xyz`).
    c0: vec4<f32>,
    /// `APIC` affine matrix row 1 (`xyz`).
    c1: vec4<f32>,
    /// `APIC` affine matrix row 2 (`xyz`).
    c2: vec4<f32>,
}

/// Global parameters, byte-identical to `FlipSimParams` in `water_flip.wesl`.
struct FlipSimParams {
    /// World-space origin of grid cell `(0, 0, 0)`'s corner (`xyz`).
    origin: vec4<f32>,
    /// Grid resolution in cells along x, y, z (`w` unused).
    dim: vec4<u32>,
    /// `MAC` cell edge length (`> 0`).
    dx: f32,
    /// Reciprocal cell edge length `1 / dx`.
    inv_dx: f32,
    /// `FLIP`/`PIC` blend in `0..=1` (`0` = `PIC`, `1` = `FLIP`).
    flip_blend: f32,
    /// Per-particle mass (unused by the gather, kept for layout parity).
    particle_mass: f32,
    /// Damped-`Jacobi` relaxation factor `ω` (unused here, layout parity).
    jacobi_omega: f32,
    /// `1` if the `APIC` affine field is carried, `0` for plain `PIC`/`FLIP`.
    use_affine: u32,
    /// Number of live particles bounding the particle dispatch.
    particle_count: u32,
    /// Total grid cell count (unused here, layout parity).
    cell_count: u32,
}

// Live particle pool (`water_flip_mac_g2p` reads and writes back).
@group(0) @binding(0) var<storage, read_write> particles: array<FlipParticle>;
// Projected staggered face velocities `[u | v | w]` (post `mac_project`).
@group(0) @binding(1) var<storage, read> faces_projected: array<f32>;
// Pre-projection staggered face velocities `[u | v | w]` (post `faces_normalize`,
// pre `mac_project`). The `FLIP` delta is `faces_projected − faces_preprojection`.
@group(0) @binding(2) var<storage, read> faces_preprojection: array<f32>;
// Simulation parameters.
@group(0) @binding(3) var<uniform> sim_params: FlipSimParams;

/// Result of gathering one velocity component from its staggered face family.
struct AxisGather {
    /// `PIC` component — the mass-weighted projected face velocity.
    pic: f32,
    /// `FLIP` delta component — the mass-weighted change the projection applied.
    delta: f32,
    /// `APIC` affine row contribution `Σ w · v_proj · (xface − xp)` (pre `D⁻¹`).
    row: vec3<f32>,
}

/// `u`-face element count `(nx+1)·ny·nz`.
fn face_u_count() -> u32 {
    return (sim_params.dim.x + 1u) * sim_params.dim.y * sim_params.dim.z;
}

/// `v`-face element count `nx·(ny+1)·nz`.
fn face_v_count() -> u32 {
    return sim_params.dim.x * (sim_params.dim.y + 1u) * sim_params.dim.z;
}

/// Flat face index for `axis` (0 = `u`, 1 = `v`, 2 = `w`) at face coordinate
/// `(i, j, k)`, matching the `mac_*_index` helpers in `water_flip_mac.wesl`.
fn face_flat_index(axis: u32, i: u32, j: u32, k: u32) -> u32 {
    let nx = sim_params.dim.x;
    let ny = sim_params.dim.y;
    if axis == 0u {
        return (k * ny + j) * (nx + 1u) + i;
    } else if axis == 1u {
        return face_u_count() + (k * (ny + 1u) + j) * nx + i;
    }
    return face_u_count() + face_v_count() + (k * ny + j) * nx + i;
}

/// Eight trilinear corner weights for fractional position `(fx, fy, fz)`,
/// ordered x fastest, then y, then z — identical to the CPU
/// `flip::trilinear_weights`. Non-negative and a partition of unity.
fn face_trilinear_weights(fx: f32, fy: f32, fz: f32) -> array<f32, 8> {
    let x = clamp(fx, 0.0, 1.0);
    let y = clamp(fy, 0.0, 1.0);
    let z = clamp(fz, 0.0, 1.0);
    let xs = array<f32, 2>(1.0 - x, x);
    let ys = array<f32, 2>(1.0 - y, y);
    let zs = array<f32, 2>(1.0 - z, z);
    var wts = array<f32, 8>(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    var idx = 0u;
    for (var cz = 0u; cz < 2u; cz = cz + 1u) {
        for (var cy = 0u; cy < 2u; cy = cy + 1u) {
            for (var cx = 0u; cx < 2u; cx = cx + 1u) {
                wts[idx] = xs[cx] * ys[cy] * zs[cz];
                idx = idx + 1u;
            }
        }
    }
    return wts;
}

/// Gathers one velocity component from its staggered face family. `axis` selects
/// the face family (0 = `u`/x, 1 = `v`/y, 2 = `w`/z); `shift` is the half-cell
/// offset of that face family's node lattice; `node_dim` is that lattice's
/// extent. Returns the `PIC` component, the `FLIP` delta component, and the
/// `APIC` affine row (before the `D⁻¹` scale).
fn gather_axis(axis: u32, shift: vec3<f32>, node_dim: vec3<u32>, pos: vec3<f32>) -> AxisGather {
    var result: AxisGather;
    result.pic = 0.0;
    result.delta = 0.0;
    result.row = vec3<f32>(0.0, 0.0, 0.0);
    let local = (pos - sim_params.origin.xyz) * sim_params.inv_dx - shift;
    let base_i = i32(floor(local.x));
    let base_j = i32(floor(local.y));
    let base_k = i32(floor(local.z));
    let frac = local - floor(local);
    let weights = face_trilinear_weights(frac.x, frac.y, frac.z);
    var corner = 0u;
    for (var cz = 0; cz < 2; cz = cz + 1) {
        for (var cy = 0; cy < 2; cy = cy + 1) {
            for (var cx = 0; cx < 2; cx = cx + 1) {
                let w = weights[corner];
                corner = corner + 1u;
                let fi = base_i + cx;
                let fj = base_j + cy;
                let fk = base_k + cz;
                // Skip faces outside this staggered lattice.
                if fi < 0 || fj < 0 || fk < 0
                    || fi >= i32(node_dim.x)
                    || fj >= i32(node_dim.y)
                    || fk >= i32(node_dim.z) {
                    continue;
                }
                if w <= 0.0 {
                    continue;
                }
                let flat = face_flat_index(axis, u32(fi), u32(fj), u32(fk));
                let v_proj = faces_projected[flat];
                let v_pre = faces_preprojection[flat];
                result.pic = result.pic + v_proj * w;
                result.delta = result.delta + (v_proj - v_pre) * w;
                let node = sim_params.origin.xyz
                    + (vec3<f32>(f32(fi), f32(fj), f32(fk)) + shift) * sim_params.dx;
                let offset = node - pos;
                result.row = result.row + offset * (v_proj * w);
            }
        }
    }
    return result;
}

/// Stage 3 (face-centered `G2P`): gather the projected staggered velocity back
/// to each particle, blend `FLIP` and `PIC`, and rebuild the `APIC` affine
/// matrix. One invocation per particle. The `FLIP` delta is read directly as the
/// per-face change `projected − preprojection`, so no pressure gradient is
/// recomputed on the particle side.
@compute @workgroup_size(64)
fn water_flip_mac_g2p(@builtin(global_invocation_id) gid: vec3<u32>) {
    let p = gid.x;
    if p >= sim_params.particle_count {
        return;
    }
    var particle = particles[p];
    if particle.pos.w <= 0.5 {
        return;
    }
    let pos = particle.pos.xyz;
    let nx = sim_params.dim.x;
    let ny = sim_params.dim.y;
    let nz = sim_params.dim.z;
    // x-velocity ← u-faces (nodes at `(i, j+0.5, k+0.5)`, `i ∈ 0..=nx`).
    let gu = gather_axis(0u, vec3<f32>(0.0, 0.5, 0.5), vec3<u32>(nx + 1u, ny, nz), pos);
    // y-velocity ← v-faces (nodes at `(i+0.5, j, k+0.5)`, `j ∈ 0..=ny`).
    let gv = gather_axis(1u, vec3<f32>(0.5, 0.0, 0.5), vec3<u32>(nx, ny + 1u, nz), pos);
    // z-velocity ← w-faces (nodes at `(i+0.5, j+0.5, k)`, `k ∈ 0..=nz`).
    let gw = gather_axis(2u, vec3<f32>(0.5, 0.5, 0.0), vec3<u32>(nx, ny, nz + 1u), pos);

    let pic = vec3<f32>(gu.pic, gv.pic, gw.pic);
    let delta = vec3<f32>(gu.delta, gv.delta, gw.delta);
    let flip_vel = particle.vel.xyz + delta;
    let alpha = clamp(sim_params.flip_blend, 0.0, 1.0);
    // (1 - alpha)·PIC + alpha·FLIP, mirroring the CPU `blend_flip_pic`.
    particle.vel = vec4<f32>(mix(pic, flip_vel, alpha), particle.vel.w);

    if sim_params.use_affine != 0u {
        let inv_d = FLIP_APIC_INV_D * sim_params.inv_dx * sim_params.inv_dx;
        particle.c0 = vec4<f32>(gu.row * inv_d, particle.c0.w);
        particle.c1 = vec4<f32>(gv.row * inv_d, particle.c1.w);
        particle.c2 = vec4<f32>(gw.row * inv_d, particle.c2.w);
    } else {
        particle.c0 = vec4<f32>(0.0, 0.0, 0.0, particle.c0.w);
        particle.c1 = vec4<f32>(0.0, 0.0, 0.0, particle.c1.w);
        particle.c2 = vec4<f32>(0.0, 0.0, 0.0, particle.c2.w);
    }
    particles[p] = particle;
}
