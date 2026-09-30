//! Face-centered (staggered `MAC`) `P2G` transfer — the missing bridge from the
//! `FLIP`/`APIC` particle pool to the unconditionally-stable staggered pressure
//! projection in `water_flip_mac.wesl`.
//!
//! ## Why a face-centered scatter
//! The legacy `water_flip.wesl` `P2G` scatters one velocity *vector* to each
//! cell centre, so the projection that follows must measure divergence with a
//! wide central difference and carries the classic odd/even (checkerboard) null
//! space — repeated projection injects energy (documented by the collocated
//! golden). `Houdini`, `UE5` `Niagara`, and `Bridson`-style solvers instead
//! store each velocity *component* on the cell face it is normal to, so the
//! compact one-sided divergence and the matching compact face gradient share the
//! exact 7-point `Laplacian` the `Jacobi` sweep relaxes and the projection is a
//! genuine orthogonal removal of the divergent component.
//!
//! This module performs the transfer half of that scheme: it splats each
//! particle's velocity component (with the `APIC` affine field) onto the three
//! staggered face grids, accumulating momentum and mass through order-independent
//! signed fixed-point atomics, then normalizes momentum/mass into the plain-`f32`
//! `[u | v | w]` `faces` buffer that `water_flip_mac.wesl`'s `mac_divergence`
//! reads. `G2P` from the projected faces is a separate stage.
//!
//! ## Face storage layout (identical to `water_flip_mac.wesl`)
//! * `u` (x-velocity) on x-faces: `(nx+1)·ny·nz`, index `(k·ny + j)·(nx+1) + i`,
//!   `i ∈ 0..=nx`, face centre `origin + (i, j+0.5, k+0.5)·dx`;
//! * `v` (y-velocity) on y-faces: `nx·(ny+1)·nz`, index
//!   `u_count + (k·(ny+1) + j)·nx + i`, `j ∈ 0..=ny`,
//!   face centre `origin + (i+0.5, j, k+0.5)·dx`;
//! * `w` (z-velocity) on z-faces: `nx·ny·(nz+1)`, index
//!   `u_count + v_count + (k·ny + j)·nx + i`, `k ∈ 0..=nz`,
//!   face centre `origin + (i+0.5, j+0.5, k)·dx`.
//! The scatter accumulator packs two signed fixed-point atomics per face
//! `[momentum, mass]` at `face_flat·2 + {0, 1}`, over the same `[u | v | w]`
//! block order.
//!
//! ## Determinism & races
//! `water_flip_mac_p2g` accumulates momentum and mass through two's-complement
//! integer `atomicAdd`, which is associative and commutative, so the scattered
//! sums are independent of invocation order even though the scatter is
//! concurrent. `water_flip_mac_faces_normalize` writes exactly one face per
//! invocation, reading only that face's two accumulators, so it is race-free.
//! Every face index is bounds-checked, so out-of-range corners are skipped and
//! never touch memory.

/// Squared/absolute epsilon guarding the mass divide (mirrors the CPU `EPS`).
const FLIP_EPS: f32 = 1.0e-6;

/// Fixed-point scale for the signed-integer momentum/mass atomics, identical to
/// `water_flip.wesl` so the two scatters share one encode/decode convention.
const FLIP_FIXED_SCALE: f32 = 65536.0;

/// Clamp applied before fixed-point encoding so a pathological velocity cannot
/// overflow the `i32` accumulator (`±2^31 / FLIP_FIXED_SCALE`).
const FLIP_FIXED_LIMIT: f32 = 30000.0;

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
    /// Per-particle mass used to weight the `P2G` scatter (`> 0`).
    particle_mass: f32,
    /// Damped-`Jacobi` relaxation factor `ω` in `(0, 1]`.
    jacobi_omega: f32,
    /// `1` if the `APIC` affine field is carried, `0` for plain `PIC`/`FLIP`.
    use_affine: u32,
    /// Number of live particles bounding the particle dispatches.
    particle_count: u32,
    /// Total grid cell count bounding the pressure dispatch.
    cell_count: u32,
}

// Live particle pool (`water_flip_mac_p2g` reads).
@group(0) @binding(0) var<storage, read> particles: array<FlipParticle>;
// Packed face scatter accumulator: `[momentum, mass]` signed fixed-point atomics
// per face, over the `[u | v | w]` block order.
@group(0) @binding(1) var<storage, read_write> face_scatter: array<atomic<u32>>;
// Packed staggered face velocities `[u | v | w]` (`water_flip_mac_faces_normalize`
// writes; `water_flip_mac.wesl`'s `mac_divergence` reads).
@group(0) @binding(2) var<storage, read_write> faces: array<f32>;
// Per-step grid scalars.
@group(0) @binding(3) var<uniform> sim_params: FlipSimParams;

/// Clamps and encodes a float as a signed fixed-point accumulator bit pattern.
fn flip_encode_fixed(value: f32) -> u32 {
    let clamped = clamp(value, -FLIP_FIXED_LIMIT, FLIP_FIXED_LIMIT);
    return bitcast<u32>(i32(round(clamped * FLIP_FIXED_SCALE)));
}

/// Decodes a signed fixed-point accumulator back to a float.
fn flip_decode_fixed(bits: u32) -> f32 {
    return f32(bitcast<i32>(bits)) / FLIP_FIXED_SCALE;
}

/// `u`-face element count `(nx+1)·ny·nz`.
fn face_u_count() -> u32 {
    return (sim_params.dim.x + 1u) * sim_params.dim.y * sim_params.dim.z;
}

/// `v`-face element count `nx·(ny+1)·nz`.
fn face_v_count() -> u32 {
    return sim_params.dim.x * (sim_params.dim.y + 1u) * sim_params.dim.z;
}

/// `w`-face element count `nx·ny·(nz+1)`.
fn face_w_count() -> u32 {
    return sim_params.dim.x * sim_params.dim.y * (sim_params.dim.z + 1u);
}

/// Total packed face count over the `[u | v | w]` blocks.
fn face_total_count() -> u32 {
    return face_u_count() + face_v_count() + face_w_count();
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

/// Scatters one velocity component onto its staggered face grid. `axis` selects
/// the face family (0 = `u`/x, 1 = `v`/y, 2 = `w`/z); `shift` is the half-cell
/// offset of that face family's node lattice; `comp_base` is the particle's
/// velocity component and `comp_row` the matching `APIC` affine row.
fn scatter_axis(
    axis: u32,
    shift: vec3<f32>,
    node_dim: vec3<u32>,
    pos: vec3<f32>,
    comp_base: f32,
    comp_row: vec3<f32>,
    mass: f32,
) {
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
                let node = sim_params.origin.xyz
                    + (vec3<f32>(f32(fi), f32(fj), f32(fk)) + shift) * sim_params.dx;
                let offset = node - pos;
                // `APIC` node velocity component `v_c + C_row·offset`.
                let node_comp = comp_base + dot(comp_row, offset);
                let momentum = node_comp * (mass * w);
                let flat = face_flat_index(axis, u32(fi), u32(fj), u32(fk));
                let slot = flat * 2u;
                atomicAdd(&face_scatter[slot + 0u], flip_encode_fixed(momentum));
                atomicAdd(&face_scatter[slot + 1u], flip_encode_fixed(mass * w));
            }
        }
    }
}

/// Stage 1 (face-centered `P2G`): scatter each particle's velocity component +
/// `APIC` affine field onto the three staggered face grids. One invocation per
/// particle; the surrounding faces receive a mass-weighted, trilinearly-blended
/// momentum contribution through the order-independent fixed-point atomics.
@compute @workgroup_size(64)
fn water_flip_mac_p2g(@builtin(global_invocation_id) gid: vec3<u32>) {
    let p = gid.x;
    if p >= sim_params.particle_count {
        return;
    }
    let particle = particles[p];
    if particle.pos.w <= 0.5 {
        return;
    }
    let mass = max(sim_params.particle_mass, 0.0);
    if mass <= FLIP_EPS {
        return;
    }
    let affine = f32(sim_params.use_affine != 0u);
    let c0 = particle.c0.xyz * affine;
    let c1 = particle.c1.xyz * affine;
    let c2 = particle.c2.xyz * affine;
    let vel = particle.vel.xyz;
    let pos = particle.pos.xyz;
    let nx = sim_params.dim.x;
    let ny = sim_params.dim.y;
    let nz = sim_params.dim.z;
    // x-velocity → u-faces (nodes at `(i, j+0.5, k+0.5)`, `i ∈ 0..=nx`).
    scatter_axis(0u, vec3<f32>(0.0, 0.5, 0.5), vec3<u32>(nx + 1u, ny, nz), pos, vel.x, c0, mass);
    // y-velocity → v-faces (nodes at `(i+0.5, j, k+0.5)`, `j ∈ 0..=ny`).
    scatter_axis(1u, vec3<f32>(0.5, 0.0, 0.5), vec3<u32>(nx, ny + 1u, nz), pos, vel.y, c1, mass);
    // z-velocity → w-faces (nodes at `(i+0.5, j+0.5, k)`, `k ∈ 0..=nz`).
    scatter_axis(2u, vec3<f32>(0.5, 0.5, 0.0), vec3<u32>(nx, ny, nz + 1u), pos, vel.z, c2, mass);
}

/// Stage 2 (`faces_normalize`): divide each face's accumulated momentum by its
/// accumulated mass to produce the plain-`f32` staggered velocity the
/// `water_flip_mac.wesl` projection consumes. One invocation per face. Faces
/// that received no mass (including all solid domain-boundary faces) are seeded
/// to zero, giving the homogeneous `Neumann` wall the projection assumes.
@compute @workgroup_size(64)
fn water_flip_mac_faces_normalize(@builtin(global_invocation_id) gid: vec3<u32>) {
    let f = gid.x;
    if f >= face_total_count() {
        return;
    }
    let slot = f * 2u;
    let mass = flip_decode_fixed(atomicLoad(&face_scatter[slot + 1u]));
    if mass <= FLIP_EPS {
        faces[f] = 0.0;
        return;
    }
    let momentum = flip_decode_fixed(atomicLoad(&face_scatter[slot + 0u]));
    faces[f] = momentum / mass;
}
