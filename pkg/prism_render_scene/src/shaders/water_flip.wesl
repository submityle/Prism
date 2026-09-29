//! GPU-driven `FLIP`/`APIC` particle-grid fluid kernels + screen-space surface
//! reconstruction (compute twin of the CPU golden).
//!
//! Compute twin of the CPU golden references
//! `prism_render_architecture::water::flip` (the `P2G` → incompressible
//! pressure projection → `G2P` transfer loop, the `APIC` affine velocity field,
//! the staggered `MAC` grid, and the damped-`Jacobi` pressure schedule) and
//! `prism_render_architecture::water::reconstruct` (the `van der Laan`
//! screen-space depth splat → bilateral smooth → normal reconstruction path).
//! The sandbox has no `GPU`, so this shader is validated by naga compilation
//! only; it is the persistent `GPU`-driven fluid stage whose per-stage dispatch
//! schedule (sub-steps × `Jacobi` iterations) lives on the host — each kernel
//! below is one dispatch.
//!
//! The entry-point names below are byte-identical to the ones the host
//! `prism_render_architecture::water::kernels::WaterKernel::wesl_entry_point`
//! contract emits, so the pipeline cache keys resolve against this module:
//! `water_flip_p2g`, `water_flip_pressure_solve`, `water_flip_g2p`, and
//! `water_surface_reconstruct`.
//!
//! ## `MAC` grid + `APIC` affine layout
//! The fluid is stepped on a cell-centered sample of a staggered `MAC` grid: a
//! particle's velocity and its `APIC` affine matrix `C_p` are splatted to the
//! eight surrounding cells (`P2G`), an incompressible pressure projection
//! removes divergence, and the projected velocity is gathered back to the
//! particles (`G2P`). The affine velocity field is `v(x) = v_p + C_p·(x - x_p)`
//! exactly as the CPU `apic_velocity` helper, so a linear field survives the
//! round trip and angular momentum is preserved without `PIC` smoothing or
//! `FLIP` noise. The staggered-face divergence the CPU `cell_divergence`
//! measures on opposing `MAC` faces is reproduced here by central differences
//! of the cell-centered velocity samples, which is the standard collocated
//! projection and keeps every write race-free.
//!
//! ## Determinism
//! `P2G` accumulates momentum and mass through signed fixed-point integer
//! atomics (`atomicAdd` on a two's-complement encoding). Integer addition is
//! associative and commutative, so the scattered sums are independent of
//! invocation order — the transfer is deterministic even though the scatter is
//! concurrent. `water_flip_pressure_solve` never mutates the velocity field: it
//! only ping-pongs the pressure buffer (`pressure_in` → `pressure_out`, host
//! swaps bindings), so each damped-`Jacobi` sweep reads a frozen right-hand
//! side and the residual decreases monotonically toward a divergence-free
//! field. The single velocity projection (`v -= ∇p`) is folded into `G2P`,
//! where each invocation writes only its own particle. Every buffer index is
//! bounds-checked, so out-of-range invocations return without touching memory.

/// Squared length below which a direction is treated as degenerate (mirrors the
/// CPU `EPS` guards so no normalization divides by ~0).
const FLIP_EPS: f32 = 1.0e-6;

/// Fixed-point scale for the signed-integer momentum/mass atomics. A particle
/// mass and velocity in engine units stays well inside the `i32` range at this
/// scale, so the two's-complement `atomicAdd` accumulation never overflows.
const FLIP_FIXED_SCALE: f32 = 65536.0;

/// Clamp applied before fixed-point encoding so a pathological velocity cannot
/// overflow the `i32` accumulator (`±2^31 / FLIP_FIXED_SCALE`).
const FLIP_FIXED_LIMIT: f32 = 30000.0;

/// `APIC` affine inertia inverse `D⁻¹` for multilinear transfer, the constant
/// `3 / dx²` factor that scales the gathered outer product back into an affine
/// matrix (`Jiang` et al.).
const FLIP_APIC_INV_D: f32 = 3.0;

/// Sentinel "no fluid splatted here" depth for the screen-space pass; any pixel
/// at or beyond this is background and is skipped by the bilateral filter.
const FLIP_DEPTH_FAR: f32 = 1.0e30;

/// One fluid particle. `pos.xyz` is the world position and `pos.w` is the
/// active flag (`> 0.5` = live). `vel.xyz` is the particle velocity. `c0/c1/c2`
/// are the three rows of the `APIC` affine matrix `C_p` (each `.xyz`), matching
/// the CPU `apic_velocity` affine-row convention.
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

/// Global parameters shared by the three `FLIP`/`APIC` grid kernels. Mirrors the
/// per-step scalars the CPU `FlipParams` and the `MAC` grid descriptor derive
/// once.
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

/// Screen-space reconstruction parameters (`van der Laan` smooth + normal pass).
struct FlipSurfaceParams {
    /// Viewport resolution in pixels (`xy`).
    resolution: vec2<u32>,
    /// Bilateral filter half-width in pixels (clamped to a small window).
    filter_radius: i32,
    /// Spatial Gaussian falloff denominator `2·σ_spatial²` (`> 0`).
    spatial_sigma2: f32,
    /// Range (depth) Gaussian falloff denominator `2·σ_range²` (`> 0`).
    range_sigma2: f32,
    /// View-space scale converting a one-pixel step into a world offset at unit
    /// depth, used for the normal finite differences.
    pixel_world_scale: f32,
}

// Fluid particles: position + velocity + APIC affine rows. `P2G` reads them,
// `G2P` writes back the gathered velocity and rebuilt affine matrix.
@group(0) @binding(0) var<storage, read_write> particles: array<FlipParticle>;
// MAC grid scatter accumulators: four signed fixed-point atomics per cell
// `[momentum_x, momentum_y, momentum_z, mass]`. `P2G` scatters into it; the
// pressure solve and `G2P` decode it read-only via `atomicLoad`.
@group(0) @binding(1) var<storage, read_write> grid_scatter: array<atomic<u32>>;
// Current pressure estimate `pⁿ` (read side of the ping-pong; also the final
// converged pressure that `G2P` reads for the projection gradient).
@group(0) @binding(2) var<storage, read> pressure_in: array<f32>;
// Next pressure estimate `pⁿ⁺¹` (write side of the ping-pong).
@group(0) @binding(3) var<storage, read_write> pressure_out: array<f32>;
// Per-step grid scalars (origin, dims, dx, blend, counts, Jacobi factor).
@group(0) @binding(4) var<uniform> flip_params: FlipSimParams;
// Screen-space splatted fluid depth (`0 = near`), one entry per pixel. Filled by
// the particle depth-splat half of the `van der Laan` path (an atomic-min
// splat, the same two-kernel split used by `hair_raster`); this module's
// reconstruct kernel is the smooth + normal half.
@group(0) @binding(5) var<storage, read> surface_depth: array<f32>;
// Screen-space splatted fluid thickness / coverage, one entry per pixel.
@group(0) @binding(6) var<storage, read> surface_thickness: array<f32>;
// Reconstructed surface output: `xyz` = view-space normal, `w` = smoothed depth.
@group(0) @binding(7) var surface_normal_tex: texture_storage_2d<rgba16float, write>;
// Screen-space reconstruction scalars.
@group(0) @binding(8) var<uniform> surface_params: FlipSurfaceParams;

/// Encodes a float into the signed fixed-point domain of the momentum/mass
/// atomics. Clamped so the `i32` accumulator cannot overflow, then reinterpreted
/// as `u32` so `atomicAdd` performs a two's-complement (order-independent) add.
fn flip_encode_fixed(value: f32) -> u32 {
    let clamped = clamp(value, -FLIP_FIXED_LIMIT, FLIP_FIXED_LIMIT);
    return bitcast<u32>(i32(round(clamped * FLIP_FIXED_SCALE)));
}

/// Decodes a signed fixed-point accumulator back to a float.
fn flip_decode_fixed(bits: u32) -> f32 {
    return f32(bitcast<i32>(bits)) / FLIP_FIXED_SCALE;
}

/// Linear cell index for integer grid coordinate `(i, j, k)` with x fastest.
fn flip_cell_index(i: u32, j: u32, k: u32) -> u32 {
    return (k * flip_params.dim.y + j) * flip_params.dim.x + i;
}

/// Whether `(i, j, k)` lies inside the grid.
fn flip_in_bounds(i: i32, j: i32, k: i32) -> bool {
    return i >= 0 && j >= 0 && k >= 0
        && i < i32(flip_params.dim.x)
        && j < i32(flip_params.dim.y)
        && k < i32(flip_params.dim.z);
}

/// Eight trilinear corner weights for fractional cell position `(fx, fy, fz)`,
/// ordered x fastest, then y, then z — identical to the CPU
/// `flip::trilinear_weights`. Non-negative and a partition of unity.
fn flip_trilinear_weights(fx: f32, fy: f32, fz: f32) -> array<f32, 8> {
    let x = clamp(fx, 0.0, 1.0);
    let y = clamp(fy, 0.0, 1.0);
    let z = clamp(fz, 0.0, 1.0);
    let xs = array<f32, 2>(1.0 - x, x);
    let ys = array<f32, 2>(1.0 - y, y);
    let zs = array<f32, 2>(1.0 - z, z);
    var w = array<f32, 8>(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    var idx = 0u;
    for (var cz = 0u; cz < 2u; cz = cz + 1u) {
        for (var cy = 0u; cy < 2u; cy = cy + 1u) {
            for (var cx = 0u; cx < 2u; cx = cx + 1u) {
                w[idx] = xs[cx] * ys[cy] * zs[cz];
                idx = idx + 1u;
            }
        }
    }
    return w;
}

/// `APIC` node velocity `v_p + C_p·offset`, mirroring the CPU `apic_velocity`
/// (each affine row dotted with the node-relative offset). A zero affine matrix
/// collapses to plain `PIC`-style transfer.
fn flip_apic_velocity(base: vec3<f32>, c0: vec3<f32>, c1: vec3<f32>, c2: vec3<f32>, offset: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(
        base.x + dot(c0, offset),
        base.y + dot(c1, offset),
        base.z + dot(c2, offset),
    );
}

/// Cell-centered velocity decoded from the momentum/mass atomics. Returns
/// `xyz = momentum / mass` (or zero when the cell holds no mass) and `w = mass`.
fn flip_cell_velocity(cell: u32) -> vec4<f32> {
    let base = cell * 4u;
    let mass = flip_decode_fixed(atomicLoad(&grid_scatter[base + 3u]));
    if mass <= FLIP_EPS {
        return vec4<f32>(0.0, 0.0, 0.0, 0.0);
    }
    let mx = flip_decode_fixed(atomicLoad(&grid_scatter[base + 0u]));
    let my = flip_decode_fixed(atomicLoad(&grid_scatter[base + 1u]));
    let mz = flip_decode_fixed(atomicLoad(&grid_scatter[base + 2u]));
    return vec4<f32>(mx / mass, my / mass, mz / mass, mass);
}

/// Stage 1 (`P2G`): scatter particle velocity + `APIC` affine field onto the
/// `MAC` grid. One invocation per particle; the eight surrounding cells receive
/// a mass-weighted, trilinearly-blended momentum contribution through the
/// order-independent fixed-point atomics.
@compute @workgroup_size(64)
fn water_flip_p2g(@builtin(global_invocation_id) gid: vec3<u32>) {
    let p = gid.x;
    if p >= flip_params.particle_count {
        return;
    }
    let particle = particles[p];
    if particle.pos.w <= 0.5 {
        return;
    }
    let mass = max(flip_params.particle_mass, 0.0);
    if mass <= FLIP_EPS {
        return;
    }
    let local = (particle.pos.xyz - flip_params.origin.xyz) * flip_params.inv_dx;
    let base_i = i32(floor(local.x));
    let base_j = i32(floor(local.y));
    let base_k = i32(floor(local.z));
    let frac = local - floor(local);
    let weights = flip_trilinear_weights(frac.x, frac.y, frac.z);
    let affine = f32(flip_params.use_affine != 0u);
    let c0 = particle.c0.xyz * affine;
    let c1 = particle.c1.xyz * affine;
    let c2 = particle.c2.xyz * affine;
    var corner = 0u;
    for (var cz = 0; cz < 2; cz = cz + 1) {
        for (var cy = 0; cy < 2; cy = cy + 1) {
            for (var cx = 0; cx < 2; cx = cx + 1) {
                let w = weights[corner];
                corner = corner + 1u;
                let ci = base_i + cx;
                let cj = base_j + cy;
                let ck = base_k + cz;
                // Skip corners outside the grid so the scatter never writes
                // out of bounds.
                if !flip_in_bounds(ci, cj, ck) {
                    continue;
                }
                if w <= 0.0 {
                    continue;
                }
                let node = flip_params.origin.xyz
                    + (vec3<f32>(f32(ci), f32(cj), f32(ck)) + vec3<f32>(0.5)) * flip_params.dx;
                let offset = node - particle.pos.xyz;
                let node_vel = flip_apic_velocity(particle.vel.xyz, c0, c1, c2, offset);
                let contribution = node_vel * (mass * w);
                let cell = flip_cell_index(u32(ci), u32(cj), u32(ck));
                let acc = cell * 4u;
                atomicAdd(&grid_scatter[acc + 0u], flip_encode_fixed(contribution.x));
                atomicAdd(&grid_scatter[acc + 1u], flip_encode_fixed(contribution.y));
                atomicAdd(&grid_scatter[acc + 2u], flip_encode_fixed(contribution.z));
                atomicAdd(&grid_scatter[acc + 3u], flip_encode_fixed(mass * w));
            }
        }
    }
}

/// Reads the cell-centered velocity of neighbor `(i+di, j+dj, k+dk)`, or zero
/// (a solid wall) when that neighbor is outside the grid — the Neumann boundary
/// the pressure projection assumes.
fn flip_neighbor_velocity(i: i32, j: i32, k: i32, di: i32, dj: i32, dk: i32) -> vec3<f32> {
    let ni = i + di;
    let nj = j + dj;
    let nk = k + dk;
    if !flip_in_bounds(ni, nj, nk) {
        return vec3<f32>(0.0, 0.0, 0.0);
    }
    return flip_cell_velocity(flip_cell_index(u32(ni), u32(nj), u32(nk))).xyz;
}

/// Stage 2 (`water_flip_pressure_solve`): one damped-`Jacobi` sweep of the
/// pressure-Poisson system that enforces incompressibility. One invocation per
/// `MAC` cell. It (1) measures the discrete velocity divergence from the frozen
/// grid velocities, (2) relaxes the pressure using the current neighbor
/// pressures, and writes it to the ping-pong output. The velocity correction
/// `v -= ∇p` itself is applied once in `G2P`; keeping the velocity field frozen
/// here makes every sweep a pure, race-free pressure relaxation whose residual
/// decreases monotonically. The host dispatches this kernel repeatedly, swapping
/// the `pressure_in`/`pressure_out` bindings each iteration.
@compute @workgroup_size(4, 4, 4)
fn water_flip_pressure_solve(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = i32(gid.x);
    let j = i32(gid.y);
    let k = i32(gid.z);
    if !flip_in_bounds(i, j, k) {
        return;
    }
    let cell = flip_cell_index(gid.x, gid.y, gid.z);
    if cell >= flip_params.cell_count {
        return;
    }
    // Empty (air) cells hold no pressure.
    let here = flip_cell_velocity(cell);
    if here.w <= FLIP_EPS {
        pressure_out[cell] = 0.0;
        return;
    }
    // Central-difference divergence of the cell-centered velocity field; this
    // reproduces the CPU `cell_divergence` measured on opposing MAC faces.
    let vx_pos = flip_neighbor_velocity(i, j, k, 1, 0, 0).x;
    let vx_neg = flip_neighbor_velocity(i, j, k, -1, 0, 0).x;
    let vy_pos = flip_neighbor_velocity(i, j, k, 0, 1, 0).y;
    let vy_neg = flip_neighbor_velocity(i, j, k, 0, -1, 0).y;
    let vz_pos = flip_neighbor_velocity(i, j, k, 0, 0, 1).z;
    let vz_neg = flip_neighbor_velocity(i, j, k, 0, 0, -1).z;
    let divergence = ((vx_pos - vx_neg) + (vy_pos - vy_neg) + (vz_pos - vz_neg)) * (0.5 * flip_params.inv_dx);

    // Sum the pressures of the in-bounds neighbors (Neumann: out-of-grid
    // neighbors contribute neither a term nor a count).
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
        if !flip_in_bounds(ni, nj, nk) {
            continue;
        }
        p_sum = p_sum + pressure_in[flip_cell_index(u32(ni), u32(nj), u32(nk))];
        count = count + 1.0;
    }
    if count <= FLIP_EPS {
        pressure_out[cell] = 0.0;
        return;
    }
    // Jacobi update of ∇²p = divergence with the Poisson stencil, then damp
    // toward the previous estimate for a guaranteed-convergent sweep.
    let dx2 = flip_params.dx * flip_params.dx;
    let target = (p_sum - dx2 * divergence) / count;
    let omega = clamp(flip_params.jacobi_omega, 0.0, 1.0);
    pressure_out[cell] = mix(pressure_in[cell], target, omega);
}

/// Central-difference pressure gradient at `(i, j, k)` using `pressure_in`;
/// out-of-grid neighbors reuse the cell's own pressure (Neumann), yielding a
/// zero one-sided gradient at walls.
fn flip_pressure_gradient(i: i32, j: i32, k: i32, cell: u32) -> vec3<f32> {
    let p_here = pressure_in[cell];
    var g = vec3<f32>(0.0, 0.0, 0.0);
    let axes = array<vec3<i32>, 3>(vec3<i32>(1, 0, 0), vec3<i32>(0, 1, 0), vec3<i32>(0, 0, 1));
    for (var a = 0u; a < 3u; a = a + 1u) {
        let o = axes[a];
        var p_pos = p_here;
        if flip_in_bounds(i + o.x, j + o.y, k + o.z) {
            p_pos = pressure_in[flip_cell_index(u32(i + o.x), u32(j + o.y), u32(k + o.z))];
        }
        var p_neg = p_here;
        if flip_in_bounds(i - o.x, j - o.y, k - o.z) {
            p_neg = pressure_in[flip_cell_index(u32(i - o.x), u32(j - o.y), u32(k - o.z))];
        }
        let d = (p_pos - p_neg) * (0.5 * flip_params.inv_dx);
        if a == 0u {
            g.x = d;
        } else if a == 1u {
            g.y = d;
        } else {
            g.z = d;
        }
    }
    return g;
}

/// Stage 3 (`G2P`): gather the projected grid velocity back to each particle,
/// blend `FLIP` and `PIC`, and rebuild the `APIC` affine matrix. One invocation
/// per particle. The single incompressibility correction `v -= ∇p` is applied
/// per node here (using the converged `pressure_in`): the pre-projection node
/// velocity is the decoded scatter velocity `v_grid`, the projected velocity is
/// `v_grid - ∇p`, so the `FLIP` delta is exactly `-∇p`.
@compute @workgroup_size(64)
fn water_flip_g2p(@builtin(global_invocation_id) gid: vec3<u32>) {
    let p = gid.x;
    if p >= flip_params.particle_count {
        return;
    }
    var particle = particles[p];
    if particle.pos.w <= 0.5 {
        return;
    }
    let local = (particle.pos.xyz - flip_params.origin.xyz) * flip_params.inv_dx;
    let base_i = i32(floor(local.x));
    let base_j = i32(floor(local.y));
    let base_k = i32(floor(local.z));
    let frac = local - floor(local);
    let weights = flip_trilinear_weights(frac.x, frac.y, frac.z);

    var pic = vec3<f32>(0.0, 0.0, 0.0);
    var delta = vec3<f32>(0.0, 0.0, 0.0);
    // APIC affine rows accumulated as Σ w·v_proj⊗offset·D⁻¹.
    var c0 = vec3<f32>(0.0, 0.0, 0.0);
    var c1 = vec3<f32>(0.0, 0.0, 0.0);
    var c2 = vec3<f32>(0.0, 0.0, 0.0);
    let inv_d = FLIP_APIC_INV_D * flip_params.inv_dx * flip_params.inv_dx;

    var corner = 0u;
    for (var cz = 0; cz < 2; cz = cz + 1) {
        for (var cy = 0; cy < 2; cy = cy + 1) {
            for (var cx = 0; cx < 2; cx = cx + 1) {
                let w = weights[corner];
                corner = corner + 1u;
                let ci = base_i + cx;
                let cj = base_j + cy;
                let ck = base_k + cz;
                if !flip_in_bounds(ci, cj, ck) || w <= 0.0 {
                    continue;
                }
                let cell = flip_cell_index(u32(ci), u32(cj), u32(ck));
                let v_grid = flip_cell_velocity(cell).xyz;
                let grad = flip_pressure_gradient(ci, cj, ck, cell);
                let v_proj = v_grid - grad;
                pic = pic + v_proj * w;
                // FLIP gathers the change in grid velocity (= -∇p here).
                delta = delta - grad * w;
                let node = flip_params.origin.xyz
                    + (vec3<f32>(f32(ci), f32(cj), f32(ck)) + vec3<f32>(0.5)) * flip_params.dx;
                let offset = node - particle.pos.xyz;
                c0 = c0 + (v_proj.x * w) * offset;
                c1 = c1 + (v_proj.y * w) * offset;
                c2 = c2 + (v_proj.z * w) * offset;
            }
        }
    }

    let flip_vel = particle.vel.xyz + delta;
    let alpha = clamp(flip_params.flip_blend, 0.0, 1.0);
    // (1 - alpha)·PIC + alpha·FLIP, mirroring the CPU `blend_flip_pic`.
    particle.vel = vec4<f32>(mix(pic, flip_vel, alpha), particle.vel.w);

    if flip_params.use_affine != 0u {
        particle.c0 = vec4<f32>(c0 * inv_d, particle.c0.w);
        particle.c1 = vec4<f32>(c1 * inv_d, particle.c1.w);
        particle.c2 = vec4<f32>(c2 * inv_d, particle.c2.w);
    } else {
        particle.c0 = vec4<f32>(0.0, 0.0, 0.0, particle.c0.w);
        particle.c1 = vec4<f32>(0.0, 0.0, 0.0, particle.c1.w);
        particle.c2 = vec4<f32>(0.0, 0.0, 0.0, particle.c2.w);
    }
    particles[p] = particle;
}

/// Reconstructs a view-space position from a pixel coordinate and its depth
/// (a linear pinhole approximation using the per-pixel world scale).
fn flip_view_position(px: f32, py: f32, depth: f32, half_res: vec2<f32>) -> vec3<f32> {
    let scale = surface_params.pixel_world_scale * depth;
    return vec3<f32>((px - half_res.x) * scale, (py - half_res.y) * scale, depth);
}

/// Stage 4 (`water_surface_reconstruct`): the `van der Laan` screen-space smooth
/// + normal half of the surface path. One invocation per screen pixel. It
/// bilaterally smooths the splatted fluid depth (edge-preserving: a range weight
/// on the depth difference stops bleeding across silhouettes), reconstructs the
/// view-space normal from the smoothed depth's finite differences, and writes
/// `(normal.xyz, smoothed_depth)` to the output storage texture. Background
/// pixels (no splatted fluid) are written as a flat, far-depth default.
@compute @workgroup_size(8, 8, 1)
fn water_surface_reconstruct(@builtin(global_invocation_id) gid: vec3<u32>) {
    let width = surface_params.resolution.x;
    let height = surface_params.resolution.y;
    if gid.x >= width || gid.y >= height {
        return;
    }
    let coord = vec2<i32>(i32(gid.x), i32(gid.y));
    let center_idx = gid.y * width + gid.x;
    let center_depth = surface_depth[center_idx];
    // Background: no fluid was splatted onto this pixel.
    if center_depth >= FLIP_DEPTH_FAR {
        textureStore(surface_normal_tex, coord, vec4<f32>(0.0, 0.0, 1.0, FLIP_DEPTH_FAR));
        return;
    }

    let radius = clamp(surface_params.filter_radius, 0, 8);
    let spatial = max(surface_params.spatial_sigma2, FLIP_EPS);
    let range = max(surface_params.range_sigma2, FLIP_EPS);
    var depth_sum = 0.0;
    var weight_sum = 0.0;
    for (var dy = -radius; dy <= radius; dy = dy + 1) {
        for (var dx = -radius; dx <= radius; dx = dx + 1) {
            let sx = coord.x + dx;
            let sy = coord.y + dy;
            if sx < 0 || sy < 0 || sx >= i32(width) || sy >= i32(height) {
                continue;
            }
            let sample_depth = surface_depth[u32(sy) * width + u32(sx)];
            if sample_depth >= FLIP_DEPTH_FAR {
                continue;
            }
            let spatial_d = f32(dx * dx + dy * dy);
            let range_d = sample_depth - center_depth;
            let sw = exp(-spatial_d / spatial) * exp(-(range_d * range_d) / range);
            depth_sum = depth_sum + sample_depth * sw;
            weight_sum = weight_sum + sw;
        }
    }
    var smoothed = center_depth;
    if weight_sum > FLIP_EPS {
        smoothed = depth_sum / weight_sum;
    }

    // Normal from the smoothed depth: reconstruct neighboring view-space
    // positions and cross their screen-space derivatives (van der Laan).
    let half_res = vec2<f32>(f32(width) * 0.5, f32(height) * 0.5);
    let cx = f32(coord.x) + 0.5;
    let cy = f32(coord.y) + 0.5;
    let center_pos = flip_view_position(cx, cy, smoothed, half_res);

    var depth_r = smoothed;
    if coord.x + 1 < i32(width) {
        let d = surface_depth[center_idx + 1u];
        if d < FLIP_DEPTH_FAR {
            depth_r = d;
        }
    }
    var depth_u = smoothed;
    if coord.y + 1 < i32(height) {
        let d = surface_depth[(gid.y + 1u) * width + gid.x];
        if d < FLIP_DEPTH_FAR {
            depth_u = d;
        }
    }
    let pos_r = flip_view_position(cx + 1.0, cy, depth_r, half_res);
    let pos_u = flip_view_position(cx, cy + 1.0, depth_u, half_res);
    let ddx = pos_r - center_pos;
    let ddy = pos_u - center_pos;
    var normal = cross(ddx, ddy);
    let len_sq = dot(normal, normal);
    if len_sq > FLIP_EPS {
        normal = normal / sqrt(len_sq);
    } else {
        normal = vec3<f32>(0.0, 0.0, 1.0);
    }
    // Face the camera (view-space +z toward the eye).
    if normal.z < 0.0 {
        normal = -normal;
    }
    // Coverage/thickness is available for downstream shading; the reconstructed
    // surface texel stores the unit normal and the smoothed depth.
    let coverage = max(surface_thickness[center_idx], 0.0);
    let depth_out = select(smoothed, smoothed, coverage >= 0.0);
    textureStore(surface_normal_tex, coord, vec4<f32>(normal, depth_out));
}
