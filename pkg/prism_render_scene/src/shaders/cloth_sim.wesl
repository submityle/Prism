//! GPU-driven XPBD cloth solver kernels (compute twin of the CPU golden).
//!
//! Compute twin of the CPU golden reference
//! `prism_render_architecture::cloth::dynamics::solve_cloth_with_collision`
//! (with the default no-op collision hook). The sandbox has no GPU, so this
//! shader is validated by naga compilation only (see `shader_tests`); it is the
//! persistent GPU-driven cloth sim stage whose per-stage dispatch schedule is
//! described by
//! `prism_render_architecture::cloth::gpu::pipeline` (the substep × iteration ×
//! per-color-batch loop lives on the host; each kernel below is one dispatch).
//!
//! Unlike the guide-strand kernel in `hair_sim.wesl`, where one invocation owns
//! a whole strand, cloth constraints connect arbitrary sim-mesh particles, so
//! the solve is split into the same per-particle / per-constraint stages the
//! CPU golden runs in sequence:
//!
//! 1. `cloth_predict` — one invocation per particle: damp velocity, add
//!    gravity, integrate position, and snapshot the pre-step position for the
//!    later velocity recovery.
//! 2. `cloth_project_distance` — one invocation per two-sided (stretch / bend /
//!    shear) constraint in the current color batch; a color batch shares no
//!    particle, so the writes never race (Jacobi within a color, Gauss-Seidel
//!    across colors, identical to the CPU order).
//! 3. `cloth_project_long_range` — one invocation per one-sided (LRA / tether)
//!    constraint; only pulls when over-extended.
//! 4. `cloth_strain_limit` — one invocation per structural (stretch) constraint;
//!    hard-clamps overstretch to `1 + limit` of rest length.
//! 5. `cloth_velocity_update` — one invocation per particle: recover velocity
//!    from the position delta (pinned particles are zeroed).
//!
//! Buffer layout mirrors `prism_render_architecture::cloth::gpu::buffers`:
//! `positions[p].xyz` is the world position and `positions[p].w` is the inverse
//! mass (`<= 0` = pinned). `velocities[p].xyz` is the explicit XPBD velocity.
//! `prev_positions[p].xyz` is the pre-substep position snapshot. Each
//! constraint is a 16-byte `ClothConstraint { a, b, rest_length, compliance }`;
//! one-sided-ness and structural-ness are encoded by which buffer / dispatch a
//! constraint is placed in (matching the colored-batch split on the host), so
//! the struct never needs a kind tag.

/// Squared-length floor below which a constraint / edge is skipped, so
/// normalization never divides by ~0 (mirrors the CPU `EPS_LEN_SQ`).
const CLOTH_EPS_LEN_SQ: f32 = 1.0e-12;

/// Global parameters shared by every cloth kernel in the dispatch. Mirrors the
/// per-substep scalars the CPU `solve_cloth_with_collision` derives once.
struct ClothSimParams {
    /// Constant external acceleration (gravity), world units per second².
    gravity: vec3<f32>,
    /// Substep duration `dt / substeps`, seconds.
    dt_sub: f32,
    /// Velocity retention `1 - damping` applied each substep.
    retain: f32,
    /// Strain limiter maximum stretch scale `1 + strain_limit`.
    strain_max_scale: f32,
    /// Number of particles bounding the per-particle dispatches.
    particle_count: u32,
    /// Number of constraints bounding the per-constraint dispatches.
    constraint_count: u32,
}

/// One distance constraint: particle indices `a`/`b`, rest length, and XPBD
/// compliance. Byte-compatible with the CPU `CONSTRAINT_STRIDE` (16 bytes).
struct ClothConstraint {
    a: u32,
    b: u32,
    rest_length: f32,
    compliance: f32,
}

// Particle state: `xyz` = world position, `w` = inverse mass (<= 0 = pinned).
@group(0) @binding(0) var<storage, read_write> positions: array<vec4<f32>>;
// Explicit XPBD velocity (`xyz` used).
@group(0) @binding(1) var<storage, read_write> velocities: array<vec4<f32>>;
// Pre-substep position snapshot for velocity recovery (`xyz` used).
@group(0) @binding(2) var<storage, read_write> prev_positions: array<vec4<f32>>;
// Constraints for the current dispatch (a color batch, LRA set, or structural
// set, depending on the stage).
@group(0) @binding(3) var<storage, read> constraints: array<ClothConstraint>;

var<immediate> params: ClothSimParams;

/// Effective inverse mass: `<= 0` (pinned) reads as `0`, mirroring the CPU
/// `inverse_mass` helper.
fn cloth_inverse_mass(w: f32) -> f32 {
    return max(w, 0.0);
}

/// Stage 1: predict positions from damped velocity + gravity, snapshotting the
/// pre-step position. Mirrors the CPU predict loop.
@compute @workgroup_size(64)
fn cloth_predict(@builtin(global_invocation_id) gid: vec3<u32>) {
    let p = gid.x;
    if p >= params.particle_count {
        return;
    }
    let pos = positions[p].xyz;
    let inv_mass = positions[p].w;
    // Snapshot the pre-step position (done for pinned particles too, so the CPU
    // velocity-recovery pass matches bit-for-bit).
    prev_positions[p] = vec4<f32>(pos, prev_positions[p].w);
    if inv_mass <= 0.0 {
        return;
    }
    let velocity = velocities[p].xyz * params.retain + params.gravity * params.dt_sub;
    velocities[p] = vec4<f32>(velocity, velocities[p].w);
    positions[p] = vec4<f32>(pos + velocity * params.dt_sub, inv_mass);
}

/// Stage 2: two-sided distance projection (stretch / bend / shear). Mirrors the
/// CPU `project_distance` without the one-sided early-out.
@compute @workgroup_size(64)
fn cloth_project_distance(@builtin(global_invocation_id) gid: vec3<u32>) {
    let c = gid.x;
    if c >= params.constraint_count {
        return;
    }
    let con = constraints[c];
    if con.a == con.b {
        return;
    }
    let a = con.a;
    let b = con.b;
    let wa = cloth_inverse_mass(positions[a].w);
    let wb = cloth_inverse_mass(positions[b].w);
    let w_sum = wa + wb;
    if w_sum <= 0.0 {
        return;
    }
    let delta = positions[a].xyz - positions[b].xyz;
    let dist_sq = dot(delta, delta);
    if dist_sq <= CLOTH_EPS_LEN_SQ {
        return;
    }
    let dist = sqrt(dist_sq);
    let error = dist - con.rest_length;
    let alpha_tilde = max(con.compliance, 0.0) / (params.dt_sub * params.dt_sub);
    let denom = w_sum + alpha_tilde;
    if denom <= 0.0 {
        return;
    }
    let d_lambda = -error / denom;
    let direction = delta / dist;
    let correction = direction * d_lambda;
    positions[a] = vec4<f32>(positions[a].xyz + correction * wa, positions[a].w);
    positions[b] = vec4<f32>(positions[b].xyz - correction * wb, positions[b].w);
}

/// Stage 3: one-sided long-range / tether projection. Mirrors
/// `project_distance` with `is_one_sided` — only pulls when over-extended.
@compute @workgroup_size(64)
fn cloth_project_long_range(@builtin(global_invocation_id) gid: vec3<u32>) {
    let c = gid.x;
    if c >= params.constraint_count {
        return;
    }
    let con = constraints[c];
    if con.a == con.b {
        return;
    }
    let a = con.a;
    let b = con.b;
    let wa = cloth_inverse_mass(positions[a].w);
    let wb = cloth_inverse_mass(positions[b].w);
    let w_sum = wa + wb;
    if w_sum <= 0.0 {
        return;
    }
    let delta = positions[a].xyz - positions[b].xyz;
    let dist_sq = dot(delta, delta);
    if dist_sq <= CLOTH_EPS_LEN_SQ {
        return;
    }
    let dist = sqrt(dist_sq);
    let error = dist - con.rest_length;
    if error <= 0.0 {
        return;
    }
    let alpha_tilde = max(con.compliance, 0.0) / (params.dt_sub * params.dt_sub);
    let denom = w_sum + alpha_tilde;
    if denom <= 0.0 {
        return;
    }
    let d_lambda = -error / denom;
    let direction = delta / dist;
    let correction = direction * d_lambda;
    positions[a] = vec4<f32>(positions[a].xyz + correction * wa, positions[a].w);
    positions[b] = vec4<f32>(positions[b].xyz - correction * wb, positions[b].w);
}

/// Stage 4: strain limiter over structural (stretch) constraints. Mirrors the
/// CPU `apply_strain_limit` mass-weighted overstretch clamp.
@compute @workgroup_size(64)
fn cloth_strain_limit(@builtin(global_invocation_id) gid: vec3<u32>) {
    let c = gid.x;
    if c >= params.constraint_count {
        return;
    }
    let con = constraints[c];
    if con.a == con.b {
        return;
    }
    let a = con.a;
    let b = con.b;
    let wa = cloth_inverse_mass(positions[a].w);
    let wb = cloth_inverse_mass(positions[b].w);
    let w_sum = wa + wb;
    if w_sum <= 0.0 {
        return;
    }
    let max_len = con.rest_length * params.strain_max_scale;
    let delta = positions[a].xyz - positions[b].xyz;
    let dist_sq = dot(delta, delta);
    let max_sq = max_len * max_len;
    if dist_sq <= max_sq || dist_sq <= CLOTH_EPS_LEN_SQ {
        return;
    }
    let dist = sqrt(dist_sq);
    let excess = dist - max_len;
    let direction = delta / dist;
    let correction = direction * excess;
    positions[a] = vec4<f32>(positions[a].xyz - correction * (wa / w_sum), positions[a].w);
    positions[b] = vec4<f32>(positions[b].xyz + correction * (wb / w_sum), positions[b].w);
}

/// Stage 5: recover velocity from the position delta. Mirrors the CPU velocity
/// recovery pass; pinned particles are zeroed.
@compute @workgroup_size(64)
fn cloth_velocity_update(@builtin(global_invocation_id) gid: vec3<u32>) {
    let p = gid.x;
    if p >= params.particle_count {
        return;
    }
    if positions[p].w <= 0.0 {
        velocities[p] = vec4<f32>(0.0, 0.0, 0.0, velocities[p].w);
        return;
    }
    if params.dt_sub <= 0.0 {
        return;
    }
    let delta = positions[p].xyz - prev_positions[p].xyz;
    velocities[p] = vec4<f32>(delta / params.dt_sub, velocities[p].w);
}
