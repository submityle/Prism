//! GPU guide-strand XPBD simulation kernel (shared, style-agnostic base).
//!
//! Compute twin of the CPU golden reference
//! `prism_render_architecture::hair::dynamics::simulate_guides`. The sandbox
//! has no GPU, so this shader is validated by naga compilation only (see
//! `shader_tests`); it is the persistent GPU-driven sim stage (design §8) that
//! advances a groom's sparse *guide* strands so the many render strands can be
//! interpolated from them.
//!
//! One invocation owns one guide strand and advances it by `params.dt`, split
//! into `params.substeps` semi-implicit steps, each projecting the same four
//! XPBD constraint families as the CPU solver over `params.iterations`
//! Gauss-Seidel sweeps, in the identical order so the GPU path stays a faithful
//! twin of the deterministic CPU golden:
//!
//! 1. edge-length (compliant distance, `alpha = compliance / dt_sub^2`),
//! 2. local shape (discrete-Laplacian bending toward the neighbour midpoint),
//! 3. global shape (pull toward the authored goal pose),
//! 4. long-range attachment (one-sided tether to the strand root).
//!
//! Buffer layout: every particle is a `vec4<f32>` whose `xyz` is the world
//! position and whose `w` is the inverse mass (`0` = pinned root riding the
//! skinned scalp). `prev_positions.xyz` carries the implicit-velocity history.
//! `strands[s]` gives each guide's flat offsets into the shared particle /
//! rest-length / goal arrays (the same offset-slicing discipline as the CPU
//! `simulate_guides`). Callers must size `rest_lengths` to cover
//! `rest_offset + particle_count - 1` and `goals` to cover
//! `goal_offset + particle_count`.

/// Segments shorter than this are skipped to avoid dividing by ~0 (mirrors the
/// CPU `EPS_LEN`).
const HAIR_EPS_LEN: f32 = 1.0e-12;

/// Global XPBD parameters shared by every strand in the dispatch. Mirrors the
/// CPU `XpbdParams` plus the strand count that bounds the dispatch.
struct HairXpbdParams {
    gravity: vec3<f32>,
    dt: f32,
    edge_compliance: f32,
    local_stiffness: f32,
    global_stiffness: f32,
    lra_stiffness: f32,
    damping: f32,
    substeps: u32,
    iterations: u32,
    strand_count: u32,
}

/// Flat layout descriptor for one guide strand.
struct HairStrand {
    particle_offset: u32,
    particle_count: u32,
    rest_offset: u32,
    goal_offset: u32,
}

// Particle state: `xyz` = position, `w` = inverse mass (0 = pinned).
@group(0) @binding(0) var<storage, read_write> positions: array<vec4<f32>>;
// Previous position history for the implicit velocity (`xyz` used).
@group(0) @binding(1) var<storage, read_write> prev_positions: array<vec4<f32>>;
// Per-particle global goal pose (`xyz` used).
@group(0) @binding(2) var<storage, read> goals: array<vec4<f32>>;
// Per-segment rest lengths, flat across all strands.
@group(0) @binding(3) var<storage, read> rest_lengths: array<f32>;
// Per-strand flat offsets.
@group(0) @binding(4) var<storage, read> strands: array<HairStrand>;

var<immediate> params: HairXpbdParams;

// Semi-implicit (Verlet) prediction. Mirrors CPU `integrate`.
fn hair_integrate(off: u32, count: u32, gravity_step: vec3<f32>, velocity_retain: f32) {
    for (var i = 0u; i < count; i = i + 1u) {
        let p = off + i;
        let inv_mass = positions[p].w;
        let pos = positions[p].xyz;
        if inv_mass <= 0.0 {
            // Pinned: stay put but roll prev forward so implied velocity is 0.
            prev_positions[p] = vec4<f32>(pos, prev_positions[p].w);
            continue;
        }
        let velocity = (pos - prev_positions[p].xyz) * velocity_retain;
        prev_positions[p] = vec4<f32>(pos, prev_positions[p].w);
        positions[p] = vec4<f32>(pos + velocity + gravity_step, inv_mass);
    }
}

// Compliant edge-length constraint over every segment. Mirrors CPU `solve_edges`.
fn hair_solve_edges(off: u32, count: u32, rest_off: u32, alpha: f32) {
    if count < 2u {
        return;
    }
    for (var i = 0u; i + 1u < count; i = i + 1u) {
        let a = off + i;
        let b = off + i + 1u;
        let rest = rest_lengths[rest_off + i];
        let w_a = positions[a].w;
        let w_b = positions[b].w;
        let w_sum = w_a + w_b;
        if w_sum <= 0.0 {
            continue;
        }
        let delta = positions[b].xyz - positions[a].xyz;
        let len = length(delta);
        if len <= HAIR_EPS_LEN {
            continue;
        }
        let dir = delta / len;
        let lambda = (len - rest) / (w_sum + alpha);
        positions[a] = vec4<f32>(positions[a].xyz + dir * (w_a * lambda), w_a);
        positions[b] = vec4<f32>(positions[b].xyz - dir * (w_b * lambda), w_b);
    }
}

// Local bending constraint over interior particles. Mirrors CPU `solve_local`.
fn hair_solve_local(off: u32, count: u32, stiffness: f32) {
    if count < 3u || stiffness <= 0.0 {
        return;
    }
    for (var i = 1u; i + 1u < count; i = i + 1u) {
        let p = off + i;
        if positions[p].w <= 0.0 {
            continue;
        }
        let prev_pos = positions[off + i - 1u].xyz;
        let next_pos = positions[off + i + 1u].xyz;
        let midpoint = (prev_pos + next_pos) * 0.5;
        let correction = (midpoint - positions[p].xyz) * stiffness;
        positions[p] = vec4<f32>(positions[p].xyz + correction, positions[p].w);
    }
}

// Global shape constraint toward the goal pose. Mirrors CPU `solve_global`.
fn hair_solve_global(off: u32, count: u32, goal_off: u32, stiffness: f32) {
    if stiffness <= 0.0 {
        return;
    }
    for (var i = 0u; i < count; i = i + 1u) {
        let p = off + i;
        if positions[p].w <= 0.0 {
            continue;
        }
        let goal = goals[goal_off + i].xyz;
        let correction = (goal - positions[p].xyz) * stiffness;
        positions[p] = vec4<f32>(positions[p].xyz + correction, positions[p].w);
    }
}

// One-sided long-range attachment (tether) to the root. Mirrors CPU `solve_lra`.
fn hair_solve_lra(off: u32, count: u32, rest_off: u32, stiffness: f32) {
    if count < 2u || stiffness <= 0.0 {
        return;
    }
    let root = positions[off].xyz;
    var max_distance = 0.0;
    for (var i = 1u; i < count; i = i + 1u) {
        max_distance = max_distance + max(rest_lengths[rest_off + i - 1u], 0.0);
        let p = off + i;
        if positions[p].w <= 0.0 {
            continue;
        }
        if max_distance <= HAIR_EPS_LEN {
            continue;
        }
        let delta = positions[p].xyz - root;
        let distance = length(delta);
        if distance > max_distance && distance > HAIR_EPS_LEN {
            let tether = root + delta * (max_distance / distance);
            let correction = (tether - positions[p].xyz) * stiffness;
            positions[p] = vec4<f32>(positions[p].xyz + correction, positions[p].w);
        }
    }
}

@compute @workgroup_size(64)
fn simulate_guides(@builtin(global_invocation_id) gid: vec3<u32>) {
    let s = gid.x;
    if s >= params.strand_count {
        return;
    }
    if params.substeps == 0u || params.dt <= 0.0 {
        return;
    }

    let strand = strands[s];
    let count = strand.particle_count;
    if count == 0u {
        return;
    }
    let off = strand.particle_offset;
    let rest_off = strand.rest_offset;
    let goal_off = strand.goal_offset;

    let sub_dt = params.dt / f32(params.substeps);
    let sub_dt_sq = sub_dt * sub_dt;
    let damping = clamp(params.damping, 0.0, 1.0);
    let local_stiffness = clamp(params.local_stiffness, 0.0, 1.0);
    let global_stiffness = clamp(params.global_stiffness, 0.0, 1.0);
    let lra_stiffness = clamp(params.lra_stiffness, 0.0, 1.0);
    let edge_compliance = max(params.edge_compliance, 0.0);
    // XPBD compliance normalized by the squared step so stiffness is
    // step-count independent (mirrors the CPU `alpha`).
    let alpha = edge_compliance / sub_dt_sq;
    let gravity_step = params.gravity * sub_dt_sq;
    let velocity_retain = 1.0 - damping;

    for (var step = 0u; step < params.substeps; step = step + 1u) {
        hair_integrate(off, count, gravity_step, velocity_retain);
        for (var it = 0u; it < params.iterations; it = it + 1u) {
            hair_solve_edges(off, count, rest_off, alpha);
            hair_solve_local(off, count, local_stiffness);
            hair_solve_global(off, count, goal_off, global_stiffness);
            hair_solve_lra(off, count, rest_off, lra_stiffness);
        }
    }
}
