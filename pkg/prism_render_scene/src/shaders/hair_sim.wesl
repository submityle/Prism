//! GPU guide-strand XPBD simulation kernel (shared, style-agnostic base).
//!
//! Compute twin of the CPU golden reference
//! `prism_render_architecture::hair::dynamics::simulate_guides`. A `GPU` is now
//! available, so this shader is compile- and type-checked through the same
//! render-world `ShaderCache` / `wesl` pipeline the runtime uses (see
//! `shader_tests`), with on-device dispatch/perf calibration still awaiting
//! real hardware; it is the persistent GPU-driven sim stage (design §8) that
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
//! 4. long-range attachment (one-sided tether to the strand root),
//!
//! then, once per substep after the constraint sweeps, projects every free
//! particle out of the analytic body colliders (spheres / capsules fitted to
//! the head, neck and shoulders) so hair settles against the body without
//! fighting the shape solve. This mirrors the CPU golden ordering exactly:
//! `simulate_strand` runs `resolve_strand_collisions` at the tail of each
//! substep (design §6.2, §7).
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

/// Squared-length epsilon for the collider projection. Mirrors the CPU
/// `collision::EPS_LEN_SQ` so both stages agree on what "degenerate" means.
const HAIR_EPS_LEN_SQ: f32 = 1.0e-24;

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
    collider_count: u32,
}

/// Flat layout descriptor for one guide strand.
struct HairStrand {
    particle_offset: u32,
    particle_count: u32,
    rest_offset: u32,
    goal_offset: u32,
}

/// Analytic collision proxy. Mirrors the CPU `collision::Collider` enum packed
/// into two `vec4`s so the storage layout is a clean 32-byte stride:
/// `center_a.xyz` is the sphere center (or capsule endpoint `a`), `center_a.w`
/// is the inflation radius, `end_b.xyz` is the capsule endpoint `b` (unused for
/// a sphere) and `end_b.w` is the kind flag (`< 0.5` = sphere, otherwise
/// capsule).
struct HairCollider {
    center_a: vec4<f32>,
    end_b: vec4<f32>,
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
// Analytic body colliders applied after every substep's constraint sweeps.
@group(0) @binding(5) var<storage, read> colliders: array<HairCollider>;

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

// Pushes `point` out to the surface of the sphere `(center, radius)`. Mirrors
// CPU `collision::push_out_sphere`: a non-positive radius or an exterior point
// is returned unchanged, a point coincident with the center escapes along `+Y`
// (a deterministic non-NaN fallback), and otherwise the point lands exactly on
// the surface along the radial direction.
fn hair_push_out_sphere(center: vec3<f32>, radius: f32, point: vec3<f32>) -> vec3<f32> {
    if radius <= 0.0 {
        return point;
    }
    let delta = point - center;
    let dist_sq = dot(delta, delta);
    if dist_sq >= radius * radius {
        return point;
    }
    if dist_sq <= HAIR_EPS_LEN_SQ {
        return center + vec3<f32>(0.0, radius, 0.0);
    }
    // Mirror CPU `normalize_or_zero` (multiply by reciprocal length) then scale
    // by the radius, in the same operation order so the twin stays bit-faithful.
    let dir = delta * (1.0 / sqrt(dist_sq));
    return center + dir * radius;
}

// Returns the point on segment `a`..`b` closest to `point`. Mirrors CPU
// `collision::closest_point_on_segment`; a zero-length segment returns `a`.
fn hair_closest_point_on_segment(a: vec3<f32>, b: vec3<f32>, point: vec3<f32>) -> vec3<f32> {
    let ab = b - a;
    let len_sq = dot(ab, ab);
    if len_sq <= HAIR_EPS_LEN_SQ {
        return a;
    }
    let t = clamp(dot(point - a, ab) / len_sq, 0.0, 1.0);
    return a + ab * t;
}

// Projects `point` out of one analytic collider. Mirrors CPU `Collider::push_out`:
// a capsule reduces to a sphere centered on the closest segment point.
fn hair_push_out_collider(collider: HairCollider, point: vec3<f32>) -> vec3<f32> {
    let radius = collider.center_a.w;
    if collider.end_b.w < 0.5 {
        return hair_push_out_sphere(collider.center_a.xyz, radius, point);
    }
    let closest = hair_closest_point_on_segment(collider.center_a.xyz, collider.end_b.xyz, point);
    return hair_push_out_sphere(closest, radius, point);
}

// Projects every free particle of the strand out of every collider, in order.
// Mirrors CPU `resolve_strand_collisions`: pinned particles are never moved, an
// empty collider set is a no-op, and colliders are applied in array order so
// the last one to push wins for a given particle (deterministic).
fn hair_resolve_collisions(off: u32, count: u32) {
    if params.collider_count == 0u {
        return;
    }
    for (var i = 0u; i < count; i = i + 1u) {
        let p = off + i;
        if positions[p].w <= 0.0 {
            continue;
        }
        var pos = positions[p].xyz;
        for (var c = 0u; c < params.collider_count; c = c + 1u) {
            pos = hair_push_out_collider(colliders[c], pos);
        }
        positions[p] = vec4<f32>(pos, positions[p].w);
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
        // Project free particles out of the body colliders once per substep,
        // after the constraint sweeps (mirrors CPU `resolve_strand_collisions`).
        hair_resolve_collisions(off, count);
    }
}
