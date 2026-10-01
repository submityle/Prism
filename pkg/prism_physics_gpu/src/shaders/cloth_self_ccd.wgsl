// GPU continuous self-collision (self-CCD) for cloth particles — the
// parallel-safe Jacobi twin of `prism_physics_core`'s
// `resolve_self_ccd_jacobi`.
//
// Two own-slot passes, each reading only a frozen snapshot so the result is
// independent of invocation order (a Jacobi iteration, never Gauss-Seidel):
//
//   * `phase1_pairs` — one invocation per candidate pair. It solves the pair's
//     swept-pair time of impact from the frozen snapshot and writes its own
//     per-partner position deltas (`pair_dp_a` / `pair_dp_b`) and velocity
//     deltas (`pair_dv_a` / `pair_dv_b`, with the `w` lane flagging whether an
//     impulse was exchanged). It reads no other pair's slot. No atomics.
//   * `phase2_apply` — one invocation per particle. It gathers the particle's
//     half of every incident pair through a host-built CSR incidence list
//     (ascending by pair index, the golden's reduction order), sums the
//     position/velocity deltas, and writes `out_positions` / `out_velocities`.
//
// The candidate pairs and the per-particle incidence list are built on the host
// (integer-exact) and uploaded, so the only floating-point work on the GPU is
// the swept-pair resolution itself, which the parity suite checks against the
// CPU golden within a tight tolerance.
//
// Provenance: the swept-pair TOI resolution is standard analytic
// continuous-collision geometry; the Jacobi own-slot accumulate/apply split is
// standard parallel position-based dynamics. No Unreal Engine source or derived
// code.

// Relative-motion floor below which a swept pair is treated as non-closing,
// mirroring `prism_physics_core`'s `EPS_REL_MOTION`.
const EPS_REL_MOTION: f32 = 1.0e-12;
// Sentinel for "no impact in [0, 1]"; any valid TOI is in [0, 1].
const NO_HIT: f32 = 1.0e30;

struct Params {
    // Number of candidate pairs (phase-1 thread count).
    pair_count: u32,
    // Number of addressable particles (phase-2 thread count).
    particle_count: u32,
    // Sanitized fabric thickness: the enforced separation.
    thickness: f32,
    // Sanitized normal restitution in 0..=1.
    restitution: f32,
    // `1 / dt`, or 0 when |dt| is (near) zero (velocity recovery disabled).
    inv_dt: f32,
    // Padding to a 32-byte (8-word) boundary.
    pad0: u32,
    pad1: u32,
    pad2: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
// Frozen frame-end positions, xyz used.
@group(0) @binding(1) var<storage, read> positions: array<vec4<f32>>;
// Frozen frame-start positions, xyz used.
@group(0) @binding(2) var<storage, read> prev_positions: array<vec4<f32>>;
// Frozen velocities, xyz used.
@group(0) @binding(3) var<storage, read> velocities: array<vec4<f32>>;
// Index-aligned inverse masses (0 marks a pinned particle).
@group(0) @binding(4) var<storage, read> inverse_masses: array<f32>;
// Candidate pairs: (first particle, second particle), ascending.
@group(0) @binding(5) var<storage, read> pairs: array<vec2<u32>>;
// CSR offsets into `vert_entries`, length `particle_count + 1`.
@group(0) @binding(6) var<storage, read> vert_offsets: array<u32>;
// Incident (pair index, side) records per particle, ascending by pair index.
@group(0) @binding(7) var<storage, read> vert_entries: array<vec2<u32>>;
// Phase-1 output: each pair's position delta for its first/second partner.
@group(0) @binding(8) var<storage, read_write> pair_dp_a: array<vec4<f32>>;
@group(0) @binding(9) var<storage, read_write> pair_dp_b: array<vec4<f32>>;
// Phase-1 output: each pair's velocity delta (xyz) + flag (w: 1 = exchanged).
@group(0) @binding(10) var<storage, read_write> pair_dv_a: array<vec4<f32>>;
@group(0) @binding(11) var<storage, read_write> pair_dv_b: array<vec4<f32>>;
// Phase-2 output: the snapshot state advanced by its corrections.
@group(0) @binding(12) var<storage, read_write> out_positions: array<vec4<f32>>;
@group(0) @binding(13) var<storage, read_write> out_velocities: array<vec4<f32>>;

// Earliest time of impact in [0, 1] of the swept pair `a: prev_a -> curr_a`,
// `b: prev_b -> curr_b` reaching separation `thickness`, or `NO_HIT`.
// Exact mirror of `prism_physics_core::swept_pair_toi`.
fn swept_pair_toi(
    prev_a: vec3<f32>,
    curr_a: vec3<f32>,
    prev_b: vec3<f32>,
    curr_b: vec3<f32>,
    thickness: f32,
) -> f32 {
    let d0 = prev_a - prev_b;
    let d1 = curr_a - curr_b;
    let dv = d1 - d0;
    let c = dot(d0, d0) - thickness * thickness;
    if (c <= 0.0) {
        return 0.0;
    }
    let a = dot(dv, dv);
    if (a <= EPS_REL_MOTION) {
        return NO_HIT;
    }
    let b = 2.0 * dot(d0, dv);
    let disc = b * b - 4.0 * a * c;
    if (disc < 0.0) {
        return NO_HIT;
    }
    let t = ((-b) - sqrt(disc)) / (2.0 * a);
    if (t >= 0.0 && t <= 1.0) {
        return t;
    }
    return NO_HIT;
}

@compute @workgroup_size(64)
fn phase1_pairs(@builtin(global_invocation_id) gid: vec3<u32>) {
    let pid = gid.x;
    if (pid >= params.pair_count) {
        return;
    }
    // Default: no contribution.
    pair_dp_a[pid] = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    pair_dp_b[pid] = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    pair_dv_a[pid] = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    pair_dv_b[pid] = vec4<f32>(0.0, 0.0, 0.0, 0.0);

    let pair = pairs[pid];
    let ia = pair.x;
    let ib = pair.y;
    let wa = max(inverse_masses[ia], 0.0);
    let wb = max(inverse_masses[ib], 0.0);
    let wsum = wa + wb;
    if (wsum <= 0.0) {
        return;
    }

    let prev_a = prev_positions[ia].xyz;
    let curr_a = positions[ia].xyz;
    let prev_b = prev_positions[ib].xyz;
    let curr_b = positions[ib].xyz;

    let t = swept_pair_toi(prev_a, curr_a, prev_b, curr_b, params.thickness);
    if (t > 1.0) {
        return;
    }

    // Positions at the time of impact.
    let a_c = prev_a + (curr_a - prev_a) * t;
    let b_c = prev_b + (curr_b - prev_b) * t;
    let delta = a_c - b_c;
    let dist_sq = dot(delta, delta);
    var normal: vec3<f32>;
    var dist: f32;
    if (dist_sq > 0.0) {
        dist = sqrt(dist_sq);
        normal = delta / dist;
    } else {
        // Coincident contact: `normalize_or_zero` yields zero, so the golden
        // falls back to the +X axis.
        normal = vec3<f32>(1.0, 0.0, 0.0);
        dist = 0.0;
    }
    let penetration = max(params.thickness - dist, 0.0);
    let inv_wsum = 1.0 / wsum;

    let pos_a = a_c + normal * (wa * inv_wsum * penetration);
    let pos_b = b_c - normal * (wb * inv_wsum * penetration);
    if (wa > 0.0) {
        pair_dp_a[pid] = vec4<f32>(pos_a - curr_a, 0.0);
    }
    if (wb > 0.0) {
        pair_dp_b[pid] = vec4<f32>(pos_b - curr_b, 0.0);
    }

    // Normal restitution impulse, recovered from the TOI approach velocity.
    let va_in = (a_c - prev_a) * params.inv_dt;
    let vb_in = (b_c - prev_b) * params.inv_dt;
    let vrel_n = dot(va_in - vb_in, normal);
    if (vrel_n < 0.0) {
        let impulse = -(1.0 + params.restitution) * vrel_n * inv_wsum;
        let new_va = va_in + normal * (wa * impulse);
        let new_vb = vb_in - normal * (wb * impulse);
        if (wa > 0.0) {
            pair_dv_a[pid] = vec4<f32>(new_va - velocities[ia].xyz, 1.0);
        }
        if (wb > 0.0) {
            pair_dv_b[pid] = vec4<f32>(new_vb - velocities[ib].xyz, 1.0);
        }
    }
}

@compute @workgroup_size(64)
fn phase2_apply(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.particle_count) {
        return;
    }
    let frozen_pos = positions[i].xyz;
    let frozen_vel = velocities[i].xyz;
    // Pinned particle: never moved, velocity never recovered.
    if (max(inverse_masses[i], 0.0) <= 0.0) {
        out_positions[i] = vec4<f32>(frozen_pos, positions[i].w);
        out_velocities[i] = vec4<f32>(frozen_vel, velocities[i].w);
        return;
    }

    let start = vert_offsets[i];
    let end = vert_offsets[i + 1u];
    var pos_acc = vec3<f32>(0.0, 0.0, 0.0);
    var vel_acc = vec3<f32>(0.0, 0.0, 0.0);
    var touched = false;
    for (var e = start; e < end; e = e + 1u) {
        let entry = vert_entries[e];
        let pid = entry.x;
        let side = entry.y;
        if (side == 0u) {
            pos_acc += pair_dp_a[pid].xyz;
            let dv = pair_dv_a[pid];
            if (dv.w != 0.0) {
                vel_acc += dv.xyz;
                touched = true;
            }
        } else {
            pos_acc += pair_dp_b[pid].xyz;
            let dv = pair_dv_b[pid];
            if (dv.w != 0.0) {
                vel_acc += dv.xyz;
                touched = true;
            }
        }
    }

    out_positions[i] = vec4<f32>(frozen_pos + pos_acc, positions[i].w);
    if (touched) {
        out_velocities[i] = vec4<f32>(frozen_vel + vel_acc, velocities[i].w);
    } else {
        out_velocities[i] = vec4<f32>(frozen_vel, velocities[i].w);
    }
}
