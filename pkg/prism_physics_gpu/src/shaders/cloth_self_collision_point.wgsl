// GPU point (vertex-vertex) self-collision for cloth particles — the
// parallel-safe Jacobi twin of `prism_physics_core`'s
// `resolve_self_collision_jacobi` / `resolve_self_collision_with_friction_jacobi`.
//
// The point self-collision tier separates every pair of cloth samples that end
// a step closer than the fabric `thickness`. The Gauss-Seidel core walks the
// pairs sequentially and scatters each half-correction in place, so a later
// pair already sees an earlier pair's moved positions — a reduction order no
// compute kernel can reproduce, because every GPU invocation reads the *same*
// frozen snapshot. This kernel therefore mirrors the engine's Jacobi golden:
// a frozen-snapshot own-slot accumulate split into two passes.
//
//   * `phase1_pairs` — one invocation per candidate pair. It computes each
//     partner's own half of the separating push (and, when friction is active,
//     its per-pair tangential displacement) from the frozen snapshot and writes
//     its own `dp_a` / `dp_b` slot. It reads no other pair's slot. No atomics.
//   * `phase2_apply` — one invocation per particle. It gathers the particle's
//     half of every incident pair through a host-built CSR incidence list
//     (ascending by pair index, the golden's reduction order), sums the
//     deltas, and writes `out_positions`.
//
// The candidate pairs and the per-particle incidence list are built on the host
// (integer-exact), so the only floating-point work on the GPU is the
// separating-push arithmetic, which the parity suite checks against the CPU
// golden within a tight tolerance.
//
// Provenance: the inverse-mass-weighted separation is standard position-based
// dynamics; the tangential-friction projection is the one published by Macklin
// et al. (2014), "Unified Particle Physics for Real-Time Applications"; the
// Jacobi own-slot accumulate/apply split is standard parallel position-based
// dynamics. No Unreal Engine source or derived code.

// Coincident-pair floor below which the offset has no defined direction,
// mirroring `prism_physics_core`'s `EPS_LEN_SQ`.
const EPS_LEN_SQ: f32 = 1.0e-12;
// Tangential-slide floor below which friction has nothing to arrest,
// mirroring `prism_physics_core`'s `EPS_FRICTION`.
const EPS_FRICTION: f32 = 1.0e-12;

struct Params {
    // Number of candidate pairs (phase-1 thread count).
    pair_count: u32,
    // Number of addressable particles (phase-2 thread count).
    particle_count: u32,
    // Sanitized fabric thickness: the enforced separation.
    thickness: f32,
    // Sanitized Coulomb friction in 0..=1; 0 selects the plain normal push.
    friction: f32,
};

@group(0) @binding(0) var<uniform> params: Params;
// Frozen frame-end positions, xyz used.
@group(0) @binding(1) var<storage, read> positions: array<vec4<f32>>;
// Frozen frame-start positions, xyz used (friction only; equals `positions`
// in the plain path so the frictionless branch never reads a stale slide).
@group(0) @binding(2) var<storage, read> prev_positions: array<vec4<f32>>;
// Index-aligned inverse masses (0 marks a pinned particle).
@group(0) @binding(3) var<storage, read> inverse_masses: array<f32>;
// Candidate pairs: (first particle, second particle), ascending.
@group(0) @binding(4) var<storage, read> pairs: array<vec2<u32>>;
// CSR offsets into `vert_entries`, length `particle_count + 1`.
@group(0) @binding(5) var<storage, read> vert_offsets: array<u32>;
// Incident (pair index, side) records per particle, ascending by pair index.
@group(0) @binding(6) var<storage, read> vert_entries: array<vec2<u32>>;
// Phase-1 output: each pair's position delta for its first/second partner.
@group(0) @binding(7) var<storage, read_write> pair_dp_a: array<vec4<f32>>;
@group(0) @binding(8) var<storage, read_write> pair_dp_b: array<vec4<f32>>;
// Phase-2 output: the snapshot positions advanced by their corrections.
@group(0) @binding(9) var<storage, read_write> out_positions: array<vec4<f32>>;

// Unit separation direction (from `ai` toward `bi`) and penetration depth for a
// pair whose offset `delta = pb - pa` has squared length `dist_sq`. Exact
// mirror of `prism_physics_core::soft::collision::self_collision_jacobi::separation`:
// a coincident pair resolves along an index-oriented `+X`/`-X` axis so the two
// own-slot roles stay antisymmetric (the lower index moves `-X`, the higher `+X`).
fn separation(delta: vec3<f32>, dist_sq: f32, ai: u32, bi: u32) -> vec4<f32> {
    if (dist_sq <= EPS_LEN_SQ) {
        let axis = select(-1.0, 1.0, ai < bi);
        return vec4<f32>(axis, 0.0, 0.0, params.thickness);
    }
    let dist = sqrt(dist_sq);
    return vec4<f32>(delta / dist, params.thickness - dist);
}

// Particle `ai`'s own half of the separating push against neighbor `bi`,
// mirroring `half_correction`. `dir` points from `ai` toward `bi`, so `ai` is
// pushed the opposite way, weighted by its inverse-mass share.
fn half_correction(ai: u32, bi: u32, thickness_sq: f32) -> vec3<f32> {
    let pa = positions[ai].xyz;
    let pb = positions[bi].xyz;
    let delta = pb - pa;
    let dist_sq = dot(delta, delta);
    if (dist_sq >= thickness_sq) {
        return vec3<f32>(0.0, 0.0, 0.0);
    }
    let wa = max(inverse_masses[ai], 0.0);
    let wb = max(inverse_masses[bi], 0.0);
    let w_sum = wa + wb;
    if (w_sum <= 0.0) {
        return vec3<f32>(0.0, 0.0, 0.0);
    }
    let sep = separation(delta, dist_sq, ai, bi);
    let dir = sep.xyz;
    let penetration = sep.w;
    return dir * (-penetration * (wa / w_sum));
}

// Particle `ai`'s own per-pair displacement against neighbor `bi` with
// position-level Coulomb friction, mirroring `half_correction_with_friction`.
// Returns the displacement (post-position minus frozen position) `ai` should
// accumulate for this pair.
fn half_correction_with_friction(ai: u32, bi: u32, thickness_sq: f32) -> vec3<f32> {
    let pa = positions[ai].xyz;
    let pb = positions[bi].xyz;
    let delta = pb - pa;
    let dist_sq = dot(delta, delta);
    if (dist_sq >= thickness_sq) {
        return vec3<f32>(0.0, 0.0, 0.0);
    }
    let wa = max(inverse_masses[ai], 0.0);
    let wb = max(inverse_masses[bi], 0.0);
    let w_sum = wa + wb;
    if (w_sum <= 0.0) {
        return vec3<f32>(0.0, 0.0, 0.0);
    }
    let sep = separation(delta, dist_sq, ai, bi);
    let dir = sep.xyz;
    let penetration = sep.w;

    // Normal separation, inverse-mass weighted (identical to `half_correction`).
    let move_a = -penetration * (wa / w_sum);
    let sep_a = pa + dir * move_a;
    let move_b = penetration * (wb / w_sum);
    let sep_b = pb + dir * move_b;

    // Relative tangential slide since frame start.
    let prev_a = prev_positions[ai].xyz;
    let prev_b = prev_positions[bi].xyz;
    let rel = (sep_a - prev_a) - (sep_b - prev_b);
    let normal_amount = dot(rel, dir);
    let tangent = rel - dir * normal_amount;
    let tan_len_sq = dot(tangent, tangent);
    if (tan_len_sq <= EPS_FRICTION) {
        // No tangential slide to arrest: just the normal displacement.
        return sep_a - pa;
    }
    let tan_len = sqrt(tan_len_sq);
    let scale = min(params.friction * penetration / tan_len, 1.0);
    let corr = tangent * scale;
    return (sep_a - corr * (wa / w_sum)) - pa;
}

@compute @workgroup_size(64)
fn phase1_pairs(@builtin(global_invocation_id) gid: vec3<u32>) {
    let pid = gid.x;
    if (pid >= params.pair_count) {
        return;
    }
    let pair = pairs[pid];
    let ia = pair.x;
    let ib = pair.y;
    let thickness_sq = params.thickness * params.thickness;

    var dp_a: vec3<f32>;
    var dp_b: vec3<f32>;
    if (params.friction > 0.0) {
        dp_a = half_correction_with_friction(ia, ib, thickness_sq);
        dp_b = half_correction_with_friction(ib, ia, thickness_sq);
    } else {
        dp_a = half_correction(ia, ib, thickness_sq);
        dp_b = half_correction(ib, ia, thickness_sq);
    }
    pair_dp_a[pid] = vec4<f32>(dp_a, 0.0);
    pair_dp_b[pid] = vec4<f32>(dp_b, 0.0);
}

@compute @workgroup_size(64)
fn phase2_apply(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.particle_count) {
        return;
    }
    let frozen = positions[i].xyz;
    let start = vert_offsets[i];
    let end = vert_offsets[i + 1u];
    var acc = vec3<f32>(0.0, 0.0, 0.0);
    for (var e = start; e < end; e = e + 1u) {
        let rec = vert_entries[e];
        let pid = rec.x;
        let side = rec.y;
        if (side == 0u) {
            acc += pair_dp_a[pid].xyz;
        } else {
            acc += pair_dp_b[pid].xyz;
        }
    }
    out_positions[i] = vec4<f32>(frozen + acc, positions[i].w);
}
