// Guide-strand XPBD solver twin: advances on-device the same Extended Position
// Based Dynamics solve the CPU golden `simulate_guides` computes, one thread
// per guide strand.
//
// A groom simulates only its sparse guide strands; the render strands are
// interpolated from them and never simulate (design 6.4/6.5). Each guide is an
// independent poly-line, so the batch is embarrassingly parallel: one thread
// owns one strand's contiguous particle range and runs the full
// substep/iteration schedule over it, exactly mirroring
// `simulate_strand` (in `prism_render_architecture::hair::dynamics`) applied to
// each strand slice by `simulate_guides`. Because every thread mutates a
// disjoint particle range, the in-place Gauss-Seidel state updates are race
// free without any barrier, and the read-after-write ordering inside a strand
// matches the scalar reference sweep for sweep.
//
// This kernel mirrors that reference term for term:
//
//   for substep in 0..substeps {
//     integrate();                       // semi-implicit Verlet + gravity
//     for _ in 0..iterations {
//       solve_edges();                   // compliant distance constraint
//       solve_local();                   // discrete-Laplacian bending
//       solve_global();                  // pull toward goal pose
//       solve_lra();                     // long-range attachment / tether
//     }
//     resolve_strand_collisions();       // analytic body push-out
//   }
//
// All host-derived scalars (`sub_dt`, `alpha = edge_compliance / sub_dt^2`, the
// clamped stiffnesses, `gravity_step = gravity * sub_dt^2`, `velocity_retain =
// 1 - damping`) are precomputed on the host in the reference's evaluation order
// and uploaded in `Params`, so the device never re-derives them and cannot
// diverge on operation ordering. The two per-strand companion arrays
// (`rest_lengths`, `goal_positions`) are all-or-nothing sub-slices in the
// reference (`get(offset..end)` is `Some` for the whole strand or `None`), so a
// single `has_rest` / `has_goal` flag per strand faithfully reproduces the
// constraint being enabled or skipped for the whole strand.
//
// Every guard is reproduced bit-for-bit: pinned particles (`inverse_mass <= 0`)
// are never integrated and never moved by any constraint, degenerate
// (zero-length or two-pinned) edges are skipped, the tether radius walk stops a
// missing rest entry, and the collider push-out reproduces
// `Collider::push_out` (sphere and capsule) exactly, including the `+Y` escape
// for a point coincident with a sphere center.
//
// It uses only the portable core-WGSL subset (sqrt via length / min / max /
// clamp / dot / mul / add) - no exp, no pow, no optional feature - so this twin
// runs unmodified on Metal, Vulkan and DX12. It is not bit-exact with the
// scalar CPU reference: a GPU may fuse a multiply-add the reference leaves
// separate, perturbing the low mantissa bits over the iterated solve. The
// parity test therefore asserts a per-component tolerance rather than exact
// equality.
//
// Provenance: standard XPBD strand solver (compliant distance + shape-matching
// + long-range attachment) plus analytic sphere/capsule push-out; no Unreal
// Engine source or derived code.

// Matches `dynamics::EPS_LEN_SQ` / `collision::EPS_LEN_SQ`: vectors shorter than
// the square root of this are treated as zero-length.
const EPS_LEN_SQ: f32 = 1.0e-24;
// Matches `dynamics::EPS_LEN`: segments shorter than this are skipped.
const EPS_LEN: f32 = 1.0e-12;

// Collider kind discriminants, matching the CPU upload encoding.
const KIND_SPHERE: u32 = 0u;
const KIND_CAPSULE: u32 = 1u;

// Solve-wide parameters, all host-derived in the reference's evaluation order.
// `48`-byte scalar-packed layout matching the `bytemuck` upload struct `Params`.
struct Params {
    // `gravity * sub_dt^2`: the per-substep gravity displacement.
    gsx: f32,
    gsy: f32,
    gsz: f32,
    // `1 - clamp(damping, 0, 1)`: implicit-velocity retention fraction.
    velocity_retain: f32,
    // `max(edge_compliance, 0) / sub_dt^2`: XPBD compliance for the edge solve.
    alpha: f32,
    // Clamped local (bending) shape-matching strength in `0..=1`.
    local_stiffness: f32,
    // Clamped global shape-matching strength in `0..=1`.
    global_stiffness: f32,
    // Clamped long-range-attachment (tether) strength in `0..=1`.
    lra_stiffness: f32,
    // Number of semi-implicit integration substeps.
    substeps: u32,
    // Constraint projection sweeps per substep.
    iterations: u32,
    // Number of strand descriptors in `strands`.
    strand_count: u32,
    // Number of active colliders in `colliders` (`0` disables collision).
    collider_count: u32,
}

// One guide-strand descriptor. `16`-byte stride matching the `bytemuck` upload
// struct `Strand`: the base particle index of this strand in the flat `state`
// pool, its particle count, and whether its rest-length and goal companion
// slices are present (all-or-nothing per strand).
struct Strand {
    point_offset: u32,
    point_count: u32,
    has_rest: u32,
    has_goal: u32,
}

// One analytic collider. `32`-byte stride matching the `bytemuck` upload struct
// `GpuCollider`. For a sphere `(ax, ay, az)` is the center; for a capsule
// `(ax, ay, az)`..`(bx, by, bz)` is the segment.
struct Collider {
    kind: u32,
    radius: f32,
    ax: f32,
    ay: f32,
    az: f32,
    bx: f32,
    by: f32,
    bz: f32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> strands: array<Strand>;
// Per-particle state, stride `8` f32: position xyz, prev_position xyz,
// inverse_mass, rest_length (the length of the segment leaving this particle).
@group(0) @binding(2) var<storage, read_write> state: array<f32>;
// Per-particle goal position, stride `3` f32.
@group(0) @binding(3) var<storage, read> goals: array<f32>;
@group(0) @binding(4) var<storage, read> colliders: array<Collider>;

fn load_pos(gi: u32) -> vec3<f32> {
    let b = 8u * gi;
    return vec3<f32>(state[b], state[b + 1u], state[b + 2u]);
}

fn store_pos(gi: u32, v: vec3<f32>) {
    let b = 8u * gi;
    state[b] = v.x;
    state[b + 1u] = v.y;
    state[b + 2u] = v.z;
}

fn load_prev(gi: u32) -> vec3<f32> {
    let b = 8u * gi;
    return vec3<f32>(state[b + 3u], state[b + 4u], state[b + 5u]);
}

fn store_prev(gi: u32, v: vec3<f32>) {
    let b = 8u * gi;
    state[b + 3u] = v.x;
    state[b + 4u] = v.y;
    state[b + 5u] = v.z;
}

fn load_inv_mass(gi: u32) -> f32 {
    return state[8u * gi + 6u];
}

fn load_rest(gi: u32) -> f32 {
    return state[8u * gi + 7u];
}

fn load_goal(gi: u32) -> vec3<f32> {
    let b = 3u * gi;
    return vec3<f32>(goals[b], goals[b + 1u], goals[b + 2u]);
}

// Unit vector along `v`, or zero when `v` is (numerically) zero, mirroring
// `Vec3::normalize_or_zero` term order (`v * (1 / sqrt(len_sq))`).
fn normalize_or_zero(v: vec3<f32>) -> vec3<f32> {
    let len_sq = dot(v, v);
    if (len_sq > EPS_LEN_SQ) {
        return v * (1.0 / sqrt(len_sq));
    }
    return vec3<f32>(0.0, 0.0, 0.0);
}

// Pushes `point` out to the sphere surface, mirroring `push_out_sphere`.
fn push_out_sphere(center: vec3<f32>, radius: f32, point: vec3<f32>) -> vec3<f32> {
    if (radius <= 0.0) {
        return point;
    }
    let delta = point - center;
    let dist_sq = dot(delta, delta);
    if (dist_sq >= radius * radius) {
        return point;
    }
    if (dist_sq <= EPS_LEN_SQ) {
        return center + vec3<f32>(0.0, radius, 0.0);
    }
    let dir = normalize_or_zero(delta);
    return center + dir * radius;
}

// Point on segment `a`..`b` closest to `point`, mirroring
// `closest_point_on_segment`.
fn closest_point_on_segment(a: vec3<f32>, b: vec3<f32>, point: vec3<f32>) -> vec3<f32> {
    let ab = b - a;
    let len_sq = dot(ab, ab);
    if (len_sq <= EPS_LEN_SQ) {
        return a;
    }
    let t = clamp(dot(point - a, ab) / len_sq, 0.0, 1.0);
    return a + ab * t;
}

// Projects `point` out of collider `col`, mirroring `Collider::push_out`.
fn push_out(col: Collider, point: vec3<f32>) -> vec3<f32> {
    if (col.kind == KIND_CAPSULE) {
        let a = vec3<f32>(col.ax, col.ay, col.az);
        let b = vec3<f32>(col.bx, col.by, col.bz);
        let closest = closest_point_on_segment(a, b, point);
        return push_out_sphere(closest, col.radius, point);
    }
    let center = vec3<f32>(col.ax, col.ay, col.az);
    return push_out_sphere(center, col.radius, point);
}

@compute @workgroup_size(64)
fn evaluate(@builtin(global_invocation_id) gid: vec3<u32>) {
    let s = gid.x;
    if (s >= params.strand_count) {
        return;
    }
    let strand = strands[s];
    let offset = strand.point_offset;
    let n = strand.point_count;
    if (n == 0u) {
        return;
    }
    let gravity_step = vec3<f32>(params.gsx, params.gsy, params.gsz);

    for (var sub: u32 = 0u; sub < params.substeps; sub = sub + 1u) {
        // integrate: semi-implicit Verlet prediction with damping + gravity.
        for (var i: u32 = 0u; i < n; i = i + 1u) {
            let gi = offset + i;
            let pos = load_pos(gi);
            if (load_inv_mass(gi) <= 0.0) {
                store_prev(gi, pos);
                continue;
            }
            let velocity = (pos - load_prev(gi)) * params.velocity_retain;
            store_prev(gi, pos);
            store_pos(gi, pos + velocity + gravity_step);
        }

        for (var it: u32 = 0u; it < params.iterations; it = it + 1u) {
            // solve_edges: compliant distance constraint per segment.
            if (strand.has_rest == 1u) {
                var i: u32 = 0u;
                while (i + 1u < n) {
                    let a = offset + i;
                    let b = a + 1u;
                    let w_a = max(load_inv_mass(a), 0.0);
                    let w_b = max(load_inv_mass(b), 0.0);
                    let w_sum = w_a + w_b;
                    if (w_sum <= 0.0) {
                        i = i + 1u;
                        continue;
                    }
                    let pa = load_pos(a);
                    let pb = load_pos(b);
                    let delta = pb - pa;
                    let len = length(delta);
                    if (len <= EPS_LEN) {
                        i = i + 1u;
                        continue;
                    }
                    let dir = delta * (1.0 / len);
                    let constraint = len - load_rest(a);
                    let lambda = constraint / (w_sum + params.alpha);
                    store_pos(a, pa + dir * (w_a * lambda));
                    store_pos(b, pb - dir * (w_b * lambda));
                    i = i + 1u;
                }
            }

            // solve_local: pull each interior free particle toward the midpoint
            // of its neighbors (discrete Laplacian bending resistance).
            if (n >= 3u && params.local_stiffness > 0.0) {
                var i: u32 = 1u;
                while (i + 1u < n) {
                    let gi = offset + i;
                    if (load_inv_mass(gi) > 0.0) {
                        let prev_pos = load_pos(offset + i - 1u);
                        let next_pos = load_pos(offset + i + 1u);
                        let pos = load_pos(gi);
                        let midpoint = (prev_pos + next_pos) * 0.5;
                        store_pos(gi, pos + (midpoint - pos) * params.local_stiffness);
                    }
                    i = i + 1u;
                }
            }

            // solve_global: pull each free particle toward its goal pose.
            if (strand.has_goal == 1u && params.global_stiffness > 0.0) {
                var i: u32 = 0u;
                while (i < n) {
                    let gi = offset + i;
                    if (load_inv_mass(gi) > 0.0) {
                        let pos = load_pos(gi);
                        store_pos(gi, pos + (load_goal(gi) - pos) * params.global_stiffness);
                    }
                    i = i + 1u;
                }
            }

            // solve_lra: cap each free particle to the cumulative rest distance
            // from the strand root (one-sided tether).
            if (n >= 2u && params.lra_stiffness > 0.0 && strand.has_rest == 1u) {
                let root = load_pos(offset);
                var max_distance: f32 = 0.0;
                var i: u32 = 1u;
                while (i < n) {
                    max_distance = max_distance + max(load_rest(offset + i - 1u), 0.0);
                    let gi = offset + i;
                    if (load_inv_mass(gi) > 0.0 && max_distance > EPS_LEN) {
                        let pos = load_pos(gi);
                        let delta = pos - root;
                        let distance = length(delta);
                        if (distance > max_distance && distance > EPS_LEN) {
                            let tether = root + delta * (max_distance / distance);
                            store_pos(gi, pos + (tether - pos) * params.lra_stiffness);
                        }
                    }
                    i = i + 1u;
                }
            }
        }

        // resolve_strand_collisions: project free particles out of every
        // collider in order, once per substep after the constraint sweeps.
        if (params.collider_count > 0u) {
            for (var i: u32 = 0u; i < n; i = i + 1u) {
                let gi = offset + i;
                if (load_inv_mass(gi) <= 0.0) {
                    continue;
                }
                var pos = load_pos(gi);
                for (var c: u32 = 0u; c < params.collider_count; c = c + 1u) {
                    pos = push_out(colliders[c], pos);
                }
                store_pos(gi, pos);
            }
        }
    }
}
