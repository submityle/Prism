// Cosserat / discrete-elastic-rod solver twin: advances on-device the same
// position-and-orientation based solve the CPU golden `simulate_guides_cosserat`
// computes, one thread per rod strand.
//
// A pure mass-spring strand (see the guide solver) has no material frame and
// therefore no twist energy; tight curls and braids need the torsional
// stiffness and natural `helix` rest shape only a `Cosserat` rod expresses.
// Each rod is an edge-vector poly-line of particles paired with one quaternion
// material frame per segment. The relative rotation between two adjacent frames
// is a discrete `Darboux` vector (two bending curvatures plus twist); driving
// that relative rotation toward a non-zero rest `Darboux` vector is what
// encodes the curl/twist rest a spring network cannot. This kernel mirrors the
// reference term for term (see `prism_render_architecture::hair::cosserat`):
//
//   for substep in 0..substeps {
//     predict();                         // semi-implicit position prediction
//     for _ in 0..8 {                    // fixed SOLVER_ITERATIONS sweeps
//       solve_stretch();                 // compliant edge-length projection
//       solve_bend_twist();              // XPBD / PB-Cosserat frame projection
//     }
//     finalize();                        // velocity write-back + damping
//   }
//
// Strands are independent poly-lines, so the batch is embarrassingly parallel:
// one thread owns one rod's contiguous particle and orientation ranges and the
// in-place Gauss-Seidel updates are race free without any barrier, matching the
// scalar reference sweep for sweep.
//
// All host-derived scalars (`sub_dt`, `inv_sub_dt = 1 / sub_dt`, `retain =
// 1 - damping`, and the three `*_alpha = compliance / sub_dt^2` values) are
// precomputed on the host in the reference's evaluation order and uploaded in
// `Params`, so the device never re-derives them and cannot diverge on operation
// ordering. The per-strand companion arrays (`rest_lengths`, `rest_darboux`,
// `orientations`) are all-or-nothing sub-slices in the reference
// (`get(offset..end)` is `Some` for the whole strand or `None`), reproduced
// here as a `has_rest` / `has_darboux` flag plus an `o_count` that is `0` when
// the orientation slice does not fit (bend-twist no-op) exactly as the golden.
//
// Every guard is reproduced: a rod of fewer than two particles early-returns,
// pinned particles (`inverse_mass <= 0`) are never predicted and never moved
// and finalize to zero velocity, degenerate (zero-length) edges are skipped,
// and a missing rest `Darboux` entry is treated as a straight `0` rest. The
// kernel re-normalizes every orientation on entry, mirroring the reference's
// per-frame `sanitized()` (= normalize for the finite inputs the parity domain
// supplies); the host guarantees finite inputs, so the kernel omits the
// reference's `is_finite` guards (core WGSL has no `isFinite`).
//
// It uses only the portable core-WGSL subset (sqrt via length / min / max /
// dot / mul / add) - no exp, no pow, no transcendental, no optional feature -
// so this twin runs unmodified on Metal, Vulkan and DX12. It is not bit-exact
// with the scalar reference: a GPU may fuse a multiply-add the reference leaves
// separate, perturbing the low mantissa bits over the iterated solve. The
// parity test therefore asserts a per-component tolerance rather than exact
// equality.
//
// Provenance: standard position-based `Cosserat` rod (compliant edge-length +
// bend-twist Darboux projection), following `Bergou` 2008/2010 and
// `Kugelstadt` 2016 public formulations; no Unreal Engine source or derived
// code.

// Matches `cosserat::EPS_LEN_SQ`: vectors whose squared length is below this
// are treated as zero so normalization never divides by (almost) zero.
const EPS_LEN_SQ: f32 = 1.0e-24;
// Matches `cosserat::EPS_LEN`: segments shorter than this are skipped.
const EPS_LEN: f32 = 1.0e-12;
// Matches `cosserat::SOLVER_ITERATIONS`: fixed Gauss-Seidel sweeps per substep.
const SOLVER_ITERATIONS: u32 = 8u;
// Uniform orientation weight shared by both frames in the bend-twist solve
// (each frame contributes an inverse inertia of one), matching the reference's
// fixed `w_sum = 2.0`.
const ORIENTATION_W_SUM: f32 = 2.0;

// Solve-wide parameters, all host-derived in the reference's evaluation order.
// `32`-byte scalar-packed layout matching the `bytemuck` upload struct `Params`.
struct Params {
    // `params.dt / substeps`: one semi-implicit substep in seconds.
    sub_dt: f32,
    // `1 / sub_dt`: reciprocal substep used by the velocity write-back.
    inv_sub_dt: f32,
    // `1 - clamp(damping, 0, 1)`: velocity-retention fraction.
    retain: f32,
    // `max(stretch_compliance, 0) / sub_dt^2`: XPBD compliance for the stretch.
    stretch_alpha: f32,
    // `max(bend_compliance, 0) / sub_dt^2`: XPBD compliance for the two bends.
    bend_alpha: f32,
    // `max(twist_compliance, 0) / sub_dt^2`: XPBD compliance for the twist.
    twist_alpha: f32,
    // Number of semi-implicit substeps.
    substeps: u32,
    // Number of rod descriptors in `strands`.
    strand_count: u32,
}

// One rod descriptor. `32`-byte stride matching the `bytemuck` upload struct
// `Strand`: the base particle index of this rod in the flat `state` pool and
// its particle count, the base orientation index in `orient` and its
// orientation count (`0` when the orientation slice does not fit, disabling the
// bend-twist solve for the whole rod), the base rest-`Darboux` index in `rd`
// and its count, and whether the rest-length / rest-`Darboux` companion slices
// are present (all-or-nothing per strand).
struct Strand {
    p_offset: u32,
    p_count: u32,
    o_offset: u32,
    o_count: u32,
    rd_offset: u32,
    rd_count: u32,
    has_rest: u32,
    has_darboux: u32,
}

// A quaternion material frame stored as `w + xi + yj + zk`, matching
// `cosserat::Quat`.
struct Q {
    w: f32,
    x: f32,
    y: f32,
    z: f32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> strands: array<Strand>;
// Per-particle state, stride `8` f32: position xyz, velocity xyz, inverse_mass,
// rest_length (the target length of the segment leaving this particle; `0` on
// the tip particle and on any particle whose rod has no rest lengths).
@group(0) @binding(2) var<storage, read_write> state: array<f32>;
// Per-segment orientation, stride `4` f32: quaternion `w, x, y, z`.
@group(0) @binding(3) var<storage, read_write> orient: array<f32>;
// Per-particle scratch previous position, stride `3` f32. Written by predict
// before it is read by finalize, so it needs no zero initialization.
@group(0) @binding(4) var<storage, read_write> prev: array<f32>;
// Flat rest-`Darboux` vectors, stride `3` f32.
@group(0) @binding(5) var<storage, read> rd: array<f32>;

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

fn load_vel(gi: u32) -> vec3<f32> {
    let b = 8u * gi;
    return vec3<f32>(state[b + 3u], state[b + 4u], state[b + 5u]);
}

fn store_vel(gi: u32, v: vec3<f32>) {
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

fn load_prev(gi: u32) -> vec3<f32> {
    let b = 3u * gi;
    return vec3<f32>(prev[b], prev[b + 1u], prev[b + 2u]);
}

fn store_prev(gi: u32, v: vec3<f32>) {
    let b = 3u * gi;
    prev[b] = v.x;
    prev[b + 1u] = v.y;
    prev[b + 2u] = v.z;
}

fn load_quat(oi: u32) -> Q {
    let b = 4u * oi;
    return Q(orient[b], orient[b + 1u], orient[b + 2u], orient[b + 3u]);
}

fn store_quat(oi: u32, q: Q) {
    let b = 4u * oi;
    orient[b] = q.w;
    orient[b + 1u] = q.x;
    orient[b + 2u] = q.y;
    orient[b + 3u] = q.z;
}

fn load_rd(ri: u32) -> vec3<f32> {
    let b = 3u * ri;
    return vec3<f32>(rd[b], rd[b + 1u], rd[b + 2u]);
}

// Hamilton product `a * b`, mirroring `Quat::mul` component for component.
fn quat_mul(a: Q, b: Q) -> Q {
    return Q(
        a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z,
        a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y,
        a.w * b.y - a.x * b.z + a.y * b.w + a.z * b.x,
        a.w * b.z + a.x * b.y - a.y * b.x + a.z * b.w,
    );
}

// Conjugate `w - xi - yj - zk`, mirroring `Quat::conjugate`.
fn quat_conj(q: Q) -> Q {
    return Q(q.w, -q.x, -q.y, -q.z);
}

fn quat_add(a: Q, b: Q) -> Q {
    return Q(a.w + b.w, a.x + b.x, a.y + b.y, a.z + b.z);
}

fn quat_sub(a: Q, b: Q) -> Q {
    return Q(a.w - b.w, a.x - b.x, a.y - b.y, a.z - b.z);
}

// Unit quaternion along `q`, falling back to the identity when `q` is
// degenerate, mirroring `Quat::normalize` (the finite guard is dropped because
// the host supplies finite inputs and core WGSL has no `isFinite`).
fn quat_normalize(q: Q) -> Q {
    let n2 = q.w * q.w + q.x * q.x + q.y * q.y + q.z * q.z;
    if (n2 > EPS_LEN_SQ) {
        let s = 1.0 / sqrt(n2);
        return Q(q.w * s, q.x * s, q.y * s, q.z * s);
    }
    return Q(1.0, 0.0, 0.0, 0.0);
}

// Discrete `Darboux` vector: imaginary part of `conj(a) * b`, mirroring
// `cosserat::darboux`.
fn darboux(a: Q, b: Q) -> vec3<f32> {
    let rel = quat_mul(quat_conj(a), b);
    return vec3<f32>(rel.x, rel.y, rel.z);
}

@compute @workgroup_size(64)
fn evaluate(@builtin(global_invocation_id) gid: vec3<u32>) {
    let s = gid.x;
    if (s >= params.strand_count) {
        return;
    }
    let strand = strands[s];
    let p_off = strand.p_offset;
    let p_count = strand.p_count;
    if (p_count < 2u) {
        return;
    }
    let o_off = strand.o_offset;
    let o_count = strand.o_count;
    let rd_off = strand.rd_offset;

    // Sanitize incoming orientations by re-normalizing them, mirroring the
    // reference's per-frame `q.sanitized()` for finite inputs.
    for (var j: u32 = 0u; j < o_count; j = j + 1u) {
        store_quat(o_off + j, quat_normalize(load_quat(o_off + j)));
    }

    for (var sub: u32 = 0u; sub < params.substeps; sub = sub + 1u) {
        // predict: record each particle's start position and advance free
        // particles by velocity * sub_dt. Pinned particles keep their position.
        for (var i: u32 = 0u; i < p_count; i = i + 1u) {
            let gi = p_off + i;
            let pos = load_pos(gi);
            store_prev(gi, pos);
            if (load_inv_mass(gi) > 0.0) {
                store_pos(gi, pos + load_vel(gi) * params.sub_dt);
            }
        }

        for (var it: u32 = 0u; it < SOLVER_ITERATIONS; it = it + 1u) {
            // solve_stretch: compliant edge-length constraint per segment.
            if (strand.has_rest == 1u) {
                var i: u32 = 0u;
                while (i + 1u < p_count) {
                    let a = p_off + i;
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
                    let lambda = constraint / (w_sum + params.stretch_alpha);
                    store_pos(a, pa + dir * (w_a * lambda));
                    store_pos(b, pb - dir * (w_b * lambda));
                    i = i + 1u;
                }
            }

            // solve_bend_twist: drive each adjacent frame pair toward its rest
            // Darboux vector. A missing rest entry is a straight 0 rest.
            var k: u32 = 0u;
            while (k + 1u < o_count) {
                let oi = o_off + k;
                let oj = oi + 1u;
                var rest = vec3<f32>(0.0, 0.0, 0.0);
                if (strand.has_darboux == 1u && k < strand.rd_count) {
                    rest = load_rd(rd_off + k);
                }
                let qi = load_quat(oi);
                let qj = load_quat(oj);
                let omega = darboux(qi, qj);
                let c = omega - rest;
                let lambda = Q(
                    0.0,
                    c.x / (ORIENTATION_W_SUM + params.bend_alpha),
                    c.y / (ORIENTATION_W_SUM + params.bend_alpha),
                    c.z / (ORIENTATION_W_SUM + params.twist_alpha),
                );
                let dqi = quat_mul(qj, lambda);
                let dqj = quat_mul(qi, lambda);
                store_quat(oi, quat_normalize(quat_add(qi, dqi)));
                store_quat(oj, quat_normalize(quat_sub(qj, dqj)));
                k = k + 1u;
            }
        }

        // finalize: velocity write-back from the position change scaled by the
        // damping retention. Pinned particles are forced to zero velocity.
        for (var i: u32 = 0u; i < p_count; i = i + 1u) {
            let gi = p_off + i;
            if (load_inv_mass(gi) <= 0.0) {
                store_vel(gi, vec3<f32>(0.0, 0.0, 0.0));
                continue;
            }
            let v = (load_pos(gi) - load_prev(gi)) * params.inv_sub_dt;
            store_vel(gi, v * params.retain);
        }
    }
}
