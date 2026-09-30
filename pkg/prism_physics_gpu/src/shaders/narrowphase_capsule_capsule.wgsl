// Capsule-capsule narrow-phase contact kernel.
//
// One invocation per candidate pair. Each invocation reads two capsules
// (segment + radius), finds the closest point pair between the two segments with
// the same clamped ClosestPtSegmentSegment routine the CPU twin runs
// (`narrowphase/capsule_capsule.rs`), collapses to a sphere-sphere test at those
// points, and writes one contact slot: a unit normal from capsule a to capsule
// b, the penetration depth, the mid-overlap contact point, and a validity flag.
// A pair that does not penetrate writes a zeroed slot with valid = 0, so the
// output index stays aligned with the input pair index.
//
// The arithmetic is bit-for-bit with the twin apart from the square root and the
// reciprocal in the normalisation, whose WGSL rounding differs in the low bits;
// the parity test therefore matches the validity flag exactly and the normal,
// depth, and point within a tight tolerance.
//
// Provenance: textbook capsule-capsule (segment-segment) closest-feature
// collision. No Unreal Engine source or derived code.

struct Params {
    // Number of candidate pairs queued in the pair buffer.
    num_pairs: u32,
    // Padding to a 16-byte uniform stride.
    pad0: u32,
    pad1: u32,
    pad2: u32,
};

// Squared-length threshold below which a segment or solver denominator is
// degenerate; kept identical to `SEG_EPS2` in `narrowphase/capsule_capsule.rs`.
const SEG_EPS2: f32 = 1.0e-12;

// Squared-distance threshold below which the closest points are coincident;
// kept identical to `COINCIDENT_EPS2` in the same module.
const COINCIDENT_EPS2: f32 = 1.0e-12;

// One capsule: (p0.xyz, radius) then (p1.xyz, unused pad).
struct Capsule {
    p0_radius: vec4<f32>,
    p1_pad: vec4<f32>,
};

// One output contact: (normal.xyz, depth) and (point.xyz, valid).
struct Contact {
    normal_depth: vec4<f32>,
    point_valid: vec4<f32>,
};

// A closest-point pair between two segments.
struct ClosestPair {
    ca: vec3<f32>,
    cb: vec3<f32>,
};

@group(0) @binding(0) var<uniform> params: Params;
// Capsule proxies, one segment + radius each.
@group(0) @binding(1) var<storage, read> capsules: array<Capsule>;
// Candidate pairs, one (a, b) capsule-index couple each.
@group(0) @binding(2) var<storage, read> pairs: array<vec2<u32>>;
// Output contacts, one slot per pair.
@group(0) @binding(3) var<storage, read_write> contacts: array<Contact>;

// Closest point pair between segment (p1, q1) and segment (p2, q2). Every
// division is guarded by SEG_EPS2 so parallel and degenerate segments never
// divide by zero.
fn closest_pt_segment_segment(p1: vec3<f32>, q1: vec3<f32>, p2: vec3<f32>, q2: vec3<f32>) -> ClosestPair {
    let d1 = q1 - p1;
    let d2 = q2 - p2;
    let r = p1 - p2;
    let a = dot(d1, d1);
    let e = dot(d2, d2);
    let f = dot(d2, r);

    var s: f32 = 0.0;
    var t: f32 = 0.0;
    if (a <= SEG_EPS2 && e <= SEG_EPS2) {
        // Both segments degenerate to points.
        s = 0.0;
        t = 0.0;
    } else if (a <= SEG_EPS2) {
        // First segment degenerate: project its point onto the second.
        s = 0.0;
        t = clamp(f / e, 0.0, 1.0);
    } else {
        let c = dot(d1, r);
        if (e <= SEG_EPS2) {
            // Second segment degenerate: project its point onto the first.
            t = 0.0;
            s = clamp(-c / a, 0.0, 1.0);
        } else {
            // General case: solve the 2x2 system, then clamp t and recompute s.
            let b = dot(d1, d2);
            let denom = a * e - b * b;
            var s0: f32 = 0.0;
            if (denom > SEG_EPS2) {
                s0 = clamp((b * f - c * e) / denom, 0.0, 1.0);
            }
            let t0 = (b * s0 + f) / e;
            if (t0 < 0.0) {
                t = 0.0;
                s = clamp(-c / a, 0.0, 1.0);
            } else if (t0 > 1.0) {
                t = 1.0;
                s = clamp((b - c) / a, 0.0, 1.0);
            } else {
                t = t0;
                s = s0;
            }
        }
    }

    var out_pair: ClosestPair;
    out_pair.ca = p1 + d1 * s;
    out_pair.cb = p2 + d2 * t;
    return out_pair;
}

@compute @workgroup_size(64)
fn narrowphase_capsule_capsule(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.num_pairs) {
        return;
    }

    let pair = pairs[i];
    let cap_a = capsules[pair.x];
    let cap_b = capsules[pair.y];
    let a_p0 = cap_a.p0_radius.xyz;
    let ra = cap_a.p0_radius.w;
    let a_p1 = cap_a.p1_pad.xyz;
    let b_p0 = cap_b.p0_radius.xyz;
    let rb = cap_b.p0_radius.w;
    let b_p1 = cap_b.p1_pad.xyz;

    let closest = closest_pt_segment_segment(a_p0, a_p1, b_p0, b_p1);
    let ca = closest.ca;
    let cb = closest.cb;
    let delta = cb - ca;
    let dist2 = dot(delta, delta);
    let sum_r = ra + rb;

    var out: Contact;
    // Strict overlap: an exactly-touching pair carries no penetration.
    if (dist2 >= sum_r * sum_r) {
        out.normal_depth = vec4<f32>(0.0, 0.0, 0.0, 0.0);
        out.point_valid = vec4<f32>(0.0, 0.0, 0.0, 0.0);
        contacts[i] = out;
        return;
    }

    var normal: vec3<f32>;
    var dist: f32;
    if (dist2 <= COINCIDENT_EPS2) {
        // Coincident closest points: a stable axis instead of a divide by zero.
        normal = vec3<f32>(1.0, 0.0, 0.0);
        dist = 0.0;
    } else {
        dist = sqrt(dist2);
        normal = delta / dist;
    }

    let depth = sum_r - dist;
    let point = ca + normal * (ra - depth * 0.5);
    out.normal_depth = vec4<f32>(normal, depth);
    out.point_valid = vec4<f32>(point, 1.0);
    contacts[i] = out;
}
