// Capsule-capsule two-point contact manifold kernel.
//
// One invocation per candidate pair. Each invocation runs the same single
// closest-feature test as the single-point capsule-capsule kernel, then, when
// the two capsule axes are near-parallel and overlap, clips the overlapping
// stretch of their axes and emits a two-point manifold spanning it. Every other
// penetrating pair emits the single deepest contact. A pair that does not
// penetrate writes a zeroed slot with count = 0, so the output index stays
// aligned with the input pair index.
//
// The arithmetic is bit-for-bit with the CPU twin
// (`narrowphase/capsule_capsule_manifold.rs`) apart from the square roots and
// the reciprocals in the normalisations, whose WGSL rounding differs in the low
// bits; the parity test therefore matches the point count exactly and the
// normal, depths, and positions within a tight tolerance.
//
// Provenance: textbook capsule-capsule manifold (segment-segment closest feature
// plus parallel-overlap interval clipping). No Unreal Engine source or derived
// code.

struct Params {
    // Number of candidate pairs queued in the pair buffer.
    num_pairs: u32,
    // Padding to a 16-byte uniform stride.
    pad0: u32,
    pad1: u32,
    pad2: u32,
};

// Squared-length threshold below which a capsule segment is a degenerate point
// (a sphere); kept identical to `SEG_EPS2` in the CPU twin.
const SEG_EPS2: f32 = 1.0e-12;

// sin^2(theta) threshold below which two axes count as parallel; kept identical
// to `PARALLEL_SIN2` in the CPU twin.
const PARALLEL_SIN2: f32 = 1.2e-3;

// World-length threshold below which the axis overlap is a single point; kept
// identical to `OVERLAP_EPS` in the CPU twin.
const OVERLAP_EPS: f32 = 1.0e-6;

// Squared-distance threshold below which closest points are coincident; kept
// identical to `COINCIDENT_EPS2` in the CPU twin.
const COINCIDENT_EPS2: f32 = 1.0e-12;

// Squared-distance threshold below which the two overlap-end corners coincide;
// kept identical to `SEP_EPS2` in the CPU twin.
const SEP_EPS2: f32 = 1.0e-12;

// One capsule: (p0.xyz, radius) then (p1.xyz, unused pad).
struct Capsule {
    p0_radius: vec4<f32>,
    p1_pad: vec4<f32>,
};

// One output manifold: (normal.xyz, count) then four (position.xyz, depth).
struct Manifold {
    normal_count: vec4<f32>,
    p0: vec4<f32>,
    p1: vec4<f32>,
    p2: vec4<f32>,
    p3: vec4<f32>,
};

// A closest-point pair between two segments.
struct ClosestPair {
    ca: vec3<f32>,
    cb: vec3<f32>,
};

// The single deepest contact: shared normal, mid-overlap point, depth, validity.
struct SingleContact {
    normal: vec3<f32>,
    point: vec3<f32>,
    depth: f32,
    valid: bool,
};

// One overlap-end corner: its world position, penetration depth, and liveness.
struct EndPoint {
    position: vec3<f32>,
    depth: f32,
    live: bool,
};

@group(0) @binding(0) var<uniform> params: Params;
// Capsule proxies, one segment + radius each.
@group(0) @binding(1) var<storage, read> capsules: array<Capsule>;
// Candidate pairs, one (a, b) capsule-index couple each.
@group(0) @binding(2) var<storage, read> pairs: array<vec2<u32>>;
// Output manifolds, one slot per pair.
@group(0) @binding(3) var<storage, read_write> manifolds: array<Manifold>;

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
        s = 0.0;
        t = 0.0;
    } else if (a <= SEG_EPS2) {
        s = 0.0;
        t = clamp(f / e, 0.0, 1.0);
    } else {
        let c = dot(d1, r);
        if (e <= SEG_EPS2) {
            t = 0.0;
            s = clamp(-c / a, 0.0, 1.0);
        } else {
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

// Single deepest capsule-capsule contact, mirroring `capsule_capsule_contact`.
fn capsule_capsule_single(a_p0: vec3<f32>, a_p1: vec3<f32>, ra: f32, b_p0: vec3<f32>, b_p1: vec3<f32>, rb: f32) -> SingleContact {
    let closest = closest_pt_segment_segment(a_p0, a_p1, b_p0, b_p1);
    let ca = closest.ca;
    let cb = closest.cb;
    let delta = cb - ca;
    let dist2 = dot(delta, delta);
    let sum_r = ra + rb;

    var out: SingleContact;
    if (dist2 >= sum_r * sum_r) {
        out.normal = vec3<f32>(0.0, 0.0, 0.0);
        out.point = vec3<f32>(0.0, 0.0, 0.0);
        out.depth = 0.0;
        out.valid = false;
        return out;
    }

    var normal: vec3<f32>;
    var dist: f32;
    if (dist2 <= COINCIDENT_EPS2) {
        normal = vec3<f32>(1.0, 0.0, 0.0);
        dist = 0.0;
    } else {
        dist = sqrt(dist2);
        normal = delta / dist;
    }
    let depth = sum_r - dist;
    out.normal = normal;
    out.point = ca + normal * (ra - depth * 0.5);
    out.depth = depth;
    out.valid = true;
    return out;
}

// Builds the manifold corner at axis parameter s on capsule a's axis, mirroring
// `overlap_end_point`.
fn overlap_end_point(s: f32, a0: vec3<f32>, ua: vec3<f32>, b0: vec3<f32>, ub: vec3<f32>, lb: f32, ra: f32, sum_r: f32, normal: vec3<f32>) -> EndPoint {
    let pa = a0 + ua * s;
    let tb = clamp(dot(pa - b0, ub), 0.0, lb);
    let pb = b0 + ub * tb;
    let delta = pb - pa;
    let d2 = dot(delta, delta);
    var dist: f32 = 0.0;
    if (d2 > COINCIDENT_EPS2) {
        dist = sqrt(d2);
    }
    let depth = sum_r - dist;

    var out: EndPoint;
    out.depth = depth;
    out.live = depth > 0.0;
    out.position = pa + normal * (ra - depth * 0.5);
    return out;
}

// A one-point manifold carrying the single deepest contact.
fn single_manifold(contact: SingleContact) -> Manifold {
    var m: Manifold;
    m.normal_count = vec4<f32>(contact.normal, 1.0);
    m.p0 = vec4<f32>(contact.point, contact.depth);
    m.p1 = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    m.p2 = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    m.p3 = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    return m;
}

@compute @workgroup_size(64)
fn narrowphase_capsule_capsule_manifold(@builtin(global_invocation_id) gid: vec3<u32>) {
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

    let contact = capsule_capsule_single(a_p0, a_p1, ra, b_p0, b_p1, rb);

    var m: Manifold;
    if (!contact.valid) {
        // No penetration: a zeroed slot with count 0.
        m.normal_count = vec4<f32>(0.0, 0.0, 0.0, 0.0);
        m.p0 = vec4<f32>(0.0, 0.0, 0.0, 0.0);
        m.p1 = vec4<f32>(0.0, 0.0, 0.0, 0.0);
        m.p2 = vec4<f32>(0.0, 0.0, 0.0, 0.0);
        m.p3 = vec4<f32>(0.0, 0.0, 0.0, 0.0);
        manifolds[i] = m;
        return;
    }

    let da = a_p1 - a_p0;
    let db = b_p1 - b_p0;
    let la2 = dot(da, da);
    let lb2 = dot(db, db);
    // A zero-length capsule is a sphere: one contact point only.
    if (la2 <= SEG_EPS2 || lb2 <= SEG_EPS2) {
        manifolds[i] = single_manifold(contact);
        return;
    }

    // Parallel test without normalisation.
    let crs = cross(da, db);
    if (dot(crs, crs) > PARALLEL_SIN2 * la2 * lb2) {
        manifolds[i] = single_manifold(contact);
        return;
    }

    let la = sqrt(la2);
    let lb = sqrt(lb2);
    let ua = da / la;
    let ub = db / lb;

    let sb0 = dot(b_p0 - a_p0, ua);
    let sb1 = dot(b_p1 - a_p0, ua);
    let lo = min(sb0, sb1);
    let hi = max(sb0, sb1);
    let ov_lo = max(0.0, lo);
    let ov_hi = min(la, hi);
    if (ov_hi - ov_lo <= OVERLAP_EPS) {
        manifolds[i] = single_manifold(contact);
        return;
    }

    let sum_r = ra + rb;
    let end0 = overlap_end_point(ov_lo, a_p0, ua, b_p0, ub, lb, ra, sum_r, contact.normal);
    let end1 = overlap_end_point(ov_hi, a_p0, ua, b_p0, ub, lb, ra, sum_r, contact.normal);

    if (end0.live && end1.live) {
        let sep = end1.position - end0.position;
        if (dot(sep, sep) <= SEP_EPS2) {
            manifolds[i] = single_manifold(contact);
            return;
        }
        m.normal_count = vec4<f32>(contact.normal, 2.0);
        m.p0 = vec4<f32>(end0.position, end0.depth);
        m.p1 = vec4<f32>(end1.position, end1.depth);
        m.p2 = vec4<f32>(0.0, 0.0, 0.0, 0.0);
        m.p3 = vec4<f32>(0.0, 0.0, 0.0, 0.0);
        manifolds[i] = m;
        return;
    }

    // Fewer than two ends penetrate: the single deepest contact is honest.
    manifolds[i] = single_manifold(contact);
}
