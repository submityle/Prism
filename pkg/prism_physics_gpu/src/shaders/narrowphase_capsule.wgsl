// Sphere-capsule narrow-phase contact kernel.
//
// One invocation per candidate pair. Each invocation reads a sphere and a
// capsule (segment + radius), runs the same closest-point-on-segment test and
// manifold construction the CPU twin runs (`narrowphase/capsule.rs`), and writes
// one contact slot: a unit normal from the capsule (b) to the sphere (a), the
// penetration depth, the mid-overlap contact point, and a validity flag. A pair
// that does not penetrate writes a zeroed slot with valid = 0, so the output
// index stays aligned with the input pair index.
//
// The arithmetic is bit-for-bit with the twin apart from the square root and the
// reciprocal in the normalisation, whose WGSL rounding differs in the low bits;
// the parity test therefore matches the validity flag exactly and the normal,
// depth, and point within a tight tolerance.
//
// Provenance: textbook sphere-capsule (segment-point) closest-feature collision.
// No Unreal Engine source or derived code.

struct Params {
    // Number of candidate pairs queued in the pair buffer.
    num_pairs: u32,
    // Padding to a 16-byte uniform stride.
    pad0: u32,
    pad1: u32,
    pad2: u32,
};

// Squared-length threshold below which a capsule segment is a single point;
// kept identical to `SEG_EPS2` in `narrowphase/capsule.rs`.
const SEG_EPS2: f32 = 1.0e-12;

// Squared-distance threshold below which the sphere centre is treated as lying
// on the segment; kept identical to `COINCIDENT_EPS2` in the same module.
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

@group(0) @binding(0) var<uniform> params: Params;
// Sphere proxies in original order: xyz centre, w radius.
@group(0) @binding(1) var<storage, read> spheres: array<vec4<f32>>;
// Capsule proxies, one segment + radius each.
@group(0) @binding(2) var<storage, read> capsules: array<Capsule>;
// Candidate pairs, one (sphere, capsule) index couple each.
@group(0) @binding(3) var<storage, read> pairs: array<vec2<u32>>;
// Output contacts, one slot per pair.
@group(0) @binding(4) var<storage, read_write> contacts: array<Contact>;

// Closest point on the capsule segment to the sphere centre `c`.
fn closest_on_segment(c: vec3<f32>, p0: vec3<f32>, p1: vec3<f32>) -> vec3<f32> {
    let ab = p1 - p0;
    let ab_len2 = dot(ab, ab);
    if (ab_len2 <= SEG_EPS2) {
        // Degenerate capsule: the segment is a single point.
        return p0;
    }
    let t = clamp(dot(c - p0, ab) / ab_len2, 0.0, 1.0);
    return p0 + ab * t;
}

@compute @workgroup_size(64)
fn narrowphase_capsule(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.num_pairs) {
        return;
    }

    let pair = pairs[i];
    let s = spheres[pair.x];
    let cap = capsules[pair.y];
    let c = s.xyz;
    let rs = s.w;
    let p0 = cap.p0_radius.xyz;
    let rc = cap.p0_radius.w;
    let p1 = cap.p1_pad.xyz;

    let q = closest_on_segment(c, p0, p1);
    let delta = c - q;
    let dist2 = dot(delta, delta);
    let sum_r = rs + rc;

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
        // Sphere centre on the segment: a stable axis instead of a divide by zero.
        normal = vec3<f32>(1.0, 0.0, 0.0);
        dist = 0.0;
    } else {
        dist = sqrt(dist2);
        normal = delta / dist;
    }

    let depth = sum_r - dist;
    let point = c - normal * (rs - depth * 0.5);
    out.normal_depth = vec4<f32>(normal, depth);
    out.point_valid = vec4<f32>(point, 1.0);
    contacts[i] = out;
}
