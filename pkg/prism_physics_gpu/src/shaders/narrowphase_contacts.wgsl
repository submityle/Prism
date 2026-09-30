// Sphere-sphere narrow-phase contact kernel.
//
// One invocation per candidate pair. Each invocation reads the two particles'
// bounding spheres, runs the same overlap test and manifold construction the
// CPU twin runs (`narrowphase/sphere.rs`), and writes one contact slot: a unit
// normal from a to b, the penetration depth, the mid-overlap contact point, and
// a validity flag. A pair whose spheres do not penetrate writes a zeroed slot
// with valid = 0, so the output index stays aligned with the input pair index.
//
// The arithmetic is bit-for-bit with the twin apart from the square root and
// the reciprocal in the normalisation, whose WGSL rounding differs in the low
// bits; the parity test therefore matches the validity flag exactly and the
// normal, depth, and point within a tight tolerance.
//
// Provenance: textbook sphere-sphere collision manifold. No Unreal Engine
// source or derived code.

struct Params {
    // Number of candidate pairs queued in the pair buffer.
    num_pairs: u32,
    // Padding to a 16-byte uniform stride.
    pad0: u32,
    pad1: u32,
    pad2: u32,
};

// Squared-distance threshold below which centres are treated as coincident;
// kept identical to `COINCIDENT_EPS2` in `narrowphase/sphere.rs`.
const COINCIDENT_EPS2: f32 = 1.0e-12;

// One output contact: (normal.xyz, depth) and (point.xyz, valid).
struct Contact {
    normal_depth: vec4<f32>,
    point_valid: vec4<f32>,
};

@group(0) @binding(0) var<uniform> params: Params;
// Particle bounding spheres in original order: xyz centre, w radius.
@group(0) @binding(1) var<storage, read> particles: array<vec4<f32>>;
// Candidate pairs, one (a, b) particle-index couple each.
@group(0) @binding(2) var<storage, read> pairs: array<vec2<u32>>;
// Output contacts, one slot per pair.
@group(0) @binding(3) var<storage, read_write> contacts: array<Contact>;

@compute @workgroup_size(64)
fn narrowphase_contacts(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.num_pairs) {
        return;
    }

    let pair = pairs[i];
    let sa = particles[pair.x];
    let sb = particles[pair.y];
    let pa = sa.xyz;
    let ra = sa.w;
    let pb = sb.xyz;
    let rb = sb.w;

    let delta = pb - pa;
    let dist2 = dot(delta, delta);
    let sum_r = ra + rb;

    var out: Contact;
    // Strict overlap: exactly-touching spheres carry no penetration.
    if (dist2 >= sum_r * sum_r) {
        out.normal_depth = vec4<f32>(0.0, 0.0, 0.0, 0.0);
        out.point_valid = vec4<f32>(0.0, 0.0, 0.0, 0.0);
        contacts[i] = out;
        return;
    }

    var normal: vec3<f32>;
    var dist: f32;
    if (dist2 <= COINCIDENT_EPS2) {
        // Coincident centres: a stable axis instead of a divide by zero.
        normal = vec3<f32>(1.0, 0.0, 0.0);
        dist = 0.0;
    } else {
        dist = sqrt(dist2);
        normal = delta / dist;
    }

    let depth = sum_r - dist;
    let point = pa + normal * (ra - depth * 0.5);
    out.normal_depth = vec4<f32>(normal, depth);
    out.point_valid = vec4<f32>(point, 1.0);
    contacts[i] = out;
}
