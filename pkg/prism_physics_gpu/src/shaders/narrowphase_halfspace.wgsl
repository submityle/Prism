// Sphere-halfspace narrow-phase contact kernel.
//
// One invocation per (sphere, plane) candidate couple. Each invocation reads
// the sphere's bounding sphere and the plane, runs the same overlap test and
// manifold construction the CPU twin runs (`narrowphase/halfspace.rs`), and
// writes one contact slot: the plane's outward normal, the penetration depth,
// the surface contact point, and a validity flag. A couple whose sphere floats
// clear writes a zeroed slot with valid = 0, so the output index stays aligned
// with the input couple index.
//
// The arithmetic is bit-for-bit with the twin: the signed distance is a dot
// product and a subtraction, the depth a subtraction, and the contact point a
// scaled subtraction, with no square root or reciprocal on this path, so the
// parity test matches the validity flag exactly and the normal, depth, and
// point to within the tightest float tolerance.
//
// Provenance: textbook sphere-plane collision manifold. No Unreal Engine
// source or derived code.

struct Params {
    // Number of candidate couples queued in the pair buffer.
    num_pairs: u32,
    // Padding to a 16-byte uniform stride.
    pad0: u32,
    pad1: u32,
    pad2: u32,
};

// One output contact: (normal.xyz, depth) and (point.xyz, valid).
struct Contact {
    normal_depth: vec4<f32>,
    point_valid: vec4<f32>,
};

@group(0) @binding(0) var<uniform> params: Params;
// Sphere particles in original order: xyz centre, w radius.
@group(0) @binding(1) var<storage, read> spheres: array<vec4<f32>>;
// Planes: xyz outward unit normal, w offset.
@group(0) @binding(2) var<storage, read> planes: array<vec4<f32>>;
// Candidate couples, one (sphere, plane) index pair each.
@group(0) @binding(3) var<storage, read> pairs: array<vec2<u32>>;
// Output contacts, one slot per couple.
@group(0) @binding(4) var<storage, read_write> contacts: array<Contact>;

@compute @workgroup_size(64)
fn narrowphase_halfspace(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.num_pairs) {
        return;
    }

    let couple = pairs[i];
    let sphere = spheres[couple.x];
    let plane = planes[couple.y];
    let c = sphere.xyz;
    let r = sphere.w;
    let n = plane.xyz;
    let d = plane.w;

    // Signed distance of the centre from the surface along the outward normal.
    let s = dot(n, c) - d;

    var out: Contact;
    // Strict overlap: a centre exactly one radius clear only grazes the surface.
    if (s >= r) {
        out.normal_depth = vec4<f32>(0.0, 0.0, 0.0, 0.0);
        out.point_valid = vec4<f32>(0.0, 0.0, 0.0, 0.0);
        contacts[i] = out;
        return;
    }

    let depth = r - s;
    let point = c - n * s;
    out.normal_depth = vec4<f32>(n, depth);
    out.point_valid = vec4<f32>(point, 1.0);
    contacts[i] = out;
}
