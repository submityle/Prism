// Sphere-versus-OBB narrow-phase contact kernel.
//
// One invocation per candidate pair. Each invocation reads a sphere (centre and
// radius) and an oriented bounding box (centre, three orthonormal local axes,
// and per-axis half extents), runs the same overlap test and manifold
// construction the CPU twin runs (`narrowphase/obb.rs`), and writes one contact
// slot: a unit normal pointing from the box surface toward the sphere (the
// push-out direction), the penetration depth, the world contact point on the
// box surface, and a validity flag. A pair that does not penetrate writes a
// zeroed slot with valid = 0, so the output index stays aligned with the input
// pair index.
//
// Geometry: project the sphere centre into the box frame, clamp it to the box,
// and inspect the offset `diff` from the clamped point to the centre. When the
// centre is outside (`dot(diff, diff) > INSIDE_EPS2`) the pair contacts only for
// a strict `dist < r`; the local normal is `diff / dist` and depth `r - dist`.
// When the centre is inside, it exits through the least-penetrated face `k`
// (ties resolve to the lower axis index), the local normal is `sign(local[k])`
// along that axis, and depth is `r + (he[k] - abs(local[k]))`. The world normal
// and contact point recombine the local vectors through the box axes.
//
// The arithmetic is bit-for-bit with the twin apart from the square root and
// the reciprocal in the outside-face normalisation, whose WGSL rounding differs
// in the low bits; the parity test therefore matches the validity flag exactly
// and the normal, depth, and point within a tight tolerance.
//
// Provenance: textbook sphere-versus-oriented-bounding-box collision manifold.
// No Unreal Engine source or derived code.

struct Params {
    // Number of candidate pairs queued in the pair buffer.
    num_pairs: u32,
    // Padding to a 16-byte uniform stride.
    pad0: u32,
    pad1: u32,
    pad2: u32,
};

// Squared-length threshold below which the clamped offset is treated as zero,
// i.e. the sphere centre is inside the box; kept identical to `INSIDE_EPS2` in
// `narrowphase/obb.rs`.
const INSIDE_EPS2: f32 = 1.0e-12;

// One oriented bounding box. The half extents ride in the w lanes of the axis
// rows to keep the box to four vec4s; `center.w` is unused padding.
struct Obb {
    center: vec4<f32>,
    axis0: vec4<f32>,
    axis1: vec4<f32>,
    axis2: vec4<f32>,
};

// One output contact: (normal.xyz, depth) and (point.xyz, valid).
struct Contact {
    normal_depth: vec4<f32>,
    point_valid: vec4<f32>,
};

@group(0) @binding(0) var<uniform> params: Params;
// Spheres in original order: xyz centre, w radius.
@group(0) @binding(1) var<storage, read> spheres: array<vec4<f32>>;
// Oriented bounding boxes, four vec4s each.
@group(0) @binding(2) var<storage, read> boxes: array<Obb>;
// Candidate pairs, one (sphere, box) index couple each.
@group(0) @binding(3) var<storage, read> pairs: array<vec2<u32>>;
// Output contacts, one slot per pair.
@group(0) @binding(4) var<storage, read_write> contacts: array<Contact>;

@compute @workgroup_size(64)
fn narrowphase_obb(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.num_pairs) {
        return;
    }

    let pair = pairs[i];
    let sphere = spheres[pair.x];
    let c = sphere.xyz;
    let r = sphere.w;

    let box_ = boxes[pair.y];
    let bc = box_.center.xyz;
    let a0 = box_.axis0.xyz;
    let a1 = box_.axis1.xyz;
    let a2 = box_.axis2.xyz;
    let he = vec3<f32>(box_.axis0.w, box_.axis1.w, box_.axis2.w);

    // Project the sphere centre into the box frame and clamp to the box.
    let d = c - bc;
    let local = vec3<f32>(dot(d, a0), dot(d, a1), dot(d, a2));
    let q = clamp(local, -he, he);
    let diff = local - q;
    let d2 = dot(diff, diff);

    var out: Contact;

    if (d2 > INSIDE_EPS2) {
        // Centre outside the box: nearest feature is the clamped point.
        let dist = sqrt(d2);
        if (dist >= r) {
            // Strict overlap: a grazing sphere carries no penetration.
            out.normal_depth = vec4<f32>(0.0, 0.0, 0.0, 0.0);
            out.point_valid = vec4<f32>(0.0, 0.0, 0.0, 0.0);
            contacts[i] = out;
            return;
        }
        let n_local = diff / dist;
        let normal = a0 * n_local.x + a1 * n_local.y + a2 * n_local.z;
        let depth = r - dist;
        let point = bc + a0 * q.x + a1 * q.y + a2 * q.z;
        out.normal_depth = vec4<f32>(normal, depth);
        out.point_valid = vec4<f32>(point, 1.0);
        contacts[i] = out;
        return;
    }

    // Centre inside the box: exit through the least-penetrated face.
    let pen = he - abs(local);
    var k: u32 = 0u;
    var min_pen: f32 = pen.x;
    if (pen.y < min_pen) {
        min_pen = pen.y;
        k = 1u;
    }
    if (pen.z < min_pen) {
        min_pen = pen.z;
        k = 2u;
    }

    var n_local = vec3<f32>(0.0, 0.0, 0.0);
    var q_in = local;
    if (k == 0u) {
        let s = select(-1.0, 1.0, local.x >= 0.0);
        n_local.x = s;
        q_in.x = s * he.x;
    } else if (k == 1u) {
        let s = select(-1.0, 1.0, local.y >= 0.0);
        n_local.y = s;
        q_in.y = s * he.y;
    } else {
        let s = select(-1.0, 1.0, local.z >= 0.0);
        n_local.z = s;
        q_in.z = s * he.z;
    }

    let normal = a0 * n_local.x + a1 * n_local.y + a2 * n_local.z;
    let depth = r + min_pen;
    let point = bc + a0 * q_in.x + a1 * q_in.y + a2 * q_in.z;
    out.normal_depth = vec4<f32>(normal, depth);
    out.point_valid = vec4<f32>(point, 1.0);
    contacts[i] = out;
}
