// OBB-versus-OBB narrow-phase contact kernel (Separating Axis Theorem).
//
// One invocation per (box, box) candidate couple. Each invocation reads two
// oriented bounding boxes (centre, three orthonormal local axes, and per-axis
// half extents), runs the same fifteen-axis separating-axis test and
// minimum-translation manifold construction the CPU twin runs
// (`narrowphase/obb_obb.rs`), and writes one contact slot: the contact normal
// (pointing from box a toward box b), the penetration depth, a representative
// mid-overlap point, and a validity flag. A couple whose boxes are separated or
// exactly grazing writes a zeroed slot with valid = 0, so the output index
// stays aligned with the input couple index.
//
// Geometry: two convex boxes are disjoint iff some axis separates their
// projected intervals. Fifteen axes suffice: box a's three face normals, box
// b's three face normals, and the nine edge-edge cross products a_i x b_j. Along
// a unit axis L the projected half-width of a box is
// |dot(L, x0)|*he.x + |dot(L, x1)|*he.y + |dot(L, x2)|*he.z, and the interval
// overlap is overlap = rA + rB - |dot(L, t)| with t = center_b - center_a. Any
// overlap <= 0 means separation (strict). Otherwise the minimum-overlap axis is
// the minimum translation vector: its overlap is the depth and its oriented
// direction is the normal.
//
// Degenerate edge axes: when two edges are parallel their cross vanishes; such
// an axis is parallel to a face axis already tested and is skipped (guarded by
// CROSS_EPS2 on the squared cross length). Face preference: edge axes carry a
// comparison penalty EDGE_BIAS so a face axis of near-equal overlap wins the
// minimum search; the reported depth is the true overlap. Normal convention:
// the MTV axis is flipped when dot(t, L) < 0 so it always runs a -> b.
//
// Single-point manifold: this slice reports the mid-overlap point between box
// a's support vertex along +normal and box b's support vertex along -normal. It
// is a deliberate, honest simplification (the point lies on the mid-overlap
// plane the solver pushes against), not a stub; a full multi-point clipped
// manifold is a separate follow-up slice.
//
// The arithmetic is bit-for-bit with the twin: the same fifteen axes in the
// same order, the same CROSS_EPS2 normalise guard, the same projection sums,
// the same strict overlap <= 0 rejection, the same biased minimum search, the
// same dot(t, L) sign flip, and the same two support vertices. Only the
// reciprocal square root in the edge-axis normalise is inexact, so the parity
// test matches the validity flag exactly and the normal, depth, and point to
// within a tight tolerance.
//
// Provenance: textbook separating-axis oriented-bounding-box collision
// manifold. No Unreal Engine source or derived code.

struct Params {
    // Number of candidate couples queued in the pair buffer.
    num_pairs: u32,
    // Padding to a 16-byte uniform stride.
    pad0: u32,
    pad1: u32,
    pad2: u32,
};

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

// Squared cross-length threshold below which an edge-edge axis is degenerate.
const CROSS_EPS2: f32 = 1.0e-12;
// Comparison-only penalty preferring face axes over near-equal edge axes.
const EDGE_BIAS: f32 = 1.0e-5;
// Sentinel overlap for a skipped (degenerate) axis.
const SKIP_OVERLAP: f32 = 1.0e30;

@group(0) @binding(0) var<uniform> params: Params;
// Oriented bounding boxes, four vec4s each.
@group(0) @binding(1) var<storage, read> boxes: array<Obb>;
// Candidate couples, one (box, box) index pair each.
@group(0) @binding(2) var<storage, read> pairs: array<vec2<u32>>;
// Output contacts, one slot per couple.
@group(0) @binding(3) var<storage, read_write> contacts: array<Contact>;

// Returns +1.0 when x >= 0.0, else -1.0 (matches sign_pos in the CPU twin).
fn sign_pos(x: f32) -> f32 {
    return select(-1.0, 1.0, x >= 0.0);
}

// Projected half-width of a box (axes a0/a1/a2, half extents he) onto `axis`.
fn projected_radius(axis: vec3<f32>, a0: vec3<f32>, a1: vec3<f32>, a2: vec3<f32>, he: vec3<f32>) -> f32 {
    return abs(dot(axis, a0)) * he.x + abs(dot(axis, a1)) * he.y + abs(dot(axis, a2)) * he.z;
}

// Support vertex of a box in direction `dir`.
fn support_vertex(center: vec3<f32>, a0: vec3<f32>, a1: vec3<f32>, a2: vec3<f32>, he: vec3<f32>, dir: vec3<f32>) -> vec3<f32> {
    return center
        + a0 * (sign_pos(dot(dir, a0)) * he.x)
        + a1 * (sign_pos(dot(dir, a1)) * he.y)
        + a2 * (sign_pos(dot(dir, a2)) * he.z);
}

@compute @workgroup_size(64)
fn narrowphase_obb_obb(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.num_pairs) {
        return;
    }

    let couple = pairs[i];
    let ba = boxes[couple.x];
    let bb = boxes[couple.y];

    let a0 = ba.axis0.xyz;
    let a1 = ba.axis1.xyz;
    let a2 = ba.axis2.xyz;
    let ea = vec3<f32>(ba.axis0.w, ba.axis1.w, ba.axis2.w);
    let b0 = bb.axis0.xyz;
    let b1 = bb.axis1.xyz;
    let b2 = bb.axis2.xyz;
    let eb = vec3<f32>(bb.axis0.w, bb.axis1.w, bb.axis2.w);
    let t = bb.center.xyz - ba.center.xyz;

    var out: Contact;

    // Build the fifteen candidate axes in the fixed order: a's faces, b's faces,
    // then the nine edge-edge crosses (normalised, degenerate ones flagged).
    var axes: array<vec3<f32>, 15>;
    var valid: array<bool, 15>;
    axes[0] = a0;
    axes[1] = a1;
    axes[2] = a2;
    axes[3] = b0;
    axes[4] = b1;
    axes[5] = b2;
    for (var m = 0; m < 6; m = m + 1) {
        valid[m] = true;
    }

    let acols = array<vec3<f32>, 3>(a0, a1, a2);
    let bcols = array<vec3<f32>, 3>(b0, b1, b2);
    var k = 6;
    for (var ii = 0; ii < 3; ii = ii + 1) {
        for (var jj = 0; jj < 3; jj = jj + 1) {
            let c = cross(acols[ii], bcols[jj]);
            let len2 = dot(c, c);
            if (len2 < CROSS_EPS2) {
                valid[k] = false;
                axes[k] = vec3<f32>(0.0, 0.0, 0.0);
            } else {
                valid[k] = true;
                axes[k] = normalize(c);
            }
            k = k + 1;
        }
    }

    var best_overlap = SKIP_OVERLAP;
    var best_cmp = SKIP_OVERLAP;
    var best_axis = vec3<f32>(0.0, 0.0, 0.0);
    var separated = false;
    for (var idx = 0; idx < 15; idx = idx + 1) {
        if (!valid[idx]) {
            continue;
        }
        let axis = axes[idx];
        let ra = projected_radius(axis, a0, a1, a2, ea);
        let rb = projected_radius(axis, b0, b1, b2, eb);
        let dist = dot(t, axis);
        let overlap = ra + rb - abs(dist);
        if (overlap <= 0.0) {
            separated = true;
        }
        var bias = 0.0;
        if (idx >= 6) {
            bias = EDGE_BIAS;
        }
        let cmp = overlap + bias;
        if (cmp < best_cmp) {
            best_cmp = cmp;
            best_overlap = overlap;
            best_axis = axis;
        }
    }

    if (separated) {
        out.normal_depth = vec4<f32>(0.0, 0.0, 0.0, 0.0);
        out.point_valid = vec4<f32>(0.0, 0.0, 0.0, 0.0);
        contacts[i] = out;
        return;
    }

    // Orient the minimum-translation axis from box a toward box b.
    var normal = best_axis;
    if (dot(t, best_axis) < 0.0) {
        normal = -best_axis;
    }

    // Representative single point: mid-overlap of the two support vertices.
    let pa = support_vertex(ba.center.xyz, a0, a1, a2, ea, normal);
    let pb = support_vertex(bb.center.xyz, b0, b1, b2, eb, -normal);
    let point = (pa + pb) * 0.5;

    out.normal_depth = vec4<f32>(normal, best_overlap);
    out.point_valid = vec4<f32>(point, 1.0);
    contacts[i] = out;
}
