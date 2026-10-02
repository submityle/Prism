// OBB-versus-triangle narrow-phase contact kernel (Separating Axis Theorem).
//
// One invocation per candidate (box, triangle) couple. Each invocation reads an
// oriented bounding box (centre, three orthonormal local axes, and per-axis half
// extents) and a triangle (three world-space vertices), runs the same thirteen-
// axis separating-axis test and minimum-translation manifold construction the
// CPU twin runs (`narrowphase/obb_triangle.rs`), and writes one contact slot: a
// unit normal pointing from the triangle toward the box (the push-out
// direction), the penetration depth, a representative contact point, and a
// validity flag. A couple that is separated or exactly grazing writes a zeroed
// slot with valid = 0, so the output index stays aligned with the input couple
// index.
//
// Geometry: a convex box and a (flat, convex) triangle are disjoint iff some
// axis separates their projected intervals. Thirteen axes suffice (the Akenine-
// Moller triangle-box overlap set): the box's three face normals, the single
// triangle face normal (b - a) x (c - a), and the nine edge-edge cross products
// box_axis_i x tri_edge_j with the three triangle edges e0 = b - a, e1 = c - b,
// e2 = a - c. Along a unit axis L the box projects to [cB - rB, cB + rB] with
// cB = dot(L, center) and rB = |dot(L, a0)|*he.x + |dot(L, a1)|*he.y +
// |dot(L, a2)|*he.z; the triangle projects to [min, max] over its three
// projected vertices. The interval overlap is overlap = min(penL, penR) with
// penL = tmax - bmin and penR = bmax - tmin. Any overlap <= 0 means separation
// (strict). Otherwise the minimum-overlap axis is the minimum translation
// vector: its overlap is the depth and its oriented direction is the normal.
//
// Normal orientation: each axis is oriented toward its shorter one-sided push --
// when penL <= penR the box clears along +L, else along -L (ties resolve to +L).
// This points the normal from the triangle toward the box with no separate
// dot(t, L) flip (the triangle has no single centre to flip against).
//
// Degenerate axes: an edge-edge cross vanishes when the box edge and triangle
// edge are parallel; such an axis is parallel to a face axis already tested and
// is skipped (guarded by CROSS_EPS2 on the squared cross length). The triangle
// normal is likewise skipped when the triangle is degenerate (collinear or
// zero-area). Face preference: edge axes (idx >= 4) carry a comparison penalty
// EDGE_BIAS so a face axis of near-equal overlap wins the minimum search; the
// reported depth is the true overlap.
//
// Single-point manifold: this slice reports the mid-overlap point between the
// box's support vertex driven deepest into the triangle (along -normal) and the
// closest point on the solid triangle to that corner. It is a deliberate, honest
// simplification, not a stub; a full multi-point clipped manifold is a separate
// follow-up slice.
//
// The arithmetic is bit-for-bit with the twin: the same thirteen axes in the
// same order, the same CROSS_EPS2 normalise guard, the same projection sums, the
// same strict overlap <= 0 rejection, the same biased minimum search, the same
// penL <= penR orientation, and the same support vertex plus closest-point
// cascade. Only the reciprocal square roots in the axis normalisation and the
// barycentric reciprocals in the closest-point helper are inexact, so the parity
// test matches the validity flag exactly and the normal, depth, and point to
// within a tight tolerance.
//
// Provenance: the thirteen-axis box-triangle separating-axis set is Tomas
// Akenine-Moller, *Fast 3D Triangle-Box Overlap Testing* (2001); the separating-
// axis minimum-translation manifold and the closest-point-on-triangle Voronoi
// cascade are Christer Ericson, *Real-Time Collision Detection* (2004), sections
// 5.2.9 and 5.1.5. No Unreal Engine source or derived code.

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

// One triangle collider. Each vertex rides in the xyz of its row; the w lanes
// are unused padding to keep the triangle to three vec4s.
struct Triangle {
    a: vec4<f32>,
    b: vec4<f32>,
    c: vec4<f32>,
};

// One output contact: (normal.xyz, depth) and (point.xyz, valid).
struct Contact {
    normal_depth: vec4<f32>,
    point_valid: vec4<f32>,
};

// Squared cross-length threshold below which an edge-edge axis (or the triangle
// normal) is degenerate and skipped; kept identical to CROSS_EPS2 in
// `narrowphase/obb_triangle.rs`.
const CROSS_EPS2: f32 = 1.0e-12;
// Comparison-only penalty preferring face axes over near-equal edge axes.
const EDGE_BIAS: f32 = 1.0e-5;
// Sentinel overlap for a skipped (degenerate) axis.
const SKIP_OVERLAP: f32 = 1.0e30;

@group(0) @binding(0) var<uniform> params: Params;
// Oriented bounding boxes, four vec4s each.
@group(0) @binding(1) var<storage, read> boxes: array<Obb>;
// Triangles, three vec4s each.
@group(0) @binding(2) var<storage, read> triangles: array<Triangle>;
// Candidate couples, one (box, triangle) index pair each.
@group(0) @binding(3) var<storage, read> pairs: array<vec2<u32>>;
// Output contacts, one slot per couple.
@group(0) @binding(4) var<storage, read_write> contacts: array<Contact>;

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

// Closest point on the solid triangle (a, b, c) to p, by Ericson's Voronoi-
// region cascade. Replicated operation for operation from the CPU twin's
// `closest_point_on_triangle` (reused from `narrowphase/sphere_triangle.rs`),
// including the fixed comparison order and the barycentric reciprocals.
fn closest_point_on_triangle(p: vec3<f32>, a: vec3<f32>, b: vec3<f32>, c: vec3<f32>) -> vec3<f32> {
    let ab = b - a;
    let ac = c - a;
    let ap = p - a;

    // Vertex region outside A.
    let d1 = dot(ab, ap);
    let d2 = dot(ac, ap);
    if (d1 <= 0.0 && d2 <= 0.0) {
        return a;
    }

    // Vertex region outside B.
    let bp = p - b;
    let d3 = dot(ab, bp);
    let d4 = dot(ac, bp);
    if (d3 >= 0.0 && d4 <= d3) {
        return b;
    }

    // Edge region AB.
    let vc = d1 * d4 - d3 * d2;
    if (vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0) {
        let v = d1 / (d1 - d3);
        return a + ab * v;
    }

    // Vertex region outside C.
    let cp = p - c;
    let d5 = dot(ab, cp);
    let d6 = dot(ac, cp);
    if (d6 >= 0.0 && d5 <= d6) {
        return c;
    }

    // Edge region AC.
    let vb = d5 * d2 - d1 * d6;
    if (vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0) {
        let w = d2 / (d2 - d6);
        return a + ac * w;
    }

    // Edge region BC.
    let va = d3 * d6 - d5 * d4;
    if (va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0) {
        let w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        return b + (c - b) * w;
    }

    // Interior face region.
    let denom = 1.0 / (va + vb + vc);
    let v = vb * denom;
    let w = vc * denom;
    return a + ab * v + ac * w;
}

@compute @workgroup_size(64)
fn narrowphase_obb_triangle(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.num_pairs) {
        return;
    }

    let pair = pairs[i];
    let bx = boxes[pair.x];
    let tri = triangles[pair.y];

    let x0 = bx.axis0.xyz;
    let x1 = bx.axis1.xyz;
    let x2 = bx.axis2.xyz;
    let he = vec3<f32>(bx.axis0.w, bx.axis1.w, bx.axis2.w);
    let center = bx.center.xyz;

    let va = tri.a.xyz;
    let vb = tri.b.xyz;
    let vc = tri.c.xyz;
    let e0 = vb - va;
    let e1 = vc - vb;
    let e2 = va - vc;

    var out: Contact;

    // Build the thirteen candidate axes in the fixed order: the box's three face
    // normals, the triangle face normal, then the nine edge-edge crosses
    // (normalised, degenerate ones flagged).
    var axes: array<vec3<f32>, 13>;
    var valid: array<bool, 13>;
    axes[0] = x0;
    axes[1] = x1;
    axes[2] = x2;
    valid[0] = true;
    valid[1] = true;
    valid[2] = true;

    let raw_normal = cross(e0, vc - va);
    let n_len2 = dot(raw_normal, raw_normal);
    if (n_len2 < CROSS_EPS2) {
        valid[3] = false;
        axes[3] = vec3<f32>(0.0, 0.0, 0.0);
    } else {
        valid[3] = true;
        axes[3] = normalize(raw_normal);
    }

    let box_axes = array<vec3<f32>, 3>(x0, x1, x2);
    let tri_edges = array<vec3<f32>, 3>(e0, e1, e2);
    var k = 4;
    for (var ii = 0; ii < 3; ii = ii + 1) {
        for (var jj = 0; jj < 3; jj = jj + 1) {
            let c = cross(box_axes[ii], tri_edges[jj]);
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

    let verts = array<vec3<f32>, 3>(va, vb, vc);

    var best_overlap = SKIP_OVERLAP;
    var best_cmp = SKIP_OVERLAP;
    var best_normal = vec3<f32>(0.0, 0.0, 0.0);
    var separated = false;
    for (var idx = 0; idx < 13; idx = idx + 1) {
        if (!valid[idx]) {
            continue;
        }
        let axis = axes[idx];

        // Box interval along the axis.
        let cb = dot(axis, center);
        let rb = projected_radius(axis, x0, x1, x2, he);
        let bmin = cb - rb;
        let bmax = cb + rb;

        // Triangle interval along the axis.
        let p0 = dot(axis, verts[0]);
        let p1 = dot(axis, verts[1]);
        let p2 = dot(axis, verts[2]);
        let tmin = min(p0, min(p1, p2));
        let tmax = max(p0, max(p1, p2));

        // Two one-sided pushes that clear the box off the triangle.
        let pen_l = tmax - bmin;
        let pen_r = bmax - tmin;
        let overlap = min(pen_l, pen_r);
        if (overlap <= 0.0) {
            separated = true;
        }

        // Orient the axis toward the shorter push (ties resolve to +L).
        var oriented = axis;
        if (pen_l > pen_r) {
            oriented = -axis;
        }

        // Edge axes (idx >= 4) carry a comparison penalty so a face axis of
        // near-equal overlap is preferred; the reported depth stays exact.
        var bias = 0.0;
        if (idx >= 4) {
            bias = EDGE_BIAS;
        }
        let cmp = overlap + bias;
        if (cmp < best_cmp) {
            best_cmp = cmp;
            best_overlap = overlap;
            best_normal = oriented;
        }
    }

    if (separated) {
        out.normal_depth = vec4<f32>(0.0, 0.0, 0.0, 0.0);
        out.point_valid = vec4<f32>(0.0, 0.0, 0.0, 0.0);
        contacts[i] = out;
        return;
    }

    // Representative single point: the mid-overlap between the box's support
    // vertex driven deepest into the triangle (along -normal) and the closest
    // point on the solid triangle to that corner.
    let box_point = support_vertex(center, x0, x1, x2, he, -best_normal);
    let tri_point = closest_point_on_triangle(box_point, va, vb, vc);
    let point = (box_point + tri_point) * 0.5;

    out.normal_depth = vec4<f32>(best_normal, best_overlap);
    out.point_valid = vec4<f32>(point, 1.0);
    contacts[i] = out;
}
