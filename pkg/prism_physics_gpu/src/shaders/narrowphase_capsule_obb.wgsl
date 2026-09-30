// Capsule-versus-OBB narrow-phase contact kernel.
//
// One invocation per candidate (capsule, box) couple. Each invocation reads a
// capsule (segment endpoints p0, p1 and a swept radius) and an oriented
// bounding box (centre, three orthonormal local axes, and per-axis half
// extents), runs the same overlap test and manifold construction the CPU twin
// runs (`narrowphase/capsule_obb.rs`), and writes one contact slot: a unit
// normal pointing from the box surface toward the capsule (the push-out
// direction), the penetration depth, the world contact point on the box
// surface, and a validity flag. A couple that does not penetrate writes a
// zeroed slot with valid = 0, so the output index stays aligned with the input
// couple index.
//
// Geometry: project both capsule endpoints into the box frame, find the segment
// parameter t whose point is closest to the box (an exact convex
// piecewise-quadratic minimisation of the segment-to-box squared distance:
// evaluate both endpoints, the six face-plane crossings, and the parabola vertex
// of every piece), then collapse to the sphere-versus-box branch at that point.
// When the closest axis point is outside (`dot(diff, diff) > INSIDE_EPS2`) the
// couple contacts only for a strict `dist < rc`; the local normal is
// `diff / dist` and depth `rc - dist`. When it is inside, it exits through the
// least-penetrated face k (ties resolve to the lower axis index), the local
// normal is `sign(local[k])` along that axis, and depth is
// `rc + (he[k] - abs(local[k]))`. The world normal and contact point recombine
// the local vectors through the box axes.
//
// The arithmetic is bit-for-bit with the twin apart from the square roots and
// reciprocals in the closest-feature search and the outside-face
// normalisation, whose WGSL rounding differs in the low bits; the parity test
// therefore matches the validity flag exactly and the normal, depth, and point
// within a tight tolerance.
//
// Provenance: textbook capsule-versus-oriented-bounding-box closest-feature
// collision. No Unreal Engine source or derived code.

struct Params {
    // Number of candidate couples queued in the pair buffer.
    num_pairs: u32,
    // Padding to a 16-byte uniform stride.
    pad0: u32,
    pad1: u32,
    pad2: u32,
};

// Squared-length threshold below which the clamped offset is treated as zero,
// i.e. the closest axis point is inside the box; kept identical to
// `INSIDE_EPS2` in `narrowphase/obb.rs`.
const INSIDE_EPS2: f32 = 1.0e-12;

// Squared-length threshold below which a capsule axis component is treated as
// parallel to a face plane; kept identical to `AXIS_EPS2` in
// `narrowphase/capsule_obb.rs`.
const AXIS_EPS2: f32 = 1.0e-12;

// One capsule: (p0.xyz, radius) then (p1.xyz, pad).
struct Capsule {
    p0_radius: vec4<f32>,
    p1_pad: vec4<f32>,
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

@group(0) @binding(0) var<uniform> params: Params;
// Capsules, two vec4s each.
@group(0) @binding(1) var<storage, read> capsules: array<Capsule>;
// Oriented bounding boxes, four vec4s each.
@group(0) @binding(2) var<storage, read> boxes: array<Obb>;
// Candidate couples, one (capsule, box) index pair each.
@group(0) @binding(3) var<storage, read> pairs: array<vec2<u32>>;
// Output contacts, one slot per couple.
@group(0) @binding(4) var<storage, read_write> contacts: array<Contact>;

// Squared distance from the segment point at `t` to the local box `[-he, he]`.
fn dist2_to_box(a: vec3<f32>, d: vec3<f32>, t: f32, he: vec3<f32>) -> f32 {
    let p = a + d * t;
    let excess = p - clamp(p, -he, he);
    return dot(excess, excess);
}

// The two face-plane crossings for one axis, each clamped to [0, 1]. A near-zero
// axis component collapses both to 0.0 (an already-evaluated endpoint).
fn axis_crossings(ai: f32, di: f32, ei: f32) -> vec2<f32> {
    if (di * di <= AXIS_EPS2) {
        return vec2<f32>(0.0, 0.0);
    }
    let inv = 1.0 / di;
    let hit_pos = clamp((ei - ai) * inv, 0.0, 1.0);
    let hit_neg = clamp((-ei - ai) * inv, 0.0, 1.0);
    return vec2<f32>(hit_pos, hit_neg);
}

// Parabola-vertex parameter of the box-distance function on the piece around
// `t_mid`, clamped to [t_lo, t_hi]. Flat (no active axis) returns t_lo.
fn piece_vertex(a: vec3<f32>, d: vec3<f32>, he: vec3<f32>, t_lo: f32, t_hi: f32, t_mid: f32) -> f32 {
    let p = a + d * t_mid;
    var alpha = 0.0;
    var half_beta = 0.0;
    if (p.x > he.x) {
        let off = a.x - he.x;
        alpha = alpha + d.x * d.x;
        half_beta = half_beta + d.x * off;
    } else if (p.x < -he.x) {
        let off = a.x + he.x;
        alpha = alpha + d.x * d.x;
        half_beta = half_beta + d.x * off;
    }
    if (p.y > he.y) {
        let off = a.y - he.y;
        alpha = alpha + d.y * d.y;
        half_beta = half_beta + d.y * off;
    } else if (p.y < -he.y) {
        let off = a.y + he.y;
        alpha = alpha + d.y * d.y;
        half_beta = half_beta + d.y * off;
    }
    if (p.z > he.z) {
        let off = a.z - he.z;
        alpha = alpha + d.z * d.z;
        half_beta = half_beta + d.z * off;
    } else if (p.z < -he.z) {
        let off = a.z + he.z;
        alpha = alpha + d.z * d.z;
        half_beta = half_beta + d.z * off;
    }
    if (alpha <= AXIS_EPS2) {
        return t_lo;
    }
    return clamp(-half_beta / alpha, t_lo, t_hi);
}

// Segment parameter t in [0, 1] whose point is closest to the local box.
fn segment_box_closest_t(a: vec3<f32>, b: vec3<f32>, he: vec3<f32>) -> f32 {
    let d = b - a;
    let cx = axis_crossings(a.x, d.x, he.x);
    let cy = axis_crossings(a.y, d.y, he.y);
    let cz = axis_crossings(a.z, d.z, he.z);

    var breaks: array<f32, 8>;
    breaks[0] = 0.0;
    breaks[1] = 1.0;
    breaks[2] = cx.x;
    breaks[3] = cx.y;
    breaks[4] = cy.x;
    breaks[5] = cy.y;
    breaks[6] = cz.x;
    breaks[7] = cz.y;

    // Fixed selection sort so the piece walk and earliest-t tie-break match the
    // CPU twin exactly.
    for (var i = 0u; i < 8u; i = i + 1u) {
        var min_idx = i;
        for (var j = i + 1u; j < 8u; j = j + 1u) {
            if (breaks[j] < breaks[min_idx]) {
                min_idx = j;
            }
        }
        let tmp = breaks[i];
        breaks[i] = breaks[min_idx];
        breaks[min_idx] = tmp;
    }

    var best_t = breaks[0];
    var best_d2 = dist2_to_box(a, d, best_t, he);
    // A candidate updates the best only on a strict improvement, so ties keep
    // the earlier (smaller) t, matching the twin's tie-break.
    for (var i = 0u; i < 7u; i = i + 1u) {
        let t_lo = breaks[i];
        let t_hi = breaks[i + 1u];
        let g_hi = dist2_to_box(a, d, t_hi, he);
        if (g_hi < best_d2) {
            best_d2 = g_hi;
            best_t = t_hi;
        }
        if (t_hi > t_lo) {
            let t_mid = 0.5 * (t_lo + t_hi);
            let t_vertex = piece_vertex(a, d, he, t_lo, t_hi, t_mid);
            let g_vertex = dist2_to_box(a, d, t_vertex, he);
            if (g_vertex < best_d2) {
                best_d2 = g_vertex;
                best_t = t_vertex;
            }
        }
    }
    return best_t;
}

@compute @workgroup_size(64)
fn narrowphase_capsule_obb(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.num_pairs) {
        return;
    }

    let pair = pairs[i];
    let cap = capsules[pair.x];
    let p0 = cap.p0_radius.xyz;
    let rc = cap.p0_radius.w;
    let p1 = cap.p1_pad.xyz;

    let box_ = boxes[pair.y];
    let bc = box_.center.xyz;
    let a0 = box_.axis0.xyz;
    let a1 = box_.axis1.xyz;
    let a2 = box_.axis2.xyz;
    let he = vec3<f32>(box_.axis0.w, box_.axis1.w, box_.axis2.w);

    // Project both capsule endpoints into the box frame.
    let d0 = p0 - bc;
    let d1 = p1 - bc;
    let local0 = vec3<f32>(dot(d0, a0), dot(d0, a1), dot(d0, a2));
    let local1 = vec3<f32>(dot(d1, a0), dot(d1, a1), dot(d1, a2));

    // Closest point on the capsule axis to the box, in the local frame.
    let t = segment_box_closest_t(local0, local1, he);
    let local = local0 + (local1 - local0) * t;

    let q = clamp(local, -he, he);
    let diff = local - q;
    let d2 = dot(diff, diff);

    var out: Contact;

    if (d2 > INSIDE_EPS2) {
        // Closest axis point outside the box: nearest feature is the clamp.
        let dist = sqrt(d2);
        if (dist >= rc) {
            // Strict overlap: a grazing capsule carries no penetration.
            out.normal_depth = vec4<f32>(0.0, 0.0, 0.0, 0.0);
            out.point_valid = vec4<f32>(0.0, 0.0, 0.0, 0.0);
            contacts[i] = out;
            return;
        }
        let n_local = diff / dist;
        let normal = a0 * n_local.x + a1 * n_local.y + a2 * n_local.z;
        let depth = rc - dist;
        let point = bc + a0 * q.x + a1 * q.y + a2 * q.z;
        out.normal_depth = vec4<f32>(normal, depth);
        out.point_valid = vec4<f32>(point, 1.0);
        contacts[i] = out;
        return;
    }

    // Closest axis point inside the box: exit through the least-penetrated face.
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
    let depth = rc + min_pen;
    let point = bc + a0 * q_in.x + a1 * q_in.y + a2 * q_in.z;
    out.normal_depth = vec4<f32>(normal, depth);
    out.point_valid = vec4<f32>(point, 1.0);
    contacts[i] = out;
}
