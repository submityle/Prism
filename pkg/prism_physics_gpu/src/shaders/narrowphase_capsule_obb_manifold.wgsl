// Capsule-versus-OBB two-point contact-manifold kernel (reference-face clip).
//
// One invocation per candidate (capsule, box) couple. Each invocation runs the
// same single deepest-feature test the single-point kernel runs
// (narrowphase_capsule_obb.wgsl), then promotes that one contact to an
// up-to-two-point manifold exactly as the CPU twin does
// (narrowphase/capsule_obb_manifold.rs): it picks the box face the contact
// normal aligns with most, clips the capsule axis to that face rectangle with
// the Liang-Barsky algorithm, and keeps the clip-boundary corners whose swept
// radius dips below the face plane.
//
// The output is one fixed-stride manifold record per couple: the shared normal
// (the reference face's outward push-out direction, from the box toward the
// capsule) with the live point count in its w lane, then four
// (position.xyz, depth) points. A separated or grazing couple writes count = 0.
// A single feature contact, a sliver clip, a clip that misses the face, or a
// pair of coincident corners collapses honestly to the single deepest contact
// (count = 1) rather than fabricating a second point.
//
// The arithmetic is operation-for-operation with the twin: the same closest
// feature search, the same reference-axis argmax with a lower-index tie-break,
// the same Liang-Barsky edge order and parallel-edge handling, the same
// depth = rc - (s * local[k] - he[k]) corner test, and the same collapse
// thresholds. Only the square roots and reciprocals in the closest-feature
// search, the outside-face normalise, and the four edge-crossing solves are
// inexact, so the parity test matches the count exactly and the normal,
// positions, and depths to within a tight tolerance.
//
// Provenance: textbook capsule-versus-oriented-bounding-box closest-feature
// collision plus Liang-Barsky segment-rectangle clipping. No Unreal Engine
// source or derived code.

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

// Parameter-span threshold below which the clipped stretch of the capsule axis
// is treated as a single point; kept identical to `CLIP_T_EPS` in
// `narrowphase/capsule_obb_manifold.rs`.
const CLIP_T_EPS: f32 = 1.0e-9;

// Squared-distance threshold below which the two clipped corners are treated as
// coincident; kept identical to `SEP_EPS2` in
// `narrowphase/capsule_obb_manifold.rs`.
const SEP_EPS2: f32 = 1.0e-12;

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

// One output manifold: (normal.xyz, count) then four (position.xyz, depth)
// points. Only the first `count` points are live.
struct Manifold {
    normal_count: vec4<f32>,
    p0: vec4<f32>,
    p1: vec4<f32>,
    p2: vec4<f32>,
    p3: vec4<f32>,
};

// The single deepest-feature contact carried between the closest-feature search
// and the manifold construction: the world push-out normal, the world contact
// point, the penetration depth, and a validity flag (0.0 for no penetration).
struct SingleContact {
    normal: vec3<f32>,
    point: vec3<f32>,
    depth: f32,
    valid: f32,
};

@group(0) @binding(0) var<uniform> params: Params;
// Capsules, two vec4s each.
@group(0) @binding(1) var<storage, read> capsules: array<Capsule>;
// Oriented bounding boxes, four vec4s each.
@group(0) @binding(2) var<storage, read> boxes: array<Obb>;
// Candidate couples, one (capsule, box) index pair each.
@group(0) @binding(3) var<storage, read> pairs: array<vec2<u32>>;
// Output manifolds, one fixed-stride record per couple.
@group(0) @binding(4) var<storage, read_write> manifolds: array<Manifold>;

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

// Runs the single deepest-feature capsule-versus-box test, mirroring
// `narrowphase_capsule_obb.wgsl`. Returns the world push-out normal, the world
// contact point, the penetration depth, and a validity flag that is 0.0 when the
// capsule does not strictly penetrate the box.
fn capsule_obb_single(
    p0: vec3<f32>,
    p1: vec3<f32>,
    rc: f32,
    bc: vec3<f32>,
    a0: vec3<f32>,
    a1: vec3<f32>,
    a2: vec3<f32>,
    he: vec3<f32>,
) -> SingleContact {
    var out: SingleContact;
    out.normal = vec3<f32>(0.0, 0.0, 0.0);
    out.point = vec3<f32>(0.0, 0.0, 0.0);
    out.depth = 0.0;
    out.valid = 0.0;

    let d0 = p0 - bc;
    let d1 = p1 - bc;
    let local0 = vec3<f32>(dot(d0, a0), dot(d0, a1), dot(d0, a2));
    let local1 = vec3<f32>(dot(d1, a0), dot(d1, a1), dot(d1, a2));

    let t = segment_box_closest_t(local0, local1, he);
    let local = local0 + (local1 - local0) * t;

    let q = clamp(local, -he, he);
    let diff = local - q;
    let d2 = dot(diff, diff);

    if (d2 > INSIDE_EPS2) {
        // Closest axis point outside the box: nearest feature is the clamp.
        let dist = sqrt(d2);
        if (dist >= rc) {
            return out;
        }
        let n_local = diff / dist;
        out.normal = a0 * n_local.x + a1 * n_local.y + a2 * n_local.z;
        out.depth = rc - dist;
        out.point = bc + a0 * q.x + a1 * q.y + a2 * q.z;
        out.valid = 1.0;
        return out;
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

    out.normal = a0 * n_local.x + a1 * n_local.y + a2 * n_local.z;
    out.depth = rc + min_pen;
    out.point = bc + a0 * q_in.x + a1 * q_in.y + a2 * q_in.z;
    out.valid = 1.0;
    return out;
}

// Component `i` of `v` (0 -> x, 1 -> y, else z); mirrors the CPU `comp` helper
// so the axis-indexed arithmetic reads the same on both paths.
fn comp(v: vec3<f32>, i: u32) -> f32 {
    if (i == 0u) {
        return v.x;
    }
    if (i == 1u) {
        return v.y;
    }
    return v.z;
}

// The reference-face axis `k`: the axis the local contact normal aligns with
// most strongly (largest |component|, ties resolving to the lower axis index).
fn reference_axis_k(n_local: vec3<f32>) -> u32 {
    let ax = abs(n_local.x);
    let ay = abs(n_local.y);
    let az = abs(n_local.z);
    var k: u32 = 0u;
    var best = ax;
    if (ay > best) {
        best = ay;
        k = 1u;
    }
    if (az > best) {
        k = 2u;
    }
    return k;
}

// Clips the segment p0 -> p1, given in one face's (u, v) coordinates, to the
// rectangle [-he_u, he_u] x [-he_v, he_v] with the Liang-Barsky algorithm.
// Returns vec3(t_enter, t_leave, 1.0) when the segment meets the rectangle, or
// vec3(0.0, 0.0, 0.0) when it lies wholly outside (the w lane is the validity
// flag). A negative edge coefficient is an entering crossing (raises t_enter), a
// positive one a leaving crossing (lowers t_leave), and a zero one is a parallel
// edge that rejects only when the segment sits on its outer side (q < 0).
fn clip_segment_to_rect(
    p0u: f32,
    p0v: f32,
    p1u: f32,
    p1v: f32,
    he_u: f32,
    he_v: f32,
) -> vec3<f32> {
    let du = p1u - p0u;
    let dv = p1v - p0v;
    var edges: array<vec2<f32>, 4>;
    edges[0] = vec2<f32>(-du, p0u + he_u); // left:   u >= -he_u
    edges[1] = vec2<f32>(du, he_u - p0u);  // right:  u <=  he_u
    edges[2] = vec2<f32>(-dv, p0v + he_v); // bottom: v >= -he_v
    edges[3] = vec2<f32>(dv, he_v - p0v);  // top:    v <=  he_v

    var t_enter = 0.0;
    var t_leave = 1.0;
    for (var e = 0u; e < 4u; e = e + 1u) {
        let p = edges[e].x;
        let q = edges[e].y;
        if (p == 0.0) {
            if (q < 0.0) {
                return vec3<f32>(0.0, 0.0, 0.0);
            }
        } else if (p < 0.0) {
            let t = q / p;
            if (t > t_leave) {
                return vec3<f32>(0.0, 0.0, 0.0);
            }
            if (t > t_enter) {
                t_enter = t;
            }
        } else {
            let t = q / p;
            if (t < t_enter) {
                return vec3<f32>(0.0, 0.0, 0.0);
            }
            if (t < t_leave) {
                t_leave = t;
            }
        }
    }
    return vec3<f32>(t_enter, t_leave, 1.0);
}

// Wraps a single deepest-feature contact as a one-point manifold, the honest
// fallback whenever a stable second point cannot be found.
fn single_manifold(c: SingleContact) -> Manifold {
    var m: Manifold;
    m.normal_count = vec4<f32>(c.normal, 1.0);
    m.p0 = vec4<f32>(c.point, c.depth);
    m.p1 = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    m.p2 = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    m.p3 = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    return m;
}

@compute @workgroup_size(64)
fn narrowphase_capsule_obb_manifold(@builtin(global_invocation_id) gid: vec3<u32>) {
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

    // The single deepest contact is both the penetration gate and the fallback.
    let contact = capsule_obb_single(p0, p1, rc, bc, a0, a1, a2, he);
    if (contact.valid == 0.0) {
        var zero: Manifold;
        zero.normal_count = vec4<f32>(0.0, 0.0, 0.0, 0.0);
        zero.p0 = vec4<f32>(0.0, 0.0, 0.0, 0.0);
        zero.p1 = vec4<f32>(0.0, 0.0, 0.0, 0.0);
        zero.p2 = vec4<f32>(0.0, 0.0, 0.0, 0.0);
        zero.p3 = vec4<f32>(0.0, 0.0, 0.0, 0.0);
        manifolds[i] = zero;
        return;
    }

    let fallback = single_manifold(contact);

    // Project both capsule endpoints into the box frame.
    let d0 = p0 - bc;
    let d1 = p1 - bc;
    let local0 = vec3<f32>(dot(d0, a0), dot(d0, a1), dot(d0, a2));
    let local1 = vec3<f32>(dot(d1, a0), dot(d1, a1), dot(d1, a2));

    // Reference face: the axis the contact normal aligns with most, and its side.
    let n_local = vec3<f32>(
        dot(contact.normal, a0),
        dot(contact.normal, a1),
        dot(contact.normal, a2),
    );
    let k = reference_axis_k(n_local);
    let u = (k + 1u) % 3u;
    let v = (k + 2u) % 3u;
    let s = select(-1.0, 1.0, comp(n_local, k) >= 0.0);

    // Clip the capsule axis to the reference face rectangle in the (u, v) plane.
    let clip = clip_segment_to_rect(
        comp(local0, u),
        comp(local0, v),
        comp(local1, u),
        comp(local1, v),
        comp(he, u),
        comp(he, v),
    );
    if (clip.z == 0.0) {
        manifolds[i] = fallback;
        return;
    }
    let t0 = clip.x;
    let t1 = clip.y;
    if (t1 - t0 <= CLIP_T_EPS) {
        // The overlap is a sliver: no stable second point, keep the deepest one.
        manifolds[i] = fallback;
        return;
    }

    // Project each clip-boundary point onto the reference face and keep the ones
    // whose swept surface actually dips below the face plane.
    let seg = local1 - local0;
    let he_k = comp(he, k);
    var pts: array<vec4<f32>, 2>;
    var count: u32 = 0u;
    for (var idx = 0u; idx < 2u; idx = idx + 1u) {
        var t = t0;
        if (idx == 1u) {
            t = t1;
        }
        let local_pt = local0 + seg * t;
        // Signed height of the axis point above the reference face plane.
        let above = s * comp(local_pt, k) - he_k;
        let depth = rc - above;
        if (depth <= 0.0) {
            continue;
        }
        let face_k = s * he_k;
        var qv = local_pt;
        if (k == 0u) {
            qv = vec3<f32>(face_k, local_pt.y, local_pt.z);
        } else if (k == 1u) {
            qv = vec3<f32>(local_pt.x, face_k, local_pt.z);
        } else {
            qv = vec3<f32>(local_pt.x, local_pt.y, face_k);
        }
        let world = bc + a0 * qv.x + a1 * qv.y + a2 * qv.z;
        pts[count] = vec4<f32>(world, depth);
        count = count + 1u;
    }

    if (count < 2u) {
        // Only one end (or neither) clears the face: keep the deepest contact.
        manifolds[i] = fallback;
        return;
    }
    // Reject a degenerate manifold whose two corners coincide.
    let sep = pts[1].xyz - pts[0].xyz;
    if (dot(sep, sep) <= SEP_EPS2) {
        manifolds[i] = fallback;
        return;
    }

    var face_axis = a0;
    if (k == 1u) {
        face_axis = a1;
    } else if (k == 2u) {
        face_axis = a2;
    }
    let normal = face_axis * s;

    var m: Manifold;
    m.normal_count = vec4<f32>(normal, 2.0);
    m.p0 = pts[0];
    m.p1 = pts[1];
    m.p2 = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    m.p3 = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    manifolds[i] = m;
}
