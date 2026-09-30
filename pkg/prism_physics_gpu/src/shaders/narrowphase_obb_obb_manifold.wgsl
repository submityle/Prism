// OBB-versus-OBB multi-point contact-manifold kernel (reference-face clipping).
//
// One invocation per (box, box) candidate couple. Each invocation runs the same
// fifteen-axis separating-axis test the single-point kernel runs
// (narrowphase_obb_obb.wgsl), records which axis won, and then promotes that one
// contact to a full manifold exactly as the CPU twin does
// (narrowphase/obb_obb_manifold.rs):
//
//   * a face axis (winning index < 6) clips the incident box's most
//     anti-parallel face against the four side planes of the reference face with
//     the Sutherland-Hodgman algorithm, keeps the clipped corners that penetrate
//     the reference face, and reduces more than four survivors to the four that
//     best bound the contact polygon;
//   * an edge-edge axis (index >= 6) degenerates to the single closest-point
//     pair between the two leaning edges.
//
// The output is one fixed-stride manifold record per couple: the shared normal
// (from box a toward box b) with the live point count in its w lane, then four
// (position.xyz, depth) points. A separated or grazing couple writes count = 0.
//
// The arithmetic is operation-for-operation with the twin: the same fifteen axes
// in the same order with the same CROSS_EPS2 guard and EDGE_BIAS face
// preference, the same reference/incident selection, the same Sutherland-Hodgman
// clip in the same plane order over the same ping-pong buffers, the same
// mid-overlap placement, and the same four-point reduction with the same
// tie-breaks. Only the reciprocal square roots in the edge-axis normalise and
// the segment solver are inexact, so the parity test matches the count exactly
// and the normal, positions, and depths to within a tight tolerance.
//
// Provenance: textbook reference/incident face-clipping contact manifold. No
// Unreal Engine source or derived code.

struct Params {
    // Number of candidate couples queued in the pair buffer.
    num_pairs: u32,
    // Padding to a 16-byte uniform stride.
    pad0: u32,
    pad1: u32,
    pad2: u32,
};

// One oriented bounding box: centre in row 0 (w unused), each local axis in the
// xyz of its row with the matching half extent in that row's w lane.
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

// Squared cross-length threshold below which an edge-edge axis is degenerate.
const CROSS_EPS2: f32 = 1.0e-12;
// Comparison-only penalty preferring face axes over near-equal edge axes.
const EDGE_BIAS: f32 = 1.0e-5;
// Sentinel overlap for a skipped (degenerate) axis.
const SKIP_OVERLAP: f32 = 1.0e30;
// Upper bound on corners the Sutherland-Hodgman clip can produce.
const MAX_CLIP_POINTS: u32 = 8u;

@group(0) @binding(0) var<uniform> params: Params;
// Oriented bounding boxes, four vec4s each.
@group(0) @binding(1) var<storage, read> boxes: array<Obb>;
// Candidate couples, one (box, box) index pair each.
@group(0) @binding(2) var<storage, read> pairs: array<vec2<u32>>;
// Output manifolds, one fixed-stride record per couple.
@group(0) @binding(3) var<storage, read_write> manifolds: array<Manifold>;

// Returns +1.0 when x >= 0.0, else -1.0 (matches sign_pos in the CPU twin).
fn sign_pos(x: f32) -> f32 {
    return select(-1.0, 1.0, x >= 0.0);
}

// Projected half-width of a box (axes a0/a1/a2, half extents he) onto `axis`.
fn projected_radius(axis: vec3<f32>, a0: vec3<f32>, a1: vec3<f32>, a2: vec3<f32>, he: vec3<f32>) -> f32 {
    return abs(dot(axis, a0)) * he.x + abs(dot(axis, a1)) * he.y + abs(dot(axis, a2)) * he.z;
}

// Support vertex of a box in direction `dir`: the corner furthest along `dir`.
fn support_vertex(center: vec3<f32>, a0: vec3<f32>, a1: vec3<f32>, a2: vec3<f32>, he: vec3<f32>, dir: vec3<f32>) -> vec3<f32> {
    return center
        + a0 * (sign_pos(dot(dir, a0)) * he.x)
        + a1 * (sign_pos(dot(dir, a1)) * he.y)
        + a2 * (sign_pos(dot(dir, a2)) * he.z);
}

// The local axis `i` of a box, indexed 0/1/2.
fn box_axis(a0: vec3<f32>, a1: vec3<f32>, a2: vec3<f32>, i: u32) -> vec3<f32> {
    if (i == 0u) { return a0; }
    if (i == 1u) { return a1; }
    return a2;
}

// The half extent along local axis `i` of a box, indexed 0/1/2.
fn box_he(he: vec3<f32>, i: u32) -> f32 {
    if (i == 0u) { return he.x; }
    if (i == 1u) { return he.y; }
    return he.z;
}

// Closest points between segments [p1, q1] and [p2, q2] (Ericson's clamped
// parametric solver). Returns the two closest points packed as a 2x vec3 via
// out-params through a small struct.
struct SegPair {
    ca: vec3<f32>,
    cb: vec3<f32>,
};

fn closest_points_segments(p1: vec3<f32>, q1: vec3<f32>, p2: vec3<f32>, q2: vec3<f32>) -> SegPair {
    let d1 = q1 - p1;
    let d2 = q2 - p2;
    let r = p1 - p2;
    let a = dot(d1, d1);
    let e = dot(d2, d2);
    let f = dot(d2, r);
    let eps = 1.0e-12;

    var s = 0.0;
    var t = 0.0;
    var out: SegPair;
    if (a <= eps && e <= eps) {
        out.ca = p1;
        out.cb = p2;
        return out;
    }
    if (a <= eps) {
        s = 0.0;
        t = clamp(f / e, 0.0, 1.0);
    } else {
        let c = dot(d1, r);
        if (e <= eps) {
            t = 0.0;
            s = clamp(-c / a, 0.0, 1.0);
        } else {
            let b = dot(d1, d2);
            let denom = a * e - b * b;
            if (denom > eps) {
                s = clamp((b * f - c * e) / denom, 0.0, 1.0);
            } else {
                s = 0.0;
            }
            t = (b * s + f) / e;
            if (t < 0.0) {
                t = 0.0;
                s = clamp(-c / a, 0.0, 1.0);
            } else if (t > 1.0) {
                t = 1.0;
                s = clamp((b - c) / a, 0.0, 1.0);
            }
        }
    }
    out.ca = p1 + d1 * s;
    out.cb = p2 + d2 * t;
    return out;
}

// Clips convex polygon `src` (its first `n` vertices) against the half-space
// dot(v - point, normal) <= 0, writing survivors into `dst` and returning the
// new vertex count. Standard Sutherland-Hodgman.
fn clip_to_plane(
    src: ptr<function, array<vec3<f32>, 8>>,
    n: u32,
    point: vec3<f32>,
    normal: vec3<f32>,
    dst: ptr<function, array<vec3<f32>, 8>>,
) -> u32 {
    var count = 0u;
    for (var i = 0u; i < n; i = i + 1u) {
        let cur = (*src)[i];
        let next = (*src)[(i + 1u) % n];
        let dc = dot(cur - point, normal);
        let dn = dot(next - point, normal);
        let cur_in = dc <= 0.0;
        let next_in = dn <= 0.0;
        if (cur_in && count < MAX_CLIP_POINTS) {
            (*dst)[count] = cur;
            count = count + 1u;
        }
        if ((cur_in && !next_in) || (!cur_in && next_in)) {
            let denom = dc - dn;
            let tt = dc / denom;
            if (count < MAX_CLIP_POINTS) {
                (*dst)[count] = cur + (next - cur) * tt;
                count = count + 1u;
            }
        }
    }
    return count;
}

@compute @workgroup_size(64)
fn narrowphase_obb_obb_manifold(@builtin(global_invocation_id) gid: vec3<u32>) {
    let gi = gid.x;
    if (gi >= params.num_pairs) {
        return;
    }

    let couple = pairs[gi];
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
    let ca = ba.center.xyz;
    let cb = bb.center.xyz;
    let t = cb - ca;

    var out: Manifold;
    out.normal_count = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    out.p0 = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    out.p1 = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    out.p2 = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    out.p3 = vec4<f32>(0.0, 0.0, 0.0, 0.0);

    // --- Fifteen-axis SAT, recording the winning axis index. ---
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
    var best_index = 0;
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
            best_index = idx;
        }
    }

    if (separated) {
        manifolds[gi] = out;
        return;
    }

    var normal = best_axis;
    if (dot(t, best_axis) < 0.0) {
        normal = -best_axis;
    }

    // --- Edge-edge contact: a single closest-point pair. ---
    if (best_index >= 6) {
        let e = u32(best_index) - 6u;
        let i = e / 3u;
        let j = e % 3u;
        let ai0 = (i + 1u) % 3u;
        let ai1 = (i + 2u) % 3u;
        let av0 = box_axis(a0, a1, a2, ai0);
        let av1 = box_axis(a0, a1, a2, ai1);
        let base_a = ca
            + av0 * (sign_pos(dot(normal, av0)) * box_he(ea, ai0))
            + av1 * (sign_pos(dot(normal, av1)) * box_he(ea, ai1));
        let eav = box_axis(a0, a1, a2, i) * box_he(ea, i);
        let pa0 = base_a - eav;
        let pa1 = base_a + eav;

        let bj0 = (j + 1u) % 3u;
        let bj1 = (j + 2u) % 3u;
        let bv0 = box_axis(b0, b1, b2, bj0);
        let bv1 = box_axis(b0, b1, b2, bj1);
        let base_b = cb
            + bv0 * (sign_pos(dot(-normal, bv0)) * box_he(eb, bj0))
            + bv1 * (sign_pos(dot(-normal, bv1)) * box_he(eb, bj1));
        let ebv = box_axis(b0, b1, b2, j) * box_he(eb, j);
        let pb0 = base_b - ebv;
        let pb1 = base_b + ebv;

        let seg = closest_points_segments(pa0, pa1, pb0, pb1);
        let point = (seg.ca + seg.cb) * 0.5;
        out.normal_count = vec4<f32>(normal, 1.0);
        out.p0 = vec4<f32>(point, best_overlap);
        manifolds[gi] = out;
        return;
    }

    // --- Face contact: reference/incident face clipping. ---
    // Reference box/face vs incident box; rn is the reference outward normal.
    var rc = ca;
    var r0 = a0;
    var r1 = a1;
    var r2 = a2;
    var rhe = ea;
    var ic = cb;
    var i0 = b0;
    var i1 = b1;
    var i2 = b2;
    var ihe = eb;
    var rn = normal;
    var ref_axis = u32(best_index);
    if (best_index >= 3) {
        rc = cb; r0 = b0; r1 = b1; r2 = b2; rhe = eb;
        ic = ca; i0 = a0; i1 = a1; i2 = a2; ihe = ea;
        rn = -normal;
        ref_axis = u32(best_index) - 3u;
    }

    let ref_sign = sign_pos(dot(rn, box_axis(r0, r1, r2, ref_axis)));
    let ref_center = rc + box_axis(r0, r1, r2, ref_axis) * (ref_sign * box_he(rhe, ref_axis));

    // Incident face: the local axis of the incident box most anti-parallel to rn.
    var inc_axis = 0u;
    var inc_best = abs(dot(rn, i0));
    let d1i = abs(dot(rn, i1));
    let d2i = abs(dot(rn, i2));
    if (d1i > inc_best) { inc_best = d1i; inc_axis = 1u; }
    if (d2i > inc_best) { inc_best = d2i; inc_axis = 2u; }
    let inc_sign = -sign_pos(dot(rn, box_axis(i0, i1, i2, inc_axis)));

    // The four corners of the incident face, wound as a loop.
    let iu = (inc_axis + 1u) % 3u;
    let iv = (inc_axis + 2u) % 3u;
    let inc_center = ic + box_axis(i0, i1, i2, inc_axis) * (inc_sign * box_he(ihe, inc_axis));
    let idu = box_axis(i0, i1, i2, iu) * box_he(ihe, iu);
    let idv = box_axis(i0, i1, i2, iv) * box_he(ihe, iv);

    var buf_a: array<vec3<f32>, 8>;
    var buf_b: array<vec3<f32>, 8>;
    buf_a[0] = inc_center + idu + idv;
    buf_a[1] = inc_center - idu + idv;
    buf_a[2] = inc_center - idu - idv;
    buf_a[3] = inc_center + idu - idv;

    // The reference face's four side planes.
    let ru = (ref_axis + 1u) % 3u;
    let rv = (ref_axis + 2u) % 3u;
    let uax = box_axis(r0, r1, r2, ru);
    let vax = box_axis(r0, r1, r2, rv);
    let hu = box_he(rhe, ru);
    let hv = box_he(rhe, rv);

    var n = 4u;
    var from_a = true;
    // Plane 0: +u.
    if (from_a) { n = clip_to_plane(&buf_a, n, ref_center + uax * hu, uax, &buf_b); }
    else { n = clip_to_plane(&buf_b, n, ref_center + uax * hu, uax, &buf_a); }
    from_a = !from_a;
    if (n != 0u) {
        // Plane 1: -u.
        if (from_a) { n = clip_to_plane(&buf_a, n, ref_center - uax * hu, -uax, &buf_b); }
        else { n = clip_to_plane(&buf_b, n, ref_center - uax * hu, -uax, &buf_a); }
        from_a = !from_a;
    }
    if (n != 0u) {
        // Plane 2: +v.
        if (from_a) { n = clip_to_plane(&buf_a, n, ref_center + vax * hv, vax, &buf_b); }
        else { n = clip_to_plane(&buf_b, n, ref_center + vax * hv, vax, &buf_a); }
        from_a = !from_a;
    }
    if (n != 0u) {
        // Plane 3: -v.
        if (from_a) { n = clip_to_plane(&buf_a, n, ref_center - vax * hv, -vax, &buf_b); }
        else { n = clip_to_plane(&buf_b, n, ref_center - vax * hv, -vax, &buf_a); }
        from_a = !from_a;
    }

    // Gather the clipped polygon into a local array in the same order the CPU
    // reads it back.
    var poly: array<vec3<f32>, 8>;
    for (var g = 0u; g < n; g = g + 1u) {
        if (from_a) { poly[g] = buf_a[g]; } else { poly[g] = buf_b[g]; }
    }

    // Keep the corners that penetrate the reference face, on the mid-overlap
    // plane along the reference normal.
    var kept_pos: array<vec3<f32>, 8>;
    var kept_depth: array<f32, 8>;
    var kept_n = 0u;
    for (var g = 0u; g < n; g = g + 1u) {
        let corner = poly[g];
        let sep = dot(corner - ref_center, rn);
        if (sep <= 0.0) {
            let depth = -sep;
            kept_pos[kept_n] = corner + rn * (depth * 0.5);
            kept_depth[kept_n] = depth;
            kept_n = kept_n + 1u;
        }
    }

    if (kept_n == 0u) {
        // Numerical fallback: report the mid-overlap single point.
        let pa = support_vertex(ca, a0, a1, a2, ea, normal);
        let pb = support_vertex(cb, b0, b1, b2, eb, -normal);
        let point = (pa + pb) * 0.5;
        out.normal_count = vec4<f32>(normal, 1.0);
        out.p0 = vec4<f32>(point, best_overlap);
        manifolds[gi] = out;
        return;
    }

    // Reduce to at most four points bounding the contact polygon.
    var out_pos: array<vec3<f32>, 4>;
    var out_depth: array<f32, 4>;
    var out_n = 0u;
    if (kept_n <= 4u) {
        for (var g = 0u; g < kept_n; g = g + 1u) {
            out_pos[g] = kept_pos[g];
            out_depth[g] = kept_depth[g];
        }
        out_n = kept_n;
    } else {
        // Deepest corner anchors the quad.
        var q0 = 0u;
        for (var g = 0u; g < kept_n; g = g + 1u) {
            if (kept_depth[g] > kept_depth[q0]) { q0 = g; }
        }
        // Farthest corner from the anchor forms the diagonal.
        var q1 = 0u;
        var best_d = -1.0;
        for (var g = 0u; g < kept_n; g = g + 1u) {
            let d = dot(kept_pos[g] - kept_pos[q0], kept_pos[g] - kept_pos[q0]);
            if (d > best_d) { best_d = d; q1 = g; }
        }
        // The two corners maximising the signed area to either side of the diagonal.
        let edge = kept_pos[q1] - kept_pos[q0];
        var q2 = q0;
        var q3 = q0;
        var max_area = 0.0;
        var min_area = 0.0;
        for (var g = 0u; g < kept_n; g = g + 1u) {
            let area = dot(normal, cross(edge, kept_pos[g] - kept_pos[q0]));
            if (area > max_area) { max_area = area; q2 = g; }
            if (area < min_area) { min_area = area; q3 = g; }
        }
        let chosen = array<u32, 4>(q0, q1, q2, q3);
        for (var ci = 0u; ci < 4u; ci = ci + 1u) {
            let idx = chosen[ci];
            var dup = false;
            for (var cj = 0u; cj < out_n; cj = cj + 1u) {
                // Compare against already-accepted chosen indices.
                if (chosen[cj] == idx) { dup = true; }
            }
            if (!dup) {
                out_pos[out_n] = kept_pos[idx];
                out_depth[out_n] = kept_depth[idx];
                out_n = out_n + 1u;
            }
        }
    }

    out.normal_count = vec4<f32>(normal, f32(out_n));
    if (out_n > 0u) { out.p0 = vec4<f32>(out_pos[0], out_depth[0]); }
    if (out_n > 1u) { out.p1 = vec4<f32>(out_pos[1], out_depth[1]); }
    if (out_n > 2u) { out.p2 = vec4<f32>(out_pos[2], out_depth[2]); }
    if (out_n > 3u) { out.p3 = vec4<f32>(out_pos[3], out_depth[3]); }
    manifolds[gi] = out;
}
