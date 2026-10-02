// OBB-versus-triangle multi-point contact-manifold kernel (reference-face clipping).
//
// One invocation per candidate (box, triangle) couple. Each invocation runs the
// same thirteen-axis separating-axis test the single-point kernel runs
// (narrowphase_obb_triangle.wgsl), records which axis won, and then promotes
// that one contact to a full manifold exactly as the CPU twin does
// (narrowphase/obb_triangle_manifold.rs):
//
//   * a box-face axis (winning index 0..2) treats the box face turned toward the
//     triangle as the reference face and clips the triangle against the four side
//     planes of the box's local slab with the Sutherland-Hodgman algorithm;
//   * the triangle-face axis (winning index 3) treats the triangle as the
//     reference face and clips the box's most anti-parallel incident face quad
//     against the three edge side planes of the triangle;
//   * an edge-edge axis (winning index 4..12) degenerates to the single
//     representative closest-point pair, identical to the single-point path.
//
// Survivors that penetrate the reference face become the manifold points, placed
// on the mid-overlap plane along the contact normal; more than four are reduced
// to the four that best bound the contact polygon. When clipping leaves no
// penetrating corner, the representative point is used so a reported manifold
// always carries at least one live point.
//
// The output is one fixed-stride manifold record per couple: the shared normal
// (from the triangle toward the box) with the live point count in its w lane,
// then four (position.xyz, depth) points. A separated or grazing couple writes
// count = 0.
//
// The arithmetic is operation-for-operation with the twin: the same thirteen
// axes in the same order with the same CROSS_EPS2 guard and EDGE_BIAS face
// preference, the same reference/incident selection, the same inward-keep
// Sutherland-Hodgman clip in the same plane order over the same ping-pong
// buffers, the same strict pen > 0 penetration filter, the same mid-overlap
// placement, and the same four-point reduction with the same tie-breaks. Only
// the reciprocal square roots in the axis normalise and the barycentric
// reciprocals in the closest-point helper are inexact, so the parity test
// matches the count exactly and the normal, positions, and depths to within a
// tight tolerance.
//
// Provenance: the thirteen-axis box-triangle separating-axis set is Tomas
// Akenine-Moller, *Fast 3D Triangle-Box Overlap Testing* (2001); the
// reference/incident face-clipping contact manifold, the Sutherland-Hodgman
// polygon clip, the four-point reduction, and the closest-point-on-triangle
// Voronoi cascade are Christer Ericson, *Real-Time Collision Detection* (2004),
// sections 5.4, 8.3, and 5.1.5. No Unreal Engine source or derived code.

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

// One triangle collider: one vertex in the xyz of each row, the w lanes unused.
struct Triangle {
    a: vec4<f32>,
    b: vec4<f32>,
    c: vec4<f32>,
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

// Squared cross-length threshold below which an edge-edge axis (or the triangle
// normal) is degenerate and skipped; identical to CROSS_EPS2 in the CPU twin.
const CROSS_EPS2: f32 = 1.0e-12;
// Comparison-only penalty preferring face axes over near-equal edge axes.
const EDGE_BIAS: f32 = 1.0e-5;
// Sentinel overlap for a skipped (degenerate) axis.
const SKIP_OVERLAP: f32 = 1.0e30;
// Upper bound on corners the Sutherland-Hodgman clip can produce (a four-vertex
// incident polygon clipped by four half-planes).
const MAX_CLIP_POINTS: u32 = 8u;

@group(0) @binding(0) var<uniform> params: Params;
// Oriented bounding boxes, four vec4s each.
@group(0) @binding(1) var<storage, read> boxes: array<Obb>;
// Triangles, three vec4s each.
@group(0) @binding(2) var<storage, read> triangles: array<Triangle>;
// Candidate couples, one (box, triangle) index pair each.
@group(0) @binding(3) var<storage, read> pairs: array<vec2<u32>>;
// Output manifolds, one fixed-stride record per couple.
@group(0) @binding(4) var<storage, read_write> manifolds: array<Manifold>;

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

// Closest point on the solid triangle (a, b, c) to p, by Ericson's Voronoi-
// region cascade. Replicated operation for operation from the CPU twin's
// `closest_point_on_triangle`, including the fixed comparison order and the
// barycentric reciprocals.
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

// Clips convex polygon `src` (its first `n` vertices) against the half-space
// dot(v - point, inward) >= 0, writing survivors into `dst` and returning the
// new vertex count. Standard Sutherland-Hodgman with the inward-keep convention
// of the CPU twin's `clip_halfspace` (a vertex is kept when it lies on the
// inward side; each crossing edge contributes its interpolated intersection).
fn clip_inward(
    src: ptr<function, array<vec3<f32>, 8>>,
    n: u32,
    point: vec3<f32>,
    inward: vec3<f32>,
    dst: ptr<function, array<vec3<f32>, 8>>,
) -> u32 {
    var count = 0u;
    for (var i = 0u; i < n; i = i + 1u) {
        let cur = (*src)[i];
        let nxt = (*src)[(i + 1u) % n];
        let dc = dot(cur - point, inward);
        let dn = dot(nxt - point, inward);
        let cur_in = dc >= 0.0;
        let nxt_in = dn >= 0.0;
        if (cur_in && count < MAX_CLIP_POINTS) {
            (*dst)[count] = cur;
            count = count + 1u;
        }
        if ((cur_in != nxt_in) && count < MAX_CLIP_POINTS) {
            // dc != dn whenever the inside flags differ, so denom is non-zero.
            let denom = dc - dn;
            let t = dc / denom;
            (*dst)[count] = cur + (nxt - cur) * t;
            count = count + 1u;
        }
    }
    return count;
}

@compute @workgroup_size(64)
fn narrowphase_obb_triangle_manifold(@builtin(global_invocation_id) gid: vec3<u32>) {
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

    var out: Manifold;
    out.normal_count = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    out.p0 = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    out.p1 = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    out.p2 = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    out.p3 = vec4<f32>(0.0, 0.0, 0.0, 0.0);

    // --- Thirteen-axis SAT, recording the winning axis index. ---
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
            let cc = cross(box_axes[ii], tri_edges[jj]);
            let len2 = dot(cc, cc);
            if (len2 < CROSS_EPS2) {
                valid[k] = false;
                axes[k] = vec3<f32>(0.0, 0.0, 0.0);
            } else {
                valid[k] = true;
                axes[k] = normalize(cc);
            }
            k = k + 1;
        }
    }

    let verts = array<vec3<f32>, 3>(va, vb, vc);

    var best_overlap = SKIP_OVERLAP;
    var best_cmp = SKIP_OVERLAP;
    var best_normal = vec3<f32>(0.0, 0.0, 0.0);
    var best_index = 0;
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

        // Orient the axis toward the shorter push (ties resolve to +L), so the
        // normal always points from the triangle toward the box.
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
            best_index = idx;
        }
    }

    if (separated) {
        manifolds[i] = out;
        return;
    }

    let normal = best_normal;

    // Build the clipped reference-face manifold points, keeping only the corners
    // that penetrate the reference face (strict pen > 0) on the mid-overlap
    // plane. An edge-edge axis (index >= 4) skips the clip and leaves kept_n = 0,
    // falling through to the single representative point below.
    var kept_pos: array<vec3<f32>, 8>;
    var kept_depth: array<f32, 8>;
    var kept_n = 0u;

    if (best_index <= 2) {
        // Box-face reference: clip the triangle against the box slab side planes.
        let axis = u32(best_index);
        let face_out_axis = box_axis(x0, x1, x2, axis);
        let s = sign_pos(dot(-normal, face_out_axis));
        let ref_out = face_out_axis * s;
        let ref_point = center + ref_out * box_he(he, axis);
        let u = (axis + 1u) % 3u;
        let v = (axis + 2u) % 3u;
        let au = box_axis(x0, x1, x2, u);
        let av = box_axis(x0, x1, x2, v);
        let hu = box_he(he, u);
        let hv = box_he(he, v);

        var buf_a: array<vec3<f32>, 8>;
        var buf_b: array<vec3<f32>, 8>;
        buf_a[0] = va;
        buf_a[1] = vb;
        buf_a[2] = vc;
        var n = 3u;

        // The four slab side planes, each (point, inward) with inward pointing
        // into the box, in the twin's order: +u, -u, +v, -v.
        var plane_pt: array<vec3<f32>, 4>;
        var plane_in: array<vec3<f32>, 4>;
        plane_pt[0] = center + au * hu; plane_in[0] = -au;
        plane_pt[1] = center - au * hu; plane_in[1] = au;
        plane_pt[2] = center + av * hv; plane_in[2] = -av;
        plane_pt[3] = center - av * hv; plane_in[3] = av;

        var from_a = true;
        for (var pi = 0u; pi < 4u; pi = pi + 1u) {
            if (n == 0u) {
                break;
            }
            if (from_a) {
                n = clip_inward(&buf_a, n, plane_pt[pi], plane_in[pi], &buf_b);
            } else {
                n = clip_inward(&buf_b, n, plane_pt[pi], plane_in[pi], &buf_a);
            }
            from_a = !from_a;
        }

        // Keep the clipped corners that penetrate the reference face.
        for (var g = 0u; g < n; g = g + 1u) {
            var p: vec3<f32>;
            if (from_a) { p = buf_a[g]; } else { p = buf_b[g]; }
            let pen = dot(ref_point - p, ref_out);
            if (pen > 0.0) {
                kept_pos[kept_n] = p + ref_out * (pen * 0.5);
                kept_depth[kept_n] = pen;
                kept_n = kept_n + 1u;
            }
        }
    } else if (best_index == 3) {
        // Triangle-face reference: clip the box incident face quad against the
        // triangle edge side planes.
        let ref_out = normal;
        let ref_point = va;

        // Incident box face: the one whose outward normal is most anti-parallel
        // to the contact normal (largest |dot(axis, normal)|).
        var axis = 0u;
        var best_abs = abs(dot(x0, normal));
        let d1a = abs(dot(x1, normal));
        if (d1a > best_abs) { best_abs = d1a; axis = 1u; }
        let d2a = abs(dot(x2, normal));
        if (d2a > best_abs) { best_abs = d2a; axis = 2u; }

        let s = -sign_pos(dot(box_axis(x0, x1, x2, axis), normal));
        let u = (axis + 1u) % 3u;
        let v = (axis + 2u) % 3u;
        let fc = center + box_axis(x0, x1, x2, axis) * (s * box_he(he, axis));
        let du = box_axis(x0, x1, x2, u) * box_he(he, u);
        let dv = box_axis(x0, x1, x2, v) * box_he(he, v);

        var buf_a: array<vec3<f32>, 8>;
        var buf_b: array<vec3<f32>, 8>;
        buf_a[0] = fc + du + dv;
        buf_a[1] = fc - du + dv;
        buf_a[2] = fc - du - dv;
        buf_a[3] = fc + du - dv;
        var n = 4u;

        // The three triangle edge side planes, each (point, inward) with inward
        // in the triangle plane pointing toward the opposite vertex.
        let tri_a = array<vec3<f32>, 3>(va, vb, vc);
        let tri_b = array<vec3<f32>, 3>(vb, vc, va);
        let tri_third = array<vec3<f32>, 3>(vc, va, vb);
        var plane_pt: array<vec3<f32>, 3>;
        var plane_in: array<vec3<f32>, 3>;
        for (var ei = 0u; ei < 3u; ei = ei + 1u) {
            var inward = cross(tri_b[ei] - tri_a[ei], ref_out);
            if (dot(inward, tri_third[ei] - tri_a[ei]) < 0.0) {
                inward = -inward;
            }
            plane_pt[ei] = tri_a[ei];
            plane_in[ei] = inward;
        }

        var from_a = true;
        for (var pi = 0u; pi < 3u; pi = pi + 1u) {
            if (n == 0u) {
                break;
            }
            if (from_a) {
                n = clip_inward(&buf_a, n, plane_pt[pi], plane_in[pi], &buf_b);
            } else {
                n = clip_inward(&buf_b, n, plane_pt[pi], plane_in[pi], &buf_a);
            }
            from_a = !from_a;
        }

        for (var g = 0u; g < n; g = g + 1u) {
            var p: vec3<f32>;
            if (from_a) { p = buf_a[g]; } else { p = buf_b[g]; }
            let pen = dot(ref_point - p, ref_out);
            if (pen > 0.0) {
                kept_pos[kept_n] = p + ref_out * (pen * 0.5);
                kept_depth[kept_n] = pen;
                kept_n = kept_n + 1u;
            }
        }
    }

    if (kept_n == 0u) {
        // Edge-edge contact or a numerical corner case: the single representative
        // closest-point pair, driven deepest into the triangle.
        let box_point = support_vertex(center, x0, x1, x2, he, -normal);
        let tri_point = closest_point_on_triangle(box_point, va, vb, vc);
        let point = (box_point + tri_point) * 0.5;
        out.normal_count = vec4<f32>(normal, 1.0);
        out.p0 = vec4<f32>(point, best_overlap);
        manifolds[i] = out;
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
            for (var cj = 0u; cj < ci; cj = cj + 1u) {
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
    manifolds[i] = out;
}
