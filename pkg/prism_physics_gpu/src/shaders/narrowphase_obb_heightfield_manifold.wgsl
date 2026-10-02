// OBB-versus-heightfield multi-point contact manifold kernel.
//
// One invocation per candidate (box, heightfield) pair. Each invocation reads an
// oriented bounding box (centre, three local axes, three half extents) and a
// heightfield (a regular rows x cols grid of height samples spaced cell_size
// apart on the XZ plane, Y up), finds the grid cells overlapping the box's XZ
// footprint, and builds the same merged multi-point manifold the CPU twin builds
// (narrowphase/heightfield_obb_manifold.rs): one shared unit normal pointing
// from the terrain toward the box, a live point count, and up to four
// (position, depth) points. A pair that touches no cell triangle writes a zero
// count, so the output index stays aligned with the input pair index.
//
// Algorithm (mirrors the twin operation for operation):
//
//   1. The box XZ footprint is the axis-aligned box its eight corners project to
//      on the world XZ plane, computed as the centre plus the per-axis absolute
//      projection of the half extents. That box maps to the inclusive rectangle
//      of candidate cells, rejecting a footprint wholly off the grid exactly as
//      Heightfield::xz_cell_range does.
//   2. Each candidate cell's two triangles, visited row-outer / column-inner /
//      triangle-zero-first, run through the shared box-versus-triangle manifold
//      (thirteen-axis SAT, reference/incident face Sutherland-Hodgman clip,
//      four-point reduction) to produce a one- to four-point sub-manifold.
//   3. The sub-manifold carrying the globally deepest point (strictly-greater,
//      first wins) sets the reference normal.
//   4. Every sub whose normal lies within COPLANAR_COS of the reference pools
//      all its points; the pool is reduced to the widest, deepest four with the
//      shared four-point reduction (deepest corner, farthest corner, then the
//      two corners of maximal signed area to either side of that diagonal,
//      deduplicated by index).
//
// Because there is no dynamic storage on the device, the pool is never
// materialised: the cell loop is re-walked for each query the reduction needs
// (reference pick, total count, and each indexed point lookup), which keeps the
// result bit-stable and independent of invocation scheduling. The arithmetic is
// bit-for-bit with the twin apart from the reciprocal square roots in the axis
// normalise and the barycentric reciprocals in the closest-point helper, whose
// WGSL rounding differs in the low bits; the parity test matches the point count
// exactly and the normal and points within a tight tolerance.
//
// Provenance: the thirteen-axis box-triangle separating-axis set is Tomas
// Akenine-Moller, *Fast 3D Triangle-Box Overlap Testing* (2001); the
// reference/incident face-clipping contact manifold, the Sutherland-Hodgman
// polygon clip, the four-point reduction, and the closest-point-on-triangle
// Voronoi cascade are Christer Ericson, *Real-Time Collision Detection* (2004),
// sections 5.4, 8.3, and 5.1.5; the heightfield cell triangulation is textbook.
// No Unreal Engine source or derived code.

struct Params {
    // Number of candidate pairs queued in the pair buffer.
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

// One heightfield's metadata. dims is (rows, cols, heights_offset, pad); geom is
// (cell_size, origin.x, origin.y, origin.z). The field's samples live in the
// shared heights array starting at heights_offset, row-major.
struct Field {
    dims: vec4<u32>,
    geom: vec4<f32>,
};

// One output manifold slot: (normal.xyz, count) then four (position.xyz, depth).
struct Manifold {
    normal_count: vec4<f32>,
    p0: vec4<f32>,
    p1: vec4<f32>,
    p2: vec4<f32>,
    p3: vec4<f32>,
};

// One cell-triangle sub-manifold: a shared normal, a live point count (0..=4),
// and four (position.xyz, depth) points. count = 0 means the box does not
// penetrate this triangle.
struct TriManifold {
    normal: vec3<f32>,
    count: u32,
    pa: vec4<f32>,
    pb: vec4<f32>,
    pc: vec4<f32>,
    pd: vec4<f32>,
};

// Squared cross-length threshold below which an edge-edge axis (or the triangle
// normal) is degenerate and skipped; identical to CROSS_EPS2 in the CPU twin.
const CROSS_EPS2: f32 = 1.0e-12;
// Comparison-only penalty preferring face axes over near-equal edge axes.
const EDGE_BIAS: f32 = 1.0e-5;
// Sentinel overlap for a skipped (degenerate) axis.
const SKIP_OVERLAP: f32 = 1.0e30;
// Upper bound on corners the Sutherland-Hodgman clip can produce.
const MAX_CLIP_POINTS: u32 = 8u;
// Minimum cosine between two sub-manifold normals for them to merge; kept
// identical to COPLANAR_COS in narrowphase/heightfield_obb_manifold.rs.
const COPLANAR_COS: f32 = 0.990;

@group(0) @binding(0) var<uniform> params: Params;
// Oriented bounding boxes, four vec4s each.
@group(0) @binding(1) var<storage, read> boxes: array<Obb>;
// Heightfield metadata, one Field each.
@group(0) @binding(2) var<storage, read> fields: array<Field>;
// Concatenated row-major height samples for every field.
@group(0) @binding(3) var<storage, read> heights: array<f32>;
// Candidate pairs, one (box, field) index couple each.
@group(0) @binding(4) var<storage, read> pairs: array<vec2<u32>>;
// Output manifolds, one slot per pair.
@group(0) @binding(5) var<storage, read_write> manifolds: array<Manifold>;

// Returns +1.0 when x >= 0.0, else -1.0 (matches sign_pos in the CPU twin).
fn sign_pos(x: f32) -> f32 {
    return select(-1.0, 1.0, x >= 0.0);
}

// Projected half-width of a box (axes a0/a1/a2, half extents he) onto axis.
fn projected_radius(axis: vec3<f32>, a0: vec3<f32>, a1: vec3<f32>, a2: vec3<f32>, he: vec3<f32>) -> f32 {
    return abs(dot(axis, a0)) * he.x + abs(dot(axis, a1)) * he.y + abs(dot(axis, a2)) * he.z;
}

// Support vertex of a box in direction dir.
fn support_vertex(center: vec3<f32>, a0: vec3<f32>, a1: vec3<f32>, a2: vec3<f32>, he: vec3<f32>, dir: vec3<f32>) -> vec3<f32> {
    return center
        + a0 * (sign_pos(dot(dir, a0)) * he.x)
        + a1 * (sign_pos(dot(dir, a1)) * he.y)
        + a2 * (sign_pos(dot(dir, a2)) * he.z);
}

// The local axis i of a box, indexed 0/1/2.
fn box_axis(a0: vec3<f32>, a1: vec3<f32>, a2: vec3<f32>, i: u32) -> vec3<f32> {
    if (i == 0u) { return a0; }
    if (i == 1u) { return a1; }
    return a2;
}

// The half extent along local axis i of a box, indexed 0/1/2.
fn box_he(he: vec3<f32>, i: u32) -> f32 {
    if (i == 0u) { return he.x; }
    if (i == 1u) { return he.y; }
    return he.z;
}

// Closest point on the solid triangle (a, b, c) to p, by Ericson's Voronoi-
// region cascade. Replicated operation for operation from the CPU twin's
// closest_point_on_triangle, including the fixed comparison order and the
// barycentric reciprocals.
fn closest_point_on_triangle(p: vec3<f32>, a: vec3<f32>, b: vec3<f32>, c: vec3<f32>) -> vec3<f32> {
    let ab = b - a;
    let ac = c - a;
    let ap = p - a;

    let d1 = dot(ab, ap);
    let d2 = dot(ac, ap);
    if (d1 <= 0.0 && d2 <= 0.0) {
        return a;
    }

    let bp = p - b;
    let d3 = dot(ab, bp);
    let d4 = dot(ac, bp);
    if (d3 >= 0.0 && d4 <= d3) {
        return b;
    }

    let vc = d1 * d4 - d3 * d2;
    if (vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0) {
        let v = d1 / (d1 - d3);
        return a + ab * v;
    }

    let cp = p - c;
    let d5 = dot(ab, cp);
    let d6 = dot(ac, cp);
    if (d6 >= 0.0 && d5 <= d6) {
        return c;
    }

    let vb = d5 * d2 - d1 * d6;
    if (vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0) {
        let w = d2 / (d2 - d6);
        return a + ac * w;
    }

    let va = d3 * d6 - d5 * d4;
    if (va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0) {
        let w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        return b + (c - b) * w;
    }

    let denom = 1.0 / (va + vb + vc);
    let v = vb * denom;
    let w = vc * denom;
    return a + ab * v + ac * w;
}

// Clips convex polygon src (its first n vertices) against the half-space
// dot(v - point, inward) >= 0, writing survivors into dst and returning the new
// vertex count. Standard Sutherland-Hodgman with the inward-keep convention of
// the CPU twin's clip_halfspace.
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
            let denom = dc - dn;
            let t = dc / denom;
            (*dst)[count] = cur + (nxt - cur) * t;
            count = count + 1u;
        }
    }
    return count;
}

// World-space position of a field's sample at grid index (r, c), matching
// Heightfield::vertex.
fn field_vertex(fi: u32, r: u32, c: u32) -> vec3<f32> {
    let f = fields[fi];
    let cols = f.dims.y;
    let off = f.dims.z;
    let h = heights[off + r * cols + c];
    let cell = f.geom.x;
    return vec3<f32>(
        f.geom.y + f32(c) * cell,
        f.geom.z + h,
        f.geom.w + f32(r) * cell,
    );
}

// Floors value to a cell index and clamps it to [0, last], matching the CPU
// twin's clamp_index.
fn clamp_index(value: f32, last: u32) -> u32 {
    let floored = floor(value);
    if (floored <= 0.0) {
        return 0u;
    }
    if (floored >= f32(last)) {
        return last;
    }
    return u32(floored);
}

// Builds the up-to-four-point box-versus-triangle sub-manifold for one cell
// triangle, matching the CPU twin's obb_triangle_manifold. count = 0 means the
// box does not penetrate this triangle.
fn obb_tri_manifold(
    center: vec3<f32>,
    x0: vec3<f32>,
    x1: vec3<f32>,
    x2: vec3<f32>,
    he: vec3<f32>,
    va: vec3<f32>,
    vb: vec3<f32>,
    vc: vec3<f32>,
) -> TriManifold {
    var m: TriManifold;
    m.normal = vec3<f32>(0.0, 0.0, 0.0);
    m.count = 0u;
    m.pa = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    m.pb = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    m.pc = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    m.pd = vec4<f32>(0.0, 0.0, 0.0, 0.0);

    let e0 = vb - va;
    let e1 = vc - vb;
    let e2 = va - vc;

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

        let cb = dot(axis, center);
        let rb = projected_radius(axis, x0, x1, x2, he);
        let bmin = cb - rb;
        let bmax = cb + rb;

        let t0 = dot(axis, verts[0]);
        let t1 = dot(axis, verts[1]);
        let t2 = dot(axis, verts[2]);
        let tmin = min(t0, min(t1, t2));
        let tmax = max(t0, max(t1, t2));

        let pen_l = tmax - bmin;
        let pen_r = bmax - tmin;
        let overlap = min(pen_l, pen_r);
        if (overlap <= 0.0) {
            separated = true;
        }

        var oriented = axis;
        if (pen_l > pen_r) {
            oriented = -axis;
        }

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
        return m;
    }

    let normal = best_normal;

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

        for (var g = 0u; g < n; g = g + 1u) {
            var pt: vec3<f32>;
            if (from_a) { pt = buf_a[g]; } else { pt = buf_b[g]; }
            let pen = dot(ref_point - pt, ref_out);
            if (pen > 0.0) {
                kept_pos[kept_n] = pt + ref_out * (pen * 0.5);
                kept_depth[kept_n] = pen;
                kept_n = kept_n + 1u;
            }
        }
    } else if (best_index == 3) {
        // Triangle-face reference: clip the box incident face quad against the
        // triangle edge side planes.
        let ref_out = normal;
        let ref_point = va;

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
            var pt: vec3<f32>;
            if (from_a) { pt = buf_a[g]; } else { pt = buf_b[g]; }
            let pen = dot(ref_point - pt, ref_out);
            if (pen > 0.0) {
                kept_pos[kept_n] = pt + ref_out * (pen * 0.5);
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
        m.normal = normal;
        m.count = 1u;
        m.pa = vec4<f32>(point, best_overlap);
        return m;
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
        var q0 = 0u;
        for (var g = 0u; g < kept_n; g = g + 1u) {
            if (kept_depth[g] > kept_depth[q0]) { q0 = g; }
        }
        var q1 = 0u;
        var best_d = -1.0;
        for (var g = 0u; g < kept_n; g = g + 1u) {
            let d = dot(kept_pos[g] - kept_pos[q0], kept_pos[g] - kept_pos[q0]);
            if (d > best_d) { best_d = d; q1 = g; }
        }
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

    m.normal = normal;
    m.count = out_n;
    if (out_n > 0u) { m.pa = vec4<f32>(out_pos[0], out_depth[0]); }
    if (out_n > 1u) { m.pb = vec4<f32>(out_pos[1], out_depth[1]); }
    if (out_n > 2u) { m.pc = vec4<f32>(out_pos[2], out_depth[2]); }
    if (out_n > 3u) { m.pd = vec4<f32>(out_pos[3], out_depth[3]); }
    return m;
}

// Rebuilds the sub-manifold for the k-th candidate cell triangle in the fixed
// visit order (row outer, column inner, triangle zero before triangle one).
fn tri_manifold_at(
    fi: u32,
    center: vec3<f32>,
    x0: vec3<f32>,
    x1: vec3<f32>,
    x2: vec3<f32>,
    he: vec3<f32>,
    min_row: u32,
    min_col: u32,
    cols_span: u32,
    k: u32,
) -> TriManifold {
    let per_row = cols_span * 2u;
    let row = min_row + k / per_row;
    let within = k % per_row;
    let col = min_col + within / 2u;
    let tri = within % 2u;

    let v00 = field_vertex(fi, row, col);
    let v01 = field_vertex(fi, row, col + 1u);
    let v10 = field_vertex(fi, row + 1u, col);
    let v11 = field_vertex(fi, row + 1u, col + 1u);

    var a = v00;
    var b = v10;
    var c = v11;
    if (tri == 1u) {
        a = v00;
        b = v11;
        c = v01;
    }
    return obb_tri_manifold(center, x0, x1, x2, he, a, b, c);
}

// The j-th live point of a sub-manifold, j in [0, count).
fn sub_point(m: TriManifold, j: u32) -> vec4<f32> {
    if (j == 0u) { return m.pa; }
    if (j == 1u) { return m.pb; }
    if (j == 2u) { return m.pc; }
    return m.pd;
}

// Greatest penetration depth among a sub-manifold's live points.
fn sub_peak_depth(m: TriManifold) -> f32 {
    var peak = m.pa.w;
    for (var j = 1u; j < m.count; j = j + 1u) {
        let d = sub_point(m, j).w;
        if (d > peak) {
            peak = d;
        }
    }
    return peak;
}

// Returns the j-th pooled coplanar point (position.xyz, depth) by re-walking the
// candidate triangles in order, counting the live points of every sub whose
// normal lies within COPLANAR_COS of ref_normal. j must be < the pooled count.
fn coplanar_point(
    fi: u32,
    center: vec3<f32>,
    x0: vec3<f32>,
    x1: vec3<f32>,
    x2: vec3<f32>,
    he: vec3<f32>,
    min_row: u32,
    min_col: u32,
    cols_span: u32,
    tri_total: u32,
    ref_normal: vec3<f32>,
    j: u32,
) -> vec4<f32> {
    var seen = 0u;
    for (var k = 0u; k < tri_total; k = k + 1u) {
        let m = tri_manifold_at(fi, center, x0, x1, x2, he, min_row, min_col, cols_span, k);
        if (m.count == 0u) {
            continue;
        }
        if (dot(m.normal, ref_normal) < COPLANAR_COS) {
            continue;
        }
        for (var pj = 0u; pj < m.count; pj = pj + 1u) {
            if (seen == j) {
                return sub_point(m, pj);
            }
            seen = seen + 1u;
        }
    }
    return vec4<f32>(0.0, 0.0, 0.0, 0.0);
}

@compute @workgroup_size(64)
fn narrowphase_obb_heightfield_manifold(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.num_pairs) {
        return;
    }

    let pair = pairs[i];
    let bx = boxes[pair.x];
    let center = bx.center.xyz;
    let x0 = bx.axis0.xyz;
    let x1 = bx.axis1.xyz;
    let x2 = bx.axis2.xyz;
    let he = vec3<f32>(bx.axis0.w, bx.axis1.w, bx.axis2.w);

    let fi = pair.y;
    let f = fields[fi];
    let rows = f.dims.x;
    let cols = f.dims.y;
    let cell = f.geom.x;
    let ox = f.geom.y;
    let oz = f.geom.w;

    var out: Manifold;
    out.normal_count = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    out.p0 = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    out.p1 = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    out.p2 = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    out.p3 = vec4<f32>(0.0, 0.0, 0.0, 0.0);

    // No complete cell, or the footprint lies wholly off the grid: no manifold.
    if (rows < 2u || cols < 2u) {
        manifolds[i] = out;
        return;
    }

    // Box XZ footprint: centre plus the per-axis absolute projection of the half
    // extents, matching obb_xz_footprint in the CPU twin.
    let span_box_x = he.x * abs(x0.x) + he.y * abs(x1.x) + he.z * abs(x2.x);
    let span_box_z = he.x * abs(x0.z) + he.y * abs(x1.z) + he.z * abs(x2.z);
    let min_x = center.x - span_box_x;
    let max_x = center.x + span_box_x;
    let min_z = center.z - span_box_z;
    let max_z = center.z + span_box_z;

    let span_grid_x = f32(cols - 1u) * cell;
    let span_grid_z = f32(rows - 1u) * cell;
    if (max_x < ox || min_x > ox + span_grid_x || max_z < oz || min_z > oz + span_grid_z) {
        manifolds[i] = out;
        return;
    }

    let last_col = cols - 2u;
    let last_row = rows - 2u;
    let inv = 1.0 / cell;
    let min_col = clamp_index((min_x - ox) * inv, last_col);
    let max_col = clamp_index((max_x - ox) * inv, last_col);
    let min_row = clamp_index((min_z - oz) * inv, last_row);
    let max_row = clamp_index((max_z - oz) * inv, last_row);

    let cols_span = max_col - min_col + 1u;
    let rows_span = max_row - min_row + 1u;
    let tri_total = rows_span * cols_span * 2u;

    // --- Pass 1: reference normal from the globally deepest sub point. ---
    var found = false;
    var best_depth = -1.0;
    var ref_normal = vec3<f32>(0.0, 0.0, 0.0);
    for (var k = 0u; k < tri_total; k = k + 1u) {
        let m = tri_manifold_at(fi, center, x0, x1, x2, he, min_row, min_col, cols_span, k);
        if (m.count == 0u) {
            continue;
        }
        let depth = sub_peak_depth(m);
        if (!found || depth > best_depth) {
            found = true;
            best_depth = depth;
            ref_normal = m.normal;
        }
    }
    if (!found) {
        manifolds[i] = out;
        return;
    }

    // --- Pass 2: total pooled coplanar point count. ---
    var n = 0u;
    for (var k = 0u; k < tri_total; k = k + 1u) {
        let m = tri_manifold_at(fi, center, x0, x1, x2, he, min_row, min_col, cols_span, k);
        if (m.count == 0u) {
            continue;
        }
        if (dot(m.normal, ref_normal) < COPLANAR_COS) {
            continue;
        }
        n = n + m.count;
    }

    // --- Reduction to the widest, deepest four. ---
    var stored: array<vec4<f32>, 4>;
    var out_count = 0u;

    if (n <= 4u) {
        for (var j = 0u; j < n; j = j + 1u) {
            stored[j] = coplanar_point(fi, center, x0, x1, x2, he, min_row, min_col, cols_span, tri_total, ref_normal, j);
        }
        out_count = n;
    } else {
        var i0 = 0u;
        var best = coplanar_point(fi, center, x0, x1, x2, he, min_row, min_col, cols_span, tri_total, ref_normal, 0u).w;
        for (var j = 1u; j < n; j = j + 1u) {
            let d = coplanar_point(fi, center, x0, x1, x2, he, min_row, min_col, cols_span, tri_total, ref_normal, j).w;
            if (d > best) {
                best = d;
                i0 = j;
            }
        }
        let anchor = coplanar_point(fi, center, x0, x1, x2, he, min_row, min_col, cols_span, tri_total, ref_normal, i0).xyz;

        var i1 = i0;
        var best_d2 = -1.0;
        for (var j = 0u; j < n; j = j + 1u) {
            let pos = coplanar_point(fi, center, x0, x1, x2, he, min_row, min_col, cols_span, tri_total, ref_normal, j).xyz;
            let d2 = dot(pos - anchor, pos - anchor);
            if (d2 > best_d2) {
                best_d2 = d2;
                i1 = j;
            }
        }
        let far = coplanar_point(fi, center, x0, x1, x2, he, min_row, min_col, cols_span, tri_total, ref_normal, i1).xyz;
        let diag = far - anchor;

        var i2 = i0;
        var i3 = i0;
        var best_pos = 0.0;
        var best_neg = 0.0;
        for (var j = 0u; j < n; j = j + 1u) {
            let pos = coplanar_point(fi, center, x0, x1, x2, he, min_row, min_col, cols_span, tri_total, ref_normal, j).xyz;
            let area = dot(cross(diag, pos - anchor), ref_normal);
            if (area > best_pos) {
                best_pos = area;
                i2 = j;
            } else if (area < best_neg) {
                best_neg = area;
                i3 = j;
            }
        }

        var idxs = array<u32, 4>(i0, i1, i2, i3);
        var chosen = array<u32, 4>(0u, 0u, 0u, 0u);
        var nc = 0u;
        for (var kk = 0u; kk < 4u; kk = kk + 1u) {
            let id = idxs[kk];
            var dup = false;
            for (var mm = 0u; mm < nc; mm = mm + 1u) {
                if (chosen[mm] == id) {
                    dup = true;
                }
            }
            if (!dup) {
                chosen[nc] = id;
                nc = nc + 1u;
            }
        }
        for (var kk = 0u; kk < nc; kk = kk + 1u) {
            stored[kk] = coplanar_point(fi, center, x0, x1, x2, he, min_row, min_col, cols_span, tri_total, ref_normal, chosen[kk]);
        }
        out_count = nc;
    }

    out.normal_count = vec4<f32>(ref_normal, f32(out_count));
    out.p0 = stored[0];
    out.p1 = stored[1];
    out.p2 = stored[2];
    out.p3 = stored[3];
    manifolds[i] = out;
}
