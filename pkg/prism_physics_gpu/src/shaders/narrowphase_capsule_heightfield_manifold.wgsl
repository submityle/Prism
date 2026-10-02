// Capsule-versus-heightfield multi-point contact manifold kernel.
//
// One invocation per candidate (capsule, heightfield) pair. Each invocation
// reads a capsule (segment + radius) and a heightfield (a regular rows x cols
// grid of height samples spaced cell_size apart on the XZ plane, Y up), finds
// the grid cells overlapping the capsule's XZ footprint, and builds the same
// merged multi-point manifold the CPU twin builds
// (narrowphase/heightfield_capsule_manifold.rs): one shared unit normal
// pointing from the terrain toward the capsule, a live point count, and up to
// four (position, depth) points. A pair that touches no cell triangle writes a
// zero count, so the output index stays aligned with the input pair index.
//
// Algorithm (mirrors the twin operation for operation):
//
//   1. The capsule XZ footprint (its two endpoints' XZ box grown by the radius)
//      maps to the inclusive rectangle of candidate cells, rejecting a
//      footprint wholly off the grid exactly as Heightfield::xz_cell_range does.
//   2. Each candidate cell's two triangles, visited row-outer / column-inner /
//      triangle-zero-first, run through the shared capsule-versus-triangle
//      manifold (single deepest contact, oriented face, Liang-Barsky axis clip)
//      to produce a one- or two-point sub-manifold.
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
// bit-for-bit with the twin apart from the square root, the floor, and the
// barycentric reciprocals, whose WGSL rounding differs in the low bits; the
// parity test matches the point count exactly and the normal and points within
// a tight tolerance.
//
// Provenance: closest-point-on-triangle is the Voronoi-region method from
// Christer Ericson, *Real-Time Collision Detection* (2004); the clamped
// segment-segment routine is Ericson section 5.1.9; the Liang-Barsky segment
// clip and the four-point manifold reduction are textbook; the heightfield cell
// triangulation is textbook. No Unreal Engine source or derived code.

struct Params {
    // Number of candidate pairs queued in the pair buffer.
    num_pairs: u32,
    // Padding to a 16-byte uniform stride.
    pad0: u32,
    pad1: u32,
    pad2: u32,
};

// Squared-length threshold below which a segment or solver denominator is
// degenerate; kept identical to SEG_EPS2 in narrowphase/capsule_triangle.rs.
const SEG_EPS2: f32 = 1.0e-12;

// Squared-distance threshold below which the closest points are coincident;
// kept identical to COINCIDENT_EPS2 in the same module.
const COINCIDENT_EPS2: f32 = 1.0e-12;

// Parameter-span threshold below which the clipped stretch collapses to a point;
// kept identical to CLIP_T_EPS in narrowphase/capsule_triangle_manifold.rs.
const CLIP_T_EPS: f32 = 1.0e-9;

// Squared-distance threshold below which the two clipped corners are coincident;
// kept identical to SEP_EPS2 in the same manifold module.
const SEP_EPS2: f32 = 1.0e-12;

// Squared-length threshold below which the triangle face normal is degenerate;
// kept identical to FACE_EPS2 in the same manifold module.
const FACE_EPS2: f32 = 1.0e-12;

// Minimum cosine between two sub-manifold normals for them to merge; kept
// identical to COPLANAR_COS in narrowphase/heightfield_capsule_manifold.rs.
const COPLANAR_COS: f32 = 0.990;

// One capsule: (p0.xyz, radius) then (p1.xyz, unused pad).
struct Capsule {
    p0_radius: vec4<f32>,
    p1_pad: vec4<f32>,
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

// A closest-point pair between two segments.
struct ClosestPair {
    ca: vec3<f32>,
    cb: vec3<f32>,
};

// Segment-triangle closest feature: s on the segment, q on the triangle, their
// squared distance, and whether the segment pierced the face (pierced != 0).
struct SegmentTriangle {
    s: vec3<f32>,
    q: vec3<f32>,
    d2: f32,
    pierced: u32,
};

// Liang-Barsky clip progress: ok flag and the surviving [t_enter, t_leave] span.
struct ClipState {
    ok: u32,
    t_enter: f32,
    t_leave: f32,
};

// One cell triangle's sub-manifold: a shared normal, a live point count (0 means
// no contact), and up to two (position.xyz, depth) points.
struct TriManifold {
    normal: vec3<f32>,
    count: u32,
    pa: vec4<f32>,
    pb: vec4<f32>,
};

@group(0) @binding(0) var<uniform> params: Params;
// Capsule proxies, one segment + radius each.
@group(0) @binding(1) var<storage, read> capsules: array<Capsule>;
// Heightfield metadata, one Field each.
@group(0) @binding(2) var<storage, read> fields: array<Field>;
// Concatenated row-major height samples for every field.
@group(0) @binding(3) var<storage, read> heights: array<f32>;
// Candidate pairs, one (capsule, field) index couple each.
@group(0) @binding(4) var<storage, read> pairs: array<vec2<u32>>;
// Output manifolds, one slot per pair.
@group(0) @binding(5) var<storage, read_write> manifolds: array<Manifold>;

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

// Closest points between segments (p1, q1) and (p2, q2), clamped to both. From
// Ericson section 5.1.9, replicated operation for operation from the CPU twin.
fn closest_pt_segment_segment(p1: vec3<f32>, q1: vec3<f32>, p2: vec3<f32>, q2: vec3<f32>) -> ClosestPair {
    let d1 = q1 - p1;
    let d2 = q2 - p2;
    let r = p1 - p2;
    let a = dot(d1, d1);
    let e = dot(d2, d2);
    let f = dot(d2, r);

    var s: f32 = 0.0;
    var t: f32 = 0.0;
    if (a <= SEG_EPS2 && e <= SEG_EPS2) {
        s = 0.0;
        t = 0.0;
    } else if (a <= SEG_EPS2) {
        s = 0.0;
        t = clamp(f / e, 0.0, 1.0);
    } else {
        let c = dot(d1, r);
        if (e <= SEG_EPS2) {
            t = 0.0;
            s = clamp(-c / a, 0.0, 1.0);
        } else {
            let b = dot(d1, d2);
            let denom = a * e - b * b;
            var s0: f32 = 0.0;
            if (denom > SEG_EPS2) {
                s0 = clamp((b * f - c * e) / denom, 0.0, 1.0);
            }
            let t0 = (b * s0 + f) / e;
            if (t0 < 0.0) {
                t = 0.0;
                s = clamp(-c / a, 0.0, 1.0);
            } else if (t0 > 1.0) {
                t = 1.0;
                s = clamp((b - c) / a, 0.0, 1.0);
            } else {
                t = t0;
                s = s0;
            }
        }
    }

    var out: ClosestPair;
    out.ca = p1 + d1 * s;
    out.cb = p2 + d2 * t;
    return out;
}

// Closest feature between the capsule segment (p0, p1) and the solid triangle
// (a, b, c). Mirrors the CPU twin's closest_segment_triangle: pierce test first,
// then the fixed five-candidate cascade with strict < updates.
fn closest_segment_triangle(p0: vec3<f32>, p1: vec3<f32>, a: vec3<f32>, b: vec3<f32>, c: vec3<f32>) -> SegmentTriangle {
    var out: SegmentTriangle;

    // --- Pierce test. ---
    let n = cross(b - a, c - a);
    let pq = p1 - p0;
    let denom = dot(pq, n);
    if (abs(denom) > SEG_EPS2) {
        let tt = dot(a - p0, n) / denom;
        if (tt >= 0.0 && tt <= 1.0) {
            let hit = p0 + pq * tt;
            let v0 = b - a;
            let v1 = c - a;
            let v2 = hit - a;
            let d00 = dot(v0, v0);
            let d01 = dot(v0, v1);
            let d11 = dot(v1, v1);
            let d20 = dot(v2, v0);
            let d21 = dot(v2, v1);
            let bary_denom = d00 * d11 - d01 * d01;
            if (abs(bary_denom) > SEG_EPS2) {
                let v = (d11 * d20 - d01 * d21) / bary_denom;
                let w = (d00 * d21 - d01 * d20) / bary_denom;
                let u = 1.0 - v - w;
                if (u >= 0.0 && v >= 0.0 && w >= 0.0) {
                    out.s = hit;
                    out.q = hit;
                    out.d2 = 0.0;
                    out.pierced = 1u;
                    return out;
                }
            }
        }
    }

    // --- Minimum over five candidates. ---
    let q0 = closest_point_on_triangle(p0, a, b, c);
    var best_s = p0;
    var best_q = q0;
    var best_d2 = dot(p0 - q0, p0 - q0);

    let q1 = closest_point_on_triangle(p1, a, b, c);
    let d2_1 = dot(p1 - q1, p1 - q1);
    if (d2_1 < best_d2) {
        best_s = p1;
        best_q = q1;
        best_d2 = d2_1;
    }

    // Edge (a, b).
    let e_ab = closest_pt_segment_segment(p0, p1, a, b);
    let d2_ab = dot(e_ab.ca - e_ab.cb, e_ab.ca - e_ab.cb);
    if (d2_ab < best_d2) {
        best_s = e_ab.ca;
        best_q = e_ab.cb;
        best_d2 = d2_ab;
    }

    // Edge (b, c).
    let e_bc = closest_pt_segment_segment(p0, p1, b, c);
    let d2_bc = dot(e_bc.ca - e_bc.cb, e_bc.ca - e_bc.cb);
    if (d2_bc < best_d2) {
        best_s = e_bc.ca;
        best_q = e_bc.cb;
        best_d2 = d2_bc;
    }

    // Edge (c, a).
    let e_ca = closest_pt_segment_segment(p0, p1, c, a);
    let d2_ca = dot(e_ca.ca - e_ca.cb, e_ca.ca - e_ca.cb);
    if (d2_ca < best_d2) {
        best_s = e_ca.ca;
        best_q = e_ca.cb;
        best_d2 = d2_ca;
    }

    out.s = best_s;
    out.q = best_q;
    out.d2 = best_d2;
    out.pierced = 0u;
    return out;
}

// Clips the capsule axis against one triangle edge half-plane, updating the
// running [t_enter, t_leave] span. Mirrors one iteration of the CPU twin's
// clip_segment_to_triangle: the inward in-plane normal is oriented toward the
// opposite vertex, and the crossing raises t_enter or lowers t_leave.
fn clip_edge(st: ClipState, p0: vec3<f32>, seg: vec3<f32>, n: vec3<f32>, e0: vec3<f32>, e1: vec3<f32>, opp: vec3<f32>) -> ClipState {
    var out = st;
    if (out.ok == 0u) {
        return out;
    }
    var inward = cross(n, e1 - e0);
    if (dot(inward, opp - e0) < 0.0) {
        inward = -inward;
    }
    let d_enter = dot(p0 - e0, inward);
    let slope = dot(seg, inward);
    let p = -slope;
    let q = d_enter;
    if (p == 0.0) {
        if (q < 0.0) {
            out.ok = 0u;
        }
    } else if (p < 0.0) {
        let t = q / p;
        if (t > out.t_leave) {
            out.ok = 0u;
            return out;
        }
        if (t > out.t_enter) {
            out.t_enter = t;
        }
    } else {
        let t = q / p;
        if (t < out.t_enter) {
            out.ok = 0u;
            return out;
        }
        if (t < out.t_leave) {
            out.t_leave = t;
        }
    }
    return out;
}

// Builds the up-to-two-point capsule-versus-triangle sub-manifold for one cell
// triangle, matching the CPU twin's capsule_triangle_manifold. count = 0 means
// the capsule does not penetrate this triangle.
fn tri_manifold(p0: vec3<f32>, p1: vec3<f32>, rc: f32, a: vec3<f32>, b: vec3<f32>, c: vec3<f32>) -> TriManifold {
    var m: TriManifold;
    m.normal = vec3<f32>(0.0, 0.0, 0.0);
    m.count = 0u;
    m.pa = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    m.pb = vec4<f32>(0.0, 0.0, 0.0, 0.0);

    // --- Step 1: single deepest contact (penetration gate and fallback). ---
    let feature = closest_segment_triangle(p0, p1, a, b, c);
    var c_normal = vec3<f32>(1.0, 0.0, 0.0);
    var c_depth = rc;
    var c_point = feature.q;
    if (feature.pierced == 0u && feature.d2 > COINCIDENT_EPS2) {
        let dist = sqrt(feature.d2);
        if (dist >= rc) {
            // Strict overlap: a grazing capsule carries no penetration.
            return m;
        }
        c_normal = (feature.s - feature.q) / dist;
        c_depth = rc - dist;
        c_point = feature.q;
    } else {
        // Axis on or through the triangle: oriented geometric face normal.
        let raw = cross(b - a, c - a);
        let len2 = dot(raw, raw);
        if (len2 > COINCIDENT_EPS2) {
            let unit = raw / sqrt(len2);
            let mid = (p0 + p1) * 0.5;
            if (dot(unit, mid - a) < 0.0) {
                c_normal = -unit;
            } else {
                c_normal = unit;
            }
        }
        c_depth = rc;
        c_point = feature.q;
    }

    // Default to the honest single-point fallback.
    var out_normal = c_normal;
    var out_count = 1u;
    var out_pa = vec4<f32>(c_point, c_depth);
    var out_pb = vec4<f32>(0.0, 0.0, 0.0, 0.0);

    // --- Step 2: reference face. A degenerate triangle keeps the single point.
    let raw = cross(b - a, c - a);
    let len2 = dot(raw, raw);
    if (len2 > FACE_EPS2) {
        let unit = raw / sqrt(len2);
        let mid = (p0 + p1) * 0.5;
        var n = unit;
        if (dot(unit, mid - a) < 0.0) {
            n = -unit;
        }

        // --- Step 3: clip the capsule axis to the triangle footprint. ---
        let seg = p1 - p0;
        var clip: ClipState;
        clip.ok = 1u;
        clip.t_enter = 0.0;
        clip.t_leave = 1.0;
        clip = clip_edge(clip, p0, seg, n, a, b, c);
        clip = clip_edge(clip, p0, seg, n, b, c, a);
        clip = clip_edge(clip, p0, seg, n, c, a, b);

        if (clip.ok == 1u && (clip.t_leave - clip.t_enter) > CLIP_T_EPS) {
            // Project each clip boundary onto the face plane; keep the ones that
            // dip below it.
            var pts: array<vec4<f32>, 2>;
            var cnt = 0u;

            let axis0 = p0 + seg * clip.t_enter;
            let above0 = dot(axis0 - a, n);
            let depth0 = rc - above0;
            if (depth0 > 0.0) {
                pts[cnt] = vec4<f32>(axis0 - n * above0, depth0);
                cnt = cnt + 1u;
            }

            let axis1 = p0 + seg * clip.t_leave;
            let above1 = dot(axis1 - a, n);
            let depth1 = rc - above1;
            if (depth1 > 0.0) {
                pts[cnt] = vec4<f32>(axis1 - n * above1, depth1);
                cnt = cnt + 1u;
            }

            if (cnt == 2u) {
                let sep = pts[1].xyz - pts[0].xyz;
                if (dot(sep, sep) > SEP_EPS2) {
                    out_normal = n;
                    out_count = 2u;
                    out_pa = pts[0];
                    out_pb = pts[1];
                }
            }
        }
    }

    m.normal = out_normal;
    m.count = out_count;
    m.pa = out_pa;
    m.pb = out_pb;
    return m;
}

// Rebuilds the sub-manifold for the k-th candidate cell triangle in the fixed
// visit order (row outer, column inner, triangle zero before triangle one).
// k ranges over [0, triangle_count), where triangle_count is
// (max_row-min_row+1) * (max_col-min_col+1) * 2.
fn tri_manifold_at(fi: u32, p0: vec3<f32>, p1: vec3<f32>, rc: f32, min_row: u32, min_col: u32, cols_span: u32, k: u32) -> TriManifold {
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
    return tri_manifold(p0, p1, rc, a, b, c);
}

// Greatest penetration depth among a sub-manifold's live points.
fn sub_peak_depth(m: TriManifold) -> f32 {
    if (m.count == 2u) {
        return max(m.pa.w, m.pb.w);
    }
    return m.pa.w;
}

// Returns the j-th pooled coplanar point (position.xyz, depth) by re-walking the
// candidate triangles in order, counting the live points of every sub whose
// normal lies within COPLANAR_COS of ref_normal. j must be < the pooled count.
fn coplanar_point(fi: u32, p0: vec3<f32>, p1: vec3<f32>, rc: f32, min_row: u32, min_col: u32, cols_span: u32, tri_total: u32, ref_normal: vec3<f32>, j: u32) -> vec4<f32> {
    var seen = 0u;
    for (var k = 0u; k < tri_total; k = k + 1u) {
        let m = tri_manifold_at(fi, p0, p1, rc, min_row, min_col, cols_span, k);
        if (m.count == 0u) {
            continue;
        }
        if (dot(m.normal, ref_normal) < COPLANAR_COS) {
            continue;
        }
        if (seen == j) {
            return m.pa;
        }
        seen = seen + 1u;
        if (m.count == 2u) {
            if (seen == j) {
                return m.pb;
            }
            seen = seen + 1u;
        }
    }
    return vec4<f32>(0.0, 0.0, 0.0, 0.0);
}

@compute @workgroup_size(64)
fn narrowphase_capsule_heightfield_manifold(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.num_pairs) {
        return;
    }

    let pair = pairs[i];
    let cap = capsules[pair.x];
    let p0 = cap.p0_radius.xyz;
    let rc = cap.p0_radius.w;
    let p1 = cap.p1_pad.xyz;

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
    let span_x = f32(cols - 1u) * cell;
    let span_z = f32(rows - 1u) * cell;
    let min_x = min(p0.x, p1.x) - rc;
    let max_x = max(p0.x, p1.x) + rc;
    let min_z = min(p0.z, p1.z) - rc;
    let max_z = max(p0.z, p1.z) + rc;
    if (max_x < ox || min_x > ox + span_x || max_z < oz || min_z > oz + span_z) {
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
        let m = tri_manifold_at(fi, p0, p1, rc, min_row, min_col, cols_span, k);
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
        let m = tri_manifold_at(fi, p0, p1, rc, min_row, min_col, cols_span, k);
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
        // Four or fewer: keep every pooled point in its original order.
        for (var j = 0u; j < n; j = j + 1u) {
            stored[j] = coplanar_point(fi, p0, p1, rc, min_row, min_col, cols_span, tri_total, ref_normal, j);
        }
        out_count = n;
    } else {
        // Deepest corner anchors the quad (strictly-greater, first wins).
        var i0 = 0u;
        var best = coplanar_point(fi, p0, p1, rc, min_row, min_col, cols_span, tri_total, ref_normal, 0u).w;
        for (var j = 1u; j < n; j = j + 1u) {
            let d = coplanar_point(fi, p0, p1, rc, min_row, min_col, cols_span, tri_total, ref_normal, j).w;
            if (d > best) {
                best = d;
                i0 = j;
            }
        }
        let anchor = coplanar_point(fi, p0, p1, rc, min_row, min_col, cols_span, tri_total, ref_normal, i0).xyz;

        // Corner farthest from the anchor.
        var i1 = i0;
        var best_d2 = -1.0;
        for (var j = 0u; j < n; j = j + 1u) {
            let pos = coplanar_point(fi, p0, p1, rc, min_row, min_col, cols_span, tri_total, ref_normal, j).xyz;
            let d2 = dot(pos - anchor, pos - anchor);
            if (d2 > best_d2) {
                best_d2 = d2;
                i1 = j;
            }
        }
        let far = coplanar_point(fi, p0, p1, rc, min_row, min_col, cols_span, tri_total, ref_normal, i1).xyz;
        let diag = far - anchor;

        // Corners of maximal signed area to either side of the diagonal.
        var i2 = i0;
        var i3 = i0;
        var best_pos = 0.0;
        var best_neg = 0.0;
        for (var j = 0u; j < n; j = j + 1u) {
            let pos = coplanar_point(fi, p0, p1, rc, min_row, min_col, cols_span, tri_total, ref_normal, j).xyz;
            let area = dot(cross(diag, pos - anchor), ref_normal);
            if (area > best_pos) {
                best_pos = area;
                i2 = j;
            } else if (area < best_neg) {
                best_neg = area;
                i3 = j;
            }
        }

        // Deduplicate [i0, i1, i2, i3] by index, preserving order.
        var idxs = array<u32, 4>(i0, i1, i2, i3);
        var chosen = array<u32, 4>(0u, 0u, 0u, 0u);
        var nc = 0u;
        for (var k = 0u; k < 4u; k = k + 1u) {
            let id = idxs[k];
            var dup = false;
            for (var m = 0u; m < nc; m = m + 1u) {
                if (chosen[m] == id) {
                    dup = true;
                }
            }
            if (!dup) {
                chosen[nc] = id;
                nc = nc + 1u;
            }
        }
        for (var k = 0u; k < nc; k = k + 1u) {
            stored[k] = coplanar_point(fi, p0, p1, rc, min_row, min_col, cols_span, tri_total, ref_normal, chosen[k]);
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
