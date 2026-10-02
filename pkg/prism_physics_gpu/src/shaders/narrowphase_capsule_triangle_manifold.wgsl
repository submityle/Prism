// Capsule-versus-triangle up-to-two-point manifold kernel.
//
// One invocation per candidate pair. Each invocation reads a capsule (segment +
// radius) and a triangle (three world-space vertices) and builds the same
// up-to-two-point contact manifold the CPU twin builds
// (`narrowphase/capsule_triangle_manifold.rs`), writing one manifold slot: a
// shared unit normal pointing from the triangle toward the capsule (the push-out
// direction), a live point count, and up to four (position, depth) points. A
// pair that does not penetrate writes a zero count, so the output index stays
// aligned with the input pair index.
//
// Geometry (mirrors the twin operation for operation):
//
//   1. Run the shared single-point closest-feature query to get the deepest
//      contact. No penetration there means no manifold (count = 0). That contact
//      is also the honest fallback whenever a stable second point cannot be
//      found.
//   2. Take the triangle's geometric face normal, oriented toward the capsule
//      segment midpoint so it pushes the capsule off the triangle. A degenerate
//      (zero-area) triangle has no face, so keep the single contact.
//   3. Clip the capsule axis segment, in its own parameter t in [0, 1], to the
//      triangle's three edge half-planes with the Liang-Barsky algorithm. Each
//      surviving boundary point is projected onto the face plane; its depth is
//      rc minus its signed height over the plane. Two live, distinct corners
//      make a two-point manifold; otherwise the manifold collapses to the single
//      deepest contact.
//
// The arithmetic is bit-for-bit with the twin apart from the square root and the
// handful of reciprocals in the segment/barycentric clamps and the clip
// crossings, whose WGSL rounding differs in the low bits; the parity test
// therefore matches the point count exactly and the normal, positions, and
// depths within a tight tolerance.
//
// Provenance: closest-point-on-triangle is the Voronoi-region method from
// Christer Ericson, *Real-Time Collision Detection* (2004); the clamped
// segment-segment routine is Ericson section 5.1.9; the Liang-Barsky segment
// clip against the triangle's edge half-planes is textbook. No Unreal Engine
// source or derived code.

struct Params {
    // Number of candidate pairs queued in the pair buffer.
    num_pairs: u32,
    // Padding to a 16-byte uniform stride.
    pad0: u32,
    pad1: u32,
    pad2: u32,
};

// Squared-length threshold below which a segment or solver denominator is
// degenerate; kept identical to `SEG_EPS2` in `narrowphase/capsule_triangle.rs`.
const SEG_EPS2: f32 = 1.0e-12;

// Squared-distance threshold below which the closest points are coincident;
// kept identical to `COINCIDENT_EPS2` in the same module.
const COINCIDENT_EPS2: f32 = 1.0e-12;

// Parameter-span threshold below which the clipped stretch collapses to a point;
// kept identical to `CLIP_T_EPS` in `narrowphase/capsule_triangle_manifold.rs`.
const CLIP_T_EPS: f32 = 1.0e-9;

// Squared-distance threshold below which the two clipped corners are coincident;
// kept identical to `SEP_EPS2` in the same manifold module.
const SEP_EPS2: f32 = 1.0e-12;

// Squared-length threshold below which the triangle face normal is degenerate;
// kept identical to `FACE_EPS2` in the same manifold module.
const FACE_EPS2: f32 = 1.0e-12;

// One capsule: (p0.xyz, radius) then (p1.xyz, unused pad).
struct Capsule {
    p0_radius: vec4<f32>,
    p1_pad: vec4<f32>,
};

// One triangle collider. Each vertex rides in the xyz of its row; the w lanes
// are unused padding to keep the triangle to three vec4s.
struct Triangle {
    a: vec4<f32>,
    b: vec4<f32>,
    c: vec4<f32>,
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

@group(0) @binding(0) var<uniform> params: Params;
// Capsule proxies, one segment + radius each.
@group(0) @binding(1) var<storage, read> capsules: array<Capsule>;
// Triangles, three vec4s each.
@group(0) @binding(2) var<storage, read> triangles: array<Triangle>;
// Candidate pairs, one (capsule, triangle) index couple each.
@group(0) @binding(3) var<storage, read> pairs: array<vec2<u32>>;
// Output manifolds, one slot per pair.
@group(0) @binding(4) var<storage, read_write> manifolds: array<Manifold>;

// Closest point on the solid triangle (a, b, c) to p, by Ericson's Voronoi-
// region cascade. Replicated operation for operation from the CPU twin.
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
// (a, b, c). Mirrors the CPU twin's `closest_segment_triangle`: pierce test
// first, then the fixed five-candidate cascade with strict < updates.
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
// `clip_segment_to_triangle`: the inward in-plane normal is oriented toward the
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

@compute @workgroup_size(64)
fn narrowphase_capsule_triangle_manifold(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.num_pairs) {
        return;
    }

    let pair = pairs[i];
    let cap = capsules[pair.x];
    let p0 = cap.p0_radius.xyz;
    let rc = cap.p0_radius.w;
    let p1 = cap.p1_pad.xyz;

    let tri = triangles[pair.y];
    let a = tri.a.xyz;
    let b = tri.b.xyz;
    let c = tri.c.xyz;

    var out: Manifold;
    out.normal_count = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    out.p0 = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    out.p1 = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    out.p2 = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    out.p3 = vec4<f32>(0.0, 0.0, 0.0, 0.0);

    // --- Step 1: single deepest contact (penetration gate and fallback). ---
    let feature = closest_segment_triangle(p0, p1, a, b, c);
    var c_normal = vec3<f32>(1.0, 0.0, 0.0);
    var c_depth = rc;
    var c_point = feature.q;
    if (feature.pierced == 0u && feature.d2 > COINCIDENT_EPS2) {
        let dist = sqrt(feature.d2);
        if (dist >= rc) {
            // Strict overlap: a grazing capsule carries no penetration.
            manifolds[i] = out;
            return;
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
    var out_p0 = vec4<f32>(c_point, c_depth);
    var out_p1 = vec4<f32>(0.0, 0.0, 0.0, 0.0);

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
                    out_p0 = pts[0];
                    out_p1 = pts[1];
                }
            }
        }
    }

    out.normal_count = vec4<f32>(out_normal, f32(out_count));
    out.p0 = out_p0;
    out.p1 = out_p1;
    manifolds[i] = out;
}
