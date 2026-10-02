// Capsule-versus-triangle narrow-phase contact kernel.
//
// One invocation per candidate pair. Each invocation reads a capsule (segment +
// radius) and a triangle (three world-space vertices), runs the same
// segment-versus-triangle closest-feature query and manifold construction the
// CPU twin runs (`narrowphase/capsule_triangle.rs`), and writes one contact
// slot: a unit normal pointing from the triangle toward the capsule (the
// push-out direction), the penetration depth, the world contact point on the
// triangle, and a validity flag. A pair that does not penetrate writes a zeroed
// slot with valid = 0, so the output index stays aligned with the input pair
// index.
//
// Geometry: find the point `s` on the capsule segment and the point `q` on the
// solid triangle that minimise their distance. First test whether the segment
// pierces the triangle face (crosses its plane inside the triangle); if so,
// `s == q == hit` and the pair is a degenerate deep contact. Otherwise take the
// minimum over five candidates: each segment endpoint projected onto the solid
// triangle with Ericson's Voronoi-region cascade, and the segment against each
// triangle edge with the clamped segment-segment routine. With `(s, q, d2)` the
// manifold is a sphere-against-triangle contact: off the triangle the pair
// contacts for a strict `dist < rc` with normal `(s - q) / dist` and depth
// `rc - dist`; on or through the triangle the normal falls back to the oriented
// geometric face normal with depth `rc`.
//
// The arithmetic is bit-for-bit with the twin apart from the square root and the
// handful of reciprocals in the segment/barycentric clamps, whose WGSL rounding
// differs in the low bits; the parity test therefore matches the validity flag
// exactly and the normal, depth, and point within a tight tolerance.
//
// Provenance: closest-point-on-triangle is the Voronoi-region method from
// Christer Ericson, *Real-Time Collision Detection* (2004), section 5.1.5; the
// clamped segment-segment routine is Ericson section 5.1.9; the segment-triangle
// pierce test is the textbook plane-crossing plus barycentric classification. No
// Unreal Engine source or derived code.

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

// One output contact: (normal.xyz, depth) and (point.xyz, valid).
struct Contact {
    normal_depth: vec4<f32>,
    point_valid: vec4<f32>,
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

@group(0) @binding(0) var<uniform> params: Params;
// Capsule proxies, one segment + radius each.
@group(0) @binding(1) var<storage, read> capsules: array<Capsule>;
// Triangles, three vec4s each.
@group(0) @binding(2) var<storage, read> triangles: array<Triangle>;
// Candidate pairs, one (capsule, triangle) index couple each.
@group(0) @binding(3) var<storage, read> pairs: array<vec2<u32>>;
// Output contacts, one slot per pair.
@group(0) @binding(4) var<storage, read_write> contacts: array<Contact>;

// Closest point on the solid triangle (a, b, c) to p, by Ericson's Voronoi-
// region cascade. Replicated operation for operation from the CPU twin's
// `closest_point_on_triangle`, including the fixed comparison order and the
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

// Closest point pair between segment (p1, q1) and segment (p2, q2). Every
// division is guarded by SEG_EPS2 so parallel and degenerate segments never
// divide by zero. Replicated operation for operation from the CPU twin's
// private copy.
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

@compute @workgroup_size(64)
fn narrowphase_capsule_triangle(@builtin(global_invocation_id) gid: vec3<u32>) {
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

    let feature = closest_segment_triangle(p0, p1, a, b, c);

    var out: Contact;

    if (feature.pierced == 0u && feature.d2 > COINCIDENT_EPS2) {
        let dist = sqrt(feature.d2);
        if (dist >= rc) {
            // Strict overlap: a grazing capsule carries no penetration.
            out.normal_depth = vec4<f32>(0.0, 0.0, 0.0, 0.0);
            out.point_valid = vec4<f32>(0.0, 0.0, 0.0, 0.0);
            contacts[i] = out;
            return;
        }
        let normal = (feature.s - feature.q) / dist;
        let depth = rc - dist;
        out.normal_depth = vec4<f32>(normal, depth);
        out.point_valid = vec4<f32>(feature.q, 1.0);
        contacts[i] = out;
        return;
    }

    // Axis on or through the triangle: fall back to the oriented face normal.
    let raw = cross(b - a, c - a);
    let len2 = dot(raw, raw);
    var normal = vec3<f32>(1.0, 0.0, 0.0);
    if (len2 > COINCIDENT_EPS2) {
        let mid = (p0 + p1) * 0.5;
        let unit = raw / sqrt(len2);
        if (dot(unit, mid - a) < 0.0) {
            normal = -unit;
        } else {
            normal = unit;
        }
    }
    out.normal_depth = vec4<f32>(normal, rc);
    out.point_valid = vec4<f32>(feature.q, 1.0);
    contacts[i] = out;
}
