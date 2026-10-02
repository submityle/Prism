// Convex-versus-convex multi-point contact-manifold kernel (GJK + EPA +
// reference-face clipping).
//
// One invocation per (hull, hull) candidate couple. Each invocation runs the
// full narrow-phase pipeline of the CPU twin (narrowphase/convex_convex_manifold.rs)
// entirely on the device, with no dynamic allocation:
//
//   * GJK (narrowphase/gjk.rs) walks a one-to-four-point simplex of the
//     Minkowski difference toward the origin; disjoint couples write count = 0.
//   * EPA (narrowphase/epa.rs) blows the terminal simplex up to a tetrahedron
//     and expands it toward the origin's nearest face to recover the
//     minimum-translation normal (B toward A) and the penetration depth.
//   * The penetration is promoted to a manifold exactly as the twin does: pick
//     the reference face (the face on A or B most parallel to the push-out),
//     clip the incident hull's most anti-parallel face against the reference
//     face's side planes with the Sutherland-Hodgman algorithm, keep the
//     clipped corners that penetrate the reference face on the mid-overlap
//     plane, and reduce more than four survivors to the widest, deepest four.
//
// The output is one fixed-stride manifold record per couple: the shared normal
// (from hull a toward hull b) with the live point count in its w lane, then
// four (position.xyz, depth) points. A separated couple writes count = 0; a
// clip that keeps no penetrating corner, or an exact tangency EPA cannot grow a
// tetrahedron from, falls back to a single mid-overlap point so an overlapping
// couple is never dropped.
//
// All simplex, polytope, horizon, and clip scratch lives in fixed-size
// var<private> arrays (one instance per invocation); the EPA polytope is capped
// so a pathological hull terminates rather than overruns. The arithmetic is
// operation-for-operation with the twin: the same support argmax tie-break, the
// same GJK Voronoi sub-distance, the same EPA horizon re-triangulation, the same
// reference/incident selection, the same clip plane order, the same mid-overlap
// placement, and the same four-point reduction with the same tie-breaks. Only
// the normalise reciprocals and barycentric divisions are inexact, so parity
// matches the count exactly and the normal, positions, and depths to a tight
// tolerance (symmetric float-tie couples excepted).
//
// Provenance: Gilbert-Johnson-Keerthi distance (1988) with Ericson's Voronoi
// sub-distance (2005), the expanding-polytope algorithm (van den Bergen, 2001)
// with Ericson horizon re-triangulation, and a textbook reference/incident
// face-clipping manifold. No Unreal Engine source or derived code.

struct Params {
    // Number of candidate couples queued in the pair buffer.
    num_pairs: u32,
    // Padding to a 16-byte uniform stride.
    pad0: u32,
    pad1: u32,
    pad2: u32,
};

// One hull header: (vert_offset, vert_count, face_offset, face_count) packed as
// a vec4<u32>. The vertex/face slices of body i begin at those offsets in the
// flattened vertex and face arrays.
struct HullHeader {
    data: vec4<u32>,
};

// One hull face: outward unit normal in plane.xyz with the plane offset in
// plane.w, then (loop_offset, loop_count, pad, pad) in loop_info. The face's
// counter-clockwise loop of hull-local vertex indices begins at loop_offset in
// the flattened face-loop array.
struct GpuFace {
    plane: vec4<f32>,
    loop_info: vec4<u32>,
};

// One rigid pose: world translation in translation.xyz (w unused), rotation as
// a unit quaternion (x, y, z, w) in rotation.
struct GpuPose {
    translation: vec4<f32>,
    rotation: vec4<f32>,
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

@group(0) @binding(0) var<uniform> params: Params;
// Hull headers, one per body.
@group(0) @binding(1) var<storage, read> hulls: array<HullHeader>;
// Flattened hull vertices (xyz, w unused), each body's slice at its vert_offset.
@group(0) @binding(2) var<storage, read> vertices: array<vec4<f32>>;
// Flattened hull faces, each body's slice at its face_offset.
@group(0) @binding(3) var<storage, read> faces: array<GpuFace>;
// Flattened face-loop vertex indices (hull-local), each face's at loop_offset.
@group(0) @binding(4) var<storage, read> face_loops: array<u32>;
// Per-body rigid poses, indexed in lockstep with hull headers.
@group(0) @binding(5) var<storage, read> poses: array<GpuPose>;
// Candidate couples, one (body, body) index pair each.
@group(0) @binding(6) var<storage, read> pairs: array<vec2<u32>>;
// Output manifolds, one fixed-stride record per couple.
@group(0) @binding(7) var<storage, read_write> manifolds: array<Manifold>;


// ----------------------------------------------------------------------------
// Tunables (operation-for-operation with the CPU twin's constants).
// ----------------------------------------------------------------------------

// Squared length below which a vector is treated as the zero vector.
const ZERO_EPS2: f32 = 1.0e-12;
// Relative progress tolerance ending the GJK search when a support stops gaining.
const PROGRESS_TOL: f32 = 1.0e-8;
// Absolute growth tolerance ending EPA when the closest plane stops moving out.
const GROWTH_TOL: f32 = 1.0e-4;
// Sentinel plane distance marking a degenerate (never-chosen) polytope face.
const DEGENERATE_DISTANCE: f32 = 1.0e30;
// Hard iteration caps so a pathological hull terminates.
const MAX_GJK_ITERS: u32 = 64u;
const MAX_EPA_ITERS: u32 = 64u;
// Fixed capacities for the per-invocation polytope and clip scratch.
const MAX_EPA_VERTS: u32 = 32u;
const MAX_EPA_FACES: u32 = 64u;
const MAX_HORIZON: u32 = 48u;
const MAX_FACE_VERTS: u32 = 16u;
const MAX_CLIP_POINTS: u32 = 32u;

// ----------------------------------------------------------------------------
// Value types.
// ----------------------------------------------------------------------------

// One posed hull resolved from its header and pose, cached per invocation.
struct Body {
    vert_offset: u32,
    vert_count: u32,
    face_offset: u32,
    face_count: u32,
    pos: vec3<f32>,
    rot: vec4<f32>,
};

// One Minkowski-difference support point with its per-body world witnesses.
struct SupportPoint {
    diff: vec3<f32>,
    on_a: vec3<f32>,
    on_b: vec3<f32>,
};

// One triangular polytope face: vertex indices into the EPA vertex arrays, its
// outward unit normal, and the origin-to-plane distance.
struct Face {
    i0: u32,
    i1: u32,
    i2: u32,
    normal: vec3<f32>,
    distance: f32,
};

// The penetration recovered by EPA; ok = false signals an exact tangency.
struct Pen {
    ok: bool,
    normal: vec3<f32>,
    depth: f32,
    point_a: vec3<f32>,
    point_b: vec3<f32>,
};

// A triangle Voronoi classification: kind 0 vertex, 1 edge, 2 face. For a vertex
// the chosen local index is in a; for an edge the two local indices are a and b.
struct TriRegion {
    kind: u32,
    a: u32,
    b: u32,
};

// A simplex reduction: kc kept points (simplex-local indices k0,k1,k2) and the
// closest point of the kept feature to the origin.
struct TriReduce {
    kc: u32,
    k0: u32,
    k1: u32,
    k2: u32,
    closest: vec3<f32>,
};

// The result of a closest-face scan over the polytope.
struct ClosestFace {
    idx: u32,
    found: bool,
};

// A most-aligned-face result: the body-local face index and achieved alignment.
struct FaceAlign {
    idx: u32,
    dotv: f32,
};

// ----------------------------------------------------------------------------
// Per-invocation scratch (var<private>: one independent instance per lane).
// ----------------------------------------------------------------------------

var<private> body_a: Body;
var<private> body_b: Body;

// GJK simplex (one to four points): difference plus the two world witnesses.
var<private> gjk_diff: array<vec3<f32>, 4>;
var<private> gjk_a: array<vec3<f32>, 4>;
var<private> gjk_b: array<vec3<f32>, 4>;
var<private> gjk_len: u32;
var<private> gjk_closest: vec3<f32>;

// EPA polytope vertices (seeded from the simplex, then grown) and its faces.
var<private> epa_diff: array<vec3<f32>, 32>;
var<private> epa_a: array<vec3<f32>, 32>;
var<private> epa_b: array<vec3<f32>, 32>;
var<private> epa_vcount: u32;
var<private> epa_faces: array<Face, 64>;
var<private> epa_fcount: u32;

// Horizon edges collected while carving the polytope for a new vertex.
var<private> horizon: array<vec2<u32>, 48>;
var<private> horizon_count: u32;

// Manifold clip ping-pong buffers and the reference-face polygon.
var<private> clip_a: array<vec3<f32>, 32>;
var<private> clip_b: array<vec3<f32>, 32>;
var<private> ref_poly: array<vec3<f32>, 16>;
var<private> ref_poly_count: u32;

// ----------------------------------------------------------------------------
// Rigid transforms (operation-matched to glam's quaternion rotation).
// ----------------------------------------------------------------------------

// Rotates v by the unit quaternion q, matching glam::Quat::mul_vec3 term by term.
fn quat_rotate(q: vec4<f32>, v: vec3<f32>) -> vec3<f32> {
    let b = q.xyz;
    let w = q.w;
    let b2 = dot(b, b);
    return v * (w * w - b2) + b * (dot(v, b) * 2.0) + cross(b, v) * (w * 2.0);
}

// The conjugate (inverse, for a unit quaternion) of q.
fn quat_conj(q: vec4<f32>) -> vec4<f32> {
    return vec4<f32>(-q.x, -q.y, -q.z, q.w);
}

// Rotates a world direction into a body's local frame.
fn quat_inv_rotate(q: vec4<f32>, v: vec3<f32>) -> vec3<f32> {
    return quat_rotate(quat_conj(q), v);
}

// Maps a body-local point into the world.
fn body_transform(body: Body, local: vec3<f32>) -> vec3<f32> {
    return body.pos + quat_rotate(body.rot, local);
}

// ----------------------------------------------------------------------------
// Support mapping over the Minkowski difference A (-) B.
// ----------------------------------------------------------------------------

// Global index of the vertex of the body slice [vert_offset, +vert_count)
// farthest along dir, resolving ties to the lower index.
fn support_local(vert_offset: u32, vert_count: u32, dir: vec3<f32>) -> u32 {
    var best = 0u;
    var best_dot = dot(vertices[vert_offset].xyz, dir);
    for (var k = 1u; k < vert_count; k = k + 1u) {
        let d = dot(vertices[vert_offset + k].xyz, dir);
        if (d > best_dot) {
            best_dot = d;
            best = k;
        }
    }
    return vert_offset + best;
}

// Support of the difference along world dir, with both world witnesses kept.
fn support(dir: vec3<f32>) -> SupportPoint {
    let la = quat_inv_rotate(body_a.rot, dir);
    let ia = support_local(body_a.vert_offset, body_a.vert_count, la);
    let on_a = body_transform(body_a, vertices[ia].xyz);
    let lb = quat_inv_rotate(body_b.rot, -dir);
    let ib = support_local(body_b.vert_offset, body_b.vert_count, lb);
    let on_b = body_transform(body_b, vertices[ib].xyz);
    var s: SupportPoint;
    s.diff = on_a - on_b;
    s.on_a = on_a;
    s.on_b = on_b;
    return s;
}


// ----------------------------------------------------------------------------
// GJK Voronoi sub-distance (Ericson 2005, specialised to the origin).
// ----------------------------------------------------------------------------

// Classifies the origin against triangle (a, b, c)'s Voronoi regions. The
// returned indices are triangle-local (0, 1, 2).
fn triangle_region(a: vec3<f32>, b: vec3<f32>, c: vec3<f32>) -> TriRegion {
    var r: TriRegion;
    let ab = b - a;
    let ac = c - a;
    let ap = -a;
    let d1 = dot(ab, ap);
    let d2 = dot(ac, ap);
    if (d1 <= 0.0 && d2 <= 0.0) {
        r.kind = 0u; r.a = 0u; r.b = 0u;
        return r;
    }
    let bp = -b;
    let d3 = dot(ab, bp);
    let d4 = dot(ac, bp);
    if (d3 >= 0.0 && d4 <= d3) {
        r.kind = 0u; r.a = 1u; r.b = 0u;
        return r;
    }
    let vc = d1 * d4 - d3 * d2;
    if (vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0) {
        r.kind = 1u; r.a = 0u; r.b = 1u;
        return r;
    }
    let cp = -c;
    let d5 = dot(ab, cp);
    let d6 = dot(ac, cp);
    if (d6 >= 0.0 && d5 <= d6) {
        r.kind = 0u; r.a = 2u; r.b = 0u;
        return r;
    }
    let vb = d5 * d2 - d1 * d6;
    if (vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0) {
        r.kind = 1u; r.a = 0u; r.b = 2u;
        return r;
    }
    let va = d3 * d6 - d5 * d4;
    if (va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0) {
        r.kind = 1u; r.a = 1u; r.b = 2u;
        return r;
    }
    r.kind = 2u; r.a = 0u; r.b = 0u;
    return r;
}

// Turns a triangle Voronoi classification into a simplex reduction over the
// three points (p0, p1, p2) whose simplex-local indices are (li0, li1, li2).
fn reduction_from_region(
    p0: vec3<f32>, p1: vec3<f32>, p2: vec3<f32>,
    li0: u32, li1: u32, li2: u32,
    region: TriRegion
) -> TriReduce {
    var red: TriReduce;
    var li = array<u32, 3>(li0, li1, li2);
    var pp = array<vec3<f32>, 3>(p0, p1, p2);
    if (region.kind == 0u) {
        red.kc = 1u;
        red.k0 = li[region.a];
        red.k1 = 0u;
        red.k2 = 0u;
        red.closest = pp[region.a];
        return red;
    }
    if (region.kind == 1u) {
        let a = pp[region.a];
        let b = pp[region.b];
        let ab = b - a;
        let t = clamp(dot(-a, ab) / max(dot(ab, ab), ZERO_EPS2), 0.0, 1.0);
        red.kc = 2u;
        red.k0 = li[region.a];
        red.k1 = li[region.b];
        red.k2 = 0u;
        red.closest = a + ab * t;
        return red;
    }
    // Face: project the origin onto the plane through p0 with the loop normal.
    let n = cross(p1 - p0, p2 - p0);
    let n2 = max(dot(n, n), ZERO_EPS2);
    red.kc = 3u;
    red.k0 = li0;
    red.k1 = li1;
    red.k2 = li2;
    red.closest = n * (dot(n, p0) / n2);
    return red;
}

// Collapses the GJK simplex to the single point at simplex-local index i.
fn keep1(i: u32) {
    let d = gjk_diff[i];
    let a = gjk_a[i];
    let b = gjk_b[i];
    gjk_diff[0] = d;
    gjk_a[0] = a;
    gjk_b[0] = b;
    gjk_len = 1u;
}

// Applies a TriReduce to the GJK simplex: gathers the kept points through a
// temporary copy so overlapping source/destination indices are safe.
fn apply_reduce(red: TriReduce) {
    var td = array<vec3<f32>, 4>(gjk_diff[0], gjk_diff[1], gjk_diff[2], gjk_diff[3]);
    var ta = array<vec3<f32>, 4>(gjk_a[0], gjk_a[1], gjk_a[2], gjk_a[3]);
    var tb = array<vec3<f32>, 4>(gjk_b[0], gjk_b[1], gjk_b[2], gjk_b[3]);
    gjk_diff[0] = td[red.k0];
    gjk_a[0] = ta[red.k0];
    gjk_b[0] = tb[red.k0];
    if (red.kc >= 2u) {
        gjk_diff[1] = td[red.k1];
        gjk_a[1] = ta[red.k1];
        gjk_b[1] = tb[red.k1];
    }
    if (red.kc >= 3u) {
        gjk_diff[2] = td[red.k2];
        gjk_a[2] = ta[red.k2];
        gjk_b[2] = tb[red.k2];
    }
    gjk_len = red.kc;
}

// Reduces the one-to-four-point simplex to the sub-feature nearest the origin,
// updating gjk_closest. Returns true when a tetrahedron encloses the origin.
fn gjk_reduce() -> bool {
    if (gjk_len == 1u) {
        gjk_closest = gjk_diff[0];
        return false;
    }
    if (gjk_len == 2u) {
        let a = gjk_diff[0];
        let b = gjk_diff[1];
        let ab = b - a;
        let t = dot(-a, ab);
        if (t <= 0.0) {
            keep1(0u);
            gjk_closest = a;
            return false;
        }
        let denom = dot(ab, ab);
        if (t >= denom) {
            keep1(1u);
            gjk_closest = b;
            return false;
        }
        let s = t / denom;
        gjk_len = 2u;
        gjk_closest = a + ab * s;
        return false;
    }
    if (gjk_len == 3u) {
        let a = gjk_diff[0];
        let b = gjk_diff[1];
        let c = gjk_diff[2];
        let region = triangle_region(a, b, c);
        let red = reduction_from_region(a, b, c, 0u, 1u, 2u, region);
        gjk_closest = red.closest;
        apply_reduce(red);
        return false;
    }
    // Tetrahedron: test each outward face, keeping the nearest sub-feature.
    var fi0 = array<u32, 4>(0u, 0u, 0u, 1u);
    var fi1 = array<u32, 4>(1u, 3u, 2u, 3u);
    var fi2 = array<u32, 4>(2u, 1u, 3u, 2u);
    var finner = array<u32, 4>(3u, 2u, 1u, 0u);
    var inside_all = true;
    var have_best = false;
    var best_d2 = 0.0;
    var best_red: TriReduce;
    for (var f = 0u; f < 4u; f = f + 1u) {
        let i0 = fi0[f];
        let i1 = fi1[f];
        let i2 = fi2[f];
        let p0 = gjk_diff[i0];
        let p1 = gjk_diff[i1];
        let p2 = gjk_diff[i2];
        let inner = gjk_diff[finner[f]];
        let nrm = cross(p1 - p0, p2 - p0);
        let origin_side = dot(nrm, -p0);
        let inner_side = dot(nrm, inner - p0);
        if (origin_side * inner_side >= 0.0) {
            continue;
        }
        inside_all = false;
        let region = triangle_region(p0, p1, p2);
        let red = reduction_from_region(p0, p1, p2, i0, i1, i2, region);
        let d2 = dot(red.closest, red.closest);
        if (!have_best || d2 < best_d2) {
            have_best = true;
            best_d2 = d2;
            best_red = red;
        }
    }
    if (inside_all) {
        return true;
    }
    if (have_best) {
        gjk_closest = best_red.closest;
        apply_reduce(best_red);
        return false;
    }
    return true;
}

// ----------------------------------------------------------------------------
// GJK main loop. Returns true when the hulls intersect, leaving the terminal
// simplex in gjk_* as the EPA seed.
// ----------------------------------------------------------------------------
fn run_gjk() -> bool {
    var seed = body_a.pos - body_b.pos;
    if (dot(seed, seed) < ZERO_EPS2) {
        seed = vec3<f32>(1.0, 0.0, 0.0);
    }
    let s0 = support(seed);
    gjk_diff[0] = s0.diff;
    gjk_a[0] = s0.on_a;
    gjk_b[0] = s0.on_b;
    gjk_len = 1u;
    gjk_closest = s0.diff;
    for (var iter = 0u; iter < MAX_GJK_ITERS; iter = iter + 1u) {
        let cc = dot(gjk_closest, gjk_closest);
        if (cc < ZERO_EPS2) {
            return true;
        }
        let dir = -gjk_closest;
        let w = support(dir);
        let advance = cc - dot(gjk_closest, w.diff);
        if (advance <= PROGRESS_TOL * max(cc, 1.0)) {
            return false;
        }
        var dup = false;
        for (var k = 0u; k < gjk_len; k = k + 1u) {
            let dd = gjk_diff[k] - w.diff;
            if (dot(dd, dd) < ZERO_EPS2) {
                dup = true;
            }
        }
        if (dup) {
            return false;
        }
        gjk_diff[gjk_len] = w.diff;
        gjk_a[gjk_len] = w.on_a;
        gjk_b[gjk_len] = w.on_b;
        gjk_len = gjk_len + 1u;
        if (gjk_reduce()) {
            return true;
        }
    }
    return false;
}

// ----------------------------------------------------------------------------
// EPA: expand the GJK simplex into the penetration normal and depth.
// Operation-for-operation with epa.rs.
// ----------------------------------------------------------------------------

// The world axis least aligned with v, for building a perpendicular.
fn least_aligned_axis(v: vec3<f32>) -> vec3<f32> {
    let ax = abs(v.x);
    let ay = abs(v.y);
    let az = abs(v.z);
    if (ax <= ay && ax <= az) {
        return vec3<f32>(1.0, 0.0, 0.0);
    } else if (ay <= az) {
        return vec3<f32>(0.0, 1.0, 0.0);
    } else {
        return vec3<f32>(0.0, 0.0, 1.0);
    }
}

// Appends a support point to the EPA polytope vertex arrays.
fn epa_add(s: SupportPoint) {
    epa_diff[epa_vcount] = s.diff;
    epa_a[epa_vcount] = s.on_a;
    epa_b[epa_vcount] = s.on_b;
    epa_vcount = epa_vcount + 1u;
}

// Grows the seeded one-to-four-point simplex (in epa_*) into a non-degenerate
// tetrahedron, adding fresh support points. Returns false when no non-flat
// tetrahedron exists.
fn blow_up() -> bool {
    // 1 -> 2: find any direction yielding a distinct second point.
    if (epa_vcount == 1u) {
        var axes = array<vec3<f32>, 6>(
            vec3<f32>(1.0, 0.0, 0.0), vec3<f32>(-1.0, 0.0, 0.0),
            vec3<f32>(0.0, 1.0, 0.0), vec3<f32>(0.0, -1.0, 0.0),
            vec3<f32>(0.0, 0.0, 1.0), vec3<f32>(0.0, 0.0, -1.0),
        );
        for (var i = 0u; i < 6u; i = i + 1u) {
            let s = support(axes[i]);
            let d = s.diff - epa_diff[0];
            if (dot(d, d) > ZERO_EPS2) {
                epa_add(s);
                break;
            }
        }
        if (epa_vcount < 2u) {
            return false;
        }
    }

    // 2 -> 3: search perpendicular to the segment for a non-collinear point.
    if (epa_vcount == 2u) {
        let ab = epa_diff[1] - epa_diff[0];
        let axis = least_aligned_axis(ab);
        let perp1 = cross(ab, axis);
        let perp2 = cross(ab, perp1);
        var dirs = array<vec3<f32>, 4>(perp1, -perp1, perp2, -perp2);
        var have_best = false;
        var best_area = 0.0;
        var best_s: SupportPoint;
        for (var i = 0u; i < 4u; i = i + 1u) {
            let d = dirs[i];
            if (dot(d, d) < ZERO_EPS2) {
                continue;
            }
            let s = support(d);
            let cr = cross(s.diff - epa_diff[0], ab);
            let area = dot(cr, cr);
            if (!have_best || area > best_area) {
                have_best = true;
                best_area = area;
                best_s = s;
            }
        }
        if (!have_best) {
            return false;
        }
        if (best_area < ZERO_EPS2) {
            return false;
        }
        epa_add(best_s);
    }

    // 3 -> 4: push off the triangle plane on whichever side reaches farther.
    if (epa_vcount == 3u) {
        let n = cross(epa_diff[1] - epa_diff[0], epa_diff[2] - epa_diff[0]);
        if (dot(n, n) < ZERO_EPS2) {
            return false;
        }
        let plus = support(n);
        let minus = support(-n);
        let off_plus = abs(dot(plus.diff - epa_diff[0], n));
        let off_minus = abs(dot(minus.diff - epa_diff[0], n));
        var s: SupportPoint;
        if (off_plus >= off_minus) {
            s = plus;
        } else {
            s = minus;
        }
        if (abs(dot(s.diff - epa_diff[0], n)) < ZERO_EPS2) {
            return false;
        }
        epa_add(s);
    }

    if (epa_vcount != 4u) {
        return false;
    }
    // Reject a flat tetrahedron.
    let vol = dot(cross(epa_diff[1] - epa_diff[0], epa_diff[2] - epa_diff[0]), epa_diff[3] - epa_diff[0]);
    if (abs(vol) < ZERO_EPS2) {
        return false;
    }
    return true;
}

// Orients triangle (i, j, k) so its winding and normal point away from the
// interior reference; a zero-area triangle is tagged with the degenerate
// sentinel distance so it is never chosen as the closest face.
fn make_oriented_face(i: u32, j: u32, k: u32, interior: vec3<f32>) -> Face {
    let vi = epa_diff[i];
    let raw = cross(epa_diff[j] - vi, epa_diff[k] - vi);
    var f: Face;
    if (dot(raw, raw) < ZERO_EPS2) {
        f.i0 = i;
        f.i1 = j;
        f.i2 = k;
        f.normal = vec3<f32>(0.0, 0.0, 0.0);
        f.distance = DEGENERATE_DISTANCE;
        return f;
    }
    let n = normalize(raw);
    if (dot(n, interior - vi) > 0.0) {
        f.i0 = i;
        f.i1 = k;
        f.i2 = j;
        f.normal = -n;
        f.distance = dot(-n, vi);
    } else {
        f.i0 = i;
        f.i1 = j;
        f.i2 = k;
        f.normal = n;
        f.distance = dot(n, vi);
    }
    return f;
}

// Builds the four outward-wound faces of the seed tetrahedron; returns false
// when the four vertices are coplanar (a degenerate seed face appears).
fn build_initial_faces() -> bool {
    let centroid = (epa_diff[0] + epa_diff[1] + epa_diff[2] + epa_diff[3]) * 0.25;
    var ti0 = array<u32, 4>(0u, 0u, 0u, 1u);
    var ti1 = array<u32, 4>(1u, 1u, 2u, 2u);
    var ti2 = array<u32, 4>(2u, 3u, 3u, 3u);
    epa_fcount = 0u;
    for (var t = 0u; t < 4u; t = t + 1u) {
        let f = make_oriented_face(ti0[t], ti1[t], ti2[t], centroid);
        if (f.distance >= DEGENERATE_DISTANCE) {
            return false;
        }
        epa_faces[epa_fcount] = f;
        epa_fcount = epa_fcount + 1u;
    }
    return true;
}

// Index of the polytope face whose supporting plane is nearest the origin,
// skipping degenerate faces; ties resolve to the first (lowest) index.
fn epa_closest_face() -> ClosestFace {
    var out: ClosestFace;
    out.idx = 0u;
    out.found = false;
    var best_dist = 0.0;
    for (var i = 0u; i < epa_fcount; i = i + 1u) {
        let d = epa_faces[i].distance;
        if (d < DEGENERATE_DISTANCE && (!out.found || d < best_dist)) {
            out.found = true;
            best_dist = d;
            out.idx = i;
        }
    }
    return out;
}

// Adds a directed edge to the horizon, or cancels it against the opposing edge
// contributed by an adjacent visible face.
fn add_or_cancel(x: u32, y: u32) {
    for (var i = 0u; i < horizon_count; i = i + 1u) {
        if (horizon[i].x == y && horizon[i].y == x) {
            horizon[i] = horizon[horizon_count - 1u];
            horizon_count = horizon_count - 1u;
            return;
        }
    }
    if (horizon_count < MAX_HORIZON) {
        horizon[horizon_count] = vec2<u32>(x, y);
        horizon_count = horizon_count + 1u;
    }
}

// Removes every face the new vertex w can see and stitches new faces from the
// resulting horizon to w, preserving outward winding. The kept faces are packed
// in place with a write cursor (write <= fi, so the in-place pack is safe).
fn carve_and_patch(w: vec3<f32>, w_index: u32) {
    horizon_count = 0u;
    var write = 0u;
    for (var fi = 0u; fi < epa_fcount; fi = fi + 1u) {
        let f = epa_faces[fi];
        let v0 = epa_diff[f.i0];
        let visible = dot(f.normal, w - v0) > 0.0;
        if (visible) {
            add_or_cancel(f.i0, f.i1);
            add_or_cancel(f.i1, f.i2);
            add_or_cancel(f.i2, f.i0);
        } else {
            epa_faces[write] = f;
            write = write + 1u;
        }
    }
    for (var h = 0u; h < horizon_count; h = h + 1u) {
        let a = horizon[h].x;
        let bb = horizon[h].y;
        let va = epa_diff[a];
        let raw = cross(epa_diff[bb] - va, w - va);
        var nf: Face;
        if (dot(raw, raw) < ZERO_EPS2) {
            nf.i0 = a;
            nf.i1 = bb;
            nf.i2 = w_index;
            nf.normal = vec3<f32>(0.0, 0.0, 0.0);
            nf.distance = DEGENERATE_DISTANCE;
        } else {
            let n = normalize(raw);
            if (dot(n, va) < 0.0) {
                nf.i0 = a;
                nf.i1 = w_index;
                nf.i2 = bb;
                nf.normal = -n;
                nf.distance = dot(-n, va);
            } else {
                nf.i0 = a;
                nf.i1 = bb;
                nf.i2 = w_index;
                nf.normal = n;
                nf.distance = dot(n, va);
            }
        }
        if (write < MAX_EPA_FACES) {
            epa_faces[write] = nf;
            write = write + 1u;
        }
    }
    epa_fcount = write;
}

// Barycentric weights (u, v, w) of p on triangle (a, b, c).
fn barycentric3(a: vec3<f32>, b: vec3<f32>, c: vec3<f32>, p: vec3<f32>) -> vec3<f32> {
    let v0 = b - a;
    let v1 = c - a;
    let v2 = p - a;
    let d00 = dot(v0, v0);
    let d01 = dot(v0, v1);
    let d11 = dot(v1, v1);
    let d20 = dot(v2, v0);
    let d21 = dot(v2, v1);
    let denom = max(d00 * d11 - d01 * d01, ZERO_EPS2);
    let v = (d11 * d20 - d01 * d21) / denom;
    let w = (d00 * d21 - d01 * d20) / denom;
    return vec3<f32>(1.0 - v - w, v, w);
}

// Expands the GJK simplex (seeded into epa_*) into the penetration. pen.ok is
// false only for an exact surface tangency (no non-flat tetrahedron).
fn run_epa() -> Pen {
    var pen: Pen;
    pen.ok = false;
    pen.normal = vec3<f32>(0.0, 0.0, 0.0);
    pen.depth = 0.0;
    pen.point_a = vec3<f32>(0.0, 0.0, 0.0);
    pen.point_b = vec3<f32>(0.0, 0.0, 0.0);

    epa_vcount = gjk_len;
    for (var k = 0u; k < gjk_len; k = k + 1u) {
        epa_diff[k] = gjk_diff[k];
        epa_a[k] = gjk_a[k];
        epa_b[k] = gjk_b[k];
    }
    if (!blow_up()) {
        return pen;
    }
    if (!build_initial_faces()) {
        return pen;
    }
    var best = epa_closest_face();
    if (!best.found) {
        return pen;
    }
    for (var iter = 0u; iter < MAX_EPA_ITERS; iter = iter + 1u) {
        let normal = epa_faces[best.idx].normal;
        let w = support(normal);
        let reach = dot(w.diff, normal);
        if (reach - epa_faces[best.idx].distance < GROWTH_TOL) {
            break;
        }
        var dup = false;
        for (var k = 0u; k < epa_vcount; k = k + 1u) {
            let dd = epa_diff[k] - w.diff;
            if (dot(dd, dd) < ZERO_EPS2) {
                dup = true;
            }
        }
        if (dup) {
            break;
        }
        if (epa_vcount >= MAX_EPA_VERTS) {
            break;
        }
        let w_index = epa_vcount;
        epa_add(w);
        carve_and_patch(w.diff, w_index);
        let nb = epa_closest_face();
        if (!nb.found) {
            break;
        }
        best = nb;
    }

    let f = epa_faces[best.idx];
    let proj = f.normal * f.distance;
    let bc = barycentric3(epa_diff[f.i0], epa_diff[f.i1], epa_diff[f.i2], proj);
    pen.ok = true;
    pen.normal = -f.normal;
    pen.depth = f.distance;
    pen.point_a = epa_a[f.i0] * bc.x + epa_a[f.i1] * bc.y + epa_a[f.i2] * bc.z;
    pen.point_b = epa_b[f.i0] * bc.x + epa_b[f.i1] * bc.y + epa_b[f.i2] * bc.z;
    return pen;
}

// ----------------------------------------------------------------------------
// Manifold build (reference/incident face clip) + compute entry point.
// Operation-for-operation with convex_convex_manifold.rs.
// ----------------------------------------------------------------------------

// Kept contact corners before the four-point reduction.
var<private> kept_pos: array<vec3<f32>, 32>;
var<private> kept_depth: array<f32, 32>;

// World outward normal of a body's face.
fn face_world_normal(body: Body, fidx: u32) -> vec3<f32> {
    return quat_rotate(body.rot, faces[body.face_offset + fidx].plane.xyz);
}

// Body-local face whose world outward normal is most parallel to dir; ties
// resolve to the lower face index.
fn most_aligned_face(body: Body, dir: vec3<f32>) -> FaceAlign {
    var out: FaceAlign;
    out.idx = 0u;
    out.dotv = dot(face_world_normal(body, 0u), dir);
    for (var i = 1u; i < body.face_count; i = i + 1u) {
        let d = dot(face_world_normal(body, i), dir);
        if (d > out.dotv) {
            out.dotv = d;
            out.idx = i;
        }
    }
    return out;
}

// Loads the world-space loop of body's face fidx into ref_poly.
fn load_ref_poly(body: Body, fidx: u32) {
    let f = faces[body.face_offset + fidx];
    let lo = f.loop_info.x;
    let lc = f.loop_info.y;
    ref_poly_count = 0u;
    for (var i = 0u; i < lc; i = i + 1u) {
        if (ref_poly_count >= MAX_FACE_VERTS) {
            break;
        }
        let vl = face_loops[lo + i];
        ref_poly[ref_poly_count] = body_transform(body, vertices[body.vert_offset + vl].xyz);
        ref_poly_count = ref_poly_count + 1u;
    }
}

// Loads the world-space loop of body's face fidx into clip_a; returns its count.
fn load_inc_poly(body: Body, fidx: u32) -> u32 {
    let f = faces[body.face_offset + fidx];
    let lo = f.loop_info.x;
    let lc = f.loop_info.y;
    var n = 0u;
    for (var i = 0u; i < lc; i = i + 1u) {
        if (n >= MAX_CLIP_POINTS) {
            break;
        }
        let vl = face_loops[lo + i];
        clip_a[n] = body_transform(body, vertices[body.vert_offset + vl].xyz);
        n = n + 1u;
    }
    return n;
}

// Sutherland-Hodgman clip of the active polygon against the half-space
// dot(v - p0, nrm) <= 0. Reads clip_a and writes clip_b when from_a is true,
// otherwise reads clip_b and writes clip_a; the two buffers never alias.
fn clip_poly(from_a: bool, count: u32, p0: vec3<f32>, nrm: vec3<f32>) -> u32 {
    if (count == 0u) {
        return 0u;
    }
    var out_n = 0u;
    for (var i = 0u; i < count; i = i + 1u) {
        let ni = (i + 1u) % count;
        var cur: vec3<f32>;
        var nxt: vec3<f32>;
        if (from_a) {
            cur = clip_a[i];
            nxt = clip_a[ni];
        } else {
            cur = clip_b[i];
            nxt = clip_b[ni];
        }
        let dc = dot(cur - p0, nrm);
        let dn = dot(nxt - p0, nrm);
        let cur_in = dc <= 0.0;
        let next_in = dn <= 0.0;
        if (cur_in) {
            if (out_n < MAX_CLIP_POINTS) {
                if (from_a) {
                    clip_b[out_n] = cur;
                } else {
                    clip_a[out_n] = cur;
                }
                out_n = out_n + 1u;
            }
        }
        if (cur_in != next_in) {
            let t = dc / (dc - dn);
            let p = cur + (nxt - cur) * t;
            if (out_n < MAX_CLIP_POINTS) {
                if (from_a) {
                    clip_b[out_n] = p;
                } else {
                    clip_a[out_n] = p;
                }
                out_n = out_n + 1u;
            }
        }
    }
    return out_n;
}

// Promotes a penetration to a manifold and writes it to slot gi.
fn build_manifold(pen: Pen, gi: u32) {
    var out: Manifold;
    out.normal_count = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    out.p0 = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    out.p1 = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    out.p2 = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    out.p3 = vec4<f32>(0.0, 0.0, 0.0, 0.0);

    let push = pen.normal;
    let normal_ab = -push;

    let maf_a = most_aligned_face(body_a, -push);
    let maf_b = most_aligned_face(body_b, push);
    let reference_is_a = maf_a.dotv >= maf_b.dotv;

    var rn: vec3<f32>;
    var ref_point: vec3<f32>;
    var inc_count: u32;
    if (reference_is_a) {
        load_ref_poly(body_a, maf_a.idx);
        rn = face_world_normal(body_a, maf_a.idx);
        ref_point = ref_poly[0];
        let inc = most_aligned_face(body_b, -rn);
        inc_count = load_inc_poly(body_b, inc.idx);
    } else {
        load_ref_poly(body_b, maf_b.idx);
        rn = face_world_normal(body_b, maf_b.idx);
        ref_point = ref_poly[0];
        let inc = most_aligned_face(body_a, -rn);
        inc_count = load_inc_poly(body_a, inc.idx);
    }

    // Clip the incident polygon against each reference side plane; the active
    // polygon ping-pongs between clip_a and clip_b.
    var n = inc_count;
    var from_a = true;
    for (var i = 0u; i < ref_poly_count; i = i + 1u) {
        if (n == 0u) {
            break;
        }
        let a0 = ref_poly[i];
        let a1 = ref_poly[(i + 1u) % ref_poly_count];
        let side = cross(a1 - a0, rn);
        n = clip_poly(from_a, n, a0, side);
        from_a = !from_a;
    }

    // Keep the clipped corners that penetrate the reference face.
    var kept_n = 0u;
    for (var i = 0u; i < n; i = i + 1u) {
        var corner: vec3<f32>;
        if (from_a) {
            corner = clip_a[i];
        } else {
            corner = clip_b[i];
        }
        let sep = dot(corner - ref_point, rn);
        if (sep <= 0.0) {
            let depth = -sep;
            if (kept_n < 32u) {
                kept_pos[kept_n] = corner + rn * (depth * 0.5);
                kept_depth[kept_n] = depth;
                kept_n = kept_n + 1u;
            }
        }
    }

    if (kept_n == 0u) {
        out.normal_count = vec4<f32>(normal_ab, 1.0);
        out.p0 = vec4<f32>((pen.point_a + pen.point_b) * 0.5, pen.depth);
        manifolds[gi] = out;
        return;
    }

    // Inline reduce_to_four (shared with the CPU twin), measured in normal_ab.
    var chosen = array<u32, 4>(0u, 0u, 0u, 0u);
    var out_count = 0u;
    if (kept_n <= 4u) {
        for (var i = 0u; i < kept_n; i = i + 1u) {
            chosen[i] = i;
        }
        out_count = kept_n;
    } else {
        var i0 = 0u;
        for (var i = 1u; i < kept_n; i = i + 1u) {
            if (kept_depth[i] > kept_depth[i0]) {
                i0 = i;
            }
        }
        let q0 = kept_pos[i0];
        var i1 = i0;
        var best_d2 = -1.0;
        for (var i = 0u; i < kept_n; i = i + 1u) {
            let dd = kept_pos[i] - q0;
            let d2 = dot(dd, dd);
            if (d2 > best_d2) {
                best_d2 = d2;
                i1 = i;
            }
        }
        let diag = kept_pos[i1] - q0;
        var i2 = i0;
        var i3 = i0;
        var best_pos = 0.0;
        var best_neg = 0.0;
        for (var i = 0u; i < kept_n; i = i + 1u) {
            let area = dot(cross(diag, kept_pos[i] - q0), normal_ab);
            if (area > best_pos) {
                best_pos = area;
                i2 = i;
            } else if (area < best_neg) {
                best_neg = area;
                i3 = i;
            }
        }
        var cand = array<u32, 4>(i0, i1, i2, i3);
        out_count = 0u;
        for (var c = 0u; c < 4u; c = c + 1u) {
            var dupc = false;
            for (var e = 0u; e < out_count; e = e + 1u) {
                if (chosen[e] == cand[c]) {
                    dupc = true;
                }
            }
            if (!dupc) {
                chosen[out_count] = cand[c];
                out_count = out_count + 1u;
            }
        }
    }

    out.normal_count = vec4<f32>(normal_ab, f32(out_count));
    if (out_count > 0u) {
        out.p0 = vec4<f32>(kept_pos[chosen[0]], kept_depth[chosen[0]]);
    }
    if (out_count > 1u) {
        out.p1 = vec4<f32>(kept_pos[chosen[1]], kept_depth[chosen[1]]);
    }
    if (out_count > 2u) {
        out.p2 = vec4<f32>(kept_pos[chosen[2]], kept_depth[chosen[2]]);
    }
    if (out_count > 3u) {
        out.p3 = vec4<f32>(kept_pos[chosen[3]], kept_depth[chosen[3]]);
    }
    manifolds[gi] = out;
}

@compute @workgroup_size(64)
fn narrowphase_convex_convex_manifold(@builtin(global_invocation_id) gid: vec3<u32>) {
    let gi = gid.x;
    if (gi >= params.num_pairs) {
        return;
    }
    let pair = pairs[gi];
    let ia = pair.x;
    let ib = pair.y;

    let ha = hulls[ia].data;
    body_a.vert_offset = ha.x;
    body_a.vert_count = ha.y;
    body_a.face_offset = ha.z;
    body_a.face_count = ha.w;
    body_a.pos = poses[ia].translation.xyz;
    body_a.rot = poses[ia].rotation;

    let hb = hulls[ib].data;
    body_b.vert_offset = hb.x;
    body_b.vert_count = hb.y;
    body_b.face_offset = hb.z;
    body_b.face_count = hb.w;
    body_b.pos = poses[ib].translation.xyz;
    body_b.rot = poses[ib].rotation;

    // Separated couples report an empty (count = 0) manifold.
    if (!run_gjk()) {
        var empty: Manifold;
        empty.normal_count = vec4<f32>(0.0, 0.0, 0.0, 0.0);
        empty.p0 = vec4<f32>(0.0, 0.0, 0.0, 0.0);
        empty.p1 = vec4<f32>(0.0, 0.0, 0.0, 0.0);
        empty.p2 = vec4<f32>(0.0, 0.0, 0.0, 0.0);
        empty.p3 = vec4<f32>(0.0, 0.0, 0.0, 0.0);
        manifolds[gi] = empty;
        return;
    }

    let pen = run_epa();
    if (!pen.ok) {
        // Exact tangency: report a zero-depth touch at the simplex witness, with
        // the a -> b centre direction as the normal.
        var witness = vec3<f32>(0.0, 0.0, 0.0);
        for (var k = 0u; k < gjk_len; k = k + 1u) {
            witness = witness + (gjk_a[k] + gjk_b[k]) * 0.5;
        }
        witness = witness / f32(gjk_len);
        let dir = body_b.pos - body_a.pos;
        var normal_ab: vec3<f32>;
        if (dot(dir, dir) > 1.0e-20) {
            normal_ab = normalize(dir);
        } else {
            normal_ab = vec3<f32>(1.0, 0.0, 0.0);
        }
        var out: Manifold;
        out.normal_count = vec4<f32>(normal_ab, 1.0);
        out.p0 = vec4<f32>(witness, 0.0);
        out.p1 = vec4<f32>(0.0, 0.0, 0.0, 0.0);
        out.p2 = vec4<f32>(0.0, 0.0, 0.0, 0.0);
        out.p3 = vec4<f32>(0.0, 0.0, 0.0, 0.0);
        manifolds[gi] = out;
        return;
    }

    build_manifold(pen, gi);
}
