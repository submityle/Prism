// Closest-point distance scene query on the device: for one world-space query
// point, the exact closest point on each target's convex core, the surface
// distance, and the outward normal there — one target per invocation.
//
// Each lane resolves its target hull and the shared query point (body 0, a
// single-vertex hull posed at the query position), then runs one
// Gilbert-Johnson-Keerthi distance walk of the Minkowski difference to recover
// the core separation, its unit normal (from the target toward the query
// point), and the closest point on the target core. There is no
// conservative-advancement loop: the query is static, so a single GJK solve per
// target is exact. The GJK core is operation-for-operation with the CPU twin
// gjk.rs (and byte-identical to the shared core in
// narrowphase_convex_convex_toi.wgsl), so a passing real-device parity test is
// direct evidence the kernel finds the same closest points and distances as the
// reference. The rounding radius and the final surface-and-inside rule are
// applied on the host by the shared closest_hit_from_core, so this kernel only
// reports the core GJK result.
//
// Buffer layout (byte-for-byte with the wrapper structs):
//   * hull headers: (vert_offset, vert_count, 0, 0) as a vec4<u32>; only the
//     vertex slice is read here (support mapping needs no faces, and the radius
//     is folded in on the host);
//   * vertices: every hull's local-space vertices concatenated as vec4<f32>
//     (xyz used, w padding), each body's slice starting at vert_offset;
//   * poses: GpuPose each, world translation in translation.xyz and the
//     rotation quaternion (x, y, z, w) in rotation;
//   * pairs: vec2<u32> each (query body index 0, target body index);
//   * results read back as two vec4<f32> each (ClosestOut): (point_b.xyz,
//     core_distance) then (normal.xyz, intersecting_flag) with the flag 1.0 when
//     the query point lies within the target core and 0.0 when it is separated.
//
// Provenance: Gilbert-Johnson-Keerthi distance (1988) with Ericson's Voronoi
// sub-distance (2005). No Unreal Engine source or derived code.

// ----------------------------------------------------------------------------
// Bindings.
// ----------------------------------------------------------------------------

struct Params {
    num_pairs: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
};

struct HullHeader {
    // (vert_offset, vert_count, 0, 0).
    data: vec4<u32>,
};

struct GpuPose {
    translation: vec4<f32>,
    rotation: vec4<f32>,
};

struct ClosestOut {
    // (point_b.xyz, core_distance).
    point_dist: vec4<f32>,
    // (normal.xyz, intersecting_flag).
    normal_flag: vec4<f32>,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> hulls: array<HullHeader>;
@group(0) @binding(2) var<storage, read> vertices: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> poses: array<GpuPose>;
@group(0) @binding(4) var<storage, read> pairs: array<vec2<u32>>;
@group(0) @binding(5) var<storage, read_write> results: array<ClosestOut>;

// ----------------------------------------------------------------------------
// Tunables (operation-for-operation with the CPU twin's constants).
// ----------------------------------------------------------------------------

// Squared length below which a vector is treated as the zero vector.
const ZERO_EPS2: f32 = 1.0e-12;
// Relative volume tolerance below which a tetrahedron is treated as
// degenerate (near-coplanar) and the containment test is distrusted.
const DEGEN_REL: f32 = 1.0e-6;
// Relative progress tolerance ending the GJK search when a support stops gaining.
const PROGRESS_TOL: f32 = 1.0e-8;
// Hard iteration cap so a pathological target terminates.
const MAX_GJK_ITERS: u32 = 64u;

// ----------------------------------------------------------------------------
// Value types.
// ----------------------------------------------------------------------------

// One posed hull resolved from its header and pose, cached per lane.
struct Body {
    vert_offset: u32,
    vert_count: u32,
    pos: vec3<f32>,
    rot: vec4<f32>,
};

// One Minkowski-difference support point with its per-body world witnesses.
struct SupportPoint {
    diff: vec3<f32>,
    on_a: vec3<f32>,
    on_b: vec3<f32>,
};

// A triangle Voronoi classification: kind 0 vertex, 1 edge, 2 face.
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

// A separated-GJK outcome: whether the hulls intersect, the separation, the
// unit normal (from B toward A), and the two world witnesses.
struct GjkResult {
    intersecting: bool,
    distance: f32,
    normal: vec3<f32>,
    point_a: vec3<f32>,
    point_b: vec3<f32>,
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

// ----------------------------------------------------------------------------
// Rigid transforms and GJK core (shared verbatim with the TOI kernel).
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
// Closest-feature reduction over all four tetrahedron faces, ignoring the
// inner-side orientation test. Robust fallback mirroring the CPU
// nearest_face_reduction used when the tetrahedron is degenerate.
fn gjk_nearest_face() {
    var fi0 = array<u32, 4>(0u, 0u, 0u, 1u);
    var fi1 = array<u32, 4>(1u, 3u, 2u, 3u);
    var fi2 = array<u32, 4>(2u, 1u, 3u, 2u);
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
        let region = triangle_region(p0, p1, p2);
        let red = reduction_from_region(p0, p1, p2, i0, i1, i2, region);
        let d2 = dot(red.closest, red.closest);
        if (!have_best || d2 < best_d2) {
            have_best = true;
            best_d2 = d2;
            best_red = red;
        }
    }
    if (have_best) {
        gjk_closest = best_red.closest;
        apply_reduce(best_red);
    }
}

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
        // Degeneracy guard mirroring the CPU reduce_tetrahedron: a near-coplanar
        // tetrahedron cannot reliably contain the origin, so fall back to the
        // closest of all four faces instead of reporting containment.
        let a = gjk_diff[0];
        let b = gjk_diff[1];
        let c = gjk_diff[2];
        let d = gjk_diff[3];
        let e0 = b - a;
        let e1 = c - a;
        let e2 = d - a;
        let vol = abs(dot(e0, cross(e1, e2)));
        let scale = length(e0) * length(e1) * length(e2);
        if (vol <= DEGEN_REL * max(scale, ZERO_EPS2)) {
            gjk_nearest_face();
            return false;
        }
        return true;
    }
    if (have_best) {
        gjk_closest = best_red.closest;
        apply_reduce(best_red);
        return false;
    }
    // Not contained yet no outward face captured: fall back to the nearest face
    // rather than wrongly claiming containment.
    gjk_nearest_face();
    return false;
}

// Barycentric weights of p over triangle (a, b, c), matching the CPU
// triangle_barycentric used by the separated-GJK witness blend.
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
    let u = 1.0 - v - w;
    return vec3<f32>(u, v, w);
}

// Blends the kept simplex's per-body witnesses by the barycentric weights of
// gjk_closest, yielding the two closest surface points.
fn separated_witnesses(out_a: ptr<function, vec3<f32>>, out_b: ptr<function, vec3<f32>>) {
    if (gjk_len == 1u) {
        *out_a = gjk_a[0];
        *out_b = gjk_b[0];
        return;
    }
    if (gjk_len == 2u) {
        let a = gjk_diff[0];
        let b = gjk_diff[1];
        let ab = b - a;
        let t = dot(gjk_closest - a, ab) / max(dot(ab, ab), ZERO_EPS2);
        *out_a = gjk_a[0] * (1.0 - t) + gjk_a[1] * t;
        *out_b = gjk_b[0] * (1.0 - t) + gjk_b[1] * t;
        return;
    }
    let w = barycentric3(gjk_diff[0], gjk_diff[1], gjk_diff[2], gjk_closest);
    *out_a = gjk_a[0] * w.x + gjk_a[1] * w.y + gjk_a[2] * w.z;
    *out_b = gjk_b[0] * w.x + gjk_b[1] * w.y + gjk_b[2] * w.z;
}

// ----------------------------------------------------------------------------
// GJK main loop producing a separated distance, normal, and witnesses, or the
// intersecting flag. Operation-for-operation with gjk.rs.
// ----------------------------------------------------------------------------
fn run_gjk() -> GjkResult {
    var res: GjkResult;
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
            res.intersecting = true;
            return res;
        }
        let dir = -gjk_closest;
        let w = support(dir);
        let advance = cc - dot(gjk_closest, w.diff);
        if (advance <= PROGRESS_TOL * max(cc, 1.0)) {
            break;
        }
        var dup = false;
        for (var k = 0u; k < gjk_len; k = k + 1u) {
            let dd = gjk_diff[k] - w.diff;
            if (dot(dd, dd) < ZERO_EPS2) {
                dup = true;
            }
        }
        if (dup) {
            break;
        }
        gjk_diff[gjk_len] = w.diff;
        gjk_a[gjk_len] = w.on_a;
        gjk_b[gjk_len] = w.on_b;
        gjk_len = gjk_len + 1u;
        if (gjk_reduce()) {
            res.intersecting = true;
            return res;
        }
    }
    // Separated: reconstruct distance, normal, and the two world witnesses.
    res.intersecting = false;
    let distance = length(gjk_closest);
    res.distance = distance;
    if (distance > sqrt(ZERO_EPS2)) {
        res.normal = gjk_closest / distance;
    } else {
        res.normal = vec3<f32>(1.0, 0.0, 0.0);
    }
    var pa: vec3<f32>;
    var pb: vec3<f32>;
    separated_witnesses(&pa, &pb);
    res.point_a = pa;
    res.point_b = pb;
    return res;
}

// ----------------------------------------------------------------------------
// Entry point: one closest-point distance solve per invocation.
// ----------------------------------------------------------------------------
@compute @workgroup_size(64)
fn narrowphase_closest_point(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.num_pairs) {
        return;
    }
    let pair = pairs[i];
    let ia = pair.x;
    let ib = pair.y;

    let header_a = hulls[ia].data;
    let header_b = hulls[ib].data;
    body_a.vert_offset = header_a.x;
    body_a.vert_count = header_a.y;
    body_b.vert_offset = header_b.x;
    body_b.vert_count = header_b.y;

    let pose_a = poses[ia];
    let pose_b = poses[ib];
    body_a.pos = pose_a.translation.xyz;
    body_a.rot = pose_a.rotation;
    body_b.pos = pose_b.translation.xyz;
    body_b.rot = pose_b.rotation;

    let res = run_gjk();

    var out: ClosestOut;
    if (res.intersecting) {
        // The query point is within the target core: distance 0, point and
        // normal undefined. The host folds this into an inside hit.
        out.point_dist = vec4<f32>(0.0, 0.0, 0.0, 0.0);
        out.normal_flag = vec4<f32>(0.0, 0.0, 0.0, 1.0);
    } else {
        // Separated: the core closest point on the target (point_b) and the
        // outward unit normal (target toward query). The host adds the radius.
        out.point_dist = vec4<f32>(res.point_b.x, res.point_b.y, res.point_b.z, res.distance);
        out.normal_flag = vec4<f32>(res.normal.x, res.normal.y, res.normal.z, 0.0);
    }
    results[i] = out;
}
