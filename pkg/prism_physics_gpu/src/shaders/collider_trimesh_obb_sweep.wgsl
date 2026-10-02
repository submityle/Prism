// Swept-oriented-box-versus-triangle-mesh scene query on the device: the
// earliest triangle a moving oriented box (OBB) touches, one triangle per lane.
//
// This is the device twin of
// collider::trimesh_obb_sweep::cpu_trimesh_obb_sweep. Each invocation takes one
// mesh triangle, models the moving box as an eight-vertex sharp (zero-radius)
// convex core posed at the sweep centre with the box rotation and translating
// along the sweep direction, and the triangle as a static zero-radius
// three-vertex core, then runs the rounded conservative-advancement time of
// impact between the two cores. The GJK distance query, the Voronoi
// sub-distance, the screw-motion sampling, and the advance step are
// operation-for-operation with the CPU rounded conservative advancement
// (conservative_advancement::conservative_advancement_toi_rounded), so a
// passing real-device parity test is direct evidence the kernel finds the same
// impacts. The host reduces the per-triangle rows to the earliest contact.
//
// Buffer layout (byte-for-byte with the wrapper structs):
//   * vertices: three vertices per triangle concatenated as vec4<f32> (xyz
//     used, w padding): tri0.a, tri0.b, tri0.c, tri1.a, ...;
//   * indices: vec4<u32> per triangle, row i is (3i, 3i + 1, 3i + 2, 0);
//   * results read back as two vec4<f32> per triangle:
//     results[2i] = (toi, point.xyz) and results[2i + 1] = (hit_flag,
//     normal.xyz) with hit_flag 1.0 on a contact and 0.0 on a miss.
//
// Provenance: Gilbert-Johnson-Keerthi distance (1988) with Ericson's Voronoi
// sub-distance (2005); conservative advancement after Mirtich (2000) and van
// den Bergen (2004) ray-casting CCD. No Unreal Engine source or derived code.

// ----------------------------------------------------------------------------
// Bindings.
// ----------------------------------------------------------------------------

struct Params {
    // xyz: box centre at the sweep start; w: max travel.
    centre_maxdist: vec4<f32>,
    // xyzw: box orientation quaternion (x, y, z, w).
    rotation: vec4<f32>,
    // xyz: box half extents along the local axes; w: padding.
    half_pad: vec4<f32>,
    // xyz: sweep direction (unit); w: padding.
    dir_pad: vec4<f32>,
    // x: triangle count; y, z, w: padding.
    counts: vec4<u32>,
};

struct Toi {
    // (toi, point.xyz).
    time_point: vec4<f32>,
    // (hit_flag, normal.xyz).
    hit_normal: vec4<f32>,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> vertices: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> indices: array<vec4<u32>>;
@group(0) @binding(3) var<storage, read_write> results: array<Toi>;

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
// Separation below which the surfaces are treated as touching.
const DISTANCE_TOL: f32 = 1.0e-4;
// Closing-speed floor below which the gap cannot close.
const CLOSING_EPS: f32 = 1.0e-8;
// Below this sampled rotation angle the rotation is treated as the identity.
const ANGLE_EPS: f32 = 1.0e-8;
// Hard iteration caps so a pathological lane terminates.
const MAX_GJK_ITERS: u32 = 64u;
const MAX_CA_ITERS: u32 = 64u;

// Fixed per-lane core sizes: an eight-vertex box then a three-vertex triangle,
// laid out back to back in the private vertex scratch.
const BOX_OFFSET: u32 = 0u;
const BOX_COUNT: u32 = 8u;
const TRIANGLE_OFFSET: u32 = 8u;
const TRIANGLE_COUNT: u32 = 3u;

// hit_flag the kernel writes on a contact.
const HIT_FLAG: f32 = 1.0;

// ----------------------------------------------------------------------------
// Value types.
// ----------------------------------------------------------------------------

// One posed hull resolved from its sampled pose, cached per lane.
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

// The eight box core vertices then the three triangle vertices, in local
// space of their respective bodies, filled per lane before the solve.
var<private> local_verts: array<vec3<f32>, 11>;

// GJK simplex (one to four points): difference plus the two world witnesses.
var<private> gjk_diff: array<vec3<f32>, 4>;
var<private> gjk_a: array<vec3<f32>, 4>;
var<private> gjk_b: array<vec3<f32>, 4>;
var<private> gjk_len: u32;
var<private> gjk_closest: vec3<f32>;

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

// Hamilton product of two quaternions (x, y, z, w), matching glam::Quat::mul.
fn quat_mul(a: vec4<f32>, b: vec4<f32>) -> vec4<f32> {
    let aw = a.w;
    let bw = b.w;
    let av = a.xyz;
    let bv = b.xyz;
    let w = aw * bw - dot(av, bv);
    let v = aw * bv + bw * av + cross(av, bv);
    return vec4<f32>(v.x, v.y, v.z, w);
}

// A unit quaternion of a rotation of angle (radians) about a unit axis.
fn quat_from_axis_angle(axis: vec3<f32>, angle: f32) -> vec4<f32> {
    let h = angle * 0.5;
    let s = sin(h);
    return vec4<f32>(axis.x * s, axis.y * s, axis.z * s, cos(h));
}

// Maps a body-local point into the world.
fn body_transform(body: Body, local: vec3<f32>) -> vec3<f32> {
    return body.pos + quat_rotate(body.rot, local);
}

// ----------------------------------------------------------------------------
// Screw-motion sampling (operation-matched to BodyMotion::pose_at).
// ----------------------------------------------------------------------------

// The pose (world translation, rotation quaternion) a body reaches after t
// units of a constant linear and angular velocity from (pos0, rot0).
fn pose_at(pos0: vec3<f32>, rot0: vec4<f32>, linear: vec3<f32>, angular: vec3<f32>, t: f32,
           out_pos: ptr<function, vec3<f32>>, out_rot: ptr<function, vec4<f32>>) {
    *out_pos = pos0 + linear * t;
    let rate = length(angular);
    let angle = rate * t;
    if (angle > ANGLE_EPS) {
        let axis = angular / rate;
        let delta = quat_from_axis_angle(axis, angle);
        *out_rot = normalize(quat_mul(delta, rot0));
    } else {
        *out_rot = rot0;
    }
}

// ----------------------------------------------------------------------------
// Support mapping over the Minkowski difference A (-) B.
// ----------------------------------------------------------------------------

// Local index of the vertex of the body slice [vert_offset, +vert_count)
// farthest along dir, resolving ties to the lower index.
fn support_local(vert_offset: u32, vert_count: u32, dir: vec3<f32>) -> u32 {
    var best = 0u;
    var best_dot = dot(local_verts[vert_offset], dir);
    for (var k = 1u; k < vert_count; k = k + 1u) {
        let d = dot(local_verts[vert_offset + k], dir);
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
    let on_a = body_transform(body_a, local_verts[ia]);
    let lb = quat_inv_rotate(body_b.rot, -dir);
    let ib = support_local(body_b.vert_offset, body_b.vert_count, lb);
    let on_b = body_transform(body_b, local_verts[ib]);
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

// ----------------------------------------------------------------------------
// Witness reconstruction for a separated simplex.
// ----------------------------------------------------------------------------

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
// Hull circumradius: the farthest local vertex distance (rotation pivot bound).
// ----------------------------------------------------------------------------
fn circumradius(vert_offset: u32, vert_count: u32) -> f32 {
    var r = 0.0;
    for (var k = 0u; k < vert_count; k = k + 1u) {
        let d = length(local_verts[vert_offset + k]);
        r = max(r, d);
    }
    return r;
}

// ----------------------------------------------------------------------------
// Entry point: one swept box-versus-triangle solve per invocation.
// ----------------------------------------------------------------------------
@compute @workgroup_size(64)
fn collider_trimesh_obb_sweep(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.counts.x) {
        return;
    }

    // Fetch this lane's triangle corners.
    let idx = indices[i];
    let va = vertices[idx.x].xyz;
    let vb = vertices[idx.y].xyz;
    let vc = vertices[idx.z].xyz;

    // Box centre at the sweep start, travel budget, orientation, half extents.
    let centre = params.centre_maxdist.xyz;
    let max_dist = params.centre_maxdist.w;
    let rot_a = params.rotation;
    let half = params.half_pad.xyz;
    let dir = params.dir_pad.xyz;
    let radius_a = 0.0;

    // Box core: the eight local corners of the half-extent box, matching
    // ConvexHull::from_box(half). The sign bits (x: bit 0, y: bit 1, z: bit 2)
    // index the corners; posing them at the centre with the box rotation maps
    // the local frame onto the oriented box.
    local_verts[0] = vec3<f32>(-half.x, -half.y, -half.z);
    local_verts[1] = vec3<f32>( half.x, -half.y, -half.z);
    local_verts[2] = vec3<f32>(-half.x,  half.y, -half.z);
    local_verts[3] = vec3<f32>( half.x,  half.y, -half.z);
    local_verts[4] = vec3<f32>(-half.x, -half.y,  half.z);
    local_verts[5] = vec3<f32>( half.x, -half.y,  half.z);
    local_verts[6] = vec3<f32>(-half.x,  half.y,  half.z);
    local_verts[7] = vec3<f32>( half.x,  half.y,  half.z);
    local_verts[8] = va;
    local_verts[9] = vb;
    local_verts[10] = vc;

    body_a.vert_offset = BOX_OFFSET;
    body_a.vert_count = BOX_COUNT;
    body_b.vert_offset = TRIANGLE_OFFSET;
    body_b.vert_count = TRIANGLE_COUNT;

    let radius_b = 0.0;

    // Box motion: pose at the centre with the box rotation, translate along the
    // sweep direction, no spin. Triangle: static identity at the world origin.
    let pos0_a = centre;
    let rot0_a = rot_a;
    let lin_a = dir;
    let ang_a = vec3<f32>(0.0, 0.0, 0.0);
    let pos0_b = vec3<f32>(0.0, 0.0, 0.0);
    let rot0_b = vec4<f32>(0.0, 0.0, 0.0, 1.0);
    let lin_b = vec3<f32>(0.0, 0.0, 0.0);
    let ang_b = vec3<f32>(0.0, 0.0, 0.0);

    // Conservative-advancement parameters (operation-matched to the CPU twin).
    let dt = max_dist;
    let target_sep = 0.0;
    let core_target = target_sep + radius_a + radius_b;

    let r_a = circumradius(body_a.vert_offset, body_a.vert_count);
    let r_b = circumradius(body_b.vert_offset, body_b.vert_count);
    let ang_bound = length(ang_a) * r_a + length(ang_b) * r_b;
    let rel_linear = lin_a - lin_b;

    var last_normal = vec3<f32>(1.0, 0.0, 0.0);
    let rel_len = length(rel_linear);
    if (rel_len > CLOSING_EPS) {
        last_normal = -rel_linear / rel_len;
    }

    var out: Toi;
    out.time_point = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    out.hit_normal = vec4<f32>(0.0, last_normal.x, last_normal.y, last_normal.z);

    var t = 0.0;
    var hit = false;
    for (var iter = 0u; iter < MAX_CA_ITERS; iter = iter + 1u) {
        var pa_pos: vec3<f32>;
        var pa_rot: vec4<f32>;
        var pb_pos: vec3<f32>;
        var pb_rot: vec4<f32>;
        pose_at(pos0_a, rot0_a, lin_a, ang_a, t, &pa_pos, &pa_rot);
        pose_at(pos0_b, rot0_b, lin_b, ang_b, t, &pb_pos, &pb_rot);
        body_a.pos = pa_pos;
        body_a.rot = pa_rot;
        body_b.pos = pb_pos;
        body_b.rot = pb_rot;

        let res = run_gjk();
        if (res.intersecting) {
            // Cores have met (or already overlapped at the substep start): the
            // CPU twin reports the moving body's translation as the point.
            out.time_point = vec4<f32>(t, pa_pos.x, pa_pos.y, pa_pos.z);
            out.hit_normal = vec4<f32>(HIT_FLAG, last_normal.x, last_normal.y, last_normal.z);
            hit = true;
            break;
        }
        if (res.distance > CLOSING_EPS) {
            last_normal = res.normal;
        }
        if (res.distance <= core_target + DISTANCE_TOL) {
            // Push each core witness out to its inflated surface along the
            // contact normal (B toward A), then report the midpoint.
            let surf_a = res.point_a - last_normal * radius_a;
            let surf_b = res.point_b + last_normal * radius_b;
            let point = (surf_a + surf_b) * 0.5;
            out.time_point = vec4<f32>(t, point.x, point.y, point.z);
            out.hit_normal = vec4<f32>(HIT_FLAG, last_normal.x, last_normal.y, last_normal.z);
            hit = true;
            break;
        }
        let lin_closing = dot(rel_linear, -last_normal);
        let mu = lin_closing + ang_bound;
        if (mu <= CLOSING_EPS) {
            break;
        }
        t = t + (res.distance - core_target) / mu;
        if (t > dt) {
            break;
        }
    }

    if (!hit) {
        out.time_point = vec4<f32>(0.0, 0.0, 0.0, 0.0);
        out.hit_normal = vec4<f32>(0.0, last_normal.x, last_normal.y, last_normal.z);
    }
    results[i] = out;
}
