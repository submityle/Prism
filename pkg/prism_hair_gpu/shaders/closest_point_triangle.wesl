//! `wgpu` compute twin of Prism's closest-point-on-triangle primitive
//! (`prism_render_architecture::hair::binding::closest_point_on_triangle`).
//!
//! The strand-root binder brute-force scans every scalp triangle for the one
//! whose closest surface point is nearest a root, so the closest-point test is
//! the inner kernel of `bind_roots` (twinned by `root_bind`). It is also a
//! stand-alone geometry primitive the collision, `SDF` and resample stages lean
//! on. This kernel is that primitive's isolated twin: one thread per
//! `(p, a, b, c)` query returns the point on triangle `abc` closest to `p`
//! plus its barycentric weights, so a passing real-device parity test is direct
//! evidence each of the seven Voronoi regions ports bit-faithfully — coverage
//! the aggregate nearest-triangle scan in `root_bind` cannot isolate, since its
//! output only reports the winning face's binding, not which region each
//! candidate resolved to.
//!
//! # Region structure
//!
//! This is the standard Voronoi-region test from Ericson's Real-Time Collision
//! Detection: three vertex regions (`A`/`B`/`C`), three edge regions
//! (`AB`/`AC`/`BC`) and the interior face region, decided in that fixed order by
//! sign tests on the six edge dot products `d1..d6` and the barycentric
//! numerators `va`/`vb`/`vc`. The branch order and every comparison match the
//! reference exactly, so both sides pick the same region for the same query.
//!
//! # Bit-faithfulness
//!
//! The vertex regions return a triangle corner and a canonical barycentric
//! basis (`[1,0,0]` etc.) with no arithmetic, so they are bit-exact. The edge
//! and interior regions divide precomputed dot-product combinations, which a
//! `GPU` may evaluate with a fused multiply-add the scalar reference cannot, so
//! the closest point and interpolated weights can differ by a few low-mantissa
//! `ULP`; the parity test keeps queries clear of region boundaries (so the
//! branch pick is unambiguous) and asserts the point within `1e-4` and each
//! weight within `abs < 1e-4` or `rel < 1e-3`, while vertex-region hits are
//! asserted exactly.
//!
//! # Portability
//!
//! The kernel uses only subtraction, `dot`, multiply, compare and one divide in
//! the portable core-`WGSL` subset — no `exp`, `pow`, atomics or optional device
//! feature — so the twin runs unmodified on Metal, Vulkan and DX12.
//!
//! Provenance: standard closest-point-on-triangle (Ericson Voronoi-region test)
//! plus `wgpu` compute dispatch; no Unreal Engine source or derived code.

// Squared-length degeneracy guard (mirrors `binding::EPS_LEN_SQ`).
const EPS_LEN_SQ: f32 = 1.0e-12;

/// Query count for the one-thread-per-query dispatch, padded to `16` bytes.
struct Params {
    query_count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

/// One closest-point query: the free point `p` and the triangle corners
/// `a`/`b`/`c`, each `xyz` in a `vec4` (w unused) so the stride is a `16`-byte
/// multiple with no `vec3` alignment hazard.
struct Query {
    p: vec4<f32>,
    a: vec4<f32>,
    b: vec4<f32>,
    c: vec4<f32>,
}

/// One result: the closest point (`xyz`) and its barycentric weights (`xyz`),
/// each packed into a `vec4` (w unused).
struct Result {
    point: vec4<f32>,
    bary: vec4<f32>,
}

/// Closest point plus barycentric weights returned from the region test.
struct Closest {
    point: vec3<f32>,
    bary: vec3<f32>,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read_write> out_results: array<Result>;

// Bit-faithful port of `binding::closest_point_on_triangle`: the standard
// Voronoi-region test from Ericson's Real-Time Collision Detection. Returns the
// point on triangle `abc` closest to `p` plus its barycentric weights; the
// branch order and every comparison match the reference exactly so both sides
// pick the same region.
fn closest_point_on_triangle(p: vec3<f32>, a: vec3<f32>, b: vec3<f32>, c: vec3<f32>) -> Closest {
    let ab = b - a;
    let ac = c - a;
    let ap = p - a;

    let d1 = dot(ab, ap);
    let d2 = dot(ac, ap);
    // Vertex region A.
    if d1 <= 0.0 && d2 <= 0.0 {
        return Closest(a, vec3<f32>(1.0, 0.0, 0.0));
    }

    let bp = p - b;
    let d3 = dot(ab, bp);
    let d4 = dot(ac, bp);
    // Vertex region B.
    if d3 >= 0.0 && d4 <= d3 {
        return Closest(b, vec3<f32>(0.0, 1.0, 0.0));
    }

    // Edge region AB.
    let vc = d1 * d4 - d3 * d2;
    if vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0 {
        let denom = d1 - d3;
        var v = 0.0;
        if denom > EPS_LEN_SQ {
            v = d1 / denom;
        }
        let point = a + ab * v;
        return Closest(point, vec3<f32>(1.0 - v, v, 0.0));
    }

    let cp = p - c;
    let d5 = dot(ab, cp);
    let d6 = dot(ac, cp);
    // Vertex region C.
    if d6 >= 0.0 && d5 <= d6 {
        return Closest(c, vec3<f32>(0.0, 0.0, 1.0));
    }

    // Edge region AC.
    let vb = d5 * d2 - d1 * d6;
    if vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0 {
        let denom = d2 - d6;
        var w = 0.0;
        if denom > EPS_LEN_SQ {
            w = d2 / denom;
        }
        let point = a + ac * w;
        return Closest(point, vec3<f32>(1.0 - w, 0.0, w));
    }

    // Edge region BC.
    let va = d3 * d6 - d5 * d4;
    let bc_num = d4 - d3;
    let bc_den = (d4 - d3) + (d5 - d6);
    if va <= 0.0 && bc_num >= 0.0 && (d5 - d6) >= 0.0 {
        var w = 0.0;
        if bc_den > EPS_LEN_SQ {
            w = bc_num / bc_den;
        }
        let point = b + (c - b) * w;
        return Closest(point, vec3<f32>(0.0, 1.0 - w, w));
    }

    // Interior face region: barycentric via the precomputed numerators.
    let denom_sum = va + vb + vc;
    if denom_sum > EPS_LEN_SQ {
        let inv = 1.0 / denom_sum;
        let v = vb * inv;
        let w = vc * inv;
        let point = a + ab * v + ac * w;
        return Closest(point, vec3<f32>(1.0 - v - w, v, w));
    }
    // Zero-area triangle: fall back to vertex A.
    return Closest(a, vec3<f32>(1.0, 0.0, 0.0));
}

@compute @workgroup_size(64)
fn evaluate(@builtin(global_invocation_id) gid: vec3<u32>) {
    let q = gid.x;
    if q >= params.query_count {
        return;
    }
    let query = queries[q];
    let hit = closest_point_on_triangle(query.p.xyz, query.a.xyz, query.b.xyz, query.c.xyz);
    out_results[q].point = vec4<f32>(hit.point, 0.0);
    out_results[q].bary = vec4<f32>(hit.bary, 0.0);
}
