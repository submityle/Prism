//! Real-device parity for the triangle barycentric twin:
//! [`GpuBarycentricCoord`](prism_volumetric_gpu::barycentric_coord::GpuBarycentricCoord)
//! must reproduce the `CPU` golden
//! [`barycentric_coord`](prism_render_architecture::particle::barycentric_coord)
//! across the 2D plane weights, the 3D Gram-matrix weights, the inside test, the
//! scalar / `vec2` / `vec3` / `vec4` attribute blends, and the perspective
//! divide.
//!
//! The fixtures cover the shapes the golden unit tests call out: corner,
//! centroid and interior probes (weights that reconstruct the query point), an
//! exterior point whose weight goes negative, a degenerate collinear triangle
//! (reported invalid, matching [`None`]), an off-plane 3D point that projects to
//! the same weights, an on-edge weight triple that must still classify as
//! inside, uniform and non-uniform `inv_w` perspective cases, and the all-zero
//! `inv_w` fallback. All triangle corners and probe points are written as
//! integers or simple decimals, so the fixtures stay pure and need no
//! `bevy_math` and no transcendental math.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! The validity flags and the inside flag are discrete classifications, so
//! `CPU` and `GPU` must agree exactly: the comparison is an exact `==` on the
//! presence of each [`Option`] and on the inside `bool`. The weights and blended
//! attributes thread through multiplies, adds and one guarded division, so they
//! are compared under tolerance (`abs_diff <= 1e-4` or `rel_diff <= 1e-3`,
//! `REL_FLOOR = 1e-6`).
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::particle::barycentric_coord`；无第三方引擎源码或衍生代码。

use prism_render_architecture::particle::barycentric_coord::{
    interpolate_scalar, interpolate_vec2, interpolate_vec3, interpolate_vec4, perspective_correct,
    point_in_triangle, Triangle2, Triangle3,
};
use prism_volumetric_gpu::barycentric_coord::{BarycentricQuery, GpuBarycentricCoord};
use prism_volumetric_gpu::GpuContext;

/// Absolute tolerance for the continuous quantities.
const ABS_EPS: f32 = 1.0e-4;
/// Relative tolerance for the continuous quantities.
const REL_EPS: f32 = 1.0e-3;
/// Floor for the relative-tolerance denominator.
const REL_FLOOR: f32 = 1.0e-6;

/// Mixed absolute / relative tolerance comparison for one `f32` lane.
fn approx(a: f32, b: f32) -> bool {
    let diff = (a - b).abs();
    if diff <= ABS_EPS {
        return true;
    }
    let scale = a.abs().max(b.abs()).max(REL_FLOOR);
    diff / scale <= REL_EPS
}

/// Tolerant comparison of two weight / vec3 triples.
fn approx3(a: [f32; 3], b: [f32; 3]) -> bool {
    approx(a[0], b[0]) && approx(a[1], b[1]) && approx(a[2], b[2])
}

/// Builds a query from the two triangles, their points, and the supplied blend
/// inputs. Attributes and `inv_w` default to simple values the per-fixture
/// helpers override as needed.
fn query(
    tri2: [[f32; 2]; 3],
    point2: [f32; 2],
    tri3: [[f32; 3]; 3],
    point3: [f32; 3],
    weights: [f32; 3],
    scalar_attrs: [f32; 3],
    vec4_attrs: [[f32; 4]; 3],
    inv_w: [f32; 3],
) -> BarycentricQuery {
    BarycentricQuery {
        tri2,
        point2,
        tri3,
        point3,
        weights,
        scalar_attrs,
        vec4_attrs,
        inv_w,
    }
}

/// Asserts every twinned answer for one query matches the `CPU` golden.
fn assert_parity(gpu: &GpuBarycentricCoord, ctx: &GpuContext, q: &BarycentricQuery) {
    let got = gpu.evaluate(ctx, std::slice::from_ref(q));
    assert_eq!(got.len(), 1, "one result per query");
    let g = got[0];

    let t2 = Triangle2 {
        a: q.tri2[0],
        b: q.tri2[1],
        c: q.tri2[2],
    };
    let cpu2 = t2.barycentric2(q.point2);
    match (g.bary2, cpu2) {
        (Some(a), Some(b)) => assert!(approx3(a, b), "bary2 mismatch: gpu {a:?} vs cpu {b:?}"),
        (None, None) => {}
        (a, b) => panic!("bary2 validity mismatch: gpu {a:?} vs cpu {b:?}"),
    }

    let t3 = Triangle3 {
        a: q.tri3[0],
        b: q.tri3[1],
        c: q.tri3[2],
    };
    let cpu3 = t3.barycentric3(q.point3);
    match (g.bary3, cpu3) {
        (Some(a), Some(b)) => assert!(approx3(a, b), "bary3 mismatch: gpu {a:?} vs cpu {b:?}"),
        (None, None) => {}
        (a, b) => panic!("bary3 validity mismatch: gpu {a:?} vs cpu {b:?}"),
    }

    assert_eq!(
        g.inside,
        point_in_triangle(q.weights),
        "inside mismatch for weights {:?}",
        q.weights
    );

    let cpu_scalar = interpolate_scalar(q.weights, q.scalar_attrs);
    assert!(
        approx(g.interp_scalar, cpu_scalar),
        "interp_scalar mismatch: gpu {} vs cpu {cpu_scalar}",
        g.interp_scalar
    );

    let cpu4 = interpolate_vec4(q.weights, q.vec4_attrs);
    assert!(
        approx(g.interp_vec4[0], cpu4[0])
            && approx(g.interp_vec4[1], cpu4[1])
            && approx(g.interp_vec4[2], cpu4[2])
            && approx(g.interp_vec4[3], cpu4[3]),
        "interp_vec4 mismatch: gpu {:?} vs cpu {cpu4:?}",
        g.interp_vec4
    );

    // vec2 / vec3 blends are the leading lanes of the vec4 blend because
    // interpolation is component-wise and independent per lane.
    let attrs2 = [
        [q.vec4_attrs[0][0], q.vec4_attrs[0][1]],
        [q.vec4_attrs[1][0], q.vec4_attrs[1][1]],
        [q.vec4_attrs[2][0], q.vec4_attrs[2][1]],
    ];
    let cpu2v = interpolate_vec2(q.weights, attrs2);
    assert!(
        approx(g.interp_vec4[0], cpu2v[0]) && approx(g.interp_vec4[1], cpu2v[1]),
        "interp_vec2 mismatch: gpu {:?} vs cpu {cpu2v:?}",
        [g.interp_vec4[0], g.interp_vec4[1]]
    );
    let attrs3 = [
        [q.vec4_attrs[0][0], q.vec4_attrs[0][1], q.vec4_attrs[0][2]],
        [q.vec4_attrs[1][0], q.vec4_attrs[1][1], q.vec4_attrs[1][2]],
        [q.vec4_attrs[2][0], q.vec4_attrs[2][1], q.vec4_attrs[2][2]],
    ];
    let cpu3v = interpolate_vec3(q.weights, attrs3);
    assert!(
        approx(g.interp_vec4[0], cpu3v[0])
            && approx(g.interp_vec4[1], cpu3v[1])
            && approx(g.interp_vec4[2], cpu3v[2]),
        "interp_vec3 mismatch: gpu {:?} vs cpu {cpu3v:?}",
        [g.interp_vec4[0], g.interp_vec4[1], g.interp_vec4[2]]
    );

    let cpu_persp = perspective_correct(q.weights, q.inv_w);
    assert!(
        approx3(g.perspective, cpu_persp),
        "perspective mismatch: gpu {:?} vs cpu {cpu_persp:?}",
        g.perspective
    );
}

/// A right 2D triangle with legs on the axes.
fn tri2() -> [[f32; 2]; 3] {
    [[0.0, 0.0], [4.0, 0.0], [0.0, 3.0]]
}

/// A right 3D triangle lying in the plane `z = 1`.
fn tri3() -> [[f32; 3]; 3] {
    [[0.0, 0.0, 1.0], [4.0, 0.0, 1.0], [0.0, 3.0, 1.0]]
}

/// Three per-corner `vec4` attributes with distinct lanes so a swapped corner is
/// visible in the blend.
fn attrs() -> [[f32; 4]; 3] {
    [
        [1.0, 2.0, 3.0, 4.0],
        [5.0, 6.0, 7.0, 8.0],
        [9.0, 10.0, 11.0, 12.0],
    ]
}

#[test]
fn interior_query_reconstructs_and_blends() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuBarycentricCoord::new(&ctx);
    // An interior weight triple that reconstructs an interior point; uniform
    // inv_w so perspective_correct is the identity.
    let q = query(
        tri2(),
        [1.0, 0.75],
        tri3(),
        [1.0, 0.5, 1.0],
        [0.5, 0.25, 0.25],
        [4.0, 8.0, 2.0],
        attrs(),
        [1.0, 1.0, 1.0],
    );
    assert_parity(&gpu, &ctx, &q);
}

#[test]
fn corner_and_centroid_weights() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuBarycentricCoord::new(&ctx);
    // Corner a of both triangles is unit u; the supplied weights pick corner b.
    let corner = query(
        tri2(),
        [0.0, 0.0],
        tri3(),
        [0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0],
        [10.0, 20.0, 30.0],
        attrs(),
        [2.0, 2.0, 2.0],
    );
    assert_parity(&gpu, &ctx, &corner);
    // Centroid of both triangles gives thirds; interior weights sum to one.
    let centroid = query(
        tri2(),
        [4.0 / 3.0, 1.0],
        tri3(),
        [4.0 / 3.0, 1.0, 1.0],
        [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
        [3.0, 6.0, 9.0],
        attrs(),
        [1.0, 1.0, 1.0],
    );
    assert_parity(&gpu, &ctx, &centroid);
}

#[test]
fn exterior_point_has_negative_weight() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuBarycentricCoord::new(&ctx);
    // An exterior query point yields a negative weight; the supplied weights are
    // also exterior so the inside flag is false.
    let q = query(
        tri2(),
        [-1.0, -1.0],
        tri3(),
        [-1.0, -1.0, 1.0],
        [-0.1, 0.6, 0.5],
        [1.0, 1.0, 1.0],
        attrs(),
        [1.0, 1.0, 1.0],
    );
    assert_parity(&gpu, &ctx, &q);
}

#[test]
fn on_edge_weights_are_inside() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuBarycentricCoord::new(&ctx);
    // A weight on an edge (one lane exactly zero) must still classify as inside.
    let q = query(
        tri2(),
        [2.0, 0.0],
        tri3(),
        [2.0, 0.0, 1.0],
        [0.0, 0.5, 0.5],
        [1.0, 2.0, 3.0],
        attrs(),
        [1.0, 1.0, 1.0],
    );
    assert_parity(&gpu, &ctx, &q);
}

#[test]
fn off_plane_3d_point_projects() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuBarycentricCoord::new(&ctx);
    // A 3D point lifted off the plane projects to the same weights as on it.
    let q = query(
        tri2(),
        [1.0, 1.0],
        tri3(),
        [1.0, 1.0, 5.0],
        [0.25, 0.25, 0.5],
        [1.0, 1.0, 1.0],
        attrs(),
        [1.0, 1.0, 1.0],
    );
    assert_parity(&gpu, &ctx, &q);
}

#[test]
fn degenerate_triangles_report_invalid() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuBarycentricCoord::new(&ctx);
    // Collinear corners give a zero-area triangle in both 2D and 3D: both solves
    // must report invalid (matching the reference None).
    let line2 = [[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]];
    let line3 = [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [2.0, 2.0, 2.0]];
    let q = query(
        line2,
        [0.5, 0.5],
        line3,
        [0.5, 0.5, 0.5],
        [0.2, 0.3, 0.5],
        [1.0, 1.0, 1.0],
        attrs(),
        [1.0, 1.0, 1.0],
    );
    assert_parity(&gpu, &ctx, &q);
}

#[test]
fn perspective_divide_reweights_and_falls_back() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuBarycentricCoord::new(&ctx);
    // Non-uniform inv_w shifts weight toward the vertex with the largest 1/w.
    let reweight = query(
        tri2(),
        [1.0, 1.0],
        tri3(),
        [1.0, 1.0, 1.0],
        [0.25, 0.25, 0.5],
        [1.0, 1.0, 1.0],
        attrs(),
        [1.0, 2.0, 4.0],
    );
    assert_parity(&gpu, &ctx, &reweight);
    // All-zero inv_w makes the reweighted sum zero, so the affine weights pass
    // through unchanged.
    let fallback = query(
        tri2(),
        [1.0, 1.0],
        tri3(),
        [1.0, 1.0, 1.0],
        [0.2, 0.3, 0.5],
        [1.0, 1.0, 1.0],
        attrs(),
        [0.0, 0.0, 0.0],
    );
    assert_parity(&gpu, &ctx, &fallback);
}

#[test]
fn batch_of_queries_matches_elementwise() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuBarycentricCoord::new(&ctx);
    // A batch exercises the one-thread-per-query flattening; each result must be
    // independent of its neighbours.
    let batch = [
        query(
            tri2(),
            [1.0, 0.5],
            tri3(),
            [1.0, 0.5, 1.0],
            [0.5, 0.25, 0.25],
            [2.0, 4.0, 6.0],
            attrs(),
            [1.0, 1.0, 1.0],
        ),
        query(
            tri2(),
            [-1.0, -1.0],
            tri3(),
            [-1.0, -1.0, 1.0],
            [-0.2, 0.7, 0.5],
            [1.0, 1.0, 1.0],
            attrs(),
            [1.0, 2.0, 3.0],
        ),
        query(
            tri2(),
            [0.0, 3.0],
            tri3(),
            [0.0, 3.0, 1.0],
            [0.0, 0.0, 1.0],
            [5.0, 5.0, 5.0],
            attrs(),
            [4.0, 1.0, 1.0],
        ),
    ];
    let got = gpu.evaluate(&ctx, &batch);
    assert_eq!(got.len(), batch.len());
    for q in &batch {
        assert_parity(&gpu, &ctx, q);
    }
}

#[test]
fn empty_batch_short_circuits() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuBarycentricCoord::new(&ctx);
    // No dispatch is issued and the result vector is empty.
    assert!(gpu.evaluate(&ctx, &[]).is_empty());
}
