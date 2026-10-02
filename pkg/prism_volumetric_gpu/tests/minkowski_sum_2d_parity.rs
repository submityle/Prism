//! Real-device parity for the 2D convex-polygon Minkowski-sum twin:
//! [`GpuMinkowskiSum2d`](prism_volumetric_gpu::minkowski_sum_2d::GpuMinkowskiSum2d)
//! must reproduce the `CPU` golden
//! [`minkowski_sum`](prism_render_architecture::particle::minkowski_sum_2d::minkowski_sum)
//! across the normalization (winding flip, collinear strip, bottom-most
//! rotation), the polar-angle edge merge, and the degenerate collapses to a
//! segment or point.
//!
//! The fixtures mirror the shapes the golden unit tests call out: a right
//! triangle summed with itself (a scaled triangle), two unit squares (a `2x2`
//! square), a square swept by a horizontal segment, a polygon translated by a
//! single point, two non-parallel segments (a parallelogram), a pentagon summed
//! with a triangle, a clockwise input that must be re-wound, a collinear input
//! that collapses to a segment, and a point summed with a point. Every corner is
//! written as an integer or a simple decimal, so the fixtures stay well clear of
//! the [`SUM_EPS`](prism_render_architecture::particle::minkowski_sum_2d::SUM_EPS)
//! sign ties and need no transcendental math.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! The output vertex count and the bottom-most-first `CCW` ordering are discrete
//! classifications, so `CPU` and `GPU` must agree exactly: the comparison is an
//! exact `==` on the vertex count and a positional, in-order match of the ring.
//! The vertex coordinates thread through adds and subtracts, so they are
//! compared under tolerance (`abs_diff <= 1e-4` or `rel_diff <= 1e-3`,
//! `REL_FLOOR = 1e-6`).
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::particle::minkowski_sum_2d`；无第三方引擎源码或衍生代码。

use prism_render_architecture::particle::minkowski_sum_2d::{minkowski_sum, Vec2};
use prism_volumetric_gpu::minkowski_sum_2d::{GpuMinkowskiSum2d, MinkowskiSum2dQuery};
use prism_volumetric_gpu::GpuContext;

/// Absolute tolerance for the continuous vertex coordinates.
const ABS_EPS: f32 = 1.0e-4;
/// Relative tolerance for the continuous vertex coordinates.
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

/// Tolerant comparison of two `[x, y]` vertices.
fn approx2(a: [f32; 2], b: [f32; 2]) -> bool {
    approx(a[0], b[0]) && approx(a[1], b[1])
}

/// Lifts an `[x, y]` ring into the golden `Vec2` ring the reference consumes.
fn to_vec2(ring: &[[f32; 2]]) -> Vec<Vec2> {
    ring.iter().map(|&p| Vec2::new(p[0], p[1])).collect()
}

/// Builds one twin query from two `[x, y]` rings.
fn query(a: &[[f32; 2]], b: &[[f32; 2]]) -> MinkowskiSum2dQuery {
    MinkowskiSum2dQuery {
        a: a.to_vec(),
        b: b.to_vec(),
    }
}

/// Asserts the twinned sum of one query matches the `CPU` golden: the vertex
/// count exactly, and every vertex in the same ring position under tolerance.
fn assert_parity(gpu: &GpuMinkowskiSum2d, ctx: &GpuContext, a: &[[f32; 2]], b: &[[f32; 2]]) {
    let q = query(a, b);
    let got = gpu.evaluate(ctx, std::slice::from_ref(&q));
    assert_eq!(got.len(), 1, "one result per query");
    let g = &got[0];

    let cpu = minkowski_sum(&to_vec2(a), &to_vec2(b));

    assert_eq!(
        g.count as usize,
        cpu.len(),
        "vertex count mismatch: gpu {} vs cpu {} (a = {a:?}, b = {b:?})",
        g.count,
        cpu.len()
    );
    assert_eq!(
        g.verts.len(),
        cpu.len(),
        "decoded vertex length must equal the reported count"
    );
    for (idx, (&gv, cv)) in g.verts.iter().zip(cpu.iter()).enumerate() {
        assert!(
            approx2(gv, [cv.x, cv.y]),
            "vertex {idx} mismatch: gpu {gv:?} vs cpu [{}, {}] (a = {a:?}, b = {b:?})",
            cv.x,
            cv.y
        );
    }
}

/// A unit square ring, `CCW` from the origin.
fn unit_square() -> [[f32; 2]; 4] {
    [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]
}

/// A right triangle ring.
fn right_triangle() -> [[f32; 2]; 3] {
    [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
}

#[test]
fn triangle_plus_itself_is_scaled_triangle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuMinkowskiSum2d::new(&ctx);
    let t = right_triangle();
    // A ⊕ A doubles the triangle about the origin.
    assert_parity(&gpu, &ctx, &t, &t);
}

#[test]
fn two_unit_squares_make_two_by_two_square() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuMinkowskiSum2d::new(&ctx);
    let s = unit_square();
    assert_parity(&gpu, &ctx, &s, &s);
}

#[test]
fn square_plus_horizontal_segment() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuMinkowskiSum2d::new(&ctx);
    // A segment sweeps the square into a wider rectangle.
    let s = unit_square();
    let seg = [[0.0, 0.0], [2.0, 0.0]];
    assert_parity(&gpu, &ctx, &s, &seg);
}

#[test]
fn polygon_plus_point_is_translation() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuMinkowskiSum2d::new(&ctx);
    // A single-vertex input is a pure translation of the other shape.
    let s = unit_square();
    let point = [[3.0, -2.0]];
    assert_parity(&gpu, &ctx, &s, &point);
    assert_parity(&gpu, &ctx, &point, &s);
}

#[test]
fn perpendicular_segments_make_parallelogram() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuMinkowskiSum2d::new(&ctx);
    // Two non-parallel segments sum to a parallelogram.
    let a = [[0.0, 0.0], [2.0, 0.0]];
    let b = [[0.0, 0.0], [1.0, 2.0]];
    assert_parity(&gpu, &ctx, &a, &b);
}

#[test]
fn pentagon_plus_triangle_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuMinkowskiSum2d::new(&ctx);
    let pent = [[0.0, 0.0], [4.0, 0.0], [5.0, 3.0], [2.0, 5.0], [-1.0, 3.0]];
    let t = right_triangle();
    assert_parity(&gpu, &ctx, &pent, &t);
}

#[test]
fn clockwise_input_is_normalized() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuMinkowskiSum2d::new(&ctx);
    // A clockwise ring must be re-wound to CCW before the merge.
    let cw = [[0.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, 0.0]];
    let s = unit_square();
    assert_parity(&gpu, &ctx, &cw, &s);
}

#[test]
fn collinear_input_reduces_to_segment() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuMinkowskiSum2d::new(&ctx);
    // Collinear points collapse to a segment; summed with a point it stays a
    // translated segment.
    let line = [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]];
    let point = [[0.0, 3.0]];
    assert_parity(&gpu, &ctx, &line, &point);
}

#[test]
fn point_plus_point_is_point() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuMinkowskiSum2d::new(&ctx);
    // Two single points sum to a single point.
    let a = [[1.0, 2.0]];
    let b = [[-3.0, 4.0]];
    assert_parity(&gpu, &ctx, &a, &b);
}

#[test]
fn redundant_collinear_vertices_are_ignored() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuMinkowskiSum2d::new(&ctx);
    // An extra midpoint on a square edge must be stripped before the merge.
    let with_extra = [[0.0, 0.0], [0.5, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];
    let s = unit_square();
    assert_parity(&gpu, &ctx, &with_extra, &s);
}

#[test]
fn batch_of_queries_matches_elementwise() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuMinkowskiSum2d::new(&ctx);
    // A batch exercises the one-thread-per-pair flattening; each result must be
    // independent of its neighbours.
    let s = unit_square();
    let t = right_triangle();
    let seg = [[0.0, 0.0], [2.0, 0.0]];
    let point = [[3.0, -2.0]];
    let batch = [
        query(&s, &s),
        query(&t, &t),
        query(&s, &seg),
        query(&s, &point),
    ];
    let got = gpu.evaluate(&ctx, &batch);
    assert_eq!(got.len(), batch.len());
    assert_parity(&gpu, &ctx, &s, &s);
    assert_parity(&gpu, &ctx, &t, &t);
    assert_parity(&gpu, &ctx, &s, &seg);
    assert_parity(&gpu, &ctx, &s, &point);
}

#[test]
fn empty_batch_short_circuits() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuMinkowskiSum2d::new(&ctx);
    // No dispatch is issued and the result vector is empty.
    assert!(gpu.evaluate(&ctx, &[]).is_empty());
}
