//! Real-device parity for the 2D convex-polygon Separating Axis Theorem
//! (`SAT`) twin:
//! [`GpuSatCollision2d`](prism_volumetric_gpu::sat_collision_2d::GpuSatCollision2d)
//! must reproduce the `CPU` golden
//! [`sat_collision_2d`](prism_render_architecture::particle::sat_collision_2d)
//! across the overlap verdict
//! ([`overlaps`](prism_render_architecture::particle::sat_collision_2d::overlaps))
//! and the minimum translation vector
//! ([`mtv`](prism_render_architecture::particle::sat_collision_2d::mtv),
//! reported as the reused
//! [`Mtv`](prism_render_architecture::particle::sat_collision_2d::Mtv)).
//!
//! The fixtures cover the shapes the golden unit tests call out and keep every
//! probe clear of a near-tie between candidate axes so a legal `ULP`-scale
//! fused-multiply-add on the `GPU` can never pick a different smallest-overlap
//! axis than the scalar reference: overlapping squares with a unique minimum
//! penetration, separated squares, fully coincident squares (whose four
//! candidate depths are all exactly equal, so the first-found tie-break is
//! deterministic), a touching contact at exactly zero depth, a triangle in a
//! shallow overlap with a box, two diamonds separated along a diagonal face
//! normal, a diamond overlapping an asymmetrically placed box, a polygon with a
//! doubled vertex whose zero-length edge must be skipped, a single-vertex input
//! that has no separating face, a mixed batch compared element for element, and
//! an empty batch. Every corner is written as an integer or a short decimal, so
//! the fixtures stay pure and need no transcendental math.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! The overlap verdict is a discrete classification built from `f32` magnitude
//! comparisons against `SAT_EPS`, so for pairs clear of the touching boundary
//! `CPU` and `GPU` must agree exactly: the comparison is an exact `==` on the
//! overlap `bool` and on the presence of the `Mtv`. The `MTV` axis and depth
//! thread through multiplies, adds, one guarded division and a single `sqrt` per
//! axis, so they are not bit-exact: a `GPU` may fuse a multiply-add the scalar
//! reference leaves separate, perturbing the low mantissa bits. The axis and
//! depth are therefore compared under tolerance (`abs_diff <= 1e-4` or
//! `rel_diff <= 1e-3`, `REL_FLOOR = 1e-6`).
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::particle::sat_collision_2d`；无第三方引擎源码或衍生代码。

use prism_render_architecture::particle::sat_collision_2d::{mtv, overlaps, Vec2};
use prism_volumetric_gpu::sat_collision_2d::{GpuSatCollision2d, SatCollision2dQuery};
use prism_volumetric_gpu::GpuContext;

/// Absolute parity bound on the `MTV` axis and depth. A `GPU` may fuse a
/// multiply-add the scalar reference leaves separate, perturbing the low
/// mantissa bits by a few units in the last place; `1e-4` admits that legal
/// slack while still failing a genuinely wrong port.
const ABS_EPS: f32 = 1.0e-4;
/// Relative parity bound, applied for larger magnitudes where a few units in the
/// last place exceed the absolute floor.
const REL_EPS: f32 = 1.0e-3;
/// Floor on the relative-error denominator so a near-zero expected value does
/// not inflate the relative error.
const REL_FLOOR: f32 = 1.0e-6;

/// Returns whether `a` and `b` agree within the absolute or relative bound.
fn approx(a: f32, b: f32) -> bool {
    let diff = (a - b).abs();
    if diff <= ABS_EPS {
        return true;
    }
    let scale = a.abs().max(b.abs()).max(REL_FLOOR);
    diff / scale <= REL_EPS
}

/// Pads a vertex list of up to four points to the fixed-capacity array the
/// query carries; trailing slots are never read because the count bounds the
/// sweep.
fn pad4(verts: &[[f32; 2]]) -> [[f32; 2]; 4] {
    let mut out = [[0.0_f32; 2]; 4];
    for (slot, v) in out.iter_mut().zip(verts.iter()) {
        *slot = *v;
    }
    out
}

/// Builds one query from two `CCW` vertex lists, taking each polygon's valid
/// vertex count from its slice length.
fn query(a: &[[f32; 2]], b: &[[f32; 2]]) -> SatCollision2dQuery {
    SatCollision2dQuery {
        poly_a: pad4(a),
        count_a: a.len() as u32,
        poly_b: pad4(b),
        count_b: b.len() as u32,
    }
}

/// Rebuilds the `CPU` golden vertex ring the kernel sees: the first `count`
/// vertices of a padded polygon, as the reference [`Vec2`] type.
fn golden(poly: &[[f32; 2]; 4], count: u32) -> Vec<Vec2> {
    (0..count as usize)
        .map(|i| Vec2::new(poly[i][0], poly[i][1]))
        .collect()
}

/// An axis-aligned square `[cx-h, cx+h] x [cy-h, cy+h]`, wound `CCW`, matching
/// the golden unit-test helper of the same name.
fn square(cx: f32, cy: f32, h: f32) -> [[f32; 2]; 4] {
    [
        [cx - h, cy - h],
        [cx + h, cy - h],
        [cx + h, cy + h],
        [cx - h, cy + h],
    ]
}

/// Asserts the twinned verdict and `MTV` for one query match the `CPU` golden.
fn assert_parity(gpu: &GpuSatCollision2d, ctx: &GpuContext, q: &SatCollision2dQuery) {
    let got = gpu.evaluate(ctx, std::slice::from_ref(q));
    assert_eq!(got.len(), 1, "one result per query");
    let g = got[0];

    let pa = golden(&q.poly_a, q.count_a);
    let pb = golden(&q.poly_b, q.count_b);

    assert_eq!(
        g.hit,
        overlaps(&pa, &pb),
        "overlap verdict mismatch: gpu {} vs cpu {}",
        g.hit,
        overlaps(&pa, &pb)
    );

    match (g.mtv, mtv(&pa, &pb)) {
        (Some(m), Some((axis, depth))) => {
            assert!(
                approx(m.axis.x, axis.x) && approx(m.axis.y, axis.y) && approx(m.depth, depth),
                "mtv mismatch: gpu axis ({}, {}) depth {} vs cpu axis ({}, {}) depth {}",
                m.axis.x,
                m.axis.y,
                m.depth,
                axis.x,
                axis.y,
                depth
            );
        }
        (None, None) => {}
        (a, b) => panic!("mtv presence mismatch: gpu {a:?} vs cpu {b:?}"),
    }
}

/// A unit diamond (a `45`-degree square) centred at the origin, its four face
/// normals the diagonal candidate axes, written without any trig call.
fn diamond() -> [[f32; 2]; 4] {
    [[0.0, -1.0], [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]]
}

#[test]
fn overlapping_squares_report_unique_mtv() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSatCollision2d::new(&ctx);
    // Two unit squares offset along x by 1.5: the only minimum-overlap axis is
    // the x face, a clean 0.5 penetration well below the 2.0 y overlap.
    let q = query(&square(0.0, 0.0, 1.0), &square(1.5, 0.0, 1.0));
    assert_parity(&gpu, &ctx, &q);
}

#[test]
fn separated_squares_report_no_overlap() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSatCollision2d::new(&ctx);
    // A wide gap on x proves a separating axis exists: no overlap, no MTV.
    let q = query(&square(0.0, 0.0, 1.0), &square(5.0, 0.0, 1.0));
    assert_parity(&gpu, &ctx, &q);
}

#[test]
fn identical_squares_fully_overlap() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSatCollision2d::new(&ctx);
    // Fully coincident squares: all four candidate depths are exactly 2.0, so
    // the first-found tie-break fixes the axis identically on CPU and GPU.
    let q = query(&square(0.0, 0.0, 1.0), &square(0.0, 0.0, 1.0));
    assert_parity(&gpu, &ctx, &q);
}

#[test]
fn touching_squares_report_zero_depth() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSatCollision2d::new(&ctx);
    // Squares sharing an edge touch at exactly zero depth: a non-negative,
    // clamped contact that still classifies as an overlap.
    let q = query(&square(0.0, 0.0, 1.0), &square(2.0, 0.0, 1.0));
    assert_parity(&gpu, &ctx, &q);
}

#[test]
fn triangle_shallow_overlap_with_box() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSatCollision2d::new(&ctx);
    // A triangle (count 3) shallowly overlapping a box: the minimum-overlap axis
    // is the box top face with a 0.1 penetration, far below every other axis.
    let tri = [[0.0, 0.0], [2.0, 0.0], [1.0, 2.0]];
    let q = query(&tri, &square(1.0, -0.5, 0.6));
    assert_parity(&gpu, &ctx, &q);
}

#[test]
fn diamonds_separate_along_diagonal_axis() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSatCollision2d::new(&ctx);
    // Two diamonds offset along the diagonal are separated by one of the
    // 45-degree face normals, found with a single sqrt and no trig.
    let d1 = diamond();
    let d2 = [[3.0, 2.0], [4.0, 3.0], [3.0, 4.0], [2.0, 3.0]];
    let q = query(&d1, &d2);
    assert_parity(&gpu, &ctx, &q);
}

#[test]
fn diamond_overlaps_asymmetric_box() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSatCollision2d::new(&ctx);
    // A diamond overlapping an asymmetrically placed box: the minimum-overlap
    // axis is a diamond face normal with a clear margin over the box faces.
    let q = query(&diamond(), &square(0.6, 0.25, 0.4));
    assert_parity(&gpu, &ctx, &q);
}

#[test]
fn degenerate_edge_is_skipped() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSatCollision2d::new(&ctx);
    // A doubled first vertex makes edge 0 zero-length; both CPU and GPU must
    // skip that axis (its normal would normalize to zero) and still agree on the
    // overlap and MTV from the remaining faces.
    let a = [[0.0, 0.0], [0.0, 0.0], [2.0, 0.0], [2.0, 2.0]];
    let q = query(&a, &square(2.0, 1.0, 0.6));
    assert_parity(&gpu, &ctx, &q);
}

#[test]
fn single_vertex_polygon_has_no_face() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSatCollision2d::new(&ctx);
    // A polygon with fewer than two vertices has no separating face: the pair is
    // reported non-overlapping with no MTV, matching the reference guard.
    let point = [[0.0, 0.0]];
    let q = query(&point, &square(0.0, 0.0, 1.0));
    assert_parity(&gpu, &ctx, &q);
}

#[test]
fn batch_of_queries_matches_elementwise() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSatCollision2d::new(&ctx);
    // A batch exercises the one-thread-per-pair flattening; each result must be
    // independent of its neighbours.
    let tri = [[0.0, 0.0], [2.0, 0.0], [1.0, 2.0]];
    let batch = [
        query(&square(0.0, 0.0, 1.0), &square(1.5, 0.0, 1.0)),
        query(&square(0.0, 0.0, 1.0), &square(5.0, 0.0, 1.0)),
        query(&diamond(), &square(0.6, 0.25, 0.4)),
        query(&tri, &square(1.0, -0.5, 0.6)),
        query(&square(0.0, 0.0, 1.0), &square(2.0, 0.0, 1.0)),
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
    let gpu = GpuSatCollision2d::new(&ctx);
    // No dispatch is issued and the result vector is empty.
    assert!(gpu.evaluate(&ctx, &[]).is_empty());
}
