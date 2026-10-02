//! Real-device parity for the 2D convex-polygon `Gilbert-Johnson-Keerthi`
//! (`GJK`) overlap twin:
//! [`GpuGjk2d`](prism_volumetric_gpu::gjk_2d::GpuGjk2d) must reproduce the `CPU`
//! golden
//! [`intersects`](prism_render_architecture::particle::gjk_2d::intersects)
//! across an empty batch, overlapping and far-apart squares, overlapping and
//! disjoint triangles, a point inside and a point outside a polygon, identical
//! and distinct single points, crossing and parallel segments, overlapping and
//! disjoint many-vertex octagons, a rotated diamond over an axis square and a
//! large pseudo-random batch compared lane for lane.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! The overlap verdict is a discrete classification built from `f32` magnitude
//! and sign comparisons against the reference's `SEP_EPS`, `DIR_EPS_SQ` and
//! `DUP_EPS_SQ` bands, so for pairs clear of the contact boundary the `CPU` and
//! `GPU` fold the identical sequence of support projections and simplex trims
//! and the comparison asserts an exact `==` on the `0`/`1` overlap flag. Every
//! fixture — named and random — is placed either clearly separated or clearly
//! overlapping (a margin far wider than any legal `ULP`-scale perturbation), so
//! no verdict sits in a tie band where a fused multiply-add could flip it.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::particle::gjk_2d`；textbook
//! 2D `GJK` convex-overlap simplex search；无第三方引擎源码或衍生代码。

use prism_volumetric_gpu::gjk_2d::{cpu_reference, GpuGjk2d, GpuGjk2dQuery, GpuGjk2dResult};
use prism_volumetric_gpu::GpuContext;

/// Fixed-capacity vertex count per polygon in the twin's `std430` layout; the
/// test fixtures stay well within it.
const CAP: usize = 16;

/// Half-width of the rejection band for the random batch: a candidate pair is
/// kept only when every axis overlap or gap exceeds this margin, so the verdict
/// is unambiguous and far from the contact boundary.
const MARGIN: f32 = 0.25;

/// Packs two vertex rings into a [`GpuGjk2dQuery`], zero-padding the unused
/// fixed-capacity slots.
fn make(a: &[[f32; 2]], b: &[[f32; 2]]) -> GpuGjk2dQuery {
    assert!(a.len() <= CAP && b.len() <= CAP, "polygon exceeds capacity");
    let mut poly_a = [[0.0_f32; 2]; CAP];
    let mut poly_b = [[0.0_f32; 2]; CAP];
    poly_a[..a.len()].copy_from_slice(a);
    poly_b[..b.len()].copy_from_slice(b);
    GpuGjk2dQuery {
        poly_a,
        count_a: a.len() as u32,
        poly_b,
        count_b: b.len() as u32,
    }
}

/// Runs the `GPU` dispatch and asserts strict lane-for-lane parity against the
/// `CPU` golden: the `0`/`1` overlap flag must match exactly. Returns the `GPU`
/// verdicts for extra per-test assertions. Use only for fixtures placed clear
/// of the contact boundary.
fn check(ctx: &GpuContext, gpu: &GpuGjk2d, queries: &[GpuGjk2dQuery]) -> Vec<GpuGjk2dResult> {
    let got = gpu.evaluate(ctx, queries);
    assert_eq!(got.len(), queries.len(), "one result per query");
    for (lane, (g, q)) in got.iter().zip(queries.iter()).enumerate() {
        let cpu = cpu_reference(q);
        assert_eq!(
            g.intersects, cpu,
            "lane {lane}: intersects gpu {} vs cpu {cpu}",
            g.intersects
        );
    }
    got
}

/// The axis-aligned unit square `[0, 1]^2`, `CCW`.
fn unit_square() -> [[f32; 2]; 4] {
    [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]
}

/// A regular octagon of circumradius `2` centered at `(cx, cy)`, matching the
/// golden fixture shape.
fn octagon(cx: f32, cy: f32) -> [[f32; 2]; 8] {
    [
        [cx + 2.0, cy + 1.0],
        [cx + 1.0, cy + 2.0],
        [cx - 1.0, cy + 2.0],
        [cx - 2.0, cy + 1.0],
        [cx - 2.0, cy - 1.0],
        [cx - 1.0, cy - 2.0],
        [cx + 1.0, cy - 2.0],
        [cx + 2.0, cy - 1.0],
    ]
}

/// A tiny integer linear-congruential generator; only integer and divide work,
/// so no transcendental appears. Returns a value in `[0, 1)`.
fn lcg(state: &mut u64) -> f32 {
    *state = state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    let bits = (*state >> 40) as u32;
    (bits & 0x00ff_ffff) as f32 / 16_777_216.0
}

/// Builds an axis-aligned rectangle `CCW` from its center and half extents.
fn rect(cx: f32, cy: f32, hx: f32, hy: f32) -> [[f32; 2]; 4] {
    [
        [cx - hx, cy - hy],
        [cx + hx, cy - hy],
        [cx + hx, cy + hy],
        [cx - hx, cy + hy],
    ]
}

#[test]
fn empty_batch_is_empty() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuGjk2d::new(&ctx);
    let got = gpu.evaluate(&ctx, &[]);
    assert!(got.is_empty(), "an empty batch yields an empty result");
}

#[test]
fn overlapping_unit_squares_intersect() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuGjk2d::new(&ctx);
    let a = unit_square();
    // Shifted half a unit on x: a wide, unambiguous interior overlap.
    let b = [[0.5, 0.0], [1.5, 0.0], [1.5, 1.0], [0.5, 1.0]];
    let q = make(&a, &b);
    let got = check(&ctx, &gpu, &[q]);
    assert_eq!(got[0].intersects, 1, "overlapping squares intersect");
}

#[test]
fn far_apart_squares_do_not_intersect() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuGjk2d::new(&ctx);
    let a = unit_square();
    let b = [
        [100.0, 100.0],
        [101.0, 100.0],
        [101.0, 101.0],
        [100.0, 101.0],
    ];
    let q = make(&a, &b);
    let got = check(&ctx, &gpu, &[q]);
    assert_eq!(got[0].intersects, 0, "far-apart squares do not intersect");
}

#[test]
fn overlapping_triangles_intersect() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuGjk2d::new(&ctx);
    let t1 = [[0.0, 0.0], [2.0, 0.0], [0.0, 2.0]];
    let t2 = [[0.5, 0.5], [2.0, 0.5], [0.5, 2.0]];
    let q = make(&t1, &t2);
    let got = check(&ctx, &gpu, &[q]);
    assert_eq!(got[0].intersects, 1, "overlapping triangles intersect");
}

#[test]
fn disjoint_triangles_do_not_intersect() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuGjk2d::new(&ctx);
    let t1 = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]];
    let t2 = [[3.0, 3.0], [4.0, 3.0], [3.0, 4.0]];
    let q = make(&t1, &t2);
    let got = check(&ctx, &gpu, &[q]);
    assert_eq!(got[0].intersects, 0, "disjoint triangles do not intersect");
}

#[test]
fn point_inside_polygon_intersects() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuGjk2d::new(&ctx);
    let square = unit_square();
    // A point well inside the square interior, clear of every edge.
    let point = [[0.5, 0.5]];
    let q = make(&square, &point);
    let got = check(&ctx, &gpu, &[q]);
    assert_eq!(got[0].intersects, 1, "interior point intersects");
}

#[test]
fn point_outside_polygon_does_not_intersect() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuGjk2d::new(&ctx);
    let square = unit_square();
    let point = [[2.0, 2.0]];
    let q = make(&square, &point);
    let got = check(&ctx, &gpu, &[q]);
    assert_eq!(got[0].intersects, 0, "exterior point does not intersect");
}

#[test]
fn identical_points_intersect() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuGjk2d::new(&ctx);
    let a = [[1.5, -3.25]];
    let b = [[1.5, -3.25]];
    let q = make(&a, &b);
    let got = check(&ctx, &gpu, &[q]);
    assert_eq!(got[0].intersects, 1, "identical points intersect");
}

#[test]
fn distinct_points_do_not_intersect() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuGjk2d::new(&ctx);
    let a = [[0.0, 0.0]];
    let b = [[1.0, 1.0]];
    let q = make(&a, &b);
    let got = check(&ctx, &gpu, &[q]);
    assert_eq!(got[0].intersects, 0, "distinct points do not intersect");
}

#[test]
fn crossing_segments_intersect() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuGjk2d::new(&ctx);
    // A horizontal and a vertical segment crossing cleanly at the origin.
    let horizontal = [[-1.0, 0.0], [1.0, 0.0]];
    let vertical = [[0.0, -1.0], [0.0, 1.0]];
    let q = make(&horizontal, &vertical);
    let got = check(&ctx, &gpu, &[q]);
    assert_eq!(got[0].intersects, 1, "crossing segments intersect");
}

#[test]
fn parallel_segments_do_not_intersect() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuGjk2d::new(&ctx);
    // Two horizontal segments a full unit apart: no shared point.
    let lower = [[-1.0, 0.0], [1.0, 0.0]];
    let upper = [[-1.0, 1.0], [1.0, 1.0]];
    let q = make(&lower, &upper);
    let got = check(&ctx, &gpu, &[q]);
    assert_eq!(got[0].intersects, 0, "parallel segments do not intersect");
}

#[test]
fn many_vertex_octagons_overlap() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuGjk2d::new(&ctx);
    let a = octagon(0.0, 0.0);
    let b = octagon(1.0, 1.0);
    let q = make(&a, &b);
    let got = check(&ctx, &gpu, &[q]);
    assert_eq!(got[0].intersects, 1, "close octagons overlap");
}

#[test]
fn many_vertex_octagons_disjoint() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuGjk2d::new(&ctx);
    let a = octagon(0.0, 0.0);
    let b = octagon(20.0, 0.0);
    let q = make(&a, &b);
    let got = check(&ctx, &gpu, &[q]);
    assert_eq!(got[0].intersects, 0, "far octagons are disjoint");
}

#[test]
fn rotated_diamond_overlaps_axis_square() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuGjk2d::new(&ctx);
    let diamond = [[0.0, 1.0], [1.0, 0.0], [0.0, -1.0], [-1.0, 0.0]];
    let square = [[-0.5, -0.5], [0.5, -0.5], [0.5, 0.5], [-0.5, 0.5]];
    let q = make(&diamond, &square);
    let got = check(&ctx, &gpu, &[q]);
    assert_eq!(got[0].intersects, 1, "diamond overlaps the axis square");
}

#[test]
fn mixed_named_batch_matches_cpu() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuGjk2d::new(&ctx);
    // A single dispatch mixing hits and misses so one batch exercises both
    // verdict classes through the shared pipeline.
    let queries = [
        make(&unit_square(), &[[0.25, 0.25], [0.75, 0.25], [0.5, 0.75]]),
        make(&unit_square(), &[[5.0, 5.0], [6.0, 5.0], [5.5, 6.0]]),
        make(&octagon(0.0, 0.0), &octagon(0.5, 0.5)),
        make(&octagon(0.0, 0.0), &octagon(10.0, 10.0)),
    ];
    let got = check(&ctx, &gpu, &queries);
    assert_eq!(got[0].intersects, 1);
    assert_eq!(got[1].intersects, 0);
    assert_eq!(got[2].intersects, 1);
    assert_eq!(got[3].intersects, 0);
}

#[test]
fn random_batch_matches_cpu() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuGjk2d::new(&ctx);

    let mut state: u64 = 0x9e37_79b9_7f4a_7c15;
    let mut queries: Vec<GpuGjk2dQuery> = Vec::with_capacity(256);
    while queries.len() < 256 {
        // Two axis-aligned rectangles with random centers and half extents.
        let acx = lcg(&mut state) * 16.0 - 8.0;
        let acy = lcg(&mut state) * 16.0 - 8.0;
        let ahx = 0.5 + lcg(&mut state) * 1.5;
        let ahy = 0.5 + lcg(&mut state) * 1.5;
        let bcx = lcg(&mut state) * 16.0 - 8.0;
        let bcy = lcg(&mut state) * 16.0 - 8.0;
        let bhx = 0.5 + lcg(&mut state) * 1.5;
        let bhy = 0.5 + lcg(&mut state) * 1.5;

        // Signed per-axis overlap of the two intervals (positive = overlap).
        let ox = (acx + ahx).min(bcx + bhx) - (acx - ahx).max(bcx - bhx);
        let oy = (acy + ahy).min(bcy + bhy) - (acy - ahy).max(bcy - bhy);

        // Keep only pairs that are clearly overlapping on both axes or clearly
        // separated on at least one axis, so the verdict is unambiguous and far
        // from the contact boundary.
        let clear_overlap = ox > MARGIN && oy > MARGIN;
        let clear_separate = ox < -MARGIN || oy < -MARGIN;
        if !(clear_overlap || clear_separate) {
            continue;
        }

        let a = rect(acx, acy, ahx, ahy);
        let b = rect(bcx, bcy, bhx, bhy);
        queries.push(make(&a, &b));
    }

    let got = gpu.evaluate(&ctx, &queries);
    assert_eq!(got.len(), queries.len(), "one result per query");

    let mut saw_hit = false;
    let mut saw_miss = false;
    for (lane, (g, q)) in got.iter().zip(queries.iter()).enumerate() {
        let cpu = cpu_reference(q);
        assert_eq!(
            g.intersects, cpu,
            "lane {lane}: intersects gpu {} vs cpu {cpu}",
            g.intersects
        );
        saw_hit |= cpu == 1;
        saw_miss |= cpu == 0;
    }

    assert!(
        saw_hit && saw_miss,
        "random batch should produce both overlaps and separations"
    );
}
