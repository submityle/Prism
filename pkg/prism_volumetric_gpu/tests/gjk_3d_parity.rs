//! Real-device parity for the 3D convex-polytope `Gilbert-Johnson-Keerthi`
//! (`GJK`) overlap twin:
//! [`GpuGjk3d`](prism_volumetric_gpu::gjk_3d::GpuGjk3d) must reproduce the `CPU`
//! golden
//! [`intersect`](prism_render_architecture::particle::gjk_3d::intersect)
//! across an empty batch, overlapping and far-apart cubes, overlapping and
//! disjoint tetrahedra, a point inside and a point outside a cube, identical
//! and distinct single points, crossing and skew segments, a mixed named batch
//! and a large pseudo-random batch of axis-aligned boxes compared lane for
//! lane.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! The overlap verdict is a discrete classification built from `f32` dot/cross
//! sign comparisons and squared-magnitude comparisons against the reference's
//! `DIR_EPS_SQ` and `DUP_EPS_SQ` bands, so for pairs clear of the contact
//! boundary the `CPU` and `GPU` fold the identical sequence of support
//! projections and simplex trims and the comparison asserts an exact `==` on
//! the `0`/`1` overlap flag. Every fixture — named and random — is placed
//! either clearly separated or clearly overlapping (a margin far wider than any
//! legal `ULP`-scale perturbation), so no verdict sits in a tie band where a
//! fused multiply-add could flip it.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::particle::gjk_3d`；textbook
//! 3D `GJK` convex-overlap simplex search；无第三方引擎源码或衍生代码。

use prism_volumetric_gpu::gjk_3d::{cpu_reference, GpuGjk3d, GpuGjk3dQuery, GpuGjk3dResult};
use prism_volumetric_gpu::GpuContext;

/// Fixed-capacity vertex count per polytope in the twin's `std430` layout; the
/// test fixtures stay well within it.
const CAP: usize = 16;

/// Half-width of the rejection band for the random batch: a candidate pair is
/// kept only when every axis overlap or gap exceeds this margin, so the verdict
/// is unambiguous and far from the contact boundary.
const MARGIN: f32 = 0.25;

/// Packs two vertex sets into a [`GpuGjk3dQuery`], zero-padding the unused
/// fixed-capacity slots.
fn make(a: &[[f32; 3]], b: &[[f32; 3]]) -> GpuGjk3dQuery {
    assert!(
        a.len() <= CAP && b.len() <= CAP,
        "polytope exceeds capacity"
    );
    let mut poly_a = [[0.0_f32; 3]; CAP];
    let mut poly_b = [[0.0_f32; 3]; CAP];
    poly_a[..a.len()].copy_from_slice(a);
    poly_b[..b.len()].copy_from_slice(b);
    GpuGjk3dQuery {
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
fn check(ctx: &GpuContext, gpu: &GpuGjk3d, queries: &[GpuGjk3dQuery]) -> Vec<GpuGjk3dResult> {
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

/// An axis-aligned unit cube with its minimum corner at `origin`, matching the
/// golden fixture shape.
fn cube(origin: [f32; 3]) -> [[f32; 3]; 8] {
    let [x, y, z] = origin;
    [
        [x, y, z],
        [x + 1.0, y, z],
        [x, y + 1.0, z],
        [x + 1.0, y + 1.0, z],
        [x, y, z + 1.0],
        [x + 1.0, y, z + 1.0],
        [x, y + 1.0, z + 1.0],
        [x + 1.0, y + 1.0, z + 1.0],
    ]
}

/// An axis-aligned box spanning `[min, max]`, matching the golden fixture
/// shape.
fn boxv(min: [f32; 3], max: [f32; 3]) -> [[f32; 3]; 8] {
    [
        [min[0], min[1], min[2]],
        [max[0], min[1], min[2]],
        [min[0], max[1], min[2]],
        [max[0], max[1], min[2]],
        [min[0], min[1], max[2]],
        [max[0], min[1], max[2]],
        [min[0], max[1], max[2]],
        [max[0], max[1], max[2]],
    ]
}

/// A regular-ish tetrahedron scaled by `s` and translated by `t`, matching the
/// golden fixture shape.
fn tetra(s: f32, t: [f32; 3]) -> [[f32; 3]; 4] {
    [
        [t[0], t[1], t[2]],
        [t[0] + s, t[1], t[2]],
        [t[0], t[1] + s, t[2]],
        [t[0], t[1], t[2] + s],
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

/// Builds an axis-aligned box from its center and per-axis half extents.
fn centered_box(c: [f32; 3], h: [f32; 3]) -> [[f32; 3]; 8] {
    boxv(
        [c[0] - h[0], c[1] - h[1], c[2] - h[2]],
        [c[0] + h[0], c[1] + h[1], c[2] + h[2]],
    )
}

#[test]
fn empty_batch_is_empty() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuGjk3d::new(&ctx);
    let got = gpu.evaluate(&ctx, &[]);
    assert!(got.is_empty(), "empty batch returns no results");
}

#[test]
fn overlapping_cubes_intersect() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuGjk3d::new(&ctx);
    // Two unit cubes offset by half a unit on every axis: deep overlap.
    let a = cube([0.0, 0.0, 0.0]);
    let b = cube([0.5, 0.5, 0.5]);
    let q = make(&a, &b);
    let got = check(&ctx, &gpu, &[q]);
    assert_eq!(got[0].intersects, 1, "overlapping cubes intersect");
}

#[test]
fn far_cubes_do_not_intersect() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuGjk3d::new(&ctx);
    // A unit cube at the origin and another five units away on x: no overlap.
    let a = cube([0.0, 0.0, 0.0]);
    let b = cube([5.0, 0.0, 0.0]);
    let q = make(&a, &b);
    let got = check(&ctx, &gpu, &[q]);
    assert_eq!(got[0].intersects, 0, "far cubes are disjoint");
}

#[test]
fn overlapping_tetrahedra_intersect() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuGjk3d::new(&ctx);
    let a = tetra(2.0, [0.0, 0.0, 0.0]);
    let b = tetra(2.0, [0.4, 0.4, 0.4]);
    let q = make(&a, &b);
    let got = check(&ctx, &gpu, &[q]);
    assert_eq!(got[0].intersects, 1, "overlapping tetrahedra intersect");
}

#[test]
fn disjoint_tetrahedra_do_not_intersect() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuGjk3d::new(&ctx);
    let a = tetra(1.0, [0.0, 0.0, 0.0]);
    let b = tetra(1.0, [5.0, 5.0, 5.0]);
    let q = make(&a, &b);
    let got = check(&ctx, &gpu, &[q]);
    assert_eq!(got[0].intersects, 0, "far tetrahedra are disjoint");
}

#[test]
fn point_inside_cube_intersects() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuGjk3d::new(&ctx);
    // A single point well inside the unit cube.
    let point = [[0.5, 0.5, 0.5]];
    let a = cube([0.0, 0.0, 0.0]);
    let q = make(&a, &point);
    let got = check(&ctx, &gpu, &[q]);
    assert_eq!(got[0].intersects, 1, "point inside the cube intersects");
}

#[test]
fn point_outside_cube_does_not_intersect() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuGjk3d::new(&ctx);
    // A single point well clear of the unit cube.
    let point = [[5.0, 5.0, 5.0]];
    let a = cube([0.0, 0.0, 0.0]);
    let q = make(&a, &point);
    let got = check(&ctx, &gpu, &[q]);
    assert_eq!(got[0].intersects, 0, "point outside the cube is disjoint");
}

#[test]
fn identical_points_intersect() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuGjk3d::new(&ctx);
    let p = [[1.0, 2.0, 3.0]];
    let q = make(&p, &p);
    let got = check(&ctx, &gpu, &[q]);
    assert_eq!(got[0].intersects, 1, "identical points share a location");
}

#[test]
fn distinct_points_do_not_intersect() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuGjk3d::new(&ctx);
    let a = [[0.0, 0.0, 0.0]];
    let b = [[1.0, 0.0, 0.0]];
    let q = make(&a, &b);
    let got = check(&ctx, &gpu, &[q]);
    assert_eq!(got[0].intersects, 0, "distinct points are disjoint");
}

#[test]
fn crossing_segments_intersect() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuGjk3d::new(&ctx);
    // A segment along x and one along y, crossing cleanly at the origin.
    let along_x = [[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]];
    let along_y = [[0.0, -1.0, 0.0], [0.0, 1.0, 0.0]];
    let q = make(&along_x, &along_y);
    let got = check(&ctx, &gpu, &[q]);
    assert_eq!(got[0].intersects, 1, "crossing segments intersect");
}

#[test]
fn skew_segments_do_not_intersect() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuGjk3d::new(&ctx);
    // Same crossing pattern but lifted a full five units apart on z: skew lines
    // that never meet.
    let along_x = [[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]];
    let along_y = [[0.0, -1.0, 5.0], [0.0, 1.0, 5.0]];
    let q = make(&along_x, &along_y);
    let got = check(&ctx, &gpu, &[q]);
    assert_eq!(got[0].intersects, 0, "skew segments are disjoint");
}

#[test]
fn mixed_named_batch_matches_cpu() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuGjk3d::new(&ctx);
    // A single dispatch mixing hits and misses so one batch exercises both
    // verdict classes through the shared pipeline.
    let queries = [
        make(&cube([0.0, 0.0, 0.0]), &cube([0.5, 0.5, 0.5])),
        make(&cube([0.0, 0.0, 0.0]), &cube([5.0, 0.0, 0.0])),
        make(&tetra(2.0, [0.0, 0.0, 0.0]), &tetra(2.0, [0.4, 0.4, 0.4])),
        make(&tetra(1.0, [0.0, 0.0, 0.0]), &tetra(1.0, [5.0, 5.0, 5.0])),
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
    let gpu = GpuGjk3d::new(&ctx);

    let mut state: u64 = 0x9e37_79b9_7f4a_7c15;
    let mut queries: Vec<GpuGjk3dQuery> = Vec::with_capacity(256);
    while queries.len() < 256 {
        // Two axis-aligned boxes with random centers and half extents.
        let acx = lcg(&mut state) * 16.0 - 8.0;
        let acy = lcg(&mut state) * 16.0 - 8.0;
        let acz = lcg(&mut state) * 16.0 - 8.0;
        let ahx = 0.5 + lcg(&mut state) * 1.5;
        let ahy = 0.5 + lcg(&mut state) * 1.5;
        let ahz = 0.5 + lcg(&mut state) * 1.5;
        let bcx = lcg(&mut state) * 16.0 - 8.0;
        let bcy = lcg(&mut state) * 16.0 - 8.0;
        let bcz = lcg(&mut state) * 16.0 - 8.0;
        let bhx = 0.5 + lcg(&mut state) * 1.5;
        let bhy = 0.5 + lcg(&mut state) * 1.5;
        let bhz = 0.5 + lcg(&mut state) * 1.5;

        // Signed per-axis overlap of the two intervals (positive = overlap).
        let ox = (acx + ahx).min(bcx + bhx) - (acx - ahx).max(bcx - bhx);
        let oy = (acy + ahy).min(bcy + bhy) - (acy - ahy).max(bcy - bhy);
        let oz = (acz + ahz).min(bcz + bhz) - (acz - ahz).max(bcz - bhz);

        // Keep only pairs that are clearly overlapping on every axis or clearly
        // separated on at least one axis, so the verdict is unambiguous and far
        // from the contact boundary.
        let clear_overlap = ox > MARGIN && oy > MARGIN && oz > MARGIN;
        let clear_separate = ox < -MARGIN || oy < -MARGIN || oz < -MARGIN;
        if !(clear_overlap || clear_separate) {
            continue;
        }

        let a = centered_box([acx, acy, acz], [ahx, ahy, ahz]);
        let b = centered_box([bcx, bcy, bcz], [bhx, bhy, bhz]);
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
