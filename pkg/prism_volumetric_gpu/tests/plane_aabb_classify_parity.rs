//! Real-device parity for the single-plane `AABB` classifier twin:
//! [`GpuPlaneAabbClassify`](prism_volumetric_gpu::plane_aabb_classify::GpuPlaneAabbClassify)
//! must reproduce the `CPU` golden
//! [`plane_aabb_classify`](prism_render_architecture::particle::plane_aabb_classify)
//! across fully-positive, fully-negative and straddling boxes, the grazing
//! boundaries where `s - r` or `s + r` reaches zero, degenerate thin and point
//! boxes, a degenerate zero normal, unnormalized-normal consistency, boxes
//! built from min/max corners, oblique normals, and a batch of pseudo-random
//! queries compared element-for-element.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! The signed distance and projected radius are short, fixed-order sums of
//! products, so `CPU` and `GPU` evaluate the same closed form. They are not
//! bit-exact: a `GPU` may fuse a multiply-add the scalar reference leaves
//! separate, perturbing the low mantissa bits by a few units in the last place.
//! The comparison therefore allows `abs_diff <= 1e-4` or `rel_diff <= 1e-3` on
//! the two `f32` outputs and asserts the side discriminant exactly. Every
//! side-bearing fixture keeps `|s - r|` and `|s + r|` well clear of the
//! reference `CMP_EPS` band so a legal fused multiply-add can never flip the
//! reported side.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::particle::plane_aabb_classify`；
//! 无第三方引擎源码或衍生代码。

use prism_render_architecture::particle::plane_aabb_classify::{
    classify, projected_radius, signed_distance_center, Aabb, Plane, Side,
};
use prism_volumetric_gpu::plane_aabb_classify::{GpuPlaneAabbClassify, PlaneAabbClassifyQuery};
use prism_volumetric_gpu::GpuContext;

/// Absolute parity bound. A `GPU` may fuse a multiply-add the scalar reference
/// leaves separate, perturbing the low mantissa bits by a few units in the last
/// place; `1e-4` admits that legal slack while still failing a genuinely wrong
/// port.
const EPS: f32 = 1.0e-4;

/// Relative parity bound, applied for larger magnitudes where a few units in
/// the last place exceed the absolute floor.
const REL: f32 = 1.0e-3;

/// Floor on the relative-error denominator so a near-zero expected value does
/// not inflate the relative error.
const REL_FLOOR: f32 = 1.0e-6;

/// Returns whether `a` and `b` agree within the absolute or relative bound.
fn close(a: f32, b: f32) -> bool {
    let abs = (a - b).abs();
    let rel = abs / a.abs().max(b.abs()).max(REL_FLOOR);
    abs <= EPS || rel <= REL
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

/// Draws a value in `[-span, span)` from `state`.
fn signed(state: &mut u64, span: f32) -> f32 {
    (lcg(state) * 2.0 - 1.0) * span
}

/// Runs the `GPU` classifier and asserts element-for-element parity against the
/// `CPU` golden: the signed distance and projected radius within tolerance and
/// the side exactly.
fn check(ctx: &GpuContext, gpu: &GpuPlaneAabbClassify, queries: &[PlaneAabbClassifyQuery]) {
    let got = gpu.eval(ctx, queries);
    assert_eq!(got.len(), queries.len(), "one result per query");
    for (idx, (q, g)) in queries.iter().zip(got.iter()).enumerate() {
        let want_s = signed_distance_center(q.plane, q.aabb);
        let want_r = projected_radius(q.plane.normal, q.aabb.half);
        let want_side = classify(q.plane, q.aabb);
        assert!(
            close(g.signed_distance, want_s),
            "query {idx}: signed_distance gpu {} vs cpu {want_s}",
            g.signed_distance
        );
        assert!(
            close(g.projected_radius, want_r),
            "query {idx}: projected_radius gpu {} vs cpu {want_r}",
            g.projected_radius
        );
        assert_eq!(
            g.side, want_side,
            "query {idx}: side gpu {:?} vs cpu {want_side:?}",
            g.side
        );
    }
}

/// A unit-half-extent box centered at `c`.
fn unit_box_at(c: [f32; 3]) -> Aabb {
    Aabb::new(c, [1.0, 1.0, 1.0])
}

/// A plane whose normal points along `+x` with constant `d`.
fn plane_px(d: f32) -> Plane {
    Plane::new([1.0, 0.0, 0.0], d)
}

#[test]
fn empty_input_is_empty() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuPlaneAabbClassify::new(&ctx);
    // Storage buffers cannot be zero-sized; the twin short-circuits an empty
    // input to an empty result with no dispatch issued.
    assert!(
        gpu.eval(&ctx, &[]).is_empty(),
        "empty input yields no results"
    );
}

#[test]
fn clear_sides_match_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuPlaneAabbClassify::new(&ctx);
    // `+x` plane through the origin, unit box. Centers far from the plane put
    // `|s - r|` and `|s + r|` well clear of the CMP_EPS band so the side is
    // deterministic on any device.
    let queries = [
        PlaneAabbClassifyQuery::new(plane_px(0.0), unit_box_at([3.0, 0.0, 0.0])),
        PlaneAabbClassifyQuery::new(plane_px(0.0), unit_box_at([-3.0, 0.0, 0.0])),
        PlaneAabbClassifyQuery::new(plane_px(0.0), unit_box_at([0.0, 0.0, 0.0])),
    ];
    let got = gpu.eval(&ctx, &queries);
    assert_eq!(got[0].side, Side::Positive, "far +x box is positive");
    assert_eq!(got[1].side, Side::Negative, "far -x box is negative");
    assert_eq!(got[2].side, Side::Intersecting, "centered box straddles");
    check(&ctx, &gpu, &queries);
}

#[test]
fn clear_straddle_well_inside_band() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuPlaneAabbClassify::new(&ctx);
    // s = 0.5, r = 1 so s - r = -0.5 and s + r = 1.5: the center sits strictly
    // inside the projected radius, a clear straddle far from either end of the
    // band, exercising a stable Intersecting that is not a mere graze.
    let q = PlaneAabbClassifyQuery::new(plane_px(-2.0), unit_box_at([2.5, 0.0, 0.0]));
    let got = gpu.eval(&ctx, &[q]);
    assert_eq!(
        got[0].side,
        Side::Intersecting,
        "center inside radius straddles"
    );
    check(&ctx, &gpu, &[q]);
}

#[test]
fn grazing_boundaries_are_intersecting() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuPlaneAabbClassify::new(&ctx);
    // Integer coordinates make s and r exactly representable, so s - r and
    // s + r evaluate to exactly 0.0 on both CPU and GPU regardless of any fused
    // multiply-add; the graze therefore lands on Intersecting deterministically
    // at the Positive/Intersecting and Negative/Intersecting flip boundaries.
    let graze_positive = PlaneAabbClassifyQuery::new(plane_px(-2.0), unit_box_at([3.0, 0.0, 0.0]));
    let graze_negative = PlaneAabbClassifyQuery::new(plane_px(2.0), unit_box_at([-3.0, 0.0, 0.0]));
    let got = gpu.eval(&ctx, &[graze_positive, graze_negative]);
    assert_eq!(
        got[0].side,
        Side::Intersecting,
        "s - r == 0 grazes, not positive"
    );
    assert_eq!(
        got[1].side,
        Side::Intersecting,
        "s + r == 0 grazes, not negative"
    );
    check(&ctx, &gpu, &[graze_positive, graze_negative]);
}

#[test]
fn center_exactly_on_plane_is_intersecting() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuPlaneAabbClassify::new(&ctx);
    // Box center exactly on the plane: s = 0, r = 1, both bands straddle.
    let q = PlaneAabbClassifyQuery::new(plane_px(0.0), unit_box_at([0.0, 7.0, -3.0]));
    let got = gpu.eval(&ctx, &[q]);
    assert_eq!(
        got[0].side,
        Side::Intersecting,
        "center on the plane straddles"
    );
    assert!(
        close(got[0].signed_distance, 0.0),
        "signed distance is zero"
    );
    check(&ctx, &gpu, &[q]);
}

#[test]
fn degenerate_thin_and_point_boxes() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuPlaneAabbClassify::new(&ctx);
    // Zero x-extent slabs and zero-extent points: r drops the collapsed lanes,
    // so the side is governed by the center distance alone.
    let thin_straddle =
        PlaneAabbClassifyQuery::new(plane_px(0.0), Aabb::new([0.0, 0.0, 0.0], [0.0, 5.0, 5.0]));
    let thin_positive =
        PlaneAabbClassifyQuery::new(plane_px(0.0), Aabb::new([2.0, 0.0, 0.0], [0.0, 5.0, 5.0]));
    let point_on_plane =
        PlaneAabbClassifyQuery::new(plane_px(0.0), Aabb::new([0.0, 4.0, -2.0], [0.0, 0.0, 0.0]));
    let point_off_plane =
        PlaneAabbClassifyQuery::new(plane_px(0.0), Aabb::new([2.0, 0.0, 0.0], [0.0, 0.0, 0.0]));
    let queries = [
        thin_straddle,
        thin_positive,
        point_on_plane,
        point_off_plane,
    ];
    let got = gpu.eval(&ctx, &queries);
    assert_eq!(
        got[0].side,
        Side::Intersecting,
        "thin slab across the plane straddles"
    );
    assert_eq!(got[1].side, Side::Positive, "thin slab on +x is positive");
    assert_eq!(
        got[2].side,
        Side::Intersecting,
        "point on the plane straddles"
    );
    assert_eq!(
        got[3].side,
        Side::Positive,
        "point off the plane is positive"
    );
    check(&ctx, &gpu, &queries);
}

#[test]
fn degenerate_zero_normal() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuPlaneAabbClassify::new(&ctx);
    // A zero normal collapses r to 0 and s to d, so the side is decided by the
    // sign of d alone (well clear of the band for |d| = 4).
    let positive = PlaneAabbClassifyQuery::new(
        Plane::new([0.0, 0.0, 0.0], 4.0),
        unit_box_at([10.0, -2.0, 3.0]),
    );
    let negative = PlaneAabbClassifyQuery::new(
        Plane::new([0.0, 0.0, 0.0], -4.0),
        unit_box_at([10.0, -2.0, 3.0]),
    );
    let got = gpu.eval(&ctx, &[positive, negative]);
    assert_eq!(
        got[0].side,
        Side::Positive,
        "zero normal with d > 0 is positive"
    );
    assert_eq!(
        got[1].side,
        Side::Negative,
        "zero normal with d < 0 is negative"
    );
    check(&ctx, &gpu, &[positive, negative]);
}

#[test]
fn unnormalized_normal_matches_scaled() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuPlaneAabbClassify::new(&ctx);
    // Scaling the normal and d by a common positive factor must not change the
    // side; the base and scaled queries must agree with the reference and with
    // each other for a spread of representative boxes.
    let base = Plane::new([1.0, 0.0, 0.0], -2.0);
    let scaled = Plane::new([3.0, 0.0, 0.0], -6.0);
    let centers = [[5.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [4.0, 0.0, 0.0]];
    let mut queries = Vec::new();
    for c in centers {
        queries.push(PlaneAabbClassifyQuery::new(base, unit_box_at(c)));
        queries.push(PlaneAabbClassifyQuery::new(scaled, unit_box_at(c)));
    }
    let got = gpu.eval(&ctx, &queries);
    for pair in got.chunks_exact(2) {
        assert_eq!(
            pair[0].side, pair[1].side,
            "scaling the plane keeps the side"
        );
    }
    check(&ctx, &gpu, &queries);
}

#[test]
fn from_min_max_matches_center_half() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuPlaneAabbClassify::new(&ctx);
    // A box built from min/max corners must classify like the equivalent
    // center/half box; both sit well clear of the band on the +x side.
    let plane = plane_px(-1.0);
    let via_min_max = PlaneAabbClassifyQuery::new(
        plane,
        Aabb::from_min_max([2.0, -1.0, -1.0], [4.0, 1.0, 1.0]),
    );
    let via_center =
        PlaneAabbClassifyQuery::new(plane, Aabb::new([3.0, 0.0, 0.0], [1.0, 1.0, 1.0]));
    let got = gpu.eval(&ctx, &[via_min_max, via_center]);
    assert_eq!(got[0].side, Side::Positive, "min/max box is positive");
    assert_eq!(got[0].side, got[1].side, "min/max matches center/half");
    check(&ctx, &gpu, &[via_min_max, via_center]);
}

#[test]
fn oblique_normal_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuPlaneAabbClassify::new(&ctx);
    // A fully oblique normal folds all three lanes into both s and r; the
    // chosen boxes stay clear of the band so the side is stable.
    let plane = Plane::new([1.0, 2.0, 2.0], -3.0);
    let queries = [
        PlaneAabbClassifyQuery::new(plane, unit_box_at([4.0, 4.0, 4.0])),
        PlaneAabbClassifyQuery::new(plane, unit_box_at([-4.0, -4.0, -4.0])),
        PlaneAabbClassifyQuery::new(plane, unit_box_at([0.0, 0.0, 0.0])),
    ];
    let got = gpu.eval(&ctx, &queries);
    assert_eq!(got[0].side, Side::Positive, "far along +normal is positive");
    assert_eq!(got[1].side, Side::Negative, "far along -normal is negative");
    check(&ctx, &gpu, &queries);
}

#[test]
fn signed_distance_and_radius_values() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuPlaneAabbClassify::new(&ctx);
    // Direct value pins: s = 1*1 + 2*1 + 3*1 + 4 = 10, r = |1|+|2|+|3| = 6 for a
    // unit box; and an L1-weighted radius for a non-uniform box.
    let value_box = PlaneAabbClassifyQuery::new(
        Plane::new([1.0, 2.0, 3.0], 4.0),
        unit_box_at([1.0, 1.0, 1.0]),
    );
    let radius_box = PlaneAabbClassifyQuery::new(
        Plane::new([1.0, -2.0, 0.5], 0.0),
        Aabb::new([0.0, 0.0, 0.0], [2.0, 3.0, 4.0]),
    );
    let got = gpu.eval(&ctx, &[value_box, radius_box]);
    assert!(close(got[0].signed_distance, 10.0), "signed distance is 10");
    assert!(close(got[0].projected_radius, 6.0), "projected radius is 6");
    // r = |1|*2 + |-2|*3 + |0.5|*4 = 2 + 6 + 2 = 10.
    assert!(close(got[1].projected_radius, 10.0), "L1 radius is 10");
    check(&ctx, &gpu, &[value_box, radius_box]);
}

#[test]
fn random_batch_clear_of_the_band_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuPlaneAabbClassify::new(&ctx);
    let mut state = 0x51ed_270b_f00d_c0de_u64;
    // Generate random planes and boxes, rejecting any sample whose decision
    // terms land within a 1e-2 guard of the band. That keeps the side
    // deterministic under a legal fused multiply-add while still sweeping
    // oblique normals, off-origin centers and non-uniform half-extents.
    let guard = 1.0e-2_f32;
    let mut queries = Vec::new();
    while queries.len() < 256 {
        let plane = Plane::new(
            [
                signed(&mut state, 2.0),
                signed(&mut state, 2.0),
                signed(&mut state, 2.0),
            ],
            signed(&mut state, 3.0),
        );
        let aabb = Aabb::new(
            [
                signed(&mut state, 5.0),
                signed(&mut state, 5.0),
                signed(&mut state, 5.0),
            ],
            [
                lcg(&mut state) * 2.0,
                lcg(&mut state) * 2.0,
                lcg(&mut state) * 2.0,
            ],
        );
        let s = signed_distance_center(plane, aabb);
        let r = projected_radius(plane.normal, aabb.half);
        if (s - r).abs() <= guard || (s + r).abs() <= guard {
            continue;
        }
        queries.push(PlaneAabbClassifyQuery::new(plane, aabb));
    }
    check(&ctx, &gpu, &queries);
}
