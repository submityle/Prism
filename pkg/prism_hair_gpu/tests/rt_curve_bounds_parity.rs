//! Real-device **bit-exact** parity for the LSS batch bounding-box union twin:
//! [`GpuHairRtCurveBounds`] must reproduce the `CPU` golden
//! [`reference_segments_aabb`](prism_hair_gpu::rt_curve_bounds::reference_segments_aabb)
//! (which forwards to
//! [`segments_aabb`](prism_render_architecture::hair::rt_curve::segments_aabb))
//! for a batch of `Linear Swept Spheres` (LSS) segments, folding them into the
//! single conservative `axis-aligned` bounding box (`AABB`) that encloses them
//! all.
//!
//! # Parity criterion
//!
//! The union of a set of boxes is a per-axis `min` of the mins and `max` of the
//! maxes. `min`/`max` select an existing operand with no arithmetic and are
//! both commutative *and* associative, so the device tree reduction's pairwise
//! fold order yields the exact same bits as the golden's left-to-right
//! `Aabb::union` walk — the twin is *bit-exact* with the scalar reference, so
//! every corner component is compared by its raw bit pattern
//! ([`f32::to_bits`]) rather than a tolerance.
//!
//! The suite drives a regular multi-segment batch, a single segment (a
//! one-element fold), asymmetric radii, a sanitised negative/non-finite radius,
//! sanitised non-finite positions, the empty no-op batch (which maps to
//! [`Aabb::EMPTY`]), and a large batch that exceeds the `256`-wide workgroup so
//! the grid-stride load and tree fold are exercised. It also asserts the
//! returned box actually encloses every swept endpoint via
//! [`Aabb::contains_point`].
//!
//! The tests skip (with a printed notice) when the host has no `wgpu` adapter,
//! so the suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device. The kernel is portable core-`WGSL`,
//! so it needs no optional device feature.
//!
//! Provenance: hardware ray-traced curve / LSS strand primitive `BLAS` contract
//! (`RTX` `DXR` / `OptiX` LSS) plus a single-workgroup shared-memory tree
//! reduction and `wgpu` compute dispatch; no Unreal Engine source or derived
//! code.

use prism_hair_gpu::rt_curve_bounds::{reference_segments_aabb, GpuHairRtCurveBounds};
use prism_hair_gpu::GpuContext;
use prism_render_architecture::hair::rt_curve::{Aabb, LssSegment};

/// Acquires a headless context, or `None` (with a skip notice) when the host has
/// no `wgpu` adapter so the suite stays green off-device.
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn context_or_skip(label: &str) -> Option<GpuContext> {
    match GpuContext::try_headless() {
        Some(ctx) => Some(ctx),
        None => {
            eprintln!("skipping {label}: no wgpu adapter on this host");
            None
        }
    }
}

/// Dispatches one batch through the device twin.
fn run(ctx: &GpuContext, segments: &[LssSegment]) -> Aabb {
    GpuHairRtCurveBounds::new(ctx).eval(ctx, segments)
}

/// Asserts the device union matches the `CPU` golden corner-for-corner by raw
/// bit pattern.
fn assert_union_exact(got: Aabb, segments: &[LssSegment]) {
    let want = reference_segments_aabb(segments);
    for axis in 0..3 {
        let gb = got.min[axis].to_bits();
        let wb = want.min[axis].to_bits();
        assert_eq!(
            gb, wb,
            "union min[{axis}]: device bits {gb:#010x} must equal golden bits {wb:#010x}"
        );
        let gb = got.max[axis].to_bits();
        let wb = want.max[axis].to_bits();
        assert_eq!(
            gb, wb,
            "union max[{axis}]: device bits {gb:#010x} must equal golden bits {wb:#010x}"
        );
    }
}

/// Asserts `bb` encloses both sanitised endpoints of every segment.
fn assert_encloses_all(bb: Aabb, segments: &[LssSegment]) {
    for s in segments {
        let clean = s.sanitized();
        assert!(bb.contains_point(clean.a), "union contains centre a");
        assert!(bb.contains_point(clean.b), "union contains centre b");
    }
}

#[test]
fn regular_batch_union_is_bit_exact() {
    let Some(ctx) = context_or_skip("regular_batch_union_is_bit_exact") else {
        return;
    };
    let segments = [
        LssSegment::new([-1.0, 0.0, 0.0], [5.0, 0.0, 0.0], 1.0, 2.0),
        LssSegment::new([0.0, 3.0, 0.0], [0.0, 3.0, 5.0], 0.5, 0.5),
        LssSegment::new([2.0, -4.0, 1.0], [2.0, -4.0, 1.0], 0.25, 0.25),
    ];
    let got = run(&ctx, &segments);
    assert_union_exact(got, &segments);
    assert!(got.is_valid(), "a non-empty batch yields a valid box");
    assert_encloses_all(got, &segments);
    // x spans sphere a of seg0 (-2) to sphere b of seg0 (7); y spans seg2
    // (-4.25) to seg0/seg1; z spans seg1's far sphere (5.5).
    assert_eq!(got.min[0].to_bits(), (-2.0f32).to_bits(), "min x");
    assert_eq!(got.max[0].to_bits(), 7.0f32.to_bits(), "max x");
    assert_eq!(got.max[2].to_bits(), 5.5f32.to_bits(), "max z");
}

#[test]
fn single_segment_fold_is_bit_exact() {
    let Some(ctx) = context_or_skip("single_segment_fold_is_bit_exact") else {
        return;
    };
    // A one-element fold must equal that segment's own box.
    let segments = [LssSegment::new(
        [3.0, -2.0, 1.0],
        [-1.0, 2.0, -3.0],
        0.5,
        1.0,
    )];
    let got = run(&ctx, &segments);
    assert_union_exact(got, &segments);
    assert!(got.is_valid(), "single-segment union is valid");
    assert_encloses_all(got, &segments);
}

#[test]
fn negative_and_nonfinite_radius_sanitise_to_zero() {
    let Some(ctx) = context_or_skip("negative_and_nonfinite_radius_sanitise_to_zero") else {
        return;
    };
    // Negative and non-finite radii collapse to 0, so those endpoints
    // contribute only their centre points to the union.
    let segments = [
        LssSegment::new([1.0, 1.0, 1.0], [2.0, 2.0, 2.0], -5.0, 1.0),
        LssSegment::new([0.0, 0.0, 0.0], [3.0, 0.0, 0.0], f32::INFINITY, f32::NAN),
    ];
    let got = run(&ctx, &segments);
    assert_union_exact(got, &segments);
    assert!(got.is_valid(), "sanitised union stays valid");
}

#[test]
fn nonfinite_positions_sanitise_to_zero() {
    let Some(ctx) = context_or_skip("nonfinite_positions_sanitise_to_zero") else {
        return;
    };
    // Non-finite position components collapse to 0 before the box is built.
    let segments = [
        LssSegment::new(
            [f32::NAN, 2.0, f32::INFINITY],
            [1.0, f32::NEG_INFINITY, 3.0],
            0.5,
            0.5,
        ),
        LssSegment::new([4.0, 4.0, 4.0], [5.0, 5.0, 5.0], 0.5, 0.5),
    ];
    let got = run(&ctx, &segments);
    assert_union_exact(got, &segments);
    assert!(got.is_valid(), "sanitised union stays valid and finite");
    assert!(
        got.min[0].is_finite() && got.max[1].is_finite(),
        "finite corners"
    );
}

#[test]
fn empty_batch_is_the_empty_box() {
    let Some(ctx) = context_or_skip("empty_batch_is_the_empty_box") else {
        return;
    };
    let got = run(&ctx, &[]);
    // The golden maps an empty set to Aabb::EMPTY (min = +inf, max = -inf).
    assert_union_exact(got, &[]);
    assert!(!got.is_valid(), "an empty batch is the empty / invalid box");
}

#[test]
fn large_batch_exceeds_workgroup_width() {
    let Some(ctx) = context_or_skip("large_batch_exceeds_workgroup_width") else {
        return;
    };
    // 500 segments > the 256-wide workgroup, so the grid-stride load folds
    // multiple segments per invocation before the tree reduction runs.
    let mut segments = Vec::with_capacity(500);
    for i in 0..500u32 {
        let f = i as f32;
        segments.push(LssSegment::new(
            [f, -f, f * 0.5],
            [f + 2.0, f - 1.0, -f],
            0.1 + f * 0.01,
            0.2 + f * 0.02,
        ));
    }
    let got = run(&ctx, &segments);
    assert_union_exact(got, &segments);
    assert!(got.is_valid(), "large-batch union is valid");
    assert_encloses_all(got, &segments);
}
