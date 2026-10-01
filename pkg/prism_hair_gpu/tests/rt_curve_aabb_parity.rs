//! Real-device **bit-exact** parity for the isolated LSS per-segment bounding
//! box twin: [`GpuHairRtCurveAabb`] must reproduce the `CPU` golden
//! [`reference_segment_aabb`](prism_hair_gpu::rt_curve_aabb::reference_segment_aabb)
//! (which forwards to
//! [`lss_segment_aabb`](prism_render_architecture::hair::rt_curve::lss_segment_aabb))
//! for a batch of `Linear Swept Spheres` (LSS) segments, computing each
//! segment's conservative `axis-aligned` bounding box (`AABB`) independently.
//!
//! # Parity criterion
//!
//! An endpoint box is the centre plus or minus its radius on each axis, and the
//! segment box is the per-axis `min`/`max` union of the two endpoint boxes —
//! only `min`, `max` and single add/sub, with **no** multiply the compiler
//! could contract into an fma. The twin is therefore *bit-exact* with the
//! scalar reference, so every corner component is compared by its raw bit
//! pattern ([`f32::to_bits`]) rather than a tolerance.
//!
//! The suite drives a regular segment, asymmetric radii, a sanitised
//! negative/non-finite radius, sanitised non-finite positions, a single-point
//! sphere (`a == b`), the empty no-op batch, and a large multi-workgroup batch
//! that crosses the 64-wide dispatch boundary. It also asserts each returned box
//! actually contains both swept endpoints via [`Aabb::contains_point`].
//!
//! The tests skip (with a printed notice) when the host has no `wgpu` adapter,
//! so the suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device. The kernel is portable core-`WGSL`,
//! so it needs no optional device feature.
//!
//! Provenance: hardware ray-traced curve / LSS strand primitive `BLAS` contract
//! (`RTX` `DXR` / `OptiX` LSS) plus `wgpu` compute dispatch; no Unreal Engine
//! source or derived code.

use prism_hair_gpu::rt_curve_aabb::{reference_segment_aabb, GpuHairRtCurveAabb};
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
fn run(ctx: &GpuContext, segments: &[LssSegment]) -> Vec<Aabb> {
    GpuHairRtCurveAabb::new(ctx).eval(ctx, segments)
}

/// Asserts one box matches the golden corner-for-corner by raw bit pattern.
fn assert_box_exact(i: usize, got: Aabb, seg: &LssSegment) {
    let want = reference_segment_aabb(seg);
    for axis in 0..3 {
        let gb = got.min[axis].to_bits();
        let wb = want.min[axis].to_bits();
        assert_eq!(
            gb, wb,
            "segment {i} min[{axis}]: device bits {gb:#010x} must equal golden bits {wb:#010x}"
        );
        let gb = got.max[axis].to_bits();
        let wb = want.max[axis].to_bits();
        assert_eq!(
            gb, wb,
            "segment {i} max[{axis}]: device bits {gb:#010x} must equal golden bits {wb:#010x}"
        );
    }
}

/// Asserts a whole batch matches the `CPU` golden **bit-for-bit**.
fn assert_batch_exact(got: &[Aabb], segments: &[LssSegment]) {
    assert_eq!(
        got.len(),
        segments.len(),
        "one box per segment (got {}, want {})",
        got.len(),
        segments.len()
    );
    for (i, (&g, s)) in got.iter().zip(segments.iter()).enumerate() {
        assert_box_exact(i, g, s);
    }
}

#[test]
fn regular_segment_box_is_bit_exact() {
    let Some(ctx) = context_or_skip("regular_segment_box_is_bit_exact") else {
        return;
    };
    let segments = [LssSegment::new([-1.0, 0.0, 0.0], [5.0, 0.0, 0.0], 1.0, 2.0)];
    let got = run(&ctx, &segments);
    assert_batch_exact(&got, &segments);

    // The box must span both swept spheres: x in [-2, 7], y/z in [-2, 2].
    let bb = got[0];
    assert_eq!(bb.min[0].to_bits(), (-2.0f32).to_bits(), "min x");
    assert_eq!(bb.max[0].to_bits(), 7.0f32.to_bits(), "max x");
    assert!(bb.is_valid(), "a regular segment yields a valid box");
    // Both endpoints (with radii) stay enclosed.
    assert!(bb.contains_point([-2.0, 0.0, 0.0]), "sphere a surface");
    assert!(bb.contains_point([7.0, 0.0, 0.0]), "sphere b surface");
    assert!(bb.contains_point([5.0, 2.0, -2.0]), "sphere b corner");
}

#[test]
fn asymmetric_radii_box_is_bit_exact() {
    let Some(ctx) = context_or_skip("asymmetric_radii_box_is_bit_exact") else {
        return;
    };
    let segments = [
        LssSegment::new([0.0, 0.0, 0.0], [0.0, 4.0, 0.0], 0.25, 1.5),
        LssSegment::new([3.0, -2.0, 1.0], [-1.0, 2.0, -3.0], 0.5, 0.5),
    ];
    let got = run(&ctx, &segments);
    assert_batch_exact(&got, &segments);
    for (bb, s) in got.iter().zip(segments.iter()) {
        let clean = s.sanitized();
        assert!(bb.contains_point(clean.a), "box contains center a");
        assert!(bb.contains_point(clean.b), "box contains center b");
    }
}

#[test]
fn negative_and_nonfinite_radius_sanitise_to_zero() {
    let Some(ctx) = context_or_skip("negative_and_nonfinite_radius_sanitise_to_zero") else {
        return;
    };
    // A negative radius and a non-finite radius both collapse to 0, so the
    // affected endpoint contributes only its centre point.
    let segments = [
        LssSegment::new([1.0, 1.0, 1.0], [2.0, 2.0, 2.0], -5.0, 1.0),
        LssSegment::new([0.0, 0.0, 0.0], [3.0, 0.0, 0.0], f32::INFINITY, f32::NAN),
    ];
    let got = run(&ctx, &segments);
    assert_batch_exact(&got, &segments);

    // Segment 1: both radii sanitise to 0, so the box is exactly the two
    // centres' span [0,3] x [0,0] x [0,0].
    let bb = got[1];
    assert_eq!(bb.min[0].to_bits(), 0.0f32.to_bits(), "seg1 min x");
    assert_eq!(bb.max[0].to_bits(), 3.0f32.to_bits(), "seg1 max x");
    assert_eq!(bb.min[1].to_bits(), 0.0f32.to_bits(), "seg1 min y");
    assert_eq!(bb.max[1].to_bits(), 0.0f32.to_bits(), "seg1 max y");
}

#[test]
fn nonfinite_positions_sanitise_to_zero() {
    let Some(ctx) = context_or_skip("nonfinite_positions_sanitise_to_zero") else {
        return;
    };
    // Non-finite position components collapse to 0 before the box is built.
    let segments = [LssSegment::new(
        [f32::NAN, 2.0, f32::INFINITY],
        [1.0, f32::NEG_INFINITY, 3.0],
        0.5,
        0.5,
    )];
    let got = run(&ctx, &segments);
    assert_batch_exact(&got, &segments);
    assert!(got[0].is_valid(), "sanitised box stays valid");
}

#[test]
fn single_point_sphere_box_is_bit_exact() {
    let Some(ctx) = context_or_skip("single_point_sphere_box_is_bit_exact") else {
        return;
    };
    // a == b with equal radii: the box is a single centred cube of half-extent r.
    let segments = [LssSegment::new(
        [2.0, -3.0, 4.0],
        [2.0, -3.0, 4.0],
        1.0,
        1.0,
    )];
    let got = run(&ctx, &segments);
    assert_batch_exact(&got, &segments);
    let bb = got[0];
    assert_eq!(bb.min[0].to_bits(), 1.0f32.to_bits(), "min x");
    assert_eq!(bb.max[0].to_bits(), 3.0f32.to_bits(), "max x");
    assert!(bb.contains_point([2.0, -3.0, 4.0]), "centre enclosed");
}

#[test]
fn empty_batch_is_a_no_op() {
    let Some(ctx) = context_or_skip("empty_batch_is_a_no_op") else {
        return;
    };
    let got = run(&ctx, &[]);
    assert!(got.is_empty(), "an empty batch yields no boxes");
}

#[test]
fn large_batch_crosses_workgroup_boundary() {
    let Some(ctx) = context_or_skip("large_batch_crosses_workgroup_boundary") else {
        return;
    };
    // 200 segments > 3 full 64-wide workgroups: every segment index must map to
    // its own box independent of the dispatch tiling.
    let mut segments = Vec::with_capacity(200);
    for i in 0..200u32 {
        let f = i as f32;
        segments.push(LssSegment::new(
            [f, -f, f * 0.5],
            [f + 2.0, f - 1.0, -f],
            0.1 + f * 0.01,
            0.2 + f * 0.02,
        ));
    }
    let got = run(&ctx, &segments);
    assert_batch_exact(&got, &segments);
    for (bb, s) in got.iter().zip(segments.iter()) {
        let clean = s.sanitized();
        assert!(bb.contains_point(clean.a), "box contains center a");
        assert!(bb.contains_point(clean.b), "box contains center b");
    }
}
