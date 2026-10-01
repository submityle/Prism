//! Real-device **bit-exact** parity for the isolated strand-to-LSS-segments
//! twin: [`GpuHairRtCurveSegments`] must reproduce the `CPU` golden
//! [`reference_strand_to_lss`](prism_hair_gpu::rt_curve_segments::reference_strand_to_lss)
//! (which forwards to
//! [`strand_to_lss`](prism_render_architecture::hair::rt_curve::strand_to_lss))
//! for a strand polyline, splitting it into one `Linear Swept Spheres` (LSS)
//! segment per adjacent vertex pair.
//!
//! # Parity criterion
//!
//! The kernel does no arithmetic beyond the per-component sanitiser — it copies
//! each sanitised position and radius into an endpoint pair — so there is **no**
//! multiply the compiler could contract into an fma. The twin is therefore
//! *bit-exact* with the scalar reference, so every endpoint component and radius
//! is compared by its raw bit pattern ([`f32::to_bits`]) rather than a tolerance.
//!
//! The suite drives a regular multi-vertex strand, a non-finite position /
//! negative / non-finite radius sanitise case, a single-vertex strand (no
//! segments), the empty no-op, and a large multi-workgroup strand that crosses
//! the 64-wide dispatch boundary.
//!
//! The tests skip (with a printed notice) when the host has no `wgpu` adapter,
//! so the suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device. The kernel is portable core-`WGSL`,
//! so it needs no optional device feature.
//!
//! Provenance: hardware ray-traced curve / LSS strand primitive `BLAS` contract
//! (`RTX` `DXR` / `OptiX` LSS) plus `wgpu` compute dispatch; no Unreal Engine
//! source or derived code.

use prism_hair_gpu::rt_curve_segments::{reference_strand_to_lss, GpuHairRtCurveSegments};
use prism_hair_gpu::GpuContext;
use prism_render_architecture::hair::rt_curve::{CurveVertex, LssSegment};

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

/// Dispatches one strand through the device twin.
fn run(ctx: &GpuContext, vertices: &[CurveVertex]) -> Vec<LssSegment> {
    GpuHairRtCurveSegments::new(ctx).eval(ctx, vertices)
}

/// Asserts two segments agree endpoint-for-endpoint by raw bit pattern.
fn assert_segment_exact(i: usize, got: LssSegment, want: LssSegment) {
    for axis in 0..3 {
        assert_eq!(
            got.a[axis].to_bits(),
            want.a[axis].to_bits(),
            "segment {i} a[{axis}] bits differ"
        );
        assert_eq!(
            got.b[axis].to_bits(),
            want.b[axis].to_bits(),
            "segment {i} b[{axis}] bits differ"
        );
    }
    assert_eq!(
        got.radius_a.to_bits(),
        want.radius_a.to_bits(),
        "segment {i} radius_a bits differ"
    );
    assert_eq!(
        got.radius_b.to_bits(),
        want.radius_b.to_bits(),
        "segment {i} radius_b bits differ"
    );
}

/// Asserts a whole strand split matches the `CPU` golden **bit-for-bit**.
fn assert_batch_exact(got: &[LssSegment], vertices: &[CurveVertex]) {
    let want = reference_strand_to_lss(vertices);
    assert_eq!(
        got.len(),
        want.len(),
        "segment count (got {}, want {})",
        got.len(),
        want.len()
    );
    for (i, (&g, &w)) in got.iter().zip(want.iter()).enumerate() {
        assert_segment_exact(i, g, w);
    }
}

#[test]
fn regular_strand_split_is_bit_exact() {
    let Some(ctx) = context_or_skip("regular_strand_split_is_bit_exact") else {
        return;
    };
    let vertices = [
        CurveVertex::new([0.0, 0.0, 0.0], 1.0),
        CurveVertex::new([1.0, 2.0, 3.0], 0.75),
        CurveVertex::new([4.0, -1.0, 2.0], 0.5),
        CurveVertex::new([5.0, 5.0, 5.0], 0.25),
    ];
    let got = run(&ctx, &vertices);
    assert_eq!(got.len(), 3, "4 vertices yield 3 segments");
    assert_batch_exact(&got, &vertices);
    // Segment 0 joins vertex 0 and vertex 1 with their radii.
    assert_eq!(got[0].a[0].to_bits(), 0.0f32.to_bits(), "seg0 a.x");
    assert_eq!(got[0].b[1].to_bits(), 2.0f32.to_bits(), "seg0 b.y");
    assert_eq!(got[0].radius_a.to_bits(), 1.0f32.to_bits(), "seg0 radius_a");
    assert_eq!(
        got[0].radius_b.to_bits(),
        0.75f32.to_bits(),
        "seg0 radius_b"
    );
}

#[test]
fn nonfinite_and_negative_sanitise_to_zero() {
    let Some(ctx) = context_or_skip("nonfinite_and_negative_sanitise_to_zero") else {
        return;
    };
    // Non-finite coordinates and negative/non-finite radii collapse to 0.
    let vertices = [
        CurveVertex::new([f32::NAN, 1.0, f32::INFINITY], -2.0),
        CurveVertex::new([2.0, f32::NEG_INFINITY, 3.0], f32::NAN),
        CurveVertex::new([4.0, 5.0, 6.0], 0.5),
    ];
    let got = run(&ctx, &vertices);
    assert_batch_exact(&got, &vertices);

    // Segment 0's endpoint a comes from the sanitised first vertex: NaN/Inf
    // coordinates and the negative radius all collapse to 0.
    let s = got[0];
    assert_eq!(s.a[0].to_bits(), 0.0f32.to_bits(), "seg0 a.x NaN -> 0");
    assert_eq!(s.a[1].to_bits(), 1.0f32.to_bits(), "seg0 a.y kept");
    assert_eq!(s.a[2].to_bits(), 0.0f32.to_bits(), "seg0 a.z Inf -> 0");
    assert_eq!(s.radius_a.to_bits(), 0.0f32.to_bits(), "seg0 radius_a -> 0");
    assert_eq!(
        s.radius_b.to_bits(),
        0.0f32.to_bits(),
        "seg0 radius_b NaN -> 0"
    );
}

#[test]
fn single_vertex_yields_no_segments() {
    let Some(ctx) = context_or_skip("single_vertex_yields_no_segments") else {
        return;
    };
    let vertices = [CurveVertex::new([1.0, 2.0, 3.0], 0.5)];
    let got = run(&ctx, &vertices);
    assert!(got.is_empty(), "a single vertex yields no segments");
}

#[test]
fn empty_strand_is_a_no_op() {
    let Some(ctx) = context_or_skip("empty_strand_is_a_no_op") else {
        return;
    };
    let got = run(&ctx, &[]);
    assert!(got.is_empty(), "an empty strand yields no segments");
}

#[test]
fn large_strand_crosses_workgroup_boundary() {
    let Some(ctx) = context_or_skip("large_strand_crosses_workgroup_boundary") else {
        return;
    };
    // 201 vertices -> 200 segments > 3 full 64-wide workgroups: every segment
    // index must map to its own vertex pair independent of the dispatch tiling.
    let mut vertices = Vec::with_capacity(201);
    for i in 0..201u32 {
        let f = i as f32;
        vertices.push(CurveVertex::new([f, -f, f * 0.25], 0.1 + f * 0.01));
    }
    let got = run(&ctx, &vertices);
    assert_eq!(got.len(), 200, "201 vertices yield 200 segments");
    assert_batch_exact(&got, &vertices);
}
