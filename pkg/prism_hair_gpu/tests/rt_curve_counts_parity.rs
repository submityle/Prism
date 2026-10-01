//! Real-device parity for the isolated per-strand LSS segment-count twin:
//! [`GpuHairRtCurveCounts`] must reproduce the `CPU` golden
//! [`reference_lss_segment_counts`](prism_hair_gpu::rt_curve_counts::reference_lss_segment_counts)
//! (which forwards to
//! [`lss_segment_counts`](prism_render_architecture::hair::rt_curve::lss_segment_counts))
//! for a batch of strands, emitting `max(0, len - 1)` segments per strand.
//!
//! # Parity criterion
//!
//! The kernel reads only each strand's vertex count, so it is pure integer
//! arithmetic — no floating point, no rounding, no fma. The twin matches the
//! scalar reference exactly, so counts are compared with integer equality
//! ([`assert_eq!`]), not a tolerance.
//!
//! The suite drives a mix of strand lengths (including empty and single-vertex
//! strands that yield zero segments), the empty no-op batch, and a large
//! multi-workgroup batch that crosses the 64-wide dispatch boundary.
//!
//! The tests skip (with a printed notice) when the host has no `wgpu` adapter,
//! so the suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device. The kernel is portable core-`WGSL`,
//! so it needs no optional device feature.
//!
//! Provenance: hardware ray-traced curve / LSS strand primitive `BLAS` contract
//! (`RTX` `DXR` / `OptiX` LSS) plus `wgpu` compute dispatch; no Unreal Engine
//! source or derived code.

use prism_hair_gpu::rt_curve_counts::{reference_lss_segment_counts, GpuHairRtCurveCounts};
use prism_hair_gpu::GpuContext;
use prism_render_architecture::hair::rt_curve::CurveVertex;

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

/// Builds a strand of `n` placeholder vertices (positions/radii are irrelevant
/// to the count).
fn strand(n: usize) -> Vec<CurveVertex> {
    (0..n)
        .map(|i| CurveVertex::new([i as f32, 0.0, 0.0], 1.0))
        .collect()
}

/// Dispatches a batch through the device twin and asserts it matches the golden.
fn assert_batch_exact(ctx: &GpuContext, strands: &[&[CurveVertex]]) -> Vec<u32> {
    let got = GpuHairRtCurveCounts::new(ctx).eval(ctx, strands);
    let want = reference_lss_segment_counts(strands);
    assert_eq!(got.len(), want.len(), "one count per strand");
    for (i, (&g, &w)) in got.iter().zip(want.iter()).enumerate() {
        assert_eq!(g as usize, w, "strand {i} segment count");
    }
    got
}

#[test]
fn mixed_lengths_counts_are_exact() {
    let Some(ctx) = context_or_skip("mixed_lengths_counts_are_exact") else {
        return;
    };
    let s0 = strand(5);
    let s1 = strand(2);
    let s2 = strand(1);
    let s3: Vec<CurveVertex> = strand(0);
    let s4 = strand(10);
    let strands: [&[CurveVertex]; 5] = [&s0, &s1, &s2, &s3, &s4];
    let got = assert_batch_exact(&ctx, &strands);
    // n vertices -> max(0, n - 1) segments.
    assert_eq!(got, vec![4, 1, 0, 0, 9], "explicit per-strand counts");
}

#[test]
fn empty_batch_is_a_no_op() {
    let Some(ctx) = context_or_skip("empty_batch_is_a_no_op") else {
        return;
    };
    let got = GpuHairRtCurveCounts::new(&ctx).eval(&ctx, &[]);
    assert!(got.is_empty(), "an empty batch yields no counts");
}

#[test]
fn large_batch_crosses_workgroup_boundary() {
    let Some(ctx) = context_or_skip("large_batch_crosses_workgroup_boundary") else {
        return;
    };
    // 200 strands > 3 full 64-wide workgroups: every strand index must map to
    // its own count independent of the dispatch tiling.
    let owned: Vec<Vec<CurveVertex>> = (0..200usize).map(|i| strand(i % 7)).collect();
    let strands: Vec<&[CurveVertex]> = owned.iter().map(Vec::as_slice).collect();
    let got = assert_batch_exact(&ctx, &strands);
    for (i, &c) in got.iter().enumerate() {
        let n = i % 7;
        let want = if n >= 2 { (n - 1) as u32 } else { 0 };
        assert_eq!(c, want, "strand {i}");
    }
}
