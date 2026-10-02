//! Real-device parity for the per-edge strand rest-length twin:
//! [`GpuHairStrandRestLengths`] must reproduce the `CPU` golden
//! [`strand_rest_lengths`](prism_render_architecture::hair::groom_import::strand_rest_lengths)
//! for a flattened batch of strand polylines, including the `n - 1` edge count,
//! the short-strand empty case, and a numerically zero-length (coincident)
//! segment.
//!
//! # Parity criterion
//!
//! The edge vector is an exact subtraction and the length is `sqrt(dot(d, d))`,
//! so the only `CPU` vs `GPU` divergence is legal fused-multiply-add
//! contraction in the squared length and the `sqrt`. Each length is asserted to
//! within `abs_diff < 1e-4` or `rel_diff < 1e-3` - tight enough to fail a
//! genuinely wrong port (a swapped endpoint, a dropped edge, a summed scalar),
//! loose enough to admit fma contraction. The per-strand edge counts are
//! asserted exactly, so a kernel that emitted the wrong number of lengths could
//! not pass.
//!
//! The suite drives a smooth multi-point strand, a two-point strand (one edge),
//! a single-point strand (no edges), a strand with a coincident-point
//! zero-length segment (a `0` rest length), a mixed batch with an empty and a
//! single-point strand, the empty batch, and a batch flattening to more than 64
//! edges that crosses the one-thread-per-edge dispatch boundary.
//!
//! The tests skip (with a printed notice) when the host has no `wgpu` adapter,
//! so the suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device. The kernel is portable core-`WGSL`,
//! so it needs no optional device feature.
//!
//! Provenance: standard polyline per-segment lengths plus a `wgpu` compute
//! dispatch; no Unreal Engine source or derived code.

use prism_hair_gpu::strand_rest_lengths::{
    reference_strand_rest_lengths, GpuHairStrandRestLengths,
};
use prism_hair_gpu::GpuContext;

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

/// True when `got` matches `want` within the rest-length tolerance.
fn close(got: f32, want: f32) -> bool {
    let diff = (got - want).abs();
    diff < 1.0e-4 || diff <= 1.0e-3 * want.abs()
}

/// Dispatches one batch through the device twin.
fn run(ctx: &GpuContext, strands: &[&[[f32; 3]]]) -> Vec<Vec<f32>> {
    GpuHairStrandRestLengths::new(ctx).eval(ctx, strands)
}

/// Asserts the device result for `strands` matches the golden per strand, with
/// exact per-strand edge counts.
fn assert_batch_parity(ctx: &GpuContext, strands: &[&[[f32; 3]]]) {
    let got = run(ctx, strands);
    assert_eq!(got.len(), strands.len(), "strand count mismatch");
    for (s, strand) in strands.iter().enumerate() {
        let want = reference_strand_rest_lengths(strand);
        assert_eq!(got[s].len(), want.len(), "strand {s}: edge count mismatch");
        for (i, (g, w)) in got[s].iter().zip(want.iter()).enumerate() {
            assert!(close(*g, *w), "strand {s} edge {i}: gpu {g} vs cpu {w}");
        }
    }
}

#[test]
fn gpu_smooth_curve_matches_cpu_golden() {
    let Some(ctx) = context_or_skip("gpu_smooth_curve_matches_cpu_golden") else {
        return;
    };
    // A four-point polyline: three edges with distinct, well-conditioned
    // lengths.
    let a: &[[f32; 3]] = &[
        [0.0, 0.0, 0.0],
        [3.0, 0.0, 0.0],
        [3.0, 4.0, 0.0],
        [3.0, 4.0, 12.0],
    ];
    let got = run(&ctx, &[a]);
    assert_eq!(got[0].len(), 3);
    // 3, 4, 12 by construction (3-4-5 and 5-12-13 style edges).
    assert!(close(got[0][0], 3.0), "{:?}", got[0]);
    assert!(close(got[0][1], 4.0), "{:?}", got[0]);
    assert!(close(got[0][2], 12.0), "{:?}", got[0]);
    assert_batch_parity(&ctx, &[a]);
}

#[test]
fn gpu_two_point_strand_single_edge() {
    let Some(ctx) = context_or_skip("gpu_two_point_strand_single_edge") else {
        return;
    };
    let two: &[[f32; 3]] = &[[1.0, 1.0, 1.0], [1.0, 1.0, 6.0]];
    let got = run(&ctx, &[two]);
    assert_eq!(got[0].len(), 1);
    assert!(close(got[0][0], 5.0), "{:?}", got[0]);
    assert_batch_parity(&ctx, &[two]);
}

#[test]
fn gpu_single_point_strand_has_no_edges() {
    let Some(ctx) = context_or_skip("gpu_single_point_strand_has_no_edges") else {
        return;
    };
    let single: &[[f32; 3]] = &[[7.0, -3.0, 2.0]];
    let got = run(&ctx, &[single]);
    assert_eq!(got.len(), 1);
    assert!(got[0].is_empty(), "single-point strand must yield no edges");
}

#[test]
fn gpu_zero_length_segment_yields_zero() {
    let Some(ctx) = context_or_skip("gpu_zero_length_segment_yields_zero") else {
        return;
    };
    // Middle point coincides with the first, so edge 0 has length 0; edge 1 is a
    // real length.
    let strand: &[[f32; 3]] = &[[2.0, 2.0, 2.0], [2.0, 2.0, 2.0], [2.0, 5.0, 2.0]];
    let got = run(&ctx, &[strand]);
    assert_eq!(got[0].len(), 2);
    assert!(close(got[0][0], 0.0), "{:?}", got[0]);
    assert!(close(got[0][1], 3.0), "{:?}", got[0]);
    assert_batch_parity(&ctx, &[strand]);
}

#[test]
fn gpu_mixed_batch_with_short_strands() {
    let Some(ctx) = context_or_skip("gpu_mixed_batch_with_short_strands") else {
        return;
    };
    let a: &[[f32; 3]] = &[[0.0, 0.0, 0.0], [0.0, 0.0, 4.0], [0.0, 2.0, 4.0]];
    let empty: &[[f32; 3]] = &[];
    let single: &[[f32; 3]] = &[[9.0, 9.0, 9.0]];
    let c: &[[f32; 3]] = &[[5.0, 5.0, 5.0], [5.0, 9.0, 5.0]];
    let got = run(&ctx, &[a, empty, single, c]);
    assert_eq!(got.len(), 4);
    assert_eq!(got[0].len(), 2);
    assert!(got[1].is_empty(), "empty strand must yield no edges");
    assert!(got[2].is_empty(), "single-point strand must yield no edges");
    assert_eq!(got[3].len(), 1);
    assert_batch_parity(&ctx, &[a, empty, single, c]);
}

#[test]
fn gpu_empty_batch_yields_empty() {
    let Some(ctx) = context_or_skip("gpu_empty_batch_yields_empty") else {
        return;
    };
    let got = run(&ctx, &[]);
    assert!(got.is_empty());
    // A batch whose every strand is too short to have an edge yields matching
    // empty inner vectors without a dispatch.
    let empty_a: &[[f32; 3]] = &[];
    let single: &[[f32; 3]] = &[[1.0, 2.0, 3.0]];
    let got2 = run(&ctx, &[empty_a, single]);
    assert_eq!(got2.len(), 2);
    assert!(got2[0].is_empty() && got2[1].is_empty());
}

#[test]
fn gpu_large_batch_crosses_workgroup_boundary() {
    let Some(ctx) = context_or_skip("gpu_large_batch_crosses_workgroup_boundary") else {
        return;
    };
    // Three strands flattening to 203 points => 200 edges (> 3 * 64), so the
    // one-thread-per-edge dispatch spans several workgroups. Points are
    // generated by integer affine arithmetic (no transcendental calls) and kept
    // non-coincident so every edge length is well conditioned.
    let build = |count: usize, seed: i32| -> Vec<[f32; 3]> {
        (0..count)
            .map(|k| {
                let i = k as i32;
                let x = (seed + i) as f32;
                let y = (2 * i - seed) as f32;
                let z = (i * i % 11) as f32;
                [x, y, z]
            })
            .collect()
    };
    let s0 = build(70, 3);
    let s1 = build(64, -5);
    let s2 = build(69, 1);
    let strands: &[&[[f32; 3]]] = &[&s0, &s1, &s2];
    assert_eq!(s0.len() + s1.len() + s2.len(), 203);
    assert_batch_parity(&ctx, strands);
}
