//! Real-device parity for the per-control-point strand-tangent twin:
//! [`GpuHairStrandTangents`] must reproduce the `CPU` golden
//! [`strand_tangents`](prism_render_architecture::hair::frames::strand_tangents)
//! for a flattened batch of strand polylines, including the forward-vs-backward
//! difference branch, the single-point fallback, and the zero-length-segment
//! fallback the reference guards.
//!
//! # Parity criterion
//!
//! The forward/backward difference is an exact subtraction, so the only `CPU`
//! vs `GPU` divergence is legal fused-multiply-add contraction in the normalize
//! (`sqrt` + reciprocal). Each tangent component is asserted to within
//! `abs_diff < 1e-4` or `rel_diff < 1e-3` — tight enough to fail a genuinely
//! wrong port (a swapped forward/backward branch, a dropped fallback, a lost
//! normalize), loose enough to admit fma contraction. Every returned tangent is
//! also asserted to be unit length, so a degenerate all-constant kernel could
//! not pass.
//!
//! The suite drives smooth multi-point strands (forward/backward difference), a
//! single-point strand (fallback), a two-point strand, a strand with a
//! coincident-point zero-length segment (fallback), a mixed batch with an empty
//! strand, the empty batch, and a 203-point batch that crosses the 64-wide
//! dispatch boundary.
//!
//! The tests skip (with a printed notice) when the host has no `wgpu` adapter,
//! so the suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device. The kernel is portable core-`WGSL`,
//! so it needs no optional device feature.
//!
//! Provenance: standard finite-difference polyline tangents plus a `wgpu`
//! compute dispatch; no Unreal Engine source or derived code.

use prism_hair_gpu::strand_tangents::{reference_strand_tangents, GpuHairStrandTangents};
use prism_hair_gpu::GpuContext;
use prism_render_architecture::hair::interpolation::Vec3;

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

/// True when `got` matches `want` within the tangent tolerance.
fn close(got: f32, want: f32) -> bool {
    let diff = (got - want).abs();
    diff < 1.0e-4 || diff <= 1.0e-3 * want.abs()
}

/// Converts a flat `[f32; 3]` control point to the golden's `Vec3`.
fn to_vec3(p: [f32; 3]) -> Vec3 {
    Vec3::new(p[0], p[1], p[2])
}

/// Dispatches one batch through the device twin.
fn run(ctx: &GpuContext, strands: &[&[[f32; 3]]]) -> Vec<Vec<[f32; 3]>> {
    GpuHairStrandTangents::new(ctx).eval(ctx, strands)
}

/// Asserts the device result for `strands` matches the golden per strand, and
/// that every tangent is a unit vector.
fn assert_batch_parity(ctx: &GpuContext, strands: &[&[[f32; 3]]]) {
    let got = run(ctx, strands);
    assert_eq!(got.len(), strands.len(), "strand count mismatch");
    for (s, strand) in strands.iter().enumerate() {
        let points: Vec<Vec3> = strand.iter().map(|p| to_vec3(*p)).collect();
        let want = reference_strand_tangents(&points);
        assert_eq!(
            got[s].len(),
            want.len(),
            "strand {s}: tangent count mismatch"
        );
        for (i, (g, w)) in got[s].iter().zip(want.iter()).enumerate() {
            for axis in 0..3 {
                assert!(
                    close(g[axis], w[axis]),
                    "strand {s} point {i} axis {axis}: gpu {} vs cpu {}",
                    g[axis],
                    w[axis]
                );
            }
            // Every golden tangent is normalized (or the unit fallback), so the
            // device result must be unit length too.
            let len = (g[0] * g[0] + g[1] * g[1] + g[2] * g[2]).sqrt();
            assert!(
                (len - 1.0).abs() < 1.0e-3,
                "strand {s} point {i}: tangent not unit length (len {len})"
            );
        }
    }
}

#[test]
fn gpu_smooth_curves_match_cpu_golden() {
    let Some(ctx) = context_or_skip("gpu_smooth_curves_match_cpu_golden") else {
        return;
    };
    // Two polylines built from affine (non-coincident) control points, so every
    // segment is a well-conditioned forward/backward difference.
    let a: &[[f32; 3]] = &[
        [0.0, 0.0, 0.0],
        [1.0, 2.0, 0.5],
        [3.0, 3.0, 2.0],
        [6.0, 1.0, 5.0],
    ];
    let b: &[[f32; 3]] = &[[-2.0, 4.0, 1.0], [-2.0, 4.0, 3.0], [-2.0, 7.0, 3.0]];
    assert_batch_parity(&ctx, &[a, b]);
}

#[test]
fn gpu_single_point_strand_takes_fallback() {
    let Some(ctx) = context_or_skip("gpu_single_point_strand_takes_fallback") else {
        return;
    };
    let single: &[[f32; 3]] = &[[7.0, -3.0, 2.0]];
    let got = run(&ctx, &[single]);
    assert_eq!(got.len(), 1);
    assert_eq!(got[0].len(), 1);
    // Single-point strand → fixed fallback tangent (0, 1, 0).
    let t = got[0][0];
    assert!(
        close(t[0], 0.0) && close(t[1], 1.0) && close(t[2], 0.0),
        "{t:?}"
    );
    assert_batch_parity(&ctx, &[single]);
}

#[test]
fn gpu_two_point_strand_matches_golden() {
    let Some(ctx) = context_or_skip("gpu_two_point_strand_matches_golden") else {
        return;
    };
    // Both points share the same (forward == backward) difference direction.
    let two: &[[f32; 3]] = &[[1.0, 1.0, 1.0], [4.0, 1.0, 1.0]];
    assert_batch_parity(&ctx, &[two]);
}

#[test]
fn gpu_zero_length_segment_falls_back() {
    let Some(ctx) = context_or_skip("gpu_zero_length_segment_falls_back") else {
        return;
    };
    // Middle point coincides with the first, so point 0's forward difference is
    // zero and must fall back; later points still take real differences.
    let strand: &[[f32; 3]] = &[[2.0, 2.0, 2.0], [2.0, 2.0, 2.0], [2.0, 5.0, 2.0]];
    let got = run(&ctx, &[strand]);
    // Point 0 forward difference is zero → fallback (0, 1, 0).
    let t0 = got[0][0];
    assert!(
        close(t0[0], 0.0) && close(t0[1], 1.0) && close(t0[2], 0.0),
        "{t0:?}"
    );
    assert_batch_parity(&ctx, &[strand]);
}

#[test]
fn gpu_mixed_batch_with_empty_strand() {
    let Some(ctx) = context_or_skip("gpu_mixed_batch_with_empty_strand") else {
        return;
    };
    let a: &[[f32; 3]] = &[[0.0, 0.0, 0.0], [0.0, 0.0, 4.0], [0.0, 2.0, 4.0]];
    let empty: &[[f32; 3]] = &[];
    let c: &[[f32; 3]] = &[[5.0, 5.0, 5.0], [5.0, 9.0, 5.0]];
    let got = run(&ctx, &[a, empty, c]);
    assert_eq!(got.len(), 3);
    assert!(got[1].is_empty(), "empty strand must yield empty tangents");
    assert_batch_parity(&ctx, &[a, empty, c]);
}

#[test]
fn gpu_empty_batch_yields_empty() {
    let Some(ctx) = context_or_skip("gpu_empty_batch_yields_empty") else {
        return;
    };
    let got = run(&ctx, &[]);
    assert!(got.is_empty());
    // A batch whose every strand is empty yields matching empty inner vectors
    // without a dispatch.
    let empty_a: &[[f32; 3]] = &[];
    let empty_b: &[[f32; 3]] = &[];
    let got2 = run(&ctx, &[empty_a, empty_b]);
    assert_eq!(got2.len(), 2);
    assert!(got2[0].is_empty() && got2[1].is_empty());
}

#[test]
fn gpu_large_batch_crosses_workgroup_boundary() {
    let Some(ctx) = context_or_skip("gpu_large_batch_crosses_workgroup_boundary") else {
        return;
    };
    // Three strands flattening to 203 points (> 3 * 64), so the one-thread-per-
    // point dispatch spans several workgroups. Points are generated by integer
    // affine arithmetic (no transcendental calls) and kept non-coincident so
    // every interior difference is well conditioned.
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
