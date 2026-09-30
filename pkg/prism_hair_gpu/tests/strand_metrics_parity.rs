//! Real-device parity for the per-strand geometry-metrics twin:
//! [`GpuStrandMetrics`] must reproduce the `CPU` goldens
//! [`strand_arc_length`](prism_render_architecture::hair::decimation::strand_arc_length)
//! and
//! [`strand_curvature`](prism_render_architecture::hair::decimation::strand_curvature)
//! for a batch of strands laid out strand-major at a fixed points-per-strand
//! stride, including the straight-strand (zero-curvature), coincident-point
//! (zero-tangent fallback), and too-short (fewer than two / three points)
//! guards, plus the empty-pool and zero-stride no-ops.
//!
//! The tests skip (with a printed notice) when the host has no `wgpu` adapter,
//! so the suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device. The kernel is portable core-WGSL,
//! so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! The reductions are closed-form geometry (the reference restricts itself to
//! `sqrt` via vector length) summed in ascending index order on both sides, so
//! the `CPU` and `GPU` evaluate the same expression and diverge only through
//! legal fused-multiply-add contraction. Parity is asserted per metric to within
//! `abs_diff < 1e-4` or `rel_diff < 1e-3` — tight enough to fail a genuinely
//! wrong port (a dropped segment, a wrong tangent guard, a missing length
//! guard), loose enough to admit the fma contraction. Curved cases also assert a
//! non-zero curvature so a no-op kernel could not pass. Point coordinates are
//! built from straight lines and polynomial samples (no `sin`/`cos`).
//!
//! Provenance: standard polyline arc-length / turning-angle geometry metrics; no
//! Unreal Engine source or derived code.

use prism_hair_gpu::strand_metrics::GpuStrandMetrics;
use prism_hair_gpu::GpuContext;
use prism_render_architecture::hair::decimation::{strand_arc_length, strand_curvature};
use prism_render_architecture::hair::interpolation::Vec3;

/// Asserts a single metric matches within the documented fma tolerance.
fn assert_close(got: f32, expected: f32, label: &str) {
    let abs_diff = (got - expected).abs();
    let rel_diff = abs_diff / expected.abs().max(1e-6);
    assert!(
        abs_diff < 1e-4 || rel_diff < 1e-3,
        "{label}: gpu {got}, cpu {expected} (abs {abs_diff}, rel {rel_diff})"
    );
}

/// Reference metrics for one strand slice via the `CPU` goldens.
fn cpu_metrics(points: &[[f32; 3]]) -> (f32, f32) {
    let vs: Vec<Vec3> = points.iter().map(|p| Vec3::new(p[0], p[1], p[2])).collect();
    (strand_arc_length(&vs), strand_curvature(&vs))
}

/// Asserts every `GPU` strand metric equals its `CPU` golden.
fn assert_parity(
    points: &[[f32; 3]],
    points_per_strand: usize,
    gpu: &[prism_hair_gpu::strand_metrics::GpuStrandMetric],
) {
    let strand_count = points.len() / points_per_strand;
    assert_eq!(gpu.len(), strand_count, "one metric per whole strand");
    for s in 0..strand_count {
        let slice = &points[s * points_per_strand..(s + 1) * points_per_strand];
        let (arc, curv) = cpu_metrics(slice);
        assert_close(gpu[s].arc_length, arc, &format!("strand {s} arc_length"));
        assert_close(gpu[s].curvature, curv, &format!("strand {s} curvature"));
    }
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn gpu_mixed_strands_match_cpu() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping strand-metrics parity: no wgpu adapter on this host");
        return;
    };

    // Five points per strand. Strand 0 is a straight line (curvature 0); strand
    // 1 is a zigzag polyline (sharp turns, high curvature); strand 2 samples the
    // parabola y = x^2 (smooth bend, moderate curvature) — no sin/cos anywhere.
    let points_per_strand = 5usize;
    let points: Vec<[f32; 3]> = vec![
        // strand 0: straight along +x, uneven spacing.
        [0.0, 0.0, 0.0],
        [0.5, 0.0, 0.0],
        [1.3, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [3.1, 0.0, 0.0],
        // strand 1: zigzag in xy.
        [0.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [2.0, 0.0, 0.0],
        [3.0, 1.0, 0.0],
        [4.0, 0.0, 0.0],
        // strand 2: parabola samples (x, x^2, 0).
        [-2.0, 4.0, 0.0],
        [-1.0, 1.0, 0.0],
        [0.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [2.0, 4.0, 0.0],
    ];
    let metrics = GpuStrandMetrics::new(&ctx);
    let gpu = metrics.eval(&ctx, &points, points_per_strand);

    assert_parity(&points, points_per_strand, &gpu);
    // The straight strand has zero curvature; the bent strands must be positive,
    // so a no-op (all-zero) kernel cannot pass.
    assert!(
        gpu[0].curvature < 1e-4,
        "straight strand flat, got {}",
        gpu[0].curvature
    );
    assert!(
        gpu[1].curvature > 1e-3,
        "zigzag strand bends, got {}",
        gpu[1].curvature
    );
    assert!(
        gpu[2].curvature > 1e-3,
        "parabola strand bends, got {}",
        gpu[2].curvature
    );
    // Arc length is positive on every strand.
    assert!(gpu[0].arc_length > 1e-3 && gpu[1].arc_length > 1e-3 && gpu[2].arc_length > 1e-3);
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn gpu_two_point_stride_has_zero_curvature() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping strand-metrics parity: no wgpu adapter on this host");
        return;
    };

    // Two points per strand: arc length is the single segment, curvature is 0
    // (fewer than three points), matching both goldens' length guards.
    let points_per_strand = 2usize;
    let points: Vec<[f32; 3]> = vec![
        [0.0, 0.0, 0.0],
        [3.0, 4.0, 0.0], // length 5
        [1.0, 1.0, 1.0],
        [1.0, 1.0, 1.0], // coincident: zero-length segment, no divide-by-zero
    ];
    let metrics = GpuStrandMetrics::new(&ctx);
    let gpu = metrics.eval(&ctx, &points, points_per_strand);

    assert_parity(&points, points_per_strand, &gpu);
    assert!(
        gpu[0].curvature.abs() < 1e-6,
        "two points => zero curvature"
    );
    assert!(
        gpu[1].curvature.abs() < 1e-6,
        "two points => zero curvature"
    );
    assert!(
        (gpu[0].arc_length - 5.0).abs() < 1e-4,
        "3-4-5 arc, got {}",
        gpu[0].arc_length
    );
    assert!(
        gpu[1].arc_length.abs() < 1e-6,
        "coincident arc 0, got {}",
        gpu[1].arc_length
    );
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn gpu_coincident_interior_point_stays_finite() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping strand-metrics parity: no wgpu adapter on this host");
        return;
    };

    // A strand with a repeated interior point exercises the normalize_or
    // zero-length fallback (a zero tangent contributes `1 - 0 = 1` to the sum)
    // without producing a NaN, matching the golden.
    let points_per_strand = 4usize;
    let points: Vec<[f32; 3]> = vec![
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0], // coincident with previous: zero incoming/outgoing edge
        [2.0, 1.0, 0.0],
    ];
    let metrics = GpuStrandMetrics::new(&ctx);
    let gpu = metrics.eval(&ctx, &points, points_per_strand);

    assert_parity(&points, points_per_strand, &gpu);
    assert!(
        gpu[0].curvature.is_finite(),
        "curvature finite despite coincident point"
    );
    assert!(gpu[0].arc_length.is_finite());
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn gpu_single_point_stride_is_all_zero() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping strand-metrics parity: no wgpu adapter on this host");
        return;
    };

    // One point per strand: both metrics are 0 (fewer than two points).
    let points_per_strand = 1usize;
    let points: Vec<[f32; 3]> = vec![[0.0, 0.0, 0.0], [5.0, 5.0, 5.0], [9.0, 1.0, 2.0]];
    let metrics = GpuStrandMetrics::new(&ctx);
    let gpu = metrics.eval(&ctx, &points, points_per_strand);

    assert_eq!(gpu.len(), 3);
    for m in &gpu {
        assert!(m.arc_length.abs() < 1e-6 && m.curvature.abs() < 1e-6);
    }
}

#[test]
fn empty_pool_and_zero_stride_are_no_ops() {
    // These no-op guards return before touching the GPU, so they run and assert
    // even on hosts without a wgpu adapter.
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let metrics = GpuStrandMetrics::new(&ctx);
    assert!(metrics.eval(&ctx, &[], 4).is_empty(), "empty pool => empty");
    let pts = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]];
    assert!(
        metrics.eval(&ctx, &pts, 0).is_empty(),
        "zero stride => empty"
    );
    // Fewer points than one stride: no whole strand fits.
    assert!(
        metrics.eval(&ctx, &pts, 5).is_empty(),
        "partial strand => empty"
    );
}
