//! Real-device parity for the isolated analytic line-coverage twin:
//! [`GpuHairLineCoverage`] must reproduce the `CPU` golden
//! [`reference_coverage`](prism_hair_gpu::line_coverage::reference_coverage)
//! (built on
//! [`pixel_coverage`](prism_render_architecture::hair::line_coverage::pixel_coverage))
//! for a batch of pixel centres against one width-carrying fibre segment,
//! computing each pixel's trapezoidal coverage ramp independently. The suite
//! drives a centre on the segment, a far pixel, a partial-ramp pixel, a
//! zero-width line, endpoint clamping, a degenerate zero-length segment, the
//! negative/non-finite width clamp, the empty no-op, and a large multi-workgroup
//! batch that crosses the 64-wide dispatch boundary.
//!
//! The tests skip (with a printed notice) when the host has no `wgpu` adapter,
//! so the suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device. The kernel is portable core-WGSL,
//! so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! Coverage is a `dot`/`sqrt` distance folded through a clamped multiply-add the
//! scalar reference may leave separate while a `GPU` fuses it, so every value is
//! asserted within `abs_diff < 1e-4` or `rel_diff < 1e-3`. No `sin`/`cos`
//! appears anywhere; all inputs are explicit literals or integer-derived grids.
//!
//! Provenance: analytic line-AA coverage (`UE5` Groom / `Weta` / `Pixar`
//! line-AA literature) plus `wgpu` compute dispatch; no Unreal Engine source or
//! derived code.

use prism_hair_gpu::line_coverage::{reference_coverage, GpuHairLineCoverage};
use prism_hair_gpu::GpuContext;
use prism_render_architecture::hair::line_coverage::ScreenSegment;

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

/// Asserts two scalars agree within the documented fma tolerance.
fn assert_close(got: f32, want: f32, what: &str) {
    let abs = (got - want).abs();
    let rel = abs / want.abs().max(1.0);
    assert!(
        abs < 1e-4 || rel < 1e-3,
        "{what}: got {got}, want {want} (abs {abs}, rel {rel})"
    );
}

/// Asserts a whole batch of coverages matches the `CPU` golden pixel by pixel.
fn assert_batch(got: &[f32], seg: ScreenSegment, centers: &[[f32; 2]]) {
    assert_eq!(got.len(), centers.len(), "one coverage per pixel");
    for (i, (&out, &center)) in got.iter().zip(centers.iter()).enumerate() {
        let want = reference_coverage(seg, center);
        assert_close(out, want, &format!("pixel {i}"));
        assert!(
            (0.0..=1.0).contains(&out),
            "pixel {i} coverage must stay in [0, 1], got {out}"
        );
    }
}

/// Dispatches one batch through the device twin.
fn run(ctx: &GpuContext, seg: ScreenSegment, centers: &[[f32; 2]]) -> Vec<f32> {
    GpuHairLineCoverage::new(ctx).eval(ctx, seg, centers)
}

#[test]
fn center_on_segment_is_full_coverage() {
    let Some(ctx) = context_or_skip("center_on_segment_is_full_coverage") else {
        return;
    };
    let seg = ScreenSegment::new([0.0, 0.0], [10.0, 0.0], 1.0);
    let centers = [[5.0, 0.0]];
    assert_batch(&run(&ctx, seg, &centers), seg, &centers);
}

#[test]
fn far_pixel_is_zero_coverage() {
    let Some(ctx) = context_or_skip("far_pixel_is_zero_coverage") else {
        return;
    };
    let seg = ScreenSegment::new([0.0, 0.0], [10.0, 0.0], 1.0);
    let centers = [[5.0, 100.0]];
    let got = run(&ctx, seg, &centers);
    assert_batch(&got, seg, &centers);
    assert_close(got[0], 0.0, "far pixel");
}

#[test]
fn partial_ramp_matches_golden() {
    let Some(ctx) = context_or_skip("partial_ramp_matches_golden") else {
        return;
    };
    // A unit-width line at perpendicular distance 0.7: coverage = clamp(0.5 +
    // 0.5 - 0.7, 0, 1) = 0.3, strictly inside the ramp.
    let seg = ScreenSegment::new([0.0, 0.0], [10.0, 0.0], 1.0);
    let centers = [[3.0, 0.7]];
    assert_batch(&run(&ctx, seg, &centers), seg, &centers);
}

#[test]
fn zero_width_line_matches_golden() {
    let Some(ctx) = context_or_skip("zero_width_line_matches_golden") else {
        return;
    };
    // A zero-width line still covers via the pixel kernel radius: coverage =
    // clamp(0 + 0.5 - 0.3, 0, 1) = 0.2 at perpendicular distance 0.3.
    let seg = ScreenSegment::new([0.0, 0.0], [10.0, 0.0], 0.0);
    let centers = [[4.0, 0.3], [4.0, 0.6]];
    assert_batch(&run(&ctx, seg, &centers), seg, &centers);
}

#[test]
fn endpoint_clamping_matches_golden() {
    let Some(ctx) = context_or_skip("endpoint_clamping_matches_golden") else {
        return;
    };
    // Pixels past both ends must use the endpoint distance (projection clamped
    // to [0, 1]), not the infinite-line distance.
    let seg = ScreenSegment::new([2.0, 2.0], [6.0, 2.0], 1.5);
    let centers = [[-1.0, 2.0], [9.0, 2.0], [2.0, 2.4], [6.0, 1.8]];
    assert_batch(&run(&ctx, seg, &centers), seg, &centers);
}

#[test]
fn degenerate_segment_matches_golden() {
    let Some(ctx) = context_or_skip("degenerate_segment_matches_golden") else {
        return;
    };
    // Zero-length segment collapses to the point distance |p - a|.
    let seg = ScreenSegment::new([3.0, 3.0], [3.0, 3.0], 1.0);
    let centers = [[3.0, 3.0], [3.3, 3.0], [3.0, 2.6], [5.0, 5.0]];
    assert_batch(&run(&ctx, seg, &centers), seg, &centers);
}

#[test]
fn negative_and_non_finite_width_clamp_to_zero() {
    let Some(ctx) = context_or_skip("negative_and_non_finite_width_clamp_to_zero") else {
        return;
    };
    // Negative, NaN and +inf widths must each collapse to 0, so coverage falls
    // back to the pixel-kernel-only ramp exactly like the golden.
    let centers = [[2.0, 0.2], [2.0, 0.4]];
    for width in [-5.0, f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        let seg = ScreenSegment::new([0.0, 0.0], [10.0, 0.0], width);
        let got = run(&ctx, seg, &centers);
        assert_batch(&got, seg, &centers);
        assert!(
            got.iter().all(|c| c.is_finite()),
            "coverage must stay finite for width {width}, got {got:?}"
        );
    }
}

#[test]
fn empty_batch_is_noop() {
    let Some(ctx) = context_or_skip("empty_batch_is_noop") else {
        return;
    };
    let seg = ScreenSegment::new([0.0, 0.0], [1.0, 0.0], 1.0);
    let got = run(&ctx, seg, &[]);
    assert!(got.is_empty(), "empty batch yields no coverages");
}

#[test]
fn large_batch_crosses_workgroup_boundary() {
    let Some(ctx) = context_or_skip("large_batch_crosses_workgroup_boundary") else {
        return;
    };
    // 130 pixel centres span three 64-wide workgroups over a deterministic grid
    // straddling a diagonal segment, so every regime (full, ramp, zero) and the
    // dispatch boundary are exercised at once.
    let seg = ScreenSegment::new([1.0, 1.0], [12.0, 9.0], 2.0);
    let mut centers = Vec::new();
    for k in 0u32..130 {
        let x = (k % 13) as f32;
        let y = ((k / 13) % 10) as f32;
        centers.push([x, y]);
    }
    assert_batch(&run(&ctx, seg, &centers), seg, &centers);
}
