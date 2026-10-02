//! Real-device parity for the Xiaolin Wu anti-aliased line twin:
//! [`GpuWuLineRasterizer`](prism_volumetric_gpu::wu_line::GpuWuLineRasterizer)
//! must reproduce the `CPU` golden
//! [`wu_line`](prism_render_architecture::particle::wu_line) across the full
//! control-flow surface — horizontal and vertical axis-aligned runs, shallow
//! positive and negative slopes, steep positive and negative slopes, negative
//! coordinates, non-integer endpoints, a mid-pixel start that shifts the first
//! column, a long `(0, 0)` to `(100, 40)` line that exercises many interior
//! steps, a mixed batch, and a randomized integer-endpoint sweep compared
//! line-for-line.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! Each line's sample set is compared after sorting both sides by `(x, y)`: the
//! sample `count` must match exactly, each sample's integer `(x, y)` must match
//! exactly, and each `coverage` must agree within `abs_diff <= 1e-4` or
//! `rel_diff <= 1e-3`. The discrete count and coordinates follow `floor`-based
//! rounding and integer stepping, so they agree exactly for the chosen
//! fixtures; `coverage` threads through multiplies and adds, so a `GPU` fused
//! multiply-add may perturb its low mantissa bits and the tolerance admits that
//! legal slack.
//!
//! # Conditioning
//!
//! Every fixture is deliberately away from a discrete tie. Integer endpoints
//! make the endpoint gap term exact (`xend - a == 0`), so the interior `intery`
//! accumulation is a pure sum of the same `f32` gradient on both devices and no
//! fused multiply-add can shift a *coverage* value across the drop threshold.
//! Integer endpoints alone, however, do not pin the discrete pixel
//! *coordinates*: a diagonal line whose integer deltas share a common factor
//! crosses interior lattice points where the accumulator equals an integer
//! exactly, and there a `floor`-based `ipart` can straddle it differently once a
//! `GPU` reciprocal-multiply and a `CPU` true-divide disagree by a unit in the
//! last place. The randomized sweep therefore rejects diagonal spans with
//! `gcd(|dx|, |dy|) > 1` (see `random_line`), which keeps every surviving
//! diagonal accumulator at least `1 / |major span|` away from any integer —
//! far beyond the `f32` accumulation slack — while axis-aligned runs stay on an
//! exact-zero gradient or constant column. The one non-integer fixture likewise
//! starts on an integer `x` so its seed is exact, and its interior steps stay
//! clear of an integer crossing. This keeps `CPU` and `GPU` on the same side of
//! every discrete decision regardless of a few units in the last place of
//! slack.
//!
//! Provenance: twinned from this repository's
//! [`wu_line`](prism_render_architecture::particle::wu_line); no third-party
//! engine source or derived code.

use prism_render_architecture::particle::wu_line::rasterize;
use prism_volumetric_gpu::wu_line::{GpuWuLineRasterizer, WuLineQuery, WuLineResult};
use prism_volumetric_gpu::GpuContext;

/// Absolute parity bound on `coverage`. A `GPU` may fuse a multiply-add the
/// scalar reference leaves separate, perturbing the low mantissa bits by a few
/// units in the last place; `1e-4` admits that legal slack while still failing a
/// wrong port.
const EPS: f32 = 1.0e-4;

/// Relative parity bound, applied for larger magnitudes where a few units in the
/// last place exceed the absolute floor.
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

/// Sorts a reference sample list by `(x, y)` for a set-wise comparison.
fn sorted_golden(mut pixels: Vec<(i32, i32, f32)>) -> Vec<(i32, i32, f32)> {
    pixels.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)));
    pixels
}

/// Sorts a `GPU` result's samples by `(x, y)` for a set-wise comparison.
fn sorted_gpu(result: &WuLineResult) -> Vec<(i32, i32, f32)> {
    let mut pixels: Vec<(i32, i32, f32)> = result
        .pixels
        .iter()
        .map(|p| (p.x, p.y, p.coverage))
        .collect();
    pixels.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)));
    pixels
}

/// Dispatches every line and pins each result against the reference: the sample
/// count exactly, every `(x, y)` exactly, and every `coverage` within tolerance.
fn check(ctx: &GpuContext, gpu: &GpuWuLineRasterizer, lines: &[WuLineQuery]) {
    let got = gpu.eval(ctx, lines);
    assert_eq!(
        got.len(),
        lines.len(),
        "result count must match the input count"
    );
    for (idx, (line, result)) in lines.iter().zip(got.iter()).enumerate() {
        let want = sorted_golden(rasterize(line.x0, line.y0, line.x1, line.y1));
        let have = sorted_gpu(result);
        assert_eq!(
            have.len(),
            want.len(),
            "line {idx} sample count: gpu {} vs cpu {}",
            have.len(),
            want.len()
        );
        for (sample, (hp, wp)) in have.iter().zip(want.iter()).enumerate() {
            assert_eq!(
                (hp.0, hp.1),
                (wp.0, wp.1),
                "line {idx} sample {sample} coord: gpu ({}, {}) vs cpu ({}, {})",
                hp.0,
                hp.1,
                wp.0,
                wp.1
            );
            assert!(
                close(hp.2, wp.2),
                "line {idx} sample {sample} coverage: gpu {} vs cpu {}",
                hp.2,
                wp.2
            );
        }
    }
}

/// A tiny integer linear-congruential generator; only integer work, so no
/// transcendental appears. Returns the raw high bits as a `u32`.
fn lcg(state: &mut u64) -> u32 {
    *state = state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    (*state >> 33) as u32
}

/// Draws an integer coordinate in `[-25, 25]` from `state`.
fn coord(state: &mut u64) -> f32 {
    ((lcg(state) % 51) as i32 - 25) as f32
}

/// Greatest common divisor of two non-negative integers, by the Euclidean
/// algorithm. Only integer work, so no transcendental appears.
fn gcd(mut a: i64, mut b: i64) -> i64 {
    while b != 0 {
        let t = a % b;
        a = b;
        b = t;
    }
    a
}

/// Builds a non-degenerate integer-endpoint line by rejection sampling and
/// conditions it away from the one remaining discrete tie.
///
/// Both endpoints are integers, so the endpoint gap term is exact and the
/// interior minor-axis accumulation is a pure sum of the same `f32` gradient on
/// both devices. That alone, however, does not pin the integer pixel
/// coordinates: a diagonal span (`dx != 0 && dy != 0`) whose integer deltas
/// share a common factor passes through interior lattice points, and at each
/// such point the real-valued accumulator equals an integer exactly. There a
/// `floor`-based `ipart` can straddle the integer differently once a `GPU`
/// reciprocal-multiply and a `CPU` true-divide perturb the gradient by a unit
/// in the last place, shifting a painted pixel by one column or row. Rejecting
/// diagonal spans with `gcd(|dx|, |dy|) > 1` removes exactly those interior
/// lattice crossings: every surviving diagonal line keeps its interior
/// accumulator at least `1 / |major span|` (here ≥ 1/50 = 0.02) away from
/// any integer, dwarfing the ≈ 3e-6 worst-case `f32` accumulation slack, so
/// both devices `floor` to the same pixel. Axis-aligned runs (one span zero)
/// carry an exact-zero gradient or a constant column and are tie-free, so they
/// are kept. The two endpoints also differ, guaranteeing a non-empty sample
/// set.
fn random_line(state: &mut u64) -> WuLineQuery {
    loop {
        let x0 = coord(state);
        let y0 = coord(state);
        let x1 = coord(state);
        let y1 = coord(state);
        // Integer values compare exactly; reject a truly coincident pair.
        let same = (x0 - x1).abs() <= 0.0 && (y0 - y1).abs() <= 0.0;
        if same {
            continue;
        }
        // Reject diagonal spans that cross an interior lattice point (a shared
        // factor between the integer deltas): there the minor-axis accumulator
        // hits an exact integer and `floor`-rounding could pick a different
        // pixel across the true-divide vs reciprocal-multiply gap. Axis-aligned
        // runs (one delta zero) are tie-free and kept.
        let dx = (x1 - x0) as i64;
        let dy = (y1 - y0) as i64;
        if dx != 0 && dy != 0 && gcd(dx.abs(), dy.abs()) > 1 {
            continue;
        }
        return WuLineQuery::new(x0, y0, x1, y1);
    }
}

/// The ten deterministic control-flow fixtures, mirroring the reference's own
/// test geometries.
fn fixture_lines() -> Vec<WuLineQuery> {
    vec![
        // Horizontal: interior columns are a single full-coverage pixel.
        WuLineQuery::new(0.0, 0.0, 5.0, 0.0),
        // Vertical: the steep path with x held constant.
        WuLineQuery::new(3.0, 0.0, 3.0, 6.0),
        // Shallow positive slope (0.3).
        WuLineQuery::new(0.0, 0.0, 10.0, 3.0),
        // Shallow negative slope (-0.4).
        WuLineQuery::new(0.0, 0.0, 10.0, -4.0),
        // Steep positive slope: the axis swap walks y.
        WuLineQuery::new(0.0, 0.0, 3.0, 10.0),
        // Steep negative slope: axis swap plus a left-to-right flip.
        WuLineQuery::new(0.0, 0.0, -4.0, 10.0),
        // Fully negative coordinates.
        WuLineQuery::new(-5.0, -5.0, -1.0, -3.0),
        // Non-integer endpoints on integer x columns.
        WuLineQuery::new(-1.0, 0.5, 9.0, 4.2),
        // Mid-pixel start shifts the first painted column off zero.
        WuLineQuery::new(0.5, 0.0, 6.0, 0.0),
        // Long line: many interior steps, about 202 samples.
        WuLineQuery::new(0.0, 0.0, 100.0, 40.0),
    ]
}

#[test]
fn empty_input_returns_empty_without_dispatch() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuWuLineRasterizer::new(&ctx);
    // An empty batch short-circuits before any dispatch (a storage buffer cannot
    // be zero-sized) and returns an empty vector.
    let got = gpu.eval(&ctx, &[]);
    assert!(got.is_empty(), "an empty batch produces no results");
}

#[test]
fn horizontal_line_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuWuLineRasterizer::new(&ctx);
    check(&ctx, &gpu, &[WuLineQuery::new(0.0, 0.0, 5.0, 0.0)]);
}

#[test]
fn vertical_line_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuWuLineRasterizer::new(&ctx);
    check(&ctx, &gpu, &[WuLineQuery::new(3.0, 0.0, 3.0, 6.0)]);
}

#[test]
fn shallow_slopes_match_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuWuLineRasterizer::new(&ctx);
    check(
        &ctx,
        &gpu,
        &[
            WuLineQuery::new(0.0, 0.0, 10.0, 3.0),
            WuLineQuery::new(0.0, 0.0, 10.0, -4.0),
        ],
    );
}

#[test]
fn steep_slopes_match_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuWuLineRasterizer::new(&ctx);
    check(
        &ctx,
        &gpu,
        &[
            WuLineQuery::new(0.0, 0.0, 3.0, 10.0),
            WuLineQuery::new(0.0, 0.0, -4.0, 10.0),
        ],
    );
}

#[test]
fn negative_coordinates_match_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuWuLineRasterizer::new(&ctx);
    check(&ctx, &gpu, &[WuLineQuery::new(-5.0, -5.0, -1.0, -3.0)]);
}

#[test]
fn non_integer_endpoints_match_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuWuLineRasterizer::new(&ctx);
    check(&ctx, &gpu, &[WuLineQuery::new(-1.0, 0.5, 9.0, 4.2)]);
}

#[test]
fn mid_pixel_start_match_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuWuLineRasterizer::new(&ctx);
    check(&ctx, &gpu, &[WuLineQuery::new(0.5, 0.0, 6.0, 0.0)]);
}

#[test]
fn long_line_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuWuLineRasterizer::new(&ctx);
    check(&ctx, &gpu, &[WuLineQuery::new(0.0, 0.0, 100.0, 40.0)]);
}

#[test]
fn mixed_batch_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuWuLineRasterizer::new(&ctx);
    // The full control-flow fixture set dispatched together so the per-thread
    // indexing and the contiguous output slots are both exercised.
    check(&ctx, &gpu, &fixture_lines());
}

#[test]
fn random_integer_sweep_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuWuLineRasterizer::new(&ctx);
    let mut state = 0x1234_5678_9abc_def0_u64;
    // A larger sweep of integer-endpoint lines (several workgroups' worth) pins
    // every reported sample across many random slopes and lengths.
    let mut lines = fixture_lines();
    for _ in 0..180 {
        lines.push(random_line(&mut state));
    }
    check(&ctx, &gpu, &lines);
}
