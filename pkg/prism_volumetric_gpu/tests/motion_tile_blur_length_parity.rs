//! Real-device parity for the per-tile motion-blur half-length twin:
//! [`GpuMotionTileBlurLength`](prism_volumetric_gpu::motion_tile_blur_length::GpuMotionTileBlurLength)
//! must reproduce the clamped half-length of the `CPU` golden
//! [`motion_blur_half_length`](prism_render_architecture::motion::tiles::motion_blur_half_length)
//! across the shutter-scaled, halved, and clamped regimes plus a randomized
//! batch compared value-for-value.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Oracle
//!
//! The golden
//! [`motion_blur_half_length`](prism_render_architecture::motion::tiles::motion_blur_half_length)
//! is `pub`, so each `GPU` result is pinned directly against the golden run on
//! the same input: a tile [`Vec2`](prism_render_architecture::motion::Vec2)
//! velocity, a shutter fraction, and a clamp radius.
//!
//! # Parity criterion
//!
//! The half-length is a single `sqrt` followed by multiplies and clamps — no
//! transcendental — so it is a continuous `f32` asserted within
//! `abs_diff <= 1e-4` or `rel_diff <= 1e-3`.
//!
//! # Conditioning
//!
//! Fixtures stay finite (the golden's `NaN` sanitization is a host concern) and
//! stay away from the clamp tie: either the scaled half-length sits clearly
//! below `max_radius_pixels` or clearly above it, so a last-bit `sqrt`
//! difference cannot flip which operand the final `min` selects. Shutter
//! fractions stay inside `[0, 1]` away from the saturation knee, and random
//! velocities keep magnitudes moderate so the summed products stay inside the
//! relative bound.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::motion::tiles::motion_blur_half_length`；无第三方引擎源码或衍生代码。

use prism_render_architecture::motion::tiles::motion_blur_half_length;
use prism_render_architecture::motion::Vec2;
use prism_volumetric_gpu::motion_tile_blur_length::{
    GpuMotionTileBlurLength, MotionTileBlurLengthQuery, MotionTileBlurLengthResult,
};
use prism_volumetric_gpu::GpuContext;

/// Absolute parity bound on the half-length.
const EPS: f32 = 1.0e-4;

/// Relative parity bound, applied for larger magnitudes.
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

/// Computes the golden result for one query, so the oracle lives beside the
/// device call and both read the same input.
fn expected(q: &MotionTileBlurLengthQuery) -> MotionTileBlurLengthResult {
    let velocity = Vec2::new(q.velocity[0], q.velocity[1]);
    let half_length = motion_blur_half_length(velocity, q.shutter_fraction, q.max_radius_pixels);
    MotionTileBlurLengthResult { half_length }
}

/// Pins one `GPU` result against the golden oracle within tolerance.
fn assert_result(idx: usize, got: &MotionTileBlurLengthResult, want: &MotionTileBlurLengthResult) {
    assert!(
        close(got.half_length, want.half_length),
        "result {idx} half_length: gpu {} vs cpu {}",
        got.half_length,
        want.half_length
    );
}

/// Runs every query on the device and pins each result against the oracle.
fn run_and_check(ctx: &GpuContext, queries: &[MotionTileBlurLengthQuery]) {
    let gpu = GpuMotionTileBlurLength::new(ctx);
    let got = gpu.evaluate(ctx, queries);
    assert_eq!(got.len(), queries.len(), "one result per query");
    for (idx, (q, g)) in queries.iter().zip(got.iter()).enumerate() {
        assert_result(idx, g, &expected(q));
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

/// Draws a float in `[0, 1)` from `state` using only integer work.
fn unit(state: &mut u64) -> f32 {
    (lcg(state) >> 8) as f32 / (1u32 << 24) as f32
}

/// Draws a float in `[lo, hi)` from `state`.
fn ranged(state: &mut u64, lo: f32, hi: f32) -> f32 {
    lo + (hi - lo) * unit(state)
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn empty_input_returns_empty_without_dispatch() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping motion_tile_blur_length parity: no wgpu adapter on this host");
        return;
    };
    let gpu = GpuMotionTileBlurLength::new(&ctx);
    let got = gpu.evaluate(&ctx, &[]);
    assert!(got.is_empty(), "an empty batch produces no results");
}

#[test]
fn unclamped_fixtures() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    // Half-length stays well below the clamp: full shutter halves the speed,
    // half shutter halves it again, and a zero velocity yields zero.
    let queries = vec![
        // 10px/frame, full shutter => 5px, clamp 100 never bites.
        MotionTileBlurLengthQuery::new([10.0, 0.0], 1.0, 100.0),
        // Half shutter => 2.5px.
        MotionTileBlurLengthQuery::new([10.0, 0.0], 0.5, 100.0),
        // Diagonal 3-4-5 velocity => length 5, 0.6 shutter => 1.5px.
        MotionTileBlurLengthQuery::new([3.0, 4.0], 0.6, 100.0),
        // Zero velocity => zero half-length.
        MotionTileBlurLengthQuery::new([0.0, 0.0], 1.0, 100.0),
        // Zero shutter => zero half-length even with large velocity.
        MotionTileBlurLengthQuery::new([40.0, 30.0], 0.0, 100.0),
    ];
    run_and_check(&ctx, &queries);
}

#[test]
fn clamped_fixtures() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    // Half-length lands well above the clamp, so the result saturates to the
    // clamp radius with a comfortable margin from the tie.
    let queries = vec![
        // 1000px/frame, full shutter => 500px, clamped to 8.
        MotionTileBlurLengthQuery::new([1000.0, 0.0], 1.0, 8.0),
        // Diagonal 60-80 => length 100, full shutter => 50px, clamped to 12.
        MotionTileBlurLengthQuery::new([60.0, 80.0], 1.0, 12.0),
        // Large velocity, moderate shutter => still far above a small clamp.
        MotionTileBlurLengthQuery::new([200.0, 0.0], 0.5, 4.0),
    ];
    run_and_check(&ctx, &queries);
}

#[test]
fn shutter_saturation_fixtures() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    // Shutter fractions above 1 saturate to 1; negatives sanitize to 0. The
    // fixtures stay clear of the clamp tie on either side.
    let queries = vec![
        // Shutter 1.5 clamps to 1 => 20px/frame * 0.5 = 10px, clamp 100.
        MotionTileBlurLengthQuery::new([20.0, 0.0], 1.5, 100.0),
        // Negative shutter sanitizes to 0 => zero half-length.
        MotionTileBlurLengthQuery::new([20.0, 0.0], -0.75, 100.0),
    ];
    run_and_check(&ctx, &queries);
}

#[test]
fn random_batch_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let mut state = 0x7b19_42c0_6d5e_1a37_u64;

    let mut queries: Vec<MotionTileBlurLengthQuery> = Vec::new();
    while queries.len() < 256 {
        let vx = ranged(&mut state, -20.0, 20.0);
        let vy = ranged(&mut state, -20.0, 20.0);
        // Shutter kept inside (0.1, 0.9) so it never touches the [0, 1] knees.
        let shutter = ranged(&mut state, 0.1, 0.9);
        let max_radius = ranged(&mut state, 2.0, 40.0);
        // Reject fixtures whose scaled half-length lands near the clamp tie, so
        // a last-bit sqrt difference cannot flip which operand `min` selects.
        let length = (vx * vx + vy * vy).sqrt();
        let half_length = length * shutter * 0.5;
        let gap = (half_length - max_radius).abs();
        if gap < 0.25 {
            continue;
        }
        queries.push(MotionTileBlurLengthQuery::new(
            [vx, vy],
            shutter,
            max_radius,
        ));
    }
    run_and_check(&ctx, &queries);
}
