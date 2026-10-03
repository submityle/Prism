//! Real-device parity for the motion-tile-classification twin:
//! [`GpuMotionTileClassify`](prism_volumetric_gpu::motion_tile_classify::GpuMotionTileClassify)
//! must reproduce the three stateless per-tile closed forms of the `CPU` golden
//! [`tiles`](prism_render_architecture::motion::tiles) — the motion bucket, the
//! `TAA` resolve flag bitmask, and the motion-blur half-length — across the
//! static / slow / fast buckets, their inclusive-downward boundaries, the
//! half-length scaling and clamp, `NaN` / negative sanitization, a mixed batch,
//! and a randomized sweep compared query-for-query.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Oracle
//!
//! The reference functions are public, so they are called directly as the
//! oracle:
//! [`classify_tile`](prism_render_architecture::motion::tiles::classify_tile)
//! for the bucket,
//! [`taa_flags`](prism_render_architecture::motion::tiles::taa_flags) plus
//! [`TaaTileFlags::bits`](prism_render_architecture::motion::tiles::TaaTileFlags::bits)
//! for the flag bitmask, and
//! [`motion_blur_half_length`](prism_render_architecture::motion::tiles::motion_blur_half_length)
//! for the half-length. A passing `GPU == oracle` run is direct evidence the
//! kernel computes the same tile decisions.
//!
//! # Parity criterion
//!
//! The class code and the flag bitmask are built from ordered comparisons, so
//! for fixtures clear of a threshold they agree exactly and are asserted with
//! `==`. The continuous half-length threads through a `sqrt`, a multiply and a
//! clamp, so it is asserted within `abs_diff <= 1e-4` or `rel_diff <= 1e-3`.
//!
//! # Conditioning
//!
//! The randomized sweep keeps every velocity magnitude a safe margin away from
//! both thresholds, so the device's squared-magnitude comparison and the
//! reference's `sqrt`-magnitude comparison bucket the tile identically and the
//! discrete answers stay bit-identical. The thresholds are built through the
//! reference's own sanitizing constructor so both sides see the same values.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::motion::tiles`；无第三方引擎源码或衍生代码。

use prism_render_architecture::motion::tiles::{
    classify_tile, motion_blur_half_length, taa_flags, MotionTileClass, TileClassifierParams,
};
use prism_render_architecture::motion::Vec2;
use prism_volumetric_gpu::motion_tile_classify::{
    GpuMotionTileClassify, MotionTileClassifyQuery, MotionTileClassifyResult,
};
use prism_volumetric_gpu::GpuContext;

/// Absolute parity bound on a half-length. A `GPU` `sqrt`/multiply/clamp may land
/// a few units in the last place from the scalar reference; `1e-4` admits that
/// legal slack while still failing a wrong port.
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

/// Maps a reference [`MotionTileClass`] to the device class code
/// (`0` static, `1` slow, `2` fast), matching the enum discriminant order.
fn class_code(class: MotionTileClass) -> u32 {
    match class {
        MotionTileClass::Static => 0,
        MotionTileClass::Slow => 1,
        MotionTileClass::Fast => 2,
    }
}

/// Builds a query whose thresholds are already sanitized by the reference's own
/// constructor, so the device and the golden see identical thresholds.
fn query(
    velocity: [f32; 2],
    static_max: f32,
    slow_max: f32,
    shutter: f32,
    max_radius: f32,
) -> MotionTileClassifyQuery {
    let params = TileClassifierParams::new(static_max, slow_max);
    MotionTileClassifyQuery::new(
        velocity,
        params.static_max_pixels,
        params.slow_max_pixels,
        shutter,
        max_radius,
    )
}

/// Computes the reference answer for one query by calling the three golden maps
/// directly: the bucket becomes the class code, the bucket's flags become the
/// bitmask, and the half-length is reproduced as-is.
fn oracle(q: &MotionTileClassifyQuery) -> MotionTileClassifyResult {
    let params = TileClassifierParams {
        static_max_pixels: q.static_max_pixels,
        slow_max_pixels: q.slow_max_pixels,
    };
    let velocity = Vec2::new(q.velocity[0], q.velocity[1]);
    let class = classify_tile(velocity, params);
    let flags = taa_flags(class).bits();
    let half_length = motion_blur_half_length(velocity, q.shutter_fraction, q.max_radius_pixels);
    MotionTileClassifyResult {
        class_code: class_code(class),
        flags,
        half_length,
    }
}

/// Pins one `GPU` result against the oracle: the class code and the flag bitmask
/// exactly, the half-length within tolerance.
fn check_result(idx: usize, got: &MotionTileClassifyResult, want: &MotionTileClassifyResult) {
    assert_eq!(
        got.class_code, want.class_code,
        "query {idx} class_code: gpu {} vs cpu {}",
        got.class_code, want.class_code
    );
    assert_eq!(
        got.flags, want.flags,
        "query {idx} flags: gpu {} vs cpu {}",
        got.flags, want.flags
    );
    assert!(
        close(got.half_length, want.half_length),
        "query {idx} half_length: gpu {} vs cpu {}",
        got.half_length,
        want.half_length
    );
}

/// Dispatches every query and pins each result against the oracle.
fn check(ctx: &GpuContext, gpu: &GpuMotionTileClassify, queries: &[MotionTileClassifyQuery]) {
    let got = gpu.evaluate(ctx, queries);
    assert_eq!(
        got.len(),
        queries.len(),
        "result count must match the input count"
    );
    for (idx, (q, result)) in queries.iter().zip(got.iter()).enumerate() {
        let want = oracle(q);
        check_result(idx, result, &want);
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

/// Default fixture thresholds (the reference defaults): static `0.5`, slow `4.0`.
const STATIC_MAX: f32 = 0.5;
/// Slow-bucket magnitude threshold for the default fixture.
const SLOW_MAX: f32 = 4.0;

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn empty_input_returns_empty_without_dispatch() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping motion_tile_classify parity: no wgpu adapter on this host");
        return;
    };
    let gpu = GpuMotionTileClassify::new(&ctx);
    // An empty batch short-circuits before any dispatch (a storage buffer cannot
    // be zero-sized) and returns an empty vector.
    let got = gpu.evaluate(&ctx, &[]);
    assert!(got.is_empty(), "an empty batch produces no results");
}

#[test]
fn static_slow_fast_buckets_match_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuMotionTileClassify::new(&ctx);
    // One velocity well inside each bucket under the default thresholds.
    let queries = [
        query([0.2, 0.0], STATIC_MAX, SLOW_MAX, 1.0, 100.0),
        query([2.0, 0.0], STATIC_MAX, SLOW_MAX, 1.0, 100.0),
        query([9.0, 0.0], STATIC_MAX, SLOW_MAX, 1.0, 100.0),
    ];
    let got = gpu.evaluate(&ctx, &queries);
    assert_eq!(got[0].class_code, 0, "sub-threshold velocity is static");
    assert_eq!(got[1].class_code, 1, "mid velocity is slow");
    assert_eq!(got[2].class_code, 2, "large velocity is fast");
    check(&ctx, &gpu, &queries);
}

#[test]
fn boundaries_are_inclusive_downward() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuMotionTileClassify::new(&ctx);
    // Integer thresholds whose magnitudes are perfect squares, so the reference
    // `sqrt` is exact and the squared comparison agrees at the boundary.
    let queries = [
        // Exactly at the static bound => static.
        query([1.0, 0.0], 1.0, 2.0, 1.0, 100.0),
        // Exactly at the slow bound => slow.
        query([2.0, 0.0], 1.0, 2.0, 1.0, 100.0),
        // Just above the slow bound => fast.
        query([2.0001, 0.0], 1.0, 2.0, 1.0, 100.0),
    ];
    let got = gpu.evaluate(&ctx, &queries);
    assert_eq!(
        got[0].class_code, 0,
        "velocity at the static bound is static"
    );
    assert_eq!(got[1].class_code, 1, "velocity at the slow bound is slow");
    assert_eq!(
        got[2].class_code, 2,
        "velocity above the slow bound is fast"
    );
    check(&ctx, &gpu, &queries);
}

#[test]
fn flag_bitmasks_match_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuMotionTileClassify::new(&ctx);
    // The three buckets map to STATIC(1), NEEDS_NEIGHBOR_CLAMP(4) and
    // FAST_MOTION|NEEDS_DILATION|NEEDS_NEIGHBOR_CLAMP(14).
    let queries = [
        query([0.2, 0.0], STATIC_MAX, SLOW_MAX, 1.0, 100.0),
        query([2.0, 0.0], STATIC_MAX, SLOW_MAX, 1.0, 100.0),
        query([9.0, 0.0], STATIC_MAX, SLOW_MAX, 1.0, 100.0),
    ];
    let got = gpu.evaluate(&ctx, &queries);
    assert_eq!(got[0].flags, 1, "static bucket sets only STATIC");
    assert_eq!(
        got[1].flags, 4,
        "slow bucket sets only NEEDS_NEIGHBOR_CLAMP"
    );
    assert_eq!(got[2].flags, 14, "fast bucket sets the conservative flags");
    check(&ctx, &gpu, &queries);
}

#[test]
fn half_length_scales_and_clamps() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuMotionTileClassify::new(&ctx);
    let queries = [
        // 10px/frame, full shutter => 5px half-length.
        query([10.0, 0.0], STATIC_MAX, SLOW_MAX, 1.0, 100.0),
        // Half shutter halves it again => 2.5px.
        query([10.0, 0.0], STATIC_MAX, SLOW_MAX, 0.5, 100.0),
        // Runaway velocity clamped to the max radius => 8px.
        query([1000.0, 0.0], STATIC_MAX, SLOW_MAX, 1.0, 8.0),
    ];
    let got = gpu.evaluate(&ctx, &queries);
    assert!(close(got[0].half_length, 5.0), "full shutter half-length");
    assert!(close(got[1].half_length, 2.5), "half shutter half-length");
    assert!(close(got[2].half_length, 8.0), "clamped half-length");
    check(&ctx, &gpu, &queries);
}

#[test]
fn nan_and_negative_shutter_sanitize_to_zero() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuMotionTileClassify::new(&ctx);
    // A NaN shutter and a negative max radius both sanitize to zero, so the
    // half-length collapses to zero while the bucket is unaffected.
    let queries = [
        query([10.0, 0.0], STATIC_MAX, SLOW_MAX, f32::NAN, -3.0),
        query([10.0, 0.0], STATIC_MAX, SLOW_MAX, -1.0, 100.0),
    ];
    let got = gpu.evaluate(&ctx, &queries);
    assert!(
        close(got[0].half_length, 0.0),
        "NaN shutter / neg radius => 0"
    );
    assert!(close(got[1].half_length, 0.0), "negative shutter => 0");
    check(&ctx, &gpu, &queries);
}

#[test]
fn mixed_batch_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuMotionTileClassify::new(&ctx);
    // A heterogeneous batch spanning buckets, shutter fractions and clamps,
    // including a diagonal velocity.
    let queries = [
        query([0.1, 0.1], STATIC_MAX, SLOW_MAX, 1.0, 50.0),
        query([3.0, 0.0], STATIC_MAX, SLOW_MAX, 0.25, 50.0),
        query([0.0, 12.0], STATIC_MAX, SLOW_MAX, 1.0, 4.0),
        query([6.0, 8.0], 1.0, 5.0, 0.5, 100.0),
        query([30.0, 40.0], 2.0, 10.0, 1.0, 12.0),
    ];
    check(&ctx, &gpu, &queries);
}

#[test]
fn random_sweep_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuMotionTileClassify::new(&ctx);
    let mut state = 0x51ed_270b_8c4f_a97d_u64;
    let mut queries = Vec::new();
    // Several workgroups' worth of queries. Each velocity magnitude is kept a
    // safe margin clear of both thresholds (via rejection) so the squared and
    // the `sqrt` comparisons bucket the tile identically.
    let margin = 0.3f32;
    while queries.len() < 300 {
        // Velocity components in [-7, 7].
        let vx = (lcg(&mut state) % 14001) as f32 / 1000.0 - 7.0;
        let vy = (lcg(&mut state) % 14001) as f32 / 1000.0 - 7.0;
        let mag = (vx * vx + vy * vy).sqrt();
        if (mag - STATIC_MAX).abs() < margin || (mag - SLOW_MAX).abs() < margin {
            continue;
        }
        // Shutter in [0, 1], max radius in [0, 20].
        let shutter = (lcg(&mut state) % 1001) as f32 / 1000.0;
        let max_radius = (lcg(&mut state) % 20001) as f32 / 1000.0;
        queries.push(query([vx, vy], STATIC_MAX, SLOW_MAX, shutter, max_radius));
    }
    check(&ctx, &gpu, &queries);
}
