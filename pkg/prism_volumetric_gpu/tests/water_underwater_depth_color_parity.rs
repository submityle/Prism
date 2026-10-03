//! Real-device parity for the underwater depth-colour twin:
//! [`GpuWaterUnderwaterDepthColor`](prism_volumetric_gpu::water_underwater_depth_color::GpuWaterUnderwaterDepthColor)
//! must reproduce the `CPU` golden
//! [`depth_color_shift`](prism_render_architecture::water::underwater::depth_color_shift)
//! across the shallow, deep, zero-depth, and per-channel-ordering regimes plus
//! a randomized batch compared channel-for-channel.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Oracle
//!
//! The golden [`depth_color_shift`](prism_render_architecture::water::underwater::depth_color_shift)
//! is `pub`, so each `GPU` channel is pinned directly against the golden run on
//! the same input.
//!
//! # Parity criterion
//!
//! Every channel is a continuous `f32` (a clamp of the twelve-step squaring
//! times a floored multiply), asserted within `abs_diff <= 1e-4` or
//! `rel_diff <= 1e-3`.
//!
//! # Conditioning
//!
//! Fixtures keep the colour and extinction well away from the sign flip at
//! zero, so the `max(_, 0)` and the `clamp` never straddle a tie, and the
//! fixed-step squaring of `exp_approx` is identical on both sides.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::water::underwater`；无第三方引擎源码或衍生代码。

use prism_render_architecture::water::underwater::{depth_color_shift, RgbColor, RgbExtinction};
use prism_volumetric_gpu::water_underwater_depth_color::{
    GpuWaterUnderwaterDepthColor, WaterUnderwaterDepthColorQuery, WaterUnderwaterDepthColorResult,
};
use prism_volumetric_gpu::GpuContext;

/// Absolute parity bound on each channel.
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
fn expected(q: &WaterUnderwaterDepthColorQuery) -> WaterUnderwaterDepthColorResult {
    let shifted = depth_color_shift(
        RgbColor {
            r: q.color_r,
            g: q.color_g,
            b: q.color_b,
        },
        RgbExtinction {
            r: q.ext_r,
            g: q.ext_g,
            b: q.ext_b,
        },
        q.depth,
    );
    WaterUnderwaterDepthColorResult {
        r: shifted.r,
        g: shifted.g,
        b: shifted.b,
    }
}

/// Pins one `GPU` result against the golden oracle within tolerance.
fn assert_result(
    idx: usize,
    got: &WaterUnderwaterDepthColorResult,
    want: &WaterUnderwaterDepthColorResult,
) {
    assert!(
        close(got.r, want.r),
        "result {idx} r: gpu {} vs cpu {}",
        got.r,
        want.r
    );
    assert!(
        close(got.g, want.g),
        "result {idx} g: gpu {} vs cpu {}",
        got.g,
        want.g
    );
    assert!(
        close(got.b, want.b),
        "result {idx} b: gpu {} vs cpu {}",
        got.b,
        want.b
    );
}

/// Runs every query on the device and pins each result against the oracle.
fn run_and_check(ctx: &GpuContext, queries: &[WaterUnderwaterDepthColorQuery]) {
    let gpu = GpuWaterUnderwaterDepthColor::new(ctx);
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
        eprintln!("skipping water_underwater_depth_color parity: no wgpu adapter on this host");
        return;
    };
    let gpu = GpuWaterUnderwaterDepthColor::new(&ctx);
    let got = gpu.evaluate(&ctx, &[]);
    assert!(got.is_empty(), "an empty batch produces no results");
}

#[test]
fn channel_ordering_reddest_fades_first() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    // White light at moderate depth with the usual r > g > b extinction: red
    // collapses first, then green, leaving the blue-green cast.
    let query = WaterUnderwaterDepthColorQuery {
        color_r: 1.0,
        color_g: 1.0,
        color_b: 1.0,
        ext_r: 0.6,
        ext_g: 0.3,
        ext_b: 0.1,
        depth: 4.0,
    };
    run_and_check(&ctx, &[query]);
    // Verify the ordering holds on the golden itself (and therefore the GPU,
    // which parity has already pinned to it).
    let want = expected(&query);
    assert!(
        want.r < want.g && want.g < want.b,
        "red must fade faster than green, green faster than blue: {want:?}"
    );
}

#[test]
fn depth_regimes() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let queries = vec![
        // Zero depth: transmittance is one, colour passes through unchanged.
        WaterUnderwaterDepthColorQuery {
            color_r: 0.8,
            color_g: 0.5,
            color_b: 0.2,
            ext_r: 0.7,
            ext_g: 0.3,
            ext_b: 0.1,
            depth: 0.0,
        },
        // Shallow depth: gentle attenuation.
        WaterUnderwaterDepthColorQuery {
            color_r: 0.9,
            color_g: 0.7,
            color_b: 0.6,
            ext_r: 0.5,
            ext_g: 0.25,
            ext_b: 0.08,
            depth: 1.5,
        },
        // Deep: red is nearly gone, blue persists.
        WaterUnderwaterDepthColorQuery {
            color_r: 1.0,
            color_g: 1.0,
            color_b: 1.0,
            ext_r: 0.8,
            ext_g: 0.4,
            ext_b: 0.12,
            depth: 8.0,
        },
        // Negative colour and extinction sanitize to zero via the floors.
        WaterUnderwaterDepthColorQuery {
            color_r: -0.5,
            color_g: 0.4,
            color_b: 0.3,
            ext_r: -0.2,
            ext_g: 0.3,
            ext_b: 0.1,
            depth: 2.0,
        },
    ];
    run_and_check(&ctx, &queries);
}

#[test]
fn random_batch_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let mut state = 0x51d3_7a0e_9c42_8bf6_u64;

    let mut queries: Vec<WaterUnderwaterDepthColorQuery> = Vec::new();
    while queries.len() < 256 {
        // Colours in [0, 1]; extinction in [0.02, 1]; depth in [0, 10]. All
        // stay clear of the sign flip at zero so neither floor straddles a tie.
        let query = WaterUnderwaterDepthColorQuery {
            color_r: ranged(&mut state, 0.0, 1.0),
            color_g: ranged(&mut state, 0.0, 1.0),
            color_b: ranged(&mut state, 0.0, 1.0),
            ext_r: ranged(&mut state, 0.02, 1.0),
            ext_g: ranged(&mut state, 0.02, 1.0),
            ext_b: ranged(&mut state, 0.02, 1.0),
            depth: ranged(&mut state, 0.0, 10.0),
        };
        queries.push(query);
    }
    run_and_check(&ctx, &queries);
}
