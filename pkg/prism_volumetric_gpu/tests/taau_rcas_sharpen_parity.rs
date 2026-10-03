//! Real-device parity for the `RCAS` sharpening per-pixel twin:
//! [`GpuTaauRcasSharpen`](prism_volumetric_gpu::taau_rcas_sharpen::GpuTaauRcasSharpen)
//! must reproduce the scalar core of the `CPU` golden
//! [`rcas`](prism_render_architecture::temporal_upscale::sharpen::rcas) — the
//! contrast-limited `5`-tap cross sharpen of one pixel — across the identity
//! cases, full-strength overshoot/undershoot, the maximum-contrast limiter, the
//! `denoise` attenuation, and a randomized `LCG` batch compared pixel-for-pixel.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Oracle
//!
//! The golden
//! [`rcas`](prism_render_architecture::temporal_upscale::sharpen::rcas) is
//! itself the public entry point, so each fixture is run through it directly to
//! produce the expected sharpened color, then compared against the `GPU` output
//! for the identical [`CrossTaps`], `sharpness` and `denoise`.
//!
//! # Parity criterion
//!
//! Every output channel threads through subtracts, reciprocals and multiplies,
//! so a `GPU` reciprocal may land a few units in the last place from the scalar
//! reference; each channel is asserted within `abs_diff <= 1e-4` or
//! `rel_diff <= 1e-3`.
//!
//! # Conditioning
//!
//! Fixtures keep the ring contrast and the `denoise` luma range comfortably
//! away from the degenerate flat-ring and zero-range branches, so the `CPU` and
//! `GPU` stay on the same side of every guard. No fixture uses an `f32`
//! transcendental method; the random batch is drawn from a host-side integer
//! `LCG`.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::temporal_upscale::sharpen`；无第三方引擎源码或衍生代码。

use prism_render_architecture::temporal_upscale::sharpen::{rcas, CrossTaps};
use prism_volumetric_gpu::taau_rcas_sharpen::{
    GpuTaauRcasSharpen, TaauRcasSharpenQuery, TaauRcasSharpenResult,
};
use prism_volumetric_gpu::GpuContext;

/// Absolute parity bound on an output channel. A `GPU` reciprocal may land a few
/// units in the last place from the scalar reference; `1e-4` admits that legal
/// slack while still failing a wrong port.
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

/// Builds the golden [`CrossTaps`] matching a twin query's five taps.
fn taps_of(q: &TaauRcasSharpenQuery) -> CrossTaps {
    CrossTaps {
        center: q.center,
        north: q.north,
        south: q.south,
        west: q.west,
        east: q.east,
    }
}

/// Runs the golden on one query to produce the expected sharpened color.
fn oracle(q: &TaauRcasSharpenQuery) -> [f32; 3] {
    rcas(&taps_of(q), q.sharpness, q.denoise)
}

/// Pins one `GPU` pixel result against the golden, channel-by-channel.
fn check_pixel(idx: usize, got: &TaauRcasSharpenResult, want: [f32; 3]) {
    for (c, &target) in want.iter().enumerate() {
        assert!(
            close(got.color[c], target),
            "pixel {idx} channel {c}: gpu {} vs cpu {}",
            got.color[c],
            target
        );
    }
}

/// Dispatches every query and pins each result against the golden.
fn check(ctx: &GpuContext, gpu: &GpuTaauRcasSharpen, queries: &[TaauRcasSharpenQuery]) {
    let got = gpu.evaluate(ctx, queries);
    assert_eq!(
        got.len(),
        queries.len(),
        "result count must match the input count"
    );
    for (idx, (q, result)) in queries.iter().zip(got.iter()).enumerate() {
        check_pixel(idx, result, oracle(q));
    }
}

/// A cross with a uniform ring and a given center, mirroring the golden's own
/// test helper.
fn cross(center: [f32; 3], ring: [f32; 3], sharpness: f32, denoise: bool) -> TaauRcasSharpenQuery {
    TaauRcasSharpenQuery::new(center, ring, ring, ring, ring, sharpness, denoise)
}

/// A tiny integer linear-congruential generator; only integer work, so no
/// transcendental appears. Returns the raw high bits as a `u32`.
fn lcg(state: &mut u64) -> u32 {
    *state = state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    (*state >> 33) as u32
}

/// Draws a channel value in `[0.0, 0.98]` at milli resolution from `state`.
fn channel(state: &mut u64) -> f32 {
    (lcg(state) % 981) as f32 / 1000.0
}

/// Draws a color triple from `state`.
fn color(state: &mut u64) -> [f32; 3] {
    [channel(state), channel(state), channel(state)]
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn empty_input_returns_empty_without_dispatch() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping taau_rcas_sharpen parity: no wgpu adapter on this host");
        return;
    };
    let gpu = GpuTaauRcasSharpen::new(&ctx);
    // An empty batch short-circuits before any dispatch (a storage buffer cannot
    // be zero-sized) and returns an empty vector.
    let got = gpu.evaluate(&ctx, &[]);
    assert!(got.is_empty(), "an empty batch produces no results");
}

#[test]
fn zero_sharpness_is_identity() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuTaauRcasSharpen::new(&ctx);
    // sharpness = 0 returns center unchanged regardless of the ring contrast.
    let q = cross([0.7, 0.3, 0.5], [0.2, 0.1, 0.4], 0.0, false);
    let got = gpu.evaluate(&ctx, &[q]);
    assert_eq!(got.len(), 1);
    check_pixel(0, &got[0], q.center);
    // And it equals the golden, which also returns center.
    check_pixel(0, &got[0], oracle(&q));
}

#[test]
fn flat_neighborhood_is_identity_at_full_strength() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuTaauRcasSharpen::new(&ctx);
    // Every tap equal: zero contrast, zero lobe, exact passthrough even at full
    // sharpness and with denoise on.
    let flat = cross([0.42, 0.42, 0.42], [0.42, 0.42, 0.42], 1.0, true);
    check(&ctx, &gpu, &[flat]);
}

#[test]
fn full_strength_overshoot_and_undershoot_match_golden() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuTaauRcasSharpen::new(&ctx);
    // Bright center on a darker ring (overshoot) and dark center on a brighter
    // ring (undershoot), both at full sharpness with denoise off.
    let bright = cross([0.6, 0.6, 0.6], [0.2, 0.2, 0.2], 1.0, false);
    let dark = cross([0.3, 0.3, 0.3], [0.8, 0.8, 0.8], 1.0, false);
    check(&ctx, &gpu, &[bright, dark]);
}

#[test]
fn denoise_on_and_off_match_golden() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuTaauRcasSharpen::new(&ctx);
    // A bright luma spike on a flat darker ring exercises the denoise term; both
    // the plain and the denoised variants must track the golden.
    let plain = cross([0.9, 0.9, 0.9], [0.1, 0.1, 0.1], 1.0, false);
    let denoised = cross([0.9, 0.9, 0.9], [0.1, 0.1, 0.1], 1.0, true);
    check(&ctx, &gpu, &[plain, denoised]);
}

#[test]
fn max_contrast_edge_matches_golden() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuTaauRcasSharpen::new(&ctx);
    // Maximum-contrast ring (0 vs near-peak) exercises the lobe limiter; the
    // output must stay finite and track the golden.
    let edge = TaauRcasSharpenQuery::new(
        [0.5, 0.5, 0.5],
        [0.99, 0.99, 0.99],
        [0.0, 0.0, 0.0],
        [0.99, 0.0, 0.0],
        [0.0, 0.99, 0.0],
        1.0,
        false,
    );
    let got = gpu.evaluate(&ctx, &[edge]);
    assert_eq!(got.len(), 1);
    assert!(
        got[0].color.iter().all(|c| c.is_finite()),
        "non-finite out {:?}",
        got[0].color
    );
    check_pixel(0, &got[0], oracle(&edge));
}

#[test]
fn sharpness_sweep_matches_golden() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuTaauRcasSharpen::new(&ctx);
    // A fixed contrast cross swept across intermediate sharpness knobs.
    let mut queries = Vec::new();
    for step in 0..=10u32 {
        let s = step as f32 / 10.0;
        queries.push(cross([0.6, 0.55, 0.5], [0.2, 0.25, 0.3], s, false));
        queries.push(cross([0.6, 0.55, 0.5], [0.2, 0.25, 0.3], s, true));
    }
    check(&ctx, &gpu, &queries);
}

#[test]
fn random_batch_matches_golden() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuTaauRcasSharpen::new(&ctx);
    let mut state = 0x0f1e_2d3c_4b5a_6978_u64;
    // Many random pixels (several workgroups' worth) over a wide span of tap
    // colors and both denoise modes.
    let mut queries = Vec::new();
    for i in 0..300 {
        let center = color(&mut state);
        let north = color(&mut state);
        let south = color(&mut state);
        let west = color(&mut state);
        let east = color(&mut state);
        let sharpness = (lcg(&mut state) % 1001) as f32 / 1000.0;
        let denoise = i % 2 == 0;
        queries.push(TaauRcasSharpenQuery::new(
            center, north, south, west, east, sharpness, denoise,
        ));
    }
    check(&ctx, &gpu, &queries);
}
