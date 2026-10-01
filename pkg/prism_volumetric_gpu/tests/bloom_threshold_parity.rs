//! Real-device parity for the `bloom` bright-pass twin:
//! [`GpuBloomThreshold`](prism_volumetric_gpu::bloom_threshold::GpuBloomThreshold)
//! must reproduce the `CPU` golden
//! [`threshold_batch`](prism_render_architecture::particle::bloom_threshold::BloomThresholdParams::threshold_batch)
//! across an empty batch, a single color, dim colors below the `knee` floor,
//! colors inside the soft-`knee` transition band, very bright colors far above
//! the threshold, a hard (`knee == 0`) cutoff and a batch of random `HDR`
//! colors compared color-for-color and channel-for-channel.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! Each output color is a fixed, non-reorderable sequence of multiplies, adds,
//! one `clamp`, two `max` and two guarded divides, so `CPU` and `GPU` evaluate
//! the same closed form in the same order. They are not bit-exact: a `GPU` may
//! fuse a multiply-add the scalar reference leaves separate, perturbing the low
//! mantissa bits by a few units in the last place. The comparison therefore
//! allows `abs_diff <= 1e-4` or `rel_diff <= 1e-3` — loose enough to admit a
//! legal fused multiply-add contraction, yet tight enough to fail a genuinely
//! wrong port (a swapped `luminance` weight, a dropped `knee` term, a missing
//! `intensity` gain, a lost hue factor).
//!
//! # Degenerate regions
//!
//! An all-zero batch is both trivially correct and vacuous (`0` vs `0`), so the
//! random fixtures draw `HDR` colors whose `luminance` straddles the threshold
//! band — some below the `knee` floor, some inside the band, some far above —
//! so the soft-`knee` curve, the hue-preserving contribution and the
//! `intensity` gain are all exercised with non-trivial values. The all-black
//! input is checked separately only to pin the guarded denominators to a
//! finite result.
//!
//! Provenance: standard `bloom` bright-pass prefilter with a soft `knee`
//! (`Jimenez`, `SIGGRAPH` 2014); mirrors the `CPU` golden
//! `prism_render_architecture::particle::bloom_threshold`; no third-party
//! engine source or derived code.

use prism_render_architecture::particle::bloom_threshold::BloomThresholdParams;
use prism_volumetric_gpu::bloom_threshold::{BloomThresholdQuery, GpuBloomThreshold};
use prism_volumetric_gpu::GpuContext;

/// Absolute parity bound. A `GPU` may fuse a multiply-add the scalar reference
/// leaves separate, perturbing the low mantissa bits by a few units in the last
/// place; `1e-4` admits that legal slack while still failing a genuinely wrong
/// port.
const EPS: f32 = 1.0e-4;

/// Relative parity bound, applied for larger magnitudes where a few units in
/// the last place exceed the absolute floor.
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

/// A tiny integer linear-congruential generator; only integer and divide work,
/// so no transcendental appears. Returns a value in `[0, 1)`.
fn lcg(state: &mut u64) -> f32 {
    *state = state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    let bits = (*state >> 40) as u32;
    (bits & 0x00ff_ffff) as f32 / 16_777_216.0
}

/// Builds `count` pseudo-random `HDR` colors in `[0, scale)` drawn from `state`.
fn random_colors(count: usize, scale: f32, state: &mut u64) -> Vec<[f32; 3]> {
    let mut colors = Vec::with_capacity(count);
    for _ in 0..count {
        colors.push([lcg(state) * scale, lcg(state) * scale, lcg(state) * scale]);
    }
    colors
}

/// Runs the `GPU` bright-pass and asserts color-for-color parity against the
/// `CPU` golden [`BloomThresholdParams::threshold_batch`], returning the `GPU`
/// result for any extra per-test assertions.
fn check(
    ctx: &GpuContext,
    gpu: &GpuBloomThreshold,
    colors: &[[f32; 3]],
    params: BloomThresholdParams,
) -> Vec<[f32; 3]> {
    let query = BloomThresholdQuery {
        colors: colors.to_vec(),
        params,
    };
    let got = gpu.eval(ctx, &query);
    let want = params.threshold_batch(colors);

    assert_eq!(
        got.len(),
        want.len(),
        "color count must match the reference"
    );
    for (idx, (g, w)) in got.iter().zip(want.iter()).enumerate() {
        for channel in 0..3 {
            assert!(
                close(g[channel], w[channel]),
                "color {idx} channel {channel}: gpu {} vs cpu {} \
                 (threshold {}, knee {}, intensity {})",
                g[channel],
                w[channel],
                params.threshold,
                params.knee,
                params.intensity,
            );
        }
    }
    got
}

#[test]
fn empty_batch_round_trips() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuBloomThreshold::new(&ctx);
    // An empty batch returns an empty vector and issues no dispatch, exactly as
    // the reference does.
    let got = check(&ctx, &gpu, &[], BloomThresholdParams::new(1.0, 0.5, 1.0));
    assert!(got.is_empty(), "an empty batch stays empty");
}

#[test]
fn single_color_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuBloomThreshold::new(&ctx);
    // A single bright color exercises the one-thread dispatch end to end.
    let colors = [[2.0, 1.5, 0.5]];
    check(
        &ctx,
        &gpu,
        &colors,
        BloomThresholdParams::new(1.0, 0.5, 1.0),
    );
}

#[test]
fn dim_colors_extract_no_bloom() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuBloomThreshold::new(&ctx);
    let params = BloomThresholdParams::new(1.0, 0.5, 1.0);
    // A grey whose luminance sits well below (threshold - knee); the twin must
    // match the reference's exact black and not leak a NaN from the divide.
    let colors = [[0.2, 0.2, 0.2], [0.1, 0.05, 0.0]];
    let got = check(&ctx, &gpu, &colors, params);
    for color in &got {
        for &c in color {
            assert!(close(c, 0.0), "a dim color extracts no bloom, got {c}");
            assert!(c.is_finite(), "the guarded divide stays finite");
        }
    }
}

#[test]
fn knee_band_colors_match_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuBloomThreshold::new(&ctx);
    let params = BloomThresholdParams::new(1.0, 0.5, 1.0);
    // Grey colors whose luminance lands inside the soft-knee band
    // [threshold - knee, threshold + knee] = [0.5, 1.5], so the quadratic
    // rational branch (not the hard floor) decides the response.
    let colors = [
        [0.6, 0.6, 0.6],
        [0.8, 0.8, 0.8],
        [1.0, 1.0, 1.0],
        [1.3, 1.3, 1.3],
    ];
    let got = check(&ctx, &gpu, &colors, params);
    // The band response is strictly positive and preserves the grey hue.
    for color in &got {
        assert!(color[0] > 0.0, "a band color contributes some bloom");
        assert!(close(color[0], color[1]) && close(color[1], color[2]));
    }
}

#[test]
fn bright_colors_preserve_hue() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuBloomThreshold::new(&ctx);
    let params = BloomThresholdParams::new(1.0, 0.5, 1.0);
    // A very bright color: contribution -> 1, so the extracted color reconverges
    // on the input while preserving the channel ratios.
    let colors = [[400.0, 300.0, 200.0]];
    let got = check(&ctx, &gpu, &colors, params);
    let out = got[0];
    // Hue preserved: out is a non-negative scalar multiple of the input.
    assert!(close(out[0] * 300.0, out[1] * 400.0));
    assert!(close(out[1] * 200.0, out[2] * 300.0));
    // Each channel lands within ~1% of the original far above the threshold.
    for (o, i) in out.iter().zip([400.0, 300.0, 200.0].iter()) {
        assert!(
            (o - i).abs() < i * 0.02,
            "bright color reconverges: {o} vs {i}"
        );
    }
}

#[test]
fn intensity_scales_linearly() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuBloomThreshold::new(&ctx);
    let colors = [[3.0, 2.0, 1.0], [2.0, 1.0, 0.5]];
    let base = check(
        &ctx,
        &gpu,
        &colors,
        BloomThresholdParams::new(1.0, 0.5, 1.0),
    );
    let doubled = check(
        &ctx,
        &gpu,
        &colors,
        BloomThresholdParams::new(1.0, 0.5, 2.0),
    );
    // Doubling the intensity doubles every extracted channel.
    for (b, d) in base.iter().zip(doubled.iter()) {
        for channel in 0..3 {
            assert!(close(d[channel], b[channel] * 2.0));
        }
    }
}

#[test]
fn zero_knee_hard_cutoff_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuBloomThreshold::new(&ctx);
    // A hard cutoff (knee == 0) stresses the guarded `4 knee + eps` denominator:
    // the GPU must reproduce the reference's exact zero just below the threshold
    // and its positive response above it without dividing by zero.
    let params = BloomThresholdParams::new(1.0, 0.0, 1.0);
    let colors = [[0.9, 0.9, 0.9], [1.5, 1.5, 1.5], [3.0, 2.0, 1.0]];
    let got = check(&ctx, &gpu, &colors, params);
    assert!(close(got[0][0], 0.0), "below the hard cutoff stays black");
    assert!(got[1][0] > 0.0, "above the hard cutoff contributes bloom");
}

#[test]
fn black_input_stays_finite() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuBloomThreshold::new(&ctx);
    // Pure black drives the luminance to zero; the guarded `max(lum, eps)`
    // denominator must keep the result at an exact, finite black, not a NaN.
    let colors = [[0.0, 0.0, 0.0]];
    let got = check(
        &ctx,
        &gpu,
        &colors,
        BloomThresholdParams::new(1.0, 0.5, 1.0),
    );
    for &c in &got[0] {
        assert!(close(c, 0.0) && c.is_finite(), "black stays a finite black");
    }
}

#[test]
fn random_colors_match_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuBloomThreshold::new(&ctx);
    let mut state = 0x5eed_b100_0ddf_11a7_u64;
    // Sweep several batch sizes and parameter sets with HDR colors whose
    // luminance straddles the threshold band (scale 4.0 against threshold ~1.0),
    // so each batch spans the below-floor, in-band and far-above regions.
    let params_sweep = [
        BloomThresholdParams::new(1.0, 0.5, 1.0),
        BloomThresholdParams::new(0.8, 0.25, 1.5),
        BloomThresholdParams::new(1.5, 0.75, 0.5),
        BloomThresholdParams::new(1.0, 0.0, 2.0),
    ];
    for count in [1usize, 2, 7, 64, 65, 200] {
        for params in params_sweep {
            let colors = random_colors(count, 4.0, &mut state);
            check(&ctx, &gpu, &colors, params);
        }
    }
}
