//! Real-device parity for the `luminance`-`histogram` twin:
//! [`GpuLuminanceHist`](prism_volumetric_gpu::luminance_hist::GpuLuminanceHist)
//! must reproduce the `CPU` golden
//! [`build_histogram`](prism_render_architecture::particle::luminance_hist::build_histogram)
//! across an empty batch (the all-zero histogram), a single sample, batches
//! that collapse entirely into the darkest or brightest bucket, a batch spread
//! one-per-bucket across the whole axis, a large pseudo-random batch, and a
//! batch of exact powers of two (for which `approx_log2` is exact).
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! A histogram bucket count is a `u32` integer, so parity is asserted with
//! **exact per-bucket equality**, not a float tolerance — exactly the check
//! that catches an `approx_log2` ported one bit wrong (an off-by-one bucket).
//! The only place `CPU` and `GPU` can legally diverge is a sample whose `EV`
//! sits within a `ULP` of a bucket boundary, where a fused multiply-add in the
//! mantissa quadratic may floor it to the neighbouring bucket. The fixtures
//! therefore keep every sample's `EV` well inside a bucket (at least `0.2 EV`
//! from a boundary with the chosen wide buckets) or use exact powers of two
//! (whose mantissa fraction, and thus the quadratic term, is identically zero),
//! so the exact-equality check stays meaningful rather than flaky.
//!
//! Provenance: standard `luminance`-`histogram` auto-exposure; no third-party
//! engine source or derived code.

use prism_render_architecture::particle::luminance_hist::{build_histogram, LumHistConfig};
use prism_volumetric_gpu::luminance_hist::{GpuLuminanceHist, LuminanceHistQuery};
use prism_volumetric_gpu::GpuContext;

/// A tiny integer linear-congruential generator; only integer and divide work,
/// so no transcendental appears. Returns a value in `[0, 1)`.
fn lcg(state: &mut u64) -> f32 {
    *state = state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    let bits = (*state >> 40) as u32;
    (bits & 0x00ff_ffff) as f32 / 16_777_216.0
}

/// Builds a `luminance` whose `EV` lands at `fpos in [0, 1)` of bucket `bin`
/// across `cfg`'s axis. Keeping `fpos` in `[0.2, 0.8]` holds the sample well
/// away from a bucket boundary, so a last-`ULP` fused multiply-add in the
/// mantissa quadratic can never floor it into a neighbouring bucket.
fn luminance_in_bucket(cfg: &LumHistConfig, bin: u32, fpos: f64) -> f32 {
    let span = f64::from(cfg.max_ev - cfg.min_ev);
    let bins = f64::from(cfg.effective_bins());
    let target_ev = f64::from(cfg.min_ev) + (f64::from(bin) + fpos) / bins * span;
    // `exp2` only appears in the host-side fixture builder, never in the twin.
    2.0f64.powf(target_ev) as f32
}

/// Runs the `GPU` histogram and asserts exact per-bucket parity against the
/// `CPU` golden [`build_histogram`], returning the counts for extra assertions.
fn check(
    ctx: &GpuContext,
    gpu: &GpuLuminanceHist,
    luminances: &[f32],
    cfg: &LumHistConfig,
) -> Vec<u32> {
    let query = LuminanceHistQuery {
        luminances: luminances.to_vec(),
        config: *cfg,
    };
    let got = gpu.eval(ctx, &query);
    let want = build_histogram(luminances, cfg);

    assert_eq!(
        got.len(),
        want.len(),
        "histogram length must match effective_bins"
    );
    for (bucket, (g, w)) in got.iter().zip(want.iter()).enumerate() {
        assert_eq!(
            g, w,
            "bucket {bucket}: gpu count {g} vs cpu count {w} (bins {})",
            cfg.bins
        );
    }
    got
}

#[test]
fn empty_input_is_the_zero_histogram() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuLuminanceHist::new(&ctx);
    let cfg = LumHistConfig::new(-8.0, 8.0, 16, 0.0, 0.0);
    // No sample issues no dispatch; the result is the correctly sized zero
    // histogram and matches the reference exactly.
    let got = check(&ctx, &gpu, &[], &cfg);
    assert_eq!(got.len(), 16, "effective_bins length is preserved");
    assert!(got.iter().all(|&c| c == 0), "every bucket is empty");
}

#[test]
fn zero_bins_still_yields_one_bucket() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuLuminanceHist::new(&ctx);
    // `bins == 0` clamps up to one effective bucket everywhere it is used, so
    // every sample lands in bucket 0.
    let cfg = LumHistConfig::new(-8.0, 8.0, 0, 0.0, 0.0);
    let luminances = [1.0f32, 2.0, 0.5];
    let got = check(&ctx, &gpu, &luminances, &cfg);
    assert_eq!(got.len(), 1, "zero bins clamps up to one bucket");
    assert_eq!(got[0], 3, "all samples fall in the single bucket");
}

#[test]
fn single_sample_lands_in_one_bucket() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuLuminanceHist::new(&ctx);
    let cfg = LumHistConfig::new(-8.0, 8.0, 16, 0.0, 0.0);
    // Luminance 1.0 is EV 0 (an exact power of two), so both paths place it in
    // the identical bucket; parity plus a total of one pins the single count.
    let got = check(&ctx, &gpu, &[1.0], &cfg);
    assert_eq!(got.iter().sum::<u32>(), 1, "exactly one sample is counted");
}

#[test]
fn all_below_min_ev_collapse_to_bucket_zero() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuLuminanceHist::new(&ctx);
    let cfg = LumHistConfig::new(-8.0, 8.0, 16, 0.0, 0.0);
    // All far below min_ev (EV -8): tiny positives, a subnormal-ish value and a
    // zero (which `approx_log2` floors to the darkest sentinel). Every one
    // clamps to bucket 0.
    let luminances = [1.0e-10f32, 1.0e-20, 1.0e-30, 0.0, 1.0e-15];
    let got = check(&ctx, &gpu, &luminances, &cfg);
    assert_eq!(got[0], 5, "every dark sample lands in bucket 0");
    assert!(
        got.iter().skip(1).all(|&c| c == 0),
        "no sample escapes the darkest bucket"
    );
}

#[test]
fn all_above_max_ev_collapse_to_the_last_bucket() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuLuminanceHist::new(&ctx);
    let cfg = LumHistConfig::new(-8.0, 8.0, 16, 0.0, 0.0);
    // All far above max_ev (EV 8): large powers of two and large non-powers.
    // Every one clamps to the final bucket.
    let luminances = [1.0e10f32, 65_536.0, 1.0e6, 262_144.0, 1.0e8];
    let got = check(&ctx, &gpu, &luminances, &cfg);
    let last = got.len() - 1;
    assert_eq!(got[last], 5, "every bright sample lands in the last bucket");
    assert!(
        got.iter().take(last).all(|&c| c == 0),
        "no sample escapes the brightest bucket"
    );
}

#[test]
fn one_sample_per_bucket_fills_the_axis() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuLuminanceHist::new(&ctx);
    // Wide 2-EV buckets ([-8, 8] / 8) leave a 0.2-EV margin trivially, so each
    // mid-bucket sample is unambiguous on both paths.
    let cfg = LumHistConfig::new(-8.0, 8.0, 8, 0.0, 0.0);
    let bins = cfg.effective_bins();
    let luminances: Vec<f32> = (0..bins)
        .map(|bin| luminance_in_bucket(&cfg, bin, 0.5))
        .collect();
    let got = check(&ctx, &gpu, &luminances, &cfg);
    assert!(
        got.iter().all(|&c| c == 1),
        "each bucket receives exactly one sample: {got:?}"
    );
}

#[test]
fn large_random_batch_conserves_totals() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuLuminanceHist::new(&ctx);
    // Wide 2-EV buckets keep every random sample at least 0.2 EV from a
    // boundary (fpos is clamped into [0.2, 0.8]), so the exact-equality parity
    // is robust against a last-ULP fused multiply-add.
    let cfg = LumHistConfig::new(-8.0, 8.0, 8, 0.0, 0.0);
    let bins = cfg.effective_bins();
    let mut state = 0x1357_9bdf_0246_8ace_u64;
    let mut luminances = Vec::with_capacity(4096);
    for _ in 0..4096 {
        let bin = (lcg(&mut state) * bins as f32) as u32 % bins;
        let fpos = 0.2 + 0.6 * f64::from(lcg(&mut state));
        luminances.push(luminance_in_bucket(&cfg, bin, fpos));
    }
    let got = check(&ctx, &gpu, &luminances, &cfg);
    assert_eq!(
        got.iter().map(|&c| u64::from(c)).sum::<u64>(),
        luminances.len() as u64,
        "every sample is counted exactly once"
    );
    assert!(
        got.iter().any(|&c| c > 0),
        "the histogram is not vacuously empty"
    );
}

#[test]
fn powers_of_two_bin_exactly() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuLuminanceHist::new(&ctx);
    // Powers of two have mantissa fraction 0, so `approx_log2` returns the
    // integer EV with no quadratic term and no rounding: CPU and GPU agree
    // bit-for-bit even though the EVs land on bucket boundaries.
    let cfg = LumHistConfig::new(-8.0, 8.0, 16, 0.0, 0.0);
    let luminances = [
        0.015_625f32, // 2^-6
        0.0625,       // 2^-4
        0.25,         // 2^-2
        1.0,          // 2^0
        4.0,          // 2^2
        16.0,         // 2^4
        64.0,         // 2^6
        1.0,          // repeat 2^0 so a bucket gets two
    ];
    let got = check(&ctx, &gpu, &luminances, &cfg);
    assert_eq!(
        got.iter().map(|&c| u64::from(c)).sum::<u64>(),
        luminances.len() as u64,
        "every power-of-two sample is counted"
    );
}

#[test]
fn mixed_batch_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuLuminanceHist::new(&ctx);
    // A spread of configs (bin counts and asymmetric ranges) each driven by a
    // mid-bucket sweep, so the normalize/floor/clamp path is exercised at
    // several axis layouts, each compared bucket-for-bucket.
    let configs = [
        LumHistConfig::new(-8.0, 8.0, 4, 0.0, 0.0),
        LumHistConfig::new(-6.0, 10.0, 8, 0.0, 0.0),
        LumHistConfig::new(-10.0, 6.0, 16, 0.0, 0.0),
    ];
    let mut state = 0x5eed_4a7d_0bad_c0de_u64;
    for cfg in &configs {
        let bins = cfg.effective_bins();
        let mut luminances = Vec::with_capacity(256);
        for _ in 0..256 {
            let bin = (lcg(&mut state) * bins as f32) as u32 % bins;
            let fpos = 0.25 + 0.5 * f64::from(lcg(&mut state));
            luminances.push(luminance_in_bucket(cfg, bin, fpos));
        }
        let got = check(&ctx, &gpu, &luminances, cfg);
        assert_eq!(
            got.iter().map(|&c| u64::from(c)).sum::<u64>(),
            luminances.len() as u64,
            "totals are conserved for bins {}",
            cfg.bins
        );
    }
}
