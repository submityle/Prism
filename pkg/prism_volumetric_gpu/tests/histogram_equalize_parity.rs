//! Real-device parity for the histogram-equalization twin:
//! [`GpuHistogramEqualize`](prism_volumetric_gpu::histogram_equalize::GpuHistogramEqualize)
//! must reproduce the two per-element pure maps of the `CPU` golden
//! [`histogram_equalize`](prism_render_architecture::particle::histogram_equalize)
//! — the `CDF`-entry to `LUT`-level map inside
//! [`equalization_lut`](prism_render_architecture::particle::histogram_equalize::equalization_lut)
//! and the sample to equalized-value map inside
//! [`apply_equalization`](prism_render_architecture::particle::histogram_equalize::apply_equalization).
//!
//! The host builds every reference input with the golden reductions
//! ([`build_histogram`](prism_render_architecture::particle::histogram_equalize::build_histogram)
//! and
//! [`cumulative_distribution`](prism_render_architecture::particle::histogram_equalize::cumulative_distribution));
//! those reductions are not twinned, only the two per-element maps are. The
//! fixtures cover the shapes the golden unit tests call out: a continuous
//! (`out_levels = 1`) sweep across a whole `CDF`, a quantized (`out_levels = 4`)
//! `CDF` whose entries are chosen so every `floor(raw * 3 + 0.5)` round sits
//! clear of a half-integer tie, a flat distribution whose degenerate denominator
//! yields an all-zero `LUT`, a sample sweep whose normalized positions are
//! reject-sampled away from a `floor` bin tie, a zero-width range, an empty
//! `LUT` pass-through, and a mixed batch exercising both tags at once.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! The integer stages (the saturated subtractions, the quantized level index
//! and the floored bin index) match the reference exactly; the surrounding
//! normalized levels and equalized values thread through a guarded divide, a
//! multiply and an add, so they are compared under tolerance (`abs_diff <= 1e-4`
//! or `rel_diff <= 1e-3`, `REL_FLOOR = 1e-6`). The fixtures keep every `floor`
//! argument clear of a tie, so a one-level or one-bin disagreement would exceed
//! the tolerance and fail the test.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::particle::histogram_equalize`；无第三方引擎源码或衍生代码。

use prism_render_architecture::particle::histogram_equalize::{
    apply_equalization, build_histogram, cumulative_distribution, equalization_lut,
};
use prism_volumetric_gpu::histogram_equalize::{GpuHistogramEqualize, HistogramEqualizeQuery};
use prism_volumetric_gpu::GpuContext;

/// Absolute tolerance for the continuous quantities.
const ABS_EPS: f32 = 1.0e-4;
/// Relative tolerance for the continuous quantities.
const REL_EPS: f32 = 1.0e-3;
/// Floor for the relative-tolerance denominator.
const REL_FLOOR: f32 = 1.0e-6;

/// Mixed absolute / relative tolerance comparison for one `f32` lane.
fn approx(a: f32, b: f32) -> bool {
    let diff = (a - b).abs();
    if diff <= ABS_EPS {
        return true;
    }
    let scale = a.abs().max(b.abs()).max(REL_FLOOR);
    diff / scale <= REL_EPS
}

/// Smallest non-zero `CDF` value, mirroring the reference `cdf_min` scan.
fn cdf_min_of(cdf: &[u32]) -> u32 {
    for &value in cdf {
        if value > 0 {
            return value;
        }
    }
    0
}

/// Builds one `LutLevel` query from a `CDF` entry and its shared scalars.
fn lut_level_query(cdf: u32, cdf_min: u32, total: u32, out_levels: u32) -> HistogramEqualizeQuery {
    HistogramEqualizeQuery::LutLevel {
        cdf,
        cdf_min,
        total,
        out_levels,
    }
}

/// A `u64` linear congruential generator (the `PCG` / `Numerical Recipes`
/// multiplier), used host-side so the fixtures need no transcendental math.
fn lcg_next(state: &mut u64) -> u64 {
    *state = state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    *state
}

/// Draws a `[0, 1)` `f32` from the top 16 bits of the next `LCG` output,
/// avoiding any lossy numeric cast via [`u16::try_from`] and [`f32::from`].
fn unit_f32(state: &mut u64) -> f32 {
    let raw = lcg_next(state) >> 48;
    let bits = u16::try_from(raw).unwrap_or(u16::MAX);
    f32::from(bits) / 65_536.0_f32
}

/// Converts a small bin count to `f32` without a lossy cast.
fn bins_as_f32(bins: usize) -> f32 {
    let narrowed = u16::try_from(bins).unwrap_or(u16::MAX);
    f32::from(narrowed)
}

/// Fractional part of `value`, used to keep a `floor` argument clear of a tie.
fn frac(value: f32) -> f32 {
    value - value.floor()
}

/// Asserts the `LUT`-level map for a whole `CDF` matches the reference.
fn assert_lut_level_parity(
    gpu: &GpuHistogramEqualize,
    ctx: &GpuContext,
    cdf: &[u32],
    total: u32,
    out_levels: u32,
) {
    let cdf_min = cdf_min_of(cdf);
    let queries: Vec<HistogramEqualizeQuery> = cdf
        .iter()
        .map(|&value| lut_level_query(value, cdf_min, total, out_levels))
        .collect();
    // The LUT is unused by LutLevel queries; pass it empty.
    let got = gpu.evaluate(ctx, &[], &queries);
    let want = equalization_lut(cdf, total, out_levels);
    assert_eq!(got.len(), want.len(), "one result per CDF entry");
    for (i, (g, w)) in got.iter().zip(want.iter()).enumerate() {
        assert!(
            approx(g.value, *w),
            "lut level mismatch at {i}: gpu {} vs cpu {w}",
            g.value
        );
    }
}

#[test]
fn lut_level_continuous_sweep_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuHistogramEqualize::new(&ctx);
    // A monotone CDF normalized continuously (out_levels = 1): the raw
    // (cdf - cdf_min) / (total - cdf_min) ratios are returned without rounding.
    let cdf = [0u32, 2, 5, 9, 10, 14, 20];
    assert_lut_level_parity(&gpu, &ctx, &cdf, 20, 1);
}

#[test]
fn lut_level_quantized_sweep_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuHistogramEqualize::new(&ctx);
    // cdf_min = 1, total = 9 -> denom = 8; the (cdf - 1) numerators 0,0,1,3,5,7,8
    // give raw eighths whose `raw * 3 + 0.5` rounds all sit clear of a half tie.
    let cdf = [0u32, 1, 2, 4, 6, 8, 9];
    assert_lut_level_parity(&gpu, &ctx, &cdf, 9, 4);
}

#[test]
fn lut_level_flat_distribution_is_all_zero() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuHistogramEqualize::new(&ctx);
    // total equals cdf_min, so the denominator degenerates and every level is 0.
    let cdf = [4u32, 4, 4];
    assert_lut_level_parity(&gpu, &ctx, &cdf, 4, 256);
}

#[test]
fn lut_level_from_real_histogram_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuHistogramEqualize::new(&ctx);
    // A CDF produced by the golden reductions, normalized continuously so no
    // quantization tie can split a level between the CPU and GPU divides.
    let samples = [0.05f32, 0.1, 0.1, 0.4, 0.42, 0.7, 0.95, 0.96, 0.3, 0.33];
    let hist = build_histogram(&samples, 8, 0.0, 1.0);
    let cdf = cumulative_distribution(&hist);
    let total = *cdf.last().unwrap_or(&0);
    assert_lut_level_parity(&gpu, &ctx, &cdf, total, 1);
}

#[test]
fn apply_sweep_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuHistogramEqualize::new(&ctx);
    // Build a real LUT, then sweep samples whose normalized position times the
    // bin count lands clear of a floor tie (fractional part in [0.15, 0.85]).
    let samples = [0.05f32, 0.2, 0.2, 0.5, 0.55, 0.8, 0.9, 0.92, 0.33, 0.6];
    let hist = build_histogram(&samples, 16, 0.0, 1.0);
    let cdf = cumulative_distribution(&hist);
    let total = *cdf.last().unwrap_or(&0);
    let lut = equalization_lut(&cdf, total, 256);
    let bins = bins_as_f32(lut.len());

    let range_min = -2.0f32;
    let range_max = 5.0f32;
    let span = range_max - range_min;

    let mut state: u64 = 0x1234_5678_9abc_def0;
    let mut queries: Vec<HistogramEqualizeQuery> = Vec::new();
    while queries.len() < 64 {
        let unit = unit_f32(&mut state);
        // Reject positions whose scaled fractional part sits near a bin tie.
        let scaled_frac = frac(unit * bins);
        if !(0.15..=0.85).contains(&scaled_frac) {
            continue;
        }
        let sample = range_min + unit * span;
        queries.push(HistogramEqualizeQuery::Apply {
            sample,
            range_min,
            range_max,
        });
    }

    let got = gpu.evaluate(&ctx, &lut, &queries);
    assert_eq!(got.len(), queries.len(), "one result per sample");
    for (g, q) in got.iter().zip(queries.iter()) {
        let HistogramEqualizeQuery::Apply {
            sample,
            range_min,
            range_max,
        } = *q
        else {
            unreachable!("sweep contains only apply queries");
        };
        let want = apply_equalization(sample, &lut, range_min, range_max);
        assert!(
            approx(g.value, want),
            "apply mismatch for sample {sample}: gpu {} vs cpu {want}",
            g.value
        );
    }
}

#[test]
fn apply_degenerate_range_returns_min() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuHistogramEqualize::new(&ctx);
    // A zero-width range returns range_min, matching the reference.
    let lut = [0.0f32, 0.5, 1.0];
    let q = HistogramEqualizeQuery::Apply {
        sample: 7.0,
        range_min: 2.0,
        range_max: 2.0,
    };
    let got = gpu.evaluate(&ctx, &lut, std::slice::from_ref(&q));
    let want = apply_equalization(7.0, &lut, 2.0, 2.0);
    assert_eq!(got.len(), 1);
    assert!(
        approx(got[0].value, want),
        "gpu {} vs cpu {want}",
        got[0].value
    );
}

#[test]
fn apply_empty_lut_is_passthrough() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuHistogramEqualize::new(&ctx);
    // An empty LUT returns the sample unchanged, matching the reference.
    let q = HistogramEqualizeQuery::Apply {
        sample: 0.42,
        range_min: 0.0,
        range_max: 1.0,
    };
    let got = gpu.evaluate(&ctx, &[], std::slice::from_ref(&q));
    let want = apply_equalization(0.42, &[], 0.0, 1.0);
    assert_eq!(got.len(), 1);
    assert!(
        approx(got[0].value, want),
        "gpu {} vs cpu {want}",
        got[0].value
    );
}

#[test]
fn mixed_batch_matches_elementwise() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuHistogramEqualize::new(&ctx);
    // One batch mixes both tags against a shared LUT: the LutLevel queries ignore
    // the LUT, the Apply queries read it, and each result is independent.
    let cdf = [0u32, 1, 2, 4, 6, 8, 9];
    let cdf_min = cdf_min_of(&cdf);
    let total = 9u32;
    let out_levels = 4u32;
    let lut = equalization_lut(&cdf, total, 256);

    let range_min = 0.0f32;
    let range_max = 1.0f32;
    let queries = [
        lut_level_query(cdf[2], cdf_min, total, out_levels),
        HistogramEqualizeQuery::Apply {
            sample: 0.23,
            range_min,
            range_max,
        },
        lut_level_query(cdf[5], cdf_min, total, out_levels),
        HistogramEqualizeQuery::Apply {
            sample: 0.77,
            range_min,
            range_max,
        },
    ];

    let got = gpu.evaluate(&ctx, &lut, &queries);
    assert_eq!(got.len(), queries.len());

    let level_lut = equalization_lut(&cdf, total, out_levels);
    assert!(approx(got[0].value, level_lut[2]));
    assert!(approx(
        got[1].value,
        apply_equalization(0.23, &lut, range_min, range_max)
    ));
    assert!(approx(got[2].value, level_lut[5]));
    assert!(approx(
        got[3].value,
        apply_equalization(0.77, &lut, range_min, range_max)
    ));
}

#[test]
fn empty_batch_short_circuits() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuHistogramEqualize::new(&ctx);
    // No dispatch is issued and the result vector is empty.
    assert!(gpu.evaluate(&ctx, &[], &[]).is_empty());
}
