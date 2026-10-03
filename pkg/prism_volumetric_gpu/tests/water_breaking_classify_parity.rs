//! Real-device parity for the breaking-wave classifier numeric-core twin:
//! [`GpuWaterBreakingClassify`](prism_volumetric_gpu::water_breaking_classify::GpuWaterBreakingClassify)
//! must reproduce the `CPU` golden
//! [`breaking`](prism_render_architecture::water::breaking) closed forms — the
//! normalized intensity, the discrete class, and the foam-source strength —
//! across calm, cresting, folded, high-intensity, zero-rate and randomized
//! fixtures.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Oracle
//!
//! The golden
//! [`breaking_intensity`](prism_render_architecture::water::breaking::breaking_intensity),
//! [`classify_breaking`](prism_render_architecture::water::breaking::classify_breaking)
//! and
//! [`foam_source_strength`](prism_render_architecture::water::breaking::foam_source_strength)
//! are public, so they are called directly as the oracle: each `GPU` result is
//! pinned against the golden evaluated on the identical sample and criteria.
//!
//! # Parity criterion
//!
//! The `class` is a discrete decision and is asserted exactly (`==`). The
//! continuous `intensity` and `foam` thread through subtracts, divides, `clamp`
//! and one multiply, so a `GPU` divide may land a few units in the last place
//! from the scalar reference; they are asserted within `abs_diff <= 1e-4` or
//! `rel_diff <= 1e-3`.
//!
//! # Conditioning
//!
//! The only class comparison whose two sides are *computed* is
//! `intensity >= breaking_intensity`; fixtures and the randomized sweep keep
//! that margin well clear of zero (replicating the golden intensity host-side)
//! so `CPU` and `GPU` stay on the same side of it. The `jacobian <= fold` and
//! `steepness > steepness` tests compare input-identical operands, so they
//! cannot diverge, but the sweep still keeps a clear margin off each for good
//! measure. The randomized sweep draws from a host-side integer generator, with
//! no transcendental method, matching the house rules.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::water::breaking`；无第三方引擎源码或衍生代码。

use prism_render_architecture::water::breaking::{
    breaking_intensity, classify_breaking, foam_source_strength, BreakingClass, BreakingCriteria,
    BreakingSample,
};
use prism_volumetric_gpu::water_breaking_classify::{
    GpuWaterBreakingClassify, WaterBreakingClass, WaterBreakingClassifyQuery,
    WaterBreakingClassifyResult,
};
use prism_volumetric_gpu::GpuContext;

/// Absolute parity bound on a scalar. A `GPU` divide may land a few units in the
/// last place from the scalar reference; `1e-4` admits that legal slack while
/// still failing a wrong port.
const EPS: f32 = 1.0e-4;

/// Relative parity bound, applied for larger magnitudes where a few units in the
/// last place exceed the absolute floor.
const REL: f32 = 1.0e-3;

/// Floor on the relative-error denominator so a near-zero expected value does
/// not inflate the relative error.
const REL_FLOOR: f32 = 1.0e-6;

/// The fixed criteria used across the suite (matching the golden unit test):
/// a steepness threshold of `1`, a Jacobian fold threshold of `0.2`, a
/// curvature threshold of `2`, and a breaking-intensity threshold of `0.5`.
const CRITERIA: BreakingCriteria = BreakingCriteria {
    steepness_threshold: 1.0,
    jacobian_fold_threshold: 0.2,
    curvature_threshold: 2.0,
    breaking_intensity: 0.5,
};

/// Returns whether `a` and `b` agree within the absolute or relative bound.
fn close(a: f32, b: f32) -> bool {
    let abs = (a - b).abs();
    let rel = abs / a.abs().max(b.abs()).max(REL_FLOOR);
    abs <= EPS || rel <= REL
}

/// Maps the golden class into the twin's class enum for an exact comparison.
fn map_class(c: BreakingClass) -> WaterBreakingClass {
    match c {
        BreakingClass::Calm => WaterBreakingClass::Calm,
        BreakingClass::Cresting => WaterBreakingClass::Cresting,
        BreakingClass::Breaking => WaterBreakingClass::Breaking,
    }
}

/// Builds a query from the three metrics and a foam `max_rate`, using the fixed
/// suite [`CRITERIA`].
fn query(
    steepness: f32,
    jacobian: f32,
    curvature: f32,
    max_rate: f32,
) -> WaterBreakingClassifyQuery {
    WaterBreakingClassifyQuery::new(
        steepness,
        jacobian,
        curvature,
        CRITERIA.steepness_threshold,
        CRITERIA.jacobian_fold_threshold,
        CRITERIA.curvature_threshold,
        CRITERIA.breaking_intensity,
        max_rate,
    )
}

/// Rebuilds the golden sample and criteria from a query.
fn parts(q: &WaterBreakingClassifyQuery) -> (BreakingSample, BreakingCriteria) {
    (
        BreakingSample {
            steepness: q.steepness,
            jacobian: q.jacobian,
            curvature: q.curvature,
        },
        BreakingCriteria {
            steepness_threshold: q.steepness_threshold,
            jacobian_fold_threshold: q.jacobian_fold_threshold,
            curvature_threshold: q.curvature_threshold,
            breaking_intensity: q.breaking_intensity,
        },
    )
}

/// Evaluates the golden closed forms on the same inputs the query carries.
fn oracle(q: &WaterBreakingClassifyQuery) -> WaterBreakingClassifyResult {
    let (sample, criteria) = parts(q);
    WaterBreakingClassifyResult {
        intensity: breaking_intensity(sample, criteria),
        foam: foam_source_strength(sample, criteria, q.max_rate),
        class: map_class(classify_breaking(sample, criteria)),
    }
}

/// Pins one `GPU` sample result against the golden oracle: both scalars within
/// tolerance and the discrete `class` exact.
fn check_pixel(idx: usize, got: &WaterBreakingClassifyResult, want: &WaterBreakingClassifyResult) {
    assert!(
        close(got.intensity, want.intensity),
        "sample {idx} intensity: gpu {} vs cpu {}",
        got.intensity,
        want.intensity
    );
    assert!(
        close(got.foam, want.foam),
        "sample {idx} foam: gpu {} vs cpu {}",
        got.foam,
        want.foam
    );
    assert_eq!(
        got.class, want.class,
        "sample {idx} class: gpu {:?} vs cpu {:?}",
        got.class, want.class
    );
}

/// Dispatches every sample and pins each result against the oracle.
fn check(ctx: &GpuContext, gpu: &GpuWaterBreakingClassify, queries: &[WaterBreakingClassifyQuery]) {
    let got = gpu.evaluate(ctx, queries);
    assert_eq!(
        got.len(),
        queries.len(),
        "result count must match the input count"
    );
    for (idx, (q, result)) in queries.iter().zip(got.iter()).enumerate() {
        let want = oracle(q);
        check_pixel(idx, result, &want);
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

/// Draws a scalar in `[0.0, span)` at milli resolution from `state`.
fn draw(state: &mut u64, span: f32) -> f32 {
    (lcg(state) % 1000) as f32 / 1000.0 * span
}

/// Whether a random sample is clear of the one computed-vs-computed guard — the
/// `intensity >= breaking_intensity` break — plus a clean margin off the fold
/// and steepness thresholds. Uses the golden intensity directly, with no
/// transcendental method.
fn well_conditioned(q: &WaterBreakingClassifyQuery) -> bool {
    let (sample, criteria) = parts(q);
    let intensity = breaking_intensity(sample, criteria);
    (intensity - criteria.breaking_intensity).abs() >= 0.02
        && (q.jacobian - criteria.jacobian_fold_threshold).abs() >= 0.02
        && (q.steepness - criteria.steepness_threshold).abs() >= 0.02
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn empty_input_returns_empty_without_dispatch() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping water_breaking_classify parity: no wgpu adapter on this host");
        return;
    };
    let gpu = GpuWaterBreakingClassify::new(&ctx);
    // An empty batch short-circuits before any dispatch (a storage buffer cannot
    // be zero-sized) and returns an empty vector.
    let got = gpu.evaluate(&ctx, &[]);
    assert!(got.is_empty(), "an empty batch produces no results");
}

#[test]
fn calm_surface_scores_zero_intensity() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuWaterBreakingClassify::new(&ctx);
    // Below every threshold: zero intensity, Calm, and no foam.
    let q = query(0.2, 1.0, 0.5, 5.0);
    check(&ctx, &gpu, &[q]);
}

#[test]
fn steep_unfolded_crest_is_cresting() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuWaterBreakingClassify::new(&ctx);
    // Steepness just past threshold, Jacobian healthy, low curvature: Cresting
    // with a low sub-threshold intensity and no foam.
    let q = query(1.2, 1.0, 0.5, 5.0);
    check(&ctx, &gpu, &[q]);
}

#[test]
fn folded_jacobian_forces_breaking() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuWaterBreakingClassify::new(&ctx);
    // A folded Jacobian forces Breaking regardless of a sub-threshold intensity,
    // and emits foam scaled by intensity.
    let q = query(0.1, 0.0, 0.1, 5.0);
    check(&ctx, &gpu, &[q]);
}

#[test]
fn high_intensity_unfolded_is_breaking() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuWaterBreakingClassify::new(&ctx);
    // Sharp steep crest with a healthy Jacobian: intensity well above the
    // breaking threshold forces Breaking without a fold.
    let q = query(3.0, 1.0, 8.0, 4.0);
    check(&ctx, &gpu, &[q]);
}

#[test]
fn breaking_with_zero_rate_emits_no_foam() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuWaterBreakingClassify::new(&ctx);
    // A breaking sample with a zero foam max_rate emits zero foam, exercising the
    // max(max_rate, 0) guard.
    let q = query(3.0, 1.0, 8.0, 0.0);
    check(&ctx, &gpu, &[q]);
}

#[test]
fn breaking_with_negative_rate_clamps_foam_to_zero() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuWaterBreakingClassify::new(&ctx);
    // A negative foam max_rate is clamped to zero, so a breaking sample still
    // emits no foam.
    let q = query(3.0, -0.3, 8.0, -2.0);
    check(&ctx, &gpu, &[q]);
}

#[test]
fn mixed_batch_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuWaterBreakingClassify::new(&ctx);
    // A deterministic batch spanning calm, cresting, folded, high-intensity and
    // zero-rate cases, dispatched together so the per-thread indexing and the
    // contiguous output slots are both exercised.
    let queries = [
        query(0.2, 1.0, 0.5, 5.0),
        query(1.2, 1.0, 0.5, 5.0),
        query(0.1, 0.0, 0.1, 5.0),
        query(3.0, 1.0, 8.0, 4.0),
        query(3.0, 1.0, 8.0, 0.0),
        query(1.6, -0.1, 3.0, 10.0),
        query(2.4, 0.5, 1.0, 7.5),
    ];
    check(&ctx, &gpu, &queries);
}

#[test]
fn random_sweep_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuWaterBreakingClassify::new(&ctx);
    let mut state = 0x51a2_c3d4_e5f6_0718_u64;
    let mut queries = Vec::new();
    // Several workgroups' worth of random samples spanning the metric ranges,
    // each clear of the intensity-break and threshold ties so CPU and GPU stay
    // on the same side of every discrete decision.
    let mut made = 0u32;
    let mut tries = 0u32;
    while made < 256 && tries < 20_000 {
        tries += 1;
        let steepness = draw(&mut state, 3.0);
        // Jacobian spans both sides of the fold threshold (0.2), including folds.
        let jacobian = draw(&mut state, 2.0) - 0.5;
        let curvature = draw(&mut state, 8.0);
        let max_rate = draw(&mut state, 10.0);
        let q = query(steepness, jacobian, curvature, max_rate);
        if !well_conditioned(&q) {
            continue;
        }
        queries.push(q);
        made += 1;
    }
    assert!(
        made >= 256,
        "expected 256 well-conditioned samples, got {made}"
    );
    check(&ctx, &gpu, &queries);
}
