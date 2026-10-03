//! Real-device parity for the hair pigment-to-absorption twin:
//! [`GpuHairMelaninAbsorption`](prism_volumetric_gpu::hair_melanin_absorption::GpuHairMelaninAbsorption)
//! must reproduce the stateless RGB `sigma_a` of the `CPU` golden
//! [`melanin_absorption`](prism_render_architecture::hair::melanin::melanin_absorption)
//! — the sanitized non-negative linear combination of the two per-pigment
//! spectra — across the zero-pigment, unit-eumelanin, unit-pheomelanin, mixed,
//! negative/non-finite, and natural-colour cases plus a randomized sweep
//! compared query-for-query.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Oracle
//!
//! The reference function is public, so it is called directly as the oracle: a
//! [`MelaninProfile`](prism_render_architecture::hair::melanin::MelaninProfile)
//! is built from each query's concentrations and
//! [`melanin_absorption`](prism_render_architecture::hair::melanin::melanin_absorption)
//! supplies the expected `sigma_a`. A passing `GPU == oracle` run is direct
//! evidence the kernel computes the same coefficient.
//!
//! # Parity criterion
//!
//! The map is a pure linear combination guarded by ordered comparisons, so
//! `CPU` and `GPU` agree to floating-point rounding and every channel is
//! asserted with `abs_diff <= 1e-4 || rel_diff <= 1e-3`.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::hair::melanin::melanin_absorption`；无第三方引擎源码或衍生代码。

use prism_render_architecture::hair::melanin::{
    melanin_absorption, MelaninProfile, NaturalHairColor, EUMELANIN_SIGMA_A, PHEOMELANIN_SIGMA_A,
};
use prism_volumetric_gpu::hair_melanin_absorption::{
    GpuHairMelaninAbsorption, HairMelaninAbsorptionQuery, HairMelaninAbsorptionResult,
};
use prism_volumetric_gpu::GpuContext;

/// Absolute tolerance for comparing one `sigma_a` channel.
const EPS: f32 = 1e-4;
/// Relative tolerance for comparing one `sigma_a` channel.
const REL: f32 = 1e-3;
/// Floor on the relative-tolerance denominator to keep near-zero comparisons
/// well-conditioned.
const REL_FLOOR: f32 = 1e-6;

/// Reports whether `a` and `b` match within the continuous tolerance.
fn close(a: f32, b: f32) -> bool {
    let abs = (a - b).abs();
    let rel = abs / a.abs().max(b.abs()).max(REL_FLOOR);
    abs <= EPS || rel <= REL
}

/// Computes the reference `sigma_a` for one query by calling the golden
/// directly.
fn oracle(q: &HairMelaninAbsorptionQuery) -> HairMelaninAbsorptionResult {
    let profile = MelaninProfile::new(q.eumelanin, q.pheomelanin);
    HairMelaninAbsorptionResult {
        sigma_a: melanin_absorption(profile),
    }
}

/// Pins one `GPU` result against the oracle: every channel within tolerance.
fn check_result(idx: usize, got: &HairMelaninAbsorptionResult, want: &HairMelaninAbsorptionResult) {
    for channel in 0..3 {
        assert!(
            close(got.sigma_a[channel], want.sigma_a[channel]),
            "query {idx} sigma_a[{channel}]: gpu {} vs cpu {}",
            got.sigma_a[channel],
            want.sigma_a[channel]
        );
    }
}

/// Dispatches every query and pins each result against the oracle.
fn check(ctx: &GpuContext, gpu: &GpuHairMelaninAbsorption, queries: &[HairMelaninAbsorptionQuery]) {
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

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn empty_input_returns_empty_without_dispatch() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping hair_melanin_absorption parity: no wgpu adapter on this host");
        return;
    };
    let gpu = GpuHairMelaninAbsorption::new(&ctx);
    // An empty batch short-circuits before any dispatch (a storage buffer cannot
    // be zero-sized) and returns an empty vector.
    let got = gpu.evaluate(&ctx, &[]);
    assert!(got.is_empty(), "an empty batch produces no results");
}

#[test]
fn zero_pigment_is_zero_absorption() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuHairMelaninAbsorption::new(&ctx);
    let queries = [HairMelaninAbsorptionQuery::new(0.0, 0.0)];
    let got = gpu.evaluate(&ctx, &queries);
    for channel in 0..3 {
        assert!(
            close(got[0].sigma_a[channel], 0.0),
            "zero pigment absorbs nothing"
        );
    }
    check(&ctx, &gpu, &queries);
}

#[test]
fn unit_pigments_reproduce_spectra() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuHairMelaninAbsorption::new(&ctx);
    // Unit eumelanin reproduces the eumelanin spectrum; unit pheomelanin
    // reproduces the pheomelanin spectrum.
    let queries = [
        HairMelaninAbsorptionQuery::new(1.0, 0.0),
        HairMelaninAbsorptionQuery::new(0.0, 1.0),
    ];
    let got = gpu.evaluate(&ctx, &queries);
    for channel in 0..3 {
        assert!(
            close(got[0].sigma_a[channel], EUMELANIN_SIGMA_A[channel]),
            "unit eumelanin reproduces its spectrum"
        );
        assert!(
            close(got[1].sigma_a[channel], PHEOMELANIN_SIGMA_A[channel]),
            "unit pheomelanin reproduces its spectrum"
        );
    }
    check(&ctx, &gpu, &queries);
}

#[test]
fn mixed_pigments_are_linear_combination() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuHairMelaninAbsorption::new(&ctx);
    let queries = [
        HairMelaninAbsorptionQuery::new(2.0, 3.0),
        HairMelaninAbsorptionQuery::new(0.5, 0.25),
        HairMelaninAbsorptionQuery::new(8.0, 0.0),
    ];
    check(&ctx, &gpu, &queries);
}

#[test]
fn negative_and_non_finite_clamp_to_zero() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuHairMelaninAbsorption::new(&ctx);
    // Negative, NaN and infinite concentrations collapse to zero on both sides,
    // and a finite positive channel paired with a non-finite channel keeps only
    // the finite contribution.
    let queries = [
        HairMelaninAbsorptionQuery::new(-5.0, -1.0),
        HairMelaninAbsorptionQuery::new(f32::NAN, f32::INFINITY),
        HairMelaninAbsorptionQuery::new(f32::NEG_INFINITY, 2.0),
        HairMelaninAbsorptionQuery::new(f32::INFINITY, 0.0),
    ];
    let got = gpu.evaluate(&ctx, &queries);
    for channel in 0..3 {
        assert!(
            close(got[0].sigma_a[channel], 0.0),
            "negative concentrations clamp to zero"
        );
        assert!(
            close(got[1].sigma_a[channel], 0.0),
            "non-finite concentrations clamp to zero"
        );
    }
    // The -inf eumelanin drops out; only 2.0 units of pheomelanin remain.
    for (channel, &pheomelanin) in PHEOMELANIN_SIGMA_A.iter().enumerate() {
        assert!(
            close(got[2].sigma_a[channel], 2.0 * pheomelanin),
            "the finite pheomelanin channel survives a non-finite eumelanin"
        );
    }
    check(&ctx, &gpu, &queries);
}

#[test]
fn natural_colours_match_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuHairMelaninAbsorption::new(&ctx);
    let queries: Vec<HairMelaninAbsorptionQuery> = [
        NaturalHairColor::Black,
        NaturalHairColor::Brown,
        NaturalHairColor::Blond,
        NaturalHairColor::Red,
    ]
    .iter()
    .map(|colour| {
        let profile = colour.profile();
        HairMelaninAbsorptionQuery::new(profile.eumelanin, profile.pheomelanin)
    })
    .collect();
    check(&ctx, &gpu, &queries);
}

#[test]
fn random_sweep_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuHairMelaninAbsorption::new(&ctx);
    let mut state = 0x5bd1_e995_1f3a_c0de_u64;
    let mut queries = Vec::new();
    // Several workgroups' worth of concentrations spanning both the clamped
    // (negative) and physical (positive) ranges in each channel, so both the
    // sanitation guard and the multiply-add are exercised across the dispatch.
    // Values are integer-derived, so no transcendental method appears, and they
    // stay away from the zero and infinity branch boundaries.
    while queries.len() < 300 {
        let eu = (lcg(&mut state) % 200_000) as f32 / 10_000.0 - 5.0;
        let ph = (lcg(&mut state) % 200_000) as f32 / 10_000.0 - 5.0;
        queries.push(HairMelaninAbsorptionQuery::new(eu, ph));
    }
    check(&ctx, &gpu, &queries);
}
