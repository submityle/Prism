//! Real-device parity for the spectral melanin absorption twin:
//! [`GpuHairSpectralAbsorption`](prism_volumetric_gpu::hair_spectral_absorption::GpuHairSpectralAbsorption)
//! must reproduce the `CPU` golden
//! [`eumelanin_sigma_a_at`](prism_render_architecture::hair::spectral_absorption::eumelanin_sigma_a_at),
//! [`pheomelanin_sigma_a_at`](prism_render_architecture::hair::spectral_absorption::pheomelanin_sigma_a_at)
//! and
//! [`melanin_sigma_a_at`](prism_render_architecture::hair::spectral_absorption::melanin_sigma_a_at)
//! across off-grid wavelengths, out-of-range clamps, non-finite inputs,
//! concentration clamping and a randomized sweep.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Oracle
//!
//! The expected values are produced by calling the golden
//! [`eumelanin_sigma_a_at`](prism_render_architecture::hair::spectral_absorption::eumelanin_sigma_a_at),
//! [`pheomelanin_sigma_a_at`](prism_render_architecture::hair::spectral_absorption::pheomelanin_sigma_a_at)
//! and
//! [`melanin_sigma_a_at`](prism_render_architecture::hair::spectral_absorption::melanin_sigma_a_at)
//! directly, so the test pins `GPU == golden`, not merely that the shader
//! compiles.
//!
//! # Parity criterion
//!
//! The kernel performs a clamp, a `floor`, a fixed-step divide and a linear
//! blend of two tabulated values, so a `GPU` reciprocal and fused multiply-add
//! may land a few units in the last place from the scalar reference. Every
//! continuous output is asserted within `abs_diff <= 1e-4` or
//! `rel_diff <= 1e-3`. Fixtures keep the interpolation parameter away from the
//! exact sample-grid boundaries so a `floor` landing a unit in the last place
//! either side still agrees within tolerance.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::hair::spectral_absorption`；无第三方引擎源码或衍生代码。

use prism_render_architecture::hair::spectral_absorption::{
    eumelanin_sigma_a_at, melanin_sigma_a_at, pheomelanin_sigma_a_at,
};
use prism_volumetric_gpu::hair_spectral_absorption::{
    GpuHairSpectralAbsorption, HairSpectralAbsorptionQuery, HairSpectralAbsorptionResult,
};
use prism_volumetric_gpu::GpuContext;

/// Absolute parity bound. A `GPU` reciprocal and fused multiply-add may land a
/// few units in the last place from the scalar reference; `1e-4` admits that
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

/// Reconstructs the golden result in-host by calling the reference samplers
/// directly.
fn oracle(q: &HairSpectralAbsorptionQuery) -> HairSpectralAbsorptionResult {
    HairSpectralAbsorptionResult {
        eumelanin_sigma: eumelanin_sigma_a_at(q.wavelength_nm),
        pheomelanin_sigma: pheomelanin_sigma_a_at(q.wavelength_nm),
        combined_sigma: melanin_sigma_a_at(q.eumelanin, q.pheomelanin, q.wavelength_nm),
    }
}

/// Pins one `GPU` result against the in-host oracle: every continuous output
/// within tolerance.
fn check_query(
    idx: usize,
    got: &HairSpectralAbsorptionResult,
    want: &HairSpectralAbsorptionResult,
) {
    assert!(
        close(got.eumelanin_sigma, want.eumelanin_sigma),
        "query {idx} eumelanin_sigma: gpu {} vs cpu {}",
        got.eumelanin_sigma,
        want.eumelanin_sigma
    );
    assert!(
        close(got.pheomelanin_sigma, want.pheomelanin_sigma),
        "query {idx} pheomelanin_sigma: gpu {} vs cpu {}",
        got.pheomelanin_sigma,
        want.pheomelanin_sigma
    );
    assert!(
        close(got.combined_sigma, want.combined_sigma),
        "query {idx} combined_sigma: gpu {} vs cpu {}",
        got.combined_sigma,
        want.combined_sigma
    );
}

/// Dispatches `queries` and checks every result against the oracle.
fn check(
    ctx: &GpuContext,
    gpu: &GpuHairSpectralAbsorption,
    queries: &[HairSpectralAbsorptionQuery],
) {
    let got = gpu.evaluate(ctx, queries);
    assert_eq!(got.len(), queries.len(), "one result per query");
    for (idx, (q, g)) in queries.iter().zip(got.iter()).enumerate() {
        let want = oracle(q);
        check_query(idx, g, &want);
    }
}

/// A small `LCG` for the randomized sweep (host-only; the kernel is portable).
fn lcg(state: &mut u64) -> u32 {
    *state = state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    (*state >> 33) as u32
}

/// Maps a raw `u32` to an `f32` in `[lo, hi]` without any transcendental call.
fn uniform(bits: u32, lo: f32, hi: f32) -> f32 {
    let unit = (bits as f32) / (u32::MAX as f32);
    lo + unit * (hi - lo)
}

#[test]
fn empty_batch_produces_no_results() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuHairSpectralAbsorption::new(&ctx);
    // An empty batch short-circuits on the host (a storage buffer cannot be
    // zero-sized) and returns an empty vector.
    let got = gpu.evaluate(&ctx, &[]);
    assert!(got.is_empty(), "an empty batch produces no results");
}

#[test]
fn off_grid_wavelengths_match_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuHairSpectralAbsorption::new(&ctx);
    // Wavelengths chosen so the interpolation parameter lands well away from the
    // 25nm sample grid: 450->2.8, 500->4.8, 520->5.6, 610->9.2, 660->11.2.
    let queries = [
        HairSpectralAbsorptionQuery::new(1.0, 1.0, 450.0),
        HairSpectralAbsorptionQuery::new(1.3, 0.4, 500.0),
        HairSpectralAbsorptionQuery::new(0.7, 0.9, 520.0),
        HairSpectralAbsorptionQuery::new(2.0, 0.1, 610.0),
        HairSpectralAbsorptionQuery::new(0.25, 1.75, 660.0),
    ];
    check(&ctx, &gpu, &queries);
}

#[test]
fn out_of_range_wavelengths_clamp_to_endpoints() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuHairSpectralAbsorption::new(&ctx);
    // Below the violet endpoint folds to the first sample; above the red
    // endpoint folds to the last sample; a non-finite wavelength folds to the
    // violet endpoint, matching the golden `sanitized_wavelength`.
    let queries = [
        HairSpectralAbsorptionQuery::new(1.0, 1.0, 100.0),
        HairSpectralAbsorptionQuery::new(1.0, 1.0, 2000.0),
        HairSpectralAbsorptionQuery::new(1.0, 1.0, f32::NAN),
        HairSpectralAbsorptionQuery::new(1.0, 1.0, f32::INFINITY),
        HairSpectralAbsorptionQuery::new(1.0, 1.0, f32::NEG_INFINITY),
    ];
    check(&ctx, &gpu, &queries);
}

#[test]
fn concentration_clamping_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuHairSpectralAbsorption::new(&ctx);
    // Non-positive or non-finite concentrations fold to 0 in the combined
    // coefficient, while the per-unit samplers stay finite table lookups.
    let queries = [
        HairSpectralAbsorptionQuery::new(0.0, 0.0, 500.0),
        HairSpectralAbsorptionQuery::new(-4.0, -2.0, 520.0),
        HairSpectralAbsorptionQuery::new(f32::NAN, 1.0, 540.0),
        HairSpectralAbsorptionQuery::new(1.0, f32::INFINITY, 560.0),
        HairSpectralAbsorptionQuery::new(f32::NEG_INFINITY, 0.0, 600.0),
    ];
    check(&ctx, &gpu, &queries);
}

#[test]
fn linear_in_concentration_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuHairSpectralAbsorption::new(&ctx);
    // The combined coefficient is linear in each concentration; doubling both
    // concentrations doubles the combined coefficient, pinned via the oracle.
    let queries = [
        HairSpectralAbsorptionQuery::new(1.0, 0.5, 505.0),
        HairSpectralAbsorptionQuery::new(2.0, 1.0, 505.0),
        HairSpectralAbsorptionQuery::new(0.5, 2.5, 635.0),
        HairSpectralAbsorptionQuery::new(1.0, 5.0, 635.0),
    ];
    check(&ctx, &gpu, &queries);
}

#[test]
fn random_sweep_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuHairSpectralAbsorption::new(&ctx);
    let mut state = 0x5a3c_7e19_2d4b_6f08_u64;
    let mut queries: Vec<HairSpectralAbsorptionQuery> = Vec::new();
    // Several workgroups' worth of random concentrations and wavelengths pin the
    // clamp, the floor and the linear blend across a wide span. The wavelength
    // band intentionally overshoots the sampled range on both sides to exercise
    // the clamp, and concentrations span negatives to exercise the fold-to-0.
    for _ in 0..512 {
        let eumelanin = uniform(lcg(&mut state), -1.0, 3.0);
        let pheomelanin = uniform(lcg(&mut state), -1.0, 3.0);
        let wavelength_nm = uniform(lcg(&mut state), 360.0, 760.0);
        queries.push(HairSpectralAbsorptionQuery::new(
            eumelanin,
            pheomelanin,
            wavelength_nm,
        ));
    }
    check(&ctx, &gpu, &queries);
}
