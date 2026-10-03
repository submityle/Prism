//! Real-device parity for the exact conductor-`Fresnel` reflectance twin:
//! [`GpuConductorFresnel`](prism_volumetric_gpu::conductor_fresnel::GpuConductorFresnel)
//! must reproduce the `CPU` closed form of the spectral oracle
//! `prism_render_architecture::reference_pt::conductor` — the per-channel exact
//! reflectance `fresnel_conductor` and the `32`-node hemispherical average
//! `average_fresnel_conductor` — across grazing incidence, normal incidence,
//! several measured metals (gold, copper, aluminium, silver) and a randomized
//! sweep compared query-for-query.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Oracle
//!
//! The reference channel kernel `fresnel_conductor_channel` is private, and this
//! wave forbids depending on the golden crate, so the host oracle is an
//! *independent* reimplementation of the same closed form (the standard `PBRT`
//! `FrComplex` decomposition into `a^2 + b^2` and `a`, then the mean of the
//! `s`- and `p`-polarized reflectances) plus the identical `32`-node midpoint
//! quadrature. Because the reference and this oracle are both scalar `f32`, a
//! `GPU == oracle` pass is direct evidence the ported kernel computes the same
//! reflectance the reference does.
//!
//! # Parity criterion
//!
//! The direct reflectance threads through `sqrt`, products and quotients, so a
//! `GPU` result may land a few units in the last place from the scalar oracle;
//! each channel is asserted within `abs_diff <= 1e-5` or `rel_diff <= 1e-4`. The
//! `32`-node quadrature average accumulates thirty-two such evaluations, so it
//! is asserted within the looser `abs_diff <= 1e-4` or `rel_diff <= 1e-3`. Both
//! use a `rel_diff` floor of `1e-6` so a near-zero expected value does not
//! inflate the relative error.
//!
//! # Conditioning
//!
//! The randomized sweep draws `eta` in `[0.1, 4]` and `k` in `[1, 5]` — the box
//! that contains every real measured conductor — and the cosine across the full
//! `[0, 1]`. That domain is deliberate: the closed form forms
//! `a^2 + b^2 = sqrt(t0^2 + 4 eta^2 k^2)` with `t0 = eta^2 - k^2 - sin^2`, then
//! takes `a = sqrt((a^2 + b^2 + t0) / 2)`. When `4 eta^2 k^2` is negligible
//! against `t0^2` (which happens as `eta -> 0`, or for `k -> 0` with `eta < 1`),
//! `a^2 + b^2` collapses onto `|t0|` and the sum `a^2 + b^2 + t0` suffers a
//! catastrophic cancellation that leaves `a` with no significant `f32` digits.
//! Those corners are the perfect-conductor and total-internal-reflection
//! idealisations rather than physical metals, so the sweep stays clear of them
//! and the near-unity / grazing extremes are instead pinned by the dedicated
//! named fixtures. Inside the drawn box every intermediate is a well-conditioned
//! non-negative square root or guarded quotient, so a `GPU` evaluation lands a
//! few units in the last place from the scalar oracle and never straddles a
//! branch cliff.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::reference_pt::conductor`；无第三方引擎源码或衍生代码。

use prism_volumetric_gpu::conductor_fresnel::{
    ConductorFresnelQuery, ConductorFresnelResult, GpuConductorFresnel,
};
use prism_volumetric_gpu::GpuContext;

/// Absolute bound on the direct per-channel reflectance. A `GPU`
/// `sqrt`/divide may land a few units in the last place from the scalar oracle;
/// `1e-5` admits that legal slack while still failing a wrong port.
const FRESNEL_ABS: f32 = 1.0e-5;

/// Relative bound on the direct per-channel reflectance, applied for larger
/// magnitudes where a few units in the last place exceed the absolute floor.
const FRESNEL_REL: f32 = 1.0e-4;

/// Absolute bound on the `32`-node quadrature average, looser because the sum
/// of thirty-two evaluations accumulates more last-place slack.
const AVG_ABS: f32 = 1.0e-4;

/// Relative bound on the `32`-node quadrature average.
const AVG_REL: f32 = 1.0e-3;

/// Floor on the relative-error denominator so a near-zero expected value does
/// not inflate the relative error.
const REL_FLOOR: f32 = 1.0e-6;

/// Number of midpoint-quadrature nodes used to pre-integrate the average
/// reflectance, matching the reference `AVG_FRESNEL_NODES`.
const AVG_FRESNEL_NODES: u32 = 32;

/// Returns whether `a` and `b` agree within the given absolute or relative
/// bound (relative error floored at `REL_FLOOR`).
fn close(a: f32, b: f32, abs_eps: f32, rel_eps: f32) -> bool {
    let abs = (a - b).abs();
    let rel = abs / a.abs().max(b.abs()).max(REL_FLOOR);
    abs <= abs_eps || rel <= rel_eps
}

/// Independent reimplementation of the reference channel kernel: the exact
/// unpolarized `Fresnel` reflectance of one wavelength channel of a conductor
/// interface from the incidence cosine and the complex index `eta + i*k`. Only
/// `sqrt`, products, quotients and a clamp appear.
fn fresnel_conductor_channel(cos_theta_i: f32, eta: f32, k: f32) -> f32 {
    let cos_i = cos_theta_i.clamp(0.0, 1.0);
    let cos2 = cos_i * cos_i;
    let sin2 = 1.0 - cos2;
    let eta2 = eta * eta;
    let k2 = k * k;
    let t0 = eta2 - k2 - sin2;
    let a2_plus_b2 = (t0 * t0 + 4.0 * eta2 * k2).max(0.0).sqrt();
    let a = (0.5 * (a2_plus_b2 + t0)).max(0.0).sqrt();
    let t1 = a2_plus_b2 + cos2;
    let t2 = 2.0 * a * cos_i;
    let denom_s = t1 + t2;
    let r_s = if denom_s > 0.0 {
        (t1 - t2) / denom_s
    } else {
        1.0
    };
    let t3 = cos2 * a2_plus_b2 + sin2 * sin2;
    let t4 = t2 * sin2;
    let denom_p = t3 + t4;
    let r_p = if denom_p > 0.0 {
        r_s * (t3 - t4) / denom_p
    } else {
        r_s
    };
    (0.5 * (r_s + r_p)).clamp(0.0, 1.0)
}

/// Independent reimplementation of the hemispherical cosine-weighted average
/// `F_avg = 2 * integral_0^1 F(mu) mu d mu` by `32`-node midpoint quadrature,
/// evaluated for a single channel.
fn average_channel(eta: f32, k: f32) -> f32 {
    let inv_nodes = 1.0 / AVG_FRESNEL_NODES as f32;
    let mut acc = 0.0;
    for i in 0..AVG_FRESNEL_NODES {
        let mu = (i as f32 + 0.5) * inv_nodes;
        let weight = 2.0 * mu * inv_nodes;
        acc += fresnel_conductor_channel(mu, eta, k) * weight;
    }
    acc
}

/// Computes the expected result from the independent host oracle, the faithful
/// reference the `GPU` is pinned against.
fn oracle(q: &ConductorFresnelQuery) -> ConductorFresnelResult {
    ConductorFresnelResult {
        fresnel_r: fresnel_conductor_channel(q.cos_theta_i, q.eta_r, q.k_r),
        fresnel_g: fresnel_conductor_channel(q.cos_theta_i, q.eta_g, q.k_g),
        fresnel_b: fresnel_conductor_channel(q.cos_theta_i, q.eta_b, q.k_b),
        avg_r: average_channel(q.eta_r, q.k_r),
        avg_g: average_channel(q.eta_g, q.k_g),
        avg_b: average_channel(q.eta_b, q.k_b),
    }
}

/// Pins one `GPU` result against the host oracle: each direct reflectance
/// channel under the tight bound, each average channel under the looser one.
fn check_one(idx: usize, got: &ConductorFresnelResult, want: &ConductorFresnelResult) {
    assert!(
        close(got.fresnel_r, want.fresnel_r, FRESNEL_ABS, FRESNEL_REL),
        "query {idx} fresnel_r: gpu {} vs cpu {}",
        got.fresnel_r,
        want.fresnel_r
    );
    assert!(
        close(got.fresnel_g, want.fresnel_g, FRESNEL_ABS, FRESNEL_REL),
        "query {idx} fresnel_g: gpu {} vs cpu {}",
        got.fresnel_g,
        want.fresnel_g
    );
    assert!(
        close(got.fresnel_b, want.fresnel_b, FRESNEL_ABS, FRESNEL_REL),
        "query {idx} fresnel_b: gpu {} vs cpu {}",
        got.fresnel_b,
        want.fresnel_b
    );
    assert!(
        close(got.avg_r, want.avg_r, AVG_ABS, AVG_REL),
        "query {idx} avg_r: gpu {} vs cpu {}",
        got.avg_r,
        want.avg_r
    );
    assert!(
        close(got.avg_g, want.avg_g, AVG_ABS, AVG_REL),
        "query {idx} avg_g: gpu {} vs cpu {}",
        got.avg_g,
        want.avg_g
    );
    assert!(
        close(got.avg_b, want.avg_b, AVG_ABS, AVG_REL),
        "query {idx} avg_b: gpu {} vs cpu {}",
        got.avg_b,
        want.avg_b
    );
}

/// Dispatches every query and pins each result against the oracle.
fn check(ctx: &GpuContext, gpu: &GpuConductorFresnel, queries: &[ConductorFresnelQuery]) {
    let got = gpu.evaluate(ctx, queries);
    assert_eq!(
        got.len(),
        queries.len(),
        "result count must match the input count"
    );
    for (idx, (q, result)) in queries.iter().zip(got.iter()).enumerate() {
        let want = oracle(q);
        check_one(idx, result, &want);
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

/// Draws a `f32` in `[lo, hi)` from the generator, using only integer-to-float
/// division (no transcendental).
fn uniform(state: &mut u64, lo: f32, hi: f32) -> f32 {
    let u = lcg(state) as f32 * (1.0 / 4_294_967_296.0);
    lo + (hi - lo) * u
}

/// Builds one well-conditioned random query over the span of real measured
/// conductors: `eta` in `[0.1, 4]`, `k` in `[1, 5]`, cosine across the full
/// `[0, 1]` (grazing included). Every real metal lives inside this box; the
/// excluded corners `eta -> 0` (the perfect-conductor idealisation) and
/// `k -> 0` with `eta < 1` (a total-internal-reflection analogue) are not
/// physical conductors and are intrinsically `f32`-ill-conditioned, so they are
/// pinned by the dedicated named fixtures instead (see `# Conditioning`).
fn random_query(state: &mut u64) -> ConductorFresnelQuery {
    ConductorFresnelQuery::new(
        uniform(state, 0.1, 4.0),
        uniform(state, 0.1, 4.0),
        uniform(state, 0.1, 4.0),
        uniform(state, 1.0, 5.0),
        uniform(state, 1.0, 5.0),
        uniform(state, 1.0, 5.0),
        uniform(state, 0.0, 1.0),
    )
}

/// Gold's measured red/green/blue real index of refraction.
const GOLD_ETA: [f32; 3] = [0.143, 0.375, 1.442];
/// Gold's measured red/green/blue extinction coefficient.
const GOLD_K: [f32; 3] = [3.983, 2.386, 1.603];
/// Copper's measured red/green/blue real index of refraction.
const COPPER_ETA: [f32; 3] = [0.2, 0.924, 1.102];
/// Copper's measured red/green/blue extinction coefficient.
const COPPER_K: [f32; 3] = [3.6, 2.577, 2.293];
/// Aluminium's measured red/green/blue real index of refraction.
const ALUMINIUM_ETA: [f32; 3] = [1.345, 0.965, 0.617];
/// Aluminium's measured red/green/blue extinction coefficient.
const ALUMINIUM_K: [f32; 3] = [7.475, 6.4, 5.303];
/// Silver's measured red/green/blue real index of refraction.
const SILVER_ETA: [f32; 3] = [0.155, 0.116, 0.138];
/// Silver's measured red/green/blue extinction coefficient.
const SILVER_K: [f32; 3] = [4.818, 3.122, 2.146];

/// Builds a query for a metal from its `eta`/`k` triples and an incidence
/// cosine.
fn metal_query(eta: [f32; 3], k: [f32; 3], cos_theta_i: f32) -> ConductorFresnelQuery {
    ConductorFresnelQuery::new(eta[0], eta[1], eta[2], k[0], k[1], k[2], cos_theta_i)
}

/// A fixed battery of named cases spanning grazing/normal incidence and several
/// measured metals, dispatched together.
fn fixture_queries() -> Vec<ConductorFresnelQuery> {
    vec![
        // Gold at grazing, mid-angle and normal incidence.
        metal_query(GOLD_ETA, GOLD_K, 0.02),
        metal_query(GOLD_ETA, GOLD_K, 0.5),
        metal_query(GOLD_ETA, GOLD_K, 1.0),
        // Copper at two angles.
        metal_query(COPPER_ETA, COPPER_K, 0.25),
        metal_query(COPPER_ETA, COPPER_K, 0.85),
        // Aluminium at two angles.
        metal_query(ALUMINIUM_ETA, ALUMINIUM_K, 0.1),
        metal_query(ALUMINIUM_ETA, ALUMINIUM_K, 0.95),
        // Silver at two angles.
        metal_query(SILVER_ETA, SILVER_K, 0.4),
        metal_query(SILVER_ETA, SILVER_K, 0.7),
        // A pure dielectric-like channel (k = 0) to exercise the a = 0 branch.
        ConductorFresnelQuery::new(1.5, 1.5, 1.5, 0.0, 0.0, 0.0, 0.6),
    ]
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn empty_input_returns_empty_without_dispatch() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping conductor_fresnel parity: no wgpu adapter on this host");
        return;
    };
    let gpu = GpuConductorFresnel::new(&ctx);
    // An empty batch short-circuits before any dispatch (a storage buffer cannot
    // be zero-sized) and returns an empty vector.
    let got = gpu.evaluate(&ctx, &[]);
    assert!(got.is_empty(), "an empty batch produces no results");
}

#[test]
fn grazing_incidence_reflects_near_one() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuConductorFresnel::new(&ctx);
    // At near-grazing incidence a conductor reflects almost everything on every
    // channel.
    let q = metal_query(GOLD_ETA, GOLD_K, 0.01);
    let got = gpu.evaluate(&ctx, &[q]);
    assert_eq!(got.len(), 1);
    check_one(0, &got[0], &oracle(&q));
    assert!(
        got[0].fresnel_r > 0.95 && got[0].fresnel_g > 0.95 && got[0].fresnel_b > 0.95,
        "grazing reflectance should be near one: {:?}",
        got[0]
    );
}

#[test]
fn gold_normal_incidence_is_warm() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuConductorFresnel::new(&ctx);
    // Gold at normal incidence reflects far more red than blue (its warm base
    // colour).
    let q = metal_query(GOLD_ETA, GOLD_K, 1.0);
    let got = gpu.evaluate(&ctx, &[q]);
    assert_eq!(got.len(), 1);
    check_one(0, &got[0], &oracle(&q));
    assert!(
        got[0].fresnel_r > got[0].fresnel_b,
        "gold is warmer in red than blue at normal incidence: {:?}",
        got[0]
    );
}

#[test]
fn average_is_cosine_independent() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuConductorFresnel::new(&ctx);
    // The hemispherical average does not depend on the query cosine, so two
    // queries that differ only in cos_theta_i must report the same average.
    let at_zero = metal_query(COPPER_ETA, COPPER_K, 0.0);
    let at_one = metal_query(COPPER_ETA, COPPER_K, 1.0);
    let got = gpu.evaluate(&ctx, &[at_zero, at_one]);
    assert_eq!(got.len(), 2);
    check_one(0, &got[0], &oracle(&at_zero));
    check_one(1, &got[1], &oracle(&at_one));
    assert!(
        close(got[0].avg_r, got[1].avg_r, AVG_ABS, AVG_REL)
            && close(got[0].avg_g, got[1].avg_g, AVG_ABS, AVG_REL)
            && close(got[0].avg_b, got[1].avg_b, AVG_ABS, AVG_REL),
        "the average must be cosine-independent: {:?} vs {:?}",
        got[0],
        got[1]
    );
}

#[test]
fn fixture_batch_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuConductorFresnel::new(&ctx);
    check(&ctx, &gpu, &fixture_queries());
}

#[test]
fn random_sweep_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuConductorFresnel::new(&ctx);
    let mut state = 0x1234_5678_9abc_def0_u64;
    let mut queries = fixture_queries();
    // Several workgroups' worth of random, well-conditioned queries pin every
    // reported reflectance across a wide span of complex indices and angles.
    for _ in 0..512 {
        queries.push(random_query(&mut state));
    }
    check(&ctx, &gpu, &queries);
}
