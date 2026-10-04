//! Real-device parity for the measured-conductor `Fresnel`-preset twin:
//! [`GpuMetalFresnel`](prism_volumetric_gpu::metal_fresnel_preset::GpuMetalFresnel)
//! must reproduce the `CPU` closed form of
//! `prism_render_architecture::reference_pt::metal::Metal::complex_ior`
//! composed with
//! `prism_render_architecture::reference_pt::conductor::fresnel_conductor` —
//! the baked complex-index lookup plus the exact unpolarized per-channel
//! conductor `Fresnel` — across the six curated metals at normal, oblique and
//! near-grazing incidence plus a randomized sweep compared query-for-query.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Oracle
//!
//! This wave forbids depending on the golden crate, so the host oracle is an
//! *independent* reimplementation of the same closed form: the same baked
//! `eta`/`k` table for the six conductors and the same per-channel
//! `fresnel_conductor_channel` arithmetic (`sqrt`, products, quotients, clamps
//! and guarded quotients). Because the reference and this oracle are both
//! scalar `f32`, a `GPU == oracle` pass is direct evidence the ported kernel
//! computes the same reflectance the reference does.
//!
//! # Parity criterion
//!
//! Each channel threads through two real square roots and a pair of guarded
//! quotients, so a `GPU` result may land a few units in the last place from the
//! scalar oracle; each reflectance channel is asserted within
//! `abs_diff <= 1e-4` or `rel_diff <= 1e-3`, with the relative error floored at
//! `1e-6` so a near-zero expected value does not inflate it. The `valid`
//! flag — set when the metal id is in range — is compared exactly.
//!
//! # Conditioning
//!
//! The randomized sweep draws a metal id in `0..6` and an incidence cosine in
//! `[0.02, 1.0]`, staying clear of `cos_theta = 0` where the `s`-polarized
//! denominator vanishes and the `select` mirror fallback kicks in. Every
//! intermediate inside that box is a well-conditioned non-negative square root
//! or guarded quotient, so a `GPU` evaluation lands a few units in the last
//! place from the scalar oracle and never straddles a discrete decision. The
//! named fixtures pin each metal's base colour at normal incidence, the warm
//! hue of gold/copper, the near-neutral tint of silver/aluminium, the grazing
//! rise toward one, and the out-of-range `valid = 0` case.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::reference_pt::metal` 与 `prism_render_architecture::reference_pt::conductor`；无第三方引擎源码或衍生代码。

use prism_volumetric_gpu::metal_fresnel_preset::{
    GpuMetalFresnel, MetalFresnelQuery, MetalFresnelResult,
};
use prism_volumetric_gpu::GpuContext;

/// Absolute bound on each reflectance channel. A `GPU` `sqrt`/divide may land a
/// few units in the last place from the scalar oracle; `1e-4` admits that legal
/// slack while still failing a wrong port.
const VALUE_ABS: f32 = 1.0e-4;

/// Relative bound on each reflectance channel, applied for larger magnitudes
/// where a few units in the last place exceed the absolute floor.
const VALUE_REL: f32 = 1.0e-3;

/// Floor on the relative-error denominator so a near-zero expected value does
/// not inflate the relative error.
const REL_FLOOR: f32 = 1.0e-6;

/// Number of curated metal presets; an id at or above this is out of range.
const METAL_COUNT: u32 = 6;

/// Per-channel real index `eta` for each preset, indexed by metal id:
/// `0` gold, `1` silver, `2` copper, `3` aluminium, `4` iron, `5` chromium.
/// Embedded independently of the golden crate.
const PRESET_ETA: [[f32; 3]; 6] = [
    [0.143, 0.375, 1.442],
    [0.155, 0.116, 0.138],
    [0.200, 0.924, 1.102],
    [1.345, 0.965, 0.617],
    [2.911, 2.950, 2.580],
    [3.181, 3.079, 2.392],
];

/// Per-channel extinction `k` for each preset, in the same id order.
const PRESET_K: [[f32; 3]; 6] = [
    [3.983, 2.386, 1.603],
    [4.818, 3.122, 2.146],
    [3.912, 2.448, 2.137],
    [7.474, 6.399, 5.303],
    [3.089, 2.931, 2.767],
    [3.329, 3.340, 3.148],
];

/// Returns whether `a` and `b` agree within the given absolute or relative
/// bound (relative error floored at `REL_FLOOR`).
fn close(a: f32, b: f32, abs_eps: f32, rel_eps: f32) -> bool {
    let abs = (a - b).abs();
    let rel = abs / a.abs().max(b.abs()).max(REL_FLOOR);
    abs <= abs_eps || rel <= rel_eps
}

/// The exact unpolarized conductor `Fresnel` for one channel, mirroring the
/// reference `fresnel_conductor_channel` line for line.
fn fresnel_conductor_channel(ci0: f32, eta: f32, k: f32) -> f32 {
    let ci = ci0.clamp(0.0, 1.0);
    let cos2 = ci * ci;
    let sin2 = 1.0 - cos2;
    let eta2 = eta * eta;
    let k2 = k * k;
    let t0 = eta2 - k2 - sin2;
    let a2b2 = (t0 * t0 + 4.0 * eta2 * k2).max(0.0).sqrt();
    let a = (0.5 * (a2b2 + t0)).max(0.0).sqrt();
    let t1 = a2b2 + cos2;
    let t2 = 2.0 * a * ci;
    let denom_s = t1 + t2;
    let r_s = if denom_s > 0.0 {
        (t1 - t2) / denom_s
    } else {
        1.0
    };
    let t3 = cos2 * a2b2 + sin2 * sin2;
    let t4 = t2 * sin2;
    let denom_p = t3 + t4;
    let r_p = if denom_p > 0.0 {
        r_s * (t3 - t4) / denom_p
    } else {
        r_s
    };
    (0.5 * (r_s + r_p)).clamp(0.0, 1.0)
}

/// Computes the expected result from the independent host oracle: a baked table
/// lookup composed with the three-channel conductor `Fresnel`. Mirrors the
/// kernel line for line.
fn oracle(q: &MetalFresnelQuery) -> MetalFresnelResult {
    if q.metal_id >= METAL_COUNT {
        return MetalFresnelResult {
            reflectance: [0.0, 0.0, 0.0],
            valid: 0,
        };
    }
    let idx = q.metal_id as usize;
    let eta = PRESET_ETA[idx];
    let k = PRESET_K[idx];
    MetalFresnelResult {
        reflectance: [
            fresnel_conductor_channel(q.cos_theta, eta[0], k[0]),
            fresnel_conductor_channel(q.cos_theta, eta[1], k[1]),
            fresnel_conductor_channel(q.cos_theta, eta[2], k[2]),
        ],
        valid: 1,
    }
}

/// Pins one `GPU` result against the host oracle: each reflectance channel
/// under the shared bound, the `valid` flag exactly.
fn check_one(idx: usize, got: &MetalFresnelResult, want: &MetalFresnelResult) {
    assert_eq!(
        got.valid, want.valid,
        "query {idx} valid: gpu {} vs cpu {}",
        got.valid, want.valid
    );
    assert!(
        close(
            got.reflectance[0],
            want.reflectance[0],
            VALUE_ABS,
            VALUE_REL
        ),
        "query {idx} reflectance_r: gpu {} vs cpu {}",
        got.reflectance[0],
        want.reflectance[0]
    );
    assert!(
        close(
            got.reflectance[1],
            want.reflectance[1],
            VALUE_ABS,
            VALUE_REL
        ),
        "query {idx} reflectance_g: gpu {} vs cpu {}",
        got.reflectance[1],
        want.reflectance[1]
    );
    assert!(
        close(
            got.reflectance[2],
            want.reflectance[2],
            VALUE_ABS,
            VALUE_REL
        ),
        "query {idx} reflectance_b: gpu {} vs cpu {}",
        got.reflectance[2],
        want.reflectance[2]
    );
}

/// Dispatches every query and pins each result against the oracle.
fn check(ctx: &GpuContext, gpu: &GpuMetalFresnel, queries: &[MetalFresnelQuery]) {
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

/// Draws one well-conditioned random query: a metal id in `0..6` and an
/// incidence cosine in `[0.02, 1.0]`, clear of the vanishing-denominator
/// grazing tie.
fn random_query(state: &mut u64) -> MetalFresnelQuery {
    let metal_id = lcg(state) % METAL_COUNT;
    let cos_theta = uniform(state, 0.02, 1.0);
    MetalFresnelQuery::new(metal_id, cos_theta)
}

/// A fixed battery of named cases: each of the six metals at normal incidence
/// (base colour), gold/copper warm hue, silver/aluminium near-neutral tint, a
/// near-grazing rise toward one, and an out-of-range id reporting `valid = 0`.
fn fixture_queries() -> Vec<MetalFresnelQuery> {
    vec![
        // Base colour (F0) at normal incidence for each metal.
        MetalFresnelQuery::new(0, 1.0),
        MetalFresnelQuery::new(1, 1.0),
        MetalFresnelQuery::new(2, 1.0),
        MetalFresnelQuery::new(3, 1.0),
        MetalFresnelQuery::new(4, 1.0),
        MetalFresnelQuery::new(5, 1.0),
        // Oblique incidence for a spread of metals.
        MetalFresnelQuery::new(0, 0.7),
        MetalFresnelQuery::new(2, 0.5),
        MetalFresnelQuery::new(3, 0.35),
        // Near grazing: reflectance rises toward one.
        MetalFresnelQuery::new(0, 0.05),
        MetalFresnelQuery::new(4, 0.02),
        // Out of range id: zero reflectance and valid = 0.
        MetalFresnelQuery::new(6, 0.8),
        MetalFresnelQuery::new(42, 0.3),
    ]
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn empty_input_returns_empty_without_dispatch() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping metal_fresnel_preset parity: no wgpu adapter on this host");
        return;
    };
    let gpu = GpuMetalFresnel::new(&ctx);
    // An empty batch short-circuits before any dispatch (a storage buffer cannot
    // be zero-sized) and returns an empty vector.
    let got = gpu.evaluate(&ctx, &[]);
    assert!(got.is_empty(), "an empty batch produces no results");
}

#[test]
fn gold_base_colour_is_warm() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuMetalFresnel::new(&ctx);
    let q = MetalFresnelQuery::new(0, 1.0);
    let got = gpu.evaluate(&ctx, &[q]);
    assert_eq!(got.len(), 1);
    check_one(0, &got[0], &oracle(&q));
    assert_eq!(got[0].valid, 1, "gold is a valid preset");
    assert!(
        got[0].reflectance[0] > got[0].reflectance[2] + 0.2,
        "gold's base colour is warm (red well above blue): {:?}",
        got[0].reflectance
    );
}

#[test]
fn copper_base_colour_is_warm() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuMetalFresnel::new(&ctx);
    let q = MetalFresnelQuery::new(2, 1.0);
    let got = gpu.evaluate(&ctx, &[q]);
    assert_eq!(got.len(), 1);
    check_one(0, &got[0], &oracle(&q));
    assert!(
        got[0].reflectance[0] > got[0].reflectance[2] + 0.2,
        "copper's base colour is warm (red well above blue): {:?}",
        got[0].reflectance
    );
}

#[test]
fn silver_base_colour_is_near_neutral() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuMetalFresnel::new(&ctx);
    let q = MetalFresnelQuery::new(1, 1.0);
    let got = gpu.evaluate(&ctx, &[q]);
    assert_eq!(got.len(), 1);
    check_one(0, &got[0], &oracle(&q));
    assert!(
        (got[0].reflectance[0] - got[0].reflectance[2]).abs() < 0.15,
        "silver's base colour is near neutral: {:?}",
        got[0].reflectance
    );
}

#[test]
fn grazing_incidence_approaches_one() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuMetalFresnel::new(&ctx);
    let q = MetalFresnelQuery::new(0, 0.02);
    let got = gpu.evaluate(&ctx, &[q]);
    assert_eq!(got.len(), 1);
    check_one(0, &got[0], &oracle(&q));
    assert!(
        got[0].reflectance[0] > 0.9 && got[0].reflectance[1] > 0.9 && got[0].reflectance[2] > 0.9,
        "near grazing every channel approaches one: {:?}",
        got[0].reflectance
    );
}

#[test]
fn out_of_range_id_reports_zero_and_invalid() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuMetalFresnel::new(&ctx);
    let q = MetalFresnelQuery::new(6, 0.8);
    let got = gpu.evaluate(&ctx, &[q]);
    assert_eq!(got.len(), 1);
    check_one(0, &got[0], &oracle(&q));
    assert_eq!(got[0].valid, 0, "an out-of-range metal id is invalid");
    assert!(
        got[0].reflectance == [0.0, 0.0, 0.0],
        "an invalid query reports zero reflectance: {:?}",
        got[0].reflectance
    );
}

#[test]
fn fixture_batch_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuMetalFresnel::new(&ctx);
    check(&ctx, &gpu, &fixture_queries());
}

#[test]
fn random_sweep_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuMetalFresnel::new(&ctx);
    let mut state = 0x1234_5678_9abc_def0_u64;
    let mut queries = fixture_queries();
    // Several workgroups' worth of random, well-conditioned queries pin the
    // reflectance across every metal and a wide span of incidence angles.
    for _ in 0..512 {
        queries.push(random_query(&mut state));
    }
    check(&ctx, &gpu, &queries);
}
