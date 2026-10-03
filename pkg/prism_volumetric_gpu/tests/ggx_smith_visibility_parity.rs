//! Real-device parity for the separable `Smith`-`GGX` masking-shadowing twin:
//! [`GpuGgxSmithVisibility`](prism_volumetric_gpu::ggx_smith_visibility::GpuGgxSmithVisibility)
//! must reproduce the `CPU` closed form of the particle microfacet oracle
//! `prism_render_architecture::particle::microfacet_ggx` — the one-sided
//! `schlick_ggx_g1`, the separable `smith_g_separable` and the visibility
//! `visibility_smith_ggx_separable` — across grazing cosines, a mirror-smooth
//! `k = 0`, asymmetric light/view geometries and a randomized sweep compared
//! query-for-query.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Oracle
//!
//! This wave forbids depending on the golden crate, so the host oracle is an
//! *independent* reimplementation of the same closed form: `clamp01`, the
//! guarded rational `G1(NoX, k) = NoX / (NoX * (1 - k) + k)`, their product and
//! the guarded visibility `G / (4 * NoL * NoV)`, with the identical `MIN_DENOM`
//! of `1e-7`. Because the reference and this oracle are both scalar `f32`, a
//! `GPU == oracle` pass is direct evidence the ported kernel computes the same
//! masking the reference does.
//!
//! # Parity criterion
//!
//! Every output threads through products and guarded quotients, so a `GPU`
//! result may land a few units in the last place from the scalar oracle; each
//! is asserted within `abs_diff <= 1e-5` or `rel_diff <= 1e-4`, with a relative
//! floor of `1e-6` so a near-zero expected value does not inflate the relative
//! error. Guarded-to-zero outputs are compared with the same absolute bound,
//! never a bare `f32` equality.
//!
//! # Conditioning
//!
//! The randomized sweep draws `NoL` and `NoV` in `[0.1, 1.0]` — away from the
//! grazing zero where `4 * NoL * NoV` falls under the `MIN_DENOM` guard — and
//! `k` in `[0.0, 0.5]`, the span the direct and `IBL` remaps produce for
//! physical roughness. Inside that box the `G1` denominator `NoX * (1 - k) + k`
//! stays comfortably positive and the visibility divisor is well clear of the
//! guard, so every intermediate is a well-conditioned quotient and a `GPU`
//! evaluation lands a few units in the last place from the scalar oracle. The
//! grazing and guarded-branch corners are instead pinned by the dedicated named
//! fixtures.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::particle::microfacet_ggx`；无第三方引擎源码或衍生代码。

use prism_volumetric_gpu::ggx_smith_visibility::{
    GgxSmithVisibilityQuery, GgxSmithVisibilityResult, GpuGgxSmithVisibility,
};
use prism_volumetric_gpu::GpuContext;

/// Generic denominator guard matching the reference `MIN_DENOM`.
const MIN_DENOM: f32 = 1.0e-7;

/// Absolute bound on each masking output. A `GPU` divide may land a few units
/// in the last place from the scalar oracle; `1e-5` admits that legal slack
/// while still failing a wrong port.
const ABS: f32 = 1.0e-5;

/// Relative bound on each masking output, applied for larger magnitudes where a
/// few units in the last place exceed the absolute floor.
const REL: f32 = 1.0e-4;

/// Floor on the relative-error denominator so a near-zero expected value does
/// not inflate the relative error.
const REL_FLOOR: f32 = 1.0e-6;

/// Returns whether `a` and `b` agree within the absolute or relative bound
/// (relative error floored at `REL_FLOOR`). This is used for every comparison,
/// including guarded-to-zero outputs, so no bare `f32` equality appears.
fn close(a: f32, b: f32) -> bool {
    let abs = (a - b).abs();
    let rel = abs / a.abs().max(b.abs()).max(REL_FLOOR);
    abs <= ABS || rel <= REL
}

/// Clamps a scalar into the `0..=1` range (used for the cosine terms).
fn clamp01(x: f32) -> f32 {
    x.clamp(0.0, 1.0)
}

/// Independent reimplementation of the reference one-sided `Schlick`-`GGX`
/// masking `G1(NoX, k) = NoX / (NoX * (1 - k) + k)` with the clamp and the
/// denominator guard.
fn schlick_ggx_g1(n_dot_x: f32, k: f32) -> f32 {
    let x = clamp01(n_dot_x);
    let denom = x * (1.0 - k) + k;
    if denom < MIN_DENOM {
        return 0.0;
    }
    x / denom
}

/// Independent reimplementation of the separable masking-shadowing
/// `G = G1(NoL) * G1(NoV)`.
fn smith_g_separable(n_dot_l: f32, n_dot_v: f32, k: f32) -> f32 {
    schlick_ggx_g1(n_dot_l, k) * schlick_ggx_g1(n_dot_v, k)
}

/// Independent reimplementation of the separable visibility
/// `V = G / (4 * NoL * NoV)` with the guarded foreshortening divisor.
fn visibility_smith_ggx_separable(n_dot_l: f32, n_dot_v: f32, k: f32) -> f32 {
    let nl = clamp01(n_dot_l);
    let nv = clamp01(n_dot_v);
    let denom = 4.0 * nl * nv;
    if denom < MIN_DENOM {
        return 0.0;
    }
    smith_g_separable(nl, nv, k) / denom
}

/// Computes the expected result from the independent host oracle, the faithful
/// reference the `GPU` is pinned against.
fn oracle(q: &GgxSmithVisibilityQuery) -> GgxSmithVisibilityResult {
    let nl = clamp01(q.n_dot_l);
    let nv = clamp01(q.n_dot_v);
    GgxSmithVisibilityResult {
        g1_l: schlick_ggx_g1(nl, q.k),
        g1_v: schlick_ggx_g1(nv, q.k),
        g_separable: smith_g_separable(nl, nv, q.k),
        visibility: visibility_smith_ggx_separable(nl, nv, q.k),
    }
}

/// Pins one `GPU` result against the host oracle, every output under the shared
/// bound.
fn check_one(idx: usize, got: &GgxSmithVisibilityResult, want: &GgxSmithVisibilityResult) {
    assert!(
        close(got.g1_l, want.g1_l),
        "query {idx} g1_l: gpu {} vs cpu {}",
        got.g1_l,
        want.g1_l
    );
    assert!(
        close(got.g1_v, want.g1_v),
        "query {idx} g1_v: gpu {} vs cpu {}",
        got.g1_v,
        want.g1_v
    );
    assert!(
        close(got.g_separable, want.g_separable),
        "query {idx} g_separable: gpu {} vs cpu {}",
        got.g_separable,
        want.g_separable
    );
    assert!(
        close(got.visibility, want.visibility),
        "query {idx} visibility: gpu {} vs cpu {}",
        got.visibility,
        want.visibility
    );
}

/// Dispatches every query and pins each result against the oracle.
fn check(ctx: &GpuContext, gpu: &GpuGgxSmithVisibility, queries: &[GgxSmithVisibilityQuery]) {
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

/// Builds one well-conditioned random query over the span of physical shading:
/// `NoL` and `NoV` in `[0.1, 1.0]` (away from the grazing guard) and `k` in
/// `[0.0, 0.5]` (the direct/`IBL` remap span). See `# Conditioning`.
fn random_query(state: &mut u64) -> GgxSmithVisibilityQuery {
    GgxSmithVisibilityQuery::new(
        uniform(state, 0.1, 1.0),
        uniform(state, 0.1, 1.0),
        uniform(state, 0.0, 0.5),
    )
}

/// A fixed battery of named cases spanning well-conditioned mid-range geometry,
/// a mirror-smooth `k = 0`, asymmetric light/view and the roughness extremes,
/// dispatched together.
fn fixture_queries() -> Vec<GgxSmithVisibilityQuery> {
    vec![
        // Well-conditioned mid-range: both cosines comfortably off grazing.
        GgxSmithVisibilityQuery::new(0.7, 0.5, 0.25),
        GgxSmithVisibilityQuery::new(0.9, 0.8, 0.125),
        // Mirror-smooth k = 0 makes each G1 the identity x / x = 1 for x > 0.
        GgxSmithVisibilityQuery::new(0.6, 0.4, 0.0),
        // Asymmetric light/view with the direct-lighting k at perceptual 0.5
        // ((0.5 + 1)^2 / 8 = 0.28125).
        GgxSmithVisibilityQuery::new(0.3, 0.85, 0.281_25),
        GgxSmithVisibilityQuery::new(0.85, 0.3, 0.281_25),
        // Rough surface near k = 0.5.
        GgxSmithVisibilityQuery::new(0.5, 0.5, 0.5),
        // Smooth surface with a tiny IBL k (alpha^2 / 2 at small alpha).
        GgxSmithVisibilityQuery::new(0.75, 0.65, 0.005),
    ]
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn empty_input_returns_empty_without_dispatch() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping ggx_smith_visibility parity: no wgpu adapter on this host");
        return;
    };
    let gpu = GpuGgxSmithVisibility::new(&ctx);
    // An empty batch short-circuits before any dispatch (a storage buffer cannot
    // be zero-sized) and returns an empty vector.
    let got = gpu.evaluate(&ctx, &[]);
    assert!(got.is_empty(), "an empty batch produces no results");
}

#[test]
fn fixture_batch_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuGgxSmithVisibility::new(&ctx);
    check(&ctx, &gpu, &fixture_queries());
}

#[test]
fn grazing_light_drives_visibility_to_guarded_zero() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuGgxSmithVisibility::new(&ctx);
    // NoL at the grazing floor drives 4 * NoL * NoV under MIN_DENOM, so the
    // visibility collapses to the guarded 0.0.
    let q = GgxSmithVisibilityQuery::new(0.0, 0.6, 0.25);
    let got = gpu.evaluate(&ctx, &[q]);
    assert_eq!(got.len(), 1);
    check_one(0, &got[0], &oracle(&q));
    assert!(
        got[0].visibility.abs() <= ABS,
        "grazing light must guard visibility to zero: {:?}",
        got[0]
    );
}

#[test]
fn grazing_view_drives_visibility_to_guarded_zero() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuGgxSmithVisibility::new(&ctx);
    // Symmetric guard: NoV at the grazing floor also collapses the visibility.
    let q = GgxSmithVisibilityQuery::new(0.55, 0.0, 0.3);
    let got = gpu.evaluate(&ctx, &[q]);
    assert_eq!(got.len(), 1);
    check_one(0, &got[0], &oracle(&q));
    assert!(
        got[0].visibility.abs() <= ABS,
        "grazing view must guard visibility to zero: {:?}",
        got[0]
    );
    // G1(NoL) stays finite and positive even while the visibility guards to 0.
    assert!(
        got[0].g1_l > 0.0,
        "the one-sided light masking stays positive off grazing: {:?}",
        got[0]
    );
}

#[test]
fn mirror_smooth_k_zero_makes_g1_identity() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuGgxSmithVisibility::new(&ctx);
    // With k = 0 the denominator x * (1 - 0) + 0 = x cancels, so G1 = 1 for any
    // positive cosine and G = 1.
    let q = GgxSmithVisibilityQuery::new(0.42, 0.73, 0.0);
    let got = gpu.evaluate(&ctx, &[q]);
    assert_eq!(got.len(), 1);
    check_one(0, &got[0], &oracle(&q));
    assert!(
        close(got[0].g1_l, 1.0) && close(got[0].g1_v, 1.0) && close(got[0].g_separable, 1.0),
        "k = 0 makes every one-sided term the identity: {:?}",
        got[0]
    );
}

#[test]
fn visibility_matches_g_over_foreshortening() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuGgxSmithVisibility::new(&ctx);
    // The reported visibility must equal the reported G divided by the
    // foreshortening factor, both read back from the device.
    let q = GgxSmithVisibilityQuery::new(0.65, 0.45, 0.2);
    let got = gpu.evaluate(&ctx, &[q]);
    assert_eq!(got.len(), 1);
    check_one(0, &got[0], &oracle(&q));
    let expected = got[0].g_separable / (4.0 * 0.65 * 0.45);
    assert!(
        close(got[0].visibility, expected),
        "visibility must equal G / (4 NoL NoV): {:?}",
        got[0]
    );
}

#[test]
fn masking_is_symmetric_under_cosine_swap() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuGgxSmithVisibility::new(&ctx);
    // Swapping NoL and NoV leaves the separable G and the visibility unchanged
    // (the product and the symmetric divisor are both commutative).
    let forward = GgxSmithVisibilityQuery::new(0.3, 0.85, 0.281_25);
    let swapped = GgxSmithVisibilityQuery::new(0.85, 0.3, 0.281_25);
    let got = gpu.evaluate(&ctx, &[forward, swapped]);
    assert_eq!(got.len(), 2);
    check_one(0, &got[0], &oracle(&forward));
    check_one(1, &got[1], &oracle(&swapped));
    assert!(
        close(got[0].g_separable, got[1].g_separable)
            && close(got[0].visibility, got[1].visibility),
        "swapping the cosines must leave G and V unchanged: {:?} vs {:?}",
        got[0],
        got[1]
    );
}

#[test]
fn g1_increases_with_cosine() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuGgxSmithVisibility::new(&ctx);
    // G1 is monotone increasing in its cosine for a fixed k; three ascending
    // cosines must report ascending G1(NoL).
    let k = 0.281_25;
    let lo = GgxSmithVisibilityQuery::new(0.2, 0.5, k);
    let mid = GgxSmithVisibilityQuery::new(0.5, 0.5, k);
    let hi = GgxSmithVisibilityQuery::new(0.9, 0.5, k);
    let got = gpu.evaluate(&ctx, &[lo, mid, hi]);
    assert_eq!(got.len(), 3);
    check_one(0, &got[0], &oracle(&lo));
    check_one(1, &got[1], &oracle(&mid));
    check_one(2, &got[2], &oracle(&hi));
    assert!(
        got[0].g1_l < got[1].g1_l && got[1].g1_l < got[2].g1_l,
        "G1 must increase with the cosine: {} {} {}",
        got[0].g1_l,
        got[1].g1_l,
        got[2].g1_l
    );
}

#[test]
fn masking_outputs_stay_in_unit_range() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuGgxSmithVisibility::new(&ctx);
    // Each G1 and the product G lie in 0..=1 across a span of well-conditioned
    // geometries and roughness.
    let queries = fixture_queries();
    let got = gpu.evaluate(&ctx, &queries);
    assert_eq!(got.len(), queries.len());
    for (idx, (q, result)) in queries.iter().zip(got.iter()).enumerate() {
        check_one(idx, result, &oracle(q));
        assert!(
            (0.0..=1.0).contains(&result.g1_l)
                && (0.0..=1.0).contains(&result.g1_v)
                && (0.0..=1.0).contains(&result.g_separable),
            "the masking terms stay in the unit range: {result:?}"
        );
    }
}

#[test]
fn rough_surface_masks_more_than_smooth() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuGgxSmithVisibility::new(&ctx);
    // A larger k (rougher surface) shadows more, so the separable G drops at the
    // same geometry.
    let smooth = GgxSmithVisibilityQuery::new(0.5, 0.5, 0.02);
    let rough = GgxSmithVisibilityQuery::new(0.5, 0.5, 0.5);
    let got = gpu.evaluate(&ctx, &[smooth, rough]);
    assert_eq!(got.len(), 2);
    check_one(0, &got[0], &oracle(&smooth));
    check_one(1, &got[1], &oracle(&rough));
    assert!(
        got[0].g_separable > got[1].g_separable,
        "a rougher surface masks more light: smooth {} vs rough {}",
        got[0].g_separable,
        got[1].g_separable
    );
}

#[test]
fn random_sweep_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuGgxSmithVisibility::new(&ctx);
    let mut state = 0x1234_5678_9abc_def0_u64;
    let mut queries = fixture_queries();
    // Several workgroups' worth of random, well-conditioned queries pin every
    // reported masking term across a wide span of cosines and roughness.
    for _ in 0..512 {
        queries.push(random_query(&mut state));
    }
    check(&ctx, &gpu, &queries);
}
