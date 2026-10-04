//! Real-device parity for the coupled diffuse-specular (Ashikhmin-Shirley
//! `Fresnel` blend) twin:
//! [`GpuFresnelBlend`](prism_volumetric_gpu::fresnel_blend_bsdf::GpuFresnelBlend)
//! must reproduce the `CPU` closed forms of
//! `prism_render_architecture::reference_pt::fresnel_blend` — the coupled
//! `BRDF` value `FresnelBlend::evaluate` and the two-strategy mixture density
//! `FresnelBlend::pdf` — across normal, oblique and near-grazing incidence plus
//! a randomized sweep compared query-for-query.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Oracle
//!
//! This wave forbids depending on the golden crate, so the host oracle is an
//! *independent* reimplementation of the same closed form: the
//! Ashikhmin-Shirley diffuse term with its `28 / (23 pi)` normalization and
//! grazing coupling, the shared isotropic GGX microfacet specular term with a
//! Schlick `Fresnel` tint, and the balanced mixture of the cosine-hemisphere
//! diffuse density and the GGX visible-normal reflection density. Because the
//! reference and this oracle are both scalar `f32`, a `GPU == oracle` pass is
//! direct evidence the ported kernel computes the same value and density the
//! reference does.
//!
//! # Parity criterion
//!
//! Both outputs thread through `sqrt`, products and quotients, so a `GPU`
//! result may land a few units in the last place from the scalar oracle; each
//! value channel and the density are asserted within `abs_diff <= 1e-4` or
//! `rel_diff <= 1e-3`, with the relative error floored at `1e-6` so a near-zero
//! expected value does not inflate it. The `valid` flag — set when both the
//! view and light cosines are positive — is compared exactly.
//!
//! # Conditioning
//!
//! The randomized sweep draws a random orientation for the normal and for both
//! directions, rejecting any draw whose view or light cosine is below `0.1`.
//! Keeping both directions comfortably inside the upper hemisphere does two
//! things: it stays clear of the `cos_o <= 0` / `cos_i <= 0` branch cliff that
//! would otherwise make the `valid` flag straddle a floating-point tie, and it
//! guarantees `wo + wi` has a strictly positive projection on the normal so the
//! half vector never degenerates. Inside that box every intermediate is a
//! well-conditioned non-negative square root or guarded quotient, so a `GPU`
//! evaluation lands a few units in the last place from the scalar oracle and
//! never straddles a discrete decision. The named fixtures pin the clear-coat
//! normal/oblique/near-grazing cases and the below-surface `valid = 0` case.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::reference_pt::fresnel_blend`；无第三方引擎源码或衍生代码。

use prism_volumetric_gpu::fresnel_blend_bsdf::{
    FresnelBlendQuery, FresnelBlendResult, GpuFresnelBlend,
};
use prism_volumetric_gpu::GpuContext;

/// Absolute bound on each value channel and on the density. A `GPU`
/// `sqrt`/divide may land a few units in the last place from the scalar oracle;
/// `1e-4` admits that legal slack while still failing a wrong port.
const VALUE_ABS: f32 = 1.0e-4;

/// Relative bound on each value channel and on the density, applied for larger
/// magnitudes where a few units in the last place exceed the absolute floor.
const VALUE_REL: f32 = 1.0e-3;

/// Floor on the relative-error denominator so a near-zero expected value does
/// not inflate the relative error.
const REL_FLOOR: f32 = 1.0e-6;

/// The mathematical constant pi, matching the reference `core::f32::consts::PI`
/// so the diffuse normalization and the inverse-pi densities use bit-identical
/// `f32` constants on both sides.
const PI: f32 = core::f32::consts::PI;

/// The Ashikhmin-Shirley diffuse normalization constant `28 / (23 pi)`.
const DIFFUSE_NORM: f32 = 28.0 / (23.0 * PI);

/// The probability of the diffuse sampling strategy in the two-lobe mixture.
const DIFFUSE_SAMPLE_PROBABILITY: f32 = 0.5;

/// Reciprocal of pi, used by the cosine-hemisphere and GGX densities.
const INV_PI: f32 = 1.0 / PI;

/// Smallest GGX width; below this the lobe is numerically a perfect mirror.
const MIN_ALPHA: f32 = 1.0e-3;

/// Squared-length threshold below which a vector is treated as the zero vector.
const EPS_LEN_SQ: f32 = 1.0e-12;

/// Returns whether `a` and `b` agree within the given absolute or relative
/// bound (relative error floored at `REL_FLOOR`).
fn close(a: f32, b: f32, abs_eps: f32, rel_eps: f32) -> bool {
    let abs = (a - b).abs();
    let rel = abs / a.abs().max(b.abs()).max(REL_FLOOR);
    abs <= abs_eps || rel <= rel_eps
}

/// Dot product of two three-vectors.
fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// Component sum of two three-vectors.
fn add(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

/// Component difference of two three-vectors.
fn sub(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

/// Scales a three-vector by a scalar.
fn scale(a: [f32; 3], s: f32) -> [f32; 3] {
    [a[0] * s, a[1] * s, a[2] * s]
}

/// Component-wise product of two three-vectors.
fn mul(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] * b[0], a[1] * b[1], a[2] * b[2]]
}

/// Unit vector along `v`, or the zero vector when `v` is numerically zero, so
/// the division never yields a `NaN`. Mirrors the reference
/// `Vec3::normalize_or_zero` and the kernel helper of the same name.
fn normalize_or_zero(v: [f32; 3]) -> [f32; 3] {
    let len_sq = dot(v, v);
    if len_sq > EPS_LEN_SQ {
        let inv = 1.0 / len_sq.sqrt();
        scale(v, inv)
    } else {
        [0.0, 0.0, 0.0]
    }
}

/// Raises `x` to the fifth power without a transcendental call.
fn quintic(x: f32) -> f32 {
    let x2 = x * x;
    x2 * x2 * x
}

/// Schlick's per-channel `Fresnel`: `f0 + (1 - f0) (1 - cos)^5`.
fn fresnel_schlick(f0: [f32; 3], cos_theta: f32) -> [f32; 3] {
    let c = (1.0 - cos_theta).clamp(0.0, 1.0);
    let c2 = c * c;
    let c5 = c2 * c2 * c;
    add(f0, scale(sub([1.0, 1.0, 1.0], f0), c5))
}

/// The GGX width `alpha = clamp(roughness, 0, 1)^2`, clamped up to `MIN_ALPHA`.
fn alpha_from_roughness(roughness: f32) -> f32 {
    let r = roughness.clamp(0.0, 1.0);
    (r * r).max(MIN_ALPHA)
}

/// The GGX normal distribution `D(h)`; zero for a back-facing half vector.
fn ggx_distribution(alpha: f32, cos_h: f32) -> f32 {
    if cos_h <= 0.0 {
        return 0.0;
    }
    let a2 = alpha * alpha;
    let c2 = cos_h * cos_h;
    let denom = c2 * (a2 - 1.0) + 1.0;
    a2 * INV_PI / (denom * denom)
}

/// The Smith `Lambda` auxiliary for a direction whose cosine to the normal is
/// `cos_w`.
fn ggx_lambda(alpha: f32, cos_w: f32) -> f32 {
    let c = cos_w.abs();
    if c >= 1.0 {
        return 0.0;
    }
    let c2 = c * c;
    let tan2 = (1.0 - c2) / c2;
    let a2 = alpha * alpha;
    0.5 * ((1.0 + a2 * tan2).sqrt() - 1.0)
}

/// The Smith single-direction masking term `G1(w)` in `[0, 1]`.
fn ggx_g1(alpha: f32, cos_w: f32) -> f32 {
    1.0 / (1.0 + ggx_lambda(alpha, cos_w))
}

/// The GGX visible-normal reflection density `G1(wo) D(h) / (4 cos_o)`.
fn ggx_reflection_pdf(alpha: f32, cos_o: f32, cos_h: f32) -> f32 {
    if cos_o <= 0.0 {
        return 0.0;
    }
    ggx_g1(alpha, cos_o) * ggx_distribution(alpha, cos_h) / (4.0 * cos_o)
}

/// The cosine-weighted hemisphere density `max(n . wi, 0) / pi`.
fn cosine_hemisphere_pdf(n: [f32; 3], wi: [f32; 3]) -> f32 {
    dot(n, wi).max(0.0) * INV_PI
}

/// Computes the expected result from the independent host oracle, the faithful
/// reference the `GPU` is pinned against. Mirrors the kernel line for line.
fn oracle(q: &FresnelBlendQuery) -> FresnelBlendResult {
    let alpha = alpha_from_roughness(q.roughness);
    let cos_o = dot(q.normal, q.wo);
    let cos_i = dot(q.normal, q.wi);

    if cos_o <= 0.0 || cos_i <= 0.0 {
        return FresnelBlendResult {
            value: [0.0, 0.0, 0.0],
            pdf: 0.0,
            valid: 0,
        };
    }

    // Ashikhmin-Shirley diffuse term.
    let fi = 1.0 - quintic(1.0 - 0.5 * cos_i);
    let fo = 1.0 - quintic(1.0 - 0.5 * cos_o);
    let diffuse = scale(
        mul(q.diffuse, sub([1.0, 1.0, 1.0], q.specular)),
        DIFFUSE_NORM * fi * fo,
    );

    // GGX specular term with a Schlick Fresnel tint.
    let sum = add(q.wo, q.wi);
    let micro_h = normalize_or_zero(sum);
    let half_len_sq = dot(micro_h, micro_h);
    let mut specular = [0.0, 0.0, 0.0];
    if half_len_sq > EPS_LEN_SQ {
        let cos_h = dot(q.normal, micro_h);
        if cos_h > 0.0 {
            let woh = dot(q.wo, micro_h);
            let denom = 4.0 * woh.abs() * cos_o.max(cos_i);
            if denom > 0.0 {
                let d = ggx_distribution(alpha, cos_h);
                let fresnel = fresnel_schlick(q.specular, woh.max(0.0));
                specular = scale(fresnel, d / denom);
            }
        }
    }
    let value = add(diffuse, specular);

    // Two-strategy mixture density.
    let diffuse_pdf = cosine_hemisphere_pdf(q.normal, q.wi);
    let specular_pdf = if half_len_sq > EPS_LEN_SQ {
        ggx_reflection_pdf(alpha, cos_o, dot(q.normal, micro_h))
    } else {
        0.0
    };
    let pdf = DIFFUSE_SAMPLE_PROBABILITY * diffuse_pdf
        + (1.0 - DIFFUSE_SAMPLE_PROBABILITY) * specular_pdf;

    FresnelBlendResult {
        value,
        pdf,
        valid: 1,
    }
}

/// Pins one `GPU` result against the host oracle: each value channel and the
/// density under the shared bound, the `valid` flag exactly.
fn check_one(idx: usize, got: &FresnelBlendResult, want: &FresnelBlendResult) {
    assert_eq!(
        got.valid, want.valid,
        "query {idx} valid: gpu {} vs cpu {}",
        got.valid, want.valid
    );
    assert!(
        close(got.value[0], want.value[0], VALUE_ABS, VALUE_REL),
        "query {idx} value_r: gpu {} vs cpu {}",
        got.value[0],
        want.value[0]
    );
    assert!(
        close(got.value[1], want.value[1], VALUE_ABS, VALUE_REL),
        "query {idx} value_g: gpu {} vs cpu {}",
        got.value[1],
        want.value[1]
    );
    assert!(
        close(got.value[2], want.value[2], VALUE_ABS, VALUE_REL),
        "query {idx} value_b: gpu {} vs cpu {}",
        got.value[2],
        want.value[2]
    );
    assert!(
        close(got.pdf, want.pdf, VALUE_ABS, VALUE_REL),
        "query {idx} pdf: gpu {} vs cpu {}",
        got.pdf,
        want.pdf
    );
}

/// Dispatches every query and pins each result against the oracle.
fn check(ctx: &GpuContext, gpu: &GpuFresnelBlend, queries: &[FresnelBlendQuery]) {
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

/// Draws a random unit vector by rejecting draws outside a thin spherical
/// shell (so the normalization is well conditioned) and normalizing. Uses only
/// `sqrt` and arithmetic.
fn random_unit(state: &mut u64) -> [f32; 3] {
    loop {
        let v = [
            uniform(state, -1.0, 1.0),
            uniform(state, -1.0, 1.0),
            uniform(state, -1.0, 1.0),
        ];
        let len_sq = dot(v, v);
        if (0.05..=1.0).contains(&len_sq) {
            let inv = 1.0 / len_sq.sqrt();
            return scale(v, inv);
        }
    }
}

/// Builds one well-conditioned random query: a random normal and two random
/// directions, redrawn until both the view and light cosines exceed `0.1`.
/// That keeps both directions comfortably in the upper hemisphere, so `valid`
/// is unambiguously `1`, `wo + wi` cannot degenerate, and every intermediate is
/// a well-conditioned square root or guarded quotient (see `# Conditioning`).
fn random_query(state: &mut u64) -> FresnelBlendQuery {
    let normal = random_unit(state);
    let (wo, wi) = loop {
        let wo = random_unit(state);
        let wi = random_unit(state);
        if dot(normal, wo) > 0.1 && dot(normal, wi) > 0.1 {
            break (wo, wi);
        }
    };
    let diffuse = [
        uniform(state, 0.0, 1.0),
        uniform(state, 0.0, 1.0),
        uniform(state, 0.0, 1.0),
    ];
    let specular = [
        uniform(state, 0.0, 0.1),
        uniform(state, 0.0, 0.1),
        uniform(state, 0.0, 0.1),
    ];
    let roughness = uniform(state, 0.05, 1.0);
    FresnelBlendQuery::new(diffuse, specular, roughness, wo, wi, normal)
}

/// A fixed battery of named clear-coat cases: normal, oblique and near-grazing
/// (all `valid`), plus a below-surface case that must report `valid = 0`.
fn fixture_queries() -> Vec<FresnelBlendQuery> {
    let up = [0.0, 0.0, 1.0];
    let diffuse = [0.5, 0.5, 0.5];
    let specular = [0.04, 0.04, 0.04];
    vec![
        // Clear coat, normal incidence: view and light both straight up.
        FresnelBlendQuery::new(diffuse, specular, 0.1, up, up, up),
        // Clear coat, oblique: both directions tilted but well above the plane.
        FresnelBlendQuery::new(
            diffuse,
            specular,
            0.25,
            normalize_or_zero([0.5, 0.0, 1.0]),
            normalize_or_zero([-0.3, 0.2, 1.0]),
            up,
        ),
        // Clear coat, near grazing but still valid (cosines around 0.15).
        FresnelBlendQuery::new(
            diffuse,
            specular,
            0.5,
            normalize_or_zero([2.0, 0.0, 0.3]),
            normalize_or_zero([-2.0, 0.0, 0.3]),
            up,
        ),
        // Rougher surface, oblique, coloured substrate.
        FresnelBlendQuery::new(
            [0.8, 0.2, 0.1],
            [0.04, 0.04, 0.04],
            0.8,
            normalize_or_zero([0.2, 0.6, 1.0]),
            normalize_or_zero([-0.4, -0.1, 0.9]),
            up,
        ),
        // Below surface: light under the plane, must report valid = 0.
        FresnelBlendQuery::new(diffuse, specular, 0.3, up, [0.0, 0.0, -1.0], up),
        // Below surface: view under the plane, must report valid = 0.
        FresnelBlendQuery::new(diffuse, specular, 0.3, [0.0, 0.0, -1.0], up, up),
    ]
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn empty_input_returns_empty_without_dispatch() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping fresnel_blend_bsdf parity: no wgpu adapter on this host");
        return;
    };
    let gpu = GpuFresnelBlend::new(&ctx);
    // An empty batch short-circuits before any dispatch (a storage buffer cannot
    // be zero-sized) and returns an empty vector.
    let got = gpu.evaluate(&ctx, &[]);
    assert!(got.is_empty(), "an empty batch produces no results");
}

#[test]
fn clear_coat_normal_incidence_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuFresnelBlend::new(&ctx);
    let up = [0.0, 0.0, 1.0];
    let q = FresnelBlendQuery::new([0.5, 0.5, 0.5], [0.04, 0.04, 0.04], 0.1, up, up, up);
    let got = gpu.evaluate(&ctx, &[q]);
    assert_eq!(got.len(), 1);
    let want = oracle(&q);
    check_one(0, &got[0], &want);
    assert_eq!(got[0].valid, 1, "normal incidence is a valid configuration");
    assert!(
        got[0].value[0] > 0.0 && got[0].pdf > 0.0,
        "a valid clear-coat response is positive: {:?}",
        got[0]
    );
}

#[test]
fn below_surface_reports_zero_and_invalid() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuFresnelBlend::new(&ctx);
    let up = [0.0, 0.0, 1.0];
    // Light below the surface: the reference early-returns a zero value and
    // zero density, and the twin reports valid = 0.
    let q = FresnelBlendQuery::new(
        [0.5, 0.5, 0.5],
        [0.04, 0.04, 0.04],
        0.3,
        up,
        [0.0, 0.0, -1.0],
        up,
    );
    let got = gpu.evaluate(&ctx, &[q]);
    assert_eq!(got.len(), 1);
    check_one(0, &got[0], &oracle(&q));
    assert_eq!(got[0].valid, 0, "a below-surface direction is invalid");
    assert!(
        close(got[0].value[0], 0.0, VALUE_ABS, VALUE_REL)
            && close(got[0].value[1], 0.0, VALUE_ABS, VALUE_REL)
            && close(got[0].value[2], 0.0, VALUE_ABS, VALUE_REL)
            && close(got[0].pdf, 0.0, VALUE_ABS, VALUE_REL),
        "an invalid query reports a zero value and density: {:?}",
        got[0]
    );
}

#[test]
fn fixture_batch_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuFresnelBlend::new(&ctx);
    check(&ctx, &gpu, &fixture_queries());
}

#[test]
fn random_sweep_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuFresnelBlend::new(&ctx);
    let mut state = 0x1234_5678_9abc_def0_u64;
    let mut queries = fixture_queries();
    // Several workgroups' worth of random, well-conditioned queries pin the
    // value and density across a wide span of orientations and roughnesses.
    for _ in 0..512 {
        queries.push(random_query(&mut state));
    }
    check(&ctx, &gpu, &queries);
}
