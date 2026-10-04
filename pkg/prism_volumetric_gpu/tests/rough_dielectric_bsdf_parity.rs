//! Real-device parity for the rough-dielectric microfacet `BSDF` twin:
//! [`GpuRoughDielectric`](prism_volumetric_gpu::rough_dielectric_bsdf::GpuRoughDielectric)
//! must reproduce the `CPU` closed form of the reference tracer
//! `prism_render_architecture::reference_pt::rough_dielectric` — the full `BSDF`
//! value `RoughDielectric::evaluate` and the mixture solid-angle density
//! `RoughDielectric::pdf` — across near-normal reflection, grazing incidence,
//! transmission through the interface, an inverted (`ior < 1`) interface and a
//! randomized, well-conditioned sweep compared query-for-query.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Oracle
//!
//! This wave forbids depending on the golden crate, so the host oracle is an
//! *independent* reimplementation of the same closed form: the `GGX`
//! (Trowbridge-Reitz) normal distribution `D`, the height-correlated Smith
//! masking-shadowing `G2`/`G1`, the unpolarized dielectric `Fresnel` `F`
//! (`PBRT` v4 `FrDielectric`), the generalized half vector `etap*wi + wo`, and
//! the visible-normal density folded through the reflect/refract Jacobian and
//! the `Fresnel`-proportional lobe-selection probability. Because the reference
//! and this oracle are both scalar `f32`, a `GPU == oracle` pass is direct
//! evidence the ported kernel computes the same scattering the reference does.
//!
//! # Parity criterion
//!
//! Both outputs thread through `sqrt`, products and quotients, so a `GPU` result
//! may land a few units in the last place from the scalar oracle; each
//! continuous channel of the value and the density is asserted within
//! `abs_diff <= 1e-4` or `rel_diff <= 1e-3` (relative floor `1e-6`). The
//! discrete `valid` word is asserted exactly on every query; the `reflect_flag`
//! lobe discriminant is asserted exactly only when `valid == 1`, since the sign
//! of `cos_i * cos_o` is immaterial at a grazing degeneracy.
//!
//! # Conditioning
//!
//! The randomized sweep forces `wo` into the shading normal's hemisphere
//! (`cos_o > 0`) so the microfacet `Fresnel` never straddles total internal
//! reflection, and rejects any draw with `|cos_o| < 0.1`, `|cos_i| < 0.1` or
//! `|cos_i * cos_o| < 0.02` so no query lands on the grazing cliff or on the
//! `reflect`/`transmit` branch boundary where the lobe discriminant flips. The
//! relative index is drawn in `[1.1, 2.0]` and the roughness in `[0.1, 0.9]`,
//! keeping every `GGX` and `Fresnel` intermediate a well-conditioned
//! non-negative square root or guarded quotient. The grazing and inverted-index
//! extremes are instead pinned by the dedicated named fixtures.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::reference_pt::rough_dielectric`；无第三方引擎源码或衍生代码。

use prism_volumetric_gpu::rough_dielectric_bsdf::{
    GpuRoughDielectric, RoughDielectricQuery, RoughDielectricResult,
};
use prism_volumetric_gpu::GpuContext;

/// Absolute bound on each continuous output. A `GPU` `sqrt`/divide may land a
/// few units in the last place from the scalar oracle; `1e-4` admits that legal
/// slack while still failing a wrong port.
const ABS: f32 = 1.0e-4;

/// Relative bound on each continuous output, applied for larger magnitudes where
/// a few units in the last place exceed the absolute floor.
const REL: f32 = 1.0e-3;

/// Floor on the relative-error denominator so a near-zero expected value does
/// not inflate the relative error.
const REL_FLOOR: f32 = 1.0e-6;

/// A direction cosine below this magnitude is a grazing degeneracy; mirrors the
/// kernel and golden `COS_EPS`.
const COS_EPS: f32 = 1.0e-8;

/// Squared-length floor below which a vector is the zero vector; mirrors the
/// kernel and golden `EPS_LEN_SQ`.
const EPS_LEN_SQ: f32 = 1.0e-12;

/// Smallest `GGX` width; mirrors the kernel and golden `MIN_ALPHA`.
const MIN_ALPHA: f32 = 1.0e-3;

/// Reciprocal of pi, matching the kernel's inline literal exactly so the two
/// `GGX` distributions share an identical constant.
const INV_PI: f32 = core::f32::consts::FRAC_1_PI;

/// Returns whether `a` and `b` agree within the given absolute or relative
/// bound (relative error floored at `REL_FLOOR`).
fn close(a: f32, b: f32, abs_eps: f32, rel_eps: f32) -> bool {
    let abs = (a - b).abs();
    let rel = abs / a.abs().max(b.abs()).max(REL_FLOOR);
    abs <= abs_eps || rel <= rel_eps
}

/// Dot product of two three-vectors.
fn dot3(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// Component-wise sum of two three-vectors.
fn add3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

/// Scales a three-vector by a scalar.
fn scale3(a: [f32; 3], s: f32) -> [f32; 3] {
    [a[0] * s, a[1] * s, a[2] * s]
}

/// Returns the unit vector along `v`, or the zero vector when `v` is numerically
/// the zero vector, matching the golden `Vec3::normalize_or_zero`.
fn normalize_or_zero3(v: [f32; 3]) -> [f32; 3] {
    let len_sq = dot3(v, v);
    if len_sq > EPS_LEN_SQ {
        scale3(v, 1.0 / len_sq.sqrt())
    } else {
        [0.0, 0.0, 0.0]
    }
}

/// Flips `a` so it lies in the same hemisphere as `reference`, matching the
/// golden `Vec3::faced_toward`.
fn faced_toward3(a: [f32; 3], reference: [f32; 3]) -> [f32; 3] {
    if dot3(a, reference) < 0.0 {
        scale3(a, -1.0)
    } else {
        a
    }
}

/// Returns `x` squared, spelled out to mirror the kernel's `sqr`.
fn sqr(x: f32) -> f32 {
    x * x
}

/// The `GGX` normal distribution `D` for a half vector whose absolute cosine to
/// the normal is `cos_h`. Zero for a back-facing half vector.
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
/// `cos_w`. Normal incidence returns zero.
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

/// The Smith single-direction masking term `G1` in `[0, 1]`.
fn ggx_g1(alpha: f32, cos_w: f32) -> f32 {
    1.0 / (1.0 + ggx_lambda(alpha, cos_w))
}

/// The height-correlated Smith masking-shadowing term `G2`.
fn ggx_g2(alpha: f32, cos_o: f32, cos_i: f32) -> f32 {
    1.0 / (1.0 + ggx_lambda(alpha, cos_o) + ggx_lambda(alpha, cos_i))
}

/// The unpolarized `Fresnel` reflectance of a dielectric interface in the
/// relative-index form (`PBRT` v4 `FrDielectric`); returns `1` beyond the
/// critical angle (total internal reflection).
fn fr_dielectric(cos_i_in: f32, eta_in: f32) -> f32 {
    let mut cos_i = cos_i_in.clamp(-1.0, 1.0);
    let mut eta = eta_in;
    if cos_i < 0.0 {
        eta = 1.0 / eta;
        cos_i = -cos_i;
    }
    let sin2_i = (1.0 - cos_i * cos_i).max(0.0);
    let sin2_t = sin2_i / (eta * eta);
    if sin2_t >= 1.0 {
        return 1.0;
    }
    let cos_t = (1.0 - sin2_t).max(0.0).sqrt();
    let r_parl = (eta * cos_i - cos_t) / (eta * cos_i + cos_t);
    let r_perp = (cos_i - eta * cos_t) / (cos_i + eta * cos_t);
    0.5 * (r_parl * r_parl + r_perp * r_perp)
}

/// The visible-normal (`VNDF`) solid-angle density of the microfacet normal `wm`
/// for the view direction `wo`.
fn visible_normal_pdf(alpha: f32, wo: [f32; 3], wm: [f32; 3], normal: [f32; 3]) -> f32 {
    let cos_o = dot3(normal, wo);
    if cos_o.abs() < COS_EPS {
        return 0.0;
    }
    let d = ggx_distribution(alpha, dot3(normal, wm).abs());
    let g1 = ggx_g1(alpha, cos_o);
    d * g1 * dot3(wo, wm).abs() / cos_o.abs()
}

/// Computes the expected result from the independent host oracle, mirroring the
/// kernel's branch structure exactly so the discrete flags agree.
fn oracle(q: &RoughDielectricQuery) -> RoughDielectricResult {
    let wo = q.wo;
    let wi = q.wi;
    let normal = q.normal;
    let ior = q.ior;

    let rough = q.roughness.clamp(0.0, 1.0);
    let alpha = (rough * rough).max(MIN_ALPHA);

    let cos_o = dot3(normal, wo);
    let cos_i = dot3(normal, wi);
    let reflect = cos_i * cos_o > 0.0;

    let mut value = [0.0_f32; 3];
    let mut pdf = 0.0_f32;
    let mut valid = 0_u32;

    if cos_o.abs() >= COS_EPS && cos_i.abs() >= COS_EPS {
        let etap = if reflect {
            1.0
        } else if cos_o > 0.0 {
            ior
        } else {
            1.0 / ior
        };
        let wm_raw = add3(scale3(wi, etap), wo);
        let wm_len_sq = dot3(wm_raw, wm_raw);
        if wm_len_sq > EPS_LEN_SQ {
            let wm = faced_toward3(normalize_or_zero3(wm_raw), normal);
            let behind = dot3(wm, wi) * cos_i < 0.0 || dot3(wm, wo) * cos_o < 0.0;
            if !behind {
                valid = 1;
                let d = ggx_distribution(alpha, dot3(normal, wm).abs());
                let g2 = ggx_g2(alpha, cos_o, cos_i);
                let f = fr_dielectric(dot3(wo, wm), ior);

                if reflect {
                    let denom = 4.0 * (cos_i * cos_o).abs();
                    if denom > 0.0 {
                        value = scale3(q.reflectance, d * g2 * f / denom);
                    }
                } else {
                    let denom = sqr(dot3(wi, wm) + dot3(wo, wm) / etap) * cos_i * cos_o;
                    if denom.abs() >= COS_EPS {
                        let ft = d * (1.0 - f) * g2 * (dot3(wi, wm) * dot3(wo, wm) / denom).abs()
                            / sqr(etap);
                        value = scale3(q.transmittance, ft);
                    }
                }

                let pr = f;
                let pt = 1.0 - f;
                if pr + pt > 0.0 {
                    let vndf = visible_normal_pdf(alpha, wo, wm, normal);
                    if reflect {
                        let woh = dot3(wo, wm).abs();
                        if woh >= COS_EPS {
                            pdf = vndf / (4.0 * woh) * pr / (pr + pt);
                        }
                    } else {
                        let denom2 = sqr(dot3(wi, wm) + dot3(wo, wm) / etap);
                        if denom2 > 0.0 {
                            let dwm_dwi = dot3(wi, wm).abs() / denom2;
                            pdf = vndf * dwm_dwi * pt / (pr + pt);
                        }
                    }
                }
            }
        }
    }

    RoughDielectricResult {
        value,
        pdf,
        valid,
        reflect_flag: u32::from(reflect),
    }
}

/// Pins one `GPU` result against the host oracle: `valid` exactly on every
/// query, each value channel and the density under tolerance, and the
/// `reflect_flag` discriminant exactly only when the pair is non-degenerate.
fn check_one(idx: usize, got: &RoughDielectricResult, want: &RoughDielectricResult) {
    assert_eq!(
        got.valid, want.valid,
        "query {idx} valid: gpu {} vs cpu {}",
        got.valid, want.valid
    );
    for (ch, (g, w)) in got.value.iter().zip(want.value.iter()).enumerate() {
        assert!(
            close(*g, *w, ABS, REL),
            "query {idx} value[{ch}]: gpu {g} vs cpu {w}"
        );
    }
    assert!(
        close(got.pdf, want.pdf, ABS, REL),
        "query {idx} pdf: gpu {} vs cpu {}",
        got.pdf,
        want.pdf
    );
    if want.valid == 1 {
        assert_eq!(
            got.reflect_flag, want.reflect_flag,
            "query {idx} reflect_flag: gpu {} vs cpu {}",
            got.reflect_flag, want.reflect_flag
        );
    }
}

/// Dispatches every query and pins each result against the oracle.
fn check(ctx: &GpuContext, gpu: &GpuRoughDielectric, queries: &[RoughDielectricQuery]) {
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

/// Normalizes a three-vector with `sqrt` only (no transcendental); used to build
/// the named direction fixtures.
fn norm3(v: [f32; 3]) -> [f32; 3] {
    scale3(v, 1.0 / dot3(v, v).sqrt())
}

/// Draws a random unit vector by rejection sampling the cube `[-1, 1]^3` and
/// normalizing, keeping only well-separated-from-zero draws so the direction is
/// numerically stable. Only `sqrt` and integer arithmetic appear.
fn rand_unit(state: &mut u64) -> [f32; 3] {
    loop {
        let v = [
            uniform(state, -1.0, 1.0),
            uniform(state, -1.0, 1.0),
            uniform(state, -1.0, 1.0),
        ];
        let len_sq = dot3(v, v);
        if (0.05..=1.0).contains(&len_sq) {
            return scale3(v, 1.0 / len_sq.sqrt());
        }
    }
}

/// Builds one well-conditioned random query: a random shading normal, `wo`
/// forced into its hemisphere (so `cos_o > 0`), and a free `wi` (either side, to
/// cover both the reflection and transmission lobes). The reject-sampling bounds
/// keep every query clear of the grazing cliff and the `reflect`/`transmit`
/// branch boundary (see `# Conditioning`).
fn random_query(state: &mut u64) -> RoughDielectricQuery {
    loop {
        let normal = rand_unit(state);
        let mut wo = rand_unit(state);
        let wi = rand_unit(state);
        if dot3(normal, wo) < 0.0 {
            wo = scale3(wo, -1.0);
        }
        let cos_o = dot3(normal, wo);
        let cos_i = dot3(normal, wi);
        if cos_o < 0.1 || cos_i.abs() < 0.1 || (cos_i * cos_o).abs() < 0.02 {
            continue;
        }
        let ior = uniform(state, 1.1, 2.0);
        let roughness = uniform(state, 0.1, 0.9);
        let reflectance = [
            uniform(state, 0.1, 1.0),
            uniform(state, 0.1, 1.0),
            uniform(state, 0.1, 1.0),
        ];
        let transmittance = [
            uniform(state, 0.1, 1.0),
            uniform(state, 0.1, 1.0),
            uniform(state, 0.1, 1.0),
        ];
        return RoughDielectricQuery::new(
            ior,
            roughness,
            reflectance,
            transmittance,
            wo,
            wi,
            normal,
        );
    }
}

/// A fixed battery of named cases spanning near-normal reflection, grazing
/// incidence, transmission and an inverted (`ior < 1`) interface.
fn fixture_queries() -> Vec<RoughDielectricQuery> {
    let normal = [0.0, 0.0, 1.0];
    let tint_r = [0.9, 0.8, 0.7];
    let tint_t = [0.6, 0.7, 0.8];
    vec![
        // Air -> glass (ior = 1.5), near-normal reflection: both wo and wi sit
        // on the +z side, so the lobe is a reflection.
        RoughDielectricQuery::new(
            1.5,
            0.25,
            tint_r,
            tint_t,
            norm3([0.2, 0.1, 1.0]),
            norm3([-0.15, 0.05, 1.0]),
            normal,
        ),
        // A rougher reflection at a steeper angle.
        RoughDielectricQuery::new(
            1.5,
            0.6,
            tint_r,
            tint_t,
            norm3([0.5, -0.3, 1.0]),
            norm3([-0.4, 0.2, 1.0]),
            normal,
        ),
        // Grazing incidence: cos_o = 0 -> degenerate, valid = 0.
        RoughDielectricQuery::new(
            1.5,
            0.3,
            tint_r,
            tint_t,
            norm3([1.0, 0.0, 0.0]),
            norm3([0.1, 0.1, 1.0]),
            normal,
        ),
        // Air -> glass transmission: wo on +z, wi on -z (opposite sides).
        RoughDielectricQuery::new(
            1.5,
            0.3,
            tint_r,
            tint_t,
            norm3([0.1, 0.1, 1.0]),
            norm3([0.05, 0.1, -1.0]),
            normal,
        ),
        // A smoother transmission lobe.
        RoughDielectricQuery::new(
            1.5,
            0.15,
            tint_r,
            tint_t,
            norm3([0.05, 0.2, 1.0]),
            norm3([0.1, 0.05, -1.0]),
            normal,
        ),
        // Inverted interface (ior < 1): near-normal reflection, directions near
        // the normal to stay clear of total internal reflection.
        RoughDielectricQuery::new(
            1.0 / 1.5,
            0.3,
            tint_r,
            tint_t,
            norm3([0.1, 0.05, 1.0]),
            norm3([-0.1, 0.08, 1.0]),
            normal,
        ),
    ]
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn empty_input_returns_empty_without_dispatch() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping rough_dielectric_bsdf parity: no wgpu adapter on this host");
        return;
    };
    let gpu = GpuRoughDielectric::new(&ctx);
    // An empty batch short-circuits before any dispatch (a storage buffer cannot
    // be zero-sized) and returns an empty vector.
    let got = gpu.evaluate(&ctx, &[]);
    assert!(got.is_empty(), "an empty batch produces no results");
}

#[test]
fn near_normal_reflection_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuRoughDielectric::new(&ctx);
    let q = RoughDielectricQuery::new(
        1.5,
        0.25,
        [0.9, 0.8, 0.7],
        [0.6, 0.7, 0.8],
        norm3([0.2, 0.1, 1.0]),
        norm3([-0.15, 0.05, 1.0]),
        [0.0, 0.0, 1.0],
    );
    let got = gpu.evaluate(&ctx, &[q]);
    assert_eq!(got.len(), 1);
    check_one(0, &got[0], &oracle(&q));
    assert_eq!(got[0].valid, 1, "a near-normal pair is non-degenerate");
    assert_eq!(got[0].reflect_flag, 1, "same-side directions reflect");
}

#[test]
fn grazing_incidence_is_degenerate() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuRoughDielectric::new(&ctx);
    // wo along the surface tangent gives cos_o = 0, which short-circuits to a
    // zero value, a zero density and valid = 0.
    let q = RoughDielectricQuery::new(
        1.5,
        0.3,
        [0.9, 0.8, 0.7],
        [0.6, 0.7, 0.8],
        norm3([1.0, 0.0, 0.0]),
        norm3([0.1, 0.1, 1.0]),
        [0.0, 0.0, 1.0],
    );
    let got = gpu.evaluate(&ctx, &[q]);
    assert_eq!(got.len(), 1);
    check_one(0, &got[0], &oracle(&q));
    assert_eq!(got[0].valid, 0, "grazing incidence is degenerate");
    assert_eq!(
        got[0].value,
        [0.0, 0.0, 0.0],
        "a degenerate pair has no value"
    );
    assert_eq!(got[0].pdf, 0.0, "a degenerate pair has no density");
}

#[test]
fn transmission_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuRoughDielectric::new(&ctx);
    // Opposite sides of the normal select the transmission lobe.
    let q = RoughDielectricQuery::new(
        1.5,
        0.3,
        [0.9, 0.8, 0.7],
        [0.6, 0.7, 0.8],
        norm3([0.1, 0.1, 1.0]),
        norm3([0.05, 0.1, -1.0]),
        [0.0, 0.0, 1.0],
    );
    let got = gpu.evaluate(&ctx, &[q]);
    assert_eq!(got.len(), 1);
    check_one(0, &got[0], &oracle(&q));
    assert_eq!(got[0].valid, 1, "the transmission pair is non-degenerate");
    assert_eq!(got[0].reflect_flag, 0, "opposite-side directions transmit");
}

#[test]
fn fixture_batch_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuRoughDielectric::new(&ctx);
    check(&ctx, &gpu, &fixture_queries());
}

#[test]
fn random_sweep_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuRoughDielectric::new(&ctx);
    let mut state = 0x1234_5678_9abc_def0_u64;
    let mut queries = fixture_queries();
    // Several workgroups' worth of random, well-conditioned queries pin every
    // reported value and density across a wide span of interfaces and angles.
    for _ in 0..512 {
        queries.push(random_query(&mut state));
    }
    check(&ctx, &gpu, &queries);
}
