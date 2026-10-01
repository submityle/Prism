//! Cook-Torrance microfacet specular `BRDF` core for particle shading
//! (design §16, §17).
//!
//! Lit sprite and mesh particles route through the shared `PBR` closure
//! ("`GGX` + 多散近似", design §17 line 415). This module owns the `CPU`
//! reference for that closure's **single-scattering** specular term so a future
//! `GPU` draw kernel can match it bit for bit. Energy-compensating multi-scatter
//! (design §17 line 417) layers on top of this core and is intentionally out of
//! scope here; this file is only the Cook-Torrance numerator/denominator.
//!
//! It is a sibling of the specular anti-aliasing reference
//! ([`super::specular_aa`], which only *widens* roughness and never evaluates a
//! `BRDF`) and the `Fresnel` rim term ([`super::fresnel_rim`], the stylized
//! edge light). This module deliberately reuses, rather than re-derives, their
//! shared pieces:
//!
//! * the `Schlick` `Fresnel` approximation [`super::fresnel_rim::fresnel_schlick`]
//!   (extended here to a three-channel `F0` via [`fresnel_schlick_f0_rgb`]), and
//! * the perceptual↔linear roughness squaring relation
//!   [`super::specular_aa::perceptual_to_linear_roughness`] /
//!   [`super::specular_aa::linear_to_perceptual_roughness`] (surfaced as
//!   [`alpha_from_perceptual_roughness`] /
//!   [`perceptual_roughness_from_alpha`]).
//!
//! # The microfacet model
//!
//! The Cook-Torrance specular `BRDF` for a half vector `h = normalize(l + v)`
//! is the product of three terms over the foreshortening denominator:
//!
//! ```text
//! f_spec(l, v) = D(NoH, alpha) * G(NoL, NoV, ...) * F(VoH) / (4 * NoL * NoV)
//! ```
//!
//! * **`D` — normal distribution function (`NDF`).** The `GGX` /
//!   Trowbridge-Reitz lobe [`ggx_distribution`]. It is a pure rational function
//!   of `NoH` and the linear roughness `alpha`; no transcendental is needed.
//! * **`G` — geometric masking-shadowing.** Two forms are provided: the
//!   separable `Schlick`-`GGX` [`smith_g_separable`] (parameterized by the `k`
//!   remap, with distinct **direct** [`smith_k_direct`] and **`IBL`**
//!   [`smith_k_ibl`] variants), and the height-correlated `Smith` `G2` of Heitz
//!   [`smith_g2_height_correlated`].
//! * **`F` — `Fresnel`.** The `Schlick` approximation evaluated at `VoH`,
//!   scalar ([`super::fresnel_rim::fresnel_schlick`]) or three-channel
//!   ([`fresnel_schlick_f0_rgb`]).
//!
//! Because the raw `G / (4 * NoL * NoV)` quotient is numerically delicate near
//! grazing angles, the height-correlated form folds the denominator into a
//! single **visibility** term `V = G / (4 * NoL * NoV)`
//! ([`visibility_smith_ggx_correlated`]), so the assembled lobe is written as
//! the better-conditioned `D * V * F` ([`specular_dvf_rgb`]) in addition to the
//! textbook `D * G * F / (4 * NoL * NoV)` ([`specular_dgf_scalar`]).
//!
//! # Determinism
//!
//! Only `f32::sqrt` and rational arithmetic are used (the `GGX` and `Smith`
//! masking terms are all rational / radical, never trigonometric or
//! exponential), so the reference is bit-reproducible against the eventual
//! `GPU` kernel. Integer-power helpers come from
//! [`super::fresnel_rim::power_u32`] by way of the reused `Fresnel` term.

use super::fresnel_rim::fresnel_schlick;
use super::specular_aa::{linear_to_perceptual_roughness, perceptual_to_linear_roughness};
use super::Vec3;

/// The circle constant, used only to normalize the `GGX` `NDF`.
///
/// This is a compile-time constant, not a transcendental function call, so it
/// does not violate the crate determinism lint.
const PI: f32 = core::f32::consts::PI;

/// Smallest linear roughness the `NDF` is evaluated at.
///
/// A true mirror (`alpha = 0`) makes the `GGX` lobe a Dirac delta that is
/// infinite at `NoH = 1` and zero elsewhere, which has no finite `f32`
/// representation. Clamping `alpha` to this floor keeps the peak large but
/// finite, so `alpha -> 0` still produces the expected sharp spike without a
/// division by zero or a `NaN`.
const MIN_ALPHA: f32 = 1.0e-4;

/// Generic denominator guard: quotients whose divisor falls below this collapse
/// to `0.0` rather than dividing by (near) zero and producing a `NaN`.
const MIN_DENOM: f32 = 1.0e-7;

/// Clamps a scalar into the `0..=1` range (used for cosine terms).
#[must_use]
fn clamp01(x: f32) -> f32 {
    x.clamp(0.0, 1.0)
}

/// Clamps a linear roughness to `[MIN_ALPHA, 1]`.
///
/// The lower floor keeps the mirror limit finite (see [`MIN_ALPHA`]); the upper
/// clamp keeps `alpha` a valid linear roughness.
#[must_use]
pub fn clamp_alpha(alpha: f32) -> f32 {
    alpha.clamp(MIN_ALPHA, 1.0)
}

/// Converts an artist-facing perceptual roughness to the `NDF` linear roughness
/// `alpha`.
///
/// This is the `alpha = perceptual^2` squaring relation; it reuses
/// [`super::specular_aa::perceptual_to_linear_roughness`] so the particle
/// subsystem keeps a single source of truth for the mapping.
#[must_use]
pub fn alpha_from_perceptual_roughness(perceptual: f32) -> f32 {
    perceptual_to_linear_roughness(perceptual)
}

/// Inverse of [`alpha_from_perceptual_roughness`] (`perceptual = sqrt(alpha)`).
///
/// Reuses [`super::specular_aa::linear_to_perceptual_roughness`].
#[must_use]
pub fn perceptual_roughness_from_alpha(alpha: f32) -> f32 {
    linear_to_perceptual_roughness(alpha)
}

/// The `GGX` / Trowbridge-Reitz normal distribution function `D(NoH, alpha)`.
///
/// ```text
/// D = alpha^2 / (pi * (NoH^2 * (alpha^2 - 1) + 1)^2)
/// ```
///
/// `n_dot_h` is clamped to `0..=1`, and `alpha` is clamped by [`clamp_alpha`].
/// At the peak (`NoH = 1`) this evaluates to `1 / (pi * alpha^2)`, which grows
/// without bound as `alpha -> 0` (a sharp mirror highlight); at `NoH = 0` it is
/// `alpha^2 / pi`. The expression is a pure rational function — no
/// transcendental is used.
#[must_use]
pub fn ggx_distribution(n_dot_h: f32, alpha: f32) -> f32 {
    let a = clamp_alpha(alpha);
    let a2 = a * a;
    let noh = clamp01(n_dot_h);
    // Trowbridge-Reitz denominator kernel, written as `NoH^2 * alpha^2 +
    // (1 - NoH^2)` rather than the algebraically equal `NoH^2 * (alpha^2 - 1)
    // + 1`: the latter catastrophically cancels in `f32` at the `NoH = 1` peak
    // (where `alpha^2 + 1 - 1` rounds the tiny `alpha^2` away), collapsing the
    // denominator to zero. This grouping keeps both summands non-negative.
    let noh2 = noh * noh;
    let kernel = noh2 * a2 + (1.0 - noh2);
    // `clamp_alpha` guarantees `kernel >= alpha^2 > 0`, so the denominator is
    // always strictly positive: no divide-by-zero guard is needed, and the
    // near-mirror peak is intentionally large rather than clamped to zero.
    let denom = PI * kernel * kernel;
    a2 / denom
}

/// `Smith` `k` remap for **direct** (analytic / punctual) lighting:
/// `k = (perceptual_roughness + 1)^2 / 8`.
///
/// This takes the artist-facing *perceptual* roughness (not `alpha`), matching
/// the disney/`UE4` convention for direct lights, and feeds
/// [`schlick_ggx_g1`] / [`smith_g_separable`].
#[must_use]
pub fn smith_k_direct(perceptual_roughness: f32) -> f32 {
    let r = clamp01(perceptual_roughness) + 1.0;
    // (r)^2 / 8 — magic 8 is the UE4/disney direct-lighting divisor.
    (r * r) / 8.0
}

/// `Smith` `k` remap for **image-based lighting** (`IBL`): `k = alpha^2 / 2`.
///
/// This takes the linear roughness `alpha` and is the environment-map variant
/// of the `k` used by [`schlick_ggx_g1`].
#[must_use]
pub fn smith_k_ibl(alpha: f32) -> f32 {
    let a = clamp_alpha(alpha);
    // alpha^2 / 2 — the IBL divisor (2 vs. the direct-lighting 8).
    (a * a) / 2.0
}

/// One-sided `Schlick`-`GGX` masking term `G1(NoX, k) = NoX / (NoX * (1 - k) + k)`.
///
/// `n_dot_x` is clamped to `0..=1`. The result lies in `0..=1` and increases
/// monotonically with `n_dot_x`. `k` is a precomputed roughness remap from
/// [`smith_k_direct`] or [`smith_k_ibl`].
#[must_use]
pub fn schlick_ggx_g1(n_dot_x: f32, k: f32) -> f32 {
    let x = clamp01(n_dot_x);
    let denom = x * (1.0 - k) + k;
    if denom < MIN_DENOM {
        return 0.0;
    }
    x / denom
}

/// Separable `Smith` masking-shadowing `G = G1(NoL, k) * G1(NoV, k)`.
///
/// This is the classic (uncorrelated) `Smith` form: the view and light
/// occlusion probabilities are assumed independent and multiplied. The result
/// lies in `0..=1`.
#[must_use]
pub fn smith_g_separable(n_dot_l: f32, n_dot_v: f32, k: f32) -> f32 {
    schlick_ggx_g1(n_dot_l, k) * schlick_ggx_g1(n_dot_v, k)
}

/// Height-correlated `Smith` `G2` of Heitz for the `GGX` masking-shadowing
/// term.
///
/// ```text
/// Lambda_v = NoL * sqrt(NoV^2 * (1 - alpha^2) + alpha^2)
/// Lambda_l = NoV * sqrt(NoL^2 * (1 - alpha^2) + alpha^2)
/// G2       = 2 * NoL * NoV / (Lambda_v + Lambda_l)
/// ```
///
/// Correlating the masking and shadowing heights removes the energy loss of the
/// separable form and uses only `sqrt`. The result lies in `0..=1`.
#[must_use]
pub fn smith_g2_height_correlated(n_dot_l: f32, n_dot_v: f32, alpha: f32) -> f32 {
    let a = clamp_alpha(alpha);
    let a2 = a * a;
    let nl = clamp01(n_dot_l);
    let nv = clamp01(n_dot_v);
    let lambda_v = nl * (nv * nv * (1.0 - a2) + a2).sqrt();
    let lambda_l = nv * (nl * nl * (1.0 - a2) + a2).sqrt();
    let denom = lambda_v + lambda_l;
    if denom < MIN_DENOM {
        return 0.0;
    }
    (2.0 * nl * nv) / denom
}

/// Height-correlated `Smith` visibility `V = G2 / (4 * NoL * NoV)`.
///
/// Folding the Cook-Torrance `1 / (4 * NoL * NoV)` denominator directly into the
/// `G2` quotient cancels the fragile `NoL`/`NoV` factors analytically:
///
/// ```text
/// V = 0.5 / (Lambda_v + Lambda_l)
/// ```
///
/// with `Lambda_v` / `Lambda_l` as in [`smith_g2_height_correlated`]. This is
/// the numerically stable term used by [`specular_dvf_rgb`]. It is non-negative.
#[must_use]
pub fn visibility_smith_ggx_correlated(n_dot_l: f32, n_dot_v: f32, alpha: f32) -> f32 {
    let a = clamp_alpha(alpha);
    let a2 = a * a;
    let nl = clamp01(n_dot_l);
    let nv = clamp01(n_dot_v);
    let lambda_v = nl * (nv * nv * (1.0 - a2) + a2).sqrt();
    let lambda_l = nv * (nl * nl * (1.0 - a2) + a2).sqrt();
    // V = 0.5 / (Lambda_v + Lambda_l) => 1 / (2 * (Lambda_v + Lambda_l)).
    let denom = 2.0 * (lambda_v + lambda_l);
    if denom < MIN_DENOM {
        return 0.0;
    }
    1.0 / denom
}

/// Separable `Smith` visibility `V = G / (4 * NoL * NoV)` from the `k` remap.
///
/// The explicit-denominator companion of [`visibility_smith_ggx_correlated`]
/// for callers that assemble the lobe with the separable `G`
/// ([`smith_g_separable`]). The `4 * NoL * NoV` divisor is guarded against zero.
#[must_use]
pub fn visibility_smith_ggx_separable(n_dot_l: f32, n_dot_v: f32, k: f32) -> f32 {
    let nl = clamp01(n_dot_l);
    let nv = clamp01(n_dot_v);
    let denom = 4.0 * nl * nv;
    if denom < MIN_DENOM {
        return 0.0;
    }
    smith_g_separable(nl, nv, k) / denom
}

/// Three-channel `Schlick` `Fresnel` with a per-channel reflectance `F0`.
///
/// Evaluates [`super::fresnel_rim::fresnel_schlick`] independently on each `RGB`
/// channel of `f0`, treating [`Vec3`] as a linear-`RGB` triple. `cos_theta` is
/// the half-vector cosine `VoH` and is clamped to `0..=1` by the scalar
/// routine. At grazing (`cos_theta = 0`) every channel saturates to `1.0`.
#[must_use]
pub fn fresnel_schlick_f0_rgb(cos_theta: f32, f0: Vec3) -> Vec3 {
    Vec3::new(
        fresnel_schlick(cos_theta, f0.x),
        fresnel_schlick(cos_theta, f0.y),
        fresnel_schlick(cos_theta, f0.z),
    )
}

/// Assembles the textbook scalar lobe `D * G * F / (4 * NoL * NoV)`.
///
/// Callers pass precomputed `D`, `G`, and `F` (for example from
/// [`ggx_distribution`], [`smith_g_separable`], and
/// [`super::fresnel_rim::fresnel_schlick`]). The `4 * NoL * NoV` divisor is
/// guarded against zero.
#[must_use]
pub fn specular_dgf_scalar(d: f32, g: f32, f: f32, n_dot_l: f32, n_dot_v: f32) -> f32 {
    let denom = 4.0 * clamp01(n_dot_l) * clamp01(n_dot_v);
    if denom < MIN_DENOM {
        return 0.0;
    }
    (d * g * f) / denom
}

/// Assembles the numerically stable scalar lobe `D * V * F`.
///
/// `vis` is a visibility term that already folds in the `1 / (4 * NoL * NoV)`
/// denominator (for example [`visibility_smith_ggx_correlated`]).
#[must_use]
pub fn specular_dvf_scalar(d: f32, vis: f32, f: f32) -> f32 {
    d * vis * f
}

/// The clamped microfacet cosines for one light/view/normal configuration.
///
/// All four cosines are clamped to `0..=1`: back-facing contributions vanish,
/// matching the physical masking of the lower hemisphere. Because the half
/// vector `h = normalize(l + v)` is symmetric in `l` and `v`, `VoH == LoH`, so
/// a single `v_dot_h` captures the `Fresnel` cosine for both.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MicrofacetDirs {
    /// Cosine between the surface normal and the light direction, `NoL`.
    pub n_dot_l: f32,
    /// Cosine between the surface normal and the view direction, `NoV`.
    pub n_dot_v: f32,
    /// Cosine between the surface normal and the half vector, `NoH`.
    pub n_dot_h: f32,
    /// Cosine between the view (equivalently light) direction and the half
    /// vector, `VoH` (`== LoH`).
    pub v_dot_h: f32,
}

impl MicrofacetDirs {
    /// Builds the clamped cosines from hand-rolled unit directions.
    ///
    /// `n`, `v`, and `l` are the surface normal, view direction, and light
    /// direction. The half vector is `normalize(l + v)`; a degenerate sum
    /// (opposing `l` and `v`) normalizes to zero, yielding `NoH = VoH = 0`.
    #[must_use]
    pub fn from_vectors(n: Vec3, v: Vec3, l: Vec3) -> Self {
        let h = v.add(l).normalize_or_zero();
        Self {
            n_dot_l: clamp01(n.dot(l)),
            n_dot_v: clamp01(n.dot(v)),
            n_dot_h: clamp01(n.dot(h)),
            v_dot_h: clamp01(v.dot(h)),
        }
    }
}

/// Evaluates the full single-scattering specular lobe as a scalar, using the
/// height-correlated `D * V * F` form.
///
/// `alpha` is the linear roughness and `f0` the scalar reflectance at normal
/// incidence. This is the recommended default assembly: `Fresnel` is taken at
/// `VoH`, visibility is the Heitz height-correlated term, and the result is
/// non-negative and reciprocal in `l`/`v`.
#[must_use]
pub fn specular_ggx_scalar(dirs: MicrofacetDirs, alpha: f32, f0: f32) -> f32 {
    let d = ggx_distribution(dirs.n_dot_h, alpha);
    let vis = visibility_smith_ggx_correlated(dirs.n_dot_l, dirs.n_dot_v, alpha);
    let f = fresnel_schlick(dirs.v_dot_h, f0);
    specular_dvf_scalar(d, vis, f)
}

/// Three-channel companion of [`specular_ggx_scalar`] with a per-channel `F0`.
///
/// Returns the linear-`RGB` specular contribution `D * V * F` where `F` is the
/// three-channel `Fresnel` ([`fresnel_schlick_f0_rgb`]). `D` and `V` are scalar
/// and shared across channels.
#[must_use]
pub fn specular_dvf_rgb(dirs: MicrofacetDirs, alpha: f32, f0: Vec3) -> Vec3 {
    let d = ggx_distribution(dirs.n_dot_h, alpha);
    let vis = visibility_smith_ggx_correlated(dirs.n_dot_l, dirs.n_dot_v, alpha);
    let scale = d * vis;
    fresnel_schlick_f0_rgb(dirs.v_dot_h, f0).scale(scale)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Comparison epsilon for the gold-standard assertions.
    const CMP_EPS: f32 = 1.0e-5;

    /// Absolute-tolerance float compare (never `==` on `f32`).
    fn approx_eq(a: f32, b: f32) -> bool {
        (a - b).abs() <= CMP_EPS
    }

    /// Relative-or-absolute compare for large-magnitude `NDF` peaks.
    fn approx_rel(a: f32, b: f32) -> bool {
        let scale = 1.0_f32.max(a.abs()).max(b.abs());
        (a - b).abs() <= CMP_EPS * scale
    }

    #[test]
    fn ggx_peak_matches_closed_form() {
        // D(NoH = 1, alpha) = 1 / (pi * alpha^2).
        for &alpha in &[0.1_f32, 0.25, 0.5, 1.0] {
            let expected = 1.0 / (PI * alpha * alpha);
            assert!(approx_rel(ggx_distribution(1.0, alpha), expected));
        }
    }

    #[test]
    fn ggx_at_zero_cosine_matches_closed_form() {
        // D(NoH = 0, alpha) = alpha^2 / pi.
        for &alpha in &[0.2_f32, 0.5, 0.9] {
            let expected = (alpha * alpha) / PI;
            assert!(approx_rel(ggx_distribution(0.0, alpha), expected));
        }
    }

    #[test]
    fn ggx_peak_sharpens_as_alpha_shrinks() {
        // Smaller alpha => a much taller, sharper highlight at the peak.
        let rough = ggx_distribution(1.0, 0.6);
        let mid = ggx_distribution(1.0, 0.2);
        let sharp = ggx_distribution(1.0, 0.05);
        assert!(sharp > mid);
        assert!(mid > rough);
        // The near-mirror peak is genuinely huge, not a token increase.
        assert!(sharp > 100.0);
    }

    #[test]
    fn ggx_lobe_widens_with_alpha_off_peak() {
        // A broad lobe pushes more energy to off-peak half angles: at NoH = 0.7
        // the rougher surface has the larger D even though its peak is lower.
        let narrow = ggx_distribution(0.7, 0.1);
        let broad = ggx_distribution(0.7, 0.6);
        assert!(broad > narrow);
    }

    #[test]
    fn ggx_is_non_negative_and_falls_off_peak() {
        let alpha = 0.3;
        let peak = ggx_distribution(1.0, alpha);
        let off = ggx_distribution(0.5, alpha);
        let far = ggx_distribution(0.0, alpha);
        assert!(peak >= 0.0 && off >= 0.0 && far >= 0.0);
        assert!(peak > off);
        assert!(off > far);
    }

    #[test]
    fn ggx_mirror_limit_stays_finite() {
        // alpha -> 0 must not divide by zero or produce a NaN/inf.
        let d = ggx_distribution(1.0, 0.0);
        assert!(d.is_finite());
        assert!(d > 0.0);
    }

    #[test]
    fn schlick_g1_is_identity_when_k_is_zero() {
        for &x in &[0.1_f32, 0.5, 1.0] {
            assert!(approx_eq(schlick_ggx_g1(x, 0.0), 1.0));
        }
    }

    #[test]
    fn schlick_g1_in_unit_range_and_monotone() {
        let k = smith_k_direct(0.5);
        let lo = schlick_ggx_g1(0.1, k);
        let mid = schlick_ggx_g1(0.5, k);
        let hi = schlick_ggx_g1(0.9, k);
        for v in [lo, mid, hi] {
            assert!((0.0..=1.0).contains(&v));
        }
        assert!(lo < mid);
        assert!(mid < hi);
    }

    #[test]
    fn separable_g_in_unit_range() {
        let k = smith_k_ibl(0.4);
        for &nl in &[0.1_f32, 0.5, 1.0] {
            for &nv in &[0.1_f32, 0.5, 1.0] {
                let g = smith_g_separable(nl, nv, k);
                assert!((0.0..=1.0).contains(&g));
            }
        }
    }

    #[test]
    fn smith_k_variants_differ_as_documented() {
        // Direct k uses (r + 1)^2 / 8 on perceptual roughness; IBL uses a^2 / 2.
        assert!(approx_eq(smith_k_direct(0.0), 1.0 / 8.0));
        assert!(approx_eq(smith_k_direct(1.0), 4.0 / 8.0));
        assert!(approx_eq(smith_k_ibl(0.5), 0.25 / 2.0));
        assert!(approx_eq(smith_k_ibl(1.0), 1.0 / 2.0));
    }

    #[test]
    fn g2_height_correlated_in_unit_range() {
        for &alpha in &[0.1_f32, 0.5, 1.0] {
            for &nl in &[0.2_f32, 0.6, 1.0] {
                for &nv in &[0.2_f32, 0.6, 1.0] {
                    let g2 = smith_g2_height_correlated(nl, nv, alpha);
                    assert!((0.0..=1.0 + CMP_EPS).contains(&g2));
                }
            }
        }
    }

    #[test]
    fn g2_matches_rough_closed_form() {
        // alpha = 1 => Lambda_v = NoL, Lambda_l = NoV => G2 = 2 NoL NoV/(NoL+NoV).
        let nl = 0.6;
        let nv = 0.3;
        let expected = (2.0 * nl * nv) / (nl + nv);
        assert!(approx_eq(smith_g2_height_correlated(nl, nv, 1.0), expected));
    }

    #[test]
    fn visibility_matches_g2_over_denominator() {
        // V must equal G2 / (4 NoL NoV) by construction.
        let (nl, nv, alpha) = (0.7_f32, 0.4_f32, 0.3_f32);
        let v = visibility_smith_ggx_correlated(nl, nv, alpha);
        let g2 = smith_g2_height_correlated(nl, nv, alpha);
        assert!(approx_eq(v, g2 / (4.0 * nl * nv)));
    }

    #[test]
    fn visibility_is_non_negative_and_finite_at_grazing() {
        // Near-grazing cosines must not blow up to a NaN/inf.
        let v = visibility_smith_ggx_correlated(1.0e-4, 1.0e-4, 0.3);
        assert!(v.is_finite());
        assert!(v >= 0.0);
    }

    #[test]
    fn separable_visibility_folds_the_denominator() {
        let (nl, nv) = (0.5_f32, 0.8_f32);
        let k = smith_k_direct(0.3);
        let v = visibility_smith_ggx_separable(nl, nv, k);
        let expected = smith_g_separable(nl, nv, k) / (4.0 * nl * nv);
        assert!(approx_eq(v, expected));
    }

    #[test]
    fn fresnel_rgb_hits_its_endpoints() {
        let f0 = Vec3::new(0.04, 0.08, 0.16);
        // Head-on: returns F0 per channel.
        let head = fresnel_schlick_f0_rgb(1.0, f0);
        assert!(approx_eq(head.x, f0.x));
        assert!(approx_eq(head.y, f0.y));
        assert!(approx_eq(head.z, f0.z));
        // Grazing: every channel saturates to 1.
        let graze = fresnel_schlick_f0_rgb(0.0, f0);
        assert!(approx_eq(graze.x, 1.0));
        assert!(approx_eq(graze.y, 1.0));
        assert!(approx_eq(graze.z, 1.0));
    }

    #[test]
    fn fresnel_rgb_is_monotone_toward_grazing() {
        let f0 = Vec3::splat(0.04);
        let head = fresnel_schlick_f0_rgb(0.9, f0).x;
        let mid = fresnel_schlick_f0_rgb(0.5, f0).x;
        let graze = fresnel_schlick_f0_rgb(0.1, f0).x;
        assert!(head < mid);
        assert!(mid < graze);
    }

    #[test]
    fn dgf_and_dvf_scalar_agree_with_shared_pieces() {
        // The explicit D*G/(4 NoL NoV)*F must equal D*V*F when V = G/(4 NoL NoV).
        let (nl, nv, noh, voh) = (0.7_f32, 0.5_f32, 0.95_f32, 0.9_f32);
        let alpha = 0.25_f32;
        let f0 = 0.04_f32;
        let d = ggx_distribution(noh, alpha);
        let g2 = smith_g2_height_correlated(nl, nv, alpha);
        let vis = visibility_smith_ggx_correlated(nl, nv, alpha);
        let f = fresnel_schlick(voh, f0);
        let dgf = specular_dgf_scalar(d, g2, f, nl, nv);
        let dvf = specular_dvf_scalar(d, vis, f);
        assert!(approx_eq(dgf, dvf));
    }

    #[test]
    fn microfacet_dirs_from_vectors_are_consistent() {
        let n = Vec3::new(0.0, 0.0, 1.0);
        let v = Vec3::new(0.0, 0.4, 1.0).normalize_or_zero();
        let l = Vec3::new(0.3, 0.0, 1.0).normalize_or_zero();
        let dirs = MicrofacetDirs::from_vectors(n, v, l);
        // All cosines land in the unit range.
        for c in [dirs.n_dot_l, dirs.n_dot_v, dirs.n_dot_h, dirs.v_dot_h] {
            assert!((0.0..=1.0).contains(&c));
        }
        // VoH == LoH because h is the normalized half vector.
        let h = v.add(l).normalize_or_zero();
        assert!(approx_eq(dirs.v_dot_h, l.dot(h).clamp(0.0, 1.0)));
    }

    #[test]
    fn specular_lobe_is_reciprocal() {
        // f(l, v) == f(v, l): swapping light and view leaves the BRDF unchanged.
        let n = Vec3::new(0.0, 0.0, 1.0);
        let v = Vec3::new(0.2, 0.1, 1.0).normalize_or_zero();
        let l = Vec3::new(-0.3, 0.25, 1.0).normalize_or_zero();
        let alpha = 0.3_f32;
        let f0 = 0.08_f32;
        let forward = specular_ggx_scalar(MicrofacetDirs::from_vectors(n, v, l), alpha, f0);
        let swapped = specular_ggx_scalar(MicrofacetDirs::from_vectors(n, l, v), alpha, f0);
        assert!(approx_eq(forward, swapped));
    }

    #[test]
    fn specular_lobe_is_non_negative() {
        let n = Vec3::new(0.0, 0.0, 1.0);
        let v = Vec3::new(0.1, 0.2, 1.0).normalize_or_zero();
        for &alpha in &[0.05_f32, 0.3, 0.7, 1.0] {
            for lx in -3..=3 {
                let l = Vec3::new(lx as f32 * 0.3, 0.1, 1.0).normalize_or_zero();
                let s = specular_ggx_scalar(MicrofacetDirs::from_vectors(n, v, l), alpha, 0.04);
                assert!(s >= 0.0);
                assert!(s.is_finite());
            }
        }
    }

    #[test]
    fn specular_rgb_is_scalar_lobe_times_fresnel_rgb() {
        // The RGB assembly must equal D*V applied to the per-channel Fresnel.
        let n = Vec3::new(0.0, 0.0, 1.0);
        let v = Vec3::new(0.15, 0.1, 1.0).normalize_or_zero();
        let l = Vec3::new(-0.2, 0.2, 1.0).normalize_or_zero();
        let alpha = 0.35_f32;
        let f0 = Vec3::new(0.04, 0.1, 0.2);
        let dirs = MicrofacetDirs::from_vectors(n, v, l);
        let rgb = specular_dvf_rgb(dirs, alpha, f0);
        let d = ggx_distribution(dirs.n_dot_h, alpha);
        let vis = visibility_smith_ggx_correlated(dirs.n_dot_l, dirs.n_dot_v, alpha);
        let scale = d * vis;
        let f = fresnel_schlick_f0_rgb(dirs.v_dot_h, f0);
        assert!(approx_eq(rgb.x, f.x * scale));
        assert!(approx_eq(rgb.y, f.y * scale));
        assert!(approx_eq(rgb.z, f.z * scale));
    }

    #[test]
    fn roughness_remap_round_trips() {
        for &p in &[0.0_f32, 0.1, 0.3, 0.5, 0.8, 1.0] {
            let alpha = alpha_from_perceptual_roughness(p);
            let back = perceptual_roughness_from_alpha(alpha);
            assert!(approx_eq(back, p));
            // alpha is the square of perceptual roughness.
            assert!(approx_eq(alpha, p * p));
        }
    }

    #[test]
    fn specular_lobe_broadens_with_roughness() {
        // A smooth surface concentrates the highlight near the mirror direction,
        // so at the exact mirror configuration the sharp lobe out-peaks the
        // broad one; the broad lobe instead spreads energy to off-peak angles.
        let n = Vec3::new(0.0, 0.0, 1.0);
        let v = Vec3::new(0.0, 0.0, 1.0);
        let l = Vec3::new(0.0, 0.0, 1.0); // perfect reflection: NoH = 1.
        let sharp = specular_ggx_scalar(MicrofacetDirs::from_vectors(n, v, l), 0.05, 0.04);
        let broad = specular_ggx_scalar(MicrofacetDirs::from_vectors(n, v, l), 0.6, 0.04);
        assert!(sharp > broad);
        // Off the mirror direction, the broad lobe carries more.
        let off_l = Vec3::new(0.6, 0.0, 0.8).normalize_or_zero();
        let sharp_off =
            specular_ggx_scalar(MicrofacetDirs::from_vectors(n, v, off_l), 0.05, 0.04);
        let broad_off =
            specular_ggx_scalar(MicrofacetDirs::from_vectors(n, v, off_l), 0.6, 0.04);
        assert!(broad_off > sharp_off);
    }
}
