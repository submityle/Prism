//! Anisotropic `GGX` base layer plus a dielectric clearcoat lobe: the `CPU`
//! gold-standard reflectance terms for brushed-metal / lacquered particle
//! shading (design §15 "各向异性/清漆" renderer matrix, §17 `PBR` closure).
//!
//! Design §15 lists the `Mesh` renderer as the one that carries the *full*
//! shading closure "含各向异性/清漆" (anisotropy + clearcoat), and §17 pins the
//! `PBR` response to a `GGX` microfacet model shared bit-for-bit between the
//! `CPU` reference and the eventual `GPU` draw kernel. This module owns that
//! reference for two coupled pieces a hair-brush, carbon-fibre, or car-paint
//! particle needs:
//!
//! 1. An **anisotropic `GGX`** base lobe whose highlight stretches along the
//!    surface tangent. The normal-distribution function (`NDF`) uses the
//!    `Burley` tangent/bitangent form and the shadowing/masking term uses the
//!    height-correlated anisotropic `Smith` visibility, so a scratched or combed
//!    surface shows the elongated streak real brushed metal does.
//! 2. A thin **clearcoat** on top: a second, isotropic `GGX` lobe for a smooth
//!    dielectric film at a fixed index of refraction (`IOR`) of 1.5, whose
//!    `Schlick` `Fresnel` both drives the coat highlight and attenuates the base
//!    layer underneath by its transmitted energy `(1 - Fc)`.
//!
//! Both anisotropic alphas are derived from a single perceptual `roughness` and
//! an `anisotropy` control in `-1..=1` through the classic `Disney`/`Frostbite`
//! mapping (`aspect = sqrt(1 - 0.9 * |anisotropy|)`, `alpha_t = alpha / aspect`,
//! `alpha_b = alpha * aspect`), so when `anisotropy = 0` the two alphas collapse
//! back to the isotropic `alpha = roughness^2` and the anisotropic `NDF`
//! reproduces the ordinary isotropic `GGX` exactly (see the module tests).
//!
//! The clearcoat `Fresnel` reuses [`super::fresnel_rim::fresnel_schlick`] rather
//! than re-deriving the `Schlick` approximation here (DRY across the particle
//! shading modules).
//!
//! **Determinism**: every term is a rational function of the geometry dot
//! products, with the single exception of the `Smith` `Lambda` and the aspect
//! ratio, which need one `sqrt` each — the only transcendental the workspace
//! determinism lint permits. No `sin`/`cos`/`tan`/`exp`/`ln`/`powf`/`powi` is
//! used: integer powers are fixed multiplications (see [`square`]). Every
//! division is guarded against a zero denominator, so a mirror-smooth
//! (`roughness = 0`) or degenerate (zero-length) input yields a finite value,
//! never a `NaN`. All vectors are the shared hand-rolled [`super::Vec3`] with
//! named methods; the tangent, bitangent, normal, and half vector are all
//! `Vec3`, and scalars are `f32`. `GPU` packing follows the shared `std430`
//! `vec4` alignment from [`super::gpu_layout`].

use super::fresnel_rim::fresnel_schlick;
use super::gpu_layout::{storage_bytes, VEC4_STRIDE};
use super::Vec3;

/// Reciprocal of pi, the `GGX` `NDF` normalization constant `1 / π`.
///
/// Taken from `core` rather than written as a literal so it carries full `f32`
/// precision; it is a compile-time constant, not a transcendental call.
const INV_PI: f32 = core::f32::consts::FRAC_1_PI;

/// Index of refraction of the clearcoat film: a polyurethane / lacquer-like
/// dielectric at `IOR = 1.5`, the industry-standard clearcoat value.
pub const CLEARCOAT_IOR: f32 = 1.5;

/// Reflectance at normal incidence (`F0`) of the clearcoat film.
///
/// For a dielectric over air the `Schlick` `F0` is `((n - 1) / (n + 1))^2`; at
/// `n = 1.5` this is `(0.5 / 2.5)^2 = 0.04`, the canonical dielectric `F0`.
pub const CLEARCOAT_F0: f32 = 0.04;

/// Slope of the anisotropy-to-aspect mapping: `aspect = sqrt(1 - 0.9 * |a|)`.
///
/// `0.9` (rather than `1.0`) is the classic `Disney` choice that keeps the
/// aspect ratio strictly positive — `1 - 0.9 = 0.1 > 0` — even at full
/// anisotropy, so neither alpha ever collapses to zero.
const ASPECT_ANISOTROPY_SCALE: f32 = 0.9;

/// Lower clamp on a `GGX` `alpha` (linear roughness).
///
/// A perfectly smooth surface (`alpha = 0`) would make the `NDF` denominator
/// vanish at the specular peak; clamping to `1e-4` keeps a very sharp but finite
/// highlight instead of an infinity, matching the "clamp near-mirror roughness"
/// rule real-time engines use.
const MIN_ALPHA: f32 = 1e-4;

/// Generic lower bound on a denominator that could otherwise reach zero (the
/// `Smith` sum, the half-vector length). Guards against division by zero
/// producing a `NaN` without perceptibly altering well-conditioned inputs.
const MIN_DENOM: f32 = 1e-8;

/// Byte stride of one [`AnisotropicClearcoatParams`] record in a `std430`
/// storage buffer.
///
/// The four scalar controls pack into a single `vec4` slot
/// `vec4(roughness, anisotropy, clearcoat_roughness, clearcoat_strength)`.
pub const ANISOTROPIC_CLEARCOAT_PARAMS_STRIDE: usize = VEC4_STRIDE;

/// Squares a scalar via a single multiplication.
///
/// Replaces `f32::powi(x, 2)` so the determinism lint's ban on `powi`/`powf`
/// holds while the intent (an integer power of two) stays explicit at the call
/// site.
#[must_use]
#[inline]
fn square(x: f32) -> f32 {
    x * x
}

/// Clamps a scalar into `0..=1` without branching on floating-point equality.
#[must_use]
#[inline]
fn clamp01(x: f32) -> f32 {
    x.clamp(0.0, 1.0)
}

/// The anisotropic `GGX` linear roughnesses along the tangent and bitangent.
///
/// Derived from a perceptual `roughness` and an `anisotropy` control; see
/// [`anisotropic_alphas`]. When `anisotropy = 0` the two are equal and the
/// surface is isotropic.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AnisoAlphas {
    /// Linear roughness along the surface tangent (`T`): the highlight is
    /// stretched *across* the direction of smaller alpha.
    pub alpha_t: f32,
    /// Linear roughness along the surface bitangent (`B`).
    pub alpha_b: f32,
}

/// The aspect ratio `sqrt(1 - 0.9 * |anisotropy|)` of the anisotropic lobe.
///
/// `anisotropy` is clamped to `-1..=1` and only its magnitude matters (the sign
/// chooses which of `T`/`B` is the "sharp" axis, which the caller encodes by how
/// it orients the tangent frame). The result lies in `(0, 1]`: it is `1` for an
/// isotropic surface (`anisotropy = 0`) and bottoms out at
/// `sqrt(0.1) ≈ 0.3162` at full anisotropy, never reaching zero.
///
/// Uses one `sqrt` (lint-permitted); everything else is rational.
#[must_use]
pub fn aspect_ratio(anisotropy: f32) -> f32 {
    let a = anisotropy.clamp(-1.0, 1.0).abs();
    // 1 - 0.9 * |a| ∈ [0.1, 1], always strictly positive, so the sqrt is real.
    (1.0 - ASPECT_ANISOTROPY_SCALE * a).sqrt()
}

/// Derives the tangent/bitangent `GGX` alphas from perceptual `roughness` and
/// `anisotropy`.
///
/// The isotropic linear roughness is `alpha = roughness^2` (the standard
/// perceptual-to-linear squaring), split by the [`aspect_ratio`] into
/// `alpha_t = alpha / aspect` (wider across the tangent) and
/// `alpha_b = alpha * aspect`. Both are floored at [`MIN_ALPHA`] so a
/// mirror-smooth input keeps the `GGX` denominator finite.
///
/// With `anisotropy = 0` the aspect is `1`, so `alpha_t == alpha_b == alpha`
/// and the surface reduces to isotropic `GGX`.
#[must_use]
pub fn anisotropic_alphas(roughness: f32, anisotropy: f32) -> AnisoAlphas {
    let alpha = square(clamp01(roughness));
    let aspect = aspect_ratio(anisotropy);
    // aspect ∈ (0, 1], so the divisions below never hit zero.
    let alpha_t = (alpha / aspect).max(MIN_ALPHA);
    let alpha_b = (alpha * aspect).max(MIN_ALPHA);
    AnisoAlphas { alpha_t, alpha_b }
}

/// Anisotropic `GGX` normal-distribution function in the `Burley` form.
///
/// Evaluates
/// `D = 1 / (π · α_t · α_b · ((ToH/α_t)² + (BoH/α_b)² + NoH²)²)`,
/// where `ToH`, `BoH`, `NoH` are the cosines of the half vector `H` against the
/// tangent, bitangent, and normal of an orthonormal shading frame. The result
/// is non-negative and purely rational apart from the shared `1/π` constant.
///
/// When `alpha_t == alpha_b == α` and `(ToH, BoH, NoH)` come from a unit `H` in
/// an orthonormal frame (so `ToH² + BoH² + NoH² = 1`), this collapses exactly to
/// the isotropic `GGX` `α² / (π · (NoH²(α²−1)+1)²)` — the module tests assert
/// this against a self-contained isotropic reference.
#[must_use]
pub fn ggx_aniso_ndf(n_dot_h: f32, t_dot_h: f32, b_dot_h: f32, alpha_t: f32, alpha_b: f32) -> f32 {
    let at = alpha_t.max(MIN_ALPHA);
    let ab = alpha_b.max(MIN_ALPHA);
    // ((ToH/α_t)² + (BoH/α_b)² + NoH²): the squared, axis-scaled slope term.
    let inner = square(t_dot_h / at) + square(b_dot_h / ab) + square(n_dot_h);
    let denom = at * ab * square(inner);
    if denom < MIN_DENOM {
        // Mirror-sharp peak: return a large-but-finite spike instead of 1/0.
        return INV_PI / MIN_DENOM;
    }
    INV_PI / denom
}

/// One side of the height-correlated anisotropic `Smith` masking term.
///
/// This is the `Filament` factorization: for a direction `ω` with cosines
/// `(ToW, BoW, NoW)` against the tangent frame, the per-direction weight is
/// `NoOther · sqrt((α_t·ToW)² + (α_b·BoW)² + NoW²)`, where `NoOther` is the
/// cosine of the *opposite* direction (the light's cosine weights the view term
/// and vice versa). Uses one `sqrt`; otherwise rational.
#[must_use]
fn smith_lambda_term(
    alpha_t: f32,
    alpha_b: f32,
    t_dot_w: f32,
    b_dot_w: f32,
    n_dot_w: f32,
    n_dot_other: f32,
) -> f32 {
    let len = (square(alpha_t * t_dot_w) + square(alpha_b * b_dot_w) + square(n_dot_w)).sqrt();
    n_dot_other * len
}

/// Height-correlated anisotropic `Smith` visibility `V = G / (4·NoV·NoL)`.
///
/// Returns the *visibility* (the geometry term already divided by the `4·NoV·NoL`
/// `BRDF` denominator), following the `Filament` height-correlated anisotropic
/// `Smith` formulation:
/// `V = 0.5 / (λ_v + λ_l)` with
/// `λ_v = NoL · sqrt((α_t·ToV)² + (α_b·BoV)² + NoV²)` and
/// `λ_l = NoV · sqrt((α_t·ToL)² + (α_b·BoL)² + NoL²)`.
///
/// `0.5` is the `Smith` `0.5 / (λ_v + λ_l)` constant that folds the `BRDF`'s
/// `1/(4·NoV·NoL)` into the correlated-masking integral. The sum is floored at
/// [`MIN_DENOM`] so a grazing (`NoV` or `NoL` → 0) configuration stays finite.
#[must_use]
#[expect(
    clippy::too_many_arguments,
    reason = "The anisotropic Smith term is defined over the full tangent-frame projection of both the view and light directions; the eight cosines are the irreducible inputs and bundling them into a struct would only move the noise to the call site."
)]
pub fn ggx_aniso_visibility(
    alpha_t: f32,
    alpha_b: f32,
    t_dot_v: f32,
    b_dot_v: f32,
    n_dot_v: f32,
    t_dot_l: f32,
    b_dot_l: f32,
    n_dot_l: f32,
) -> f32 {
    let lambda_v = smith_lambda_term(alpha_t, alpha_b, t_dot_v, b_dot_v, n_dot_v, n_dot_l);
    let lambda_l = smith_lambda_term(alpha_t, alpha_b, t_dot_l, b_dot_l, n_dot_l, n_dot_v);
    let denom = (lambda_v + lambda_l).max(MIN_DENOM);
    // 0.5 folds the BRDF's 1/(4·NoV·NoL) into the correlated-masking integral.
    0.5 / denom
}

/// Isotropic `GGX` normal-distribution function for the clearcoat lobe.
///
/// `D = α² / (π · (NoH²·(α²−1) + 1)²)`, the standard `Trowbridge-Reitz` form.
/// `alpha` is the clearcoat's own linear roughness (`clearcoat_roughness²`),
/// independent of the base layer. Non-negative and rational apart from `1/π`;
/// the denominator is floored at [`MIN_DENOM`].
#[must_use]
pub fn clearcoat_ggx_ndf(n_dot_h: f32, alpha: f32) -> f32 {
    let a2 = square(alpha.max(MIN_ALPHA));
    // NoH²·(α²−1) + 1: the Trowbridge-Reitz denominator kernel.
    let kernel = square(clamp01(n_dot_h)) * (a2 - 1.0) + 1.0;
    let denom = square(kernel);
    if denom < MIN_DENOM {
        return a2 * INV_PI / MIN_DENOM;
    }
    a2 * INV_PI / denom
}

/// Height-correlated isotropic `Smith` visibility for the clearcoat lobe.
///
/// `V = 0.5 / (NoL·√(NoV²(1−α²)+α²) + NoV·√(NoL²(1−α²)+α²))`, the isotropic
/// specialization of [`ggx_aniso_visibility`]. Uses two `sqrt`s; otherwise
/// rational, with the sum floored at [`MIN_DENOM`].
#[must_use]
pub fn clearcoat_visibility(n_dot_v: f32, n_dot_l: f32, alpha: f32) -> f32 {
    let a2 = square(alpha.max(MIN_ALPHA));
    let nv = clamp01(n_dot_v);
    let nl = clamp01(n_dot_l);
    let ggx_v = nl * (square(nv) * (1.0 - a2) + a2).sqrt();
    let ggx_l = nv * (square(nl) * (1.0 - a2) + a2).sqrt();
    let denom = (ggx_v + ggx_l).max(MIN_DENOM);
    0.5 / denom
}

/// The clearcoat `Schlick` `Fresnel` `Fc` at a given cosine.
///
/// Reuses [`super::fresnel_rim::fresnel_schlick`] with the fixed dielectric
/// [`CLEARCOAT_F0`] of `0.04`. `cos_theta` is the cosine between the half vector
/// and the view (or the normal and the view, depending on the caller's `Fresnel`
/// convention). Head-on (`cos_theta = 1`) returns `F0`; grazing
/// (`cos_theta = 0`) saturates to `1`.
#[must_use]
pub fn clearcoat_fresnel(cos_theta: f32) -> f32 {
    fresnel_schlick(cos_theta, CLEARCOAT_F0)
}

/// Energy left for the base layer after the clearcoat reflects `fc`.
///
/// A thin dielectric coat transmits `(1 - Fc)` of the incident energy to the
/// layer beneath it; light also exits back through the coat, so a physically
/// motivated two-pass attenuation would be `(1 - Fc)²`. This returns the single
/// transmission `(1 - Fc)` as the design specifies ("清漆对基底层的能量衰减因子
/// `(1 − Fc)`"); callers wanting the round trip can square it. `fc` is clamped
/// to `0..=1`, so the result is always in `0..=1`.
#[must_use]
pub fn clearcoat_attenuation(fc: f32) -> f32 {
    1.0 - clamp01(fc)
}

/// Artist-facing controls for the anisotropic base plus clearcoat response.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AnisotropicClearcoatParams {
    /// Perceptual roughness of the base layer, in `0..=1` (`alpha = roughness²`).
    pub roughness: f32,
    /// Anisotropy in `-1..=1`; `0` is isotropic, magnitude stretches the lobe.
    pub anisotropy: f32,
    /// Perceptual roughness of the clearcoat lobe, in `0..=1` (usually small).
    pub clearcoat_roughness: f32,
    /// Clearcoat presence in `0..=1` (`0` disables the coat entirely).
    pub clearcoat_strength: f32,
}

/// Evaluated reflectance terms for one shading fragment.
///
/// These are the *geometry-dependent* microfacet terms (distribution,
/// visibility, `Fresnel`, attenuation); the caller multiplies them by the base
/// albedo / `F0`, the light radiance, and `NoL` to form the final `BRDF`
/// contribution. Keeping them separate lets the same sample drive both the
/// `CPU` reference and a `GPU` kernel without re-deriving the geometry.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AnisotropicClearcoatSample {
    /// Anisotropic `GGX` `NDF` of the base layer.
    pub base_ndf: f32,
    /// Height-correlated anisotropic `Smith` visibility of the base layer.
    pub base_visibility: f32,
    /// Fraction of energy the clearcoat passes to the base layer, `(1 - Fc)`.
    pub base_attenuation: f32,
    /// Isotropic `GGX` `NDF` of the clearcoat lobe.
    pub clearcoat_ndf: f32,
    /// Isotropic `Smith` visibility of the clearcoat lobe.
    pub clearcoat_visibility: f32,
    /// Clearcoat `Schlick` `Fresnel` `Fc` (also the coat's specular weight).
    pub clearcoat_fresnel: f32,
}

impl AnisotropicClearcoatParams {
    /// Builds parameters from all fields.
    #[must_use]
    pub const fn new(
        roughness: f32,
        anisotropy: f32,
        clearcoat_roughness: f32,
        clearcoat_strength: f32,
    ) -> Self {
        Self {
            roughness,
            anisotropy,
            clearcoat_roughness,
            clearcoat_strength,
        }
    }

    /// Evaluates the base + clearcoat microfacet terms for a shading frame.
    ///
    /// `tangent`, `bitangent`, and `normal` form the (assumed orthonormal)
    /// shading basis; `view` and `light` point from the surface toward the eye
    /// and the light respectively. All five are normalized in place (a
    /// degenerate zero-length input normalizes to [`Vec3::ZERO`], i.e. grazing),
    /// the half vector `H = normalize(view + light)` is formed, and the geometry
    /// cosines feed [`ggx_aniso_ndf`], [`ggx_aniso_visibility`],
    /// [`clearcoat_ggx_ndf`], [`clearcoat_visibility`], and [`clearcoat_fresnel`].
    ///
    /// The clearcoat `Fresnel` is evaluated at `LoH` (the light/half cosine), the
    /// physically correct angle for the microfacet `Fresnel`, and multiplied by
    /// [`AnisotropicClearcoatParams::clearcoat_strength`] so a strength of `0`
    /// fully disables the coat (`Fc = 0`, attenuation `= 1`).
    #[must_use]
    pub fn evaluate(
        &self,
        tangent: Vec3,
        bitangent: Vec3,
        normal: Vec3,
        view: Vec3,
        light: Vec3,
    ) -> AnisotropicClearcoatSample {
        let t = tangent.normalize_or_zero();
        let b = bitangent.normalize_or_zero();
        let n = normal.normalize_or_zero();
        let v = view.normalize_or_zero();
        let l = light.normalize_or_zero();
        let h = v.add(l).normalize_or_zero();

        let alphas = anisotropic_alphas(self.roughness, self.anisotropy);

        let n_dot_h = n.dot(h);
        let t_dot_h = t.dot(h);
        let b_dot_h = b.dot(h);
        let base_ndf = ggx_aniso_ndf(n_dot_h, t_dot_h, b_dot_h, alphas.alpha_t, alphas.alpha_b);

        let n_dot_v = clamp01(n.dot(v));
        let n_dot_l = clamp01(n.dot(l));
        let base_visibility = ggx_aniso_visibility(
            alphas.alpha_t,
            alphas.alpha_b,
            t.dot(v),
            b.dot(v),
            n_dot_v,
            t.dot(l),
            b.dot(l),
            n_dot_l,
        );

        let cc_alpha = square(clamp01(self.clearcoat_roughness)).max(MIN_ALPHA);
        let clearcoat_ndf = clearcoat_ggx_ndf(n_dot_h, cc_alpha);
        let clearcoat_vis = clearcoat_visibility(n_dot_v, n_dot_l, cc_alpha);

        let l_dot_h = clamp01(l.dot(h));
        let strength = clamp01(self.clearcoat_strength);
        let fc = clearcoat_fresnel(l_dot_h) * strength;

        AnisotropicClearcoatSample {
            base_ndf,
            base_visibility,
            base_attenuation: clearcoat_attenuation(fc),
            clearcoat_ndf,
            clearcoat_visibility: clearcoat_vis,
            clearcoat_fresnel: fc,
        }
    }

    /// Packs the four controls into their `std430` `vec4`-aligned bit layout.
    ///
    /// Layout: `[roughness, anisotropy, clearcoat_roughness, clearcoat_strength]`
    /// as raw `f32::to_bits` patterns, matching
    /// [`ANISOTROPIC_CLEARCOAT_PARAMS_STRIDE`] so the mixed buffer round-trips
    /// exactly without a lossy cast.
    #[must_use]
    pub fn to_std430_bits(&self) -> [u32; 4] {
        [
            self.roughness.to_bits(),
            self.anisotropy.to_bits(),
            self.clearcoat_roughness.to_bits(),
            self.clearcoat_strength.to_bits(),
        ]
    }

    /// Total `std430` byte size of `count` packed records (minimum one element,
    /// per the shared [`super::gpu_layout`] clamp-to-one rule).
    #[must_use]
    pub fn storage_size(count: usize) -> usize {
        storage_bytes(ANISOTROPIC_CLEARCOAT_PARAMS_STRIDE, count)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Absolute tolerance for approximate floating-point comparisons.
    const EPS: f32 = 1e-5;

    /// Self-contained isotropic `GGX` reference, independent of any sibling
    /// module, used to validate the anisotropic `NDF`'s isotropic collapse.
    fn iso_ggx_reference(n_dot_h: f32, alpha: f32) -> f32 {
        let a2 = alpha * alpha;
        let kernel = n_dot_h * n_dot_h * (a2 - 1.0) + 1.0;
        a2 * INV_PI / (kernel * kernel)
    }

    fn approx(a: f32, b: f32, eps: f32) -> bool {
        (a - b).abs() <= eps
    }

    #[test]
    fn aspect_is_unit_when_isotropic_and_in_open_unit_interval_otherwise() {
        assert!(approx(aspect_ratio(0.0), 1.0, EPS), "isotropic aspect is 1");
        for &a in &[-1.0_f32, -0.5, -0.1, 0.1, 0.5, 1.0] {
            let aspect = aspect_ratio(a);
            assert!(aspect > 0.0, "aspect must be strictly positive: {aspect}");
            assert!(aspect <= 1.0, "aspect must not exceed 1: {aspect}");
        }
        // Full anisotropy bottoms out at sqrt(1 - 0.9) = sqrt(0.1).
        assert!(
            approx(aspect_ratio(1.0), 0.1_f32.sqrt(), EPS),
            "full anisotropy aspect equals sqrt(0.1)"
        );
    }

    #[test]
    fn aspect_clamps_out_of_range_anisotropy() {
        assert!(approx(aspect_ratio(5.0), aspect_ratio(1.0), EPS));
        assert!(approx(aspect_ratio(-5.0), aspect_ratio(-1.0), EPS));
    }

    #[test]
    fn alphas_collapse_to_isotropic_when_anisotropy_zero() {
        let a = anisotropic_alphas(0.5, 0.0);
        let expected = 0.5_f32 * 0.5; // alpha = roughness²
        assert!(approx(a.alpha_t, expected, EPS));
        assert!(approx(a.alpha_b, expected, EPS));
    }

    #[test]
    fn alphas_split_in_opposite_directions_with_anisotropy() {
        let a = anisotropic_alphas(0.5, 0.8);
        // alpha_t = alpha / aspect >= alpha >= alpha * aspect = alpha_b.
        assert!(
            a.alpha_t > a.alpha_b,
            "tangent alpha widens, bitangent narrows"
        );
        // Product is preserved: (alpha/aspect)*(alpha*aspect) = alpha².
        let alpha = 0.5_f32 * 0.5;
        assert!(approx(a.alpha_t * a.alpha_b, alpha * alpha, EPS));
    }

    #[test]
    fn alphas_floored_for_mirror_smooth_input() {
        let a = anisotropic_alphas(0.0, 0.0);
        assert!(a.alpha_t >= MIN_ALPHA, "alpha_t floored");
        assert!(a.alpha_b >= MIN_ALPHA, "alpha_b floored");
    }

    #[test]
    fn aniso_ndf_matches_isotropic_reference_when_alphas_equal() {
        let alpha = 0.3_f32;
        // Sweep a unit half vector in the T-N plane of an orthonormal frame so
        // that ToH² + BoH² + NoH² = 1 holds exactly.
        for &n_dot_h in &[1.0_f32, 0.95, 0.8, 0.6, 0.4] {
            let t_dot_h = (1.0 - n_dot_h * n_dot_h).max(0.0).sqrt();
            let b_dot_h = 0.0;
            let got = ggx_aniso_ndf(n_dot_h, t_dot_h, b_dot_h, alpha, alpha);
            let want = iso_ggx_reference(n_dot_h, alpha);
            assert!(
                approx(got, want, 1e-3 * want.max(1.0)),
                "aniso NDF must equal isotropic GGX when alpha_t==alpha_b: got {got}, want {want}"
            );
        }
    }

    #[test]
    fn aniso_ndf_is_non_negative() {
        let a = anisotropic_alphas(0.4, 0.6);
        for &n_dot_h in &[0.0_f32, 0.2, 0.5, 0.9, 1.0] {
            let t_dot_h = (1.0 - n_dot_h * n_dot_h).max(0.0).sqrt();
            let d = ggx_aniso_ndf(n_dot_h, t_dot_h, 0.0, a.alpha_t, a.alpha_b);
            assert!(d >= 0.0, "NDF must be non-negative: {d}");
            assert!(d.is_finite(), "NDF must be finite: {d}");
        }
    }

    #[test]
    fn aniso_ndf_peaks_along_tangent_for_positive_anisotropy() {
        let a = anisotropic_alphas(0.4, 0.9);
        // A half vector tilted off-normal along the (wider) tangent axis spreads
        // the lobe more than the same tilt along the (narrower) bitangent axis,
        // so for an equal off-axis angle the bitangent tilt gives a higher D.
        let off = 0.3_f32;
        let n_dot_h = (1.0 - off * off).sqrt();
        let along_t = ggx_aniso_ndf(n_dot_h, off, 0.0, a.alpha_t, a.alpha_b);
        let along_b = ggx_aniso_ndf(n_dot_h, 0.0, off, a.alpha_t, a.alpha_b);
        assert!(
            along_t > along_b,
            "the wider tangent axis (alpha_t > alpha_b) keeps more density for an equal off-axis tilt, so the lobe stretches along the tangent: along_t {along_t}, along_b {along_b}"
        );
    }

    #[test]
    fn aniso_visibility_is_positive_and_finite() {
        let a = anisotropic_alphas(0.5, 0.4);
        let v = ggx_aniso_visibility(a.alpha_t, a.alpha_b, 0.1, 0.2, 0.9, 0.15, 0.1, 0.8);
        assert!(v > 0.0, "visibility must be positive: {v}");
        assert!(v.is_finite(), "visibility must be finite: {v}");
    }

    #[test]
    fn aniso_visibility_survives_grazing_angles() {
        let a = anisotropic_alphas(0.5, 0.4);
        // NoV = NoL = 0 (fully grazing) must not divide by zero.
        let v = ggx_aniso_visibility(a.alpha_t, a.alpha_b, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0);
        assert!(v.is_finite(), "grazing visibility must be finite: {v}");
        assert!(v > 0.0, "grazing visibility must be positive: {v}");
    }

    #[test]
    fn clearcoat_f0_derives_from_ior() {
        // F0 = ((n-1)/(n+1))² at n = 1.5 is 0.04.
        let n = CLEARCOAT_IOR;
        let r0 = {
            let num = n - 1.0;
            let den = n + 1.0;
            (num / den) * (num / den)
        };
        assert!(approx(r0, CLEARCOAT_F0, EPS), "IOR 1.5 yields F0 0.04");
    }

    #[test]
    fn clearcoat_fresnel_head_on_is_f0_and_grazing_is_one() {
        assert!(
            approx(clearcoat_fresnel(1.0), CLEARCOAT_F0, EPS),
            "head-on Fresnel equals F0"
        );
        assert!(
            approx(clearcoat_fresnel(0.0), 1.0, EPS),
            "grazing Fresnel saturates to 1"
        );
    }

    #[test]
    fn clearcoat_attenuation_is_within_unit_range() {
        for &fc in &[-0.5_f32, 0.0, 0.04, 0.5, 1.0, 2.0] {
            let att = clearcoat_attenuation(fc);
            assert!(att >= 0.0, "attenuation must be >= 0: {att}");
            assert!(att <= 1.0, "attenuation must be <= 1: {att}");
        }
        // Specifically (1 - Fc) for an in-range Fc.
        assert!(approx(clearcoat_attenuation(0.04), 0.96, EPS));
    }

    #[test]
    fn clearcoat_ndf_matches_isotropic_reference() {
        let alpha = 0.1_f32;
        for &n_dot_h in &[1.0_f32, 0.9, 0.7, 0.5] {
            let got = clearcoat_ggx_ndf(n_dot_h, alpha);
            let want = iso_ggx_reference(n_dot_h, alpha);
            assert!(
                approx(got, want, 1e-3 * want.max(1.0)),
                "clearcoat NDF must equal isotropic GGX: got {got}, want {want}"
            );
        }
    }

    #[test]
    fn clearcoat_ndf_and_visibility_are_positive_finite() {
        let d = clearcoat_ggx_ndf(0.8, 0.08);
        let v = clearcoat_visibility(0.7, 0.6, 0.08);
        assert!(
            d > 0.0 && d.is_finite(),
            "clearcoat NDF positive finite: {d}"
        );
        assert!(v > 0.0 && v.is_finite(), "clearcoat V positive finite: {v}");
    }

    #[test]
    fn evaluate_disables_coat_at_zero_strength() {
        let params = AnisotropicClearcoatParams::new(0.4, 0.5, 0.05, 0.0);
        let t = Vec3::new(1.0, 0.0, 0.0);
        let b = Vec3::new(0.0, 1.0, 0.0);
        let n = Vec3::new(0.0, 0.0, 1.0);
        let v = Vec3::new(0.0, 0.3, 1.0);
        let l = Vec3::new(0.2, 0.0, 1.0);
        let s = params.evaluate(t, b, n, v, l);
        assert!(
            approx(s.clearcoat_fresnel, 0.0, EPS),
            "zero strength kills Fc"
        );
        assert!(
            approx(s.base_attenuation, 1.0, EPS),
            "no coat, full base energy"
        );
        assert!(s.base_ndf > 0.0 && s.base_ndf.is_finite());
        assert!(s.base_visibility > 0.0 && s.base_visibility.is_finite());
    }

    #[test]
    fn evaluate_isotropic_base_matches_standalone_ndf() {
        let params = AnisotropicClearcoatParams::new(0.5, 0.0, 0.1, 1.0);
        let t = Vec3::new(1.0, 0.0, 0.0);
        let b = Vec3::new(0.0, 1.0, 0.0);
        let n = Vec3::new(0.0, 0.0, 1.0);
        // View == light == normal makes H == N, so NoH = 1, ToH = BoH = 0.
        let dir = Vec3::new(0.0, 0.0, 1.0);
        let s = params.evaluate(t, b, n, dir, dir);
        let alpha = 0.5_f32 * 0.5;
        let want = iso_ggx_reference(1.0, alpha);
        assert!(
            approx(s.base_ndf, want, 1e-3 * want.max(1.0)),
            "isotropic evaluate base NDF matches reference at the peak: {} vs {}",
            s.base_ndf,
            want
        );
        // With strength 1 and head-on H, Fc = F0 and attenuation = 1 - F0.
        assert!(approx(s.clearcoat_fresnel, CLEARCOAT_F0, EPS));
        assert!(approx(s.base_attenuation, 1.0 - CLEARCOAT_F0, EPS));
    }

    #[test]
    fn evaluate_survives_degenerate_vectors() {
        let params = AnisotropicClearcoatParams::new(0.0, 0.0, 0.0, 1.0);
        let z = Vec3::ZERO;
        let s = params.evaluate(z, z, z, z, z);
        assert!(
            s.base_ndf.is_finite(),
            "degenerate base NDF finite: {}",
            s.base_ndf
        );
        assert!(s.base_visibility.is_finite());
        assert!(s.clearcoat_ndf.is_finite());
        assert!(s.clearcoat_visibility.is_finite());
        assert!(s.clearcoat_fresnel.is_finite());
        assert!((0.0..=1.0).contains(&s.base_attenuation));
    }

    #[test]
    fn params_pack_round_trips_bits() {
        let params = AnisotropicClearcoatParams::new(0.3, -0.4, 0.07, 0.9);
        let bits = params.to_std430_bits();
        assert!(
            approx(f32::from_bits(bits[0]), 0.3, 0.0),
            "roughness round-trips"
        );
        assert!(
            approx(f32::from_bits(bits[1]), -0.4, 0.0),
            "anisotropy round-trips"
        );
        assert!(
            approx(f32::from_bits(bits[2]), 0.07, 0.0),
            "cc roughness round-trips"
        );
        assert!(
            approx(f32::from_bits(bits[3]), 0.9, 0.0),
            "cc strength round-trips"
        );
    }

    #[test]
    fn storage_size_clamps_to_one_element() {
        assert_eq!(
            AnisotropicClearcoatParams::storage_size(0),
            ANISOTROPIC_CLEARCOAT_PARAMS_STRIDE
        );
        assert_eq!(
            AnisotropicClearcoatParams::storage_size(3),
            3 * ANISOTROPIC_CLEARCOAT_PARAMS_STRIDE
        );
    }
}
