//! Independent `OpenPBR` reference oracle for the base layer — the *other* side
//! of the §4.5 constraint-1 conformance gate (design doc §4.5 / roadmap P2-f).
//!
//! ## Why this module exists
//!
//! `openpbr_conformance` (in `prism_render_material`) only checks that the
//! *packed parameter block* is a fixed point of the extension projection. It says
//! nothing about the surface **response**. The design doc's constraint 1 is
//! stronger: a material that uses only `OpenPBR` fields must produce the same
//! BSDF response as the `OpenPBR` reference. Proving that needs a second,
//! *independent* implementation to compare against — if the gate reused the
//! real-time helpers it would only be checking the code against itself (the
//! "self-certifying" failure mode this gate is meant to eliminate).
//!
//! So this module is a **from-the-specification transcription** that shares no
//! code with `lighting.rs` / `clearcoat.rs`. It is intentionally written in the
//! scalar, equation-by-equation style of the specification so a reviewer can
//! diff it against the source.
//!
//! ## Sources (`AcademySoftwareFoundation/OpenPBR`, v1.1.1)
//!
//! * `index.html` §*Microfacet model* — the single-scattering microfacet form
//!   `f(ωi, ωo) ∝ F(ωi, h) · D(h) · G(ωi, ωo)` (equation
//!   `microfacet_brdf_ss`), the GGX normal distribution `D` (equation `GGX` and
//!   its anisotropic form), and the height-correlated Smith `G` (Heitz 2014).
//! * `index.html` §*Anisotropy* — equation `openpbr-anisotropy-formula`,
//!   `α_t = r²·sqrt(2/(1+(1−a)²))`, `α_b = (1−a)·α_t`.
//! * `index.html` §*Dielectric base* — equation `unmodulated_fresnel`
//!   (`F_s = |(1−η_s)/(1+η_s)|²`) and equation `modulated_ior`
//!   (`η' = (1+ε)/(1−ε)`, `ε = sgn(η−1)·sqrt(ξ_s·F_s)`), applied through the
//!   exact angle-dependent dielectric Fresnel `F(μ, η')` (Walter 2007).
//!
//! Only the parts the real-time über currently implements are transcribed here;
//! the module grows lobe-by-lobe together with the gate (the gate asserts the
//! *coverage* it actually has, so it can never silently over-claim).

use core::f32::consts::PI;

use bevy_math::ops;

/// Lower bound on a relative IOR, matching the real-time interface guard.
pub const MIN_IOR: f32 = 1.0 + 1.0e-3;
/// Lower bound on a GGX width, so a degenerate axis never divides by zero.
pub const MIN_ALPHA: f32 = 1.0e-3;

/// `OpenPBR` equation `unmodulated_fresnel`: dielectric Fresnel reflectance at
/// normal incidence, `F_s = |(1 − η_s) / (1 + η_s)|²`.
#[inline]
#[must_use]
pub fn fresnel_normal_incidence(eta: f32) -> f32 {
    let r = (1.0 - eta) / (1.0 + eta);
    r * r
}

/// `OpenPBR` equation `modulated_ior`: the index ratio whose *exact*
/// angle-dependent Fresnel reproduces the `specular_weight`-scaled reflectance.
///
/// `ε = sgn(η_s − 1)·sqrt(ξ_s·F_s)`, `η' = (1 + ε)/(1 − ε)`, with `ξ_s =
/// specular_weight` and `F_s` from [`fresnel_normal_incidence`]. Reductions:
/// `ξ_s = 1` gives back `η_s` (the identity), and `ξ_s = 0` gives `η' = 1`
/// (no reflection at any angle).
#[inline]
#[must_use]
pub fn modulated_ior(eta: f32, specular_weight: f32) -> f32 {
    let w = specular_weight.max(0.0);
    let f_s = fresnel_normal_incidence(eta);
    let sign = if eta >= 1.0 { 1.0 } else { -1.0 };
    // ξ_s == 0 removes the interface entirely: η' == 1, i.e. F(μ) == 0 at every
    // angle (pinned exactly; a clamp to `MIN_IOR` would leak a false rim).
    if w == 0.0 {
        return 1.0;
    }
    // ξ_s > 1 is legal (the spec allows it) but ε must stay below 1 or η' goes
    // negative; the guard maps the unbounded-reflectivity limit onto a large
    // finite index, i.e. F ≈ 1 for every angle.
    let epsilon = (sign * ops::sqrt(w * f_s)).clamp(-(1.0 - 1.0e-4), 1.0 - 1.0e-4);
    let one_minus = 1.0 - epsilon;
    let eta_prime = (1.0 + epsilon) / one_minus;
    if !eta_prime.is_finite() || eta_prime <= 0.0 {
        return MIN_IOR;
    }
    eta_prime
}

/// Exact unpolarized dielectric Fresnel reflectance `F(μ, η)` (Walter 2007),
/// the angle-dependent form the specification plugs the *modulated* index into.
///
/// `μ` is the cosine of the incident angle, `η` the relative index (`η > 1`
/// entering a denser medium). Includes the total-internal-reflection branch.
/// Transcribed independently of `lighting.rs::fresnel_dielectric` — the two are
/// kept honest by the conformance test rather than by sharing code.
#[inline]
#[must_use]
pub fn dielectric_fresnel(mu: f32, eta: f32) -> f32 {
    let mu = mu.clamp(0.0, 1.0);
    let eta = eta.max(1.0e-6);
    // A relative index of exactly one is a non-interface: nothing reflects at
    // any angle (the grazing limit of the formula below would wrongly return 1).
    if (eta - 1.0).abs() <= f32::EPSILON {
        return 0.0;
    }
    // Snell: sin²θ_t = (1 − μ²)/η².
    let sin2_t = (1.0 - mu * mu).max(0.0) / (eta * eta);
    if sin2_t > 1.0 {
        return 1.0;
    }
    let cos_t = ops::sqrt((1.0 - sin2_t).max(0.0));
    // Fresnel's two polarisations, in the η·μ form (equivalent to the
    // (μ − η·μt)/(μ + η·μt) form by Snell's law).
    let r_parl = (eta * mu - cos_t) / (eta * mu + cos_t);
    let r_perp = (mu - eta * cos_t) / (mu + eta * cos_t);
    0.5 * (r_parl * r_parl + r_perp * r_perp)
}

/// The base dielectric's specular Fresnel factor for a **color3** specular tint,
/// per the specification: the exact Fresnel at the *modulated* index, tinted by
/// `specular_color` (which the spec applies only to the initial reflection from
/// above).
#[inline]
#[must_use]
pub fn base_dielectric_fresnel_rgb(
    mu: f32,
    eta: f32,
    specular_weight: f32,
    specular_color: [f32; 3],
) -> [f32; 3] {
    let f = dielectric_fresnel(mu, modulated_ior(eta, specular_weight));
    [
        f * specular_color[0],
        f * specular_color[1],
        f * specular_color[2],
    ]
}

/// The `OpenPBR` §*Metal* "F82-tint" conductor Fresnel (Kutz 2021, extending
/// Hoffman 2019), which the specification stitches from four equations:
///
/// ```text
/// F0            = base_weight · base_color
/// F_Schlick(μ)  = F0 + (1 − F0)(1 − μ)^5
/// F(μ̄)          = specular_color · F_Schlick(μ̄)          with μ̄ = 1/7
/// F82(μ)        = F_Schlick(μ) − μ(1−μ)^6 / (μ̄(1−μ̄)^6) · (F_Schlick(μ̄) − F(μ̄))
/// F_metal(μ)    = specular_weight · F82(μ)
/// ```
///
/// At the default `specular_weight = 1`, `specular_color = white` the correction
/// term vanishes and the curve reduces to plain Schlick — the property the
/// specification calls out, and which the tests pin.
#[inline]
#[must_use]
pub fn base_metal_fresnel_rgb(
    mu: f32,
    base_weight: f32,
    base_color: [f32; 3],
    specular_weight: f32,
    specular_color: [f32; 3],
) -> [f32; 3] {
    /// The specification's fixed "grazing edge" cosine, `μ̄ = 1/7` (≈ 82°).
    const MU_BAR: f32 = 1.0 / 7.0;
    let bw = base_weight.clamp(0.0, 1.0);
    let f0 = [bw * base_color[0], bw * base_color[1], bw * base_color[2]];
    let schlick = |mu: f32| -> [f32; 3] {
        let om = 1.0 - mu.clamp(0.0, 1.0);
        let om2 = om * om;
        let f = om2 * om2 * om;
        [
            f0[0] + (1.0 - f0[0]) * f,
            f0[1] + (1.0 - f0[1]) * f,
            f0[2] + (1.0 - f0[2]) * f,
        ]
    };
    let f_bar_curve = schlick(MU_BAR);
    let om_bar = 1.0 - MU_BAR;
    let om_bar2 = om_bar * om_bar;
    let denom = MU_BAR * om_bar2 * om_bar2 * om_bar2;
    let mu_c = mu.clamp(0.0, 1.0);
    let om = 1.0 - mu_c;
    let om2 = om * om;
    let weight = mu_c * om2 * om2 * om2 / denom;
    let mut out = [0.0_f32; 3];
    for (c, slot) in out.iter_mut().enumerate() {
        let f_edge = specular_color[c].clamp(0.0, 1.0) * f_bar_curve[c];
        let f82 = schlick(mu_c)[c] - weight * (f_bar_curve[c] - f_edge);
        *slot = specular_weight.clamp(0.0, 1.0) * f82;
    }
    out
}

/// `OpenPBR` §Coat *Darkening*: the hemispherical (average) albedo of the coat
/// Fresnel, `E_F(η_c)`, using the specification's closed-form fit
///
/// ```text
/// E_F(η) ≈ ln( (10893·η − 1438.2) / (−774.4·η² + 10212·η + 1) )     for η ≥ 1
/// E_F(η) = 1 − η²·(1 − E_F(1/η))                                    for η < 1
/// ```
#[inline]
#[must_use]
pub fn hemispherical_fresnel(eta: f32) -> f32 {
    let eta = eta.max(MIN_IOR);
    if eta >= 1.0 {
        let num = 10893.0 * eta - 1438.2;
        let den = -774.4 * eta * eta + 10212.0 * eta + 1.0;
        if num <= 0.0 || den <= 0.0 {
            return 0.0;
        }
        ops::ln(num / den).clamp(0.0, 1.0)
    } else {
        let up = hemispherical_fresnel(1.0 / eta);
        (1.0 - eta * eta * (1.0 - up)).clamp(0.0, 1.0)
    }
}

/// `OpenPBR` equation `internal_diffuse_reflection_coefficient_for_rough_base`:
/// `K_r = 1 − (1 − E_F(η_c))/η_c²`.
#[inline]
#[must_use]
pub fn internal_reflection_rough(eta_c: f32) -> f32 {
    let e = hemispherical_fresnel(eta_c);
    (1.0 - (1.0 - e) / (eta_c * eta_c)).clamp(0.0, 1.0)
}

/// `OpenPBR` equations `dielectric_roughness_estimate` / `base_roughness_estimate`:
/// the effective base roughness `r_b` that blends the smooth-base
/// (`internal_diffuse_reflection_coefficient_for_smooth_base`) and rough-base
/// darkening limits.
#[inline]
#[must_use]
pub fn base_roughness_estimate(
    specular_roughness: f32,
    specular_weight: f32,
    specular_ior: f32,
    base_metalness: f32,
) -> f32 {
    let r = specular_roughness.clamp(0.0, 1.0);
    let xi = specular_weight.clamp(0.0, 1.0);
    let f_s = fresnel_normal_incidence(specular_ior);
    // `ξ_s·F_s` must be clamped into [0, 1] before it is used as a lerp weight.
    let w = (xi * f_s).clamp(0.0, 1.0);
    let r_d = 1.0 + (r - 1.0) * w; // lerp(1, r, w)
    let r_m = r;
    let m = base_metalness.clamp(0.0, 1.0);
    r_d + (r_m - r_d) * m
}

/// `OpenPBR` equation `internal_diffuse_reflection_coefficient_for_general_base`:
/// `K = lerp(K_s, K_r, r_b)` with `K_s = F(ω_o, η_c)` and `K_r` from
/// [`internal_reflection_rough`].
#[inline]
#[must_use]
pub fn internal_reflection_general(eta_c: f32, mu_o: f32, r_b: f32) -> f32 {
    let k_s = dielectric_fresnel(mu_o, eta_c);
    let k_r = internal_reflection_rough(eta_c);
    (k_s + (k_r - k_s) * r_b.clamp(0.0, 1.0)).clamp(0.0, 1.0)
}

/// `OpenPBR` equation `general_darkening_formula`: `Δ = (1 − K)/(1 − E_b·K)`.
#[inline]
#[must_use]
pub fn coat_darkening_delta(base_albedo: f32, k: f32) -> f32 {
    let eb = base_albedo.clamp(0.0, 1.0);
    let k = k.clamp(0.0, 1.0);
    let den = 1.0 - eb * k;
    if den.abs() <= f32::MIN_POSITIVE {
        return 1.0;
    }
    ((1.0 - k) / den).clamp(0.0, 1.0)
}

/// `OpenPBR` equation `modulated_darkening_factor`:
/// `(1 − C) + C·B(δ)·Δ = lerp(1, Δ, C·δ)`, the uniform multiplier applied to
/// the whole base BSDF beneath the coat.
#[inline]
#[must_use]
pub fn modulated_darkening_factor(coat_weight: f32, coat_darkening: f32, delta: f32) -> f32 {
    let c = coat_weight.clamp(0.0, 1.0);
    let d = coat_darkening.clamp(0.0, 1.0);
    1.0 + (delta.clamp(0.0, 1.0) - 1.0) * (c * d)
}

/// `OpenPBR` §Coat *Roughening*: a rough coat convolves the base NDF, so the
/// base's effective roughness becomes
/// `r'_B = lerp(r_B, min(1, r_B⁴ + 2·r_C⁴)^(1/4), C)`.
///
/// `C == 0` is the exact identity and `r_C == 0` leaves the base untouched.
#[inline]
#[must_use]
pub fn coat_roughened_base_roughness(r_base: f32, r_coat: f32, coat_weight: f32) -> f32 {
    let rb = r_base.clamp(0.0, 1.0);
    let rc = r_coat.clamp(0.0, 1.0);
    let c = coat_weight.clamp(0.0, 1.0);
    let rb2 = rb * rb;
    let rc2 = rc * rc;
    let convolved = ops::sqrt(ops::sqrt((rb2 * rb2 + 2.0 * rc2 * rc2).min(1.0)));
    rb + (convolved - rb) * c
}

/// `OpenPBR` §Coat *View-dependent absorption*: the volumetric transmittance
/// along the input and output rays,
/// `T_coat^(1/μᵗᵢ + 1/μᵗₒ)` with the ray cosines refracted into the coat,
/// `μᵗ = sqrt(1 − (1 − μ²)/η²_ca)`.
///
/// `coat_color` is the *two-way* normal-incidence transmittance (the
/// specification's `T²_coat`), so the per-pass transmittance is its square root;
/// at normal incidence the factor is exactly `coat_color`. A ray that is totally
/// internally reflected inside the coat (`μᵗ ≤ 0`) travels an unbounded path and
/// therefore contributes no transmittance (`0`).
#[inline]
#[must_use]
pub fn coat_absorption_rgb(
    coat_color: [f32; 3],
    mu_i: f32,
    mu_o: f32,
    eta_ca: f32,
) -> [f32; 3] {
    let mu_t = |mu: f32| -> f32 {
        let m = mu.clamp(0.0, 1.0);
        let s = 1.0 - (1.0 - m * m) / (eta_ca * eta_ca);
        if s <= 0.0 {
            0.0
        } else {
            ops::sqrt(s)
        }
    };
    let ti = mu_t(mu_i);
    let to = mu_t(mu_o);
    if ti <= 0.0 || to <= 0.0 {
        return [0.0; 3];
    }
    // The per-pass transmittance is `sqrt(coat_color)`, so the two-way exponent
    // halves: `coat_color^((1/μᵗᵢ + 1/μᵗₒ)/2)`.
    let exponent = 0.5 * (1.0 / ti + 1.0 / to);
    let mut out = [0.0_f32; 3];
    for (c, slot) in out.iter_mut().enumerate() {
        *slot = ops::powf(coat_color[c].clamp(0.0, 1.0), exponent);
    }
    out
}

/// `OpenPBR` equation `openpbr-anisotropy-formula`: perceptual roughness `r` and
/// anisotropy `a ∈ [0, 1]` → the two GGX widths along the tangent and bitangent.
#[inline]
#[must_use]
pub fn anisotropy_to_alpha(r: f32, a: f32) -> (f32, f32) {
    let r = r.clamp(0.0, 1.0);
    let a = a.clamp(0.0, 1.0);
    let alpha = r * r;
    let one_minus = 1.0 - a;
    let alpha_t = (alpha * ops::sqrt(2.0 / (1.0 + one_minus * one_minus))).max(MIN_ALPHA);
    let alpha_b = (one_minus * alpha_t).max(MIN_ALPHA);
    (alpha_t, alpha_b)
}

/// Anisotropic GGX normal distribution `D(h)` (specification equation `GGX`,
/// anisotropic form), normalised so `∫ D(h) (n·h) dω = 1`.
#[inline]
#[must_use]
pub fn ggx_d(
    x_dot_h: f32,
    y_dot_h: f32,
    n_dot_h: f32,
    alpha_t: f32,
    alpha_b: f32,
) -> f32 {
    let x = x_dot_h / alpha_t;
    let y = y_dot_h / alpha_b;
    let denom = x * x + y * y + n_dot_h * n_dot_h;
    1.0 / (PI * alpha_t * alpha_b * denom * denom)
}

/// Height-correlated Smith masking-shadowing `G2 = 0.5/(Λ_v + Λ_l)`
/// (Heitz 2014), the form the specification prescribes for the GGX NDF.
#[inline]
#[must_use]
pub fn smith_g2(
    x_dot_v: f32,
    y_dot_v: f32,
    x_dot_l: f32,
    y_dot_l: f32,
    n_dot_v: f32,
    n_dot_l: f32,
    alpha_t: f32,
    alpha_b: f32,
) -> f32 {
    fn lambda(x: f32, y: f32, n: f32, alpha_t: f32, alpha_b: f32) -> f32 {
        if n <= 0.0 {
            return f32::INFINITY;
        }
        let x = x / alpha_t;
        let y = y / alpha_b;
        let tan2 = (x * x + y * y) / (n * n);
        0.5 * (ops::sqrt(1.0 + tan2) - 1.0)
    }
    let lv = lambda(x_dot_v, y_dot_v, n_dot_v, alpha_t, alpha_b);
    let ll = lambda(x_dot_l, y_dot_l, n_dot_l, alpha_t, alpha_b);
    let denom = lv + ll;
    if !denom.is_finite() || denom <= 0.0 {
        return 0.0;
    }
    1.0 / (2.0 * denom)
}

/// The single-scattering microfacet BRDF of the base specular lobe under the
/// specification's equations, excluding the `n·l` cosine.
///
/// `F` is supplied by the caller (dielectric per
/// [`base_dielectric_fresnel_rgb`], or conductor per the Metal section) so the
/// gate can exercise either branch.
#[inline]
#[must_use]
#[expect(
    clippy::too_many_arguments,
    reason = "the parameters are exactly the specification's single-scattering \
              BRDF arguments (F, the half-vector/frame cosines and the two GGX \
              widths), kept 1:1 with the source equations for reviewability"
)]
pub fn microfacet_brdf(
    f: [f32; 3],
    x_dot_h: f32,
    y_dot_h: f32,
    n_dot_h: f32,
    x_dot_v: f32,
    y_dot_v: f32,
    x_dot_l: f32,
    y_dot_l: f32,
    n_dot_v: f32,
    n_dot_l: f32,
    alpha_t: f32,
    alpha_b: f32,
) -> [f32; 3] {
    if n_dot_v <= 0.0 || n_dot_l <= 0.0 || n_dot_h <= 0.0 {
        return [0.0; 3];
    }
    let d = ggx_d(x_dot_h, y_dot_h, n_dot_h, alpha_t, alpha_b);
    let g = smith_g2(
        x_dot_v, y_dot_v, x_dot_l, y_dot_l, n_dot_v, n_dot_l, alpha_t, alpha_b,
    );
    let scale = d * g / (4.0 * n_dot_v * n_dot_l);
    [f[0] * scale, f[1] * scale, f[2] * scale]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f32, b: f32, eps: f32) -> bool {
        (a - b).abs() <= eps + eps * b.abs()
    }

    #[test]
    fn unmodulated_fresnel_matches_the_specification_equation() {
        // η = 1.5 ⇒ F_s = (0.5/2.5)² = 0.04 exactly, the spec's worked example.
        assert!(approx(fresnel_normal_incidence(1.5), 0.04, 1e-7));
        assert_eq!(fresnel_normal_incidence(1.0), 0.0);
        // Monotone in η over the physical range.
        let mut prev = -1.0;
        for i in 10..=40 {
            let eta = i as f32 / 10.0;
            let f = fresnel_normal_incidence(eta);
            assert!(f > prev, "not monotone at η={eta}");
            prev = f;
        }
    }

    #[test]
    fn modulated_ior_reduces_to_the_specification_endpoints() {
        // ξ_s = 1 is the identity (the reflectance stays at the authored IOR).
        assert!(approx(modulated_ior(1.5, 1.0), 1.5, 1e-5));
        // ξ_s = 0 removes the reflection: η' = 1, so F(μ) = 0 at every angle.
        let eta0 = modulated_ior(1.5, 0.0);
        assert!(approx(eta0, 1.0, 1e-3), "η'={eta0}");
        for mu in [1.0_f32, 0.7, 0.3, 0.05] {
            assert!(dielectric_fresnel(mu, eta0) < 1e-4);
        }
        // A weight above 1 stays physically meaningful (η' > η_s).
        assert!(modulated_ior(1.5, 1.44) > 1.5);
    }

    #[test]
    fn dielectric_fresnel_has_the_expected_shape() {
        // Normal incidence reproduces the closed-form F_s.
        assert!(approx(dielectric_fresnel(1.0, 1.5), 0.04, 1e-6));
        assert_eq!(dielectric_fresnel(0.0, 1.5), 1.0);
        // Schlick is only an approximation: the exact curve is *not* the Schlick
        // polynomial, which is exactly the deviation the gate must catch.
        let mu = 0.5_f32;
        let om = 1.0 - mu;
        let om2 = om * om;
        let schlick = 0.04 + (1.0 - 0.04) * (om2 * om2 * om);
        assert!(
            (dielectric_fresnel(mu, 1.5) - schlick).abs() > 1e-4,
            "exact={} schlick={schlick}",
            dielectric_fresnel(mu, 1.5)
        );
        // Total internal reflection when entering a rarer medium.
        assert_eq!(dielectric_fresnel(0.3, 0.5), 1.0);
    }

    #[test]
    fn ggx_ndf_is_normalised_and_isotropic_at_a_zero() {
        // a == 0 collapses to the isotropic width r².
        let (at, ab) = anisotropy_to_alpha(0.3, 0.0);
        assert_eq!(at, ab);
        assert!(approx(at, 0.09, 1e-6));
        // The anisotropic widths follow the specification formula.
        let (at_a, ab_a) = anisotropy_to_alpha(0.3, 0.8);
        let expected_t = 0.09 * (2.0_f32 / (1.0 + 0.2 * 0.2)).sqrt();
        assert!(approx(at_a, expected_t, 1e-6));
        assert!(approx(ab_a, 0.2 * expected_t, 1e-6));
        // Numerically integrate D(h)·(n·h) over the hemisphere: 1.0. Integrate in
        // `u = cos²θ` (du = −2 cosθ sinθ dθ) so the thin `θ ≈ α` ridge is
        // resolved by construction rather than by brute-force θ sampling.
        let n = 200_000;
        let mut sum = 0.0_f64;
        for i in 0..n {
            let u = (i as f64 + 0.5) / n as f64;
            let c = u.sqrt();
            let s = (1.0 - u).max(0.0).sqrt();
            let d = ggx_d(s as f32, 0.0, c as f32, at, ab) as f64;
            sum += d; // ∫ D cosθ sinθ dθ dφ = 2π ∫ D du/2 = π ∫ D du
        }
        let integral = sum * core::f64::consts::PI / n as f64;
        assert!(approx(integral as f32, 1.0, 2e-3), "∫D·(n·h) = {integral}");
    }

    #[test]
    fn smith_g2_is_bounded_and_reciprocal() {
        let (at, ab) = anisotropy_to_alpha(0.4, 0.3);
        let (xv, yv, xl, yl) = (0.1, 0.3, 0.2, -0.4);
        let (nv, nl) = (0.9, 0.8);
        let g = smith_g2(xv, yv, xl, yl, nv, nl, at, ab);
        assert!((0.0..=1.0).contains(&g), "G={g}");
        let swapped = smith_g2(xl, yl, xv, yv, nl, nv, at, ab);
        assert!((g - swapped).abs() < 1e-6, "{g} vs {swapped}");
        // Below the horizon the masking term vanishes.
        assert_eq!(smith_g2(xv, yv, xl, yl, -0.1, nl, at, ab), 0.0);
    }

    #[test]
    fn microfacet_brdf_is_reciprocal_and_energy_bounded() {
        // The single-scattering microfacet BRDF is reciprocal (swapping the
        // incident and outgoing directions leaves `f` unchanged) and a white
        // interface cannot reflect more than it receives.
        let (at, ab) = anisotropy_to_alpha(0.4, 0.5);
        let f = [1.0_f32; 3];
        let brdf = |v: [f32; 3], l: [f32; 3]| {
            let h = {
                let s = [v[0] + l[0], v[1] + l[1], v[2] + l[2]];
                let n = (s[0] * s[0] + s[1] * s[1] + s[2] * s[2]).sqrt();
                [s[0] / n, s[1] / n, s[2] / n]
            };
            microfacet_brdf(
                f, h[0], h[2], h[1], v[0], v[2], l[0], l[2], v[1], l[1], at, ab,
            )
        };
        // Normal-up frame: `n = +y`, tangent `+x`, bitangent `+z`.
        let v = [0.2_f32, 0.95, 0.24];
        let l = [-0.3_f32, 0.9, 0.31];
        let a = brdf(v, l);
        let b = brdf(l, v);
        for c in 0..3 {
            assert!((a[c] - b[c]).abs() <= 1.0e-3 * a[c].abs(), "{a:?} vs {b:?}");
            assert!(a[c] >= 0.0 && a[c].is_finite());
        }
        // Below the horizon the BRDF vanishes.
        assert_eq!(
            microfacet_brdf(f, 0.1, 0.1, 0.9, 0.1, 0.1, 0.1, 0.1, -0.5, 0.9, at, ab),
            [0.0; 3]
        );
    }

    #[test]
    fn metal_f82_reduces_to_schlick_and_hits_the_edge_tint() {
        let base_color = [0.9_f32, 0.8, 0.5];
        let bw = 0.75_f32;
        // Default `specular_color` (white) and `specular_weight` (1): the
        // correction term vanishes and the curve is plain Schlick from
        // `base_weight · base_color`.
        for mu in [1.0_f32, 0.9, 0.6, 0.3, 0.1] {
            let got = base_metal_fresnel_rgb(mu, bw, base_color, 1.0, [1.0; 3]);
            for c in 0..3 {
                let f0 = bw * base_color[c];
                let om = 1.0 - mu;
                let om2 = om * om;
                let want = f0 + (1.0 - f0) * (om2 * om2 * om);
                assert!(approx(got[c], want, 1e-6), "mu={mu} c={c}: {got:?}");
            }
        }
        // The defining property of the F82 fit: at `μ̄ = 1/7` the curve is exactly
        // `specular_color · F_Schlick(μ̄)`.
        let tint = [0.6_f32, 0.7, 0.8];
        let mu_bar = 1.0 / 7.0;
        let f_bar_curve = {
            let mut f = [0.0_f32; 3];
            for c in 0..3 {
                let f0 = bw * base_color[c];
                let om = 1.0 - mu_bar;
                let om2 = om * om;
                f[c] = f0 + (1.0 - f0) * (om2 * om2 * om);
            }
            f
        };
        let got = base_metal_fresnel_rgb(mu_bar, bw, base_color, 1.0, tint);
        for c in 0..3 {
            assert!(
                approx(got[c], tint[c] * f_bar_curve[c], 1e-6),
                "c={c}: {} vs {}",
                got[c],
                tint[c] * f_bar_curve[c]
            );
        }
        // The edge tint can only darken the Schlick curve (the spec calls this
        // out as a guarantee), and `specular_weight == 0` removes the lobe.
        for mu in [1.0_f32, 0.5, 0.1] {
            let tinted = base_metal_fresnel_rgb(mu, bw, base_color, 1.0, tint);
            let schlick = base_metal_fresnel_rgb(mu, bw, base_color, 1.0, [1.0; 3]);
            for c in 0..3 {
                assert!(tinted[c] <= schlick[c] + 1e-6);
            }
            assert_eq!(
                base_metal_fresnel_rgb(mu, bw, base_color, 0.0, tint),
                [0.0; 3]
            );
        }
    }
}

/// §4.5 constraint-1 conformance gate: the real-time base dielectric specular
/// must reproduce the specification's Fresnel factor, not merely a lookalike.
///
/// This is the *response-level* half of the gate the design doc calls for
/// (roadmap P2-f); the parameter-block half lives in
/// `prism_render_material::openpbr_conformance`. It compares Prism's shading
/// evaluator against [`super::base_dielectric_fresnel_rgb`], which is transcribed
/// from the specification and shares no code with `lighting.rs`.
#[cfg(test)]
mod conformance {
    use super::*;
    use crate::lighting::base_dielectric_fresnel_rgb as realtime;

    /// The specification's dielectric specular Fresnel factor is
    /// `specular_color · F(μ, η'(specular_ior, specular_weight))` with the
    /// **exact** angle-dependent Fresnel `F` (Walter 2007) and the
    /// `specular_weight`-modulated index of equation `modulated_ior`.
    #[test]
    fn base_dielectric_fresnel_matches_the_openpbr_reference() {
        let mut worst = 0.0_f32;
        let mut worst_at = (0.0_f32, 0.0_f32, 0.0_f32);
        for ior in [1.2_f32, 1.33, 1.5, 2.0, 3.0] {
            for weight in [0.0_f32, 0.25, 0.5, 1.0] {
                for color in [[1.0_f32; 3], [0.8, 0.9, 1.0]] {
                    for cos in [1.0_f32, 0.9, 0.7, 0.5, 0.3, 0.1, 0.02] {
                        // `super::base_dielectric_fresnel_rgb` is the independent
                        // oracle; `realtime` is what `evaluate_principled_direct`
                        // feeds into the dielectric specular lobe.
                        let reference = base_dielectric_fresnel_rgb(cos, ior, weight, color);
                        let prism = realtime(cos, ior, weight, color);
                        for c in 0..3 {
                            let d = (reference[c] - prism[c]).abs();
                            if d > worst {
                                worst = d;
                                worst_at = (ior, weight, cos);
                            }
                        }
                    }
                }
            }
        }
        assert!(
            worst <= 1.0e-4,
            "base dielectric Fresnel deviates from the OpenPBR reference by \
             {worst} (at ior={}, weight={}, cos={}); the real-time evaluator must \
             use the exact Fresnel at the modulated index",
            worst_at.0,
            worst_at.1,
            worst_at.2,
        );
    }

    /// The conductor branch is the specification's `F_82`-tint model: F0 is
    /// `base_weight · base_color`, the near-grazing edge is tinted by
    /// `specular_color` at `μ̄ = 1/7`, and the whole lobe scales by
    /// `specular_weight`.
    #[test]
    fn base_metal_fresnel_matches_the_openpbr_reference() {
        let mut worst = 0.0_f32;
        let mut worst_at = (0.0_f32, 0.0_f32, 0.0_f32);
        for base_color in [[0.9_f32, 0.8, 0.5], [0.2, 0.9, 0.4]] {
            for base_weight in [0.0_f32, 0.5, 1.0] {
                for specular_weight in [0.0_f32, 0.5, 1.0] {
                    for specular_color in [[1.0_f32; 3], [0.6, 0.7, 0.8]] {
                        for cos in [1.0_f32, 0.9, 0.7, 0.5, 0.3, 0.1, 0.02, 1.0 / 7.0] {
                            let reference = base_metal_fresnel_rgb(
                                cos,
                                base_weight,
                                base_color,
                                specular_weight,
                                specular_color,
                            );
                            // What `evaluate_principled_direct` feeds the metal
                            // branch of the base specular lobe.
                            let prism = crate::lighting::base_metal_fresnel_rgb(
                                cos,
                                base_weight,
                                base_color,
                                specular_weight,
                                specular_color,
                            );
                            for c in 0..3 {
                                let d = (reference[c] - prism[c]).abs();
                                if d > worst {
                                    worst = d;
                                    worst_at = (base_weight, specular_weight, cos);
                                }
                            }
                        }
                    }
                }
            }
        }
        assert!(
            worst <= 1.0e-5,
            "metal Fresnel deviates from the OpenPBR F82-tint reference by \
             {worst} (at base_weight={}, specular_weight={}, cos={}); the \
             real-time evaluator must apply the edge tint and the weight",
            worst_at.0,
            worst_at.1,
            worst_at.2,
        );
    }

    /// `OpenPBR` §Coat *Darkening*: the hemispherical Fresnel fit, the
    /// internal-reflection coefficients and the `modulated_darkening_factor` the
    /// real-time Clearcoat lobe applies to the base beneath the coat.
    #[test]
    fn coat_darkening_matches_the_openpbr_reference() {
        // Hemispherical Fresnel: zero at a vanishing interface, monotone, and
        // the reciprocity identity holds across the physical index range.
        // The fit is exact only in form: at a vanishing interface (η = 1) it
        // lands within ~2e-3 of the true zero, which is the fit's documented
        // accuracy, not a bug to paper over.
        assert!(hemispherical_fresnel(1.0) < 5.0e-3);
        let mut prev = -1.0;
        for i in 10..=40 {
            let eta = i as f32 / 10.0;
            let e = hemispherical_fresnel(eta);
            assert!((0.0..=1.0).contains(&e), "E_F({eta}) = {e}");
            assert!(e >= prev, "not monotone at η={eta}");
            prev = e;
        }

        let mut worst = 0.0_f32;
        let mut worst_at = (0.0_f32, 0.0_f32, 0.0_f32, 0.0_f32);
        for eta_c in [1.1_f32, 1.3, 1.6, 2.0] {
            for mu_o in [1.0_f32, 0.8, 0.5, 0.2, 0.05] {
                for r_b in [0.0_f32, 0.35, 0.7, 1.0] {
                    // Prism's real-time Clearcoat helpers.
                    let k_rt = crate::clearcoat::coat_internal_reflection_coefficient(
                        eta_c, mu_o, r_b,
                    );
                    let k_ref = internal_reflection_general(eta_c, mu_o, r_b);
                    for base_albedo in [0.05_f32, 0.25, 0.5, 0.9] {
                        let d_rt = crate::clearcoat::coat_darkening_delta(base_albedo, k_rt);
                        let d_ref = coat_darkening_delta(base_albedo, k_ref);
                        for coat_weight in [0.0_f32, 0.5, 1.0] {
                            for coat_darkening in [0.0_f32, 0.5, 1.0] {
                                let m_rt = crate::clearcoat::coat_modulated_darkening(
                                    coat_weight,
                                    coat_darkening,
                                    d_rt,
                                );
                                let m_ref = modulated_darkening_factor(
                                    coat_weight,
                                    coat_darkening,
                                    d_ref,
                                );
                                let d = (m_rt - m_ref).abs();
                                if d > worst {
                                    worst = d;
                                    worst_at = (eta_c, mu_o, r_b, base_albedo);
                                }
                            }
                        }
                    }
                }
            }
        }
        assert!(
            worst <= 1.0e-4,
            "coat darkening deviates from the OpenPBR reference by {worst} \
             (at eta_c={}, mu_o={}, r_b={}, E_b={}); the real-time Clearcoat lobe \
             must use the specification's modulated darkening factor",
            worst_at.0,
            worst_at.1,
            worst_at.2,
            worst_at.3,
        );

        // The specification's defining reductions.
        // δ = 0 restores the un-darkened base exactly (the artist escape hatch).
        assert_eq!(modulated_darkening_factor(1.0, 0.0, 0.2), 1.0);
        // C = 0 leaves the base untouched whatever the darkening.
        assert_eq!(modulated_darkening_factor(0.0, 1.0, 0.2), 1.0);
        // A white base cannot darken: Δ(E_b = 1) == 1.
        assert!((coat_darkening_delta(1.0, internal_reflection_rough(1.6)) - 1.0).abs() < 1e-6);
        // And the real-time roughness estimate agrees with the specification's.
        let mut worst_r = 0.0_f32;
        for r in [0.0_f32, 0.3, 0.7, 1.0] {
            for weight in [0.0_f32, 0.5, 1.0] {
                for ior in [1.2_f32, 1.5, 2.0, 3.0] {
                    for m in [0.0_f32, 0.5, 1.0] {
                        let rt = crate::clearcoat::coat_base_roughness_estimate(r, weight, ior, m);
                        let rf = base_roughness_estimate(r, weight, ior, m);
                        worst_r = worst_r.max((rt - rf).abs());
                    }
                }
            }
        }
        assert!(worst_r <= 1.0e-6, "base roughness estimate deviates by {worst_r}");
    }

    /// `OpenPBR` §Coat *Roughening*: the real-time base roughness under the coat
    /// must be the specification's convolution `lerp(r_B, min(1, r_B⁴+2r_C⁴)^¼, C)`.
    #[test]
    fn coat_roughening_matches_the_openpbr_reference() {
        let mut worst = 0.0_f32;
        for r_base in [0.0_f32, 0.15, 0.4, 0.75, 1.0] {
            for r_coat in [0.0_f32, 0.1, 0.35, 0.8, 1.0] {
                for c in [0.0_f32, 0.25, 0.5, 1.0] {
                    let rt = crate::clearcoat::coat_roughened_base_roughness(r_base, r_coat, c);
                    let rf = coat_roughened_base_roughness(r_base, r_coat, c);
                    worst = worst.max((rt - rf).abs());
                }
            }
        }
        assert!(worst <= 1.0e-6, "coat roughening deviates by {worst}");
        // Defining reductions: no coat ⇒ identity; smooth coat ⇒ identity; and
        // the roughened base never gets smoother than the authored base.
        for r in [0.0_f32, 0.3, 0.9] {
            assert_eq!(coat_roughened_base_roughness(r, 0.7, 0.0), r);
            assert_eq!(coat_roughened_base_roughness(r, 0.0, 1.0), r);
            assert!(coat_roughened_base_roughness(r, 0.7, 1.0) >= r);
        }
    }

    /// `OpenPBR` §Coat *View-dependent absorption*: the base beneath an absorbing
    /// coat is tinted by the two-way transmittance, and the coat's own highlight
    /// is not.
    #[test]
    fn coat_absorption_matches_the_openpbr_reference() {
        let mut worst = 0.0_f32;
        for coat_color in [[1.0_f32; 3], [0.8, 0.5, 0.2], [0.0; 3]] {
            for eta in [1.0_f32, 1.3, 1.6, 2.0] {
                for mu_i in [1.0_f32, 0.8, 0.5, 0.2, 0.05] {
                    for mu_o in [1.0_f32, 0.6, 0.1] {
                        let rt =
                            crate::clearcoat::coat_absorption_rgb(coat_color, mu_i, mu_o, eta);
                        let rf = coat_absorption_rgb(coat_color, mu_i, mu_o, eta);
                        for c in 0..3 {
                            worst = worst.max((rt[c] - rf[c]).abs());
                        }
                    }
                }
            }
        }
        assert!(worst <= 1.0e-5, "coat absorption deviates by {worst}");
        // At normal incidence the two-way transmittance is exactly `coat_color`
        // (the specification's definition of the parameter).
        let color = [0.8_f32, 0.5, 0.2];
        let normal = coat_absorption_rgb(color, 1.0, 1.0, 1.6);
        for c in 0..3 {
            assert!((normal[c] - color[c]).abs() < 1.0e-5, "{normal:?} vs {color:?}");
        }
        // A grazing view travels a longer path through the medium, so it can only
        // darken further (never brighten), and white is the exact identity.
        let grazing = coat_absorption_rgb(color, 0.1, 0.1, 1.6);
        for c in 0..3 {
            assert!(grazing[c] <= normal[c] + 1e-6);
        }
        assert_eq!(coat_absorption_rgb([1.0; 3], 0.1, 0.2, 1.6), [1.0; 3]);
        // Total internal reflection inside the coat blocks transmission entirely.
        assert_eq!(coat_absorption_rgb(color, 0.1, 0.2, 0.5), [0.0; 3]);
    }
}
