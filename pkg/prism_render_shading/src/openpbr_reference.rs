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

// ---------------------------------------------------------------------------
// §Base dielectric / *glossy-diffuse*: the `EON` (energy-preserving Oren-Nayar)
// diffuse lobe.
//
// The specification defines the diffuse BRDF as `f_ON + f_comp` (equations
// `EON_brdf`, `FON_brdf`, `EON_comp`). Prism transcribes `f_ON` — the Fujii form
// — directly, and derives the albedo inputs the specification leaves to
// [Portsmouth2024] (`Ê_ON(ω)`, `⟨Ê_ON⟩`) by **numerical quadrature of the
// specification's own `FON_brdf` equation**, rather than inventing a fit. Those
// two albedos are the only inputs `EON_comp` needs, so this keeps the
// compensation term spec-derived and verifiable.
// ---------------------------------------------------------------------------

/// Oren-Nayar `A(σ)` coefficient (specification `FON_brdf`, Fujii 2012 form).
#[inline]
#[must_use]
pub fn fon_a(sigma: f32) -> f32 {
    let s2 = sigma * sigma;
    1.0 - 0.5 * s2 / (s2 + 0.33)
}

/// Oren-Nayar `B(σ)` coefficient (specification `FON_brdf`).
#[inline]
#[must_use]
pub fn fon_b(sigma: f32) -> f32 {
    let s2 = sigma * sigma;
    0.45 * s2 / (s2 + 0.09)
}

/// The unit-`ρ` Fujii Oren-Nayar BRDF `(1/π)(A(σ) + B(σ)·s/t)` for a direction
/// pair given by its cosines and the `s = sinθ_i·sinθ_o·cosΔφ` term. Mirrors
/// `lighting.rs::oren_nayar_factor` (which omits the `1/π`).
#[inline]
#[must_use]
pub fn fon_unit_brdf(cos_i: f32, cos_o: f32, s: f32, sigma: f32) -> f32 {
    let t = if s > 0.0 {
        cos_i.max(cos_o).max(1.0e-6)
    } else {
        1.0
    };
    (fon_a(sigma) + fon_b(sigma) * s / t).max(0.0) / PI
}

/// The unit-`ρ` directional albedo `Ê_ON(μ_o, σ)` of the Fujii Oren-Nayar lobe,
/// `∫ f_ON(ω_i, ω_o)·(n·ω_i) dω_i` with `ρ = 1`.
///
/// Evaluated by deterministic Gauss-free midpoint quadrature of the
/// specification's `FON_brdf`; the specification states an analytic form exists
/// but defers it to [Portsmouth2024], so quadrature keeps this oracle
/// independent of any invented fit.
#[must_use]
pub fn eon_directional_albedo(mu_o: f32, sigma: f32) -> f32 {
    const N_THETA: usize = 64;
    const N_PHI: usize = 64;
    let cos_o = mu_o.clamp(0.0, 1.0);
    let sin_o = ops::sqrt((1.0 - cos_o * cos_o).max(0.0));
    let d_theta = core::f32::consts::FRAC_PI_2 / N_THETA as f32;
    let d_phi = 2.0 * PI / N_PHI as f32;
    let mut sum = 0.0_f32;
    for i in 0..N_THETA {
        let theta = (i as f32 + 0.5) * d_theta;
        let (sin_i, cos_i) = (ops::sin(theta), ops::cos(theta));
        for j in 0..N_PHI {
            let phi = (j as f32 + 0.5) * d_phi;
            let s = sin_i * sin_o * ops::cos(phi);
            sum += fon_unit_brdf(cos_i, cos_o, s, sigma) * cos_i * sin_i * d_theta * d_phi;
        }
    }
    sum
}

/// The cosine-weighted average albedo `⟨Ê_ON⟩(σ) = 2∫ Ê_ON(μ, σ) μ dμ`.
#[must_use]
pub fn eon_average_albedo(sigma: f32) -> f32 {
    const N: usize = 64;
    let mut sum = 0.0_f32;
    for i in 0..N {
        let mu = (i as f32 + 0.5) / N as f32;
        sum += eon_directional_albedo(mu, sigma) * mu * (1.0 / N as f32);
    }
    2.0 * sum
}

/// The `EON_comp` multiple-scattering albedo `ρ_ms` for a `ρ`-scaled base.
///
/// **Specification erratum (verified numerically).** The published LaTeX writes
/// `ρ_ms = (ρ²/π)·(⟨Ê⟩/(1−⟨Ê⟩))/(1 − ρ(1−⟨Ê⟩))`, but that factor of `1/π`
/// contradicts the invariant the same paragraph states — *"as ρ → 1, the total
/// directional albedo `E_diffuse(ω)` → 1"* — and is dimensionally inconsistent
/// (`f_comp` already carries a `1/π`, so `ρ_ms` must be dimensionless). Working
/// the integral for `ρ = 1`:
///
/// ```text
/// ∫ f_comp·(n·ωi) dωi = ρ_ms·(1 − Ê(ωo))·π(1 − ⟨Ê⟩)
/// ```
///
/// so `E_diffuse = Ê + (1 − Ê)/π` with the literal formula (≈ 0.70 at
/// `σ = π/2`, not 1), and exactly `Ê + (1 − Ê) = 1` without it. Prism therefore
/// drops the `/π` from `ρ_ms` — keeping the `1/π` where it belongs, in
/// `f_comp` — which is the only reading consistent with the specification's own
/// white-furnace guarantee. `eon_lobe_passes_the_white_furnace` pins it.
#[inline]
#[must_use]
pub fn eon_rho_ms(rho: f32, sigma: f32) -> f32 {
    let e = eon_average_albedo(sigma).clamp(0.0, 1.0 - 1.0e-4);
    let r = rho.clamp(0.0, 1.0);
    let num = e / (1.0 - e);
    let den = (1.0 - r * (1.0 - e)).max(1.0e-4);
    r * r * num / den
}

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
pub fn coat_absorption_rgb(coat_color: [f32; 3], mu_i: f32, mu_o: f32, eta_ca: f32) -> [f32; 3] {
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

/// `OpenPBR` equation `specular_ior_ratio` with the TIR fix
/// (`specular_ior_ratio_with_tir_fix`): the base dielectric's index ratio when a
/// partial coat makes its surrounding medium statistically the coat,
/// `η_s = lerp(n_b/n_a, η_bc_fixed, C)` with `n_a = 1` and `η_bc_fixed =
/// n_c/n_b` when the coat is denser than the base (otherwise `n_b/n_c`).
///
/// The fix replaces the naive `n_b/n_c`, which would let a refracted ray reach
/// the base beyond the critical angle and produce a spurious bright ring near
/// grazing. Both branches yield `≥ 1`, so the ratio is always physical.
#[inline]
#[must_use]
pub fn coat_specular_ior_ratio(specular_ior: f32, coat_ior: f32, coat_weight: f32) -> f32 {
    let n_b = specular_ior.clamp(1.0, 3.0);
    let n_c = coat_ior.max(MIN_IOR);
    let eta_bc_fixed = if n_c > n_b { n_c / n_b } else { n_b / n_c };
    (n_b + (eta_bc_fixed - n_b) * coat_weight.clamp(0.0, 1.0)).clamp(1.0, 3.0)
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
pub fn ggx_d(x_dot_h: f32, y_dot_h: f32, n_dot_h: f32, alpha_t: f32, alpha_b: f32) -> f32 {
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

    /// `OpenPBR` §*Testing*: the `EON` diffuse lobe (`f_ON + f_comp`) must pass
    /// the white furnace — its directional albedo has to reach 1 for `ρ = 1`,
    /// which is exactly what equation `EON_comp` exists to guarantee. This also
    /// validates the quadrature-derived albedo the specification defers to
    /// [Portsmouth2024].
    #[test]
    fn eon_lobe_passes_the_white_furnace() {
        const N_THETA: usize = 96;
        const N_PHI: usize = 96;
        let d_theta = core::f32::consts::FRAC_PI_2 / N_THETA as f32;
        let d_phi = 2.0 * PI / N_PHI as f32;
        let mut worst = (0.0_f32, String::new());
        for roughness in [0.0_f32, 0.3, 0.6, 1.0] {
            let sigma = roughness * core::f32::consts::FRAC_PI_2;
            // `Ê_ON` depends only on the polar angle, so evaluate it once per
            // ring (and once for the view) rather than inside the quadrature.
            let e_i: Vec<f32> = (0..N_THETA)
                .map(|i| {
                    let theta = (i as f32 + 0.5) * d_theta;
                    eon_directional_albedo(ops::cos(theta), sigma)
                })
                .collect();
            let rho_ms = eon_rho_ms(1.0, sigma);
            for mu_o in [1.0_f32, 0.7, 0.3, 0.1] {
                let cos_o = mu_o;
                let sin_o = (1.0 - cos_o * cos_o).max(0.0).sqrt();
                let e_o = eon_directional_albedo(cos_o, sigma);
                let mut sum = 0.0_f32;
                for (i, e_i_ring) in e_i.iter().enumerate() {
                    let theta = (i as f32 + 0.5) * d_theta;
                    let (sin_i, cos_i) = (ops::sin(theta), ops::cos(theta));
                    let comp = (rho_ms / PI) * (1.0 - e_i_ring).max(0.0) * (1.0 - e_o).max(0.0);
                    for j in 0..N_PHI {
                        let phi = (j as f32 + 0.5) * d_phi;
                        let s = sin_i * sin_o * ops::cos(phi);
                        let f = fon_unit_brdf(cos_i, cos_o, s, sigma) + comp;
                        sum += f * cos_i * sin_i * d_theta * d_phi;
                    }
                }
                let err = (sum - 1.0).abs();
                if err > worst.0 {
                    worst = (err, format!("roughness={roughness} mu_o={mu_o} -> {sum}"));
                }
            }
        }
        assert!(
            worst.0 <= 0.03,
            "EON diffuse must pass the white furnace: {} (error {})",
            worst.1,
            worst.0
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
    #![allow(clippy::print_stdout, reason = "diagnostic prints furnace convergence")]
    use super::*;
    use crate::lighting::base_dielectric_fresnel_rgb as realtime;
    use crate::{evaluate_principled_direct, DirectLightSample, ShadingFrame, SurfaceSample};

    /// Directional albedo of the real-time base lobe: the hemisphere integral of
    /// `f·(n·l)` under a unit uniform environment, i.e. exactly what the
    /// specification's **white furnace test** checks. Deterministic stratified
    /// quadrature in `(θ, φ)` with a cosine-friendly midpoint rule.
    /// Same quadrature for any direct-lighting closure (the Cloth and
    /// Subsurface shading classes have their own entry points).
    fn furnace_albedo_fn(
        f: impl Fn(SurfaceSample, ShadingFrame, DirectLightSample) -> [f32; 3],
        surface: SurfaceSample,
        view: [f32; 3],
        n_theta: usize,
        n_phi: usize,
    ) -> [f32; 3] {
        let n = [0.0_f32, 1.0, 0.0];
        let len = (view[0] * view[0] + view[1] * view[1] + view[2] * view[2]).sqrt();
        let v = [view[0] / len, view[1] / len, view[2] / len];
        let frame = ShadingFrame {
            normal: n,
            view: v,
            tangent: [1.0, 0.0, 0.0],
            bitangent: [0.0, 0.0, -1.0],
        };
        let d_theta = core::f32::consts::FRAC_PI_2 / n_theta as f32;
        let d_phi = 2.0 * PI / n_phi as f32;
        let mut sum = [0.0_f32; 3];
        for i in 0..n_theta {
            let theta = (i as f32 + 0.5) * d_theta;
            let (sin_t, cos_t) = (ops::sin(theta), ops::cos(theta));
            for j in 0..n_phi {
                let phi = (j as f32 + 0.5) * d_phi;
                let l = [sin_t * ops::cos(phi), cos_t, sin_t * ops::sin(phi)];
                let r = f(
                    surface,
                    frame,
                    DirectLightSample {
                        direction: l,
                        illuminance: [1.0; 3],
                        visibility: 1.0,
                    },
                );
                let w = sin_t * d_theta * d_phi;
                for c in 0..3 {
                    sum[c] += r[c] * w;
                }
            }
        }
        sum
    }

    /// Quadrature with an explicit resolution, so tests can demonstrate
    /// convergence for a peaked (low-roughness) lobe.
    fn furnace_albedo(
        surface: SurfaceSample,
        view: [f32; 3],
        n_theta: usize,
        n_phi: usize,
    ) -> [f32; 3] {
        let (n_theta, n_phi) = (n_theta, n_phi);
        let n = [0.0_f32, 1.0, 0.0];
        let v = {
            let len = (view[0] * view[0] + view[1] * view[1] + view[2] * view[2]).sqrt();
            [view[0] / len, view[1] / len, view[2] / len]
        };
        let frame = ShadingFrame {
            normal: n,
            view: v,
            tangent: [1.0, 0.0, 0.0],
            bitangent: [0.0, 0.0, -1.0],
        };
        let d_theta = core::f32::consts::FRAC_PI_2 / n_theta as f32;
        let d_phi = 2.0 * PI / n_phi as f32;
        let mut sum = [0.0_f32; 3];
        for i in 0..n_theta {
            let theta = (i as f32 + 0.5) * d_theta;
            let (sin_t, cos_t) = (ops::sin(theta), ops::cos(theta));
            for j in 0..n_phi {
                let phi = (j as f32 + 0.5) * d_phi;
                let l = [sin_t * ops::cos(phi), cos_t, sin_t * ops::sin(phi)];
                let radiance = evaluate_principled_direct(
                    surface,
                    frame,
                    DirectLightSample {
                        direction: l,
                        illuminance: [1.0; 3],
                        visibility: 1.0,
                    },
                );
                // `evaluate_principled_direct` already carries `n·l`; the
                // hemisphere measure is `sinθ dθ dφ`.
                let w = sin_t * d_theta * d_phi;
                for c in 0..3 {
                    sum[c] += radiance[c] * w;
                }
            }
        }
        sum
    }

    /// Deterministic multiple-importance estimate of the same white-furnace
    /// integral. Half of the samples follow the GGX NDF and half follow a
    /// cosine hemisphere. The mixture resolves near-mirror highlights without
    /// losing the broad multiscatter compensation term.
    fn furnace_albedo_mis(
        surface: SurfaceSample,
        view: [f32; 3],
        sample_count_per_technique: u32,
    ) -> [f32; 3] {
        fn radical_inverse_vdc(bits: u32) -> f32 {
            let reversed = bits.reverse_bits();
            (reversed as f64 * (1.0 / 4_294_967_296.0)) as f32
        }

        fn normalize(v: [f32; 3]) -> [f32; 3] {
            let length = ops::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
            [v[0] / length, v[1] / length, v[2] / length]
        }

        fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
            a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
        }

        fn ggx_half_vector(u1: f32, u2: f32, alpha: f32) -> [f32; 3] {
            let phi = 2.0 * PI * u1;
            let alpha_sq = alpha * alpha;
            let cos_theta = ops::sqrt(((1.0 - u2) / (1.0 + (alpha_sq - 1.0) * u2)).clamp(0.0, 1.0));
            let sin_theta = ops::sqrt((1.0 - cos_theta * cos_theta).max(0.0));
            [
                sin_theta * ops::cos(phi),
                cos_theta,
                sin_theta * ops::sin(phi),
            ]
        }

        fn cosine_direction(u1: f32, u2: f32) -> [f32; 3] {
            let radius = ops::sqrt(u1);
            let phi = 2.0 * PI * u2;
            [
                radius * ops::cos(phi),
                ops::sqrt((1.0 - u1).max(0.0)),
                radius * ops::sin(phi),
            ]
        }

        assert!(sample_count_per_technique > 0);
        let n = [0.0_f32, 1.0, 0.0];
        let v = normalize(view);
        let frame = ShadingFrame {
            normal: n,
            view: v,
            tangent: [1.0, 0.0, 0.0],
            bitangent: [0.0, 0.0, -1.0],
        };
        let roughness = surface.perceptual_roughness.clamp(0.045, 1.0);
        let alpha = roughness * roughness;
        let mut sum = [0.0_f64; 3];

        for technique in 0..2_u32 {
            for index in 0..sample_count_per_technique {
                let u1 = (index as f32 + 0.5) / sample_count_per_technique as f32;
                let u2 = radical_inverse_vdc(index);
                let l = if technique == 0 {
                    let h = ggx_half_vector(u1, u2, alpha);
                    let v_dot_h = dot(v, h);
                    if v_dot_h <= 0.0 {
                        continue;
                    }
                    normalize([
                        2.0 * v_dot_h * h[0] - v[0],
                        2.0 * v_dot_h * h[1] - v[1],
                        2.0 * v_dot_h * h[2] - v[2],
                    ])
                } else {
                    cosine_direction(u1, u2)
                };
                let n_dot_l = l[1];
                if n_dot_l <= 0.0 {
                    continue;
                }
                let h = normalize([v[0] + l[0], v[1] + l[1], v[2] + l[2]]);
                let n_dot_h = h[1].max(0.0);
                let v_dot_h = dot(v, h).max(1.0e-8);
                // The PDF must match the exact inverse-CDF sampler above. The
                // production evaluator deliberately floors its denominator for
                // finite shader arithmetic; reusing that clamped value here
                // would no longer integrate to one for a near-mirror lobe.
                let alpha_sq = alpha * alpha;
                let denominator = n_dot_h * n_dot_h * (alpha_sq - 1.0) + 1.0;
                let d = alpha_sq / (PI * denominator * denominator);
                let ggx_pdf = d * n_dot_h / (4.0 * v_dot_h);
                let cosine_pdf = n_dot_l / PI;
                let mixture_pdf = 0.5 * (ggx_pdf + cosine_pdf);
                if !mixture_pdf.is_finite() || mixture_pdf <= 0.0 {
                    continue;
                }
                let radiance = evaluate_principled_direct(
                    surface,
                    frame,
                    DirectLightSample {
                        direction: l,
                        illuminance: [1.0; 3],
                        visibility: 1.0,
                    },
                );
                for c in 0..3 {
                    sum[c] += f64::from(radiance[c] / mixture_pdf);
                }
            }
        }

        let denominator = f64::from(2 * sample_count_per_technique);
        [
            (sum[0] / denominator) as f32,
            (sum[1] / denominator) as f32,
            (sum[2] / denominator) as f32,
        ]
    }

    /// `OpenPBR` §*Testing*: the specification lists the configurations in which
    /// a **white furnace test** must pass (directional albedo → 1 under a unit
    /// uniform environment). This is the energy-conservation half of the
    /// conformance gate: it needs no external reference implementation, only the
    /// specification's stated invariant.
    /// Diagnostic: white-furnace quadrature convergence for the peaked
    /// low-roughness metal case (run with `--ignored --nocapture`).
    #[test]
    #[ignore = "diagnostic: prints furnace convergence vs quadrature resolution"]
    fn print_furnace_convergence() {
        for roughness in [0.1_f32, 0.3, 0.8] {
            let surface = SurfaceSample {
                base_color: [1.0; 3],
                metallic: 1.0,
                perceptual_roughness: roughness,
                specular_color: [1.0; 3],
                specular_weight: 1.0,
                ..Default::default()
            };
            let mut line = format!("roughness={roughness}: ");
            for res in [96usize, 192, 384, 768, 1536] {
                let a = furnace_albedo(surface, [0.95, 0.31, 0.0], res, res);
                line += &format!("{res}->{:.4} ", (a[0] + a[1] + a[2]) / 3.0);
            }
            println!("{line}");
        }
    }

    #[test]
    fn white_furnace_conserves_energy_for_the_specification_configurations() {
        let white = [1.0_f32; 3];
        let metal = |roughness: f32| SurfaceSample {
            base_color: white,
            metallic: 1.0,
            perceptual_roughness: roughness,
            specular_color: white,
            specular_weight: 1.0,
            ..Default::default()
        };
        // `specular_weight = 0` removes the dielectric interface, isolating the
        // diffuse lobe so the furnace measures the EON energy alone. With the
        // interface present the *coupling* term confounds it: Prism weights the
        // diffuse by `1 − F(v·h)` where the specification prescribes
        // `1 − E_dielectric(ωo)` (equation `glossy_diffuse_albedo_constraint`),
        // which over-delivers ~12% at grazing (measured below and tracked as the
        // next fix).
        let pure_diffuse = |diffuse_roughness: f32| SurfaceSample {
            base_color: white,
            metallic: 0.0,
            perceptual_roughness: 0.3,
            diffuse_roughness,
            specular_color: white,
            specular_weight: 0.0,
            ..Default::default()
        };
        // With the dielectric interface present the diffuse *coupling* is under
        // test as well, which is the invariant the `1 − F(v·h)` weight used to
        // break at grazing (it measured 1.12).
        let coupled = |diffuse_roughness: f32| SurfaceSample {
            base_color: white,
            metallic: 0.0,
            perceptual_roughness: 0.3,
            diffuse_roughness,
            specular_color: white,
            specular_weight: 1.0,
            ..Default::default()
        };
        // Specification: translucent base (`transmission_weight = 1`) with white
        // `transmission_color` at `transmission_depth == 0`.
        let translucent = SurfaceSample {
            base_color: white,
            metallic: 0.0,
            perceptual_roughness: 0.3,
            specular_color: white,
            specular_weight: 1.0,
            transmission: 1.0,
            transmission_color: white,
            index_of_refraction: 1.5,
            thickness: 0.0,
            ..Default::default()
        };
        // Specification: the same dielectric base under a white coat / white fuzz
        // (the coat and fuzz layers must not create or destroy energy).
        // `clearcoat_roughness` must be resolvable by the uniform hemisphere
        // grid: the default (0) is a near-mirror lobe whose energy the
        // quadrature misses entirely (measured coat-only albedo 0.001 vs the true
        // 0.28), exactly like the `metal(0.1)` case above.
        let coated = SurfaceSample {
            clearcoat: 1.0,
            clearcoat_roughness: 0.25,
            coat_color: white,
            ..coupled(0.0)
        };
        // Specification: fuzz on top (`fuzz_weight > 0`, white `fuzz_color`). In
        // Prism the fuzz layer lives in the Cloth shading class, so it has its own
        // closure and its own furnace check below.
        let fuzzy = SurfaceSample {
            fuzz_color: white,
            sheen: 1.0,
            ..coupled(0.0)
        };
        let configs: [(&str, SurfaceSample); 8] = [
            ("metal, default roughness", metal(0.3)),
            ("metal, rough", metal(0.8)),
            ("pure diffuse base, Lambertian", pure_diffuse(0.0)),
            (
                "pure diffuse base, maximum diffuse roughness",
                pure_diffuse(1.0),
            ),
            ("dielectric base, Lambertian", coupled(0.0)),
            ("dielectric base, maximum diffuse roughness", coupled(1.0)),
            ("translucent base (transmission_weight=1)", translucent),
            ("dielectric base under a white coat", coated),
        ];
        let mut worst = (0.0_f32, String::new());
        // One gate, three energy invariants it has already caught: the diffuse
        // `EON_comp` gap (white rough diffuse reflected 0.577), the multiscatter
        // compensation's view-independent albedo (white metal read 1.11), and
        // the glossy-diffuse coupling weight at grazing.
        //
        // Still missing from the specification's list, and *measured* here:
        //
        // * **coat** (`coat_weight = 1`, white `coat_color`) reads **0.72** at
        //   grazing. Diagnosis: the base is attenuated by the closed-form slab
        //   transmission `(1−f_v)(1−f_l)/(1−f_v·f_l)`, but the coat's own lobe is
        //   a *single-scatter* GGX microfacet lobe whose Fresnel is evaluated at
        //   `v·h`, so it does not carry the complementary slab reflectance
        //   `r_slab` that the transmission withholds. Two attempts were measured
        //   and rejected rather than shipped: adding the same multiscatter
        //   compensation the base uses (0.72 → 0.736, still 26% short — the
        //   Turquin coloured scale is ~0.009 at `F0 = 0.053`), and renormalising
        //   the lobe by `r_slab / (F(v)·E_white)` (0.72 → 0.719). Closing this
        //   needs a deliberate coat-energy design (e.g. distributing `r_slab`
        //   over the lobe's shape with an accurate lobe albedo), not another
        //   correction term.
        // * **fuzz** on top and **subsurface** (a separate shading class) are not
        //   covered yet either.
        for (name, surface) in configs {
            // Normal, mid and near-grazing view directions all have to conserve.
            for view in [[0.0_f32, 1.0, 0.0], [0.6, 0.8, 0.0], [0.95, 0.31, 0.0]] {
                // 384² is converged for the configurations below (the probe shows
                // the metal cases stable from 384² on and the diffuse cases are
                // smooth); the dielectric cases are cheap at 192².
                let res = if name.starts_with("metal") { 384 } else { 192 };
                let albedo = furnace_albedo(surface, view, res, res);
                let mean = (albedo[0] + albedo[1] + albedo[2]) / 3.0;
                // Every configuration listed by the specification must reach
                // 1, metal and dielectric alike: the metal case is what exposed
                // the compensation's view-independent albedo (it used to read
                // 1.11), so it is asserted to the same tolerance.
                let err = (mean - 1.0).abs();
                if err > worst.0 {
                    worst = (err, format!("{name} @ view {view:?}: {albedo:?}"));
                }
            }
        }
        // The specification's smooth-metal configuration is too narrow for the
        // uniform grid above. Exercise the production evaluator with a balanced
        // GGX/cosine MIS estimator so both the sharp single-scatter peak and the
        // broad multiscatter compensation are measured by the same gate.
        for view in [[0.0_f32, 1.0, 0.0], [0.6, 0.8, 0.0], [0.95, 0.31, 0.0]] {
            let albedo = furnace_albedo_mis(metal(0.1), view, 32_768);
            let mean = (albedo[0] + albedo[1] + albedo[2]) / 3.0;
            let err = (mean - 1.0).abs();
            if err > worst.0 {
                worst = (
                    err,
                    format!("metal, smooth (roughness=0.1) @ view {view:?}: {albedo:?}"),
                );
            }
        }
        // The Subsurface class's own white furnace.
        {
            let sub = SurfaceSample {
                base_color: white,
                subsurface: 1.0,
                subsurface_color: white,
                thickness: 0.0,
                ..coupled(0.0)
            };
            let albedo = furnace_albedo_fn(
                crate::subsurface::evaluate_subsurface_direct,
                sub,
                [0.6, 0.8, 0.0],
                192,
                192,
            );
            let mean = (albedo[0] + albedo[1] + albedo[2]) / 3.0;
            let err = (mean - 1.0).abs();
            if err > worst.0 {
                worst = (err, format!("subsurface class @ mid: {albedo:?}"));
            }
        }
        // The Cloth class's own white furnace (its closure is separate).
        {
            let albedo = furnace_albedo_fn(
                crate::cloth::evaluate_cloth_direct,
                fuzzy,
                [0.6, 0.8, 0.0],
                192,
                192,
            );
            let mean = (albedo[0] + albedo[1] + albedo[2]) / 3.0;
            let err = (mean - 1.0).abs();
            if err > worst.0 {
                worst = (err, format!("cloth class (fuzz) @ mid: {albedo:?}"));
            }
        }
        assert!(
            worst.0 <= 0.02,
            "white furnace must conserve energy: {} (worst error {})",
            worst.1,
            worst.0
        );
    }

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
                    let k_rt =
                        crate::clearcoat::coat_internal_reflection_coefficient(eta_c, mu_o, r_b);
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
                                let m_ref =
                                    modulated_darkening_factor(coat_weight, coat_darkening, d_ref);
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
        assert!(
            worst_r <= 1.0e-6,
            "base roughness estimate deviates by {worst_r}"
        );
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
                        let rt = crate::clearcoat::coat_absorption_rgb(coat_color, mu_i, mu_o, eta);
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
            assert!(
                (normal[c] - color[c]).abs() < 1.0e-5,
                "{normal:?} vs {color:?}"
            );
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

    /// `OpenPBR` equations `specular_ior_ratio` + `specular_ior_ratio_with_tir_fix`:
    /// a coat shifts the base dielectric's surrounding index, and the real-time
    /// Clearcoat lobe must apply the TIR-fixed blend rather than the naive ratio.
    #[test]
    fn coat_specular_ior_ratio_matches_the_openpbr_reference() {
        let mut worst = 0.0_f32;
        for specular_ior in [1.0_f32, 1.2, 1.5, 2.0, 3.0] {
            for coat_ior in [1.0_f32, 1.3, 1.5, 1.6, 2.4] {
                for c in [0.0_f32, 0.25, 0.5, 0.75, 1.0] {
                    let rt = crate::clearcoat::coat_specular_ior_ratio(specular_ior, coat_ior, c);
                    let rf = coat_specular_ior_ratio(specular_ior, coat_ior, c);
                    worst = worst.max((rt - rf).abs());
                }
            }
        }
        assert!(worst <= 1.0e-6, "coat IOR ratio deviates by {worst}");
        // `C == 0` is the exact identity.
        for n in [1.0_f32, 1.5, 2.7] {
            assert_eq!(coat_specular_ior_ratio(n, 2.0, 0.0), n);
        }
        // The TIR fix: a denser coat inverts the ratio instead of implying a
        // rarer medium (`n_b/n_c < 1`). At `C == 1` the ratio is the plain
        // `n_c/n_b` in that case, never below 1.
        assert!((coat_specular_ior_ratio(1.5, 2.0, 1.0) - 2.0 / 1.5).abs() < 1e-6);
        // A rarer coat keeps the naive `n_b/n_c` (already `≥ 1`).
        assert!((coat_specular_ior_ratio(2.0, 1.5, 1.0) - 2.0 / 1.5).abs() < 1e-6);
        // The blend is a plain `lerp(n_b, η_bc_fixed, C)`, so it moves
        // monotonically from the authored index toward the fixed ratio in
        // *either* direction and never leaves the physical `[1, 3]` band.
        let target = coat_specular_ior_ratio(1.5, 1.9, 1.0);
        for i in 0..=10 {
            let c = i as f32 / 10.0;
            let v = coat_specular_ior_ratio(1.5, 1.9, c);
            let expected = 1.5 + (target - 1.5) * c;
            assert!(
                (v - expected).abs() < 1e-6 && (1.0..=3.0).contains(&v),
                "not a lerp at C={c}: {v} vs {expected}"
            );
        }
    }
}

/// Table generator (run manually with `--ignored --nocapture`); prints the
/// `Ê_ON(μ, σ)` grid baked into `eon_albedo_table.rs` and `eon_albedo.wesl`.
#[cfg(test)]
mod table_gen {
    #![allow(clippy::print_stdout, reason = "generator prints the baked table")]
    use super::*;

    #[test]
    #[ignore = "generator: prints the baked OpenPBR EON albedo table"]
    fn print_eon_table() {
        const N_MU: usize = 17;
        const N_SIGMA: usize = 9;
        println!("N_MU={N_MU} N_SIGMA={N_SIGMA}");
        for j in 0..N_SIGMA {
            let sigma = j as f32 / (N_SIGMA - 1) as f32 * core::f32::consts::FRAC_PI_2;
            let row: Vec<String> = (0..N_MU)
                .map(|i| {
                    let mu = i as f32 / (N_MU - 1) as f32;
                    format!("{:.6}", eon_directional_albedo(mu, sigma))
                })
                .collect();
            println!("{},", row.join(", "));
        }
        let avg: Vec<String> = (0..N_SIGMA)
            .map(|j| {
                let sigma = j as f32 / (N_SIGMA - 1) as f32 * core::f32::consts::FRAC_PI_2;
                format!("{:.6}", eon_average_albedo(sigma))
            })
            .collect();
        println!("AVG: {}", avg.join(", "));
    }
}

/// Generator for the GGX single-scatter albedo table (run with
/// `--ignored --nocapture`). Uses the repository's own GGX importance-sampled
/// bake path (`gi::env_brdf::multiscatter`) so the table is the *true* albedo of
/// the lobe the shader evaluates, not a fit.
#[cfg(test)]
mod albedo_table_gen {
    use super::*;

    #[test]
    #[ignore = "generator: prints the baked GGX single-scatter albedo table"]
    fn print_ggx_albedo_table() {
        use crate::gi::env_brdf::multiscatter::{average_albedo, directional_albedo};
        const N_MU: usize = 17;
        const N_R: usize = 9;
        const SAMPLES: u32 = 32768;
        println!("N_MU={N_MU} N_R={N_R}");
        for j in 0..N_R {
            let r = j as f32 / (N_R - 1) as f32;
            // `mu = 0` is the grazing ray, where VNDF importance sampling
            // degenerates (its `taken == 0` fallback reports a mirror, i.e. 1).
            // The shader floors `n·v` at `MIN_N_DOT`, so clamp the first column
            // onto the first resolved sample instead of publishing the artifact.
            let col1 = if r == 0.0 {
                1.0
            } else {
                directional_albedo(1.0 / (N_MU - 1) as f32, r, SAMPLES)
            };
            let row: Vec<String> = (0..N_MU)
                .map(|i| {
                    let mu = i as f32 / (N_MU - 1) as f32;
                    // A perfect mirror (`r == 0`) reflects everything at every
                    // angle; the VNDF estimator is degenerate there.
                    let e = if r == 0.0 {
                        1.0
                    } else if i == 0 {
                        col1
                    } else {
                        directional_albedo(mu, r, SAMPLES)
                    };
                    format!("{:.6}", e)
                })
                .collect();
            println!("{},", row.join(", "));
        }
        let avg: Vec<String> = (0..N_R)
            .map(|j| {
                let r = j as f32 / (N_R - 1) as f32;
                format!("{:.6}", average_albedo(r, 64, SAMPLES))
            })
            .collect();
        println!("AVG: {}", avg.join(", "));
    }
}

/// Generator for the *dielectric* specular lobe's directional albedo
/// `E_dielectric(µ, r)` — the quantity `OpenPBR`'s
/// `glossy_diffuse_albedo_constraint` weights the diffuse slab by. Baked with
/// VNDF importance sampling of the exact-Fresnel lobe at the specification's
/// default dielectric (`η = 1.5`), because a uniform hemisphere quadrature
/// cannot resolve a near-mirror GGX lobe.
#[cfg(test)]
mod dielectric_albedo_gen {
    use super::*;
    use crate::lighting::fresnel_dielectric;

    pub(super) fn vndf(wo: [f32; 3], alpha: f32, u1: f32, u2: f32) -> [f32; 3] {
        let vh = {
            let s = [alpha * wo[0], alpha * wo[1], wo[2]];
            let n = ops::sqrt(s[0] * s[0] + s[1] * s[1] + s[2] * s[2]);
            [s[0] / n, s[1] / n, s[2] / n]
        };
        let lensq = vh[0] * vh[0] + vh[1] * vh[1];
        let t1 = if lensq > 0.0 {
            let inv = 1.0 / ops::sqrt(lensq);
            [-vh[1] * inv, vh[0] * inv, 0.0]
        } else {
            [1.0, 0.0, 0.0]
        };
        let t2 = [
            vh[1] * t1[2] - vh[2] * t1[1],
            vh[2] * t1[0] - vh[0] * t1[2],
            vh[0] * t1[1] - vh[1] * t1[0],
        ];
        let r = ops::sqrt(u1);
        let phi = 2.0 * PI * u2;
        let p1 = r * ops::cos(phi);
        let s = 0.5 * (1.0 + vh[2]);
        let p2 = (1.0 - s) * ops::sqrt((1.0 - p1 * p1).max(0.0)) + s * (r * ops::sin(phi));
        let pz = ops::sqrt((1.0 - p1 * p1 - p2 * p2).max(0.0));
        let nh = [
            p1 * t1[0] + p2 * t2[0] + pz * vh[0],
            p1 * t1[1] + p2 * t2[1] + pz * vh[1],
            p1 * t1[2] + p2 * t2[2] + pz * vh[2],
        ];
        let s = [alpha * nh[0], alpha * nh[1], nh[2].max(1.0e-6)];
        let n = ops::sqrt(s[0] * s[0] + s[1] * s[1] + s[2] * s[2]);
        [s[0] / n, s[1] / n, s[2] / n]
    }

    /// `E_dielectric(µ, r)` at relative index `eta`, VNDF-importance-sampled.
    pub(super) fn bake_dielectric_albedo(mu_o: f32, perceptual_roughness: f32, eta: f32) -> f32 {
        const N: u32 = 65536;
        let alpha = perceptual_roughness * perceptual_roughness;
        let c = mu_o.clamp(1.0e-3, 1.0);
        let wo = [(1.0 - c * c).max(0.0).sqrt(), 0.0, c];
        let g1 = crate::glint::smith_g1(c, alpha).max(1.0e-6);
        let strata = 256u32;
        let mut acc = 0.0_f64;
        for s in 0..N {
            let i = s / strata;
            let j = s % strata;
            let u1 = (i as f32 + 0.5) / strata as f32;
            let u2 = (j as f32 + 0.5) / strata as f32;
            let h = vndf(wo, alpha, u1, u2);
            let v_dot_h = (wo[0] * h[0] + wo[1] * h[1] + wo[2] * h[2]).clamp(0.0, 1.0);
            let wi = [
                2.0 * v_dot_h * h[0] - wo[0],
                2.0 * v_dot_h * h[1] - wo[1],
                2.0 * v_dot_h * h[2] - wo[2],
            ];
            if wi[2] <= 0.0 {
                continue;
            }
            let g2 = crate::glint::smith_g2_height_correlated(c, wi[2], alpha);
            let weight = (g2 / g1) as f64;
            let f = fresnel_dielectric(v_dot_h, eta) as f64;
            acc += f * weight;
        }
        (acc / N as f64).clamp(0.0, 1.0) as f32
    }

    #[test]
    #[ignore = "generator: prints the baked dielectric specular albedo table"]
    fn print_dielectric_albedo_table() {
        const N_MU: usize = 17;
        const N_R: usize = 9;
        for j in 0..N_R {
            let r = j as f32 / (N_R - 1) as f32;
            let row: Vec<String> = (0..N_MU)
                .map(|i| {
                    let mu = i as f32 / (N_MU - 1) as f32;
                    format!("{:.6}", bake_dielectric_albedo(mu, r, 1.5))
                })
                .collect();
            println!("{},", row.join(", "));
        }
    }
}

/// Generator for `B(µ, r)`: the GGX lobe's albedo with the Schlick *bias* weight
/// `(1 − v·h)^5`. Together with the white-lobe albedo `E_white` it gives the
/// **exact** Schlick-Fresnel lobe albedo `F0·E_white + (1−F0)·B`, which the coat
/// energy closure needs (the hemispherical-average and `F(n·v)` estimates were
/// 2.3× low and 6% high respectively).
#[cfg(test)]
mod schlick_bias_gen {
    use super::*;

    /// The Schlick bias component of the split-sum DFG, from the repository's
    /// own validated VNDF estimator (`integrate_dfg`) — the *same* estimator the
    /// white-lobe albedo `E_white` comes from, so `F0·E_white + (1−F0)·B` is
    /// internally consistent. (A hand-rolled VNDF differed enough to leave a
    /// view-dependent ±8% error in the coat furnace.)
    fn bias_albedo(mu_o: f32, perceptual_roughness: f32) -> f32 {
        crate::gi::env_brdf::dfg_lut::integrate_dfg(mu_o, perceptual_roughness, 65536)
            .y
            .clamp(0.0, 1.0)
    }

    #[test]
    #[ignore = "generator: prints the baked Schlick-bias albedo table"]
    fn print_schlick_bias_table() {
        const N_MU: usize = 17;
        const N_R: usize = 9;
        for j in 0..N_R {
            let r = j as f32 / (N_R - 1) as f32;
            let row: Vec<String> = (0..N_MU)
                .map(|i| {
                    let mu = i as f32 / (N_MU - 1) as f32;
                    format!("{:.6}", if r == 0.0 { 0.0 } else { bias_albedo(mu, r) })
                })
                .collect();
            println!("{},", row.join(", "));
        }
    }
}

/// Generator for the Charlie (fuzz) lobe's directional albedo `E_fuzz(µ, r)`,
/// needed by the specification's `fuzz-layering-approx`
/// (`layer = F·f_fuzz + (1 − F·E_fuzz)·f_base`). Prism's fuzz lobe is Charlie +
/// Ashikhmin visibility (broad, so a direct 2-D quadrature is accurate).
#[cfg(test)]
mod charlie_albedo_gen {
    use super::*;

    fn charlie_albedo(mu_o: f32, roughness: f32) -> f32 {
        use crate::cloth::{distribution_charlie, visibility_ashikhmin};
        const N_THETA: usize = 256;
        const N_PHI: usize = 256;
        let cos_o = mu_o.clamp(1.0e-3, 1.0);
        let sin_o = (1.0 - cos_o * cos_o).max(0.0).sqrt();
        let d_theta = core::f32::consts::FRAC_PI_2 / N_THETA as f32;
        let d_phi = 2.0 * PI / N_PHI as f32;
        let mut sum = 0.0_f32;
        for i in 0..N_THETA {
            let theta = (i as f32 + 0.5) * d_theta;
            let (sin_i, cos_i) = (ops::sin(theta), ops::cos(theta));
            let n_dot_h = {
                let hy = cos_i + cos_o;
                let hx = sin_i * ops::cos((0.0f32).max(0.0)) + sin_o;
                let hz = sin_i * ops::sin(0.0);
                let len = ops::sqrt(hx * hx + hy * hy + hz * hz);
                (hy / len.max(1.0e-6)).clamp(0.0, 1.0)
            };
            for j in 0..N_PHI {
                let phi = (j as f32 + 0.5) * d_phi;
                // Half-vector between the fixed view and this light direction.
                let l = [sin_i * ops::cos(phi), cos_i, sin_i * ops::sin(phi)];
                let v = [sin_o, cos_o, 0.0];
                let s = [v[0] + l[0], v[1] + l[1], v[2] + l[2]];
                let len = ops::sqrt(s[0] * s[0] + s[1] * s[1] + s[2] * s[2]).max(1.0e-6);
                let n_dot_h = ((s[1] / len).clamp(0.0, 1.0)).max(1.0e-5);
                let _ = n_dot_h;
                let d = distribution_charlie(roughness.clamp(1.0e-2, 1.0), n_dot_h);
                let vis = visibility_ashikhmin(cos_o, cos_i.max(1.0e-5));
                sum += d * vis * cos_i * sin_i * d_theta * d_phi;
            }
        }
        sum
    }

    #[test]
    #[ignore = "generator: prints the baked Charlie (fuzz) albedo table"]
    fn print_charlie_albedo_table() {
        const N_MU: usize = 17;
        const N_R: usize = 9;
        for j in 0..N_R {
            let r = j as f32 / (N_R - 1) as f32;
            let row: Vec<String> = (0..N_MU)
                .map(|i| {
                    let mu = i as f32 / (N_MU - 1) as f32;
                    format!("{:.6}", charlie_albedo(mu, r))
                })
                .collect();
            println!("{},", row.join(", "));
        }
    }
}
