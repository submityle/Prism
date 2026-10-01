//! Thin-film iridescence (Belcour-Barla style Airy reflectance) — CPU golden.
//!
//! A thin dielectric film of thickness `d` sitting on a substrate reflects
//! light from two interfaces — the outer (incident medium → film) and the
//! inner (film → substrate).  The two reflected wave trains differ by an
//! optical phase that depends on the extra path length inside the film,
//!
//! ```text
//! phi = 4*pi * n_film * d * cos(theta_film) / lambda
//! ```
//!
//! and therefore interfere constructively or destructively as a function of
//! wavelength, thickness, and viewing angle.  That wavelength-dependent
//! interference is what tints soap bubbles, oil slicks, and anodised metals.
//!
//! Following Belcour & Barla's *A Practical Extension to Microfacet Theory for
//! the Modeling of Varying Iridescence* (SIGGRAPH 2017) we evaluate the
//! single-film **Airy** reflectance.  For real (lossless dielectric) Fresnel
//! amplitudes `r1` (outer interface) and `r2` (inner interface) the intensity
//! reflectance of the stack, per polarization, has the closed form
//!
//! ```text
//! R = (r1^2 + r2^2 + 2*r1*r2*cos(phi)) / (1 + r1^2*r2^2 + 2*r1*r2*cos(phi))
//! ```
//!
//! which is averaged over the s- and p-polarizations for unpolarised light.
//! The spectral response is reconstructed on three fixed representative
//! wavelengths (R = 630 nm, G = 532 nm, B = 465 nm) and packed straight into an
//! RGB [`Vec3`] with unit per-channel weights — the classic, allocation-free
//! three-wavelength approximation the GPU twin mirrors.
//!
//! A small anisotropic-GGX helper is included because iridescent coatings are
//! usually layered over a glossy base whose GI reflection lobe must be sized
//! consistently.  The Trowbridge-Reitz anisotropic NDF, the Disney roughness →
//! `(alpha_t, alpha_b)` remap, and the Smith `G1` masking term are all pure
//! closed forms.
//!
//! # Conventions
//! * `no_std`: math via `bevy_math`; transcendentals via `bevy_math::ops`
//!   (never `f32::exp`); square roots via the `f32::sqrt` method.
//! * All indices of refraction are clamped to be `>= 1` and all cosines to
//!   `[1e-4, 1]`, so grazing angles and degenerate media never divide by zero
//!   or produce `NaN`.
//! * Every reflectance is physically bounded and additionally clamped to
//!   `[0, 1]`; a film thickness of zero makes `phi = 0` and the Airy formula
//!   collapses *exactly* to the direct outer-medium → substrate Fresnel
//!   reflectance (proven in the tests).
//! * Every function is a deterministic, allocation-free pure function: no RNG,
//!   no I/O, no GPU, no global state.

use bevy_math::{ops, Vec3};
use core::f32::consts::PI;

/// Representative sampling wavelengths in nanometres for the R, G, and B
/// channels respectively (`630`, `532`, `465` nm).  These are the classic
/// primaries used by three-wavelength spectral-to-RGB iridescence shaders.
pub const RGB_WAVELENGTHS_NM: [f32; 3] = [630.0, 532.0, 465.0];

/// Smallest cosine of an incidence angle we evaluate, keeping grazing angles
/// away from the exact `cos = 0` singularity of the Fresnel denominators.
const MIN_COS: f32 = 1.0e-4;

/// Smallest index of refraction accepted; physical dielectrics have `n >= 1`.
const MIN_IOR: f32 = 1.0;

/// Smallest NDF denominator / roughness floor, keeping the GGX closed forms
/// well-conditioned for mirror-smooth inputs.
const MIN_ALPHA: f32 = 1.0e-4;

/// Clamps an index of refraction to the physical dielectric range `[1, inf)`.
#[inline]
fn clamp_ior(n: f32) -> f32 {
    if n.is_finite() {
        n.max(MIN_IOR)
    } else {
        MIN_IOR
    }
}

/// Clamps a cosine of an incidence angle to `[MIN_COS, 1]`.
#[inline]
fn clamp_cos(cos: f32) -> f32 {
    if cos.is_finite() {
        cos.clamp(MIN_COS, 1.0)
    } else {
        MIN_COS
    }
}

/// Cosine of the transmitted angle through an interface via Snell's law.
///
/// Given the incident cosine `cos_i` in a medium of index `n_i` crossing into a
/// medium of index `n_t`, returns `Some(cos_t)` with `cos_t >= 0`, or `None`
/// when the configuration is beyond the critical angle (total internal
/// reflection), in which case no real transmitted direction exists.
#[inline]
pub fn transmitted_cos(cos_i: f32, n_i: f32, n_t: f32) -> Option<f32> {
    let cos_i = clamp_cos(cos_i);
    let n_i = clamp_ior(n_i);
    let n_t = clamp_ior(n_t);
    let sin_i2 = (1.0 - cos_i * cos_i).max(0.0);
    let eta = n_i / n_t;
    let sin_t2 = eta * eta * sin_i2;
    if sin_t2 >= 1.0 {
        None
    } else {
        Some((1.0 - sin_t2).max(0.0).sqrt())
    }
}

/// Real Fresnel *amplitude* reflection coefficients `(r_s, r_p)` for a single
/// lossless dielectric interface.
///
/// `cos_i` / `cos_t` are the incident and transmitted cosines (both `>= 0`) and
/// `n_i` / `n_t` the two indices of refraction.  Returns the s- and
/// p-polarised amplitude coefficients
///
/// ```text
/// r_s = (n_i*cos_i - n_t*cos_t) / (n_i*cos_i + n_t*cos_t)
/// r_p = (n_t*cos_i - n_i*cos_t) / (n_t*cos_i + n_i*cos_t)
/// ```
///
/// Each lies in `[-1, 1]`; the sign carries the half-wave phase flip at the
/// interface, which the Airy recombination needs.
#[inline]
fn fresnel_amplitudes(cos_i: f32, cos_t: f32, n_i: f32, n_t: f32) -> (f32, f32) {
    let a = n_i * cos_i;
    let b = n_t * cos_t;
    let c = n_t * cos_i;
    let d = n_i * cos_t;
    let r_s = (a - b) / (a + b);
    let r_p = (c - d) / (c + d);
    (r_s, r_p)
}

/// Unpolarised intensity reflectance of a single dielectric interface.
///
/// Returns the average of the s- and p-polarised power reflectances
/// `(r_s^2 + r_p^2) / 2` for light crossing from medium `n_i` into medium
/// `n_t` at incident cosine `cos_i`.  Beyond the critical angle (total internal
/// reflection) the interface reflects everything and the result is `1`.
///
/// This is the physical reference the thin-film Airy reflectance must collapse
/// to when the film thickness is zero.
#[inline]
pub fn fresnel_dielectric_unpolarized(cos_i: f32, n_i: f32, n_t: f32) -> f32 {
    let cos_i = clamp_cos(cos_i);
    let n_i = clamp_ior(n_i);
    let n_t = clamp_ior(n_t);
    match transmitted_cos(cos_i, n_i, n_t) {
        None => 1.0,
        Some(cos_t) => {
            let (r_s, r_p) = fresnel_amplitudes(cos_i, cos_t, n_i, n_t);
            (0.5 * (r_s * r_s + r_p * r_p)).clamp(0.0, 1.0)
        }
    }
}

/// Accumulated optical phase difference `phi` across the film, in radians.
///
/// ```text
/// phi = 4*pi * n_film * d * cos(theta_film) / lambda
/// ```
///
/// where `cos_film` is the cosine of the propagation angle *inside* the film,
/// `thickness_nm` is the geometric film thickness `d`, and `wavelength_nm` is
/// the vacuum wavelength.  The phase is returned unwrapped (monotonically
/// increasing in thickness), which the tests rely on to verify interference
/// advances steadily with `d`.
#[inline]
pub fn optical_phase(film_ior: f32, cos_film: f32, thickness_nm: f32, wavelength_nm: f32) -> f32 {
    let film_ior = clamp_ior(film_ior);
    let cos_film = cos_film.clamp(0.0, 1.0);
    let d = thickness_nm.max(0.0);
    let lambda = wavelength_nm.max(1.0);
    4.0 * PI * film_ior * d * cos_film / lambda
}

/// Single-film Airy intensity reflectance at one wavelength.
///
/// The stack is: incident medium (`outer_ior`, usually air) → film
/// (`film_ior`, thickness `thickness_nm`) → substrate (`base_ior`), observed at
/// incident cosine `cos_outer`.  Returns the unpolarised reflectance in
/// `[0, 1]` for the given vacuum `wavelength_nm`.
///
/// Total internal reflection at either interface — which can happen when the
/// film or substrate index is lower than the medium above it — returns `1`
/// (the stack is a perfect mirror at that angle).  At `thickness_nm == 0` the
/// phase vanishes and the result equals
/// [`fresnel_dielectric_unpolarized(cos_outer, outer_ior, base_ior)`].
#[inline]
pub fn airy_reflectance(
    outer_ior: f32,
    film_ior: f32,
    base_ior: f32,
    cos_outer: f32,
    thickness_nm: f32,
    wavelength_nm: f32,
) -> f32 {
    let n0 = clamp_ior(outer_ior);
    let n1 = clamp_ior(film_ior);
    let n2 = clamp_ior(base_ior);
    let cos0 = clamp_cos(cos_outer);

    // Angle inside the film (outer -> film). TIR here is a perfect mirror.
    let cos1 = match transmitted_cos(cos0, n0, n1) {
        Some(c) => c,
        None => return 1.0,
    };
    // Angle inside the substrate (film -> base). TIR here is a perfect mirror.
    let cos2 = match transmitted_cos(cos1, n1, n2) {
        Some(c) => c,
        None => return 1.0,
    };

    let (r01_s, r01_p) = fresnel_amplitudes(cos0, cos1, n0, n1);
    let (r12_s, r12_p) = fresnel_amplitudes(cos1, cos2, n1, n2);

    let phi = optical_phase(n1, cos1, thickness_nm, wavelength_nm);
    let cos_phi = ops::cos(phi);

    let airy = |r1: f32, r2: f32| -> f32 {
        let num = r1 * r1 + r2 * r2 + 2.0 * r1 * r2 * cos_phi;
        let den = 1.0 + r1 * r1 * r2 * r2 + 2.0 * r1 * r2 * cos_phi;
        // `den` is bounded away from zero for |r1|,|r2| < 1, but clamp anyway.
        if den.abs() > MIN_ALPHA {
            (num / den).clamp(0.0, 1.0)
        } else {
            1.0
        }
    };

    0.5 * (airy(r01_s, r12_s) + airy(r01_p, r12_p))
}

/// Iridescent RGB reflectance of a thin film, sampled on the three fixed
/// [`RGB_WAVELENGTHS_NM`] primaries.
///
/// Evaluates [`airy_reflectance`] at 630 / 532 / 465 nm and packs the three
/// unpolarised reflectances into an RGB [`Vec3`] with unit per-channel weights
/// — the standard three-wavelength spectral approximation.  Each channel is in
/// `[0, 1]`, so the returned colour is a valid, energy-conserving reflectance.
///
/// With `thickness_nm == 0` all three channels equal the achromatic base
/// Fresnel reflectance, so the film has no tint — exactly the uncoated
/// substrate.
#[inline]
pub fn iridescent_reflectance_rgb(
    outer_ior: f32,
    film_ior: f32,
    base_ior: f32,
    cos_outer: f32,
    thickness_nm: f32,
) -> Vec3 {
    Vec3::new(
        airy_reflectance(
            outer_ior,
            film_ior,
            base_ior,
            cos_outer,
            thickness_nm,
            RGB_WAVELENGTHS_NM[0],
        ),
        airy_reflectance(
            outer_ior,
            film_ior,
            base_ior,
            cos_outer,
            thickness_nm,
            RGB_WAVELENGTHS_NM[1],
        ),
        airy_reflectance(
            outer_ior,
            film_ior,
            base_ior,
            cos_outer,
            thickness_nm,
            RGB_WAVELENGTHS_NM[2],
        ),
    )
}

/// Disney anisotropic roughness remap to the GGX `(alpha_t, alpha_b)` pair.
///
/// Given a perceptual `roughness` in `[0, 1]` and `anisotropy` in `[-1, 1]`
/// (0 = isotropic, positive = stretched along the tangent), returns the tangent
/// and bitangent GGX widths following Burley's convention:
///
/// ```text
/// aspect  = sqrt(1 - 0.9 * anisotropy)
/// alpha_t = roughness^2 / aspect
/// alpha_b = roughness^2 * aspect
/// ```
///
/// Both outputs are clamped to be `>= MIN_ALPHA` so the NDF stays finite for
/// mirror-smooth surfaces.
#[inline]
pub fn anisotropic_alphas(roughness: f32, anisotropy: f32) -> (f32, f32) {
    let r = roughness.clamp(0.0, 1.0);
    let a = anisotropy.clamp(-1.0, 1.0);
    let alpha = r * r;
    let aspect = (1.0 - 0.9 * a).max(MIN_ALPHA).sqrt();
    let alpha_t = (alpha / aspect).max(MIN_ALPHA);
    let alpha_b = (alpha * aspect).max(MIN_ALPHA);
    (alpha_t, alpha_b)
}

/// Normalises `v`, returning `fallback` for a degenerate (near-zero) input.
#[inline]
fn normalize_or(v: Vec3, fallback: Vec3) -> Vec3 {
    let len_sq = v.length_squared();
    if len_sq > f32::MIN_POSITIVE {
        v * len_sq.sqrt().recip()
    } else {
        fallback
    }
}

/// Anisotropic Trowbridge-Reitz (GGX) normal-distribution function.
///
/// Evaluates the microfacet NDF for a half-vector `half` in the local frame
/// spanned by unit `tangent`, `bitangent`, and `normal`, with GGX widths
/// `alpha_t` and `alpha_b`:
///
/// ```text
/// D = 1 / (pi * at * ab * ((ht/at)^2 + (hb/ab)^2 + hn^2)^2)
/// ```
///
/// where `ht`, `hb`, `hn` are the projections of the (normalised) half-vector
/// onto the tangent, bitangent, and normal.  Returns `0` for half-vectors in or
/// below the tangent plane (`hn <= 0`); otherwise a non-negative density whose
/// peak at `hn = 1` is `1 / (pi * at * ab)`.
#[inline]
pub fn ggx_aniso_ndf(
    half: Vec3,
    tangent: Vec3,
    bitangent: Vec3,
    normal: Vec3,
    alpha_t: f32,
    alpha_b: f32,
) -> f32 {
    let n = normalize_or(normal, Vec3::Z);
    let t = normalize_or(tangent, Vec3::X);
    let b = normalize_or(bitangent, Vec3::Y);
    let h = normalize_or(half, n);
    let at = alpha_t.max(MIN_ALPHA);
    let ab = alpha_b.max(MIN_ALPHA);

    let hn = h.dot(n);
    if hn <= 0.0 {
        return 0.0;
    }
    let ht = h.dot(t);
    let hb = h.dot(b);

    let a = ht / at;
    let c = hb / ab;
    let inner = a * a + c * c + hn * hn;
    let denom = PI * at * ab * inner * inner;
    if denom > f32::MIN_POSITIVE {
        (1.0 / denom).max(0.0)
    } else {
        0.0
    }
}

/// Smith `G1` masking-shadowing term for the anisotropic GGX distribution.
///
/// Returns the fraction of microfacets visible along direction `v` (view or
/// light) in the `(tangent, bitangent, normal)` frame, using Heitz's `Lambda`:
///
/// ```text
/// Lambda = (-1 + sqrt(1 + (at^2*vt^2 + ab^2*vb^2) / vn^2)) / 2
/// G1     = 1 / (1 + Lambda)
/// ```
///
/// The result is in `(0, 1]`, tends to `1` for a mirror-smooth surface or a
/// head-on direction, and falls off at grazing angles.  Directions in or below
/// the tangent plane (`vn <= 0`) are fully masked and return `0`.
#[inline]
pub fn smith_g1_aniso(
    v: Vec3,
    tangent: Vec3,
    bitangent: Vec3,
    normal: Vec3,
    alpha_t: f32,
    alpha_b: f32,
) -> f32 {
    let n = normalize_or(normal, Vec3::Z);
    let t = normalize_or(tangent, Vec3::X);
    let b = normalize_or(bitangent, Vec3::Y);
    let dir = normalize_or(v, n);
    let at = alpha_t.max(MIN_ALPHA);
    let ab = alpha_b.max(MIN_ALPHA);

    let vn = dir.dot(n);
    if vn <= 0.0 {
        return 0.0;
    }
    let vt = dir.dot(t);
    let vb = dir.dot(b);
    let ratio = (at * at * vt * vt + ab * ab * vb * vb) / (vn * vn);
    let lambda = 0.5 * (-1.0 + (1.0 + ratio).max(0.0).sqrt());
    (1.0 / (1.0 + lambda)).clamp(0.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f32 = 1.0e-5;

    #[test]
    fn zero_thickness_reduces_to_base_fresnel() {
        // Air -> film (1.5) -> glass (1.7). At d = 0 the Airy stack must equal
        // the direct air -> glass Fresnel reflectance at every angle and
        // wavelength (the film is optically absent).
        for &cos in &[1.0, 0.9, 0.6, 0.3, 0.05] {
            let base = fresnel_dielectric_unpolarized(cos, 1.0, 1.7);
            for &lambda in &RGB_WAVELENGTHS_NM {
                let airy = airy_reflectance(1.0, 1.5, 1.7, cos, 0.0, lambda);
                assert!(
                    (airy - base).abs() < 1.0e-4,
                    "cos={cos} lambda={lambda}: airy={airy} base={base}"
                );
            }
        }
    }

    #[test]
    fn zero_thickness_rgb_is_achromatic() {
        // No film => all three channels equal the base Fresnel => no tint.
        let rgb = iridescent_reflectance_rgb(1.0, 1.4, 2.0, 0.8, 0.0);
        assert!((rgb.x - rgb.y).abs() < 1.0e-4);
        assert!((rgb.y - rgb.z).abs() < 1.0e-4);
    }

    #[test]
    fn reflectance_is_energy_conserving() {
        // Reflectance must be non-negative and never exceed one, for a wide
        // sweep of thickness, angle, and wavelength.
        for ti in 0..20 {
            let d = ti as f32 * 80.0; // 0..1520 nm
            for ci in 1..=10 {
                let cos = ci as f32 / 10.0;
                for &lambda in &RGB_WAVELENGTHS_NM {
                    let r = airy_reflectance(1.0, 1.4, 1.9, cos, d, lambda);
                    assert!(r.is_finite() && (0.0..=1.0).contains(&r), "r={r}");
                }
            }
        }
    }

    #[test]
    fn rgb_channels_are_bounded() {
        let rgb = iridescent_reflectance_rgb(1.0, 1.33, 1.5, 0.7, 380.0);
        for c in [rgb.x, rgb.y, rgb.z] {
            assert!(c.is_finite() && (0.0..=1.0).contains(&c), "channel={c}");
        }
    }

    #[test]
    fn phase_advances_monotonically_with_thickness() {
        let mut prev = f32::NEG_INFINITY;
        for ti in 0..=30 {
            let d = ti as f32 * 25.0;
            let phi = optical_phase(1.5, 0.9, d, 532.0);
            assert!(phi.is_finite());
            assert!(phi > prev - EPS, "non-monotonic: {prev} -> {phi}");
            prev = phi;
        }
        // The first thickness is zero -> zero phase.
        assert_eq!(optical_phase(1.5, 0.9, 0.0, 532.0), 0.0);
    }

    #[test]
    fn phase_scales_with_thickness_linearly() {
        // Doubling thickness doubles the phase (fixed n, cos, lambda).
        let a = optical_phase(1.5, 0.8, 100.0, 500.0);
        let b = optical_phase(1.5, 0.8, 200.0, 500.0);
        assert!((b - 2.0 * a).abs() < 1.0e-3, "a={a} b={b}");
    }

    #[test]
    fn interference_oscillates_in_thickness() {
        // Between a constructive and a destructive thickness the reflectance
        // must actually change (the film is iridescent, not constant).
        let r_a = airy_reflectance(1.0, 1.5, 1.0, 1.0, 0.0, 550.0);
        let r_b = airy_reflectance(1.0, 1.5, 1.0, 1.0, 140.0, 550.0);
        assert!((r_a - r_b).abs() > 1.0e-3, "no interference: {r_a} vs {r_b}");
    }

    #[test]
    fn total_internal_reflection_is_mirror() {
        // Dense film/medium (n0=1.6) over a thin low-index film (n1=1.0):
        // beyond the critical angle the outer interface reflects everything.
        // critical cos: sin_c = n1/n0 = 0.625 -> cos_c = 0.78; use cos=0.2.
        let r = airy_reflectance(1.6, 1.0, 1.5, 0.2, 300.0, 550.0);
        assert_eq!(r, 1.0);
    }

    #[test]
    fn fresnel_normal_incidence_matches_schlick_f0() {
        // At normal incidence the dielectric reflectance equals the F0 formula.
        let r = fresnel_dielectric_unpolarized(1.0, 1.0, 1.5);
        let d = (1.5_f32 - 1.0) / (1.5 + 1.0);
        let f0 = d * d;
        assert!((r - f0).abs() < 1.0e-5, "r={r} f0={f0}");
    }

    #[test]
    fn transmitted_cos_tir_detection() {
        // Water (1.33) -> air (1.0): critical angle ~ 48.75 deg (cos ~ 0.659).
        assert!(transmitted_cos(0.9, 1.33, 1.0).is_some()); // steep, transmits
        assert!(transmitted_cos(0.3, 1.33, 1.0).is_none()); // grazing, TIR
    }

    #[test]
    fn anisotropic_alphas_isotropic_case() {
        let (at, ab) = anisotropic_alphas(0.5, 0.0);
        assert!((at - ab).abs() < 1.0e-6, "at={at} ab={ab}");
        assert!((at - 0.25).abs() < 1.0e-5, "alpha should be roughness^2");
    }

    #[test]
    fn anisotropic_alphas_stretch_ordering() {
        // Positive anisotropy stretches the tangent lobe: alpha_t > alpha_b.
        let (at, ab) = anisotropic_alphas(0.6, 0.8);
        assert!(at > ab, "at={at} ab={ab}");
        // Both stay positive and finite.
        assert!(at.is_finite() && ab.is_finite() && at > 0.0 && ab > 0.0);
    }

    #[test]
    fn ggx_ndf_peaks_at_normal() {
        let (at, ab) = anisotropic_alphas(0.4, 0.0);
        let n = Vec3::Z;
        let t = Vec3::X;
        let b = Vec3::Y;
        let peak = ggx_aniso_ndf(n, t, b, n, at, ab);
        let expected = 1.0 / (PI * at * ab);
        assert!((peak - expected).abs() < 1.0e-3, "peak={peak} exp={expected}");
        // An off-axis half-vector is strictly dimmer than the peak.
        let off = normalize_or(Vec3::new(0.4, 0.0, 1.0), Vec3::Z);
        let d_off = ggx_aniso_ndf(off, t, b, n, at, ab);
        assert!(d_off < peak, "off={d_off} peak={peak}");
    }

    #[test]
    fn ggx_ndf_below_horizon_is_zero() {
        let (at, ab) = anisotropic_alphas(0.4, 0.0);
        let d = ggx_aniso_ndf(Vec3::NEG_Z, Vec3::X, Vec3::Y, Vec3::Z, at, ab);
        assert_eq!(d, 0.0);
    }

    #[test]
    fn ggx_ndf_is_nonnegative_and_finite_everywhere() {
        let (at, ab) = anisotropic_alphas(0.3, 0.5);
        for xi in -3..=3 {
            for zi in -3..=3 {
                let h = Vec3::new(xi as f32, 0.5, zi as f32);
                let d = ggx_aniso_ndf(h, Vec3::X, Vec3::Y, Vec3::Z, at, ab);
                assert!(d.is_finite() && d >= 0.0, "d={d}");
            }
        }
    }

    #[test]
    fn smith_g1_bounds_and_limits() {
        // Head-on view with a smooth surface is (almost) fully visible.
        let g_head = smith_g1_aniso(Vec3::Z, Vec3::X, Vec3::Y, Vec3::Z, 0.01, 0.01);
        assert!(g_head > 0.99 && g_head <= 1.0, "g_head={g_head}");
        // Below-horizon direction is fully masked.
        let g_down = smith_g1_aniso(Vec3::NEG_Z, Vec3::X, Vec3::Y, Vec3::Z, 0.3, 0.3);
        assert_eq!(g_down, 0.0);
        // Grazing on a rough surface is partially masked (strictly < 1).
        let graze = normalize_or(Vec3::new(1.0, 0.0, 0.15), Vec3::Z);
        let g_graze = smith_g1_aniso(graze, Vec3::X, Vec3::Y, Vec3::Z, 0.6, 0.6);
        assert!(g_graze.is_finite() && g_graze > 0.0 && g_graze < 1.0, "g={g_graze}");
    }

    #[test]
    fn is_deterministic() {
        let a = airy_reflectance(1.0, 1.45, 1.8, 0.73, 312.0, 532.0);
        let b = airy_reflectance(1.0, 1.45, 1.8, 0.73, 312.0, 532.0);
        assert_eq!(a, b);
        let p = iridescent_reflectance_rgb(1.0, 1.45, 1.8, 0.73, 312.0);
        let q = iridescent_reflectance_rgb(1.0, 1.45, 1.8, 0.73, 312.0);
        assert_eq!(p, q);
    }

    #[test]
    fn degenerate_inputs_are_clamped() {
        // Zero/negative IOR, NaN cosine, negative thickness must not NaN.
        let r = airy_reflectance(0.0, -1.0, f32::NAN, f32::NAN, -50.0, 0.0);
        assert!(r.is_finite() && (0.0..=1.0).contains(&r), "r={r}");
        let rgb = iridescent_reflectance_rgb(f32::NAN, 0.0, 0.0, 2.0, -1.0);
        for c in [rgb.x, rgb.y, rgb.z] {
            assert!(c.is_finite() && (0.0..=1.0).contains(&c), "c={c}");
        }
    }
}
