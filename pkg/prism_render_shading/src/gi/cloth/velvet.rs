//! Ashikhmin-Shirley velvet lobe: the inverted-Gaussian NDF.
//!
//! Golden CPU reference for the classic velvet / microfiber sheen used before
//! (and alongside) the Charlie model in [`super::charlie`]. Velvet fibres stand
//! roughly normal to the surface, so the microfacet distribution does the
//! opposite of a glossy GGX lobe: it is *suppressed* around the surface normal
//! and *amplified* toward grazing half angles, producing the bright rim seen on
//! velvet, suede, and brushed fabric.
//!
//! The distribution used here is the Ashikhmin inverted Gaussian popularised by
//! Neubelt & Pettineo (*The Order: 1886*):
//!
//! ```text
//! D(θ_h) = (1 / (π·(1 + 4r))) · (1 + 4r · exp(−cot²θ_h / r) / sin⁴θ_h)
//! ```
//!
//! with `r = roughness`, `cot²θ_h = cos²θ_h / sin²θ_h`, and `cosθ_h = n·h`. The
//! `exp(−cot²/r)/sin⁴` ring term is what brightens grazing angles; it decays to
//! zero at the normal faster than `sin⁴θ_h` vanishes, so the distribution stays
//! finite there and reduces to the constant floor `1/(π·(1 + 4r))`.
//!
//! The lobe reuses the Ashikhmin "no closed-form" visibility
//! [`super::charlie::v_neubelt`] rather than re-deriving it, keeping the two
//! fabric lobes consistent and this file focused on the distribution.
//!
//! # Conventions
//! * Local shading frame with the normal at `+Z`; a direction's cosine with the
//!   normal is its `z` component. `wo`/`wi` point away from the surface.
//! * `roughness ∈ [0, 1]`, floored at [`MIN_ROUGHNESS`]; it is used directly
//!   (perceptually linear), not squared.
//! * Transcendental maths goes through [`bevy_math::ops`]; `sqrt` uses the
//!   inherent method. Every routine is a deterministic pure function (no RNG,
//!   I/O, GPU, globals, or `unsafe`) and clamps defensively so it never returns
//!   `NaN` or infinity.
//!
//! # References
//! * Ashikhmin, Premože & Shirley 2000, *A Microfacet-Based BRDF Generator*.
//! * Neubelt & Pettineo 2013, *Crafting a Next-Gen Material Pipeline for
//!   The Order: 1886*.

use bevy_math::{Vec3, ops};
use core::f32::consts::FRAC_1_PI;

use super::charlie::v_neubelt;

/// Smallest velvet roughness, so the `1/r` and `cot²/r` terms stay finite.
pub const MIN_ROUGHNESS: f32 = 1.0e-3;

/// `sin²θ_h` below this is treated as the normal-incidence limit (the ring term
/// vanishes), avoiding the `0/0` in `exp(−cot²/r)/sin⁴θ_h`.
pub const MIN_SIN2: f32 = 1.0e-6;

/// Ashikhmin inverted-Gaussian velvet normal-distribution function.
///
/// `D(θ_h) = (1/(π·(1 + 4r)))·(1 + 4r·exp(−cot²θ_h/r)/sin⁴θ_h)` with
/// `cosθ_h = n·h` and `r = roughness` floored at [`MIN_ROUGHNESS`]. Returns `0`
/// for a back-facing half vector (`µ_h ≤ 0`). At grazing half angles the ring
/// term approaches `4r`, so `D → 1/π`; near the normal the ring term decays to
/// `0` and `D → 1/(π·(1 + 4r))`. The value is always non-negative and finite.
#[inline]
pub fn velvet_ndf(n_dot_h: f32, roughness: f32) -> f32 {
    if !n_dot_h.is_finite() {
        return 0.0;
    }
    let cos_h = n_dot_h.clamp(-1.0, 1.0);
    if cos_h <= 0.0 {
        return 0.0;
    }
    let r = roughness.clamp(MIN_ROUGHNESS, 1.0);
    let cos2 = cos_h * cos_h;
    let sin2 = (1.0 - cos2).max(0.0);
    let base = FRAC_1_PI / (1.0 + 4.0 * r);
    // Near the normal the ring term is a 0/0 whose true limit is 0; skip it.
    if sin2 <= MIN_SIN2 {
        return base;
    }
    let cot2 = cos2 / sin2;
    let sin4 = sin2 * sin2;
    let ring = 4.0 * r * ops::exp(-cot2 / r) / sin4;
    let d = base * (1.0 + ring);
    if d.is_finite() {
        d.max(0.0)
    } else {
        0.0
    }
}

/// Ashikhmin "no closed-form" visibility, shared with the Charlie lobe.
///
/// Thin re-export of [`super::charlie::v_neubelt`]:
/// `V = 1 / (4·(µ_l + µ_v − µ_l·µ_v))`. Returns `0` below the horizon.
#[inline]
pub fn velvet_visibility(n_dot_v: f32, n_dot_l: f32) -> f32 {
    v_neubelt(n_dot_v, n_dot_l)
}

/// Colourless velvet sheen BRDF value `D·V` for local directions `wo`, `wi`.
///
/// Builds the half vector `h = normalize(wo + wi)`, evaluates [`velvet_ndf`] at
/// `µ_h = h.z`, and multiplies by [`velvet_visibility`]. The velvet tint and the
/// `n·l` cosine are applied by the caller. Returns `0` for below-horizon
/// directions or a degenerate half vector.
#[inline]
pub fn velvet_brdf(wo: Vec3, wi: Vec3, roughness: f32) -> f32 {
    if wo.z <= 0.0 || wi.z <= 0.0 {
        return 0.0;
    }
    let h = (wo + wi).normalize_or_zero();
    if h.length_squared() <= 0.0 {
        return 0.0;
    }
    let d = velvet_ndf(h.z, roughness);
    let v = velvet_visibility(wo.z, wi.z);
    let f = d * v;
    if f.is_finite() {
        f.max(0.0)
    } else {
        0.0
    }
}

/// Colourless velvet sheen lobe `D·V` from cosines (frame-independent).
///
/// Cosine-domain companion to [`velvet_brdf`]: evaluates [`velvet_ndf`] at the
/// supplied `µ_h = n·h` and multiplies by [`velvet_visibility`]. Preferred when
/// the caller already has the half-vector cosine (e.g. the combined cloth BRDF
/// in [`super::fabric`]). Returns `0` below the horizon or for `µ_h ≤ 0`.
#[inline]
pub fn velvet_lobe(n_dot_v: f32, n_dot_l: f32, n_dot_h: f32, roughness: f32) -> f32 {
    let f = velvet_ndf(n_dot_h, roughness) * velvet_visibility(n_dot_v, n_dot_l);
    if f.is_finite() {
        f.max(0.0)
    } else {
        0.0
    }
}

/// Grazing-rim weight `(1 − µ_v)^p` used to tint velvet toward the silhouette.
///
/// A small helper for layering the velvet lobe: it returns a smooth
/// `[0, 1]` falloff that is `0` head-on (`µ_v = 1`) and `1` at the grazing
/// silhouette (`µ_v = 0`). `power` is floored at `1` and the cosine clamped to
/// `[0, 1]`. Purely a convenience; the core lobe does not depend on it.
#[inline]
pub fn grazing_rim(n_dot_v: f32, power: f32) -> f32 {
    let mu = n_dot_v.clamp(0.0, 1.0);
    let p = power.max(1.0);
    let w = ops::powf((1.0 - mu).max(0.0), p);
    if w.is_finite() {
        w.clamp(0.0, 1.0)
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::f32::consts::{FRAC_PI_2, TAU};

    /// The inverted Gaussian is brighter toward grazing half angles than at the
    /// normal — the defining velvet behaviour.
    #[test]
    fn velvet_ndf_grazing_brighter_than_normal() {
        for &r in &[0.2f32, 0.5, 0.9] {
            let near_grazing = velvet_ndf(0.02, r);
            let near_normal = velvet_ndf(0.999, r);
            assert!(
                near_grazing > near_normal,
                "grazing={near_grazing} normal={near_normal} r={r}"
            );
        }
    }

    /// Approaching the normal the distribution relaxes to the constant floor
    /// `1/(π(1 + 4r))` (the ring term vanishes), confirming normal attenuation.
    #[test]
    fn velvet_ndf_normal_floor() {
        for &r in &[0.3f32, 0.7, 1.0] {
            let floor = FRAC_1_PI / (1.0 + 4.0 * r);
            let at_normal = velvet_ndf(1.0, r);
            assert!(
                (at_normal - floor).abs() < 1.0e-6,
                "at_normal={at_normal} floor={floor} r={r}"
            );
            // Just off the normal must be very close to the same floor.
            let near = velvet_ndf(0.9995, r);
            assert!((near - floor).abs() < 1.0e-2, "near={near} floor={floor}");
        }
    }

    #[test]
    fn velvet_ndf_backface_is_zero() {
        assert_eq!(velvet_ndf(0.0, 0.5), 0.0);
        assert_eq!(velvet_ndf(-0.3, 0.5), 0.0);
    }

    #[test]
    fn velvet_ndf_non_negative_and_finite() {
        for &r in &[0.05f32, 0.4, 1.0] {
            for i in 0..=50 {
                let c = i as f32 / 50.0;
                let d = velvet_ndf(c, r);
                assert!(d.is_finite() && d >= 0.0, "D={d} c={c} r={r}");
            }
        }
    }

    #[test]
    fn velvet_brdf_reciprocal() {
        let a = Vec3::new(0.2, 0.1, 0.974).normalize();
        let b = Vec3::new(-0.3, 0.25, 0.92).normalize();
        let fab = velvet_brdf(a, b, 0.4);
        let fba = velvet_brdf(b, a, 0.4);
        assert!((fab - fba).abs() < 1.0e-6, "fab={fab} fba={fba}");
    }

    #[test]
    fn velvet_brdf_below_horizon_is_zero() {
        let up = Vec3::new(0.1, 0.0, 0.995).normalize();
        let down = Vec3::new(0.1, 0.0, -0.995).normalize();
        assert_eq!(velvet_brdf(down, up, 0.5), 0.0);
        assert_eq!(velvet_brdf(up, down, 0.5), 0.0);
    }

    #[test]
    fn velvet_brdf_positive_and_finite() {
        let wo = Vec3::new(0.3, 0.0, 0.954).normalize();
        for i in 1..20 {
            let a = i as f32 / 20.0 * FRAC_PI_2;
            let (s, c) = ops::sin_cos(a);
            let wi = Vec3::new(s, 0.0, c);
            let f = velvet_brdf(wo, wi, 0.5);
            assert!(f.is_finite() && f >= 0.0, "f={f}");
        }
    }

    /// The cosine-domain lobe agrees with the vector-domain BRDF when fed the
    /// matching half-vector cosine.
    #[test]
    fn velvet_lobe_matches_brdf() {
        let wo = Vec3::new(0.25, 0.1, 0.962).normalize();
        let wi = Vec3::new(-0.2, 0.3, 0.933).normalize();
        let h = (wo + wi).normalize();
        let r = 0.45;
        let via_vec = velvet_brdf(wo, wi, r);
        let via_cos = velvet_lobe(wo.z, wi.z, h.z, r);
        assert!((via_vec - via_cos).abs() < 1.0e-6, "vec={via_vec} cos={via_cos}");
    }

    /// The cosine-weighted hemispherical integral of the NDF is finite and
    /// strictly positive (velvet is not normalised, but must stay well behaved).
    #[test]
    fn velvet_ndf_hemisphere_integral_finite_positive() {
        for &r in &[0.25f32, 0.6, 1.0] {
            let n = 100_000usize;
            let mut acc = 0.0f64;
            for i in 0..n {
                let cos_t = (i as f32 + 0.5) / n as f32;
                acc += (velvet_ndf(cos_t, r) * cos_t) as f64;
            }
            let integral = (acc / n as f64 * TAU as f64) as f32;
            assert!(
                integral.is_finite() && integral > 0.0,
                "integral={integral} r={r}"
            );
        }
    }

    #[test]
    fn grazing_rim_endpoints() {
        assert!((grazing_rim(1.0, 4.0) - 0.0).abs() < 1.0e-7);
        assert!((grazing_rim(0.0, 4.0) - 1.0).abs() < 1.0e-7);
        // Monotonic decrease with the cosine.
        assert!(grazing_rim(0.2, 3.0) > grazing_rim(0.8, 3.0));
    }

    #[test]
    fn degenerate_inputs_never_nan() {
        assert_eq!(velvet_ndf(f32::NAN, 0.5), 0.0);
        assert!(velvet_ndf(1.0, 0.0).is_finite());
        assert_eq!(velvet_brdf(Vec3::Z, Vec3::NEG_Z, 0.5), 0.0);
        assert_eq!(velvet_brdf(Vec3::ZERO, Vec3::ZERO, 0.5), 0.0);
        assert!(grazing_rim(f32::NAN, 0.0).is_finite());
    }
}
