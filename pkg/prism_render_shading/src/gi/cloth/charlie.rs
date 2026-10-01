//! Estevez-Kulla "Charlie" sheen lobe: distribution and visibility.
//!
//! Golden CPU reference for the retro-reflective cloth sheen highlight. This
//! file evaluates the *actual* sheen BRDF lobe `D·V` (the microfacet normal
//! distribution together with its masking-shadowing term); it is deliberately
//! distinct from [`crate::gi::env_brdf::sheen_clearcoat`], which only bakes the
//! pre-integrated split-sum directional-albedo (DFG) LUT used for energy
//! compensation. Nothing here duplicates that LUT.
//!
//! Two visibility models are provided for the same Charlie distribution:
//!
//! * [`v_neubelt`] — the Ashikhmin "no closed-form" visibility as popularised by
//!   Neubelt & Pettineo (*The Order: 1886*). It has no analytic inverse, so the
//!   fitted rational `1 / (4·(µ_l + µ_v − µ_l·µ_v))` is used. Cheap, stable, and
//!   the common run-time choice for cloth.
//! * [`v_charlie`] — the Estevez-Kulla soft-shadowing term built from the fitted
//!   `Λ(cosθ)` ("lambda") analytic approximation. More faithful to a measured
//!   fabric response and the reference the GPU twin must match.
//!
//! # Conventions
//! * All angles arrive as cosines against the shading normal: `µ_v = n·v`,
//!   `µ_l = n·l`, `µ_h = n·h`. These are frame-independent, so the lobe helpers
//!   do not care whether the caller works in a local `+Z` frame or world space.
//! * `roughness ∈ [0, 1]` maps directly to the Charlie width `α` (perceptually
//!   linear, *not* squared), following Imageworks / glTF `KHR_materials_sheen`.
//!   It is floored at [`MIN_ROUGHNESS`] so `1/α` stays finite.
//! * [`d_charlie`] is normalised so `∫ D(h)·cosθ_h dω = 1` over the hemisphere.
//! * Transcendental maths goes through [`bevy_math::ops`]; `sqrt` uses the
//!   inherent method. Every routine is a deterministic pure function (no RNG,
//!   I/O, GPU, globals, or `unsafe`) and clamps defensively so it never returns
//!   `NaN` or infinity.
//!
//! # References
//! * Estevez & Kulla 2017, *Production Friendly Microfacet Sheen BRDF*.
//! * Neubelt & Pettineo 2013, *Crafting a Next-Gen Material Pipeline for
//!   The Order: 1886* (the Ashikhmin visibility).
//! * Khronos `KHR_materials_sheen` reference implementation.

use bevy_math::ops;
use core::f32::consts::TAU;

/// Smallest sheen roughness, so the Charlie exponent `1/α` stays finite.
pub const MIN_ROUGHNESS: f32 = 1.0e-3;

/// Floor applied to cosines inside visibility denominators to avoid divide-by-zero.
pub const MIN_COS: f32 = 1.0e-4;

/// Estevez-Kulla "Charlie" sheen normal-distribution function.
///
/// `D(h) = (2 + 1/α) · sinθ_h^{1/α} / (2π)`, with `cosθ_h = n·h` and
/// `α = roughness` floored at [`MIN_ROUGHNESS`]. Returns `0` for a back-facing
/// half vector (`µ_h ≤ 0`). The distribution is normalised so that
/// `∫ D(h)·cosθ_h dω = 1` over the upper hemisphere.
///
/// The inverted-Gaussian shape (a `sinθ` power) peaks at grazing half angles,
/// which is what produces the soft rim seen on velvet and brushed fabric.
#[inline]
pub fn d_charlie(roughness: f32, n_dot_h: f32) -> f32 {
    let cos_h = n_dot_h.clamp(-1.0, 1.0);
    if cos_h <= 0.0 {
        return 0.0;
    }
    let alpha = roughness.clamp(MIN_ROUGHNESS, 1.0);
    let inv_alpha = 1.0 / alpha;
    let sin2 = (1.0 - cos_h * cos_h).max(0.0);
    let sin_h = sin2.sqrt();
    // sinθ^{1/α}; guard the base so `powf(0, …)` yields 0 rather than NaN.
    let pow = if sin_h <= 0.0 {
        0.0
    } else {
        ops::powf(sin_h, inv_alpha)
    };
    let d = (2.0 + inv_alpha) * pow / TAU;
    if d.is_finite() {
        d.max(0.0)
    } else {
        0.0
    }
}

/// Ashikhmin "no closed-form" sheen visibility (Neubelt & Pettineo variant).
///
/// `V = 1 / (4·(µ_l + µ_v − µ_l·µ_v))`. This fitted rational stands in for the
/// Ashikhmin masking term, which has no analytic form, and already folds in the
/// `1/(4 µ_l µ_v)` BRDF denominator so the caller multiplies it straight onto
/// [`d_charlie`]. Returns `0` when either direction is below the horizon. The
/// denominator is floored so the result stays finite at grazing angles.
#[inline]
pub fn v_neubelt(n_dot_v: f32, n_dot_l: f32) -> f32 {
    if n_dot_v <= 0.0 || n_dot_l <= 0.0 {
        return 0.0;
    }
    let mu_v = n_dot_v.clamp(MIN_COS, 1.0);
    let mu_l = n_dot_l.clamp(MIN_COS, 1.0);
    // 4·(µ_l + µ_v − µ_l·µ_v) = 4·(1 − (1−µ_l)(1−µ_v)) ≥ 0.
    let denom = 4.0 * (mu_l + mu_v - mu_l * mu_v);
    if denom <= MIN_COS {
        return 0.0;
    }
    let v = 1.0 / denom;
    if v.is_finite() {
        v.max(0.0)
    } else {
        0.0
    }
}

/// Fitted helper for the Estevez-Kulla Charlie soft-shadowing `Λ` term.
///
/// Interpolates the published coefficients between the `α = 1` and `α = 0`
/// fits by `(1 − α)²` and evaluates `a / (1 + b·x^c) + d·x + e`. Private because
/// only [`lambda_sheen`] stitches the two halves together.
#[inline]
fn lambda_sheen_helper(x: f32, alpha: f32) -> f32 {
    let one_minus = (1.0 - alpha).clamp(0.0, 1.0);
    let t = one_minus * one_minus;
    let a = lerp(21.5473, 25.3245, t);
    let b = lerp(3.829_87, 3.324_35, t);
    let c = lerp(0.198_23, 0.168_01, t);
    let d = lerp(-1.977_60, -1.273_93, t);
    let e = lerp(-4.320_54, -4.859_67, t);
    let xc = ops::powf(x.max(0.0), c);
    let v = a / (1.0 + b * xc) + d * x + e;
    if v.is_finite() {
        v
    } else {
        0.0
    }
}

/// Estevez-Kulla Charlie soft-shadowing `Λ(cosθ)` ("lambda") analytic fit.
///
/// Uses the published symmetric split at `cosθ = 0.5`: the fitted rational is
/// evaluated directly below the split and mirrored above it. `α = roughness`,
/// floored at [`MIN_ROUGHNESS`]. The result is non-negative and finite; it feeds
/// the masking term `1 / (1 + Λ(µ_v) + Λ(µ_l))`.
#[inline]
pub fn lambda_sheen(cos_theta: f32, roughness: f32) -> f32 {
    let alpha = roughness.clamp(MIN_ROUGHNESS, 1.0);
    let c = cos_theta.clamp(0.0, 1.0);
    let l = if c < 0.5 {
        ops::exp(lambda_sheen_helper(c, alpha))
    } else {
        ops::exp(2.0 * lambda_sheen_helper(0.5, alpha) - lambda_sheen_helper(1.0 - c, alpha))
    };
    if l.is_finite() {
        l.max(0.0)
    } else {
        0.0
    }
}

/// Estevez-Kulla Charlie soft-shadowing visibility built from [`lambda_sheen`].
///
/// `V = 1 / ((1 + Λ(µ_v) + Λ(µ_l)) · 4·µ_v·µ_l)`. This is the physically
/// motivated alternative to [`v_neubelt`]; it folds in the `1/(4 µ_v µ_l)`
/// denominator so it, too, multiplies directly onto [`d_charlie`]. Returns `0`
/// below the horizon and floors the cosines to stay finite.
#[inline]
pub fn v_charlie(n_dot_v: f32, n_dot_l: f32, roughness: f32) -> f32 {
    if n_dot_v <= 0.0 || n_dot_l <= 0.0 {
        return 0.0;
    }
    let mu_v = n_dot_v.clamp(MIN_COS, 1.0);
    let mu_l = n_dot_l.clamp(MIN_COS, 1.0);
    let shadow = 1.0 + lambda_sheen(mu_v, roughness) + lambda_sheen(mu_l, roughness);
    let denom = shadow * 4.0 * mu_v * mu_l;
    if denom <= MIN_COS * MIN_COS {
        return 0.0;
    }
    let v = 1.0 / denom;
    if v.is_finite() {
        v.max(0.0)
    } else {
        0.0
    }
}

/// Colourless Charlie sheen lobe `D·V` using the Neubelt visibility.
///
/// Multiplies [`d_charlie`] by [`v_neubelt`]. The sheen colour and the `n·l`
/// cosine are applied by the caller. Returns `0` for a below-horizon
/// configuration or a degenerate half vector.
#[inline]
pub fn sheen_lobe_neubelt(n_dot_v: f32, n_dot_l: f32, n_dot_h: f32, roughness: f32) -> f32 {
    let f = d_charlie(roughness, n_dot_h) * v_neubelt(n_dot_v, n_dot_l);
    if f.is_finite() {
        f.max(0.0)
    } else {
        0.0
    }
}

/// Colourless Charlie sheen lobe `D·V` using the soft-shadowing visibility.
///
/// Multiplies [`d_charlie`] by [`v_charlie`]. The sheen colour and the `n·l`
/// cosine are applied by the caller. Returns `0` for a below-horizon
/// configuration or a degenerate half vector.
#[inline]
pub fn sheen_lobe_charlie(n_dot_v: f32, n_dot_l: f32, n_dot_h: f32, roughness: f32) -> f32 {
    let f = d_charlie(roughness, n_dot_h) * v_charlie(n_dot_v, n_dot_l, roughness);
    if f.is_finite() {
        f.max(0.0)
    } else {
        0.0
    }
}

/// Linear interpolation `a + (b − a)·t`, `t` unclamped (callers pass in-range).
#[inline]
fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `∫ D(h)·cosθ_h dω = 1` over the hemisphere. `D` is isotropic in azimuth,
    /// so a dense `cosθ` sweep with the uniform-hemisphere weight `2π/N`
    /// suffices.
    #[test]
    fn charlie_ndf_normalises_over_hemisphere() {
        for &r in &[0.3f32, 0.6, 1.0] {
            let n = 200_000usize;
            let mut acc = 0.0f64;
            for i in 0..n {
                let cos_t = (i as f32 + 0.5) / n as f32;
                let d = d_charlie(r, cos_t);
                acc += (d * cos_t) as f64;
            }
            let integral = (acc / n as f64 * TAU as f64) as f32;
            assert!(
                (integral - 1.0).abs() < 0.02,
                "charlie D integral={integral} r={r}"
            );
        }
    }

    #[test]
    fn charlie_ndf_backface_is_zero() {
        assert_eq!(d_charlie(0.5, 0.0), 0.0);
        assert_eq!(d_charlie(0.5, -0.4), 0.0);
    }

    /// The inverted-Gaussian NDF grows toward grazing half angles.
    #[test]
    fn charlie_ndf_peaks_at_grazing() {
        let r = 0.5;
        let near_grazing = d_charlie(r, 0.05);
        let mid = d_charlie(r, 0.5);
        let near_normal = d_charlie(r, 0.98);
        assert!(
            near_grazing > mid && mid > near_normal,
            "grazing={near_grazing} mid={mid} normal={near_normal}"
        );
    }

    #[test]
    fn neubelt_visibility_positive_and_finite() {
        for &c in &[0.02f32, 0.3, 0.7, 1.0] {
            for &l in &[0.02f32, 0.5, 1.0] {
                let v = v_neubelt(c, l);
                assert!(v.is_finite() && v >= 0.0, "V={v} c={c} l={l}");
            }
        }
        assert_eq!(v_neubelt(-0.1, 0.5), 0.0);
        assert_eq!(v_neubelt(0.5, -0.1), 0.0);
    }

    #[test]
    fn neubelt_visibility_is_symmetric() {
        let a = v_neubelt(0.3, 0.8);
        let b = v_neubelt(0.8, 0.3);
        assert!((a - b).abs() < 1.0e-7, "a={a} b={b}");
    }

    #[test]
    fn charlie_visibility_positive_and_finite() {
        for &r in &[0.2f32, 0.5, 1.0] {
            for &c in &[0.05f32, 0.3, 0.7, 1.0] {
                let v = v_charlie(c, 0.6, r);
                assert!(v.is_finite() && v >= 0.0, "V={v} c={c} r={r}");
            }
        }
        assert_eq!(v_charlie(-0.1, 0.5, 0.5), 0.0);
    }

    #[test]
    fn lambda_sheen_non_negative_and_finite() {
        for &r in &[0.1f32, 0.5, 1.0] {
            for i in 0..=20 {
                let c = i as f32 / 20.0;
                let l = lambda_sheen(c, r);
                assert!(l.is_finite() && l >= 0.0, "lambda={l} c={c} r={r}");
            }
        }
    }

    /// Both lobe helpers are reciprocal: swapping view and light leaves `D·V`
    /// unchanged (the half vector and both visibilities are symmetric).
    #[test]
    fn sheen_lobe_is_reciprocal() {
        let (mu_v, mu_l, mu_h, r) = (0.42f32, 0.77, 0.63, 0.4);
        let n = sheen_lobe_neubelt(mu_v, mu_l, mu_h, r);
        let n_swapped = sheen_lobe_neubelt(mu_l, mu_v, mu_h, r);
        assert!((n - n_swapped).abs() < 1.0e-7, "neubelt {n} vs {n_swapped}");
        let c = sheen_lobe_charlie(mu_v, mu_l, mu_h, r);
        let c_swapped = sheen_lobe_charlie(mu_l, mu_v, mu_h, r);
        assert!((c - c_swapped).abs() < 1.0e-7, "charlie {c} vs {c_swapped}");
    }

    #[test]
    fn sheen_lobe_below_horizon_is_zero() {
        assert_eq!(sheen_lobe_neubelt(0.0, 0.5, 0.5, 0.5), 0.0);
        assert_eq!(sheen_lobe_charlie(0.5, 0.0, 0.5, 0.5), 0.0);
        assert_eq!(sheen_lobe_neubelt(0.5, 0.5, -0.2, 0.5), 0.0);
    }

    #[test]
    fn degenerate_inputs_never_nan() {
        assert_eq!(d_charlie(0.5, f32::NAN), 0.0);
        assert!(d_charlie(0.0, 0.5).is_finite());
        assert!(v_neubelt(0.0, 0.0).is_finite());
        assert!(v_charlie(0.0, 0.0, 0.0).is_finite());
        assert!(lambda_sheen(f32::NAN, 0.5).is_finite());
        assert!(sheen_lobe_neubelt(f32::NAN, 0.5, 0.5, 0.5).is_finite());
    }
}
