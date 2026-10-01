//! Geometric specular anti-aliasing from screen-space normal derivatives —
//! CPU golden reference (Kaplanyan 2016 / Tokuyoshi–Kaplanyan 2019).
//!
//! A curved or bumpy surface packs many micro-orientations into a single
//! shaded pixel.  Under motion this sub-pixel normal distribution is sampled
//! at ever-shifting points, so a tight specular lobe flickers between frames.
//! The geometric specular AA filter estimates the *screen-space normal
//! variance* `σ²` directly from the per-pixel partial derivatives of the
//! shading normal and widens the GGX lobe just enough to cover it, trading a
//! slightly softer highlight for a stable one.
//!
//! Given the screen-space derivatives `∂n/∂x` and `∂n/∂y` of the (interpolated,
//! renormalised) shading normal, the projected variance estimate is
//!
//! ```text
//! σ² = SIGMA2 · ( ⟨∂n/∂x, ∂n/∂x⟩ + ⟨∂n/∂y, ∂n/∂y⟩ ),
//! ```
//!
//! the kernel roughness is `κ = 2σ²`, and the filtered GGX width follows the
//! additive-variance rule
//!
//! ```text
//! alpha'² = clamp( alpha² + min(2σ², KAPPA_MAX),  0, 1 ).
//! ```
//!
//! The `min(·, KAPPA_MAX)` clamp (Tokuyoshi–Kaplanyan's `κ` cap) bounds how
//! much a silhouette or normal-map discontinuity may blur the lobe, preventing
//! grazing edges from turning mirror-like materials fully rough.  The default
//! constants `SIGMA2 = 0.15915494` and `KAPPA_MAX = 0.18` are the values
//! recommended by Tokuyoshi & Kaplanyan 2019.
//!
//! # Conventions
//! * `no_std`: allocation-free; only [`bevy_math::Vec3`] (the normal-derivative
//!   vectors) is imported.  No transcendental functions are required, so
//!   [`bevy_math::ops`] is *not* imported; the inherent `f32::sqrt` suffices.
//! * `∂n/∂x`, `∂n/∂y` are world- or view-space normal differences per pixel;
//!   only their squared lengths enter the estimate, so the frame choice is
//!   irrelevant as long as it is consistent.
//! * Perceptual `roughness ∈ [0, 1]` maps to the GGX width `alpha = roughness²`
//!   through [`roughness_to_alpha`]; this module never re-derives it.
//! * All inputs are sanitised: non-finite derivatives contribute zero variance,
//!   the scale and cap are floored non-negative, and every result is finite and
//!   clamped to a physical range (never `NaN`/`inf`).
//!
//! # References
//! * Anton Kaplanyan et al. 2016, *Filtering Distributions of Normals for
//!   Shading Antialiasing* (HPG) — the screen-space normal-variance filter.
//! * Yusuke Tokuyoshi & Anton Kaplanyan 2019, *Improved Geometric Specular
//!   Antialiasing* (I3D) — the projected-variance estimate and the `κ` cap used
//!   here, including the recommended `SIGMA2`/`KAPPA_MAX` constants.

use crate::gi::spec_gi::ggx_lobe::{MIN_ALPHA, roughness_to_alpha};
use bevy_math::Vec3;

/// Default projection constant `SIGMA2` scaling the summed squared normal
/// derivatives into a variance (Tokuyoshi–Kaplanyan 2019, ≈ `1/(2π)`).
pub const DEFAULT_SIGMA2: f32 = 0.159_154_94;

/// Default kernel-roughness cap `KAPPA_MAX` bounding `min(2σ², κ)`
/// (Tokuyoshi–Kaplanyan 2019).
pub const DEFAULT_KAPPA_MAX: f32 = 0.18;

/// Returns the squared length of a normal-derivative vector, treating any
/// non-finite component as a zero contribution.
#[inline]
fn safe_sq_len(v: Vec3) -> f32 {
    if v.is_finite() {
        v.dot(v).max(0.0)
    } else {
        0.0
    }
}

/// Projected screen-space normal variance
/// `σ² = scale · (⟨∂n/∂x, ∂n/∂x⟩ + ⟨∂n/∂y, ∂n/∂y⟩)`.
///
/// `scale` is the `SIGMA2` projection constant (see [`DEFAULT_SIGMA2`]); it is
/// floored non-negative.  This is the Tokuyoshi 2019 "screen-space variance"
/// estimate: it rises with surface curvature and normal-map detail and is `0`
/// on a locally flat, static surface.
#[inline]
pub fn screen_space_variance(ddx_normal: Vec3, ddy_normal: Vec3, scale: f32) -> f32 {
    let s = if scale.is_finite() { scale.max(0.0) } else { 0.0 };
    let sum = safe_sq_len(ddx_normal) + safe_sq_len(ddy_normal);
    let variance = s * sum;
    if variance.is_finite() { variance.max(0.0) } else { 0.0 }
}

/// Kernel roughness `κ = min(2σ², max_kernel)` — the extra GGX variance the
/// filter is allowed to inject.
///
/// `max_kernel` is the `KAPPA_MAX` cap (see [`DEFAULT_KAPPA_MAX`]); it is
/// floored non-negative.  The `min` prevents discontinuities from blurring the
/// lobe without bound.
#[inline]
pub fn kernel_roughness_sq(variance: f32, max_kernel: f32) -> f32 {
    let v = if variance.is_finite() { variance.max(0.0) } else { 0.0 };
    let cap = if max_kernel.is_finite() { max_kernel.max(0.0) } else { 0.0 };
    (2.0 * v).min(cap)
}

/// Adds the capped kernel variance to a base GGX variance:
/// `alpha'² = clamp(alpha² + min(2σ², max_kernel), MIN_ALPHA², 1)`.
///
/// Returns the filtered `alpha'²` as a valid, finite GGX width that is never
/// sharper than the base.
#[inline]
pub fn filter_alpha_sq(base_alpha_sq: f32, variance: f32, max_kernel: f32) -> f32 {
    let base = if base_alpha_sq.is_finite() { base_alpha_sq.max(0.0) } else { 0.0 };
    let eff = base + kernel_roughness_sq(variance, max_kernel);
    let floor = MIN_ALPHA * MIN_ALPHA;
    if eff.is_finite() { eff.clamp(floor, 1.0) } else { base.clamp(floor, 1.0) }
}

/// Closed-form additional GGX variance `Δα² = min(2σ², max_kernel)` contributed
/// by the screen-space normal derivatives.
///
/// Convenience wrapper that chains [`screen_space_variance`] and
/// [`kernel_roughness_sq`] so callers can add the result to their own `alpha²`.
#[inline]
pub fn delta_alpha_sq_from_derivatives(
    ddx_normal: Vec3,
    ddy_normal: Vec3,
    scale: f32,
    max_kernel: f32,
) -> f32 {
    let variance = screen_space_variance(ddx_normal, ddy_normal, scale);
    kernel_roughness_sq(variance, max_kernel)
}

/// High-level helper: filters a perceptual `base_roughness` using the
/// screen-space normal derivatives, returning the anti-aliased perceptual
/// roughness in `[0, 1]`.
///
/// Pipeline: `roughness → alpha² → + min(2σ², max_kernel) → alpha'² →
/// roughness`.  Because the extra variance is non-negative, the output is
/// always `≥` the input (the filter only softens).
#[inline]
pub fn geometric_specular_aa_roughness(
    base_roughness: f32,
    ddx_normal: Vec3,
    ddy_normal: Vec3,
    scale: f32,
    max_kernel: f32,
) -> f32 {
    let alpha = roughness_to_alpha(base_roughness);
    let variance = screen_space_variance(ddx_normal, ddy_normal, scale);
    let eff_alpha_sq = filter_alpha_sq(alpha * alpha, variance, max_kernel);
    // roughness = (α²)^(1/4) since alpha = roughness² and alpha² is the input.
    let roughness = eff_alpha_sq.max(0.0).sqrt().sqrt();
    roughness.clamp(0.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f32 = 1.0e-6;

    #[test]
    fn flat_surface_adds_no_variance() {
        let v = screen_space_variance(Vec3::ZERO, Vec3::ZERO, DEFAULT_SIGMA2);
        assert!(v.abs() < EPS);
        let r = geometric_specular_aa_roughness(
            0.2,
            Vec3::ZERO,
            Vec3::ZERO,
            DEFAULT_SIGMA2,
            DEFAULT_KAPPA_MAX,
        );
        assert!((r - 0.2).abs() < 1.0e-4, "r = {r}");
    }

    #[test]
    fn variance_scales_with_derivative_magnitude() {
        let small = screen_space_variance(
            Vec3::new(0.1, 0.0, 0.0),
            Vec3::new(0.0, 0.1, 0.0),
            DEFAULT_SIGMA2,
        );
        let large = screen_space_variance(
            Vec3::new(0.4, 0.0, 0.0),
            Vec3::new(0.0, 0.4, 0.0),
            DEFAULT_SIGMA2,
        );
        assert!(large > small);
        assert!(small > 0.0);
    }

    #[test]
    fn kappa_cap_bounds_kernel() {
        // Enormous variance must still be capped at max_kernel.
        let k = kernel_roughness_sq(1.0e3, DEFAULT_KAPPA_MAX);
        assert!((k - DEFAULT_KAPPA_MAX).abs() < EPS, "k = {k}");
    }

    #[test]
    fn uncapped_region_is_two_sigma_sq() {
        let v = 0.01_f32;
        let k = kernel_roughness_sq(v, DEFAULT_KAPPA_MAX);
        assert!((k - 2.0 * v).abs() < EPS, "k = {k}");
    }

    #[test]
    fn roughness_only_increases_and_is_bounded() {
        let ddx = Vec3::new(0.3, 0.1, 0.0);
        let ddy = Vec3::new(0.1, 0.3, 0.0);
        for &r in &[0.0_f32, 0.05, 0.2, 0.5, 0.9, 1.0] {
            let out = geometric_specular_aa_roughness(
                r,
                ddx,
                ddy,
                DEFAULT_SIGMA2,
                DEFAULT_KAPPA_MAX,
            );
            assert!(out >= r - 1.0e-4, "r {r} out {out}");
            assert!((0.0..=1.0).contains(&out));
        }
    }

    #[test]
    fn filter_alpha_sq_clamped_to_unit() {
        // Base already at the ceiling plus kernel must saturate at 1, not blow up.
        let out = filter_alpha_sq(0.95, 10.0, DEFAULT_KAPPA_MAX);
        assert!((0.0..=1.0).contains(&out));
    }

    #[test]
    fn delta_matches_pipeline() {
        let ddx = Vec3::new(0.2, 0.05, 0.01);
        let ddy = Vec3::new(0.03, 0.22, 0.0);
        let variance = screen_space_variance(ddx, ddy, DEFAULT_SIGMA2);
        let direct = delta_alpha_sq_from_derivatives(
            ddx,
            ddy,
            DEFAULT_SIGMA2,
            DEFAULT_KAPPA_MAX,
        );
        let staged = kernel_roughness_sq(variance, DEFAULT_KAPPA_MAX);
        assert!((direct - staged).abs() < EPS);
    }

    #[test]
    fn defends_against_garbage() {
        let nan = Vec3::new(f32::NAN, 0.0, 0.0);
        let inf = Vec3::splat(f32::INFINITY);
        assert!(screen_space_variance(nan, inf, DEFAULT_SIGMA2).is_finite());
        assert!(kernel_roughness_sq(f32::NAN, f32::NAN).is_finite());
        let out = geometric_specular_aa_roughness(
            f32::NAN,
            nan,
            inf,
            f32::NAN,
            f32::NAN,
        );
        assert!((0.0..=1.0).contains(&out));
    }

    #[test]
    fn negative_scale_is_floored() {
        let v = screen_space_variance(Vec3::ONE, Vec3::ONE, -5.0);
        assert!(v.abs() < EPS);
    }
}
