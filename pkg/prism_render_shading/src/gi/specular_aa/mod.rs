//! Geometric specular anti-aliasing — CPU golden reference.
//!
//! Specular aliasing is the shimmer that appears when a tight glossy highlight
//! is sampled at sub-pixel frequencies the raster cannot resolve: as the camera
//! or surface moves, the lobe snaps between pixels and sparkles.  The classical
//! fix is to widen the lobe just enough to cover the sub-pixel normal
//! distribution, converting invisible high-frequency detail into a slightly
//! rougher but *stable* highlight.  This subsystem collects the three standard
//! families of that technique as backend-neutral, allocation-free references:
//!
//! * [`toksvig`] — Toksvig 2005 normal-map filtering: derive an anti-aliasing
//!   factor from the mip-averaged normal length `|avg_normal|` and fold it back
//!   into effective shininess / GGX roughness.
//! * [`normal_variance`] — Kaplanyan 2016 / Tokuyoshi–Kaplanyan 2019 geometric
//!   AA: estimate screen-space normal variance `σ²` from the per-pixel normal
//!   derivatives and add the capped kernel roughness `min(2σ², κ)` to `alpha²`.
//! * [`lean`] — LEAN / LEADR slope-moment mapping: accumulate the first/second
//!   slope moments, form the micro-slope covariance `Σ = M − B⊗B`, and add it to
//!   the GGX width to recover an *anisotropic* filtered roughness.
//!
//! All three share one principle: specular AA is additive in lobe **variance**
//! and may only ever *coarsen* a material, never sharpen it.  The high-level
//! [`apply_specular_aa`] entry point drives the derivative-based filter — the
//! one that needs no extra authored data — and returns a possibly-anisotropic
//! [`AnisoRoughness`].  The Toksvig and LEAN paths can be layered on top via
//! their own helpers when averaged-normal or slope-moment data is available.
//!
//! # Conventions
//! * `no_std`, allocation-free; math via [`bevy_math`], transcendentals via
//!   [`bevy_math::ops`].  Perceptual `roughness ∈ [0, 1]` maps to the GGX width
//!   `alpha = roughness²` through
//!   [`roughness_to_alpha`](crate::gi::spec_gi::ggx_lobe::roughness_to_alpha),
//!   which this subsystem reuses rather than re-implements.
//! * Every public routine is a deterministic pure function that sanitises its
//!   inputs (non-finite → neutral) and clamps its outputs to a valid, finite
//!   range, so the reference can never inject `NaN`/`inf` into the pipeline.
//!
//! # References
//! * Toksvig 2005, *Mipmapping Normal Maps*.
//! * Kaplanyan et al. 2016, *Filtering Distributions of Normals for Shading
//!   Antialiasing*.
//! * Tokuyoshi & Kaplanyan 2019, *Improved Geometric Specular Antialiasing*.
//! * Olano & Baker 2010, *LEAN Mapping*; Dupuy et al. 2013, *LEADR*.

pub mod lean;
pub mod normal_variance;
pub mod toksvig;

pub use lean::{
    LeanMoments, clamp_psd, covariance_to_anisotropic_roughness, normal_to_slope, principal_axes,
};
pub use normal_variance::{
    DEFAULT_KAPPA_MAX, DEFAULT_SIGMA2, delta_alpha_sq_from_derivatives,
    geometric_specular_aa_roughness, kernel_roughness_sq, screen_space_variance,
};
pub use toksvig::{
    combine_roughness, effective_shininess, toksvig_delta_alpha_sq, toksvig_factor,
    toksvig_roughness,
};

use bevy_math::Vec3;

/// Screen-space partial derivatives of the shading normal, `(∂n/∂x, ∂n/∂y)`.
///
/// These are the per-pixel finite differences of the interpolated, renormalised
/// shading normal and drive the geometric variance estimate in
/// [`apply_specular_aa`].
#[derive(Clone, Copy, Debug, Default)]
pub struct NormalDerivatives {
    /// Horizontal derivative `∂n/∂x`.
    pub ddx: Vec3,
    /// Vertical derivative `∂n/∂y`.
    pub ddy: Vec3,
}

impl NormalDerivatives {
    /// Builds a derivative pair from the two screen-space normal differences.
    #[inline]
    pub fn new(ddx: Vec3, ddy: Vec3) -> Self {
        Self { ddx, ddy }
    }
}

/// Tuning parameters for geometric specular anti-aliasing.
///
/// Defaults are the Tokuyoshi–Kaplanyan 2019 recommendations
/// (`screen_variance_scale = 0.15915494`, `max_kernel = 0.18`).
#[derive(Clone, Copy, Debug)]
pub struct SpecularAaParams {
    /// Cap on the injected kernel variance `min(2σ², max_kernel)`; bounds how
    /// much a silhouette or normal discontinuity may blur the lobe.
    pub max_kernel: f32,
    /// `SIGMA2` projection constant scaling the summed squared normal
    /// derivatives into a variance.
    pub screen_variance_scale: f32,
}

impl Default for SpecularAaParams {
    #[inline]
    fn default() -> Self {
        Self {
            max_kernel: DEFAULT_KAPPA_MAX,
            screen_variance_scale: DEFAULT_SIGMA2,
        }
    }
}

/// Two-axis perceptual roughness, each component in `[0, 1]`.
///
/// The isotropic derivative-based filter sets `x == y`; the LEAN path can
/// produce genuinely anisotropic axes.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct AnisoRoughness {
    /// Roughness along the tangent (`x`) axis.
    pub x: f32,
    /// Roughness along the bitangent (`y`) axis.
    pub y: f32,
}

impl AnisoRoughness {
    /// Builds an anisotropic roughness, clamping both axes to `[0, 1]`.
    #[inline]
    pub fn new(x: f32, y: f32) -> Self {
        Self {
            x: sanitize_roughness(x),
            y: sanitize_roughness(y),
        }
    }

    /// Builds an isotropic roughness (`x == y`).
    #[inline]
    pub fn splat(roughness: f32) -> Self {
        let r = sanitize_roughness(roughness);
        Self { x: r, y: r }
    }

    /// Returns `true` when the two axes are equal within `tol`.
    #[inline]
    pub fn is_isotropic(&self, tol: f32) -> bool {
        (self.x - self.y).abs() <= tol.max(0.0)
    }

    /// Collapses to a single isotropic roughness by taking the coarser axis.
    #[inline]
    pub fn max_axis(&self) -> f32 {
        self.x.max(self.y)
    }
}

/// Clamps a roughness to `[0, 1]`, mapping non-finite inputs to `0`.
#[inline]
fn sanitize_roughness(r: f32) -> f32 {
    if r.is_finite() { r.clamp(0.0, 1.0) } else { 0.0 }
}

/// Applies derivative-based geometric specular anti-aliasing to a base
/// perceptual roughness.
///
/// Estimates the screen-space normal variance from `normal_derivatives` and
/// widens the GGX lobe by the capped kernel roughness, returning the
/// anti-aliased [`AnisoRoughness`].  The derivative estimate is isotropic, so
/// both axes are equal; feed the result through the [`lean`] helpers when
/// anisotropic bump statistics are available.  The output is always `≥` the
/// input (specular AA only softens) and finite.
#[inline]
pub fn apply_specular_aa(
    base_roughness: f32,
    normal_derivatives: NormalDerivatives,
    params: SpecularAaParams,
) -> AnisoRoughness {
    let filtered = geometric_specular_aa_roughness(
        base_roughness,
        normal_derivatives.ddx,
        normal_derivatives.ddy,
        params.screen_variance_scale,
        params.max_kernel,
    );
    AnisoRoughness::splat(filtered)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_params_match_paper_constants() {
        let p = SpecularAaParams::default();
        assert!((p.max_kernel - DEFAULT_KAPPA_MAX).abs() < 1.0e-9);
        assert!((p.screen_variance_scale - DEFAULT_SIGMA2).abs() < 1.0e-9);
    }

    #[test]
    fn flat_surface_passes_roughness_through() {
        let out = apply_specular_aa(
            0.3,
            NormalDerivatives::new(Vec3::ZERO, Vec3::ZERO),
            SpecularAaParams::default(),
        );
        assert!(out.is_isotropic(1.0e-6));
        assert!((out.x - 0.3).abs() < 1.0e-4, "x = {}", out.x);
    }

    #[test]
    fn curved_surface_only_coarsens() {
        let d = NormalDerivatives::new(Vec3::new(0.3, 0.1, 0.0), Vec3::new(0.1, 0.3, 0.0));
        for &r in &[0.0_f32, 0.1, 0.5, 0.9, 1.0] {
            let out = apply_specular_aa(r, d, SpecularAaParams::default());
            assert!(out.x >= r - 1.0e-4, "r {r} out {}", out.x);
            assert!((0.0..=1.0).contains(&out.x));
        }
    }

    #[test]
    fn aniso_roughness_helpers() {
        let a = AnisoRoughness::new(2.0, -1.0);
        assert!((0.0..=1.0).contains(&a.x) && (0.0..=1.0).contains(&a.y));
        let s = AnisoRoughness::splat(0.4);
        assert!(s.is_isotropic(0.0));
        assert!((s.max_axis() - 0.4).abs() < 1.0e-6);
        let n = AnisoRoughness::splat(f32::NAN);
        assert!(n.x == 0.0 && n.y == 0.0);
    }

    #[test]
    fn garbage_inputs_stay_valid() {
        let out = apply_specular_aa(
            f32::NAN,
            NormalDerivatives::new(Vec3::splat(f32::NAN), Vec3::splat(f32::INFINITY)),
            SpecularAaParams::default(),
        );
        assert!((0.0..=1.0).contains(&out.x) && (0.0..=1.0).contains(&out.y));
    }

    #[test]
    fn reexports_are_reachable() {
        // Touch one symbol from each submodule to lock the public surface.
        let _ = toksvig_roughness(0.2, 0.8);
        let _ = screen_space_variance(Vec3::ZERO, Vec3::ZERO, DEFAULT_SIGMA2);
        let _ = LeanMoments::from_normal(Vec3::Z);
    }
}
