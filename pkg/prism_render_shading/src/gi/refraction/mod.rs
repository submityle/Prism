//! Screen-space refraction CPU golden references.
//!
//! Deterministic, GPU-free rough-refraction models for transmissive surfaces
//! (glass, gems, liquids).  Distinct from [`crate::gi::material`] wet-surface
//! blending: here the background buffer is bent and absorbed per Snell + Beer:
//!
//! * [`bend`] — Snell view-ray refraction and thickness-scaled screen-UV
//!   displacement of the background buffer.
//! * [`absorption`] — Beer-Lambert path-length tint plus Fresnel split between
//!   reflected and transmitted radiance.
//! * [`rough`] — roughness-to-blur-LOD mapping and dispersion (per-channel IOR)
//!   for frosted / dispersive refraction.
//!
//! The top-level [`refract_background`] stitches these three references into a
//! single pass: it bends the background UV per channel (with chromatic
//! dispersion), picks the pre-filtered mip level and gather radius implied by
//! the surface roughness, and splits the incident energy with the Fresnel
//! equations.  The caller then samples the background buffer at the returned
//! per-channel UVs and feeds those samples to [`RefractedBackground::resolve`],
//! which applies the Fresnel transmit weight and Beer-Lambert volume
//! absorption to produce the final transmitted radiance.
//!
//! # Conventions
//! * Direction vectors follow the sub-modules: `view` is the incident ray
//!   pointing into the surface, `normal` points out toward the eye, and view
//!   space has `+z` toward the camera with the screen in the `xy` plane.
//! * UVs are clamped into `[0, 1]^2`; indices of refraction into `[1, inf)`;
//!   roughness, dispersion, and cosines into their natural ranges.
//! * `no_std`: math via `bevy_math`; transcendentals via `bevy_math::ops`.
//!   Nothing is allocated and every result is finite.
//! * Every item is a deterministic, allocation-free pure function (or a method
//!   on a plain-data struct) with `#[cfg(test)]` coverage.

pub mod absorption;
pub mod bend;
pub mod rough;

use bevy_math::{Vec2, Vec3};

/// Inputs describing a transmissive fragment and its surrounding media.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RefractionParams {
    /// Screen UV of the fragment itself, in `[0, 1]^2`.
    pub base_uv: Vec2,
    /// Incident view ray, pointing from the eye into the surface.
    pub view: Vec3,
    /// Surface normal, pointing out of the surface toward the eye.
    pub normal: Vec3,
    /// Index of refraction of the incident medium (air `~= 1`).
    pub ior_in: f32,
    /// Base index of refraction of the transmissive body (green channel).
    pub ior_out: f32,
    /// Geometric thickness of the body along the view axis.
    pub thickness: f32,
    /// Perceptual roughness in `[0, 1]`; drives the transmitted-blur LOD.
    pub roughness: f32,
    /// Chromatic dispersion strength in `[0, 1]`; `0` disables colour fringing.
    pub dispersion: f32,
    /// Per-channel absorption coefficient (inverse distance) of the medium.
    pub sigma_a: Vec3,
    /// Converts a view-space tangential shift into UV units (focal/aspect).
    pub projection_scale: f32,
    /// Top mip index of the background pyramid used for roughness blur.
    pub max_lod: f32,
    /// Maximum gather-kernel radius in texels at full roughness.
    pub max_kernel_radius: f32,
}

/// The resolved description of a screen-space refraction sample.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RefractedBackground {
    /// Per-channel `[0, 1]^2` background sample UVs `[red, green, blue]`.
    pub uv: [Vec2; 3],
    /// Fractional mip level to sample for the roughness blur.
    pub lod: f32,
    /// Gather-kernel radius in texels for the roughness blur.
    pub kernel_radius: f32,
    /// Fresnel-reflected energy fraction, in `[0, 1]`.
    pub reflect_weight: f32,
    /// Fresnel-transmitted energy fraction, in `[0, 1]`; `1 - reflect_weight`.
    pub transmit_weight: f32,
    /// `true` when the interface totally internally reflected the view ray.
    pub total_internal_reflection: bool,
    /// Optical path length through the body, for Beer-Lambert absorption.
    pub path_length: f32,
}

impl RefractedBackground {
    /// Combines the background samples into the transmitted radiance.
    ///
    /// Each channel takes its matching component from the background colour
    /// sampled at that channel's UV (`bg_r.x`, `bg_g.y`, `bg_b.z`), then applies
    /// Beer-Lambert absorption over [`Self::path_length`] and the Fresnel
    /// transmit weight.  The result is the transmitted contribution a refraction
    /// pass adds; the reflected lobe (weight [`Self::reflect_weight`]) is
    /// composited separately by the caller.
    #[inline]
    pub fn resolve(&self, bg_r: Vec3, bg_g: Vec3, bg_b: Vec3, sigma_a: Vec3) -> Vec3 {
        let transmittance = absorption::beer_lambert_transmittance(sigma_a, self.path_length);
        let gathered = Vec3::new(bg_r.x.max(0.0), bg_g.y.max(0.0), bg_b.z.max(0.0));
        gathered * transmittance * self.transmit_weight
    }
}

/// Builds a full screen-space refraction sample from a transmissive fragment.
///
/// Pipeline:
/// 1. Compute the green-channel refracted UV displacement via
///    [`bend::bend_background_uv`] and record total-internal-reflection.
/// 2. Spread that displacement into per-channel offsets with
///    [`rough::dispersive_offsets`] for chromatic fringing, re-clamping each UV.
/// 3. Map `roughness` to a mip [`rough::roughness_to_lod`] and a gather radius
///    [`rough::roughness_to_kernel_radius`].
/// 4. Split the incident energy with the exact Fresnel
///    [`absorption::reflect_transmit_split`] at the incidence cosine.
/// 5. Derive the Beer-Lambert path length through the slab.
///
/// On total internal reflection the per-channel UVs collapse to `base_uv`, the
/// transmit weight is `0`, and the reflect weight is `1`.
#[inline]
pub fn refract_background(params: &RefractionParams) -> RefractedBackground {
    let bend = bend::bend_background_uv(
        params.base_uv,
        params.view,
        params.normal,
        params.ior_in,
        params.ior_out,
        params.thickness,
        params.projection_scale,
    );

    // Green-channel displacement, then disperse into the three channels.
    let base_offset = bend.uv - params.base_uv;
    let offsets = rough::dispersive_offsets(
        base_offset,
        params.ior_out,
        params.ior_in,
        params.dispersion,
    );
    let uv = [
        bend::clamp_uv(params.base_uv + offsets[0]),
        bend::clamp_uv(params.base_uv + offsets[1]),
        bend::clamp_uv(params.base_uv + offsets[2]),
    ];

    let lod = rough::roughness_to_lod(params.roughness, params.max_lod);
    let kernel_radius =
        rough::roughness_to_kernel_radius(params.roughness, params.max_kernel_radius);

    // Fresnel energy split at the incidence cosine.
    let n = normalize_or(params.normal, Vec3::Z);
    let i = normalize_or(params.view, Vec3::NEG_Z);
    let cos_i = (-n.dot(i)).clamp(0.0, 1.0);
    let split = absorption::reflect_transmit_split(cos_i, params.ior_in, params.ior_out);

    // Path length: straight through the slab on TIR (no refracted ray), else
    // along the refracted direction.
    let path_length = match bend.refracted {
        Some(r) => bend::slab_path_length(r, params.thickness),
        None => bend::slab_path_length(Vec3::NEG_Z, params.thickness),
    };

    // TIR transmits nothing regardless of the smooth-interface Fresnel value.
    let (reflect_weight, transmit_weight) = if bend.total_internal_reflection {
        (1.0, 0.0)
    } else {
        (split.reflect, split.transmit)
    };

    RefractedBackground {
        uv,
        lod,
        kernel_radius,
        reflect_weight,
        transmit_weight,
        total_internal_reflection: bend.total_internal_reflection,
        path_length,
    }
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

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f32 = 1.0e-4;

    fn glass_params() -> RefractionParams {
        RefractionParams {
            base_uv: Vec2::new(0.5, 0.5),
            view: Vec3::new(0.3, -0.15, -1.0).normalize(),
            normal: Vec3::Z,
            ior_in: bend::AIR_IOR,
            ior_out: 1.5,
            thickness: 1.0,
            roughness: 0.3,
            dispersion: 0.6,
            sigma_a: Vec3::new(0.1, 0.3, 0.6),
            projection_scale: 0.2,
            max_lod: 8.0,
            max_kernel_radius: 16.0,
        }
    }

    fn assert_uv_in_range(uv: Vec2) {
        assert!(uv.x >= 0.0 && uv.x <= 1.0 && uv.y >= 0.0 && uv.y <= 1.0, "{uv:?}");
    }

    #[test]
    fn integration_produces_valid_sample() {
        let out = refract_background(&glass_params());
        for uv in out.uv {
            assert_uv_in_range(uv);
        }
        assert!(out.lod >= 0.0 && out.lod <= 8.0 + EPS);
        assert!(out.kernel_radius >= 0.0 && out.kernel_radius <= 16.0 + EPS);
        assert!(out.path_length.is_finite() && out.path_length > 0.0);
        assert!(!out.total_internal_reflection);
    }

    #[test]
    fn energy_split_conserves_and_bounds() {
        let out = refract_background(&glass_params());
        assert!((out.reflect_weight + out.transmit_weight - 1.0).abs() < EPS);
        assert!(out.reflect_weight >= 0.0 && out.reflect_weight <= 1.0);
        assert!(out.transmit_weight >= 0.0 && out.transmit_weight <= 1.0);
    }

    #[test]
    fn dispersion_separates_channel_uvs() {
        let out = refract_background(&glass_params());
        // With dispersion and a real bend, the three UVs must differ.
        assert!((out.uv[0] - out.uv[2]).length() > EPS, "red vs blue identical");
    }

    #[test]
    fn no_dispersion_collapses_channel_uvs() {
        let mut p = glass_params();
        p.dispersion = 0.0;
        let out = refract_background(&p);
        assert!((out.uv[0] - out.uv[1]).length() < EPS);
        assert!((out.uv[1] - out.uv[2]).length() < EPS);
    }

    #[test]
    fn total_internal_reflection_path() {
        let mut p = glass_params();
        // Glass -> air at a steep angle total-internal-reflects.
        p.ior_in = 1.5;
        p.ior_out = bend::AIR_IOR;
        let theta = 70.0f32.to_radians();
        p.view = Vec3::new(theta.sin(), 0.0, -theta.cos());
        let out = refract_background(&p);
        assert!(out.total_internal_reflection);
        assert_eq!(out.transmit_weight, 0.0);
        assert_eq!(out.reflect_weight, 1.0);
        // No transmitted ray -> every channel samples the fragment's own UV.
        for uv in out.uv {
            assert!((uv - p.base_uv).length() < EPS);
        }
    }

    #[test]
    fn thicker_body_absorbs_more() {
        let bg = Vec3::ONE;
        let mut thin = glass_params();
        thin.thickness = 0.5;
        let mut thick = glass_params();
        thick.thickness = 4.0;

        let thin_out = refract_background(&thin);
        let thick_out = refract_background(&thick);
        let thin_rgb = thin_out.resolve(bg, bg, bg, thin.sigma_a);
        let thick_rgb = thick_out.resolve(bg, bg, bg, thick.sigma_a);
        // Longer path -> darker transmitted colour on the absorbing channel.
        assert!(thick_rgb.z < thin_rgb.z, "{thick_rgb:?} vs {thin_rgb:?}");
    }

    #[test]
    fn resolve_is_tinted_by_absorption() {
        let out = refract_background(&glass_params());
        let bg = Vec3::ONE;
        let rgb = out.resolve(bg, bg, bg, Vec3::new(0.1, 0.3, 0.6));
        // Red absorbs least -> warm transmitted tint; all non-negative.
        assert!(rgb.x > rgb.y && rgb.y > rgb.z);
        assert!(rgb.min_element() >= 0.0);
    }

    #[test]
    fn roughness_drives_lod_monotonically() {
        let mut smooth = glass_params();
        smooth.roughness = 0.1;
        let mut frosted = glass_params();
        frosted.roughness = 0.9;
        assert!(refract_background(&frosted).lod > refract_background(&smooth).lod);
    }

    #[test]
    fn degenerate_params_stay_finite() {
        let p = RefractionParams {
            base_uv: Vec2::new(f32::NAN, 2.0),
            view: Vec3::ZERO,
            normal: Vec3::ZERO,
            ior_in: f32::NAN,
            ior_out: -3.0,
            thickness: -1.0,
            roughness: 5.0,
            dispersion: -2.0,
            sigma_a: Vec3::new(-1.0, f32::INFINITY, 0.2),
            projection_scale: f32::NAN,
            max_lod: -4.0,
            max_kernel_radius: f32::NAN,
        };
        let out = refract_background(&p);
        for uv in out.uv {
            assert_uv_in_range(uv);
        }
        assert!(out.lod.is_finite() && out.kernel_radius.is_finite());
        assert!(out.path_length.is_finite());
        let rgb = out.resolve(Vec3::ONE, Vec3::ONE, Vec3::ONE, p.sigma_a);
        assert!(rgb.x.is_finite() && rgb.y.is_finite() && rgb.z.is_finite());
    }
}
