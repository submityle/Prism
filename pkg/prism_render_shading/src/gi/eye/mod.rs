//! Physically based eye-shading CPU golden references.
//!
//! Deterministic, GPU-free models for the cornea/iris stack used by AAA
//! character eyes:
//!
//! * [`cornea`] — Snell refraction of the view ray through the corneal
//!   interface plus Fresnel reflectance of the wet surface.
//! * [`iris`] — parallax-corrected iris sampling with pupil dilation and
//!   radial limbal-ring darkening.
//! * [`caustic`] — approximate corneal focusing gain applied to iris radiance.
//!
//! # Conventions
//! * All vectors are unit directions in the eye's *local frame*: the optical
//!   axis / apex normal is `+Z`, pointing out of the eye toward the camera, so
//!   a ray refracted into the eye has `z < 0`.  The iris lies in a plane
//!   parallel to `XY`.
//! * The high-level [`shade_eye`] entry point chains the three stages:
//!   refract the view ray at the cornea, trace it to the iris plane with
//!   parallax, re-map for pupil dilation, apply limbal darkening and finally
//!   the corneal focusing gain — returning everything a shader needs in a
//!   single [`EyeShadeSample`].
//! * Every helper is a deterministic pure function (no RNG, I/O, GPU, globals
//!   or `unsafe`), defends against degeneracy and never returns `NaN`/`inf`.
//!   Under total internal reflection the iris cannot be seen, so the sample
//!   reports it and leaves the iris lookup at the (un-parallaxed) centre with
//!   zero focusing gain.

pub mod cornea;
pub mod iris;
pub mod caustic;

pub use caustic::CausticParams;
pub use cornea::{CorneaInterface, CorneaSample};
pub use iris::{IrisGeometry, IrisSample, IrisStyle};

use bevy_math::{Vec2, Vec3};

/// Complete parameter set for [`shade_eye`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EyeParams {
    /// Refracting interface (air→cornea, or an effective air→aqueous surface).
    pub cornea: CorneaInterface,
    /// Anterior-chamber depth and iris radius.
    pub geometry: IrisGeometry,
    /// Pupil rest radius, iris edge and limbal styling (UV units).
    pub style: IrisStyle,
    /// Corneal focusing-gain tuning.
    pub caustic: CausticParams,
}

impl Default for EyeParams {
    #[inline]
    fn default() -> Self {
        Self {
            // The effective air→aqueous surface sends the view ray straight to
            // the iris, matching the caustic's destination medium.
            cornea: CorneaInterface::effective(),
            geometry: IrisGeometry::default(),
            style: IrisStyle::default(),
            caustic: CausticParams::default(),
        }
    }
}

/// Everything the shader needs for one eye pixel.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EyeShadeSample {
    /// Exact dielectric Fresnel reflectance of the cornea — the specular/wet
    /// highlight weight in `[0, 1]`.
    pub cornea_reflectance: f32,
    /// Iris texture coordinate (parallax- and pupil-corrected).
    pub iris_uv: Vec2,
    /// Limbal darkening multiplier in `[0, 1]`.
    pub limbal: f32,
    /// Corneal focusing gain multiplier (`≥ 1`) for the iris radiance.
    pub caustic_gain: f32,
    /// `true` when the view ray was totally internally reflected (iris hidden).
    pub total_internal_reflection: bool,
}

/// Shades one eye sample end to end.
///
/// `view` points away from the surface toward the camera, `normal` is the
/// outward corneal normal (both in the eye-local frame), `center_uv` is the
/// iris centre in texture space (usually `Vec2::splat(0.5)`) and `pupil_now`
/// the live pupil radius in UV units.
///
/// The pipeline refracts the view ray at the cornea, projects it onto the iris
/// plane with parallax, re-maps the radius for pupil dilation, computes limbal
/// darkening and the corneal focusing gain.  On total internal reflection the
/// iris is unreachable, so the iris UV is left at `center_uv`, the gain is `1`
/// and `total_internal_reflection` is set.
#[inline]
pub fn shade_eye(
    view: Vec3,
    normal: Vec3,
    center_uv: Vec2,
    pupil_now: f32,
    params: EyeParams,
) -> EyeShadeSample {
    let surface = cornea::shade(view, normal, params.cornea);

    match surface.refracted {
        Some(refracted) => {
            let iris = iris::sample_iris(
                center_uv,
                refracted,
                pupil_now,
                params.geometry,
                params.style,
            );
            let caustic_gain =
                caustic::corneal_focus_gain(surface.cos_incidence, iris.radius_norm, params.caustic);
            EyeShadeSample {
                cornea_reflectance: surface.reflectance,
                iris_uv: iris.uv,
                limbal: iris.limbal,
                caustic_gain,
                total_internal_reflection: false,
            }
        }
        None => EyeShadeSample {
            cornea_reflectance: surface.reflectance,
            iris_uv: center_uv,
            limbal: iris::limbal_darkening(0.0, params.style.limbal_start, params.style.limbal_strength),
            caustic_gain: 1.0,
            total_internal_reflection: true,
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn head_on_eye_samples_iris_center() {
        // Looking straight down the axis: no parallax, iris centre, strong
        // focusing, weak Fresnel.
        let s = shade_eye(Vec3::Z, Vec3::Z, Vec2::splat(0.5), 0.2, EyeParams::default());
        assert!(!s.total_internal_reflection);
        assert!((s.iris_uv - Vec2::splat(0.5)).length() < 1e-5, "uv={:?}", s.iris_uv);
        assert!(s.caustic_gain > 1.0, "gain={}", s.caustic_gain);
        assert!(s.cornea_reflectance < 0.1, "F={}", s.cornea_reflectance);
        assert!((s.limbal - 1.0).abs() < 1e-6);
    }

    #[test]
    fn oblique_view_shifts_iris_and_raises_fresnel() {
        let head_on = shade_eye(Vec3::Z, Vec3::Z, Vec2::splat(0.5), 0.2, EyeParams::default());
        let view = Vec3::new(0.5, 0.0, 0.866).normalize();
        let oblique = shade_eye(view, Vec3::Z, Vec2::splat(0.5), 0.2, EyeParams::default());

        assert!(!oblique.total_internal_reflection);
        // Parallax moves the iris sample off centre.
        let shift = (oblique.iris_uv - Vec2::splat(0.5)).length();
        assert!(shift > 1e-3, "no parallax shift: {shift}");
        // Grazing boosts the wet-surface reflectance.
        assert!(
            oblique.cornea_reflectance > head_on.cornea_reflectance,
            "oblique F {} should exceed head-on {}",
            oblique.cornea_reflectance,
            head_on.cornea_reflectance
        );
    }

    #[test]
    fn all_outputs_finite_over_a_sweep() {
        let params = EyeParams::default();
        for i in 0..=16 {
            let a = i as f32 / 16.0 * core::f32::consts::FRAC_PI_2;
            let view = Vec3::new(bevy_math::ops::sin(a), 0.0, bevy_math::ops::cos(a)).normalize();
            for p in [0.08f32, 0.14, 0.25, 0.4] {
                let s = shade_eye(view, Vec3::Z, Vec2::splat(0.5), p, params);
                assert!(s.cornea_reflectance.is_finite());
                assert!((0.0..=1.0).contains(&s.cornea_reflectance));
                assert!(s.iris_uv.is_finite());
                assert!((0.0..=1.0).contains(&s.limbal));
                assert!(s.caustic_gain.is_finite() && s.caustic_gain >= 1.0 - 1e-4);
            }
        }
    }

    #[test]
    fn pupil_dilation_moves_interior_samples() {
        // A sample inside the pupil region should shift when the pupil dilates.
        let view = Vec3::new(0.1, 0.0, 0.995).normalize();
        let narrow = shade_eye(view, Vec3::Z, Vec2::splat(0.5), 0.1, EyeParams::default());
        let wide = shade_eye(view, Vec3::Z, Vec2::splat(0.5), 0.3, EyeParams::default());
        let d = (narrow.iris_uv - wide.iris_uv).length();
        assert!(d > 1e-4, "pupil dilation had no effect: {d}");
    }

    #[test]
    fn degenerate_view_is_handled() {
        let s = shade_eye(Vec3::ZERO, Vec3::ZERO, Vec2::splat(0.5), 0.2, EyeParams::default());
        assert!(s.cornea_reflectance.is_finite());
        assert!(s.iris_uv.is_finite());
        assert!(s.caustic_gain.is_finite());
    }

    #[test]
    fn total_internal_reflection_hides_iris() {
        // Force a dense->rare interface so grazing view triggers TIR.
        let mut params = EyeParams::default();
        params.cornea = CorneaInterface::new(cornea::IOR_CORNEA, cornea::IOR_AIR);
        // Very grazing view.
        let view = Vec3::new(0.999, 0.0, 0.0447).normalize();
        let s = shade_eye(view, Vec3::Z, Vec2::splat(0.5), 0.2, params);
        assert!(s.total_internal_reflection);
        assert!((s.iris_uv - Vec2::splat(0.5)).length() < 1e-6);
        assert!((s.caustic_gain - 1.0).abs() < 1e-6);
        assert!((s.cornea_reflectance - 1.0).abs() < 1e-6);
    }
}
