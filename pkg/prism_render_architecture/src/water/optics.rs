//! Deterministic water-optics planning that fuses spectral refraction with
//! underwater transport.
//!
//! This module composes the dispersion and underwater primitives into a single
//! pure planning step. Given a static optical [`OpticsProfile`] and per-view
//! [`OpticsInputs`], [`plan_optics`] produces an [`OpticsPlan`] describing the
//! per-channel index of refraction (`IOR`), screen-space dispersion offsets,
//! depth-attenuated color, single-scatter phase, godray inscatter, a
//! multiple-scatter boost, and a coarse visibility flag.
//!
//! Everything here is side-effect free and reproducible so that the same inputs
//! always yield the same plan, which keeps `GPU` uploads stable across frames.
//! The refraction stage relies on the `Cauchy` law and `Snell`-style transmitted
//! sines from the dispersion module, while the transport stage uses
//! `Beer-Lambert` transmittance and the `Henyey-Greenstein` phase function from
//! the underwater module.

use super::dispersion::{dispersion_offsets, spectral_iors, RgbIor};
use super::underwater::{
    beer_lambert_transmittance, depth_color_shift, godray_inscatter, henyey_greenstein, is_visible,
    multiple_scatter_boost, RgbColor, RgbExtinction,
};

/// Static optical description of a body of water.
///
/// The `Cauchy` coefficients drive spectral `IOR`, the extinction triple drives
/// per-channel `Beer-Lambert` attenuation, and the remaining scalars tune the
/// scattering phase and visibility behavior.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct OpticsProfile {
    /// `Cauchy` law constant term `A` (baseline `IOR`).
    pub cauchy_a: f32,
    /// `Cauchy` law dispersion term `B` (wavelength spread); `b > 0` yields
    /// `r < g < b` ordering across the `RGB` channels.
    pub cauchy_b: f32,
    /// Scale applied to screen-space dispersion offsets.
    pub refraction_strength: f32,
    /// Per-channel `Beer-Lambert` extinction coefficients.
    pub extinction: RgbExtinction,
    /// Single-scattering albedo used by inscatter and the multiple-scatter
    /// boost.
    pub scatter_albedo: f32,
    /// `Henyey-Greenstein` asymmetry parameter `g` in `(-1, 1)`.
    pub asymmetry_g: f32,
    /// Transmittance threshold below which geometry is considered invisible.
    pub visibility_threshold: f32,
}

/// Per-view inputs sampled each frame for a single optics evaluation.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct OpticsInputs {
    /// Sine of the incidence angle at the water interface.
    pub sin_incidence: f32,
    /// View-ray depth through the medium, in meters.
    pub view_depth: f32,
    /// Surface `RGB` color prior to depth attenuation.
    pub surface_color: RgbColor,
    /// Incident surface light intensity.
    pub surface_light: f32,
    /// Cosine of the scattering angle for the phase function.
    pub cos_scatter: f32,
    /// Length of the light shaft used for godray inscatter.
    pub shaft_length: f32,
}

/// Fully resolved, deterministic optics plan ready for shading upload.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct OpticsPlan {
    /// Per-channel `IOR` from the `Cauchy` law.
    pub iors: RgbIor,
    /// Non-negative per-channel screen-space refraction offsets.
    pub dispersion_offsets: [f32; 3],
    /// Depth-attenuated surface color after `Beer-Lambert` shift.
    pub depth_color: RgbColor,
    /// `Henyey-Greenstein` phase value for the scatter angle.
    pub phase: f32,
    /// Godray inscatter contribution along the shaft.
    pub inscatter: f32,
    /// Multiple-scatter boosted single-scatter estimate.
    pub scatter_boost: f32,
    /// Whether geometry at `view_depth` remains visible.
    pub visible: bool,
}

/// Compose a full [`OpticsPlan`] from a profile and per-view inputs.
///
/// The green channel (`extinction.g`) is used as the representative
/// single-channel extinction for the scalar transport functions. The result is
/// pure and deterministic: identical arguments always produce identical plans.
#[must_use]
pub fn plan_optics(profile: OpticsProfile, inputs: OpticsInputs) -> OpticsPlan {
    let iors = spectral_iors(profile.cauchy_a, profile.cauchy_b);
    let dispersion_offsets =
        dispersion_offsets(iors, inputs.sin_incidence, profile.refraction_strength);
    let depth_color =
        depth_color_shift(inputs.surface_color, profile.extinction, inputs.view_depth);
    let phase = henyey_greenstein(inputs.cos_scatter, profile.asymmetry_g);
    let inscatter = godray_inscatter(
        inputs.surface_light,
        profile.scatter_albedo,
        profile.extinction.g,
        inputs.shaft_length,
    );
    let single = beer_lambert_transmittance(profile.extinction.g, inputs.view_depth)
        * inputs.surface_light.max(0.0)
        * phase;
    let scatter_boost = multiple_scatter_boost(single, profile.scatter_albedo);
    let visible = is_visible(
        profile.extinction.g,
        inputs.view_depth,
        profile.visibility_threshold,
    );

    OpticsPlan {
        iors,
        dispersion_offsets,
        depth_color,
        phase,
        inscatter,
        scatter_boost,
        visible,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture_profile() -> OpticsProfile {
        OpticsProfile {
            cauchy_a: 1.324,
            cauchy_b: 0.0032,
            refraction_strength: 0.05,
            extinction: RgbExtinction {
                r: 0.45,
                g: 0.15,
                b: 0.08,
            },
            scatter_albedo: 0.7,
            asymmetry_g: 0.6,
            visibility_threshold: 0.02,
        }
    }

    fn fixture_inputs() -> OpticsInputs {
        OpticsInputs {
            sin_incidence: 0.7,
            view_depth: 3.0,
            surface_color: RgbColor {
                r: 1.0,
                g: 1.0,
                b: 1.0,
            },
            surface_light: 1.0,
            cos_scatter: 0.5,
            shaft_length: 5.0,
        }
    }

    #[test]
    fn channels_ordered_red_below_green_below_blue() {
        let plan = plan_optics(fixture_profile(), fixture_inputs());
        assert!(plan.iors.r < plan.iors.g, "expected r < g for spectral IOR");
        assert!(plan.iors.g < plan.iors.b, "expected g < b for spectral IOR");
    }

    #[test]
    fn depth_color_attenuates_and_orders_channels() {
        let inputs = fixture_inputs();
        let plan = plan_optics(fixture_profile(), inputs);

        // Each channel cannot exceed the incoming surface color.
        assert!(plan.depth_color.r <= inputs.surface_color.r);
        assert!(plan.depth_color.g <= inputs.surface_color.g);
        assert!(plan.depth_color.b <= inputs.surface_color.b);

        // Red is extinguished fastest, so red < green < blue after the shift.
        assert!(plan.depth_color.r < plan.depth_color.g);
        assert!(plan.depth_color.g < plan.depth_color.b);
    }

    #[test]
    fn transmittance_stays_within_unit_bounds() {
        let profile = fixture_profile();
        let inputs = fixture_inputs();
        let t = beer_lambert_transmittance(profile.extinction.g, inputs.view_depth);
        assert!((0.0..=1.0).contains(&t), "transmittance out of range: {t}");
    }

    #[test]
    fn transmittance_decays_monotonically_with_depth() {
        let profile = fixture_profile();
        let mut prev = beer_lambert_transmittance(profile.extinction.g, 0.0);
        let mut depth = 0.5;
        while depth <= 20.0 {
            let cur = beer_lambert_transmittance(profile.extinction.g, depth);
            assert!(cur <= prev, "transmittance increased at depth {depth}");
            prev = cur;
            depth += 0.5;
        }
    }

    #[test]
    fn inscatter_rises_with_shaft_length() {
        let profile = fixture_profile();
        let mut inputs = fixture_inputs();
        inputs.shaft_length = 1.0;
        let short = plan_optics(profile, inputs).inscatter;
        inputs.shaft_length = 8.0;
        let long = plan_optics(profile, inputs).inscatter;
        assert!(long > short, "expected inscatter to grow with shaft length");
    }

    #[test]
    fn scatter_boost_is_at_least_single_scatter() {
        let profile = fixture_profile();
        let inputs = fixture_inputs();
        let plan = plan_optics(profile, inputs);
        let phase = henyey_greenstein(inputs.cos_scatter, profile.asymmetry_g);
        let single = beer_lambert_transmittance(profile.extinction.g, inputs.view_depth)
            * inputs.surface_light.max(0.0)
            * phase;
        assert!(
            plan.scatter_boost >= single,
            "scatter boost {} below single scatter {single}",
            plan.scatter_boost
        );
    }

    #[test]
    fn visibility_flips_with_depth() {
        let profile = fixture_profile();
        let mut inputs = fixture_inputs();
        inputs.view_depth = 3.0;
        assert!(
            plan_optics(profile, inputs).visible,
            "expected visible at shallow depth"
        );
        inputs.view_depth = 1000.0;
        assert!(
            !plan_optics(profile, inputs).visible,
            "expected invisible at large depth"
        );
    }

    #[test]
    fn plan_is_deterministic() {
        let profile = fixture_profile();
        let inputs = fixture_inputs();
        let first = plan_optics(profile, inputs);
        let second = plan_optics(profile, inputs);
        assert_eq!(first, second, "plan_optics must be deterministic");
    }
}
