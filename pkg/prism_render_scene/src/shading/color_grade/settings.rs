//! Render-world resource gating the colour-grade pass and feeding the golden
//! tunables into the [`GpuColorGradeParams`] immediate block.
//!
//! Folds the golden [`prism_render_shading::ColorGradeParams`] defaults so the
//! render-world resource and the CPU reference stay in lockstep, with two
//! device-side additions the golden does not model: a master `enabled` gate and
//! a global `scale` that fades the whole grade toward identity.
//!
//! The name is prefixed `Prism` to keep the render-world resource distinct from
//! the golden's own `ColorGradeParams`; a game can overwrite the resource to
//! retune globally without touching any pass code, mirroring
//! [`super::super::vignette`]'s `PrismVignetteSettings`.

use bevy_ecs::prelude::Resource;
use bevy_math::UVec2;
use prism_render_shading::ColorGradeParams;

use super::abi::GpuColorGradeParams;

/// Global colour-grade settings consumed by the grade pass.
///
/// Disabled by default (the subsystem is opt-in; when `false` the pass
/// allocates nothing and dispatches nothing). Every artist control ships the
/// golden neutral value, so the default grade is the identity to
/// floating-point tolerance; a host raises `enabled` and dials in the wheels.
#[derive(Resource, Clone, Copy, Debug, PartialEq)]
pub(crate) struct PrismColorGradeSettings {
    /// Master enable; when `false` the pass allocates nothing and dispatches
    /// nothing.
    pub enabled: bool,
    /// Normalised von Kries white-balance temperature (golden `temperature`).
    pub temperature: f32,
    /// Normalised von Kries white-balance tint (golden `tint`).
    pub tint: f32,
    /// ASC CDL lift (per-channel offset, golden `lift`).
    pub lift: [f32; 3],
    /// ASC CDL gamma (per-channel power, golden `gamma`).
    pub gamma: [f32; 3],
    /// ASC CDL gain (per-channel slope, golden `gain`).
    pub gain: [f32; 3],
    /// Linear contrast about `pivot` (golden `contrast`).
    pub contrast: f32,
    /// Contrast pivot / middle grey (golden `pivot`).
    pub pivot: f32,
    /// Luma-preserving saturation (golden `saturation`).
    pub saturation: f32,
    /// Global effect scale in `[0, 1]` (device-only); fades the grade toward the
    /// input. `1` reproduces the golden grade exactly.
    pub scale: f32,
}

impl Default for PrismColorGradeSettings {
    fn default() -> Self {
        // Fold the golden `ColorGradeParams` defaults so the render-world
        // resource and the CPU reference stay in lockstep. Every default is the
        // golden neutral, so the default grade is the identity; `enabled` is off
        // and `scale` is full-strength for when a host dials the wheels in.
        let golden = ColorGradeParams::default();
        Self {
            // Opt-in: the subsystem allocates and dispatches nothing until the
            // host flips this on.
            enabled: false,
            temperature: golden.temperature,
            tint: golden.tint,
            lift: golden.lift,
            gamma: golden.gamma,
            gain: golden.gain,
            contrast: golden.contrast,
            pivot: golden.pivot,
            saturation: golden.saturation,
            // Full-strength grade by default; scale = 1 is the exact golden twin.
            scale: 1.0,
        }
    }
}

impl PrismColorGradeSettings {
    /// Builds the [`GpuColorGradeParams`] immediate block for a framebuffer of
    /// the given extent from these settings.
    pub(crate) fn params(&self, screen_size: UVec2) -> GpuColorGradeParams {
        GpuColorGradeParams::from_settings(screen_size, self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_mirror_the_golden_color_grade_params() {
        let settings = PrismColorGradeSettings::default();
        let golden = ColorGradeParams::default();
        assert_eq!(settings.temperature, golden.temperature);
        assert_eq!(settings.tint, golden.tint);
        assert_eq!(settings.lift, golden.lift);
        assert_eq!(settings.gamma, golden.gamma);
        assert_eq!(settings.gain, golden.gain);
        assert_eq!(settings.contrast, golden.contrast);
        assert_eq!(settings.pivot, golden.pivot);
        assert_eq!(settings.saturation, golden.saturation);
        // Opt-in gate off by default.
        assert!(!settings.enabled);
        // Full strength; scale = 1 is the exact golden twin.
        assert_eq!(settings.scale, 1.0);
    }

    #[test]
    fn default_settings_are_a_golden_identity_grade() {
        // The device defaults are the golden neutral values, so grading any
        // pixel with them through the CPU golden returns it unchanged to
        // floating-point tolerance. This guards the device defaults against
        // drift from the golden's identity contract.
        let settings = PrismColorGradeSettings::default();
        let golden = ColorGradeParams {
            temperature: settings.temperature,
            tint: settings.tint,
            lift: settings.lift,
            gamma: settings.gamma,
            gain: settings.gain,
            contrast: settings.contrast,
            pivot: settings.pivot,
            saturation: settings.saturation,
        };
        for rgb in [[0.3_f32, 0.5, 0.8], [0.02, 0.4, 1.7], [0.5, 0.5, 0.5]] {
            let graded = prism_render_shading::apply_color_grade(rgb, &golden);
            for channel in 0..3 {
                assert!(
                    (graded[channel] - rgb[channel]).abs() < 1.0e-4,
                    "default grade must be identity: {} != {}",
                    graded[channel],
                    rgb[channel]
                );
            }
        }
    }

    #[test]
    fn params_packs_the_extent_and_controls() {
        let settings = PrismColorGradeSettings::default();
        let params = settings.params(UVec2::new(1280, 720));
        assert_eq!(params.screen_size, [1280, 720]);
        assert_eq!(params.pivot_sat_scale[0], settings.pivot);
        assert_eq!(params.pivot_sat_scale[1], settings.saturation);
        assert_eq!(params.pivot_sat_scale[2], settings.scale);
    }
}
