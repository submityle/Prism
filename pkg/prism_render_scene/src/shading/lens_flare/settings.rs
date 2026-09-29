//! Render-world resource gating the lens-flare pass and feeding the golden
//! tunables into the [`GpuLensFlareParams`] immediate block.
//!
//! Folds the golden [`prism_render_shading::LensFlareParams`] defaults so the
//! render-world resource and the CPU reference stay in lockstep, with one
//! device-side addition the golden does not model: a master `enabled` gate.
//!
//! The name is prefixed `Prism` to keep the render-world resource distinct from
//! the golden's own `LensFlareParams`; a game can overwrite the resource to
//! retune globally without touching any pass code, mirroring
//! [`super::super::color_grade`]'s `PrismColorGradeSettings`.

use bevy_ecs::prelude::Resource;
use bevy_math::UVec2;
use prism_render_shading::LensFlareParams;

use super::abi::GpuLensFlareParams;

/// Global lens-flare settings consumed by the flare pass.
///
/// Disabled by default (the subsystem is opt-in; when `false` the pass
/// allocates nothing and dispatches nothing). Every artist control ships the
/// golden neutral value, so the default flare is the identity (the golden
/// default is `intensity == 0`, which leaves the scene untouched); a host
/// raises `enabled` and dials in the ghost/halo controls.
#[derive(Resource, Clone, Copy, Debug, PartialEq)]
pub(crate) struct PrismLensFlareSettings {
    /// Master enable; when `false` the pass allocates nothing and dispatches
    /// nothing.
    pub enabled: bool,
    /// Composite blend weight of the accumulated flare over the scene (golden
    /// `intensity`); `0` disables the pass.
    pub intensity: f32,
    /// Luminance above which pixels contribute to the flare (golden
    /// `threshold`).
    pub threshold: f32,
    /// Number of `ghost` discs sampled along the centre-facing vector (golden
    /// `ghost_count`).
    pub ghost_count: u32,
    /// Spacing of the `ghost` chain toward the optical centre (golden
    /// `dispersal`).
    pub dispersal: f32,
    /// Radial offset of the `halo` ring (golden `halo_width`).
    pub halo_width: f32,
    /// Per-channel chromatic dispersion of the `ghost` discs (golden
    /// `distortion`).
    pub distortion: f32,
}

impl Default for PrismLensFlareSettings {
    fn default() -> Self {
        // Fold the golden `LensFlareParams` defaults so the render-world
        // resource and the CPU reference stay in lockstep. Every default is the
        // golden neutral (all zero), so the default flare is the identity;
        // `enabled` is off for when a host dials the ghost/halo controls in.
        let golden = LensFlareParams::default();
        Self {
            // Opt-in: the subsystem allocates and dispatches nothing until the
            // host flips this on.
            enabled: false,
            intensity: golden.intensity,
            threshold: golden.threshold,
            ghost_count: golden.ghost_count,
            dispersal: golden.dispersal,
            halo_width: golden.halo_width,
            distortion: golden.distortion,
        }
    }
}

impl PrismLensFlareSettings {
    /// Builds the [`GpuLensFlareParams`] immediate block for a framebuffer of
    /// the given extent from these settings.
    pub(crate) fn params(&self, screen_size: UVec2) -> GpuLensFlareParams {
        GpuLensFlareParams::from_settings(screen_size, self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_mirror_the_golden_lens_flare_params() {
        let settings = PrismLensFlareSettings::default();
        let golden = LensFlareParams::default();
        assert_eq!(settings.intensity, golden.intensity);
        assert_eq!(settings.threshold, golden.threshold);
        assert_eq!(settings.ghost_count, golden.ghost_count);
        assert_eq!(settings.dispersal, golden.dispersal);
        assert_eq!(settings.halo_width, golden.halo_width);
        assert_eq!(settings.distortion, golden.distortion);
        // Opt-in gate off by default.
        assert!(!settings.enabled);
    }

    #[test]
    fn default_settings_are_a_golden_identity_flare() {
        // The golden default is a disabled flare: `intensity == 0` leaves the
        // scene untouched regardless of the accumulated flare. This guards the
        // device defaults against drift from the golden's identity contract.
        let settings = PrismLensFlareSettings::default();
        let golden = LensFlareParams {
            intensity: settings.intensity,
            threshold: settings.threshold,
            ghost_count: settings.ghost_count,
            dispersal: settings.dispersal,
            halo_width: settings.halo_width,
            distortion: settings.distortion,
        };
        for scene in [[0.3_f32, 0.5, 0.8], [0.02, 0.4, 1.7], [0.5, 0.5, 0.5]] {
            let composited =
                prism_render_shading::apply_lens_flare(scene, [9.0, 9.0, 9.0], golden.intensity);
            for channel in 0..3 {
                assert!(
                    (composited[channel] - scene[channel]).abs() < 1.0e-4,
                    "default flare must be identity: {} != {}",
                    composited[channel],
                    scene[channel]
                );
            }
        }
    }

    #[test]
    fn params_packs_the_extent_and_controls() {
        let settings = PrismLensFlareSettings::default();
        let params = settings.params(UVec2::new(1280, 720));
        assert_eq!(params.screen_size, [1280, 720]);
        assert_eq!(params.ghost_count, settings.ghost_count);
        assert_eq!(params.intensity, settings.intensity);
        assert_eq!(params.distortion, settings.distortion);
    }
}
