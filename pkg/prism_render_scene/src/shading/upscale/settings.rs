//! Scene-side resource driving the temporal-upscale subsystem.
//!
//! [`UpscaleSettings`] is the render-world mirror of the architecture contract
//! [`TemporalUpscaleSettings`]: the `render_scale` the current frame is drawn
//! at, the RCAS `sharpness` the finishing pass runs, the RCAS `denoise` toggle,
//! and the [`InvalidationMask`] of history-invalidation events this feature
//! depends on. The accumulation-pass tunables that the architecture contract
//! does not carry (depth tolerance, variance gamma, thin-feature contrast, min
//! alpha) travel in the golden [`UpscaleConfig`], stored alongside so the whole
//! pipeline is configured from one resource.
//!
//! It converts losslessly to and from [`TemporalUpscaleSettings`]
//! ([`UpscaleSettings::to_upscale_settings`] / [`From`]) and hands the RCAS pass
//! its [`RcasParams`] via [`UpscaleSettings::rcas_params`], so the ABI builders
//! in [`super::abi`] can be fed straight from this resource without the graph
//! wiring re-deriving any golden default.

use bevy_ecs::resource::Resource;
use prism_render_architecture::history::InvalidationMask;
use prism_render_architecture::temporal_upscale::TemporalUpscaleSettings;
use prism_render_shading::upscale::{RcasParams, UpscaleConfig};

/// Render-world temporal-upscale configuration, mapping the architecture
/// [`TemporalUpscaleSettings`] plus the golden accumulation-pass tunables.
#[derive(Resource, Clone, Copy, Debug, PartialEq)]
pub(crate) struct UpscaleSettings {
    /// render / display in `(0, 1]`; the fraction of the display resolution the
    /// scene is drawn at each frame. `1.0` is native (no upscaling).
    pub render_scale: f32,
    /// RCAS lobe scale in `[0, 1]`. `0.0` is an exact identity (no sharpening),
    /// `1.0` full-strength RCAS.
    pub sharpness: f32,
    /// Whether the RCAS pass attenuates its lobe by the noise estimate.
    pub denoise: bool,
    /// The history-invalidation events this feature resets its accumulation on.
    pub invalidation_dependencies: InvalidationMask,
    /// The accumulation-pass tunables not carried by the architecture contract.
    pub config: UpscaleConfig,
}

impl Default for UpscaleSettings {
    /// Matches the architecture [`TemporalUpscaleSettings::default`]
    /// (`render_scale = 1.0`, `sharpness = 0.0`, every dependency) and the
    /// golden [`UpscaleConfig::default`] / [`RcasParams::default`] (`denoise`
    /// off).
    fn default() -> Self {
        let settings = TemporalUpscaleSettings::default();
        Self {
            render_scale: settings.render_scale,
            sharpness: settings.sharpness,
            denoise: RcasParams::default().denoise,
            invalidation_dependencies: settings.invalidation_dependencies,
            config: UpscaleConfig::default(),
        }
    }
}

impl UpscaleSettings {
    /// Project back to the architecture [`TemporalUpscaleSettings`] contract
    /// (`render_scale`, `sharpness`, `invalidation_dependencies`).
    #[must_use]
    pub(crate) fn to_upscale_settings(self) -> TemporalUpscaleSettings {
        TemporalUpscaleSettings {
            render_scale: self.render_scale,
            sharpness: self.sharpness,
            invalidation_dependencies: self.invalidation_dependencies,
        }
    }

    /// The golden [`RcasParams`] for the finishing sharpen pass: the sharpness
    /// carried straight through (so `0.0` stays an exact identity) and the
    /// denoise toggle.
    #[must_use]
    pub(crate) fn rcas_params(&self) -> RcasParams {
        RcasParams {
            sharpness: self.sharpness,
            denoise: self.denoise,
        }
    }
}

impl From<TemporalUpscaleSettings> for UpscaleSettings {
    /// Adopt an architecture contract, keeping the golden [`UpscaleConfig`] and
    /// `denoise` defaults for the tunables it does not carry.
    fn from(settings: TemporalUpscaleSettings) -> Self {
        Self {
            render_scale: settings.render_scale,
            sharpness: settings.sharpness,
            denoise: RcasParams::default().denoise,
            invalidation_dependencies: settings.invalidation_dependencies,
            config: UpscaleConfig::default(),
        }
    }
}

impl From<UpscaleSettings> for TemporalUpscaleSettings {
    fn from(settings: UpscaleSettings) -> Self {
        settings.to_upscale_settings()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_matches_the_architecture_contract() {
        let ours = UpscaleSettings::default();
        let arch = TemporalUpscaleSettings::default();
        assert_eq!(ours.render_scale, arch.render_scale);
        assert_eq!(ours.sharpness, arch.sharpness);
        assert_eq!(
            ours.invalidation_dependencies,
            arch.invalidation_dependencies
        );
        // Native, identity sharpen, golden accumulation tunables.
        assert_eq!(ours.render_scale, 1.0);
        assert_eq!(ours.sharpness, 0.0);
        assert!(!ours.denoise);
        assert_eq!(ours.config, UpscaleConfig::default());
    }

    #[test]
    fn round_trips_through_the_architecture_contract() {
        let arch = TemporalUpscaleSettings {
            render_scale: 0.5,
            sharpness: 0.8,
            invalidation_dependencies: InvalidationMask::CAMERA_CUT,
        };
        let ours: UpscaleSettings = arch.into();
        assert_eq!(ours.render_scale, 0.5);
        assert_eq!(ours.sharpness, 0.8);
        assert_eq!(
            ours.invalidation_dependencies,
            InvalidationMask::CAMERA_CUT
        );

        let back: TemporalUpscaleSettings = ours.into();
        assert_eq!(back.render_scale, arch.render_scale);
        assert_eq!(back.sharpness, arch.sharpness);
        assert_eq!(
            back.invalidation_dependencies,
            arch.invalidation_dependencies
        );
    }

    #[test]
    fn rcas_params_carry_sharpness_and_denoise() {
        let mut settings = UpscaleSettings::default();
        // Default is the exact-identity sharpen.
        assert_eq!(settings.rcas_params(), RcasParams::default());

        settings.sharpness = 0.6;
        settings.denoise = true;
        let params = settings.rcas_params();
        assert_eq!(params.sharpness, 0.6);
        assert!(params.denoise);
    }
}
