//! Render-world resource gating the film-grain pass and feeding the golden
//! tunables into the [`GpuFilmGrainParams`] immediate block.
//!
//! Folds the golden [`prism_render_shading::FilmGrainParams`] defaults so the
//! render-world resource and the CPU reference stay in lockstep, with two
//! device-side additions the golden does not model: a master `enabled` gate and
//! a monotonically advancing `frame` counter that seeds the per-frame grain
//! animation.
//!
//! The name is prefixed `Prism` to keep the render-world resource distinct from
//! the golden's own `FilmGrainParams`; a game can overwrite the resource to
//! retune globally without touching any pass code, mirroring
//! [`super::super::vignette`]'s `PrismVignetteSettings`.

use bevy_ecs::prelude::Resource;
use bevy_math::UVec2;
use prism_render_shading::FilmGrainParams;

use super::abi::GpuFilmGrainParams;

/// Global film-grain settings consumed by the film-grain pass.
///
/// Disabled by default (the subsystem is opt-in; when `false` the pass
/// allocates nothing and dispatches nothing). Unlike the golden's neutral
/// `intensity = 0`, the render-world resource ships a subtle but visible default
/// so flipping `enabled` on is immediately meaningful; set `intensity = 0` for
/// the golden identity.
#[derive(Resource, Clone, Copy, Debug, PartialEq)]
pub(crate) struct PrismFilmGrainSettings {
    /// Master enable; when `false` the pass allocates nothing and dispatches
    /// nothing.
    pub enabled: bool,
    /// Artist grain strength (golden `intensity`).
    pub intensity: f32,
    /// Luminance response in `[0, 1]` (golden `response`).
    pub response: f32,
    /// Grain cell size (golden `size`).
    pub size: f32,
    /// Coloured (`true`) vs monochrome (`false`) grain (golden `colored`).
    pub colored: bool,
    /// Monotonically advancing frame index (device-only), advanced once per
    /// frame by the prepare system and folded into the per-frame animation seed.
    /// Wraps at [`Self::FRAME_SEED_PERIOD`] to keep the `f32` seed small so the
    /// hash keeps full precision.
    pub frame: u32,
}

impl PrismFilmGrainSettings {
    /// Period the `frame` counter wraps at before it is cast to the `f32`
    /// animation seed. Kept well below the `f32` integer-exact range so the
    /// hash coordinate never loses precision as the app runs for hours.
    pub(crate) const FRAME_SEED_PERIOD: u32 = 4096;
}

impl Default for PrismFilmGrainSettings {
    fn default() -> Self {
        // Fold the golden `FilmGrainParams` defaults so the render-world resource
        // and the CPU reference stay in lockstep for every field the golden
        // models; `intensity` is the one deliberate override (see below).
        let golden = FilmGrainParams::default();
        Self {
            // Opt-in: the subsystem allocates and dispatches nothing until the
            // host flips this on.
            enabled: false,
            // Device-visible default: the golden's neutral intensity is 0
            // (identity), but a render-world default of 0 would make enabling the
            // pass a no-op, so ship a subtle grain. Set to 0 for the golden
            // identity.
            intensity: 0.08,
            response: golden.response,
            size: golden.size,
            colored: golden.colored,
            frame: 0,
        }
    }
}

impl PrismFilmGrainSettings {
    /// Builds the [`GpuFilmGrainParams`] immediate block for a framebuffer of
    /// the given extent from these settings, folding the current `frame` counter
    /// into the per-frame animation seed.
    pub(crate) fn params(&self, screen_size: UVec2) -> GpuFilmGrainParams {
        let time_seed = (self.frame % Self::FRAME_SEED_PERIOD) as f32;
        GpuFilmGrainParams::from_settings(screen_size, time_seed, self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_mirror_the_golden_film_grain_params() {
        let settings = PrismFilmGrainSettings::default();
        let golden = FilmGrainParams::default();
        assert_eq!(settings.response, golden.response);
        assert_eq!(settings.size, golden.size);
        assert_eq!(settings.colored, golden.colored);
        // Opt-in gate off by default.
        assert!(!settings.enabled);
    }

    #[test]
    fn device_default_is_visible_when_enabled() {
        let settings = PrismFilmGrainSettings::default();
        // A subtle but visible grain so enabling the pass is immediately
        // meaningful (the golden neutral intensity is 0).
        assert!(settings.intensity > 0.0);
    }

    #[test]
    fn params_folds_the_extent_and_wrapped_frame_seed() {
        let mut settings = PrismFilmGrainSettings::default();
        settings.frame = PrismFilmGrainSettings::FRAME_SEED_PERIOD + 3;
        let params = settings.params(UVec2::new(1280, 720));
        assert_eq!(params.screen_size, [1280, 720]);
        // The seed wraps at FRAME_SEED_PERIOD so it stays small and precise.
        assert_eq!(params.time_seed, 3.0);
        assert_eq!(params.intensity, settings.intensity);
    }
}
