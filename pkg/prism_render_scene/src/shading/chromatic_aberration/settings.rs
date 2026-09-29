//! Frame-constant tunables for Prism's chromatic-aberration subsystem.
//!
//! [`PrismChromaticAberrationSettings`] is the single render-world resource the
//! aberration pass reads to gate the subsystem and drive the radial split. It
//! folds the shading golden
//! [`prism_render_shading::ChromaticAberrationParams`]
//! field-for-field and default-for-default, so the `center`, `intensity` and
//! `samples` the GPU twin (`chromatic_aberration.wesl`) uses are exactly the
//! ones the CPU golden was validated against.
//!
//! The name is prefixed `Prism` to keep the render-world resource distinct from
//! the golden's own `ChromaticAberrationParams`; a game can overwrite the
//! resource to retune globally without touching any pass code, mirroring
//! [`super::super::dof`]'s `PrismDofSettings`.

use bevy_ecs::prelude::Resource;
use bevy_math::UVec2;
use prism_render_shading::ChromaticAberrationParams;

use super::abi::GpuChromaticAberrationParams;

/// Global chromatic-aberration settings consumed by the aberration pass.
///
/// Disabled by default (matching the golden's neutral `intensity == 0`), so
/// bringing the subsystem online is an opt-in the host flips on this resource.
#[derive(Resource, Clone, Copy, Debug, PartialEq)]
pub(crate) struct PrismChromaticAberrationSettings {
    /// Master enable; when `false` the pass allocates nothing and dispatches
    /// nothing.
    pub enabled: bool,
    /// Split strength; `0` disables the effect (all channel offsets vanish)
    /// (golden `intensity`).
    pub intensity: f32,
    /// Optical centre in `uv` space; the radial split fans out from here
    /// (golden `center`).
    pub center: [f32; 2],
    /// Spectral tap count carried for parity with the golden; the shipped
    /// three-tap pass reads the red/green/blue endpoints (golden `samples`).
    pub samples: u32,
}

impl Default for PrismChromaticAberrationSettings {
    fn default() -> Self {
        // Fold the golden `ChromaticAberrationParams` defaults so the
        // render-world resource and the CPU reference stay in lockstep.
        let golden = ChromaticAberrationParams::default();
        Self {
            // Opt-in: the golden's identity split (`intensity == 0`) is the
            // disabled state, so the master enable defaults off to match.
            enabled: false,
            intensity: golden.intensity,
            center: golden.center,
            samples: golden.samples,
        }
    }
}

impl PrismChromaticAberrationSettings {
    /// Builds the [`GpuChromaticAberrationParams`] immediate block for a
    /// framebuffer of `screen_size` texels from these settings.
    #[must_use]
    pub(crate) fn params(&self, screen_size: UVec2) -> GpuChromaticAberrationParams {
        GpuChromaticAberrationParams::from_settings(screen_size, self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_mirror_the_golden_params() {
        let settings = PrismChromaticAberrationSettings::default();
        let golden = ChromaticAberrationParams::default();
        assert_eq!(settings.intensity, golden.intensity);
        assert_eq!(settings.center, golden.center);
        assert_eq!(settings.samples, golden.samples);
        // The subsystem is opt-in and the golden default split is the identity.
        assert!(!settings.enabled);
        assert_eq!(settings.intensity, 0.0);
    }

    #[test]
    fn params_carry_the_screen_size_and_optics() {
        let settings = PrismChromaticAberrationSettings {
            enabled: true,
            intensity: 0.2,
            center: [0.4, 0.6],
            samples: 3,
        };
        let params = settings.params(UVec2::new(1920, 1080));
        assert_eq!(params.screen_size, [1920, 1080]);
        assert_eq!(params.center, [0.4, 0.6]);
        assert_eq!(params.intensity, 0.2);
        assert_eq!(params.samples, 3);
    }
}
