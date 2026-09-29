//! ABI shared between the film-grain compute pass and `shaders/film_grain.wesl`.
//!
//! Like the other single-pass post effects, the film-grain pass carries one
//! immediate (push-constant) block, [`GpuFilmGrainParams`], mirroring the
//! shader's single `var<immediate>` global. Every field mirrors the shader
//! struct byte-for-byte so machines with and without a GPU agree with the CPU
//! golden in [`prism_render_shading::film_grain`].
//!
//! WGSL gives `vec2` an 8-byte alignment, so the block leads with the
//! `screen_size` `vec2<u32>` at offset 0, then five trailing 4-byte scalars
//! (`intensity`, `response`, `size`, `time_seed`, `colored`) and one `_padding`
//! word: 8 (`screen_size`) + 6 * 4 = 32 bytes, a multiple of the 8-byte
//! alignment the `vec2` forces, with no implicit padding beyond the explicit
//! trailing word.

use bevy_math::UVec2;
use bytemuck::{Pod, Zeroable};

use super::settings::PrismFilmGrainSettings;

/// Workgroup size (per axis) of the film-grain compute entry point.
///
/// Must match `@workgroup_size(N, N, 1)` in `film_grain.wesl`; the dispatch
/// rounds its target extent up to a multiple of this on both axes and the
/// shader bounds-checks every invocation.
pub(crate) const FILM_GRAIN_WORKGROUP_SIZE: u32 = 8;

/// Immediate (push-constant) block consumed by the `film_grain_main` entry
/// point.
///
/// Mirrors the shader's `GpuFilmGrainParams`: the artist controls (the golden
/// [`prism_render_shading::FilmGrainParams`] fields — `intensity`, `response`,
/// `size`, `colored`) plus the framebuffer extent and a per-frame `time_seed`
/// that animates the hash-noise field.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable, PartialEq)]
pub(crate) struct GpuFilmGrainParams {
    /// Full-resolution framebuffer extent in texels (`vec2<u32>`, 8-byte
    /// aligned). Leads the block for its alignment.
    pub screen_size: [u32; 2],
    /// Artist grain strength added to the radiance; `0` is the golden identity.
    pub intensity: f32,
    /// Luminance response in `[0, 1]`: `0` grains uniformly, `1` biases the
    /// grain fully toward the shadows.
    pub response: f32,
    /// Grain cell size; larger values sample a lower frequency (coarser grain).
    pub size: f32,
    /// Per-frame animation seed (a wrapped frame index cast to `f32`); offsets
    /// the hash so the grain field animates frame to frame.
    pub time_seed: f32,
    /// Coloured-grain flag: non-zero hashes each channel with an offset seed,
    /// zero shares one value across channels (monochrome).
    pub colored: u32,
    /// Padding to a multiple of the `vec2<u32>` 8-byte alignment.
    pub _padding: u32,
}

impl GpuFilmGrainParams {
    /// Builds the immediate block from the framebuffer extent, the per-frame
    /// animation seed and the live [`PrismFilmGrainSettings`].
    ///
    /// The artist controls are carried through verbatim so the on-device grain
    /// is the exact golden twin; `time_seed` animates the field per frame.
    pub(crate) fn from_settings(
        size: UVec2,
        time_seed: f32,
        settings: &PrismFilmGrainSettings,
    ) -> Self {
        Self {
            screen_size: [size.x, size.y],
            intensity: settings.intensity,
            response: settings.response,
            size: settings.size,
            time_seed,
            colored: u32::from(settings.colored),
            _padding: 0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn film_grain_params_is_the_32_byte_immediate_block() {
        // screen_size (8) + five trailing 4-byte scalars (20) + one padding word
        // (4) fill 32 bytes, a multiple of the 8-byte alignment the `vec2`
        // forces, with no implicit padding.
        assert_eq!(size_of::<GpuFilmGrainParams>(), 32);
        assert_eq!(align_of::<GpuFilmGrainParams>(), 4);
    }

    #[test]
    fn workgroup_constant_matches_the_shader() {
        assert_eq!(FILM_GRAIN_WORKGROUP_SIZE, 8);
    }

    #[test]
    fn from_settings_folds_the_extent_seed_and_artist_controls() {
        let settings = PrismFilmGrainSettings::default();
        let params = GpuFilmGrainParams::from_settings(UVec2::new(1920, 1080), 7.0, &settings);
        assert_eq!(params.screen_size, [1920, 1080]);
        assert_eq!(params.intensity, settings.intensity);
        assert_eq!(params.response, settings.response);
        assert_eq!(params.size, settings.size);
        assert_eq!(params.time_seed, 7.0);
        assert_eq!(params.colored, u32::from(settings.colored));
        assert_eq!(params._padding, 0);
    }
}
