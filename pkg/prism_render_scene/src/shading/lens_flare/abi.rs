//! ABI shared between the lens-flare compute pass and
//! `shaders/lens_flare.wesl`.
//!
//! Like the other single-pass post effects, the flare pass carries one
//! immediate (push-constant) block, [`GpuLensFlareParams`], mirroring the
//! shader's single `var<immediate>` global. Every field mirrors the shader
//! struct byte-for-byte so machines with and without a GPU agree with the CPU
//! golden in [`prism_render_shading::lens_flare`].
//!
//! The golden [`prism_render_shading::LensFlareParams`] is all scalars plus the
//! framebuffer extent, so the block leads with the `vec2<u32>` `screen_size`
//! (8-byte aligned) and trails the six artist scalars (`ghost_count`,
//! `intensity`, `threshold`, `dispersal`, `halo_width`, `distortion`):
//! `8 + 4 + 5 * 4 = 32` bytes, a multiple of 16 with no implicit padding. There
//! is no `vec3` control, so no lane packing is needed.

use bevy_math::UVec2;
use bytemuck::{Pod, Zeroable};

use super::settings::PrismLensFlareSettings;

/// Workgroup size (per axis) of the lens-flare compute entry point.
///
/// Must match `@workgroup_size(N, N, 1)` in `lens_flare.wesl`; the dispatch
/// rounds its target extent up to a multiple of this on both axes and the
/// shader bounds-checks every invocation.
pub(crate) const LENS_FLARE_WORKGROUP_SIZE: u32 = 8;

/// Immediate (push-constant) block consumed by the `lens_flare_main` entry
/// point.
///
/// Mirrors the shader's `GpuLensFlareParams`: the framebuffer extent leads for
/// its 8-byte alignment, then the golden `LensFlareParams` artist controls (the
/// `ghost_count` disc taps, the composite `intensity`, the bright-tail
/// `threshold`, the `dispersal` ghost spacing, the `halo_width` ring radius and
/// the `distortion` chromatic dispersion).
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable, PartialEq)]
pub(crate) struct GpuLensFlareParams {
    /// Full-resolution framebuffer extent in texels (`vec2<u32>`). Leads the
    /// block for its 8-byte alignment.
    pub screen_size: [u32; 2],
    /// Number of `ghost` discs sampled along the centre-facing vector (golden
    /// `ghost_count`); the GPU twin fixes the fan-out at four taps and masks
    /// beyond this count.
    pub ghost_count: u32,
    /// Composite blend weight of the accumulated flare over the scene (golden
    /// `intensity`); `0` leaves the scene untouched.
    pub intensity: f32,
    /// Luminance above which pixels contribute to the flare (golden
    /// `threshold`).
    pub threshold: f32,
    /// Spacing of the `ghost` chain toward the optical centre (golden
    /// `dispersal`).
    pub dispersal: f32,
    /// Radial offset of the `halo` ring (golden `halo_width`).
    pub halo_width: f32,
    /// Per-channel chromatic dispersion of the `ghost` discs (golden
    /// `distortion`).
    pub distortion: f32,
}

impl GpuLensFlareParams {
    /// Builds the immediate block from the framebuffer extent and the live
    /// [`PrismLensFlareSettings`].
    ///
    /// The artist controls are carried through verbatim so the on-device flare
    /// is the exact golden twin.
    pub(crate) fn from_settings(size: UVec2, settings: &PrismLensFlareSettings) -> Self {
        Self {
            screen_size: [size.x, size.y],
            ghost_count: settings.ghost_count,
            intensity: settings.intensity,
            threshold: settings.threshold,
            dispersal: settings.dispersal,
            halo_width: settings.halo_width,
            distortion: settings.distortion,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lens_flare_params_is_the_32_byte_immediate_block() {
        // The `vec2<u32>` extent (8) + the `u32` ghost count (4) + five `f32`
        // artist scalars (20) fill 32 bytes, a multiple of 16 with no implicit
        // padding.
        assert_eq!(size_of::<GpuLensFlareParams>(), 32);
        assert_eq!(align_of::<GpuLensFlareParams>(), 4);
    }

    #[test]
    fn workgroup_constant_matches_the_shader() {
        assert_eq!(LENS_FLARE_WORKGROUP_SIZE, 8);
    }

    #[test]
    fn from_settings_packs_the_extent_and_artist_controls() {
        let settings = PrismLensFlareSettings::default();
        let params = GpuLensFlareParams::from_settings(UVec2::new(1920, 1080), &settings);
        assert_eq!(params.screen_size, [1920, 1080]);
        assert_eq!(params.ghost_count, settings.ghost_count);
        assert_eq!(params.intensity, settings.intensity);
        assert_eq!(params.threshold, settings.threshold);
        assert_eq!(params.dispersal, settings.dispersal);
        assert_eq!(params.halo_width, settings.halo_width);
        assert_eq!(params.distortion, settings.distortion);
    }
}
