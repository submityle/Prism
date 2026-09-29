//! ABI shared between the vignette compute pass and `shaders/vignette.wesl`.
//!
//! Like the other single-pass post effects, the vignette pass carries one
//! immediate (push-constant) block, [`GpuVignetteParams`], mirroring the
//! shader's single `var<immediate>` global. Every field mirrors the shader
//! struct byte-for-byte so machines with and without a GPU agree with the CPU
//! golden in [`prism_render_shading::vignette`].
//!
//! WGSL gives `vec2` an 8-byte alignment, so the block leads with the two
//! `vec2`s (`center`, `screen_size`) at offsets 0 and 8, then eight trailing
//! 4-byte scalars: 8 (center) + 8 (extent) + 8 * 4 (intensity, smoothness,
//! feather, roundness, `aspect_ratio`, `focal_ratio`, mode, scale) = 48 bytes, a
//! multiple of the 8-byte alignment the `vec2`s force, with no implicit padding.

use bevy_math::UVec2;
use bytemuck::{Pod, Zeroable};

use super::settings::PrismVignetteSettings;

/// Workgroup size (per axis) of the vignette compute entry point.
///
/// Must match `@workgroup_size(N, N, 1)` in `vignette.wesl`; the dispatch rounds
/// its target extent up to a multiple of this on both axes and the shader
/// bounds-checks every invocation.
pub(crate) const VIGNETTE_WORKGROUP_SIZE: u32 = 8;

/// Immediate (push-constant) block consumed by the `vignette_main` entry point.
///
/// Mirrors the shader's `GpuVignetteParams`: the artist controls (the golden
/// `VignetteParams` fields — `center`, `intensity`, `smoothness`, `feather`,
/// `roundness`, `aspect_ratio`, `focal_ratio`, `mode`) plus the framebuffer
/// extent and a global `scale` in `[0, 1]` that fades the whole effect toward
/// identity (`scale = 1` reproduces the golden factor exactly).
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable, PartialEq)]
pub(crate) struct GpuVignetteParams {
    /// Darkening origin (normalised `[0, 1]` screen coordinate) for the artistic
    /// model. Frame centre `(0.5, 0.5)` by default. `vec2` leads the block for
    /// its 8-byte alignment.
    pub center: [f32; 2],
    /// Full-resolution framebuffer extent in texels (`vec2<u32>`, 8-byte aligned).
    pub screen_size: [u32; 2],
    /// Artistic darkening strength; `0` is the golden identity.
    pub intensity: f32,
    /// Artistic transition width.
    pub smoothness: f32,
    /// Extra artistic transition softness, added to `smoothness`.
    pub feather: f32,
    /// Square (`0`, Chebyshev) to circular (`1`, Euclidean) shape blend.
    pub roundness: f32,
    /// Horizontal aspect correction for the artistic model.
    pub aspect_ratio: f32,
    /// Focal ratio for the natural `cos^4` model.
    pub focal_ratio: f32,
    /// Fall-off model: `0` = natural `cos^4`, `1` = artistic `smoothstep`.
    pub mode: u32,
    /// Global effect scale in `[0, 1]`; fades the darkening toward identity.
    /// `1` reproduces the golden factor, `0` disables the effect per pixel.
    pub scale: f32,
}

impl GpuVignetteParams {
    /// Builds the immediate block from the framebuffer extent and the live
    /// [`PrismVignetteSettings`].
    ///
    /// The `scale` is clamped to `[0, 1]` on device; the artist controls are
    /// carried through verbatim so the on-device factor is the exact golden
    /// twin (`scale = 1`).
    pub(crate) fn from_settings(size: UVec2, settings: &PrismVignetteSettings) -> Self {
        Self {
            center: settings.center,
            screen_size: [size.x, size.y],
            intensity: settings.intensity,
            smoothness: settings.smoothness,
            feather: settings.feather,
            roundness: settings.roundness,
            aspect_ratio: settings.aspect_ratio,
            focal_ratio: settings.focal_ratio,
            mode: settings.mode,
            scale: settings.scale,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vignette_params_is_the_48_byte_immediate_block() {
        // center (8) + extent (8) + eight trailing 4-byte scalars (32) fill 48
        // bytes, a multiple of the 8-byte alignment the `vec2`s force, with no
        // implicit padding.
        assert_eq!(size_of::<GpuVignetteParams>(), 48);
        assert_eq!(align_of::<GpuVignetteParams>(), 4);
    }

    #[test]
    fn workgroup_constant_matches_the_shader() {
        assert_eq!(VIGNETTE_WORKGROUP_SIZE, 8);
    }

    #[test]
    fn from_settings_folds_the_extent_and_artist_controls() {
        let settings = PrismVignetteSettings::default();
        let params = GpuVignetteParams::from_settings(UVec2::new(1920, 1080), &settings);
        assert_eq!(params.screen_size, [1920, 1080]);
        assert_eq!(params.center, settings.center);
        assert_eq!(params.intensity, settings.intensity);
        assert_eq!(params.smoothness, settings.smoothness);
        assert_eq!(params.feather, settings.feather);
        assert_eq!(params.roundness, settings.roundness);
        assert_eq!(params.aspect_ratio, settings.aspect_ratio);
        assert_eq!(params.focal_ratio, settings.focal_ratio);
        assert_eq!(params.mode, settings.mode);
        assert_eq!(params.scale, settings.scale);
    }
}
