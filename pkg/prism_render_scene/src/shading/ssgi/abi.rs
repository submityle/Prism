//! ABI shared between the SSGI compute pass and `shaders/ssgi.wesl`.
//!
//! Screen-space global illumination reuses the reflection subsystem's rebuilt
//! inputs — the reverse-Z Hi-Z depth pyramid, the packed view-space
//! `normal_roughness`, and the current-frame colour pyramid — so this slice
//! only introduces the per-dispatch [`GpuSsgiConfig`] immediate block the
//! diffuse-hemisphere trace consumes. The GPU pass, its resources and the
//! resolve consumption land in following slices so every committed ABI struct
//! has a live consumer, matching the SSR/GTAO precedent.
//!
//! Every field mirrors the shader's `SsgiConfig` byte-for-byte so machines with
//! and without a GPU agree with the CPU golden in
//! [`prism_render_shading::screen_space::gi`].

use bevy_math::{Mat4, UVec2};
use bytemuck::{Pod, Zeroable};

/// Compute workgroup edge the SSGI trace and composite dispatch in, matching
/// the `@workgroup_size(8, 8)` in `ssgi.wesl` / `ssgi_composite.wesl`. One
/// invocation per pixel, one workgroup per 8x8 tile.
pub(crate) const SSGI_WORKGROUP_SIZE: u32 = 8;

/// Immediate (push-constant) block consumed by `ssgi.wesl`'s `trace_ssgi`
/// entry point.
///
/// Mirrors the shader's `SsgiConfig`: the reverse-Z projection and its inverse
/// (used to reconstruct the view-space position each hemisphere ray starts
/// from), the framebuffer extent, the near-plane distance and march length, the
/// thin-surface `thickness` and iteration cap forwarded to the shared
/// hierarchical march, the finest refined mip, the artistic `intensity`, the
/// hit `distance_falloff`, and the per-pixel `sample_count`.
///
/// `screen_size` leads the scalar block so the `vec2<f32>` lands on its 8-byte
/// alignment with no implicit pad, and two trailing `u32`s round the struct up
/// to the 16-byte immediate alignment the `mat4x4` fields force.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable, PartialEq)]
pub(crate) struct GpuSsgiConfig {
    /// View -> clip (reverse-Z perspective).
    pub clip_from_view: [f32; 16],
    /// Clip -> view (inverse projection), used to reconstruct view positions.
    pub view_from_clip: [f32; 16],
    /// Full-resolution framebuffer extent in texels.
    pub screen_size: [f32; 2],
    /// Positive near-plane distance in front of the camera along -Z.
    pub near: f32,
    /// View-space hemisphere-ray march length.
    pub max_distance: f32,
    /// Device-depth thin-surface tolerance for accepting a hit.
    pub thickness: f32,
    /// Hard iteration cap for the hierarchical march.
    pub max_iterations: u32,
    /// Finest mip the march refines to (usually 0).
    pub most_detailed_mip: u32,
    /// Artistic gain applied to the gathered indirect radiance.
    pub intensity: f32,
    /// Normalized travel a hit fades out over, in `(0, 1]`; `1.0` disables it.
    pub distance_falloff: f32,
    /// Number of cosine-weighted hemisphere rays traced per pixel (>= 1).
    pub sample_count: u32,
    /// Padding to the 16-byte immediate alignment.
    pub _pad0: u32,
    /// Padding to the 16-byte immediate alignment.
    pub _pad1: u32,
}

impl GpuSsgiConfig {
    /// Builds the SSGI config from the view/projection matrices and framebuffer
    /// extent, folding in the golden march (`SsrMarchConfig`) and gather
    /// (`SsgiParams`) defaults. Both matrices upload column-major (via
    /// [`Mat4::to_cols_array`]) so the WGSL `mat4x4<f32>` multiply agrees
    /// byte-for-byte. `sample_count` is clamped to at least one ray so a zero
    /// request still traces a single hemisphere sample.
    pub(crate) fn from_view(
        clip_from_view: Mat4,
        view_from_clip: Mat4,
        near: f32,
        max_distance: f32,
        screen_size: UVec2,
        sample_count: u32,
    ) -> Self {
        Self {
            clip_from_view: clip_from_view.to_cols_array(),
            view_from_clip: view_from_clip.to_cols_array(),
            screen_size: [screen_size.x as f32, screen_size.y as f32],
            near,
            max_distance,
            // Golden `SsrMarchConfig` defaults (shared march).
            thickness: 0.02,
            max_iterations: 128,
            most_detailed_mip: 0,
            // Golden `SsgiParams` defaults.
            intensity: 1.0,
            distance_falloff: 1.0,
            sample_count: sample_count.max(1),
            _pad0: 0,
            _pad1: 0,
        }
    }
}

/// Immediate (push-constant) block consumed by `ssgi_composite.wesl`'s two
/// entry points.
///
/// Both the base copy and the energy-conserving GI fold only need the
/// framebuffer extent to bounds-check each invocation; the two trailing `u32`s
/// round the block up to the 16-byte immediate alignment, mirroring
/// `GpuSsrCompositeParams` byte-for-byte.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable, PartialEq)]
pub(crate) struct GpuSsgiCompositeParams {
    /// Framebuffer width in texels.
    pub width: u32,
    /// Framebuffer height in texels.
    pub height: u32,
    /// Padding to satisfy the 16-byte immediate alignment.
    pub _pad0: u32,
    /// Padding to satisfy the 16-byte immediate alignment.
    pub _pad1: u32,
}

impl GpuSsgiCompositeParams {
    /// Builds the composite params from the framebuffer extent.
    pub(crate) fn new(width: u32, height: u32) -> Self {
        Self {
            width,
            height,
            _pad0: 0,
            _pad1: 0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::mem::{align_of, size_of};

    #[test]
    fn config_matches_the_shader_immediate_layout() {
        // Two mat4x4 (128) + a vec2<f32> (8) + nine scalars (36) fill 172 bytes;
        // two u32 pads round the block up to 176 bytes, a multiple of the
        // 16-byte immediate alignment the mat4x4 fields force on the struct.
        assert_eq!(size_of::<GpuSsgiConfig>(), 176);
        assert_eq!(align_of::<GpuSsgiConfig>(), 4);
    }

    #[test]
    fn config_folds_in_the_golden_defaults_and_uploads_matrices_column_major() {
        let clip = Mat4::from_cols_array(&[
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
        ]);
        let inv = Mat4::from_cols_array(&[
            17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0,
            31.0, 32.0,
        ]);
        let config = GpuSsgiConfig::from_view(clip, inv, 0.5, 12.0, UVec2::new(1920, 1080), 8);
        assert_eq!(config.clip_from_view, clip.to_cols_array());
        assert_eq!(config.view_from_clip, inv.to_cols_array());
        assert_eq!(config.screen_size, [1920.0, 1080.0]);
        assert_eq!(config.near, 0.5);
        assert_eq!(config.max_distance, 12.0);
        // Golden `SsrMarchConfig` defaults.
        assert_eq!(config.thickness, 0.02);
        assert_eq!(config.max_iterations, 128);
        assert_eq!(config.most_detailed_mip, 0);
        // Golden `SsgiParams` defaults.
        assert_eq!(config.intensity, 1.0);
        assert_eq!(config.distance_falloff, 1.0);
        assert_eq!(config.sample_count, 8);
    }

    #[test]
    fn sample_count_is_clamped_to_at_least_one_ray() {
        let single = GpuSsgiConfig::from_view(
            Mat4::IDENTITY,
            Mat4::IDENTITY,
            0.1,
            8.0,
            UVec2::new(64, 64),
            0,
        );
        assert_eq!(single.sample_count, 1);
    }

    #[test]
    fn composite_params_match_the_shader_immediate_layout() {
        // Framebuffer extent (two u32) plus two padding u32 fill the 16-byte
        // immediate alignment, mirroring the SSR composite block.
        assert_eq!(size_of::<GpuSsgiCompositeParams>(), 16);
        assert_eq!(align_of::<GpuSsgiCompositeParams>(), 4);
        let params = GpuSsgiCompositeParams::new(1920, 1080);
        assert_eq!(params.width, 1920);
        assert_eq!(params.height, 1080);
    }

    #[test]
    fn workgroup_edge_matches_the_shader() {
        assert_eq!(SSGI_WORKGROUP_SIZE, 8);
    }
}
