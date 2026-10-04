//! ABI shared between the specular-GI *temporal denoise* passes
//! (`shaders/spec_denoise_reproject.wesl` + `shaders/spec_denoise_history_clamp.wesl`)
//! and their host-side bind-group uploads.
//!
//! The temporal path is the two-pass ReBLUR/ReLAX-style accumulator that sits
//! between the `spec_gi` reuse resolve and the spatial pre-filter:
//!
//! * **reproject** rebuilds each pixel's world-space surface from the SSR
//!   prepass reverse-Z depth, follows the specular *virtual* reflection point
//!   back into the previous frame, and samples the prior accumulated history
//!   (radiance + age), metadata (normal + roughness) and dual-rate luminance
//!   EMAs under a world-space disocclusion guard, emitting the reprojected
//!   planes the clamp consumes;
//! * **history-clamp** fuses that reprojected history with the current-frame
//!   noisy `spec_gi` resolve under an AABB colour clamp (anti-ghosting) and a
//!   normal/roughness consistency gate, advancing the age + luminance EMAs and
//!   writing both the next-frame history planes and the denoised specular the
//!   spatial pass filters.
//!
//! Each pass reads a single `#[repr(C)]` uniform record from `@group(0)
//! @binding(0)`; the two structs below mirror their WESL twins
//! (`SpecDenoiseReprojectConfig` / `SpecDenoiseHistoryClampConfig`)
//! byte-for-byte, and the `size_of`/`offset_of` contract tests guard the layout
//! against drift. The matrices upload column-major (via [`Mat4::to_cols_array`])
//! so the WGSL `mat4x4<f32>` multiplies agree, and the `vec3 + f32` camera rows
//! pack exactly as WGSL's 16-byte-aligned `vec3<f32>` followed by a scalar.

use bevy_math::{Mat4, Vec3};
use bytemuck::{Pod, Zeroable};

/// Workgroup size (per axis) of both temporal compute entry points.
///
/// Must match `@workgroup_size(N, N, 1)` in `spec_denoise_reproject.wesl` and
/// `spec_denoise_history_clamp.wesl`; the dispatch rounds its target extent up
/// to a multiple of this on both axes and both kernels bounds-check every
/// invocation.
pub(crate) const SPEC_DENOISE_TEMPORAL_WORKGROUP_SIZE: u32 = 8;

/// Uniform block consumed by `spec_denoise_reproject.wesl` (one per dispatch).
///
/// Mirrors the shader's `SpecDenoiseReprojectConfig`: four `mat4x4` (256 bytes)
/// followed by two `vec3 + f32` camera rows (32 bytes) and a trailing row of
/// four scalars (16 bytes) = 304 bytes. The matrices upload column-major; the
/// `curr_cam_pos`/`prev_cam_pos` triples pack with their trailing scalar exactly
/// as WGSL packs a 16-byte-aligned `vec3<f32>` ahead of an `f32`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable, PartialEq)]
pub(crate) struct GpuSpecDenoiseReprojectConfig {
    /// Current camera clip (NDC, reverse-Z) -> world, to rebuild the surface.
    pub world_from_clip: [f32; 16],
    /// Previous camera world -> clip (NDC), to find the history texel.
    pub prev_clip_from_world: [f32; 16],
    /// Previous camera clip (NDC) -> world, to rebuild the stored prev surface
    /// for the world-space disocclusion guard.
    pub prev_world_from_clip: [f32; 16],
    /// Current camera view -> world rotation, lifting the view-space G-buffer
    /// normal into the world space the reprojection maths runs in.
    pub world_from_view: [f32; 16],
    /// Current camera world-space position (parallax origin).
    pub curr_cam_pos: [f32; 3],
    /// Virtual-reflection parallax sensitivity (`SdrParams::parallax_sensitivity`).
    pub parallax_sensitivity: f32,
    /// Previous camera world-space position (parallax origin last frame).
    pub prev_cam_pos: [f32; 3],
    /// Virtual-reflection depth exponent (`SdrParams::virtual_exponent`).
    pub virtual_exponent: f32,
    /// Framebuffer width in texels (invocations round up / bounds-check).
    pub width: u32,
    /// Framebuffer height in texels.
    pub height: u32,
    /// Minimum reprojection confidence floor (`SdrParams::min_confidence`).
    pub min_confidence: f32,
    /// Relative world-space disocclusion tolerance (fraction of view distance).
    pub depth_rejection: f32,
}

impl GpuSpecDenoiseReprojectConfig {
    /// Builds the reprojection config from the current/previous camera
    /// transforms, parallax tunables and framebuffer extent. All matrices
    /// upload column-major so the WGSL `mat4x4<f32>` multiplies agree
    /// byte-for-byte.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        world_from_clip: Mat4,
        prev_clip_from_world: Mat4,
        prev_world_from_clip: Mat4,
        world_from_view: Mat4,
        curr_cam_pos: Vec3,
        parallax_sensitivity: f32,
        prev_cam_pos: Vec3,
        virtual_exponent: f32,
        width: u32,
        height: u32,
        min_confidence: f32,
        depth_rejection: f32,
    ) -> Self {
        Self {
            world_from_clip: world_from_clip.to_cols_array(),
            prev_clip_from_world: prev_clip_from_world.to_cols_array(),
            prev_world_from_clip: prev_world_from_clip.to_cols_array(),
            world_from_view: world_from_view.to_cols_array(),
            curr_cam_pos: curr_cam_pos.to_array(),
            parallax_sensitivity,
            prev_cam_pos: prev_cam_pos.to_array(),
            virtual_exponent,
            width,
            height,
            min_confidence,
            depth_rejection,
        }
    }
}

/// Uniform block consumed by `spec_denoise_history_clamp.wesl` (one per
/// dispatch).
///
/// Mirrors the shader's `SpecDenoiseHistoryClampConfig`: twelve 4-byte scalars
/// (five `f32`, four `u32` extent/frame counts and three pad words) = 48 bytes,
/// 4-byte aligned. The three trailing pad words round the record to a 16-byte
/// multiple so the WGSL `var<uniform>` stride matches.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable, PartialEq)]
pub(crate) struct GpuSpecDenoiseHistoryClampConfig {
    /// Colour-box AABB half-width in standard deviations (anti-ghosting clamp).
    pub clamp_sigma: f32,
    /// Specular-lobe normal/roughness consistency tolerance.
    pub lobe_tolerance: f32,
    /// Fast-EMA responsiveness used to detect lighting change.
    pub fast_sensitivity: f32,
    /// Blend rate of the responsive (fast) luminance EMA.
    pub fast_rate: f32,
    /// Blend rate of the stable (slow) luminance EMA.
    pub slow_rate: f32,
    /// Minimum accumulated frame count before the clamp fully trusts history.
    pub min_frames: u32,
    /// Maximum accumulated frame count (history age cap).
    pub max_frames: u32,
    /// Framebuffer width in texels.
    pub width: u32,
    /// Framebuffer height in texels.
    pub height: u32,
    /// Padding to the 16-byte row alignment.
    pub _pad0: u32,
    /// Padding to the 16-byte row alignment.
    pub _pad1: u32,
    /// Padding to the 16-byte row alignment.
    pub _pad2: u32,
}

impl GpuSpecDenoiseHistoryClampConfig {
    /// Builds the history-clamp config from the clamp/EMA tunables, age bounds
    /// and framebuffer extent.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        clamp_sigma: f32,
        lobe_tolerance: f32,
        fast_sensitivity: f32,
        fast_rate: f32,
        slow_rate: f32,
        min_frames: u32,
        max_frames: u32,
        width: u32,
        height: u32,
    ) -> Self {
        Self {
            clamp_sigma,
            lobe_tolerance,
            fast_sensitivity,
            fast_rate,
            slow_rate,
            min_frames,
            max_frames,
            width,
            height,
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::mem::{align_of, offset_of, size_of};

    #[test]
    fn reproject_config_layout_matches_wgsl() {
        // 4 mat4 (256) + 2 (vec3 + f32) rows (32) + 4 trailing scalars (16) = 304.
        assert_eq!(size_of::<GpuSpecDenoiseReprojectConfig>(), 304);
        assert_eq!(align_of::<GpuSpecDenoiseReprojectConfig>(), 4);
        assert_eq!(
            offset_of!(GpuSpecDenoiseReprojectConfig, world_from_clip),
            0
        );
        assert_eq!(
            offset_of!(GpuSpecDenoiseReprojectConfig, prev_clip_from_world),
            64
        );
        assert_eq!(
            offset_of!(GpuSpecDenoiseReprojectConfig, prev_world_from_clip),
            128
        );
        assert_eq!(
            offset_of!(GpuSpecDenoiseReprojectConfig, world_from_view),
            192
        );
        assert_eq!(offset_of!(GpuSpecDenoiseReprojectConfig, curr_cam_pos), 256);
        assert_eq!(
            offset_of!(GpuSpecDenoiseReprojectConfig, parallax_sensitivity),
            268
        );
        assert_eq!(offset_of!(GpuSpecDenoiseReprojectConfig, prev_cam_pos), 272);
        assert_eq!(
            offset_of!(GpuSpecDenoiseReprojectConfig, virtual_exponent),
            284
        );
        assert_eq!(offset_of!(GpuSpecDenoiseReprojectConfig, width), 288);
        assert_eq!(offset_of!(GpuSpecDenoiseReprojectConfig, height), 292);
        assert_eq!(
            offset_of!(GpuSpecDenoiseReprojectConfig, min_confidence),
            296
        );
        assert_eq!(
            offset_of!(GpuSpecDenoiseReprojectConfig, depth_rejection),
            300
        );
    }

    #[test]
    fn reproject_config_uploads_matrices_column_major() {
        let m = Mat4::from_cols_array(&[
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
        ]);
        let cfg = GpuSpecDenoiseReprojectConfig::new(
            m,
            m,
            m,
            m,
            Vec3::new(1.0, 2.0, 3.0),
            8.0,
            Vec3::new(4.0, 5.0, 6.0),
            2.0,
            1920,
            1080,
            0.0,
            0.05,
        );
        assert_eq!(cfg.world_from_clip, m.to_cols_array());
        assert_eq!(cfg.curr_cam_pos, [1.0, 2.0, 3.0]);
        assert_eq!(cfg.prev_cam_pos, [4.0, 5.0, 6.0]);
        assert_eq!((cfg.width, cfg.height), (1920, 1080));
    }

    #[test]
    fn history_clamp_config_layout_matches_wgsl() {
        // 9 meaningful 4-byte scalars + 3 pad words = 48 bytes, 4-byte aligned.
        assert_eq!(size_of::<GpuSpecDenoiseHistoryClampConfig>(), 48);
        assert_eq!(align_of::<GpuSpecDenoiseHistoryClampConfig>(), 4);
        assert_eq!(offset_of!(GpuSpecDenoiseHistoryClampConfig, clamp_sigma), 0);
        assert_eq!(
            offset_of!(GpuSpecDenoiseHistoryClampConfig, lobe_tolerance),
            4
        );
        assert_eq!(
            offset_of!(GpuSpecDenoiseHistoryClampConfig, fast_sensitivity),
            8
        );
        assert_eq!(offset_of!(GpuSpecDenoiseHistoryClampConfig, fast_rate), 12);
        assert_eq!(offset_of!(GpuSpecDenoiseHistoryClampConfig, slow_rate), 16);
        assert_eq!(offset_of!(GpuSpecDenoiseHistoryClampConfig, min_frames), 20);
        assert_eq!(offset_of!(GpuSpecDenoiseHistoryClampConfig, max_frames), 24);
        assert_eq!(offset_of!(GpuSpecDenoiseHistoryClampConfig, width), 28);
        assert_eq!(offset_of!(GpuSpecDenoiseHistoryClampConfig, height), 32);
        assert_eq!(offset_of!(GpuSpecDenoiseHistoryClampConfig, _pad0), 36);
        assert_eq!(offset_of!(GpuSpecDenoiseHistoryClampConfig, _pad1), 40);
        assert_eq!(offset_of!(GpuSpecDenoiseHistoryClampConfig, _pad2), 44);
    }

    #[test]
    fn history_clamp_config_zeroes_pad_words() {
        let cfg = GpuSpecDenoiseHistoryClampConfig::new(2.0, 2.0, 4.0, 0.5, 0.08, 2, 32, 1280, 720);
        assert_eq!((cfg._pad0, cfg._pad1, cfg._pad2), (0, 0, 0));
        assert_eq!((cfg.min_frames, cfg.max_frames), (2, 32));
        assert_eq!((cfg.width, cfg.height), (1280, 720));
    }
}
