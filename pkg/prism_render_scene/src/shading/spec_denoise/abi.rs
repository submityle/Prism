//! ABI shared between the specular-GI *spatial denoise* pass and
//! `shaders/spec_denoise_spatial.wesl`.
//!
//! The spatial pass is the on-device twin of the edge-aware cross-bilateral
//! filter in `prism_render_shading::gi::spec_denoise::spatial`. It consumes the
//! noisy per-pixel specular estimate produced by the `spec_gi` reuse/composite
//! chain together with the SSR trace's per-pixel hit distance, and widens the
//! blur anisotropically along the surface's reflection footprint while gating
//! on depth, normal and roughness so reflections stay crisp on silhouettes and
//! material boundaries (the ReBLUR/ReLAX-style spatial pre-filter the temporal
//! history-clamp block consumes next).
//!
//! This slice lands the single `#[repr(C)]` uniform record the kernel reads:
//! [`GpuSpecDenoiseSpatialConfig`]. Every field mirrors its
//! `spec_denoise_spatial.wesl` struct byte-for-byte so hosts with and without a
//! GPU agree with the CPU golden; the `size_of`/`offset_of` contract tests
//! below guard the layout against drift.

use bevy_math::Mat4;
use bytemuck::{Pod, Zeroable};

/// Workgroup size (per axis) of the spatial-denoise compute entry point.
///
/// Must match `@workgroup_size(N, N, 1)` in `spec_denoise_spatial.wesl`; the
/// dispatch rounds its target extent up to a multiple of this on both axes and
/// the kernel bounds-checks every invocation.
pub(crate) const SPEC_DENOISE_WORKGROUP_SIZE: u32 = 8;

/// Uniform block consumed by `spec_denoise_spatial.wesl` (one per dispatch).
///
/// Mirrors the shader's `SpecDenoiseSpatialConfig`: the `mat4x4` forces 16-byte
/// struct alignment, so the ten trailing scalars plus three pad words pack into
/// the three 16-byte rows that follow the matrix (112 bytes total). The spatial
/// filter parameters match the CPU golden's `SpatialParams`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable, PartialEq)]
pub(crate) struct GpuSpecDenoiseSpatialConfig {
    /// Clip -> view (inverse projection); rebuilds the view-space position of
    /// each tap from the SSR prepass reverse-Z depth. Uploaded column-major.
    pub view_from_clip: [f32; 16],
    /// Framebuffer width in texels (invocations round up / bounds-check).
    pub width: u32,
    /// Framebuffer height in texels.
    pub height: u32,
    /// Maximum blur radius in pixels at the roughest/most-distant extreme
    /// (`SpatialParams::max_radius`).
    pub max_radius: f32,
    /// Depth plane-distance gate sigma (`SpatialParams::phi_depth`).
    pub phi_depth: f32,
    /// Normal alignment gate exponent (`SpatialParams::phi_normal`).
    pub phi_normal: f32,
    /// Roughness gate sigma (`SpatialParams::phi_roughness`).
    pub phi_roughness: f32,
    /// Contact-hardening strength (`SpatialParams::contact_hardening`): shrinks
    /// the kernel toward crisp contacts for short reflection hit distances.
    pub contact_hardening: f32,
    /// Maximum anisotropy ratio (`SpatialParams::max_anisotropy`) between the
    /// major and minor kernel axes at grazing angles.
    pub max_anisotropy: f32,
    /// Positive near-plane distance (front of camera along `-Z`).
    pub near: f32,
    /// Padding to the 16-byte row alignment.
    pub _pad0: u32,
    /// Padding to the 16-byte row alignment.
    pub _pad1: u32,
    /// Padding to the 16-byte row alignment.
    pub _pad2: u32,
}

impl GpuSpecDenoiseSpatialConfig {
    /// Builds the spatial-denoise config from the inverse projection,
    /// framebuffer extent, spatial filter parameters and near plane.
    /// `view_from_clip` is uploaded column-major (via [`Mat4::to_cols_array`])
    /// so the WGSL `mat4x4<f32>` multiply agrees byte-for-byte.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        view_from_clip: Mat4,
        width: u32,
        height: u32,
        max_radius: f32,
        phi_depth: f32,
        phi_normal: f32,
        phi_roughness: f32,
        contact_hardening: f32,
        max_anisotropy: f32,
        near: f32,
    ) -> Self {
        Self {
            view_from_clip: view_from_clip.to_cols_array(),
            width,
            height,
            max_radius,
            phi_depth,
            phi_normal,
            phi_roughness,
            contact_hardening,
            max_anisotropy,
            near,
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
    fn gpu_spec_denoise_spatial_config_layout_matches_wgsl() {
        // mat4 (64) + 10 scalars (40) + 3 pad words (12) = 116 -> rounds to 112?
        // No: 64 + 40 + 12 = 116; the matrix forces 16-byte struct align so the
        // tail rounds up to a multiple of 16 -> but the explicit pads already
        // fill the final row exactly, giving 112.
        assert_eq!(size_of::<GpuSpecDenoiseSpatialConfig>(), 112);
        assert_eq!(align_of::<GpuSpecDenoiseSpatialConfig>(), 4);
        assert_eq!(offset_of!(GpuSpecDenoiseSpatialConfig, view_from_clip), 0);
        assert_eq!(offset_of!(GpuSpecDenoiseSpatialConfig, width), 64);
        assert_eq!(offset_of!(GpuSpecDenoiseSpatialConfig, height), 68);
        assert_eq!(offset_of!(GpuSpecDenoiseSpatialConfig, max_radius), 72);
        assert_eq!(offset_of!(GpuSpecDenoiseSpatialConfig, phi_depth), 76);
        assert_eq!(offset_of!(GpuSpecDenoiseSpatialConfig, phi_normal), 80);
        assert_eq!(offset_of!(GpuSpecDenoiseSpatialConfig, phi_roughness), 84);
        assert_eq!(
            offset_of!(GpuSpecDenoiseSpatialConfig, contact_hardening),
            88
        );
        assert_eq!(offset_of!(GpuSpecDenoiseSpatialConfig, max_anisotropy), 92);
        assert_eq!(offset_of!(GpuSpecDenoiseSpatialConfig, near), 96);
        assert_eq!(offset_of!(GpuSpecDenoiseSpatialConfig, _pad0), 100);
        assert_eq!(offset_of!(GpuSpecDenoiseSpatialConfig, _pad1), 104);
        assert_eq!(offset_of!(GpuSpecDenoiseSpatialConfig, _pad2), 108);
    }

    #[test]
    fn spatial_config_uploads_matrix_column_major() {
        let m = Mat4::from_cols_array(&[
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
        ]);
        let cfg =
            GpuSpecDenoiseSpatialConfig::new(m, 1920, 1080, 32.0, 0.5, 128.0, 0.08, 1.0, 3.0, 0.1);
        assert_eq!(cfg.view_from_clip, m.to_cols_array());
        assert_eq!((cfg.width, cfg.height), (1920, 1080));
        assert_eq!((cfg._pad0, cfg._pad1, cfg._pad2), (0, 0, 0));
    }
}
