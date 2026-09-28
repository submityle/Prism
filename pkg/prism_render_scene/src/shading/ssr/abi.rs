//! ABI shared between the SSR geometry prepass and `shaders/ssr_prepass.wesl`.
//!
//! The Prism renderer is a visibility-buffer deferred pipeline with no
//! depth/normal G-buffer, so screen-space reflection must first *rebuild* the
//! inputs its trace consumes. This slice lands the geometry prepass, which
//! reconstructs the reverse-Z device depth (`scene_depth`) and the view-space
//! surface normal (`view_normal`) one texel per framebuffer pixel, driven by
//! [`GpuSsrPrepassParams`]. The Hi-Z pyramid build, the material-roughness
//! repack and the trace itself (and their `GpuSsrHzbParams`/`GpuSsrConfig`
//! immediate blocks) land in following slices so every committed ABI struct has
//! a live consumer.
//!
//! Every field mirrors its shader struct byte-for-byte so machines with and
//! without a GPU agree with the CPU golden in
//! [`prism_render_shading::screen_space`].

use bevy_math::Mat4;
use bytemuck::{Pod, Zeroable};

/// Workgroup size (per axis) of every SSR compute entry point.
///
/// Must match `@workgroup_size(N, N, 1)` in the SSR shaders; each dispatch
/// rounds its target extent up to a multiple of this on both axes and the
/// shaders bounds-check every invocation.
pub(crate) const SSR_WORKGROUP_SIZE: u32 = 8;

/// Immediate (push-constant) block consumed by `ssr_prepass.wesl`.
///
/// Mirrors the shader's `PrepassParams`: the rigid `view_from_world` transform
/// used to push world-space vertices into the camera-at-origin view frame, the
/// `clip_from_view` projection used to derive reverse-Z device depth, and the
/// framebuffer extent. Two trailing `u32`s pad the block to the 16-byte
/// immediate alignment WGSL requires after the two matrices.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable, PartialEq)]
pub(crate) struct GpuSsrPrepassParams {
    /// World -> view (rigid, camera-at-origin, looking down `-Z`).
    pub view_from_world: [f32; 16],
    /// View -> clip (reverse-Z perspective), used to derive device depth.
    pub clip_from_view: [f32; 16],
    /// Framebuffer width in texels.
    pub width: u32,
    /// Framebuffer height in texels.
    pub height: u32,
    /// Padding to satisfy the 16-byte immediate alignment.
    pub _pad0: u32,
    /// Padding to satisfy the 16-byte immediate alignment.
    pub _pad1: u32,
}

impl GpuSsrPrepassParams {
    /// Builds the prepass params from the view/projection matrices and the
    /// framebuffer extent. Both matrices are uploaded column-major (via
    /// [`Mat4::to_cols_array`]) so the WGSL `mat4x4<f32>` multiply agrees
    /// byte-for-byte.
    pub(crate) fn new(view_from_world: Mat4, clip_from_view: Mat4, width: u32, height: u32) -> Self {
        Self {
            view_from_world: view_from_world.to_cols_array(),
            clip_from_view: clip_from_view.to_cols_array(),
            width,
            height,
            _pad0: 0,
            _pad1: 0,
        }
    }
}

/// Immediate (push-constant) block consumed by the two `ssr_hzb.wesl` entry
/// points that build the reverse-Z "nearest depth" Hi-Z pyramid the trace
/// marches.
///
/// Both the mip-0 copy (`ssr_hzb_copy`, which lifts the full-resolution
/// `scene_depth` into pyramid level 0) and each coarser 2x2 max-reduction
/// (`ssr_hzb_reduce`) are driven by the destination/source extents so the GPU
/// build reproduces `prism_render_shading::DepthPyramid::from_nearest_reduction`
/// exactly: odd source dimensions clamp their trailing column/row, and the
/// per-cell maximum keeps the *nearest* surface under a coarse cell.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable, PartialEq)]
pub(crate) struct GpuSsrHzbParams {
    /// Destination mip width in texels (invocations round up to this).
    pub dst_width: u32,
    /// Destination mip height in texels.
    pub dst_height: u32,
    /// Source mip width in texels; used to clamp the 2x2 footprint edge.
    pub src_width: u32,
    /// Source mip height in texels.
    pub src_height: u32,
}

impl GpuSsrHzbParams {
    /// Builds the pyramid-build params from the destination/source extents.
    pub(crate) fn new(dst: bevy_math::UVec2, src: bevy_math::UVec2) -> Self {
        Self {
            dst_width: dst.x,
            dst_height: dst.y,
            src_width: src.x,
            src_height: src.y,
        }
    }
}

/// Immediate (push-constant) block consumed by `ssr_repack.wesl`, the stage
/// that folds the geometry prepass's signed view-space normal and the material
/// roughness into the trace's `normal_roughness` input.
///
/// Mirrors the shader's `RepackParams`: the framebuffer extent plus two trailing
/// `u32`s that pad the block to the 16-byte immediate alignment WGSL requires.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable, PartialEq)]
pub(crate) struct GpuSsrRepackParams {
    /// Framebuffer width in texels.
    pub width: u32,
    /// Framebuffer height in texels.
    pub height: u32,
    /// Padding to satisfy the 16-byte immediate alignment.
    pub _pad0: u32,
    /// Padding to satisfy the 16-byte immediate alignment.
    pub _pad1: u32,
}

impl GpuSsrRepackParams {
    /// Builds the repack params from the framebuffer extent.
    pub(crate) fn new(width: u32, height: u32) -> Self {
        Self {
            width,
            height,
            _pad0: 0,
            _pad1: 0,
        }
    }
}

/// Immediate (push-constant) block consumed by `ssr.wesl`'s `trace_ssr` entry
/// point — the screen-space march itself.
///
/// Mirrors the shader's `SsrConfig`: the reverse-Z projection (`clip_from_view`)
/// and its inverse (`view_from_clip`) used to project the reflected ray and
/// reconstruct view positions, the near-plane/march tunables, the confidence
/// fades (mirroring the golden `SsrConfidenceParams`) and march limits
/// (mirroring `SsrMarchConfig`), plus the finest colour mip and the pyramid's
/// top mip index used to pick the roughness-selected reflection blur, and the
/// full-resolution framebuffer extent.
///
/// The two `mat4x4<f32>` fields (64 B each, offsets 0 and 64) precede the
/// scalars; `screen_size` (a `vec2<f32>`) lands at offset 168, already 8-byte
/// aligned, and the block ends at 176 bytes — a multiple of the 16-byte
/// immediate alignment WGSL requires, so no trailing pad is needed.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable, PartialEq)]
pub(crate) struct GpuSsrConfig {
    /// View -> clip (reverse-Z perspective); projects the marched ray.
    pub clip_from_view: [f32; 16],
    /// Clip -> view (inverse projection); reconstructs view-space positions.
    pub view_from_clip: [f32; 16],
    /// Positive near-plane distance in front of the camera along `-Z`.
    pub near: f32,
    /// View-space march length the reflected ray is extended along.
    pub max_distance: f32,
    /// Device-depth thin-surface tolerance for accepting a hit.
    pub thickness: f32,
    /// Hard iteration cap for the hierarchical march.
    pub max_iterations: u32,
    /// UV border margin where the edge fade begins ramping.
    pub edge_fade_start: f32,
    /// Perceptual roughness below which SSR is fully trusted.
    pub full_roughness: f32,
    /// Perceptual roughness at/above which SSR is fully replaced by IBL.
    pub max_roughness: f32,
    /// Normalized travel where the distance fade begins ramping.
    pub distance_fade_start: f32,
    /// Finest Hi-Z mip the march refines down to (usually 0).
    pub most_detailed_mip: u32,
    /// Top colour-pyramid mip index (`color_mip_count - 1`), the roughest blur.
    pub color_max_mip: f32,
    /// Full-resolution framebuffer extent in texels.
    pub screen_size: [f32; 2],
}

impl GpuSsrConfig {
    /// Builds the trace config from the view matrices and framebuffer extent,
    /// folding in the golden `SsrConfidenceParams`/`SsrMarchConfig` defaults from
    /// [`prism_render_shading::screen_space`] so the GPU march agrees with the
    /// CPU reference. Both matrices upload column-major (via
    /// [`Mat4::to_cols_array`]) so the WGSL `mat4x4<f32>` multiply matches
    /// byte-for-byte.
    pub(crate) fn from_view(
        clip_from_view: Mat4,
        view_from_clip: Mat4,
        near: f32,
        max_distance: f32,
        screen_size: bevy_math::UVec2,
        color_max_mip: f32,
    ) -> Self {
        Self {
            clip_from_view: clip_from_view.to_cols_array(),
            view_from_clip: view_from_clip.to_cols_array(),
            near,
            max_distance,
            // Golden `SsrMarchConfig::default()`.
            thickness: 0.02,
            max_iterations: 128,
            // Golden `SsrConfidenceParams::default()`.
            edge_fade_start: 0.1,
            full_roughness: 0.2,
            max_roughness: 0.6,
            distance_fade_start: 0.7,
            most_detailed_mip: 0,
            color_max_mip: color_max_mip.max(0.0),
            screen_size: [screen_size.x as f32, screen_size.y as f32],
        }
    }
}

/// Immediate (push-constant) block consumed by `ssr_composite.wesl`, the stage
/// that blends the trace's reflection buffer over the shaded `scene_color`.
///
/// Mirrors the shader's `CompositeParams`: the framebuffer extent plus two
/// trailing `u32`s padding the block to the 16-byte immediate alignment WGSL
/// requires (identical shape to [`GpuSsrRepackParams`]).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable, PartialEq)]
pub(crate) struct GpuSsrCompositeParams {
    /// Framebuffer width in texels.
    pub width: u32,
    /// Framebuffer height in texels.
    pub height: u32,
    /// Padding to satisfy the 16-byte immediate alignment.
    pub _pad0: u32,
    /// Padding to satisfy the 16-byte immediate alignment.
    pub _pad1: u32,
}

impl GpuSsrCompositeParams {
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

    #[test]
    fn prepass_params_match_the_shader_immediate_layout() {
        // Two mat4x4 (128) + width/height/pad0/pad1 (16) = 144 bytes, 16-byte
        // aligned as an immediate block.
        assert_eq!(size_of::<GpuSsrPrepassParams>(), 144);
        assert_eq!(align_of::<GpuSsrPrepassParams>(), 4);
        assert_eq!(SSR_WORKGROUP_SIZE, 8);
    }

    #[test]
    fn hzb_params_match_the_shader_immediate_layout() {
        // Four tightly packed u32 extents = 16 bytes, the WGSL immediate
        // alignment, and the fields round-trip in declaration order.
        assert_eq!(size_of::<GpuSsrHzbParams>(), 16);
        assert_eq!(align_of::<GpuSsrHzbParams>(), 4);
        let params =
            GpuSsrHzbParams::new(bevy_math::UVec2::new(960, 540), bevy_math::UVec2::new(1920, 1080));
        assert_eq!(params.dst_width, 960);
        assert_eq!(params.dst_height, 540);
        assert_eq!(params.src_width, 1920);
        assert_eq!(params.src_height, 1080);
    }

    #[test]
    fn repack_params_match_the_shader_immediate_layout() {
        // width/height/pad0/pad1 = 16 bytes, the WGSL immediate alignment, and
        // the fields round-trip in declaration order.
        assert_eq!(size_of::<GpuSsrRepackParams>(), 16);
        assert_eq!(align_of::<GpuSsrRepackParams>(), 4);
        let params = GpuSsrRepackParams::new(1920, 1080);
        assert_eq!(params.width, 1920);
        assert_eq!(params.height, 1080);
        assert_eq!(params._pad0, 0);
        assert_eq!(params._pad1, 0);
    }

    #[test]
    fn prepass_params_upload_matrices_column_major() {
        let view = Mat4::from_cols_array(&[
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
        ]);
        // A distinct (non-perspective) column-major matrix keeps this a pure
        // byte-layout check without pulling in the deprecated `glam`
        // `perspective_rh`; only column-major fidelity is under test here.
        let clip = Mat4::from_cols_array(&[
            17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0,
            31.0, 32.0,
        ]);
        let params = GpuSsrPrepassParams::new(view, clip, 1920, 1080);
        assert_eq!(params.view_from_world, view.to_cols_array());
        assert_eq!(params.clip_from_view, clip.to_cols_array());
        assert_eq!(params.width, 1920);
        assert_eq!(params.height, 1080);
    }

    #[test]
    fn config_matches_the_shader_immediate_layout() {
        // Two mat4x4 (128) + ten scalars (40) + a vec2<f32> (8) = 176 bytes,
        // already a multiple of the 16-byte immediate alignment (screen_size
        // lands 8-byte aligned at offset 168), so no trailing pad is needed.
        assert_eq!(size_of::<GpuSsrConfig>(), 176);
        assert_eq!(align_of::<GpuSsrConfig>(), 4);
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
        let config =
            GpuSsrConfig::from_view(clip, inv, 0.5, 100.0, bevy_math::UVec2::new(1920, 1080), 7.0);
        assert_eq!(config.clip_from_view, clip.to_cols_array());
        assert_eq!(config.view_from_clip, inv.to_cols_array());
        assert_eq!(config.near, 0.5);
        assert_eq!(config.max_distance, 100.0);
        // Golden `SsrMarchConfig`/`SsrConfidenceParams` defaults.
        assert_eq!(config.thickness, 0.02);
        assert_eq!(config.max_iterations, 128);
        assert_eq!(config.edge_fade_start, 0.1);
        assert_eq!(config.full_roughness, 0.2);
        assert_eq!(config.max_roughness, 0.6);
        assert_eq!(config.distance_fade_start, 0.7);
        assert_eq!(config.most_detailed_mip, 0);
        assert_eq!(config.color_max_mip, 7.0);
        assert_eq!(config.screen_size, [1920.0, 1080.0]);
    }

    #[test]
    fn composite_params_match_the_shader_immediate_layout() {
        // width/height/pad0/pad1 = 16 bytes, the WGSL immediate alignment, and
        // the fields round-trip in declaration order.
        assert_eq!(size_of::<GpuSsrCompositeParams>(), 16);
        assert_eq!(align_of::<GpuSsrCompositeParams>(), 4);
        let params = GpuSsrCompositeParams::new(1920, 1080);
        assert_eq!(params.width, 1920);
        assert_eq!(params.height, 1080);
        assert_eq!(params._pad0, 0);
        assert_eq!(params._pad1, 0);
    }
}
