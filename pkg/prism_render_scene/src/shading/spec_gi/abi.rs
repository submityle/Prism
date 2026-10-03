//! ABI shared between the glossy-specular ReSTIR *reuse* pass and
//! `shaders/spec_gi_reuse.wesl`.
//!
//! The reuse pass is the on-device twin of the spatio-temporal resampling layer
//! in `prism_render_shading::gi::spec_gi::glossy_reservoir`. It layers reservoir
//! reuse on top of the screen-space GGX candidate rays (the same importance
//! sampling the SSR trace already performs) so a handful of noisy glossy
//! candidates converge to a low-variance specular estimate before the
//! `spec_denoise` passes and the energy-conserving composite consume it.
//!
//! This slice lands the two `#[repr(C)]` records the kernel reads:
//!
//! * [`GpuSpecrReservoir`] — the per-pixel reservoir *storage* layout. The
//!   WESL twin's working reservoir (`SpecrReservoir`) nests three `vec3<f32>`
//!   payload vectors; to pin a stable, host-mirrorable byte layout we store the
//!   flattened form where every `vec3` shares its 16-byte row with the trailing
//!   scalar it pairs with (`visible_point|w_sum`, `sample_point|m`,
//!   `radiance|w`), then the `has_sample` flag and three pad words. This is the
//!   classic WGSL "`vec3<f32>` + `f32` packs into 16 bytes" rule, so the Rust
//!   and WGSL views agree byte-for-byte.
//! * [`GpuSpecGiReuseConfig`] — the uniform block driving one reuse dispatch:
//!   the inverse projection used to rebuild the view-space glossy point from the
//!   SSR prepass depth, the framebuffer extent, the temporal M-cap, the
//!   roughness confidence-cap base, the roughness reuse sigma, the near plane,
//!   and the temporal/spatial reuse toggles.
//!
//! Every field mirrors its `spec_gi_reuse.wesl` struct byte-for-byte so hosts
//! with and without a GPU agree with the CPU golden; the `size_of`/`offset_of`
//! contract tests below guard the layout against drift.

use bevy_math::Mat4;
use bytemuck::{Pod, Zeroable};

/// Workgroup size (per axis) of the glossy-specular reuse compute entry point.
///
/// Must match `@workgroup_size(N, N, 1)` in `spec_gi_reuse.wesl`; the dispatch
/// rounds its target extent up to a multiple of this on both axes and the
/// kernel bounds-checks every invocation.
#[allow(dead_code)] // consumed by the reuse dispatch slice (next).
pub(crate) const SPEC_GI_WORKGROUP_SIZE: u32 = 8;

/// Per-pixel glossy reservoir *storage* record (the ping-pong buffer element).
///
/// Flattened twin of `spec_gi_reuse.wesl`'s `GpuSpecrReservoir`: three
/// `vec3<f32>` payload vectors, each sharing its 16-byte row with the scalar it
/// pairs with, then the `has_sample` flag padded out to the next 16-byte row.
/// 64 bytes total, 16-byte aligned.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable, PartialEq)]
pub(crate) struct GpuSpecrReservoir {
    /// Visible-point (shading point) world/view position `x_v`.
    pub visible_point: [f32; 3],
    /// Running sum of resampling weights of every candidate considered.
    pub w_sum: f32,
    /// Secondary sample-point position `x_s`.
    pub sample_point: [f32; 3],
    /// Confidence weight: fractional candidate count this reservoir summarises.
    pub m: f32,
    /// Linear RGB radiance leaving `x_s` towards `x_v`.
    pub radiance: [f32; 3],
    /// Finalised unbiased contribution weight `W` (valid after finalize).
    pub w: f32,
    /// `1` when the reservoir holds a selected candidate, else `0`
    /// (`Option<GiSample>` modelled as a flag, matching the WESL twin).
    pub has_sample: u32,
    /// Padding to the 16-byte row alignment.
    pub _pad0: u32,
    /// Padding to the 16-byte row alignment.
    pub _pad1: u32,
    /// Padding to the 16-byte row alignment.
    pub _pad2: u32,
}

/// Uniform block consumed by `spec_gi_reuse.wesl` (one per reuse dispatch).
///
/// Mirrors the shader's `SpecGiReuseConfig`: the `mat4x4` forces 16-byte struct
/// alignment, so the eight trailing scalars pack into the two 16-byte rows that
/// follow the matrix with no extra padding (96 bytes total).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable, PartialEq)]
#[allow(dead_code)] // fields read by `spec_gi_reuse.wesl` via the bind-group slice (next).
pub(crate) struct GpuSpecGiReuseConfig {
    /// Clip -> view (inverse projection); rebuilds the view-space glossy point
    /// from the SSR prepass reverse-Z depth. Uploaded column-major.
    pub view_from_clip: [f32; 16],
    /// Framebuffer width in texels (invocations round up / bounds-check).
    pub width: u32,
    /// Framebuffer height in texels.
    pub height: u32,
    /// Temporal confidence cap `M` applied to the reprojected prior reservoir
    /// before merging (the ReSTIR history clamp; `glossy_reservoir` M-cap).
    pub m_cap: f32,
    /// Base confidence cap fed to `specr_roughness_confidence_cap`, tightened
    /// for sharp (low-roughness) surfaces so stale history is dropped faster.
    pub roughness_cap_base: f32,
    /// Normal/roughness reuse sigma gating `specr_roughness_reuse_weight`.
    pub sigma_roughness: f32,
    /// Positive near-plane distance (front of camera along `-Z`).
    pub near: f32,
    /// `1` enables temporal (prior-frame) reservoir reuse, else `0`.
    pub temporal_enabled: u32,
    /// `1` enables spatial (neighbour) reservoir reuse, else `0`.
    pub spatial_enabled: u32,
}

#[allow(dead_code)] // ctor used by the bind-group/dispatch slice (next) and the tests.
impl GpuSpecGiReuseConfig {
    /// Builds the reuse config from the inverse projection, framebuffer extent
    /// and reuse parameters. `view_from_clip` is uploaded column-major (via
    /// [`Mat4::to_cols_array`]) so the WGSL `mat4x4<f32>` multiply agrees
    /// byte-for-byte.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        view_from_clip: Mat4,
        width: u32,
        height: u32,
        m_cap: f32,
        roughness_cap_base: f32,
        sigma_roughness: f32,
        near: f32,
        temporal_enabled: bool,
        spatial_enabled: bool,
    ) -> Self {
        Self {
            view_from_clip: view_from_clip.to_cols_array(),
            width,
            height,
            m_cap,
            roughness_cap_base,
            sigma_roughness,
            near,
            temporal_enabled: temporal_enabled as u32,
            spatial_enabled: spatial_enabled as u32,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::mem::{align_of, offset_of, size_of};

    #[test]
    fn gpu_specr_reservoir_layout_matches_wgsl() {
        // 4 rows x 16 bytes; `vec3 + scalar` share each of the first three rows.
        assert_eq!(size_of::<GpuSpecrReservoir>(), 64);
        assert_eq!(align_of::<GpuSpecrReservoir>(), 4);
        assert_eq!(offset_of!(GpuSpecrReservoir, visible_point), 0);
        assert_eq!(offset_of!(GpuSpecrReservoir, w_sum), 12);
        assert_eq!(offset_of!(GpuSpecrReservoir, sample_point), 16);
        assert_eq!(offset_of!(GpuSpecrReservoir, m), 28);
        assert_eq!(offset_of!(GpuSpecrReservoir, radiance), 32);
        assert_eq!(offset_of!(GpuSpecrReservoir, w), 44);
        assert_eq!(offset_of!(GpuSpecrReservoir, has_sample), 48);
        assert_eq!(offset_of!(GpuSpecrReservoir, _pad2), 60);
    }

    #[test]
    fn gpu_spec_gi_reuse_config_layout_matches_wgsl() {
        // mat4 (64) + 8 scalars (32) = 96, matrix forces 16-byte struct align.
        assert_eq!(size_of::<GpuSpecGiReuseConfig>(), 96);
        assert_eq!(offset_of!(GpuSpecGiReuseConfig, view_from_clip), 0);
        assert_eq!(offset_of!(GpuSpecGiReuseConfig, width), 64);
        assert_eq!(offset_of!(GpuSpecGiReuseConfig, height), 68);
        assert_eq!(offset_of!(GpuSpecGiReuseConfig, m_cap), 72);
        assert_eq!(offset_of!(GpuSpecGiReuseConfig, roughness_cap_base), 76);
        assert_eq!(offset_of!(GpuSpecGiReuseConfig, sigma_roughness), 80);
        assert_eq!(offset_of!(GpuSpecGiReuseConfig, near), 84);
        assert_eq!(offset_of!(GpuSpecGiReuseConfig, temporal_enabled), 88);
        assert_eq!(offset_of!(GpuSpecGiReuseConfig, spatial_enabled), 92);
    }

    #[test]
    fn reuse_config_uploads_matrix_column_major() {
        let m = Mat4::from_cols_array(&[
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
        ]);
        let cfg = GpuSpecGiReuseConfig::new(m, 1920, 1080, 32.0, 16.0, 0.1, 0.1, true, false);
        assert_eq!(cfg.view_from_clip, m.to_cols_array());
        assert_eq!((cfg.width, cfg.height), (1920, 1080));
        assert_eq!((cfg.temporal_enabled, cfg.spatial_enabled), (1, 0));
    }
}
