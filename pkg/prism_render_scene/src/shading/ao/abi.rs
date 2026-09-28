//! ABI shared between the GTAO geometry prepass and `shaders/gtao_prepass.wesl`.
//!
//! The prepass is dispatched once over the whole framebuffer; the only
//! per-dispatch state it needs is the `view_from_world` transform (to push the
//! world-space vertices the visibility buffer decodes into the RH,
//! camera-at-origin view space GTAO integrates in) plus the framebuffer
//! dimensions used to reject out-of-bounds invocations.

use bytemuck::{Pod, Zeroable};

/// Workgroup size (per axis) of the `gtao_prepass` compute entry point.
///
/// Must match `@workgroup_size(N, N, 1)` in `shaders/gtao_prepass.wesl`; the
/// dispatch rounds the viewport up to a multiple of this on both axes.
pub(crate) const GTAO_PREPASS_WORKGROUP_SIZE: u32 = 8;

/// Immediate (push-constant) block consumed by `gtao_prepass.wesl`.
///
/// `view_from_world` is a column-major `Mat4` (uploaded via
/// [`glam::Mat4::to_cols_array`]) so the WGSL `mat4x4<f32>` multiply agrees
/// byte-for-byte. The 64-byte matrix is already 16-byte aligned, so the four
/// trailing `u32`s (two live, two padding) fill the final 16 bytes exactly.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable, PartialEq)]
pub(crate) struct GpuGtaoPrepassParams {
    /// Column-major world -> view transform.
    pub view_from_world: [f32; 16],
    /// Framebuffer width in pixels; invocations at or beyond it early-out.
    pub width: u32,
    /// Framebuffer height in pixels.
    pub height: u32,
    /// Padding to round the block out to a 16-byte multiple.
    pub _pad0: u32,
    /// Padding to round the block out to a 16-byte multiple.
    pub _pad1: u32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prepass_params_layout_matches_the_wgsl_immediate_block() {
        // 64-byte mat4x4 (16-byte aligned) + four u32 scalars = 80 bytes.
        assert_eq!(size_of::<GpuGtaoPrepassParams>(), 80);
        assert_eq!(align_of::<GpuGtaoPrepassParams>(), 4);
        assert_eq!(GTAO_PREPASS_WORKGROUP_SIZE, 8);
    }
}
