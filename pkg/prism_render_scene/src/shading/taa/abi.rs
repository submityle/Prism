//! Immediate (push-constant) block shared with `shaders/taa_resolve.wesl`.
//!
//! Kept byte-for-byte in sync with the shader's `TaaParams` struct: the
//! framebuffer extent, the golden [`prism_render_shading::taa::TaaParams`]
//! tunables, and the history-validity flag the resolve consults before it
//! trusts the ping-pong history.

use bytemuck::{Pod, Zeroable};

/// 8x8 pixel tile per workgroup, matching the shader's `@workgroup_size(8,8,1)`.
pub(crate) const TAA_WORKGROUP_SIZE: u32 = 8;

/// Framebuffer extent + golden TAA tunables + history-valid flag, uploaded as
/// the `taa_resolve.wesl` immediate block.
///
/// All fields are 4-byte scalars laid out back to back, so the `#[repr(C)]`
/// record is 20 bytes with no padding and matches the WESL `TaaParams` struct
/// exactly.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub(crate) struct GpuTaaResolveParams {
    /// Framebuffer width in texels.
    pub width: u32,
    /// Framebuffer height in texels.
    pub height: u32,
    /// Fraction of the (clipped) reprojected history kept when trusted.
    pub history_blend: f32,
    /// Standard-deviation multiplier for the `YCoCg` variance clip box.
    pub variance_gamma: f32,
    /// `1` when a valid previous frame exists (no resize / camera cut), else `0`.
    pub valid_history: u32,
}

impl GpuTaaResolveParams {
    /// Builds the resolve params from the framebuffer extent, folding in the
    /// golden [`prism_render_shading::taa::TaaParams`] defaults
    /// (`history_blend = 0.9`, `variance_gamma = 1.0`). `valid_history` gates
    /// whether the shader trusts the ping-pong history at all (cleared on the
    /// first frame a view is seen and on a resize).
    pub(crate) fn new(width: u32, height: u32, valid_history: bool) -> Self {
        Self {
            width,
            height,
            // Golden `TaaParams::default()`.
            history_blend: 0.9,
            variance_gamma: 1.0,
            valid_history: u32::from(valid_history),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn taa_workgroup_size_matches_shader() {
        assert_eq!(TAA_WORKGROUP_SIZE, 8);
    }

    #[test]
    fn resolve_params_layout_matches_the_wesl_immediate_block() {
        // Five 4-byte scalars, no padding.
        assert_eq!(size_of::<GpuTaaResolveParams>(), 20);
        assert_eq!(align_of::<GpuTaaResolveParams>(), 4);

        let params = GpuTaaResolveParams::new(1920, 1080, true);
        assert_eq!(params.width, 1920);
        assert_eq!(params.height, 1080);
        assert_eq!(params.history_blend, 0.9);
        assert_eq!(params.variance_gamma, 1.0);
        assert_eq!(params.valid_history, 1);

        let invalid = GpuTaaResolveParams::new(1, 1, false);
        assert_eq!(invalid.valid_history, 0);
    }
}
