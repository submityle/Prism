//! Device ABI for the world-space `ReSTIR` composite's two entry points.
//!
//! Both the `scene_color` -> `gi_base` copy and the energy-conserving direct
//! substitution only need the framebuffer extent to bounds-check each
//! invocation; the two trailing `u32`s round the block up to the 16-byte
//! immediate alignment, mirroring [`super::super::super::world_space_gi`]'s
//! `GpuWorldSpaceGiCompositeParams` byte-for-byte.

use bytemuck::{Pod, Zeroable};

/// Workgroup edge (in pixels) both composite entry points tile the framebuffer
/// with; mirrors the `@workgroup_size(8, 8, 1)` in
/// `shaders/world_restir_composite.wesl`.
pub(crate) const COMPOSITE_WORKGROUP_SIZE: u32 = 8;

/// Immediate (push-constant) block consumed by `world_restir_composite.wesl`'s
/// two entry points (`wr_copy_base`, `wr_composite`).
///
/// Both the base copy and the direct-illumination substitution only need the
/// framebuffer extent to bounds-check each invocation; the two trailing `u32`s
/// round the block up to the 16-byte immediate alignment, matching the
/// world-space GI composite byte-for-byte.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable, PartialEq)]
pub(crate) struct GpuWorldRestirCompositeParams {
    /// Framebuffer width in texels.
    pub width: u32,
    /// Framebuffer height in texels.
    pub height: u32,
    /// Padding to satisfy the 16-byte immediate alignment.
    pub _pad0: u32,
    /// Padding to satisfy the 16-byte immediate alignment.
    pub _pad1: u32,
}

impl GpuWorldRestirCompositeParams {
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
    fn composite_params_is_the_16_byte_immediate_block() {
        // Framebuffer extent (two u32) + two padding u32 fill 16 bytes, the
        // immediate alignment; a drift here would mismatch `set_immediates`
        // against the shader's `var<immediate>` block.
        assert_eq!(size_of::<GpuWorldRestirCompositeParams>(), 16);
        assert_eq!(align_of::<GpuWorldRestirCompositeParams>(), 4);
    }

    #[test]
    fn builder_packs_extent_and_zeroes_padding() {
        let params = GpuWorldRestirCompositeParams::new(1920, 1080);
        assert_eq!(params.width, 1920);
        assert_eq!(params.height, 1080);
        assert_eq!(params._pad0, 0);
        assert_eq!(params._pad1, 0);
    }

    #[test]
    fn dispatch_constant_matches_the_wesl_twin() {
        assert_eq!(COMPOSITE_WORKGROUP_SIZE, 8);
    }
}
