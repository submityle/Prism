//! ABI shared between the shading-resolve compute pass and
//! `shaders/shading_resolve.wesl`.
//!
//! The resolve entry point is dispatched once per `MaterialShadingClass` with
//! an indirect argument buffer, so the only per-dispatch state it needs is the
//! active class index plus the framebuffer dimensions and the world-space view
//! position used to build the shading frame.  Everything else is read from the
//! storage buffers bound by [`super::bind_groups`].

use bytemuck::{Pod, Zeroable};

/// Workgroup size of the `shading_resolve` compute entry point.
///
/// Must match `@workgroup_size(...)` in `shaders/shading_resolve.wesl` and the
/// `local_invocation` bound checks against the per-class work count.
pub(crate) const RESOLVE_WORKGROUP_SIZE: u32 = 64;

/// Immediate (push-constant) block consumed by `shading_resolve.wesl`.
///
/// `shading_class` selects which compacted worklist slice this dispatch drains
/// (`work[class_offsets[class] + local]`).  `view_position` is padded to a
/// 16-byte boundary so the trailing `vec4<f32>` lands on its natural alignment
/// and the Rust/WGSL layouts agree byte-for-byte.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable, PartialEq)]
pub(crate) struct GpuShadingResolveParams {
    /// Active `MaterialShadingClass` discriminant for this dispatch.
    pub shading_class: u32,
    /// Framebuffer width in pixels (used to unflatten `pixel_index`).
    pub width: u32,
    /// Framebuffer height in pixels.
    pub height: u32,
    /// Packed feature bits; see [`RESOLVE_FLAG_GTAO`] and
    /// [`RESOLVE_FLAG_IBL_SPECULAR`].  Doubles as the 16-byte alignment word
    /// ahead of the `vec4<f32>`.
    pub flags: u32,
    /// World-space camera position; `w` is unused padding.
    pub view_position: [f32; 4],
}

/// `flags` bit selecting the screen-space GTAO occlusion multiply in the
/// resolve shader.  Mirrors `RESOLVE_FLAG_GTAO` in `shading_resolve.wesl`.
pub(crate) const RESOLVE_FLAG_GTAO: u32 = 1 << 0;

/// `flags` bit signalling that the prefiltered environment cube and the DFG
/// table are resident, so the resolve samples the real split-sum specular
/// reflection instead of the low-frequency SH-radiance fallback.  Mirrors
/// `RESOLVE_FLAG_IBL_SPECULAR` in `shading_resolve.wesl`.
pub(crate) const RESOLVE_FLAG_IBL_SPECULAR: u32 = 1 << 1;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolve_params_layout_matches_the_wgsl_immediate_block() {
        // vec4<f32> forces 16-byte alignment; the four leading u32 scalars fill
        // the first 16 bytes exactly, so the whole block is 32 bytes.
        assert_eq!(size_of::<GpuShadingResolveParams>(), 32);
        assert_eq!(align_of::<GpuShadingResolveParams>(), 4);
        assert_eq!(RESOLVE_WORKGROUP_SIZE, 64);
    }

    #[test]
    fn resolve_feature_flags_are_distinct_single_bits() {
        // The shader ANDs `flags` with each mask independently, so they must be
        // disjoint powers of two.
        assert_eq!(RESOLVE_FLAG_GTAO, 1);
        assert_eq!(RESOLVE_FLAG_IBL_SPECULAR, 2);
        assert_eq!(RESOLVE_FLAG_GTAO & RESOLVE_FLAG_IBL_SPECULAR, 0);
    }
}
