//! GPU-ABI record for the virtual-shadow-map caster depth pass's per-page
//! uniform.
//!
//! [`GpuVsmCasterDepthView`] is the CPU twin of the `VsmCasterDepthView` uniform
//! block declared in `shaders/vsm_caster_depth.wesl`. One element is uploaded
//! per resident clipmap page into a
//! [`DynamicUniformBuffer`](bevy_render::render_resource::DynamicUniformBuffer);
//! the draw that fills a page's atlas tile binds the matching slice by dynamic
//! offset so the vertex stage transforms shadow casters through *that* page's
//! orthographic light projection.
//!
//! The record carries a single column-major `world -> light-clip` matrix. Unlike
//! the classic shadow atlas ([`super::super::super::shadow`]) the VSM caster pass
//! always stores raw wgpu NDC depth (`z in [0, 1]`), so it needs neither a light
//! position nor a storage-mode selector -- the projection matrix is the whole
//! per-page ABI. Field order and packing therefore match the shader's
//! `struct VsmCasterDepthView` exactly, and the layout is pinned by the unit
//! test below so a shader-side reorder cannot silently desync the upload.

use bevy_math::Mat4;
use bevy_render::render_resource::ShaderType;

/// GPU-ABI mirror of the `VsmCasterDepthView` uniform block in
/// `shaders/vsm_caster_depth.wesl`.
///
/// One element per resident clipmap page is packed into the caster-depth
/// [`DynamicUniformBuffer`](bevy_render::render_resource::DynamicUniformBuffer);
/// the page's draw selects its slice by dynamic offset.
#[derive(Clone, Copy, Default, ShaderType)]
pub(crate) struct GpuVsmCasterDepthView {
    /// Column-major `world -> light-clip` matrix for this page's orthographic
    /// light projection (wgpu clip: `z` in `[0, 1]`). Built on the CPU by
    /// [`super::projection::page_light_projection`].
    pub view_projection: Mat4,
}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy_render::render_resource::ShaderType;

    #[test]
    fn gpu_view_is_a_single_mat4() {
        // The uniform is exactly one column-major 4x4 matrix: 16 f32 lanes, no
        // padding. If the shader block or this record ever grows a field the
        // size assertion trips before an upload can desync the two.
        let size = u64::from(<GpuVsmCasterDepthView as ShaderType>::min_size().get());
        assert_eq!(size, 64, "VsmCasterDepthView must stay a bare mat4x4<f32>");
    }

    #[test]
    fn default_is_the_identity_matrix() {
        // A defaulted record is only ever used as the never-referenced
        // placeholder slice on a page-less frame (so the dynamic uniform buffer
        // is non-empty and its bind group stays valid). `#[derive(Default)]`
        // yields `Mat4`'s default, the identity, which is a well-formed matrix
        // that is trivially valid to upload and bind even though no draw ever
        // selects it.
        assert_eq!(
            GpuVsmCasterDepthView::default().view_projection,
            Mat4::IDENTITY
        );
    }
}
