//! Render-world rows for the compute-friendly surface reconstruction table.

use bevy_render::{
    impl_atomic_pod,
    render_resource::{AtomicPod, ShaderType},
};
use bytemuck::{Pod, Zeroable};
use prism_render_shading::{GpuShadingPrimitive, GpuShadingVertex};

/// Geometry-table header addressed by a stable geometry index.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Pod, ShaderType, Zeroable)]
pub struct RenderShadingGeometryHeader {
    pub generation: u32,
    pub revision: u32,
    pub vertex_offset: u32,
    pub vertex_count: u32,
    pub primitive_offset: u32,
    pub primitive_count: u32,
    pub flags: u32,
    pub _padding: u32,
}
impl_atomic_pod!(RenderShadingGeometryHeader, RenderShadingGeometryHeaderBlob);

impl RenderShadingGeometryHeader {
    /// Resolves a geometry-local vertex index into the packed vertex table.
    ///
    /// Primitive rows intentionally retain local indices. Consumers that bind
    /// the packed render-world vertex buffer must use this checked translation
    /// rather than indexing that buffer with the primitive row directly.
    #[must_use]
    pub const fn global_vertex_index(self, local_index: u32) -> Option<u32> {
        if local_index >= self.vertex_count {
            return None;
        }
        self.vertex_offset.checked_add(local_index)
    }
}
/// Stable vertex row used by compute resolve and offline export.
pub type RenderShadingVertex = GpuShadingVertex;

/// Stable triangle row used by compute resolve and offline export.
pub type RenderShadingPrimitive = GpuShadingPrimitive;

pub type RenderShadingGeometry = prism_render_shading::ShadingGeometry;
pub type ShadingGeometryBuildError = prism_render_shading::ShadingGeometryBuildError;

pub use prism_render_shading::{
    build_shading_geometry, SHADING_GEOMETRY_FLAG_ACTIVE,
    SHADING_GEOMETRY_FLAG_INVALID, SHADING_GEOMETRY_FLAG_MISSING_NORMAL,
    SHADING_GEOMETRY_FLAG_MISSING_TANGENT, SHADING_GEOMETRY_FLAG_MISSING_UV,
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rows_match_compute_surface_abi() {
        assert_eq!(size_of::<RenderShadingGeometryHeader>(), 32);
        assert_eq!(size_of::<RenderShadingVertex>(), 64);
        assert_eq!(size_of::<RenderShadingPrimitive>(), 16);
    }

    #[test]
    fn header_resolves_local_indices_against_the_packed_vertex_base() {
        let header = RenderShadingGeometryHeader {
            vertex_offset: 11,
            vertex_count: 3,
            ..Default::default()
        };
        assert_eq!(header.global_vertex_index(0), Some(11));
        assert_eq!(header.global_vertex_index(2), Some(13));
        assert_eq!(header.global_vertex_index(3), None);
    }
}
