use bevy_render::{
    impl_atomic_pod,
    render_resource::{AtomicPod, ShaderType},
};
use bytemuck::{Pod, Zeroable};
use prism_render_architecture::geometry::{GeometryLodRecord, GeometryPrimitiveKind};

pub(crate) const GEOMETRY_FLAG_ACTIVE: u32 = 1 << 0;
pub(crate) const GEOMETRY_FLAG_INDEXED: u32 = 1 << 1;
pub(crate) const GEOMETRY_FLAG_RESIDENT: u32 = 1 << 2;
pub(crate) const GEOMETRY_FLAG_FALLBACK: u32 = 1 << 3;

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Pod, ShaderType, Zeroable)]
pub struct RenderGeometryHeader {
    pub generation: u32,
    pub revision: u32,
    pub lod_offset: u32,
    pub lod_count: u32,
    pub primitive_start: u32,
    pub primitive_count: u32,
    pub flags: u32,
    pub _padding: u32,
}
impl_atomic_pod!(RenderGeometryHeader, RenderGeometryHeaderBlob);

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, ShaderType, Zeroable)]
pub struct RenderGeometryLod {
    pub element_count: u32,
    pub first_element: u32,
    pub base_vertex: i32,
    pub vertex_count: u32,
    pub vertex_buffer_class: u32,
    pub index_buffer_class: u32,
    pub flags: u32,
    pub level: u32,
    pub screen_error: f32,
    pub _padding: [u32; 3],
}
impl_atomic_pod!(RenderGeometryLod, RenderGeometryLodBlob);

impl RenderGeometryLod {
    pub(crate) fn from_record(
        lod: GeometryLodRecord,
        vertex_buffer_class: u32,
        index_buffer_class: u32,
    ) -> Self {
        let mut flags = GEOMETRY_FLAG_ACTIVE;
        flags |= u32::from(lod.primitive_kind == GeometryPrimitiveKind::Indexed)
            * GEOMETRY_FLAG_INDEXED;
        flags |= u32::from(lod.resident) * GEOMETRY_FLAG_RESIDENT;
        flags |= u32::from(lod.fallback) * GEOMETRY_FLAG_FALLBACK;
        Self {
            element_count: lod.element_count,
            first_element: lod.first_element,
            base_vertex: lod.base_vertex,
            vertex_count: lod.vertex_count,
            vertex_buffer_class,
            index_buffer_class,
            flags,
            level: lod.level,
            screen_error: lod.screen_error,
            _padding: [0; 3],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn geometry_rows_match_shader_contract() {
        assert_eq!(size_of::<RenderGeometryHeader>(), 32);
        assert_eq!(size_of::<RenderGeometryLod>(), 48);
    }
}
