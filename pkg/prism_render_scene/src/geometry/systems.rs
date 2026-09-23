use bevy_asset::AssetId;
use bevy_ecs::prelude::*;
use bevy_mesh::Mesh;
use bevy_render::{
    mesh::{allocator::MeshAllocator, RenderMesh, RenderMeshBufferInfo},
    render_asset::RenderAssets,
    renderer::{RenderDevice, RenderQueue},
};
use prism_render_architecture::geometry::{
    GeometryLodRecord, GeometryPrimitiveKind, GeometryRecord,
};

use super::{buffers::RenderGeometryBuffers, runtime::RenderGeometryRegistry};
use crate::scene::RenderGpuScene;

pub(crate) fn sync_geometry_registry(
    scene: Res<RenderGpuScene>,
    meshes: Res<RenderAssets<RenderMesh>>,
    allocator: Res<MeshAllocator>,
    mut registry: ResMut<RenderGeometryRegistry>,
    mut buffers: ResMut<RenderGeometryBuffers>,
) {
    let assets: Vec<(AssetId<Mesh>, _)> = scene.geometry_assets().collect();
    let mut changed = false;
    for (asset, handle) in assets {
        let Some(mesh) = meshes.get(asset) else { continue };
        let Some(vertices) = allocator.mesh_vertex_slice(&asset) else { continue };
        let vertex_buffer_class = registry.buffer_class(vertices.buffer);
        let topology = mesh.primitive_topology();
        let (primitive_kind, element_count, first_element, base_vertex, index_buffer_class) = match mesh.buffer_info {
            RenderMeshBufferInfo::Indexed { count, .. } => {
                let Some(indices) = allocator.mesh_index_slice(&asset) else { continue };
                let index_buffer_class = registry.buffer_class(indices.buffer);
                (GeometryPrimitiveKind::Indexed, count, indices.range.start, vertices.range.start as i32, index_buffer_class)
            }
            RenderMeshBufferInfo::NonIndexed => {
                (GeometryPrimitiveKind::NonIndexed, mesh.vertex_count, vertices.range.start, 0, 0)
            }
        };
        let record = GeometryRecord {
            handle,
            revision: 1,
            vertex_buffer_class,
            index_buffer_class,
            primitive_start: 0,
            primitive_count: primitive_count(topology, element_count),
            lods: vec![GeometryLodRecord {
                level: 0,
                primitive_kind,
                element_count,
                first_element,
                base_vertex,
                vertex_count: mesh.vertex_count,
                screen_error: 0.0,
                resident: true,
                fallback: true,
            }],
        };
        if registry.record(handle) != Some(&record) {
            registry.upsert(asset, record);
            changed = true;
        }
    }
    if changed || registry.is_dirty() {
        let _ = registry.take_dirty();
        buffers.rebuild(&registry);
    }
}

fn primitive_count(topology: bevy_render::render_resource::PrimitiveTopology, elements: u32) -> u32 {
    use bevy_render::render_resource::PrimitiveTopology::*;
    match topology {
        PointList => elements,
        LineList => elements / 2,
        LineStrip => elements.saturating_sub(1),
        TriangleList => elements / 3,
        TriangleStrip => elements.saturating_sub(2),
    }
}

pub(crate) fn upload_geometry_buffers(
    mut buffers: ResMut<RenderGeometryBuffers>,
    device: Res<RenderDevice>,
    queue: Res<RenderQueue>,
) {
    buffers.upload(&device, &queue);
}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy_render::render_resource::PrimitiveTopology;

    #[test]
    fn primitive_ranges_follow_topology() {
        assert_eq!(primitive_count(PrimitiveTopology::TriangleList, 12), 4);
        assert_eq!(primitive_count(PrimitiveTopology::TriangleStrip, 12), 10);
        assert_eq!(primitive_count(PrimitiveTopology::LineList, 12), 6);
        assert_eq!(primitive_count(PrimitiveTopology::LineStrip, 12), 11);
        assert_eq!(primitive_count(PrimitiveTopology::PointList, 12), 12);
    }
}
