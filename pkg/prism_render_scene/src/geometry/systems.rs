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
        let Some(mesh) = meshes.get(asset) else {
            continue;
        };
        let base = draw_range(&asset, mesh, &allocator, &mut registry);
        let Some(base) = base else { continue };
        let mut lods = vec![GeometryLodRecord {
            level: 0,
            primitive_kind: base.primitive_kind,
            element_count: base.element_count,
            first_element: base.first_element,
            base_vertex: base.base_vertex,
            vertex_count: mesh.vertex_count,
            screen_error: 0.0,
            resident: true,
            fallback: true,
        }];
        let mut lod_assets = Vec::new();
        for (level, lod_asset, screen_error) in scene.geometry_lod_assets(handle) {
            let Some(lod_mesh) = meshes.get(lod_asset) else {
                lods.push(GeometryLodRecord {
                    level,
                    screen_error,
                    ..Default::default()
                });
                continue;
            };
            let Some(range) = draw_range(&lod_asset, lod_mesh, &allocator, &mut registry) else {
                lods.push(GeometryLodRecord {
                    level,
                    screen_error,
                    ..Default::default()
                });
                continue;
            };
            lods.push(GeometryLodRecord {
                level,
                primitive_kind: range.primitive_kind,
                element_count: range.element_count,
                first_element: range.first_element,
                base_vertex: range.base_vertex,
                vertex_count: lod_mesh.vertex_count,
                screen_error,
                resident: true,
                fallback: false,
            });
            lod_assets.push((
                level,
                lod_asset,
                range.vertex_buffer_class,
                range.index_buffer_class,
            ));
        }
        lods.sort_by_key(|lod| lod.level);
        let record = GeometryRecord {
            handle,
            revision: 1,
            vertex_buffer_class: base.vertex_buffer_class,
            index_buffer_class: base.index_buffer_class,
            primitive_start: 0,
            primitive_count: primitive_count(mesh.primitive_topology(), base.element_count),
            lods,
        };
        if registry.record(handle) != Some(&record) {
            registry.upsert(asset, record);
            changed = true;
        }
        // The geometry record describes draw ranges, while this mapping owns
        // the prepared Bevy asset that supplies each range. Those identities
        // may change during reload without changing any range metadata, so
        // refresh the mapping independently of record equality.
        for (level, lod_asset, vertex_buffer_class, index_buffer_class) in lod_assets {
            if (registry.lod_asset(handle, level) != Some(lod_asset)
                || registry.buffer_classes_for_lod(handle, level)
                    != Some((vertex_buffer_class, index_buffer_class)))
                && registry.upsert_lod_asset(
                    handle,
                    lod_asset,
                    level,
                    vertex_buffer_class,
                    index_buffer_class,
                )
            {
                changed = true;
            }
        }
    }
    if changed || registry.is_dirty() {
        let _ = registry.take_dirty();
        buffers.rebuild(&registry);
    }
}

#[derive(Clone, Copy)]
struct DrawRange {
    primitive_kind: GeometryPrimitiveKind,
    element_count: u32,
    first_element: u32,
    base_vertex: i32,
    vertex_buffer_class: u32,
    index_buffer_class: u32,
}

fn draw_range(
    asset: &AssetId<Mesh>,
    mesh: &RenderMesh,
    allocator: &MeshAllocator,
    registry: &mut RenderGeometryRegistry,
) -> Option<DrawRange> {
    let vertices = allocator.mesh_vertex_slice(asset)?;
    let vertex_buffer_class = registry.buffer_class(vertices.buffer);
    match mesh.buffer_info {
        RenderMeshBufferInfo::Indexed { count, .. } => {
            let indices = allocator.mesh_index_slice(asset)?;
            Some(DrawRange {
                primitive_kind: GeometryPrimitiveKind::Indexed,
                element_count: count,
                first_element: indices.range.start,
                base_vertex: vertices.range.start as i32,
                vertex_buffer_class,
                index_buffer_class: registry.buffer_class(indices.buffer),
            })
        }
        RenderMeshBufferInfo::NonIndexed => Some(DrawRange {
            primitive_kind: GeometryPrimitiveKind::NonIndexed,
            element_count: mesh.vertex_count,
            first_element: vertices.range.start,
            base_vertex: 0,
            vertex_buffer_class,
            index_buffer_class: 0,
        }),
    }
}

fn primitive_count(
    topology: bevy_render::render_resource::PrimitiveTopology,
    elements: u32,
) -> u32 {
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
