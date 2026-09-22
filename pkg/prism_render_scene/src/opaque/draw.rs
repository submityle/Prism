use bevy_core_pipeline::core_3d::Opaque3d;
use bevy_ecs::{
    query::ROQueryItem,
    system::{lifetimeless::SRes, SystemParamItem},
};
use bevy_pbr::{RenderMeshInstances, SetMeshViewBindGroup, SetMeshViewEmptyBindGroup};
use bevy_render::{
    mesh::{allocator::MeshAllocator, RenderMesh, RenderMeshBufferInfo},
    render_asset::RenderAssets,
    render_phase::{
        PhaseItem, RenderCommand, RenderCommandResult, SetItemPipeline, TrackedRenderPass,
    },
};

use crate::{buffers::GpuSceneBindGroup, GpuSceneInstanceAddress};

pub(crate) type DrawGpuSceneOpaque = (
    SetItemPipeline,
    SetMeshViewBindGroup<0>,
    SetMeshViewEmptyBindGroup<1>,
    SetGpuSceneBindGroup<2>,
    DrawGpuSceneMesh,
);

pub(crate) struct SetGpuSceneBindGroup<const I: usize>;

impl<const I: usize> RenderCommand<Opaque3d> for SetGpuSceneBindGroup<I> {
    type Param = SRes<GpuSceneBindGroup>;
    type ViewQuery = ();
    type ItemQuery = ();

    fn render<'w>(
        _: &Opaque3d,
        _: (),
        _: Option<()>,
        bindings: SystemParamItem<'w, '_, Self::Param>,
        pass: &mut TrackedRenderPass<'w>,
    ) -> RenderCommandResult {
        let Some(bind_group) = &bindings.into_inner().bind_group else {
            return RenderCommandResult::Skip;
        };
        pass.set_bind_group(I, bind_group, &[]);
        RenderCommandResult::Success
    }
}

pub(crate) struct DrawGpuSceneMesh;

impl RenderCommand<Opaque3d> for DrawGpuSceneMesh {
    type Param = (
        SRes<RenderAssets<RenderMesh>>,
        SRes<RenderMeshInstances>,
        SRes<MeshAllocator>,
    );
    type ViewQuery = ();
    type ItemQuery = &'static GpuSceneInstanceAddress;

    fn render<'w>(
        item: &Opaque3d,
        _: (),
        address: Option<ROQueryItem<'w, '_, Self::ItemQuery>>,
        (meshes, instances, allocator): SystemParamItem<'w, '_, Self::Param>,
        pass: &mut TrackedRenderPass<'w>,
    ) -> RenderCommandResult {
        let Some(address) = address else {
            return RenderCommandResult::Skip;
        };
        let Some(mesh_id) = instances.mesh_asset_id(item.main_entity()) else {
            return RenderCommandResult::Skip;
        };
        let Some(mesh) = meshes.into_inner().get(mesh_id) else {
            return RenderCommandResult::Skip;
        };
        let allocator = allocator.into_inner();
        let Some(vertices) = allocator.mesh_vertex_slice(&mesh_id) else {
            return RenderCommandResult::Skip;
        };
        pass.set_immediates(
            0,
            bytemuck::cast_slice(&[address.index, address.generation]),
        );
        pass.set_vertex_buffer(0, vertices.buffer.slice(..));
        match mesh.buffer_info {
            RenderMeshBufferInfo::Indexed {
                index_format,
                count,
            } => {
                let Some(indices) = allocator.mesh_index_slice(&mesh_id) else {
                    return RenderCommandResult::Skip;
                };
                pass.set_index_buffer(indices.buffer.slice(..), index_format);
                pass.draw_indexed(
                    indices.range.start..indices.range.start + count,
                    vertices.range.start as i32,
                    0..1,
                );
            }
            RenderMeshBufferInfo::NonIndexed => pass.draw(vertices.range, 0..1),
        }
        RenderCommandResult::Success
    }
}
