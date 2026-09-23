use bevy_core_pipeline::core_3d::Opaque3d;
use bevy_ecs::{
    query::ROQueryItem,
    system::{lifetimeless::SRes, SystemParamItem},
};
use bevy_pbr::RenderMeshInstances;
use bevy_render::{
    mesh::{allocator::MeshAllocator, RenderMesh, RenderMeshBufferInfo},
    render_asset::RenderAssets,
    render_phase::{PhaseItem, RenderCommand, RenderCommandResult, TrackedRenderPass},
};

use crate::{
    visibility::runtime::UnifiedVisibilityState, GpuSceneInstanceAddress, RenderGpuScene,
};

const INDEXED_COMMAND_SIZE: u64 = size_of::<bevy_render::render_resource::DrawIndexedIndirectArgs>() as u64;
const NON_INDEXED_COMMAND_SIZE: u64 = size_of::<bevy_render::render_resource::DrawIndirectArgs>() as u64;

pub(crate) struct DrawGpuSceneIndirectBin;

impl RenderCommand<Opaque3d> for DrawGpuSceneIndirectBin {
    type Param = (
        SRes<RenderAssets<RenderMesh>>,
        SRes<RenderMeshInstances>,
        SRes<MeshAllocator>,
        SRes<UnifiedVisibilityState>,
        SRes<super::GpuSceneOpaqueIndirectEnabled>,
        SRes<crate::visibility::runtime::UnifiedVisibilitySettings>,
        SRes<crate::visibility::buffers::UnifiedVisibilityBuffers>,
        SRes<RenderGpuScene>,
    );
    type ViewQuery = &'static bevy_render::view::ExtractedView;
    type ItemQuery = &'static GpuSceneInstanceAddress;

    fn render<'w>(
        item: &Opaque3d,
        view: ROQueryItem<'w, '_, Self::ViewQuery>,
        address: Option<ROQueryItem<'w, '_, Self::ItemQuery>>,
        (meshes, instances, allocator, visibility, enabled, settings, buffers, scene): SystemParamItem<
            'w,
            '_,
            Self::Param,
        >,
        pass: &mut TrackedRenderPass<'w>,
    ) -> RenderCommandResult {
        let indirect_capable = enabled.0
            && settings.indirect_first_instance
            && visibility
                .draw_bins
                .iter()
                .any(|bins| visibility.retained_view(bins.view) == Some(view.retained_view_entity));
        if !indirect_capable {
            return super::draw::draw_direct_mesh(
                item,
                address,
                meshes.into_inner(),
                instances.into_inner(),
                allocator.into_inner(),
                pass,
            );
        }
        let Some(handle) = scene.handle_for_entity(item.entity()) else {
            return RenderCommandResult::Skip;
        };
        let Some(view_bins) = visibility.draw_bins.iter().find(|bins| {
            visibility.retained_view(bins.view) == Some(view.retained_view_entity)
        }) else {
            return RenderCommandResult::Skip;
        };
        if !bin_indirect_is_safe(view_bins, handle) {
            return super::draw::draw_direct_mesh(
                item,
                address,
                meshes.into_inner(),
                instances.into_inner(),
                allocator.into_inner(),
                pass,
            );
        }
        let Some(bin) = view_bins.bins.iter().find(|bin| bin.representative_scene == handle) else {
            return RenderCommandResult::Skip;
        };
        let meshes = meshes.into_inner();
        let instances = instances.into_inner();
        let allocator = allocator.into_inner();
        let buffers = buffers.into_inner();
        let Some(mesh_id) = instances.mesh_asset_id(item.main_entity()) else {
            return RenderCommandResult::Skip;
        };
        let Some(mesh) = meshes.get(mesh_id) else {
            return RenderCommandResult::Skip;
        };
        let Some(vertices) = allocator.mesh_vertex_slice(&mesh_id) else {
            return RenderCommandResult::Skip;
        };
        let Some((indexed, non_indexed)) = buffers.indirect() else {
            return RenderCommandResult::Skip;
        };
        pass.set_vertex_buffer(0, vertices.buffer.slice(..));
        match mesh.buffer_info {
            RenderMeshBufferInfo::Indexed { index_format, .. } => {
                let Some(indices) = allocator.mesh_index_slice(&mesh_id) else {
                    return RenderCommandResult::Skip;
                };
                pass.set_index_buffer(indices.buffer.slice(..), index_format);
                pass.multi_draw_indexed_indirect(
                    indexed,
                    (view_bins.global_candidate_start + bin.command_start) as u64
                        * INDEXED_COMMAND_SIZE,
                    bin.command_capacity,
                );
            }
            RenderMeshBufferInfo::NonIndexed => pass.multi_draw_indirect(
                non_indexed,
                (view_bins.global_candidate_start + bin.command_start) as u64
                    * NON_INDEXED_COMMAND_SIZE,
                bin.command_capacity,
            ),
        }
        RenderCommandResult::Success
    }
}

fn bin_indirect_is_safe(
    bins: &prism_render_visibility::ViewDrawBins,
    handle: prism_render_architecture::gpu_scene::SceneHandle,
) -> bool {
    bins.bins
        .iter()
        .find(|bin| bin.representative_scene == handle)
        .is_some_and(|bin| bin.command_capacity > 0)
}
