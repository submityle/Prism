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
        if bin.key.pass_mask & prism_render_visibility::RenderPassMask::OPAQUE.0 == 0 {
            return RenderCommandResult::Skip;
        }
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
        pass.set_vertex_buffer(0, vertices.buffer.slice(..));
        match mesh.buffer_info {
            RenderMeshBufferInfo::Indexed { index_format, .. } => {
                let Some(indices) = allocator.mesh_index_slice(&mesh_id) else {
                    return RenderCommandResult::Skip;
                };
                pass.set_index_buffer(indices.buffer.slice(..), index_format);
                for late in indirect_streams(settings.hzb_occlusion) {
                    let Some((indexed, _)) = buffers.indirect_for(late) else {
                        return RenderCommandResult::Skip;
                    };
                    pass.multi_draw_indexed_indirect(
                        indexed,
                        indirect_command_index(view_bins, bin) as u64 * INDEXED_COMMAND_SIZE,
                        bin.command_capacity,
                    );
                }
            }
            RenderMeshBufferInfo::NonIndexed => {
                for late in indirect_streams(settings.hzb_occlusion) {
                    let Some((_, non_indexed)) = buffers.indirect_for(late) else {
                        return RenderCommandResult::Skip;
                    };
                    pass.multi_draw_indirect(
                        non_indexed,
                        indirect_command_index(view_bins, bin) as u64 * NON_INDEXED_COMMAND_SIZE,
                        bin.command_capacity,
                    );
                }
            }
        }
        RenderCommandResult::Success
    }
}

fn indirect_streams(hzb_occlusion: bool) -> impl Iterator<Item = bool> {
    [false, true]
        .into_iter()
        .take(1 + usize::from(hzb_occlusion))
}

fn indirect_command_index(
    bins: &prism_render_visibility::ViewDrawBins,
    bin: &prism_render_visibility::DrawBinRange,
) -> u32 {
    bins.command_buffer_start.saturating_add(bin.command_start)
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

#[cfg(test)]
mod tests {
    use prism_render_architecture::{
        abi::GenerationalHandle,
        geometry::GeometryPrimitiveKind,
    };
    use prism_render_visibility::{
        build_view_draw_bins, DrawBinCandidate, DrawBinKey, RenderPassMask,
    };

    use super::{indirect_command_index, indirect_streams};

    fn handle(index: u32) -> GenerationalHandle {
        GenerationalHandle {
            index,
            generation: 1,
        }
    }

    #[test]
    fn multiview_commands_do_not_use_sparse_candidate_table_offsets() {
        let key = DrawBinKey {
            geometry: handle(2),
            lod_or_cluster: 0,
            pipeline_class: 3,
            vertex_buffer_class: 4,
            index_buffer_class: 5,
            indexed: true,
            primitive_kind: GeometryPrimitiveKind::Indexed,
            pass_mask: RenderPassMask::OPAQUE.0,
        };
        // Scene capacity is 16, but each view only owns two live command slots.
        let mut first = build_view_draw_bins(
            handle(10),
            16,
            [DrawBinCandidate {
                scene: handle(1),
                key,
                visibility_stages: prism_render_visibility::VisibilityStageMask::EARLY,
            }],
        );
        let mut second = build_view_draw_bins(
            handle(11),
            16,
            [DrawBinCandidate {
                scene: handle(9),
                key,
                visibility_stages: prism_render_visibility::VisibilityStageMask::EARLY,
            }],
        );
        first.command_buffer_start = 0;
        first.global_candidate_start = 0;
        second.command_buffer_start = 2;
        second.global_candidate_start = 16;

        assert_eq!(indirect_command_index(&first, &first.bins[0]), 0);
        assert_eq!(indirect_command_index(&second, &second.bins[0]), 2);
        assert_ne!(
            indirect_command_index(&second, &second.bins[0]),
            second.global_candidate_start + second.bins[0].command_start
        );
    }

    #[test]
    fn late_stream_is_consumed_only_when_two_phase_hzb_is_enabled() {
        assert_eq!(indirect_streams(false).collect::<Vec<_>>(), [false]);
        assert_eq!(indirect_streams(true).collect::<Vec<_>>(), [false, true]);
    }
}
