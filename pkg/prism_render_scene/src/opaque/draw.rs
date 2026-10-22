use bevy_core_pipeline::core_3d::Opaque3d;
use bevy_ecs::{
    query::ROQueryItem,
    system::{lifetimeless::SRes, SystemParamItem},
};
use bevy_pbr::{SetMeshViewBindGroup, SetMeshViewEmptyBindGroup};
use bevy_render::{
    mesh::{allocator::MeshAllocator, RenderMesh, RenderMeshBufferInfo},
    render_asset::RenderAssets,
    render_phase::{RenderCommand, RenderCommandResult, SetItemPipeline, TrackedRenderPass},
};

use crate::{
    buffers::GpuSceneBindGroup,
    geometry::{indexed_draw_range, non_indexed_draw_range, select_render_mesh_asset},
    GpuSceneInstanceAddress, MaterialBindGroup,
};
use prism_render_architecture::geometry::MeshConsumer;

pub(crate) type DrawGpuSceneOpaque = (
    SetItemPipeline,
    SetMeshViewBindGroup<0>,
    SetMeshViewEmptyBindGroup<1>,
    SetGpuSceneBindGroup<2>,
    SetMaterialBindGroup<3>,
    SetOpaqueDebugMetadataBindGroup<4>,
    super::indirect::DrawGpuSceneIndirectBin,
);

pub(crate) struct SetOpaqueDebugMetadataBindGroup<const I: usize>;

impl<const I: usize> RenderCommand<Opaque3d> for SetOpaqueDebugMetadataBindGroup<I> {
    type Param = (
        SRes<super::debug_metadata::OpaqueDebugMetadataBuffers>,
        SRes<crate::visibility::runtime::UnifiedVisibilityState>,
        SRes<super::GpuSceneDebugView>,
    );
    type ViewQuery = &'static bevy_render::view::ExtractedView;
    type ItemQuery = ();

    fn render<'w>(
        _: &Opaque3d,
        view: ROQueryItem<'w, '_, Self::ViewQuery>,
        _: Option<()>,
        (metadata, visibility, debug_view): SystemParamItem<'w, '_, Self::Param>,
        pass: &mut TrackedRenderPass<'w>,
    ) -> RenderCommandResult {
        if !matches!(
            *debug_view,
            debug if super::pipeline::uses_debug_metadata(debug)
        ) {
            return RenderCommandResult::Success;
        }
        let metadata = metadata.into_inner();
        let visibility = visibility.into_inner();
        let Some(handle) = visibility
            .views
            .iter()
            .find(|record| {
                visibility
                    .retained_view(record.handle)
                    .is_some_and(|retained| retained == view.retained_view_entity)
            })
            .map(|record| record.handle)
        else {
            return RenderCommandResult::Skip;
        };
        let Some(bind_group) = metadata.bind_group(handle) else {
            return RenderCommandResult::Skip;
        };
        pass.set_bind_group(I, bind_group, &[]);
        RenderCommandResult::Success
    }
}

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

pub(crate) struct SetMaterialBindGroup<const I: usize>;

impl<const I: usize> RenderCommand<Opaque3d> for SetMaterialBindGroup<I> {
    type Param = SRes<MaterialBindGroup>;
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

pub(super) fn draw_direct_mesh<'w>(
    item: &Opaque3d,
    address: Option<ROQueryItem<'w, '_, &'static GpuSceneInstanceAddress>>,
    meshes: &'w RenderAssets<RenderMesh>,
    allocator: &'w MeshAllocator,
    registry: &'w crate::RenderGeometryRegistry,
    pass: &mut TrackedRenderPass<'w>,
) -> RenderCommandResult {
    let Some(address) = address else {
        return RenderCommandResult::Skip;
    };
    let Ok(mesh_id) = item.bin_key.asset_id.try_typed::<bevy_mesh::Mesh>() else {
        return RenderCommandResult::Skip;
    };
    let Ok(selection) = select_render_mesh_asset(registry, mesh_id, MeshConsumer::MainView) else {
        return RenderCommandResult::Skip;
    };
    let Some(mesh) = meshes.get(selection.asset) else {
        return RenderCommandResult::Skip;
    };
    let Some(vertices) = allocator.mesh_vertex_slice(&selection.asset) else {
        return RenderCommandResult::Skip;
    };
    pass.set_vertex_buffer(0, vertices.buffer.slice(..));
    let instances = address.index..address.index.saturating_add(1);
    match mesh.buffer_info {
        RenderMeshBufferInfo::Indexed {
            index_format,
            count,
        } => {
            let Some(indices) = allocator.mesh_index_slice(&selection.asset) else {
                return RenderCommandResult::Skip;
            };
            pass.set_index_buffer(indices.buffer.slice(..), index_format);
            let Some(range) = indexed_draw_range(
                &selection.lod,
                vertices.range.clone(),
                indices.range.clone(),
                count,
            ) else {
                return RenderCommandResult::Skip;
            };
            pass.draw_indexed(range, selection.lod.base_vertex, instances);
        }
        RenderMeshBufferInfo::NonIndexed => {
            let Some(range) =
                non_indexed_draw_range(&selection.lod, vertices.range.clone(), mesh.vertex_count)
            else {
                return RenderCommandResult::Skip;
            };
            pass.draw(range, instances);
        }
    }
    RenderCommandResult::Success
}
