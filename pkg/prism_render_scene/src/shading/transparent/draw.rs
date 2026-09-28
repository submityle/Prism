//! Render commands for the transparent forward (WBOIT) draw pass.
//!
//! Structurally identical to [`super::super::raster`]'s draw path - the same
//! four bind groups (`view.main`, `view.empty`, scene, material) followed by an
//! instanced GPU-Scene mesh draw - but the [`bevy_render::render_phase::RenderCommand`]
//! trait is generic over the phase item, so each command is re-implemented for
//! [`TransparentOit3d`]. The draw itself is the exact same GPU-Scene instanced
//! draw (one instance addressed by [`GpuSceneInstanceAddress`]).

use bevy_ecs::{
    query::ROQueryItem,
    system::{lifetimeless::SRes, SystemParamItem},
};
use bevy_pbr::{MeshViewBindGroup, RenderMeshInstances};
use bevy_render::{
    mesh::{allocator::MeshAllocator, RenderMesh, RenderMeshBufferInfo},
    render_asset::RenderAssets,
    render_phase::{
        PhaseItem, RenderCommand, RenderCommandResult, SetItemPipeline, TrackedRenderPass,
    },
};

use crate::{buffers::GpuSceneBindGroup, GpuSceneInstanceAddress, MaterialBindGroup};

use super::phase::TransparentOit3d;

/// Full draw command tuple bound for each [`TransparentOit3d`] item: pipeline,
/// the four bind groups, then the instanced GPU-Scene mesh draw.
pub(crate) type DrawTransparentOit = (
    SetItemPipeline,
    SetOitViewBindGroup<0>,
    SetOitEmptyBindGroup<1>,
    SetOitSceneBindGroup<2>,
    SetOitMaterialBindGroup<3>,
    DrawOitMesh,
);

/// Binds the mesh view bind group (group 0) with its dynamic offsets.
pub(crate) struct SetOitViewBindGroup<const I: usize>;
impl<const I: usize> RenderCommand<TransparentOit3d> for SetOitViewBindGroup<I> {
    type Param = ();
    type ViewQuery = &'static MeshViewBindGroup;
    type ItemQuery = ();
    fn render<'w>(
        _: &TransparentOit3d,
        view: ROQueryItem<'w, '_, Self::ViewQuery>,
        _: Option<()>,
        _: SystemParamItem<'w, '_, Self::Param>,
        pass: &mut TrackedRenderPass<'w>,
    ) -> RenderCommandResult {
        pass.set_bind_group(I, &view.main, &view.main_offsets);
        RenderCommandResult::Success
    }
}

/// Binds the empty view bind group (group 1).
pub(crate) struct SetOitEmptyBindGroup<const I: usize>;
impl<const I: usize> RenderCommand<TransparentOit3d> for SetOitEmptyBindGroup<I> {
    type Param = ();
    type ViewQuery = &'static MeshViewBindGroup;
    type ItemQuery = ();
    fn render<'w>(
        _: &TransparentOit3d,
        view: ROQueryItem<'w, '_, Self::ViewQuery>,
        _: Option<()>,
        _: SystemParamItem<'w, '_, Self::Param>,
        pass: &mut TrackedRenderPass<'w>,
    ) -> RenderCommandResult {
        pass.set_bind_group(I, &view.empty, &[]);
        RenderCommandResult::Success
    }
}

macro_rules! set_global_bind_group {
    ($name:ident, $resource:ty, $field:ident) => {
        pub(crate) struct $name<const I: usize>;
        impl<const I: usize> RenderCommand<TransparentOit3d> for $name<I> {
            type Param = SRes<$resource>;
            type ViewQuery = ();
            type ItemQuery = ();
            fn render<'w>(
                _: &TransparentOit3d,
                _: (),
                _: Option<()>,
                resource: SystemParamItem<'w, '_, Self::Param>,
                pass: &mut TrackedRenderPass<'w>,
            ) -> RenderCommandResult {
                let Some(group) = &resource.into_inner().$field else {
                    return RenderCommandResult::Skip;
                };
                pass.set_bind_group(I, group, &[]);
                RenderCommandResult::Success
            }
        }
    };
}
set_global_bind_group!(SetOitSceneBindGroup, GpuSceneBindGroup, bind_group);
set_global_bind_group!(SetOitMaterialBindGroup, MaterialBindGroup, bind_group);

/// Instanced GPU-Scene mesh draw. Identical to
/// [`super::super::raster::DrawVisibilityMesh`]: the single instance is
/// addressed by the item's [`GpuSceneInstanceAddress`].
pub(crate) struct DrawOitMesh;
impl RenderCommand<TransparentOit3d> for DrawOitMesh {
    type Param = (
        SRes<RenderAssets<RenderMesh>>,
        SRes<RenderMeshInstances>,
        SRes<MeshAllocator>,
    );
    type ViewQuery = ();
    type ItemQuery = &'static GpuSceneInstanceAddress;
    fn render<'w>(
        item: &TransparentOit3d,
        _: (),
        address: Option<ROQueryItem<'w, '_, Self::ItemQuery>>,
        (meshes, instances, allocator): SystemParamItem<'w, '_, Self::Param>,
        pass: &mut TrackedRenderPass<'w>,
    ) -> RenderCommandResult {
        let meshes = meshes.into_inner();
        let instances = instances.into_inner();
        let allocator = allocator.into_inner();
        let Some(address) = address else {
            return RenderCommandResult::Skip;
        };
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
        let instance = address.index..address.index.saturating_add(1);
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
                    instance,
                );
            }
            RenderMeshBufferInfo::NonIndexed => pass.draw(vertices.range, instance),
        }
        RenderCommandResult::Success
    }
}
