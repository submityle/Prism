use core::ops::Range;

use bevy_asset::{load_embedded_asset, Handle};
use bevy_core_pipeline::core_3d::CORE_3D_DEPTH_FORMAT;
use bevy_ecs::{
    prelude::*,
    query::ROQueryItem,
    system::{lifetimeless::SRes, SystemParamItem},
};
use bevy_material::{descriptor::BindGroupLayoutDescriptor, labels::DrawFunctionId};
use bevy_mesh::{Mesh, MeshAttributeCompressionFlags, MeshVertexBufferLayoutRef};
use bevy_pbr::{
    MeshPipeline, MeshPipelineKey, MeshPipelineViewLayoutKey, MeshViewBindGroup,
    RenderMeshInstances, ViewKeyCache,
};
use bevy_render::{
    camera::ExtractedCamera,
    mesh::{allocator::MeshAllocator, RenderMesh, RenderMeshBufferInfo},
    render_asset::RenderAssets,
    render_phase::{
        BinnedPhaseItem, BinnedRenderPhaseType, CachedRenderPipelinePhaseItem, DrawFunctions,
        InputUniformIndex, PhaseItem, PhaseItemBatchSetKey, PhaseItemExtraIndex, RenderCommand,
        RenderCommandResult, SetItemPipeline, TrackedRenderPass, ViewBinnedRenderPhases,
    },
    render_resource::*,
    renderer::{RenderContext, ViewQuery},
    sync_world::MainEntity,
    view::{ExtractedView, Msaa, ViewDepthStencilTexture},
};
use bevy_shader::Shader;

use crate::{buffers::GpuSceneBindGroup, GpuSceneInstanceAddress, MaterialBindGroup};

use super::{resources::ViewVisibilityBuffer, runtime::PrismShadingSettings};

/// Integer clear values consumed by the visibility resolve. The all-ones scene and
/// material indices make an untouched pixel distinguishable from valid index zero.
pub(crate) const VISIBILITY_IDS_CLEAR: wgpu_types::Color = wgpu_types::Color {
    r: u32::MAX as f64,
    g: 0.0,
    b: u32::MAX as f64,
    a: u32::MAX as f64,
};
pub(crate) const VISIBILITY_METADATA_CLEAR: wgpu_types::Color = wgpu_types::Color {
    r: u32::MAX as f64,
    g: 0.0,
    b: 0.0,
    a: 0.0,
};

#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(crate) struct VisibilityBatchSetKey {
    pipeline: CachedRenderPipelineId,
    draw_function: DrawFunctionId,
    indexed: bool,
}

impl PhaseItemBatchSetKey for VisibilityBatchSetKey {
    fn indexed(&self) -> bool {
        self.indexed
    }
}

#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(crate) struct VisibilityBinKey(bevy_asset::UntypedAssetId);

pub(crate) struct Visibility3d {
    batch_set_key: VisibilityBatchSetKey,
    _bin_key: VisibilityBinKey,
    representative_entity: (Entity, MainEntity),
    batch_range: Range<u32>,
    extra_index: PhaseItemExtraIndex,
}

impl PhaseItem for Visibility3d {
    fn entity(&self) -> Entity {
        self.representative_entity.0
    }
    fn main_entity(&self) -> MainEntity {
        self.representative_entity.1
    }
    fn draw_function(&self) -> DrawFunctionId {
        self.batch_set_key.draw_function
    }
    fn batch_range(&self) -> &Range<u32> {
        &self.batch_range
    }
    fn batch_range_mut(&mut self) -> &mut Range<u32> {
        &mut self.batch_range
    }
    fn extra_index(&self) -> PhaseItemExtraIndex {
        self.extra_index.clone()
    }
    fn batch_range_and_extra_index_mut(&mut self) -> (&mut Range<u32>, &mut PhaseItemExtraIndex) {
        (&mut self.batch_range, &mut self.extra_index)
    }
}

impl BinnedPhaseItem for Visibility3d {
    type BatchSetKey = VisibilityBatchSetKey;
    type BinKey = VisibilityBinKey;
    fn new(
        batch_set_key: Self::BatchSetKey,
        bin_key: Self::BinKey,
        representative_entity: (Entity, MainEntity),
        batch_range: Range<u32>,
        extra_index: PhaseItemExtraIndex,
    ) -> Self {
        Self {
            batch_set_key,
            _bin_key: bin_key,
            representative_entity,
            batch_range,
            extra_index,
        }
    }
}

impl CachedRenderPipelinePhaseItem for Visibility3d {
    fn cached_pipeline(&self) -> CachedRenderPipelineId {
        self.batch_set_key.pipeline
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(crate) struct VisibilityRasterPipelineKey {
    mesh: MeshPipelineKey,
}

#[derive(Resource)]
pub(crate) struct VisibilityRasterPipeline {
    mesh_pipeline: MeshPipeline,
    scene_layout: BindGroupLayoutDescriptor,
    material_layout: BindGroupLayoutDescriptor,
    shader: Handle<Shader>,
}

pub(crate) fn init_visibility_raster(
    mut commands: Commands,
    mesh_pipeline: Res<MeshPipeline>,
    scene: Res<GpuSceneBindGroup>,
    material: Res<MaterialBindGroup>,
    asset_server: Res<bevy_asset::AssetServer>,
) {
    commands.insert_resource(VisibilityRasterPipeline {
        mesh_pipeline: mesh_pipeline.clone(),
        scene_layout: scene.layout_descriptor.clone(),
        material_layout: material.layout_descriptor.clone(),
        shader: load_embedded_asset!(asset_server.as_ref(), "../shaders/visibility_raster.wesl"),
    });
}

impl SpecializedMeshPipeline for VisibilityRasterPipeline {
    type Key = VisibilityRasterPipelineKey;

    fn specialize(
        &self,
        key: Self::Key,
        layout: &MeshVertexBufferLayoutRef,
    ) -> Result<RenderPipelineDescriptor, SpecializedMeshPipelineError> {
        let mut shader_defs = Vec::new();
        if layout
            .0
            .get_attribute_compression()
            .contains(MeshAttributeCompressionFlags::COMPRESS_POSITION)
        {
            shader_defs.push("VERTEX_POSITIONS_COMPRESSED".into());
        }
        let vertex_layout = layout
            .0
            .get_layout(&[Mesh::ATTRIBUTE_POSITION.at_shader_location(0)])?;
        let view = self
            .mesh_pipeline
            .get_view_layout(MeshPipelineViewLayoutKey::from(key.mesh));
        Ok(RenderPipelineDescriptor {
            label: Some("prism visibility raster".into()),
            layout: vec![
                view.main_layout,
                view.empty_layout,
                self.scene_layout.clone(),
                self.material_layout.clone(),
            ],
            immediate_size: 0,
            vertex: VertexState {
                shader: self.shader.clone(),
                shader_defs: shader_defs.clone(),
                buffers: vec![vertex_layout],
                ..Default::default()
            },
            fragment: Some(FragmentState {
                shader: self.shader.clone(),
                shader_defs,
                targets: vec![
                    Some(ColorTargetState {
                        format: TextureFormat::Rgba32Uint,
                        blend: None,
                        write_mask: ColorWrites::ALL,
                    }),
                    Some(ColorTargetState {
                        format: TextureFormat::Rgba32Uint,
                        blend: None,
                        write_mask: ColorWrites::ALL,
                    }),
                ],
                ..Default::default()
            }),
            primitive: PrimitiveState {
                topology: key.mesh.primitive_topology(),
                strip_index_format: key.mesh.strip_index_format(),
                front_face: FrontFace::Ccw,
                cull_mode: Some(Face::Back),
                ..Default::default()
            },
            depth_stencil: Some(DepthStencilState {
                format: CORE_3D_DEPTH_FORMAT,
                depth_write_enabled: Some(true),
                depth_compare: Some(CompareFunction::GreaterEqual),
                stencil: StencilState::default(),
                bias: DepthBiasState::default(),
            }),
            multisample: MultisampleState::default(),
            ..Default::default()
        })
    }
}

pub(crate) type DrawVisibilityRaster = (
    SetItemPipeline,
    SetVisibilityViewBindGroup<0>,
    SetVisibilityEmptyBindGroup<1>,
    SetVisibilitySceneBindGroup<2>,
    SetVisibilityMaterialBindGroup<3>,
    DrawVisibilityMesh,
);

pub(crate) struct SetVisibilityViewBindGroup<const I: usize>;
impl<const I: usize> RenderCommand<Visibility3d> for SetVisibilityViewBindGroup<I> {
    type Param = ();
    type ViewQuery = &'static MeshViewBindGroup;
    type ItemQuery = ();
    fn render<'w>(
        _: &Visibility3d,
        view: ROQueryItem<'w, '_, Self::ViewQuery>,
        _: Option<()>,
        _: SystemParamItem<'w, '_, Self::Param>,
        pass: &mut TrackedRenderPass<'w>,
    ) -> RenderCommandResult {
        pass.set_bind_group(I, &view.main, &view.main_offsets);
        RenderCommandResult::Success
    }
}
pub(crate) struct SetVisibilityEmptyBindGroup<const I: usize>;
impl<const I: usize> RenderCommand<Visibility3d> for SetVisibilityEmptyBindGroup<I> {
    type Param = ();
    type ViewQuery = &'static MeshViewBindGroup;
    type ItemQuery = ();
    fn render<'w>(
        _: &Visibility3d,
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
        impl<const I: usize> RenderCommand<Visibility3d> for $name<I> {
            type Param = SRes<$resource>;
            type ViewQuery = ();
            type ItemQuery = ();
            fn render<'w>(
                _: &Visibility3d,
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
set_global_bind_group!(SetVisibilitySceneBindGroup, GpuSceneBindGroup, bind_group);
set_global_bind_group!(
    SetVisibilityMaterialBindGroup,
    MaterialBindGroup,
    bind_group
);

pub(crate) struct DrawVisibilityMesh;
impl RenderCommand<Visibility3d> for DrawVisibilityMesh {
    type Param = (
        SRes<RenderAssets<RenderMesh>>,
        SRes<RenderMeshInstances>,
        SRes<MeshAllocator>,
    );
    type ViewQuery = ();
    type ItemQuery = &'static GpuSceneInstanceAddress;
    fn render<'w>(
        item: &Visibility3d,
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

#[expect(
    clippy::too_many_arguments,
    reason = "Visibility queue joins retained scene, view, and mesh state."
)]
pub(crate) fn queue_visibility_raster(
    settings: Res<PrismShadingSettings>,
    pipeline_cache: Res<PipelineCache>,
    pipeline: Res<VisibilityRasterPipeline>,
    mut pipelines: ResMut<SpecializedMeshPipelines<VisibilityRasterPipeline>>,
    draw_functions: Res<DrawFunctions<Visibility3d>>,
    mut phases: ResMut<ViewBinnedRenderPhases<Visibility3d>>,
    views: Query<&ExtractedView>,
    view_keys: Res<ViewKeyCache>,
    render_meshes: Res<RenderAssets<RenderMesh>>,
    render_instances: Res<RenderMeshInstances>,
    scene_instances: Query<&GpuSceneInstanceAddress>,
    visibility: Res<crate::visibility::runtime::UnifiedVisibilityState>,
    scene: Res<crate::RenderGpuScene>,
    mut previously_queued: Local<
        bevy_platform::collections::HashMap<
            bevy_render::view::RetainedViewEntity,
            bevy_render::sync_world::MainEntityHashSet,
        >,
    >,
) {
    if !settings.enable_visibility_buffer {
        phases.clear();
        previously_queued.clear();
        return;
    }
    let draw_function = draw_functions.read().id::<DrawVisibilityRaster>();
    for view in &views {
        if view_keys
            .get(&view.retained_view_entity)
            .is_none_or(|key| key.msaa_samples() != 1)
        {
            if let Some(phase) = phases.get_mut(&view.retained_view_entity)
                && let Some(previous) = previously_queued.remove(&view.retained_view_entity)
            {
                for entity in previous {
                    phase.remove(entity);
                }
            }
            continue;
        }
        let Some(view_output) = visibility
            .frame
            .views
            .iter()
            .find(|(handle, _)| {
                visibility.retained_view(**handle) == Some(view.retained_view_entity)
            })
            .map(|(_, output)| output)
        else {
            if let Some(phase) = phases.get_mut(&view.retained_view_entity)
                && let Some(previous) = previously_queued.remove(&view.retained_view_entity)
            {
                for entity in previous {
                    phase.remove(entity);
                }
            }
            continue;
        };
        let Some(&view_key) = view_keys.get(&view.retained_view_entity) else {
            continue;
        };
        phases.prepare_for_new_frame(
            view.retained_view_entity,
            bevy_render::batching::gpu_preprocessing::GpuPreprocessingMode::None,
        );
        let Some(phase) = phases.get_mut(&view.retained_view_entity) else {
            continue;
        };
        let previous = previously_queued.remove(&view.retained_view_entity);
        let mut current = bevy_render::sync_world::MainEntityHashSet::default();
        let mut candidates = Vec::new();
        for work in &visibility.frame.work_items[view_output.visible_instances.start as usize
            ..view_output
                .visible_instances
                .start
                .saturating_add(view_output.visible_instances.count) as usize]
        {
            if work.pass_mask.0 & prism_render_visibility::RenderPassMask::OPAQUE.0 == 0 {
                continue;
            }
            let Some((render_entity, main_entity)) = scene.entity_binding_for_handle(work.scene)
            else {
                continue;
            };
            candidates.push((render_entity, main_entity));
        }
        for (render_entity, main_entity) in candidates {
            let Ok(address) = scene_instances.get(render_entity) else {
                continue;
            };
            let Some(mesh_id) = render_instances.mesh_asset_id(main_entity) else {
                continue;
            };
            let Some(mesh) = render_meshes.get(mesh_id) else {
                continue;
            };
            let key = view_key
                | MeshPipelineKey::from_primitive_topology_and_strip_index(
                    mesh.primitive_topology(),
                    mesh.index_format(),
                );
            let Ok(pipeline_id) = pipelines.specialize(
                &pipeline_cache,
                &pipeline,
                VisibilityRasterPipelineKey { mesh: key },
                &mesh.layout,
            ) else {
                continue;
            };
            current.insert(main_entity);
            phase.add(
                VisibilityBatchSetKey {
                    pipeline: pipeline_id,
                    draw_function,
                    indexed: matches!(mesh.buffer_info, RenderMeshBufferInfo::Indexed { .. }),
                },
                VisibilityBinKey(mesh_id.into()),
                (render_entity, main_entity),
                InputUniformIndex(address.index),
                BinnedRenderPhaseType::NonMesh,
            );
        }
        if let Some(previous) = previous {
            for entity in previous {
                if !current.contains(&entity) {
                    phase.remove(entity);
                }
            }
        }
        previously_queued.insert(view.retained_view_entity, current);
    }
}

pub(crate) fn visibility_raster_pass(
    world: &World,
    view: ViewQuery<(
        &ExtractedCamera,
        &ExtractedView,
        &ViewDepthStencilTexture,
        &ViewVisibilityBuffer,
        Option<&Msaa>,
    )>,
    phases: Res<ViewBinnedRenderPhases<Visibility3d>>,
    settings: Res<PrismShadingSettings>,
    mut ctx: RenderContext,
) {
    if !settings.enable_visibility_buffer {
        return;
    }
    let view_entity = view.entity();
    let (camera, extracted, depth, targets, msaa) = view.into_inner();
    if msaa.is_some_and(|value| value.samples() != 1) {
        return;
    }
    let Some(phase) = phases.get(&extracted.retained_view_entity) else {
        return;
    };
    let (ids, metadata) = targets.attachments();
    let colors = [
        Some(RenderPassColorAttachment {
            view: ids,
            depth_slice: None,
            resolve_target: None,
            ops: Operations {
                load: LoadOp::Clear(VISIBILITY_IDS_CLEAR),
                store: StoreOp::Store,
            },
        }),
        Some(RenderPassColorAttachment {
            view: metadata,
            depth_slice: None,
            resolve_target: None,
            ops: Operations {
                load: LoadOp::Clear(VISIBILITY_METADATA_CLEAR),
                store: StoreOp::Store,
            },
        }),
    ];
    let mut pass = ctx.begin_tracked_render_pass(RenderPassDescriptor {
        label: Some("prism_visibility_raster"),
        color_attachments: &colors,
        depth_stencil_attachment: Some(depth.get_attachment(StoreOp::Store)),
        timestamp_writes: None,
        occlusion_query_set: None,
        multiview_mask: None,
    });
    if let Some(viewport) = camera.viewport.as_ref() {
        pass.set_camera_viewport(viewport);
    }
    let _ = phase.render(&mut pass, world, view_entity);
}

#[cfg(test)]
mod tests {
    use super::{VISIBILITY_IDS_CLEAR, VISIBILITY_METADATA_CLEAR};

    #[test]
    fn clear_values_encode_invalid_visibility_handles() {
        assert_eq!(
            [
                VISIBILITY_IDS_CLEAR.r as u32,
                VISIBILITY_IDS_CLEAR.g as u32,
                VISIBILITY_IDS_CLEAR.b as u32,
                VISIBILITY_IDS_CLEAR.a as u32,
            ],
            [u32::MAX, 0, u32::MAX, u32::MAX]
        );
        assert_eq!(
            [
                VISIBILITY_METADATA_CLEAR.r as u32,
                VISIBILITY_METADATA_CLEAR.g as u32,
                VISIBILITY_METADATA_CLEAR.b as u32,
                VISIBILITY_METADATA_CLEAR.a as u32,
            ],
            [u32::MAX, 0, 0, 0]
        );
    }
}
