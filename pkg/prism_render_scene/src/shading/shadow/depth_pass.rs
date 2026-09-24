//! Shadow-map depth rasterization: the render-graph node and its two feeder
//! systems that fill the shadow atlas.
//!
//! This is the runtime half of the depth pass whose device objects live in
//! [`super::pipeline`].  Three pieces cooperate each frame:
//!
//! * [`queue_shadow_depth`] (in [`RenderSystems::QueueMeshes`]) specializes the
//!   [`ShadowDepthPipeline`] for every GPU-scene instance's mesh and records a
//!   flat [`ShadowDepthDrawList`] of `(pipeline, entity, instance-index)` draws.
//!   The atlas is shared by every shadow view, so the same geometry list is
//!   replayed once per admitted layer.
//! * [`prepare_shadow_depth_uniform`] (in [`RenderSystems::PrepareBindGroups`],
//!   after the shadow buffers are written) packs one
//!   [`GpuShadowDepthView`](super::pipeline::GpuShadowDepthView) per planned
//!   [`ShadowDepthDraw`](prism_render_shading::ShadowDepthDraw) into the
//!   dynamic-offset uniform buffer, records the per-draw offsets in
//!   [`ShadowDepthViewOffsets`], and rebuilds the `@group(0)` bind group.
//! * [`shadow_depth_pass`] (a `Core3d` node before the visibility raster) walks
//!   the planned draws in slot order, opening one render pass per atlas layer
//!   (its own colour target plus the shared transient depth buffer) and
//!   replaying the geometry list with that layer's per-view dynamic offset.
//!
//! # Known limitations (require on-device validation)
//! * The node has no `ViewQuery`, so under multiple cameras `Core3d` runs it
//!   once per view and each run refills the whole atlas identically
//!   (last-write-wins).  This is correct but redundant; a future slice should
//!   gate it to run once per frame.
//! * The stored depth / distance values, the white (`far`) clear, and the
//!   embedded-asset resolution of `shadow_depth.wesl` at runtime all need GPU
//!   validation against the CPU golden reference.

use bevy_ecs::prelude::*;
use bevy_pbr::{MeshPipelineKey, RenderMeshInstances};
use bevy_render::{
    mesh::{allocator::MeshAllocator, RenderMesh, RenderMeshBufferInfo},
    render_asset::RenderAssets,
    render_resource::*,
    renderer::{RenderContext, RenderDevice, RenderQueue},
    sync_world::MainEntity,
};

use crate::{buffers::GpuSceneBindGroup, ExtractedSceneInstance, GpuSceneInstanceAddress};

use super::pipeline::{
    GpuShadowDepthView, ShadowDepthPipeline, ShadowDepthPipelineKey, ShadowDepthViewUniform,
};
use super::resources::{ExtractedShadows, ShadowAtlas};

/// One replayable geometry draw for the shadow depth pass.
///
/// The same list is drawn into every admitted atlas layer; only the bound
/// per-view uniform (selected by dynamic offset) changes between layers.
#[derive(Clone, Copy)]
struct ShadowDepthDrawEntry {
    /// The mesh-specialized depth pipeline for this instance.
    pipeline_id: CachedRenderPipelineId,
    /// Main-world entity used to resolve the mesh asset id at draw time.
    main_entity: MainEntity,
    /// Row of this instance in the GPU-scene transform / instance tables.
    instance_index: u32,
}

/// The per-frame list of shadow-caster geometry draws, rebuilt in
/// [`queue_shadow_depth`] and replayed per atlas layer by [`shadow_depth_pass`].
#[derive(Resource, Default)]
pub(crate) struct ShadowDepthDrawList(Vec<ShadowDepthDrawEntry>);

/// Dynamic-offset table into [`ShadowDepthViewUniform`], one entry per planned
/// depth draw in the same order as [`ExtractedShadows::depth_draws`].
#[derive(Resource, Default)]
pub(crate) struct ShadowDepthViewOffsets(Vec<u32>);

/// Specializes the shadow depth pipeline for every GPU-scene instance and
/// records the flat draw list replayed into each atlas layer.
///
/// Gated on [`PrismShadingSettings::enable_visibility_buffer`]; the list is
/// cleared every frame so a disabled pass leaves no stale draws behind.
pub(crate) fn queue_shadow_depth(
    settings: Res<super::super::runtime::PrismShadingSettings>,
    pipeline_cache: Res<PipelineCache>,
    pipeline: Res<ShadowDepthPipeline>,
    mut pipelines: ResMut<SpecializedMeshPipelines<ShadowDepthPipeline>>,
    meshes: Res<RenderAssets<RenderMesh>>,
    render_instances: Res<RenderMeshInstances>,
    scene_instances: Query<(&GpuSceneInstanceAddress, &ExtractedSceneInstance)>,
    mut draw_list: ResMut<ShadowDepthDrawList>,
) {
    draw_list.0.clear();
    if !settings.enable_visibility_buffer {
        return;
    }
    for (address, instance) in &scene_instances {
        let Some(mesh_id) = render_instances.mesh_asset_id(instance.main_entity) else {
            continue;
        };
        let Some(mesh) = meshes.get(mesh_id) else {
            continue;
        };
        let key = MeshPipelineKey::from_primitive_topology_and_strip_index(
            mesh.primitive_topology(),
            mesh.index_format(),
        );
        let Ok(pipeline_id) = pipelines.specialize(
            &pipeline_cache,
            &pipeline,
            ShadowDepthPipelineKey { mesh: key },
            &mesh.layout,
        ) else {
            continue;
        };
        draw_list.0.push(ShadowDepthDrawEntry {
            pipeline_id,
            main_entity: instance.main_entity,
            instance_index: address.index,
        });
    }
}

/// Uploads one per-view uniform slice per planned depth draw and rebuilds the
/// `@group(0)` bind group.
///
/// Runs after `write_shadow_buffers` so the freshly extracted
/// [`ExtractedShadows::depth_draws`] plan is visible.  The buffer is kept
/// non-empty even with no draws so [`DynamicUniformBuffer::binding`] always
/// yields a valid resource and the bind group can be built unconditionally.
pub(crate) fn prepare_shadow_depth_uniform(
    extracted: Res<ExtractedShadows>,
    mut uniform: ResMut<ShadowDepthViewUniform>,
    mut offsets: ResMut<ShadowDepthViewOffsets>,
    device: Res<RenderDevice>,
    queue: Res<RenderQueue>,
) {
    uniform.buffer.clear();
    offsets.0.clear();
    for draw in &extracted.depth_draws {
        let offset = uniform.buffer.push(&GpuShadowDepthView::from_view(&draw.view));
        offsets.0.push(offset);
    }
    if extracted.depth_draws.is_empty() {
        // Keep the buffer non-empty so `binding()` yields a resource even on a
        // shadowless frame; the placeholder slice is never referenced.
        uniform.buffer.push(&GpuShadowDepthView::default());
    }
    uniform.buffer.write_buffer(&device, &queue);
    let Some(binding) = uniform.buffer.binding() else {
        uniform.bind_group = None;
        return;
    };
    let bind_group = device.create_bind_group(
        "prism shadow depth view",
        &uniform.layout,
        &BindGroupEntries::single(binding),
    );
    uniform.bind_group = Some(bind_group);
}

/// `Core3d` node: rasterizes shadow-caster geometry into every admitted atlas
/// layer.
///
/// One render pass is opened per planned [`ShadowDepthDraw`] layer: its colour
/// target is that layer's single-layer view (cleared to the far value `1.0`)
/// and its depth-stencil target is the shared transient depth buffer (cleared
/// to `1.0` and reused per layer).  The whole [`ShadowDepthDrawList`] is
/// replayed inside each pass with the layer's per-view dynamic offset bound.
pub(crate) fn shadow_depth_pass(world: &World, mut ctx: RenderContext) {
    let settings = world.resource::<super::super::runtime::PrismShadingSettings>();
    if !settings.enable_visibility_buffer {
        return;
    }
    let extracted = world.resource::<ExtractedShadows>();
    if extracted.depth_draws.is_empty() {
        return;
    }
    let uniform = world.resource::<ShadowDepthViewUniform>();
    let Some(view_bind_group) = uniform.bind_group.as_ref() else {
        return;
    };
    let scene = world.resource::<GpuSceneBindGroup>();
    let Some(scene_bind_group) = scene.bind_group.as_ref() else {
        return;
    };
    let draw_list = world.resource::<ShadowDepthDrawList>();
    if draw_list.0.is_empty() {
        return;
    }
    let atlas = world.resource::<ShadowAtlas>();
    let offsets = world.resource::<ShadowDepthViewOffsets>();
    let pipeline_cache = world.resource::<PipelineCache>();
    let meshes = world.resource::<RenderAssets<RenderMesh>>();
    let render_instances = world.resource::<RenderMeshInstances>();
    let allocator = world.resource::<MeshAllocator>();

    for (draw_index, draw) in extracted.depth_draws.iter().enumerate() {
        let Some(layer_view) = atlas.layer_view(draw.layer) else {
            continue;
        };
        let Some(&offset) = offsets.0.get(draw_index) else {
            continue;
        };
        let color_attachments = [Some(RenderPassColorAttachment {
            view: layer_view,
            depth_slice: None,
            resolve_target: None,
            ops: Operations {
                // The atlas stores depth / normalized distance in `.r`; the far
                // value `1.0` means "no occluder", so uncovered texels clear to
                // white and cast no shadow.
                load: LoadOp::Clear(wgpu_types::Color {
                    r: 1.0,
                    g: 1.0,
                    b: 1.0,
                    a: 1.0,
                }),
                store: StoreOp::Store,
            },
        })];
        let mut pass = ctx.begin_tracked_render_pass(RenderPassDescriptor {
            label: Some("prism_shadow_depth"),
            color_attachments: &color_attachments,
            depth_stencil_attachment: Some(RenderPassDepthStencilAttachment {
                view: atlas.depth_view(),
                depth_ops: Some(Operations {
                    load: LoadOp::Clear(1.0),
                    store: StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        for entry in &draw_list.0 {
            let Some(render_pipeline) = pipeline_cache.get_render_pipeline(entry.pipeline_id)
            else {
                continue;
            };
            let Some(mesh_id) = render_instances.mesh_asset_id(entry.main_entity) else {
                continue;
            };
            let Some(mesh) = meshes.get(mesh_id) else {
                continue;
            };
            let Some(vertices) = allocator.mesh_vertex_slice(&mesh_id) else {
                continue;
            };
            pass.set_render_pipeline(render_pipeline);
            pass.set_bind_group(0, view_bind_group, &[offset]);
            pass.set_bind_group(1, scene_bind_group, &[]);
            pass.set_vertex_buffer(0, vertices.buffer.slice(..));
            let instance = entry.instance_index..entry.instance_index.saturating_add(1);
            match mesh.buffer_info {
                RenderMeshBufferInfo::Indexed {
                    index_format,
                    count,
                } => {
                    let Some(indices) = allocator.mesh_index_slice(&mesh_id) else {
                        continue;
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
        }
    }
}
