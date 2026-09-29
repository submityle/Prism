//! Runtime half of the virtual-shadow-map caster depth pass: the three feeder
//! systems and the `Core3d` render-graph node that fill the physical atlas with
//! shadow-caster depth, one resident clipmap page per tile.
//!
//! This is the VSM twin of [`super::super::super::shadow::depth_pass`]. Four pieces
//! cooperate each frame, split so each is independently schedulable and the node
//! stays a thin replay:
//!
//! * [`queue_vsm_caster_depth`] ([`RenderSystems::QueueMeshes`]) specializes the
//!   [`VsmCasterDepthPipeline`](super::pipeline::VsmCasterDepthPipeline) for every
//!   GPU-scene instance's mesh and records a flat
//!   [`VsmCasterDepthDrawList`](super::raster::VsmCasterDepthDrawList) of
//!   `(pipeline, entity, instance-index)` draws. The same geometry list is
//!   replayed into every resident page (only the bound per-page projection
//!   changes), exactly as the classic shadow pass replays its list per atlas
//!   layer.
//! * [`prepare_vsm_caster_depth_targets`] ([`RenderSystems::PrepareResources`])
//!   ensures the atlas-sized transient depth buffer
//!   ([`VsmCasterDepthTargets`](super::raster::VsmCasterDepthTargets)) exists and
//!   matches the current atlas edge.
//! * [`prepare_vsm_caster_depth_views`] ([`RenderSystems::PrepareBindGroups`])
//!   packs one per-page orthographic light projection
//!   ([`GpuVsmCasterDepthView`](super::abi::GpuVsmCasterDepthView)) into the
//!   dynamic-offset uniform buffer, records the per-page
//!   [`VsmCasterDepthEntry`](super::raster::VsmCasterDepthEntry) list on each view as
//!   [`ViewVsmCasterDepth`](super::raster::ViewVsmCasterDepth), and rebuilds the
//!   `@group(0)` bind group.
//! * [`vsm_caster_depth_pass`] (a `Core3d` node) opens one render pass per view
//!   whose colour target is that view's shared physical atlas and whose depth
//!   target is the transient depth buffer, then for each resident page sets the
//!   page's tile as the viewport / scissor and replays the geometry list with
//!   the page's dynamic offset. Rendering straight into the atlas with a
//!   per-page viewport writes each page's depth into exactly the texels
//!   [`shaders/vsm_sample.wesl`] reads it back from.
//!
//! # Frame ordering
//! Placed after the page-request / allocation pass (`super::super::page_mark`) so
//! the resident page list is known, and before the resolve pass that samples the
//! atlas. Because resident pages occupy disjoint tiles, the single atlas-wide
//! depth clear at the start of the pass gives every rendered tile a clean
//! far-depth start without the pages interfering with one another.
//!
//! # Known limitations (require on-device validation)
//! * The stored NDC depth, the far (`1.0`) colour clear, and the embedded-asset
//!   resolution of `vsm_caster_depth.wesl` at runtime all need GPU validation
//!   against the CPU golden reference.
//! * The depth slab is sized to the coarsest clip window
//!   ([`caster_depth_half_extent`](super::projection::caster_depth_half_extent));
//!   a tighter per-scene fit is deferred to a later slice.

use bevy_ecs::prelude::*;
use bevy_pbr::{MeshPipelineKey, RenderMeshInstances};
use bevy_render::{
    mesh::{allocator::MeshAllocator, RenderMesh, RenderMeshBufferInfo},
    render_asset::RenderAssets,
    render_resource::{
        BindGroupEntries, CachedRenderPipelineId, LoadOp, Operations, PipelineCache,
        RenderPassColorAttachment, RenderPassDepthStencilAttachment, RenderPassDescriptor,
        SpecializedMeshPipelines, StoreOp,
    },
    renderer::{RenderContext, RenderDevice, RenderQueue, ViewQuery},
};

use crate::{buffers::GpuSceneBindGroup, ExtractedSceneInstance, GpuSceneInstanceAddress};

use super::abi::GpuVsmCasterDepthView;
use super::pipeline::{VsmCasterDepthPipeline, VsmCasterDepthPipelineKey, VsmCasterDepthViewUniform};
use super::projection::{caster_depth_half_extent, page_light_projection, page_viewport_rect};
use super::raster::{
    VsmCasterDepthDrawEntry, VsmCasterDepthDrawList, VsmCasterDepthEntry, VsmCasterDepthTargets,
    ViewVsmCasterDepth, ViewVsmCasterPages,
};
use super::resources::{physical_pages_per_edge, ViewVsmPhysicalAtlas};

use super::super::extract::VsmPrimaryLight;
use super::super::settings::PrismVirtualShadowSettings;
use super::super::super::runtime::PrismShadingSettings;

/// Specializes the caster depth pipeline for every GPU-scene instance and
/// records the flat draw list replayed into each resident page.
///
/// Mirrors [`queue_shadow_depth`](super::super::super::shadow::depth_pass) exactly:
/// gated on [`PrismShadingSettings::enable_virtual_shadow`], the list is cleared
/// every frame so a disabled pass leaves no stale draws behind.
pub(crate) fn queue_vsm_caster_depth(
    settings: Res<PrismShadingSettings>,
    pipeline_cache: Res<PipelineCache>,
    pipeline: Res<VsmCasterDepthPipeline>,
    mut pipelines: ResMut<SpecializedMeshPipelines<VsmCasterDepthPipeline>>,
    meshes: Res<RenderAssets<RenderMesh>>,
    render_instances: Res<RenderMeshInstances>,
    scene_instances: Query<(&GpuSceneInstanceAddress, &ExtractedSceneInstance)>,
    mut draw_list: ResMut<VsmCasterDepthDrawList>,
) {
    draw_list.clear();
    if !settings.enable_virtual_shadow {
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
        let Ok(pipeline_id): Result<CachedRenderPipelineId, _> = pipelines.specialize(
            &pipeline_cache,
            &pipeline,
            VsmCasterDepthPipelineKey { mesh: key },
            &mesh.layout,
        ) else {
            continue;
        };
        draw_list.push(VsmCasterDepthDrawEntry {
            pipeline_id,
            main_entity: instance.main_entity,
            instance_index: address.index,
        });
    }
}

/// Ensures the atlas-sized transient depth buffer exists and matches the current
/// physical-atlas edge.
///
/// The atlas edge is recomputed exactly as
/// [`prepare_vsm_physical_atlas`](super::resources::prepare_vsm_physical_atlas)
/// does (`physical_pages_per_edge * page_size`) so the transient depth attachment
/// always covers the whole atlas framebuffer the node renders per-page viewports
/// into. Gated on the enable so a disabled frame does not allocate.
pub(crate) fn prepare_vsm_caster_depth_targets(
    settings: Res<PrismShadingSettings>,
    vsm_settings: Res<PrismVirtualShadowSettings>,
    device: Res<RenderDevice>,
    mut targets: ResMut<VsmCasterDepthTargets>,
) {
    if !settings.enable_virtual_shadow {
        return;
    }
    let page_size = u32::from(vsm_settings.page_size).max(1);
    let atlas_edge = physical_pages_per_edge(vsm_settings.physical_pages)
        .saturating_mul(page_size)
        .max(1);
    targets.ensure(&device, atlas_edge);
}

/// Uploads one per-page orthographic light projection slice for every resident
/// page across every view, records each view's per-page
/// [`VsmCasterDepthEntry`] list, and rebuilds the `@group(0)` bind group.
///
/// Mirrors [`prepare_shadow_depth_uniform`](super::super::super::shadow::depth_pass):
/// the buffer is kept non-empty even with no resident page so
/// [`DynamicUniformBuffer::binding`](bevy_render::render_resource::DynamicUniformBuffer::binding)
/// always yields a resource and the bind group can be built unconditionally. All
/// pages share the light direction and depth slab, so the projection differs
/// only in the page footprint each maps onto the NDC cube.
pub(crate) fn prepare_vsm_caster_depth_views(
    settings: Res<PrismShadingSettings>,
    vsm_settings: Res<PrismVirtualShadowSettings>,
    primary_light: Res<VsmPrimaryLight>,
    mut uniform: ResMut<VsmCasterDepthViewUniform>,
    views: Query<(Entity, &ViewVsmCasterPages)>,
    device: Res<RenderDevice>,
    queue: Res<RenderQueue>,
    mut commands: Commands,
) {
    uniform.buffer.clear();

    let enabled = settings.enable_virtual_shadow;
    let light_direction = primary_light.direction;
    let depth_half_extent = caster_depth_half_extent(&vsm_settings.clipmap());

    // Only pack real projections when the pass will actually run and a light
    // exists; otherwise every view gets an empty entry list and the pass skips.
    if let (true, Some(direction)) = (enabled, light_direction) {
        for (entity, pages) in &views {
            let mut entries = Vec::with_capacity(pages.pages.len());
            for page in &pages.pages {
                let view_projection = page_light_projection(
                    page.world_origin,
                    page.world_size,
                    direction,
                    depth_half_extent,
                );
                let dynamic_offset = uniform
                    .buffer
                    .push(&GpuVsmCasterDepthView { view_projection });
                entries.push(VsmCasterDepthEntry {
                    dynamic_offset,
                    atlas_tile_origin: page.atlas_tile_origin,
                });
            }
            commands.entity(entity).insert(ViewVsmCasterDepth { entries });
        }
    }

    if uniform.buffer.is_empty() {
        // Keep the buffer non-empty so `binding()` yields a resource even on a
        // page-less / disabled frame; the placeholder slice is never referenced.
        uniform.buffer.push(&GpuVsmCasterDepthView::default());
    }
    uniform.buffer.write_buffer(&device, &queue);

    let Some(binding) = uniform.buffer.binding() else {
        uniform.bind_group = None;
        return;
    };
    let bind_group = device.create_bind_group(
        "prism vsm caster depth view",
        &uniform.layout,
        &BindGroupEntries::single(binding),
    );
    uniform.bind_group = Some(bind_group);
}

/// `Core3d` node: rasterizes shadow-caster geometry into every resident page of
/// the current view's physical atlas.
///
/// One render pass is opened for the view: its colour target is the shared
/// physical atlas (cleared to the far value `1.0`) and its depth-stencil target is
/// the atlas-sized transient depth buffer (cleared to `1.0`). For each resident
/// page the page's atlas tile is set as the viewport / scissor and the whole
/// [`VsmCasterDepthDrawList`] is replayed with the page's per-page dynamic offset
/// bound, so each page's casters are projected through that page's orthographic
/// light projection and written into exactly its tile.
///
/// Uses a [`ViewQuery`]: `Core3d` runs the node per view, and a view lacking the
/// atlas / entry components simply skips (the query fails validation) rather than
/// panicking.
pub(crate) fn vsm_caster_depth_pass(
    settings: Res<PrismShadingSettings>,
    view: ViewQuery<(&ViewVsmPhysicalAtlas, &ViewVsmCasterDepth)>,
    vsm_settings: Res<PrismVirtualShadowSettings>,
    uniform: Res<VsmCasterDepthViewUniform>,
    scene: Res<GpuSceneBindGroup>,
    draw_list: Res<VsmCasterDepthDrawList>,
    targets: Res<VsmCasterDepthTargets>,
    pipeline_cache: Res<PipelineCache>,
    meshes: Res<RenderAssets<RenderMesh>>,
    render_instances: Res<RenderMeshInstances>,
    allocator: Res<MeshAllocator>,
    mut ctx: RenderContext,
) {
    if !settings.enable_virtual_shadow {
        return;
    }
    let (atlas, caster_depth) = view.into_inner();
    if caster_depth.entries.is_empty() {
        return;
    }
    let Some(view_bind_group) = uniform.bind_group.as_ref() else {
        return;
    };
    let Some(scene_bind_group) = scene.bind_group.as_ref() else {
        return;
    };
    if draw_list.draws().is_empty() {
        return;
    }
    let Some(depth_view) = targets.depth_view() else {
        return;
    };

    let page_size = u32::from(vsm_settings.page_size).max(1);

    let mut pass = ctx.begin_tracked_render_pass(RenderPassDescriptor {
        label: Some("prism_vsm_caster_depth"),
        color_attachments: &[Some(RenderPassColorAttachment {
            view: atlas.atlas_view(),
            depth_slice: None,
            resolve_target: None,
            ops: Operations {
                // The atlas stores NDC depth in `.r`; the far value `1.0` means
                // "no occluder", so any texel a page does not cover clears to the
                // far plane and casts no shadow. Resident pages occupy disjoint
                // tiles, so this single clear seeds every rendered tile cleanly.
                load: LoadOp::Clear(wgpu_types::Color {
                    r: 1.0,
                    g: 1.0,
                    b: 1.0,
                    a: 1.0,
                }),
                store: StoreOp::Store,
            },
        })],
        depth_stencil_attachment: Some(RenderPassDepthStencilAttachment {
            view: depth_view,
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

    for entry in &caster_depth.entries {
        // Restrict rendering to this page's atlas tile: the viewport maps the
        // page's NDC cube onto the tile and the scissor discards anything the
        // projection pushes outside it, so a page can only ever write its own
        // tile of the shared atlas.
        let (tile_x, tile_y, tile_w, tile_h) =
            page_viewport_rect(entry.atlas_tile_origin, page_size);
        pass.set_viewport(
            tile_x as f32,
            tile_y as f32,
            tile_w as f32,
            tile_h as f32,
            0.0,
            1.0,
        );
        pass.set_scissor_rect(tile_x, tile_y, tile_w, tile_h);

        for draw in draw_list.draws() {
            let Some(render_pipeline) = pipeline_cache.get_render_pipeline(draw.pipeline_id) else {
                continue;
            };
            let Some(mesh_id) = render_instances.mesh_asset_id(draw.main_entity) else {
                continue;
            };
            let Some(mesh) = meshes.get(mesh_id) else {
                continue;
            };
            let Some(vertices) = allocator.mesh_vertex_slice(&mesh_id) else {
                continue;
            };
            pass.set_render_pipeline(render_pipeline);
            pass.set_bind_group(0, view_bind_group, &[entry.dynamic_offset]);
            pass.set_bind_group(1, scene_bind_group, &[]);
            pass.set_vertex_buffer(0, vertices.buffer.slice(..));
            let instance = draw.instance_index..draw.instance_index.saturating_add(1);
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
