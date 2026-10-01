//! `Core3d` graph node that rasterizes the displaced water surface on screen.
//!
//! Terminal step of the water render chain. Upstream, [`super::dispatch`] ran
//! the compute sweep that filled each body's four `@group(0)` storage arrays
//! (base positions / surface UVs / displacement / normal-foam), and
//! [`super::surface_pipeline`] specialized a raster pipeline per view and
//! lighting frontend. This node only records the indexed draws, loading the
//! view target the shading composite already wrote so the water blends over the
//! resolved opaque radiance (premultiplied alpha) and tests against — but never
//! writes — the main-pass depth.
//!
//! It runs *after* [`crate::shading::composite_shading`] (so opaque pixels are
//! present to refract through `scene_color`) and *before* Bevy's post-process,
//! and inherits the same single-sample / `enable_visibility_buffer` gate as the
//! rest of the Prism visibility path: a view without both a
//! [`ViewWaterSurfacePipelines`] and a [`ViewVisibilityBuffer`] is skipped by
//! the query, and a multisampled view bails explicitly.

use bevy_ecs::prelude::*;
use bevy_render::{
    render_resource::{
        BindGroup, BindGroupEntries, Buffer, BufferInitDescriptor, BufferUsages, IndexFormat,
        PipelineCache, RenderPassDescriptor, RenderPipeline, StoreOp,
    },
    renderer::{RenderContext, RenderDevice, ViewQuery},
    view::{ExtractedView, Msaa, ViewDepthStencilTexture, ViewTarget},
};

use crate::lighting::LightBindGroup;
use crate::shading::{PrismShadingSettings, ViewVisibilityBuffer};

use super::bind_groups::filtering_sampler;
use super::resources::WaterGpuBodies;
use super::surface_mesh::{build_surface_view, surface_index_data};
use super::surface_pipeline::{ViewWaterSurfacePipelines, WaterSurfacePipelines};
use super::surface_shading::SurfaceViewInputs;

/// One body's device resources, built before the pass so the index/uniform
/// buffers and bind group outlive the single tracked pass that references them.
struct PreparedSurfaceDraw<'a> {
    pipeline: &'a RenderPipeline,
    #[expect(
        dead_code,
        reason = "RAII handle keeping the uniform buffer alive for the bind \
                  group that references it; the pass binds through `bind_group` \
                  and never reads this field directly."
    )]
    uniform: Buffer,
    index_buffer: Buffer,
    bind_group: BindGroup,
    index_count: u32,
}

/// Record the indexed water-surface draws for every covered view.
pub(crate) fn draw_water_surface(
    settings: Res<PrismShadingSettings>,
    cache: Res<PipelineCache>,
    bodies: Res<WaterGpuBodies>,
    surface_pipeline: Res<WaterSurfacePipelines>,
    light_bindings: Res<LightBindGroup>,
    device: Res<RenderDevice>,
    view: ViewQuery<(
        &ExtractedView,
        &ViewTarget,
        &ViewDepthStencilTexture,
        &ViewWaterSurfacePipelines,
        &ViewVisibilityBuffer,
        Option<&Msaa>,
    )>,
    mut ctx: RenderContext,
) {
    if !settings.enable_visibility_buffer {
        return;
    }
    if bodies.is_empty() {
        return;
    }
    let (extracted, target, depth, view_pipelines, visibility, msaa) = view.into_inner();
    // The Prism visibility path is single-sample only; a multisampled view never
    // wrote `scene_color`, so refraction has nothing to sample.
    if msaa.is_some_and(|value| value.samples() != 1) {
        return;
    }

    // The camera transform is constant across every body this view; scene
    // lighting is bound separately through the shared `group(1)` light table.
    let clip_from_world = extracted.clip_from_world.unwrap_or_else(|| {
        extracted.clip_from_view * extracted.world_from_view.to_matrix().inverse()
    });
    let view_inputs = SurfaceViewInputs {
        clip_from_world: clip_from_world.to_cols_array_2d(),
        camera_world_position: extracted.world_from_view.translation().to_array(),
        viewport_size: [
            viewport_dimension(extracted.viewport.z),
            viewport_dimension(extracted.viewport.w),
        ],
    };

    let layout = cache.get_bind_group_layout(&surface_pipeline.layout);
    let sampler = filtering_sampler(&device, "prism water surface refraction");

    // Build every body's device resources before opening the pass: the index
    // and uniform buffers plus the bind group must outlive the single tracked
    // pass that records all the draws.
    let mut prepared: Vec<PreparedSurfaceDraw> = Vec::new();
    for body in &bodies.bodies {
        let Some(draw) = body.surface_draw.as_ref() else {
            continue;
        };
        // The compute sweep fills exactly `vertex_count` vertices; a grid that
        // disagrees would index past the storage arrays, so skip it.
        if draw.grid.vertex_count() != draw.vertex_count {
            continue;
        }
        let Some(pipeline) =
            cache.get_render_pipeline(view_pipelines.id_for(draw.shading.frontend))
        else {
            continue;
        };

        let gpu_view = build_surface_view(&draw.shading.view_params(&view_inputs));
        let uniform = device.create_buffer_with_data(&BufferInitDescriptor {
            label: Some("prism water surface view"),
            contents: bytemuck::bytes_of(&gpu_view),
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        });
        let indices = surface_index_data(draw.grid);
        let index_buffer = device.create_buffer_with_data(&BufferInitDescriptor {
            label: Some("prism water surface indices"),
            contents: bytemuck::cast_slice(&indices),
            usage: BufferUsages::INDEX | BufferUsages::COPY_DST,
        });
        let bind_group = device.create_bind_group(
            "prism water surface",
            &layout,
            &BindGroupEntries::sequential((
                uniform.as_entire_binding(),
                body.buffers.surface_mesh_base_positions.as_entire_binding(),
                body.buffers.surface_mesh_surface_uvs.as_entire_binding(),
                body.buffers.surface_mesh_displacement.as_entire_binding(),
                body.buffers.surface_mesh_normal_foam.as_entire_binding(),
                visibility.scene_color_view(),
                &sampler,
            )),
        );
        prepared.push(PreparedSurfaceDraw {
            pipeline,
            uniform,
            index_buffer,
            bind_group,
            index_count: draw.grid.index_count(),
        });
    }
    if prepared.is_empty() {
        return;
    }
    // The shared light table must have uploaded this frame; without it the
    // fragment stage has no `group(1)` to read, so skip exactly like the
    // opaque resolve pass does on its first frame.
    let Some(light_group) = light_bindings.bind_group.as_ref() else {
        return;
    };

    // Single tracked pass: the composite already wrote the view target, so the
    // color attachment loads, and the main-pass depth loads read-only (the
    // surface pipeline disables depth writes).
    let color = target.get_color_attachment();
    let mut pass = ctx.begin_tracked_render_pass(RenderPassDescriptor {
        label: Some("prism water surface"),
        color_attachments: &[Some(color)],
        depth_stencil_attachment: Some(depth.get_attachment(StoreOp::Store)),
        timestamp_writes: None,
        occlusion_query_set: None,
        multiview_mask: None,
    });
    for draw in &prepared {
        pass.set_render_pipeline(draw.pipeline);
        pass.set_bind_group(0, &draw.bind_group, &[]);
        pass.set_bind_group(1, light_group, &[]);
        pass.set_index_buffer(draw.index_buffer.slice(..), IndexFormat::Uint32);
        pass.draw_indexed(0..draw.index_count, 0, 0..1);
    }
}

/// Framebuffer extent as the shader's `f32`; viewport dimensions are small
/// screen pixel counts, so the `u32 -> f32` widening never loses precision.
#[expect(
    clippy::cast_precision_loss,
    reason = "Viewport width/height are screen pixel counts well below 2^24, so \
              the widening to the shader's f32 is exact."
)]
fn viewport_dimension(pixels: u32) -> f32 {
    pixels as f32
}
