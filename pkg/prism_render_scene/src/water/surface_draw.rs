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
        LoadOp, Operations, PipelineCache, RenderPassColorAttachment, RenderPassDescriptor,
        RenderPipeline, StoreOp,
    },
    renderer::{RenderContext, RenderDevice, ViewQuery},
    view::{ExtractedView, Msaa, ViewDepthStencilTexture, ViewTarget},
};

use bevy_math::{Vec3, Vec4};
use prism_render_shading::ReceiverProjection;

use crate::lighting::LightBindGroup;
use crate::shading::{
    GpuVsmResolveParams, PrismShadingSettings, PrismVirtualShadowSettings, ViewSsrTextures,
    ViewVisibilityBuffer, ViewVsmPageTable, ViewVsmPhysicalAtlas, VsmPrimaryLight,
};

use super::bind_groups::filtering_sampler;
use super::resources::WaterGpuBodies;
use super::surface_gtao::GpuWaterGtaoConfig;
use super::surface_mesh::{build_surface_view, surface_index_data};
use super::surface_motion::ViewWaterMotionUniform;
use super::surface_pipeline::{ViewWaterSurfacePipelines, WaterSurfacePipelines};
use super::surface_shading::SurfaceViewInputs;
use super::surface_ssr::{GpuWaterSsrConfig, WaterSsrFallback};
use super::surface_vsm::WaterVsmFallback;

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
    /// The `@group(5)` underwater froxel-volume group for this body: the sampled
    /// single-scatter + transmittance volume the surface fragment stage
    /// composites into its refraction.
    froxel_group: BindGroup,
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
    vsm_fallback: Res<WaterVsmFallback>,
    ssr_fallback: Res<WaterSsrFallback>,
    vsm_settings: Option<Res<PrismVirtualShadowSettings>>,
    primary_light: Option<Res<VsmPrimaryLight>>,
    view: ViewQuery<(
        &ExtractedView,
        &ViewTarget,
        &ViewDepthStencilTexture,
        &ViewWaterSurfacePipelines,
        &ViewVisibilityBuffer,
        &ViewWaterMotionUniform,
        Option<&ViewSsrTextures>,
        Option<&ViewVsmPhysicalAtlas>,
        Option<&ViewVsmPageTable>,
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
    let (
        extracted,
        target,
        depth,
        view_pipelines,
        visibility,
        motion_uniform,
        ssr_textures,
        vsm_atlas,
        vsm_page_table,
        msaa,
    ) = view.into_inner();
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
    let froxel_layout = cache.get_bind_group_layout(&surface_pipeline.froxel_layout);
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
                // @binding(7) previous-frame displacement for wave self-motion.
                body.buffers
                    .surface_mesh_displacement_prev
                    .as_entire_binding(),
            )),
        );
        // The `@group(5)` underwater froxel volume: the `water_underwater_volume`
        // kernel already filled this body's `vec4(inscatter, transmittance)`
        // volume this frame; the fragment stage samples it to composite the
        // participating medium into the refraction. Per-body (the volume lives
        // on the body), so it is built here rather than once per view.
        let froxel_group = device.create_bind_group(
            "prism water surface froxel",
            &froxel_layout,
            &BindGroupEntries::sequential((
                &body.buffers.underwater_out,
                &sampler,
                body.buffers.surface_froxel_params.as_entire_binding(),
            )),
        );
        prepared.push(PreparedSurfaceDraw {
            pipeline,
            uniform,
            index_buffer,
            bind_group,
            froxel_group,
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

    // The `@group(2)` virtual-shadow-map bind group, built once for the view
    // (its addressing basis is camera/light constant across every body). A
    // byte-for-byte twin of the opaque resolve pass's VSM group: it binds the
    // resident page table + physical atlas when both exist and the feature is
    // on, else the format-correct fallbacks with the `sample_enable` bit clear
    // so the shader keeps the analytic directional visibility. The params buffer
    // and the group are declared here so they outlive the single tracked pass.
    let vsm_layout = cache.get_bind_group_layout(&surface_pipeline.vsm_layout);
    let light_direction = primary_light.as_ref().and_then(|light| light.direction);
    let enable = settings.enable_virtual_shadow
        && vsm_settings.is_some()
        && vsm_atlas.is_some()
        && vsm_page_table.is_some()
        && light_direction.is_some();
    let clipmap = vsm_settings.as_ref().map_or_else(
        || PrismVirtualShadowSettings::default().clipmap(),
        |s| s.clipmap(),
    );
    let pcf_radius = vsm_settings.as_ref().map_or(0, |s| s.pcf_radius);
    let (physical_pages, physical_pages_per_edge) = vsm_atlas.map_or((1, 1), |atlas| {
        (atlas.physical_pages(), atlas.physical_pages_per_edge())
    });
    let direction = light_direction.unwrap_or(Vec3::NEG_Y);
    let projection = ReceiverProjection::from_light_direction(direction, Vec3::ZERO, 0.0);
    let light_forward = {
        let normalized = direction.normalize_or_zero();
        if normalized == Vec3::ZERO {
            Vec3::NEG_Y
        } else {
            normalized
        }
    };
    let vsm_params = GpuVsmResolveParams::new(
        &clipmap,
        physical_pages,
        physical_pages_per_edge,
        pcf_radius,
        enable,
        projection.light_right,
        projection.light_up,
        light_forward,
    );
    let vsm_params_buffer = device.create_buffer_with_data(&BufferInitDescriptor {
        label: Some("prism water surface vsm params"),
        contents: bytemuck::bytes_of(&vsm_params),
        usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
    });
    let page_table_binding = vsm_page_table.map_or_else(
        || vsm_fallback.page_table.as_entire_binding(),
        |table| table.buffer.as_entire_binding(),
    );
    let atlas_view = vsm_atlas.map_or(&vsm_fallback.atlas_view, |atlas| atlas.atlas_view());
    let vsm_group = device.create_bind_group(
        "prism water surface vsm",
        &vsm_layout,
        &BindGroupEntries::sequential((
            page_table_binding,
            atlas_view,
            &vsm_fallback.sampler,
            vsm_params_buffer.as_entire_binding(),
        )),
    );

    // The `@group(3)` screen-space-reflection group, built once for the view
    // (its march basis is camera-constant across every body). The water
    // fragment reconstructs its own view-space position/normal and marches the
    // opaque prepass's reverse-Z Hi-Z pyramid itself (see `surface_ssr`); it
    // never samples the resolved opaque SSR buffer, which would reflect the
    // submerged terrain rather than the surface. When the feature is off or no
    // pyramid is resident this frame it binds the format-correct fallback with
    // the `sample_enable` bit clear, so the shader keeps the image-based
    // reflection. The params buffer and group are declared here so they outlive
    // the single tracked pass.
    let ssr_layout = cache.get_bind_group_layout(&surface_pipeline.ssr_layout);
    // Recover the positive near-plane distance from the inverse projection: in
    // Prism's reverse-Z convention device depth 1.0 is the near plane, so
    // inverse-projecting clip (0, 0, 1, 1) yields a view-space point at `-near`
    // along the camera's `-Z` (byte-for-byte the opaque `ssr::trace` recovery).
    let view_from_clip = extracted.clip_from_view.inverse();
    let near_view = view_from_clip * Vec4::new(0.0, 0.0, 1.0, 1.0);
    let near = if near_view.w.abs() > f32::EPSILON {
        (near_view.z / near_view.w).abs().max(1.0e-3)
    } else {
        1.0e-3
    };
    let view_from_world = extracted.world_from_view.to_matrix().inverse();
    let ssr_enable = settings.enable_ssr && ssr_textures.is_some();
    let ssr_config = GpuWaterSsrConfig::new(
        extracted.clip_from_view,
        view_from_clip,
        view_from_world,
        near,
        40.0,
        ssr_enable,
    );
    let ssr_params_buffer = device.create_buffer_with_data(&BufferInitDescriptor {
        label: Some("prism water surface ssr params"),
        contents: bytemuck::bytes_of(&ssr_config),
        usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
    });
    let ssr_hzb = ssr_textures.map_or(&ssr_fallback.hzb_view, ViewSsrTextures::hzb_view);
    let ssr_group = device.create_bind_group(
        "prism water surface ssr",
        &ssr_layout,
        &BindGroupEntries::sequential((ssr_hzb, ssr_params_buffer.as_entire_binding())),
    );

    // The `@group(6)` ground-truth ambient-occlusion group, built once for the
    // view. The water fragment runs the horizon `GTAO` search over the same
    // reverse-Z Hi-Z pyramid bound at `@group(3)` (reusing its reconstruction
    // matrices), occluding the image-based ambient term with the surface's own
    // occlusion rather than the submerged terrain's resolved opaque `GTAO`.
    // The search needs that depth pyramid, so it is gated on both the `GTAO`
    // feature flag and a resident `ViewSsrTextures`; otherwise the config's
    // `sample_enable` bit is clear and the shader leaves the ambient term fully
    // lit. The buffer and group are declared here so they outlive the pass.
    let gtao_layout = cache.get_bind_group_layout(&surface_pipeline.gtao_layout);
    let gtao_enable = settings.enable_gtao && ssr_textures.is_some();
    let gtao_config = GpuWaterGtaoConfig::new(gtao_enable);
    let gtao_params_buffer = device.create_buffer_with_data(&BufferInitDescriptor {
        label: Some("prism water surface gtao params"),
        contents: bytemuck::bytes_of(&gtao_config),
        usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
    });
    let gtao_group = device.create_bind_group(
        "prism water surface gtao",
        &gtao_layout,
        &BindGroupEntries::single(gtao_params_buffer.as_entire_binding()),
    );

    // The `@group(4)` motion-vector group: the per-view current+previous
    // view-projection uniform `prepare_water_surface_motion` built this frame.
    // The fragment stage reprojects the surface's world position through both
    // matrices and writes `cur_uv - prev_uv` into the second render target.
    let motion_layout = cache.get_bind_group_layout(&surface_pipeline.motion_layout);
    let motion_group = device.create_bind_group(
        "prism water surface motion",
        &motion_layout,
        &BindGroupEntries::single(motion_uniform.buffer.as_entire_binding()),
    );

    // Single tracked pass: the composite already wrote the view target, so the
    // color attachment loads, and the main-pass depth loads read-only (the
    // surface pipeline disables depth writes).
    let color = target.get_color_attachment();
    // Second render target (MRT): the shared `Rg16Float` motion G-buffer. It
    // loads the opaque resolve pass's camera+object motion and the water
    // fragment replaces it (write mask RED|GREEN) for exactly the pixels the
    // surface covers, so the depth-tested water owns the reprojection basis for
    // its own pixels without disturbing the submerged geometry behind it.
    let motion_attachment = RenderPassColorAttachment {
        view: visibility.motion_vectors_view(),
        depth_slice: None,
        resolve_target: None,
        ops: Operations {
            load: LoadOp::Load,
            store: StoreOp::Store,
        },
    };
    let mut pass = ctx.begin_tracked_render_pass(RenderPassDescriptor {
        label: Some("prism water surface"),
        color_attachments: &[Some(color), Some(motion_attachment)],
        depth_stencil_attachment: Some(depth.get_attachment(StoreOp::Store)),
        timestamp_writes: None,
        occlusion_query_set: None,
        multiview_mask: None,
    });
    for draw in &prepared {
        pass.set_render_pipeline(draw.pipeline);
        pass.set_bind_group(0, &draw.bind_group, &[]);
        pass.set_bind_group(1, light_group, &[]);
        pass.set_bind_group(2, &vsm_group, &[]);
        pass.set_bind_group(3, &ssr_group, &[]);
        pass.set_bind_group(4, &motion_group, &[]);
        pass.set_bind_group(5, &draw.froxel_group, &[]);
        pass.set_bind_group(6, &gtao_group, &[]);
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
