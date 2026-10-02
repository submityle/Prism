//! SSR screen-space trace: pipeline, per-view bind group, and the `Core3d`
//! dispatch node.
//!
//! This is the reflection march itself, the GPU twin of
//! [`prism_render_shading::screen_space`]. For every covered pixel it reflects
//! the view ray off the reconstructed surface, marches the reflected ray across
//! the reverse-Z Hi-Z "nearest depth" pyramid, and samples the current-frame
//! colour pyramid at the hit (picking a roughness-selected mip so rougher
//! surfaces read a wider pre-blur), writing reflected radiance plus a `[0, 1]`
//! blend confidence into the reflection output the composite folds back over the
//! shaded scene colour.
//!
//! It reads one bind group (group 0, matching `shaders/ssr.wesl`):
//!
//! * `0` the multi-mip Hi-Z pyramid (`textureLoad`ed per level in the march),
//! * `1` the full-resolution device depth (view-position reconstruction),
//! * `2` the packed view-normal + roughness the repack produced,
//! * `3` the current-frame colour pyramid (sampled with a filtering sampler so
//!   the fractional roughness mip interpolates),
//! * `4` that filtering sampler, and
//! * `5` the write-only `rgba16float` reflection output.
//!
//! The view/camera matrices and the golden fade/march tunables travel in the
//! [`GpuSsrConfig`] immediate block. Runs after the Hi-Z build, the repack and
//! the colour-pyramid build (all its inputs) and before the composite.

use bevy_asset::{load_embedded_asset, Handle};
use bevy_ecs::prelude::*;
use bevy_material::{
    bind_group_layout_entries::{
        binding_types::{sampler, texture_2d, texture_storage_2d},
        BindGroupLayoutEntries,
    },
    descriptor::BindGroupLayoutDescriptor,
};
use bevy_math::Vec4;
use bevy_render::{
    render_resource::{
        AddressMode, BindGroup, BindGroupEntries, BindGroupLayout, CachedComputePipelineId,
        ComputePassDescriptor, ComputePipelineDescriptor, FilterMode, MipmapFilterMode,
        PipelineCache, Sampler, SamplerBindingType, SamplerDescriptor, ShaderStages,
        StorageTextureAccess, TextureSampleType,
    },
    renderer::{RenderContext, RenderDevice, ViewQuery},
    view::ExtractedView,
};
use bevy_shader::Shader;

use super::abi::{GpuSsrConfig, SSR_WORKGROUP_SIZE};
use super::resources::{ViewSsrTextures, SSR_OUT_FORMAT};

/// GGX importance-sampled reflection rays traced per pixel.
///
/// Smooth surfaces collapse every sample onto the mirror direction, so this is
/// effectively a single ray there; rougher surfaces spread the samples across
/// the GGX lobe. Eight balances lobe coverage against the per-ray hierarchical
/// march cost ahead of the spatial-reconstruction stage that will let the count
/// drop again.
const SSR_SAMPLE_COUNT: u32 = 8;

/// Compute pipeline, its owned group-0 layout, and the filtering sampler the
/// trace reads the colour pyramid through.
#[derive(Resource)]
pub(crate) struct SsrTracePipeline {
    /// `trace_ssr` compute entry point, specialized against the group-0 layout
    /// and the 192-byte [`GpuSsrConfig`] immediate block.
    trace: CachedComputePipelineId,
    /// group 0: Hi-Z + depth + `normal_roughness` reads, colour pyramid + its
    /// sampler, and the write-only reflection output.
    layout: BindGroupLayout,
    /// Trilinear clamp sampler bound at binding 4 so the fractional
    /// roughness-selected colour mip interpolates. Pass-owned (not per-view).
    sampler: Sampler,
}

/// group-0 layout mirroring `ssr.wesl`: three non-filterable float reads (the
/// Hi-Z pyramid, device depth and packed normal/roughness, all `textureLoad`ed),
/// the *filterable* colour pyramid + its filtering sampler (sampled with an
/// explicit LOD), then the write-only `rgba16float` reflection output.
fn layout_entries() -> BindGroupLayoutEntries<6> {
    BindGroupLayoutEntries::sequential(
        ShaderStages::COMPUTE,
        (
            texture_2d(TextureSampleType::Float { filterable: false }),
            texture_2d(TextureSampleType::Float { filterable: false }),
            texture_2d(TextureSampleType::Float { filterable: false }),
            texture_2d(TextureSampleType::Float { filterable: true }),
            sampler(SamplerBindingType::Filtering),
            texture_storage_2d(SSR_OUT_FORMAT, StorageTextureAccess::WriteOnly),
        ),
    )
}

/// `RenderStartup` initializer for [`SsrTracePipeline`].
pub(crate) fn init_ssr_trace_pipeline(
    mut commands: Commands,
    device: Res<RenderDevice>,
    cache: Res<PipelineCache>,
    asset_server: Res<bevy_asset::AssetServer>,
) {
    let entries = layout_entries();
    let descriptor = BindGroupLayoutDescriptor::new("prism SSR trace", &entries);
    let layout = device.create_bind_group_layout("prism SSR trace", &entries);

    // Trilinear clamp: linear min/mag plus linear mip so the fractional
    // roughness-selected colour LOD interpolates across pyramid levels.
    let sampler = device.create_sampler(&SamplerDescriptor {
        label: Some("prism SSR trace colour sampler"),
        address_mode_u: AddressMode::ClampToEdge,
        address_mode_v: AddressMode::ClampToEdge,
        address_mode_w: AddressMode::ClampToEdge,
        mag_filter: FilterMode::Linear,
        min_filter: FilterMode::Linear,
        mipmap_filter: MipmapFilterMode::Linear,
        ..Default::default()
    });

    let shader: Handle<Shader> = load_embedded_asset!(asset_server.as_ref(), "../shaders/ssr.wesl");

    let trace = cache.queue_compute_pipeline(ComputePipelineDescriptor {
        label: Some("prism SSR trace".into()),
        layout: vec![descriptor],
        immediate_size: size_of::<GpuSsrConfig>() as u32,
        shader,
        entry_point: Some("trace_ssr".into()),
        ..Default::default()
    });

    commands.insert_resource(SsrTracePipeline {
        trace,
        layout,
        sampler,
    });
}

/// The trace's group-0 bind group for a single view. Present only when every
/// backing SSR texture (Hi-Z, depth, `normal_roughness`, colour pyramid, output)
/// is resident.
#[derive(Component)]
pub(crate) struct ViewSsrTraceBindGroup {
    group: BindGroup,
}

/// `PrepareBindGroups` system building the trace bind group for every view with
/// resident [`ViewSsrTextures`].
pub(crate) fn prepare_ssr_trace_bind_groups(
    mut commands: Commands,
    pipeline: Res<SsrTracePipeline>,
    device: Res<RenderDevice>,
    views: Query<(Entity, &ViewSsrTextures)>,
) {
    for (entity, textures) in &views {
        let group = device.create_bind_group(
            "prism SSR trace",
            &pipeline.layout,
            &BindGroupEntries::sequential((
                textures.hzb_view(),
                textures.scene_depth_sampled(),
                textures.normal_roughness_view(),
                textures.color_sampled_view(),
                &pipeline.sampler,
                textures.ssr_out_view(),
            )),
        );
        commands
            .entity(entity)
            .insert(ViewSsrTraceBindGroup { group });
    }
}

/// `Core3d` node recording the screen-space trace dispatch for every view.
///
/// Runs after the Hi-Z build, the roughness repack and the colour-pyramid build
/// (all of which fill its group-0 inputs) and before the composite that folds
/// the reflection back into `scene_color`. Dispatches one workgroup per 8x8
/// pixel tile; the shader's per-pixel guards drop the background and off-screen
/// rays.
pub(crate) fn ssr_trace_pass(
    settings: Res<super::super::runtime::PrismShadingSettings>,
    view: ViewQuery<(&ViewSsrTextures, &ViewSsrTraceBindGroup, &ExtractedView)>,
    pipeline: Res<SsrTracePipeline>,
    cache: Res<PipelineCache>,
    mut ctx: RenderContext,
) {
    if !settings.enable_ssr {
        return;
    }
    let (textures, group, extracted) = view.into_inner();

    let Some(trace) = cache.get_compute_pipeline(pipeline.trace) else {
        return;
    };

    let size = textures.size;
    if size.x == 0 || size.y == 0 {
        return;
    }

    let clip_from_view = extracted.clip_from_view;
    let view_from_clip = clip_from_view.inverse();
    // Recover the positive near-plane distance from the inverse projection: in
    // Prism's reverse-Z convention device depth 1.0 is the near plane, so
    // inverse-projecting clip (0, 0, 1, 1) yields a view-space point at `-near`
    // along the camera's `-Z`. Robust for both finite and infinite reverse-Z.
    let near_view = view_from_clip * Vec4::new(0.0, 0.0, 1.0, 1.0);
    let near = if near_view.w.abs() > f32::EPSILON {
        (near_view.z / near_view.w).abs().max(1.0e-3)
    } else {
        1.0e-3
    };
    // View-space march length (view units). A fixed budget keeps the trace
    // bounded independent of scene scale; the confidence distance-fade tapers
    // the tail so an over-long ray never hard-cuts.
    let max_distance = 100.0_f32;
    let color_max_mip = (textures.color_mip_count().max(1) - 1) as f32;

    let config = GpuSsrConfig::from_view(
        clip_from_view,
        view_from_clip,
        near,
        max_distance,
        size,
        color_max_mip,
        SSR_SAMPLE_COUNT,
    );

    let workgroups_x = size.x.div_ceil(SSR_WORKGROUP_SIZE);
    let workgroups_y = size.y.div_ceil(SSR_WORKGROUP_SIZE);

    let mut pass = ctx
        .command_encoder()
        .begin_compute_pass(&ComputePassDescriptor {
            label: Some("prism SSR trace"),
            timestamp_writes: None,
        });
    pass.set_pipeline(trace);
    pass.set_bind_group(0, &group.group, &[]);
    pass.set_immediates(0, bytemuck::bytes_of(&config));
    pass.dispatch_workgroups(workgroups_x, workgroups_y, 1);
}
