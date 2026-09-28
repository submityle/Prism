//! SSGI screen-space trace: pipeline, per-view bind group, and the `Core3d`
//! dispatch node.
//!
//! This is the indirect-diffuse gather itself, the GPU twin of
//! [`prism_render_shading::screen_space::gi`]. For every covered pixel it casts
//! several cosine-weighted hemisphere rays around the reconstructed surface
//! normal, marches each across the reverse-Z Hi-Z "nearest depth" pyramid the
//! reflection subsystem already builds, and samples the current-frame colour at
//! the hit (picking up one indirect bounce of on-screen radiance / colour
//! bleeding). Rays that escape the framebuffer take the resolve's IBL/SH
//! ambient, so the gather augments the ambient term without energy
//! discontinuities. It writes the pre-albedo mean indirect radiance (rgb) plus
//! a `[0, 1]` blend confidence (a) the composite folds back over `scene_color`.
//!
//! It reads one bind group (group 0, matching `shaders/ssgi.wesl`):
//!
//! * `0` the multi-mip Hi-Z pyramid (`textureLoad`ed per level in the march),
//! * `1` the full-resolution device depth (view-position reconstruction),
//! * `2` the packed view-normal + roughness the repack produced (SSGI reads the
//!   normal to orient the hemisphere),
//! * `3` the current-frame colour pyramid (sampled at LOD0 with a filtering
//!   sampler so a hit gathers the shaded surface radiance),
//! * `4` that filtering sampler,
//! * `5` the resolve's pre-albedo IBL/SH diffuse irradiance (the miss/sky
//!   fallback the gather blends toward), and
//! * `6` the write-only `rgba16float` SSGI output.
//!
//! The view/camera matrices and the golden march/gather tunables travel in the
//! [`GpuSsgiConfig`] immediate block. Because SSGI reuses SSR's rebuilt inputs
//! it runs after the Hi-Z build, the repack and the colour-pyramid build (all
//! its inputs) and before the composite.

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

use super::super::resources::ViewVisibilityBuffer;
use super::abi::{GpuSsgiConfig, SSGI_WORKGROUP_SIZE};
use super::resources::{ViewSsgiTextures, SSGI_OUT_FORMAT};
use super::super::ssr::ViewSsrTextures;

/// Compute pipeline, its owned group-0 layout, and the filtering sampler the
/// gather reads the colour pyramid through.
#[derive(Resource)]
pub(crate) struct SsgiTracePipeline {
    /// `trace_ssgi` compute entry point, specialized against the group-0 layout
    /// and the 176-byte [`GpuSsgiConfig`] immediate block.
    trace: CachedComputePipelineId,
    /// group 0: Hi-Z + depth + `normal_roughness` reads, colour pyramid + its
    /// sampler, the pre-albedo ambient, and the write-only SSGI output.
    layout: BindGroupLayout,
    /// Trilinear clamp sampler bound at binding 4 so the LOD0 colour lookup
    /// filters. Pass-owned (not per-view).
    sampler: Sampler,
}

/// group-0 layout mirroring `ssgi.wesl`: three non-filterable float reads (the
/// Hi-Z pyramid, device depth and packed normal/roughness, all `textureLoad`ed),
/// the *filterable* colour pyramid + its filtering sampler (sampled at LOD0),
/// the non-filterable pre-albedo ambient irradiance, then the write-only
/// `rgba16float` SSGI output.
fn layout_entries() -> BindGroupLayoutEntries<7> {
    BindGroupLayoutEntries::sequential(
        ShaderStages::COMPUTE,
        (
            texture_2d(TextureSampleType::Float { filterable: false }),
            texture_2d(TextureSampleType::Float { filterable: false }),
            texture_2d(TextureSampleType::Float { filterable: false }),
            texture_2d(TextureSampleType::Float { filterable: true }),
            sampler(SamplerBindingType::Filtering),
            texture_2d(TextureSampleType::Float { filterable: false }),
            texture_storage_2d(SSGI_OUT_FORMAT, StorageTextureAccess::WriteOnly),
        ),
    )
}

/// `RenderStartup` initializer for [`SsgiTracePipeline`].
pub(crate) fn init_ssgi_trace_pipeline(
    mut commands: Commands,
    device: Res<RenderDevice>,
    cache: Res<PipelineCache>,
    asset_server: Res<bevy_asset::AssetServer>,
) {
    let entries = layout_entries();
    let descriptor = BindGroupLayoutDescriptor::new("prism SSGI trace", &entries);
    let layout = device.create_bind_group_layout("prism SSGI trace", &entries);

    // Trilinear clamp: linear min/mag plus linear mip. The gather samples the
    // colour pyramid at LOD0, but a filtering sampler keeps the bilinear tap
    // smooth across the fractional hit UV.
    let sampler = device.create_sampler(&SamplerDescriptor {
        label: Some("prism SSGI trace colour sampler"),
        address_mode_u: AddressMode::ClampToEdge,
        address_mode_v: AddressMode::ClampToEdge,
        address_mode_w: AddressMode::ClampToEdge,
        mag_filter: FilterMode::Linear,
        min_filter: FilterMode::Linear,
        mipmap_filter: MipmapFilterMode::Linear,
        ..Default::default()
    });

    let shader: Handle<Shader> =
        load_embedded_asset!(asset_server.as_ref(), "../shaders/ssgi.wesl");

    let trace = cache.queue_compute_pipeline(ComputePipelineDescriptor {
        label: Some("prism SSGI trace".into()),
        layout: vec![descriptor],
        immediate_size: size_of::<GpuSsgiConfig>() as u32,
        shader,
        entry_point: Some("trace_ssgi".into()),
        ..Default::default()
    });

    commands.insert_resource(SsgiTracePipeline {
        trace,
        layout,
        sampler,
    });
}

/// The gather's group-0 bind group for a single view. Present only when every
/// backing input (SSR's Hi-Z/depth/normal_roughness/colour, the resolve's
/// ambient export, and the SSGI output) is resident.
#[derive(Component)]
pub(crate) struct ViewSsgiTraceBindGroup {
    group: BindGroup,
}

/// `PrepareBindGroups` system building the SSGI trace bind group for every view
/// that has resident SSR textures, a visibility buffer (for the ambient export)
/// and SSGI targets.
pub(crate) fn prepare_ssgi_trace_bind_groups(
    mut commands: Commands,
    pipeline: Res<SsgiTracePipeline>,
    device: Res<RenderDevice>,
    views: Query<(
        Entity,
        &ViewSsrTextures,
        &ViewVisibilityBuffer,
        &ViewSsgiTextures,
    )>,
) {
    for (entity, ssr, visibility, ssgi) in &views {
        let group = device.create_bind_group(
            "prism SSGI trace",
            &pipeline.layout,
            &BindGroupEntries::sequential((
                ssr.hzb_view(),
                ssr.scene_depth_sampled(),
                ssr.normal_roughness_view(),
                ssr.color_sampled_view(),
                &pipeline.sampler,
                visibility.ssgi_ambient_view(),
                ssgi.ssgi_out_view(),
            )),
        );
        commands
            .entity(entity)
            .insert(ViewSsgiTraceBindGroup { group });
    }
}

/// `Core3d` node recording the SSGI gather dispatch for every view.
///
/// Runs after the Hi-Z build, the roughness repack and the colour-pyramid build
/// (all of which fill its group-0 inputs) and before the composite that folds
/// the indirect radiance into `scene_color`. Dispatches one workgroup per 8x8
/// pixel tile; the shader's per-pixel guards drop the background and off-screen
/// rays.
pub(crate) fn ssgi_trace_pass(
    settings: Res<super::super::runtime::PrismShadingSettings>,
    view: ViewQuery<(&ViewSsgiTextures, &ViewSsgiTraceBindGroup, &ExtractedView)>,
    pipeline: Res<SsgiTracePipeline>,
    cache: Res<PipelineCache>,
    mut ctx: RenderContext,
) {
    if !settings.enable_ssgi {
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

    let config = GpuSsgiConfig::from_view(
        clip_from_view,
        view_from_clip,
        near,
        settings.ssgi_max_distance,
        size,
        settings.ssgi_sample_count,
    );

    let workgroups_x = size.x.div_ceil(SSGI_WORKGROUP_SIZE);
    let workgroups_y = size.y.div_ceil(SSGI_WORKGROUP_SIZE);

    let mut pass = ctx
        .command_encoder()
        .begin_compute_pass(&ComputePassDescriptor {
            label: Some("prism SSGI trace"),
            timestamp_writes: None,
        });
    pass.set_pipeline(trace);
    pass.set_bind_group(0, &group.group, &[]);
    pass.set_immediates(0, bytemuck::bytes_of(&config));
    pass.dispatch_workgroups(workgroups_x, workgroups_y, 1);
}
