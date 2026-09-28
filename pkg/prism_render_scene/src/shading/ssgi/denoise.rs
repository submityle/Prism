//! SSGI edge-aware spatial denoiser: pipeline, per-view bind group, and the
//! `Core3d` dispatch node.
//!
//! The GPU twin of [`prism_render_shading::screen_space::denoise_ssgi`]. A
//! single SSGI gather traces only a handful of cosine-weighted rays per pixel,
//! so the raw [`super::resources::ViewSsgiTextures::ssgi_out_view`] buffer is
//! heavy with Monte-Carlo noise — grainy on open surfaces, splotchy where few
//! rays find a valid hit. This pass is the spatial half of the AAA denoiser: a
//! joint **bilateral** blur that averages each pixel with its
//! `(2 * radius + 1)^2` neighbours while depth *and* normal edge-stopping terms
//! keep the indirect radiance from bleeding across geometry seams — the same
//! `XeGTAO`-style kernel [`super::super::ao`] applies to AO, generalised to the
//! four-channel SSGI output (rgb pre-albedo radiance, a confidence).
//!
//! It reads one bind group (group 0, matching `shaders/ssgi_denoise.wesl`):
//!
//! * `0` the raw SSGI gather the trace wrote (`textureLoad`ed per tap),
//! * `1` the full-resolution reverse-Z device depth (reprojected to a linear
//!   view depth for the depth edge stop), and
//! * `2` the packed view-space normal + roughness (its `.xyz` view normal
//!   drives the normal edge stop),
//! * `3` the write-only `rgba16float` denoised output the composite folds in.
//!
//! The inverse projection, framebuffer extent and the golden bilateral tunables
//! travel in the [`GpuSsgiDenoiseConfig`] immediate block. It runs after the
//! trace fills `ssgi_out` and before the composite that folds the denoised
//! radiance into `scene_color`.

use bevy_asset::{load_embedded_asset, Handle};
use bevy_ecs::prelude::*;
use bevy_material::{
    bind_group_layout_entries::{
        binding_types::{texture_2d, texture_storage_2d},
        BindGroupLayoutEntries,
    },
    descriptor::BindGroupLayoutDescriptor,
};
use bevy_render::{
    render_resource::{
        BindGroup, BindGroupEntries, BindGroupLayout, CachedComputePipelineId,
        ComputePassDescriptor, ComputePipelineDescriptor, PipelineCache, ShaderStages,
        StorageTextureAccess, TextureSampleType,
    },
    renderer::{RenderContext, RenderDevice, ViewQuery},
    view::ExtractedView,
};
use bevy_shader::Shader;

use super::super::ssr::ViewSsrTextures;
use super::abi::{GpuSsgiDenoiseConfig, SSGI_WORKGROUP_SIZE};
use super::resources::{ViewSsgiTextures, SSGI_DENOISED_FORMAT};

/// Compute pipeline and its owned group-0 layout for the SSGI spatial denoiser.
#[derive(Resource)]
pub(crate) struct SsgiDenoisePipeline {
    /// `denoise_ssgi` compute entry point, specialized against the group-0
    /// layout and the 96-byte [`GpuSsgiDenoiseConfig`] immediate block.
    denoise: CachedComputePipelineId,
    /// group 0: raw SSGI + device depth + `normal_roughness` reads, then the
    /// write-only `rgba16float` denoised output.
    layout: BindGroupLayout,
}

/// group-0 layout mirroring `ssgi_denoise.wesl`: three non-filterable float
/// reads (the raw gather, device depth and packed view normal, all
/// `textureLoad`ed), then the write-only `rgba16float` denoised output.
fn layout_entries() -> BindGroupLayoutEntries<4> {
    BindGroupLayoutEntries::sequential(
        ShaderStages::COMPUTE,
        (
            texture_2d(TextureSampleType::Float { filterable: false }),
            texture_2d(TextureSampleType::Float { filterable: false }),
            texture_2d(TextureSampleType::Float { filterable: false }),
            texture_storage_2d(SSGI_DENOISED_FORMAT, StorageTextureAccess::WriteOnly),
        ),
    )
}

/// `RenderStartup` initializer for [`SsgiDenoisePipeline`].
pub(crate) fn init_ssgi_denoise_pipeline(
    mut commands: Commands,
    device: Res<RenderDevice>,
    cache: Res<PipelineCache>,
    asset_server: Res<bevy_asset::AssetServer>,
) {
    let entries = layout_entries();
    let descriptor = BindGroupLayoutDescriptor::new("prism SSGI denoise", &entries);
    let layout = device.create_bind_group_layout("prism SSGI denoise", &entries);

    let shader: Handle<Shader> =
        load_embedded_asset!(asset_server.as_ref(), "../shaders/ssgi_denoise.wesl");

    let denoise = cache.queue_compute_pipeline(ComputePipelineDescriptor {
        label: Some("prism SSGI denoise".into()),
        layout: vec![descriptor],
        immediate_size: size_of::<GpuSsgiDenoiseConfig>() as u32,
        shader,
        entry_point: Some("denoise_ssgi".into()),
        ..Default::default()
    });

    commands.insert_resource(SsgiDenoisePipeline { denoise, layout });
}

/// The denoiser's group-0 bind group for a single view. Present only when every
/// backing input (SSR's device depth and packed normal, plus the SSGI raw and
/// denoised targets) is resident.
#[derive(Component)]
pub(crate) struct ViewSsgiDenoiseBindGroup {
    group: BindGroup,
}

/// `PrepareBindGroups` system building the SSGI denoise bind group for every
/// view that has resident SSR textures and SSGI targets.
pub(crate) fn prepare_ssgi_denoise_bind_groups(
    mut commands: Commands,
    pipeline: Res<SsgiDenoisePipeline>,
    device: Res<RenderDevice>,
    views: Query<(Entity, &ViewSsrTextures, &ViewSsgiTextures)>,
) {
    for (entity, ssr, ssgi) in &views {
        let group = device.create_bind_group(
            "prism SSGI denoise",
            &pipeline.layout,
            &BindGroupEntries::sequential((
                ssgi.ssgi_out_view(),
                ssr.scene_depth_view(),
                ssr.normal_roughness_view(),
                ssgi.ssgi_denoised_view(),
            )),
        );
        commands
            .entity(entity)
            .insert(ViewSsgiDenoiseBindGroup { group });
    }
}

/// `Core3d` node recording the SSGI denoise dispatch for every view.
///
/// Runs after the trace fills `ssgi_out` and before the composite folds the
/// denoised radiance into `scene_color`. Dispatches one workgroup per 8x8 pixel
/// tile; the shader's per-pixel guard drops the off-screen invocations, and the
/// reverse-Z background stops the blur pass-through untouched.
pub(crate) fn ssgi_denoise_pass(
    settings: Res<super::super::runtime::PrismShadingSettings>,
    view: ViewQuery<(&ViewSsgiTextures, &ViewSsgiDenoiseBindGroup, &ExtractedView)>,
    pipeline: Res<SsgiDenoisePipeline>,
    cache: Res<PipelineCache>,
    mut ctx: RenderContext,
) {
    if !settings.enable_ssgi {
        return;
    }
    let (textures, group, extracted) = view.into_inner();

    let Some(denoise) = cache.get_compute_pipeline(pipeline.denoise) else {
        return;
    };

    let size = textures.size;
    if size.x == 0 || size.y == 0 {
        return;
    }

    let view_from_clip = extracted.clip_from_view.inverse();
    let config = GpuSsgiDenoiseConfig::from_view(
        view_from_clip,
        size,
        settings.ssgi_denoise_radius,
        settings.ssgi_denoise_spatial_sigma,
        settings.ssgi_denoise_depth_sigma,
        settings.ssgi_denoise_normal_power,
    );

    let workgroups_x = size.x.div_ceil(SSGI_WORKGROUP_SIZE);
    let workgroups_y = size.y.div_ceil(SSGI_WORKGROUP_SIZE);

    let mut pass = ctx
        .command_encoder()
        .begin_compute_pass(&ComputePassDescriptor {
            label: Some("prism SSGI denoise"),
            timestamp_writes: None,
        });
    pass.set_pipeline(denoise);
    pass.set_bind_group(0, &group.group, &[]);
    pass.set_immediates(0, bytemuck::bytes_of(&config));
    pass.dispatch_workgroups(workgroups_x, workgroups_y, 1);
}
