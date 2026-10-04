//! Glossy-specular ReSTIR composite: pipeline, per-view bind group and the
//! `Core3d` dispatch node that folds the reuse pass's resolved specular back
//! over the shaded scene under an energy-conserving substitution.
//!
//! The reuse pass (`spec_gi_reuse.wesl`) leaves `rgb` = the resolved glossy
//! contribution (already BRDF-weighted: `throughput * radiance * W`, so it is
//! directly comparable to the IBL specular radiance and is *not* re-scaled by
//! the split-sum env-BRDF) and `a` = the reservoir's normalised confidence in
//! its resolved target. This stage substitutes that reflection for the IBL
//! specular the shading resolve already folded into `scene_color`, the GPU twin
//! of option C in the composite step of
//! [`prism_render_shading::gi::spec_gi`], computing
//! `scene_color.rgb = base + (contrib - env_specular) * confidence` (UE-style:
//! a hit replaces the environment specular, a miss falls back to IBL).
//!
//! This pass owns the specular substitution whenever the glossy-ReSTIR
//! subsystem is enabled; [`super::super::ssr::ssr_composite_pass`]
//! early-returns under the same gate so the IBL specular is swapped exactly
//! once (no double subtract of `env_specular`).
//!
//! `scene_color` is an `rgba16float` storage image, which is not read-write
//! storage-capable, so the *base* colour is read from colour-pyramid level 0 —
//! a byte-for-byte copy of `scene_color` the colour-mip build takes immediately
//! before this pass — and only the write side touches `scene_color`, keeping the
//! pass free of any read/write aliasing hazard on a single resource.
//!
//! It reads one bind group (group 0, matching `shaders/spec_gi_composite.wesl`):
//!
//! * `0` colour-pyramid level 0 (the untouched shaded colour, `textureLoad`ed),
//! * `1` the reuse pass's resolved specular+confidence target (`textureLoad`ed),
//! * `2` the write-only `rgba16float` `scene_color` substituted in place, and
//! * `3` the IBL specular the resolve folded in (subtracted under confidence).
//!
//! The framebuffer extent travels in the [`GpuSpecGiCompositeParams`] immediate
//! block. Runs after the reuse dispatch (its resolved input) and the colour-mip
//! build (its base copy) and before the main pass that presents `scene_color`.

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
};
use bevy_shader::Shader;

use super::super::resources::{ViewVisibilityBuffer, SCENE_COLOR_FORMAT};
use super::super::runtime::PrismShadingSettings;
use super::super::ssr::ViewSsrTextures;
use super::abi::{GpuSpecGiCompositeParams, SPEC_GI_WORKGROUP_SIZE};
use super::resources::ViewSpecGiReuse;

/// Compute pipeline and its owned group-0 layout for the glossy-specular
/// composite.
#[derive(Resource)]
pub(crate) struct SpecGiCompositePipeline {
    /// `spec_gi_composite` compute entry point, specialized against the group-0
    /// layout and the 16-byte [`GpuSpecGiCompositeParams`] immediate block.
    composite: CachedComputePipelineId,
    /// group 0: colour-pyramid level 0 + resolved reads, `scene_color`
    /// substituted in place, IBL specular read for the energy-conserving swap.
    layout: BindGroupLayout,
}

/// group-0 layout mirroring `spec_gi_composite.wesl`: the colour-pyramid level-0
/// copy and the reuse pass's resolved target (both `textureLoad`ed, so
/// non-filterable), then the write-only `rgba16float` `scene_color` output, then
/// the IBL specular read subtracted under confidence.
fn layout_entries() -> BindGroupLayoutEntries<4> {
    BindGroupLayoutEntries::sequential(
        ShaderStages::COMPUTE,
        (
            texture_2d(TextureSampleType::Float { filterable: false }),
            texture_2d(TextureSampleType::Float { filterable: false }),
            texture_storage_2d(SCENE_COLOR_FORMAT, StorageTextureAccess::WriteOnly),
            texture_2d(TextureSampleType::Float { filterable: false }),
        ),
    )
}

/// `RenderStartup` initializer for [`SpecGiCompositePipeline`].
pub(crate) fn init_spec_gi_composite_pipeline(
    mut commands: Commands,
    device: Res<RenderDevice>,
    cache: Res<PipelineCache>,
    asset_server: Res<bevy_asset::AssetServer>,
) {
    let entries = layout_entries();
    let descriptor = BindGroupLayoutDescriptor::new("prism spec_gi composite", &entries);
    let layout = device.create_bind_group_layout("prism spec_gi composite", &entries);

    let shader: Handle<Shader> =
        load_embedded_asset!(asset_server.as_ref(), "../shaders/spec_gi_composite.wesl");

    let composite = cache.queue_compute_pipeline(ComputePipelineDescriptor {
        label: Some("prism spec_gi composite".into()),
        layout: vec![descriptor],
        immediate_size: size_of::<GpuSpecGiCompositeParams>() as u32,
        shader,
        entry_point: Some("spec_gi_composite".into()),
        ..Default::default()
    });

    commands.insert_resource(SpecGiCompositePipeline { composite, layout });
}

/// The composite's group-0 bind group for a single view. Present only when the
/// colour pyramid's level 0 (the base copy) is resident; the reuse resources and
/// visibility buffer are guaranteed live together by the subsystem gate.
#[derive(Component)]
pub(crate) struct ViewSpecGiCompositeBindGroup {
    group: BindGroup,
}

/// `PrepareBindGroups` system building [`ViewSpecGiCompositeBindGroup`] for every
/// view that has a visibility buffer (for `scene_color` + IBL specular), resident
/// SSR textures (the colour pyramid) and resident reuse resources (the resolved
/// specular target). Clears any stale group when the colour pyramid has no level
/// 0 to lift the base colour from.
pub(crate) fn prepare_spec_gi_composite_bind_groups(
    mut commands: Commands,
    pipeline: Res<SpecGiCompositePipeline>,
    device: Res<RenderDevice>,
    views: Query<(
        Entity,
        &ViewVisibilityBuffer,
        &ViewSsrTextures,
        &ViewSpecGiReuse,
        // The spatial denoiser's filtered specular target, present only while
        // the denoise pass is enabled (same gate as the reuse resolve). When
        // resident the composite reads the denoised specular instead of the
        // raw reuse resolve; otherwise it falls back to the resolve directly.
        Option<&super::super::spec_denoise::ViewSpecDenoise>,
    )>,
) {
    for (entity, visibility, textures, spec_gi, denoise) in &views {
        let Some(color_l0) = textures.color_mip_view(0) else {
            commands
                .entity(entity)
                .remove::<ViewSpecGiCompositeBindGroup>();
            continue;
        };
        let group = device.create_bind_group(
            "prism spec_gi composite",
            &pipeline.layout,
            &BindGroupEntries::sequential((
                color_l0,
                // Prefer the spatially denoised specular when the denoise pass
                // is resident; fall back to the raw reuse resolve otherwise.
                // Both carry the ReSTIR confidence in `.a`, so the composite's
                // confidence read is valid either way.
                denoise
                    .map(|d| d.filtered_view())
                    .unwrap_or_else(|| spec_gi.resolved_view()),
                visibility.scene_color_view(),
                visibility.ssr_env_specular_view(),
            )),
        );
        commands
            .entity(entity)
            .insert(ViewSpecGiCompositeBindGroup { group });
    }
}

/// `Core3d` node recording the glossy-specular composite dispatch for every
/// view.
///
/// Gated on `enable_spec_gi`; runs after the reuse dispatch (its resolved input)
/// and the colour-mip build (its base copy) and before the main pass that
/// presents `scene_color`. Dispatches one workgroup per 8x8 pixel tile; the
/// shader bounds-checks every invocation against the framebuffer extent.
pub(crate) fn spec_gi_composite_pass(
    settings: Res<PrismShadingSettings>,
    view: ViewQuery<(&ViewSsrTextures, &ViewSpecGiCompositeBindGroup)>,
    pipeline: Res<SpecGiCompositePipeline>,
    cache: Res<PipelineCache>,
    mut ctx: RenderContext,
) {
    if !settings.enable_spec_gi {
        return;
    }
    let (textures, group) = view.into_inner();

    let Some(composite) = cache.get_compute_pipeline(pipeline.composite) else {
        return;
    };

    let size = textures.size;
    if size.x == 0 || size.y == 0 {
        return;
    }

    let params = GpuSpecGiCompositeParams::new(size.x, size.y);
    let workgroups_x = size.x.div_ceil(SPEC_GI_WORKGROUP_SIZE);
    let workgroups_y = size.y.div_ceil(SPEC_GI_WORKGROUP_SIZE);

    let mut pass = ctx
        .command_encoder()
        .begin_compute_pass(&ComputePassDescriptor {
            label: Some("prism spec_gi composite"),
            timestamp_writes: None,
        });
    pass.set_pipeline(composite);
    pass.set_bind_group(0, &group.group, &[]);
    pass.set_immediates(0, bytemuck::bytes_of(&params));
    pass.dispatch_workgroups(workgroups_x, workgroups_y, 1);
}
