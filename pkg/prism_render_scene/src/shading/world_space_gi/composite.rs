//! World-space GI composite: the copy + fold pipelines, per-view bind groups,
//! and the `Core3d` dispatch that folds the resolved diffuse GI irradiance
//! back over the shaded scene.
//!
//! The resolve (`world_space_gi_resolve.wesl`) leaves `rgb` = the pre-albedo
//! diffuse GI irradiance (golden `evaluate_irradiance`, times the artistic
//! intensity) and `a` = a `[0, 1]` blend confidence in `gi_out`. This is the
//! *same* quantity the resolve pass exports at `ssgi_ambient` (pre-albedo IBL /
//! SH irradiance), so this stage performs the *same* energy-conserving
//! substitution SSGI does, swapping the flat IBL ambient the shading resolve
//! already folded into `scene_color` for the world-space gather under its
//! confidence:
//!
//! ```text
//! scene = base + confidence * albedo * (gi_out - ambient)
//! ```
//!
//! A fully confident pixel (`a == 1`) replaces the flat ambient with the
//! world-space indirect irradiance; a miss (`a == 0`) leaves the shaded colour
//! untouched, preserving the IBL ambient. The CPU golden in
//! [`prism_render_shading::gi::world_space`] has no composite step — it stops
//! at the resolved irradiance — so this substitution matches golden semantics:
//! `gi_out.rgb` is exactly the golden irradiance, and folding it in under
//! confidence never creates energy.
//!
//! `scene_color` is an `rgba16float` storage image, which is not read-write
//! storage-capable, so the pass runs two passes in one encoder (wgpu inserts
//! the barrier between them):
//!
//! * a **copy** pass lifts the shading-resolved `scene_color` into a scratch
//!   `gi_base` texture (`textureLoad(scene_color) -> gi_base`), and
//! * a **fold** pass reads that base plus the GI gather and the resolve's
//!   albedo / ambient exports and writes the substitution into `scene_color`.
//!
//! Reading the base from a distinct texture and writing `scene_color` keeps the
//! pass free of any storage read/write aliasing hazard. It must run after
//! `world_space_gi_pass` (its `gi_out` input) and, being a `scene_color`
//! writer, is sequenced into the `scene_color`-writer chain after the SSGI
//! composite.

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
use super::abi::{GpuWorldSpaceGiCompositeParams, WORLD_SPACE_GI_WORKGROUP_SIZE};
use super::resources::ViewWorldSpaceGi;
use super::settings::PrismWorldSpaceGiSettings;

/// The two compute pipelines and their owned group-0 layouts for the
/// world-space GI composite: the `scene_color` -> `gi_base` copy and the
/// energy-conserving GI fold, both entry points of
/// `shaders/world_space_gi_composite.wesl`.
#[derive(Resource)]
pub(crate) struct WorldSpaceGiCompositePipeline {
    /// `wsgi_copy_base` entry point: lifts `scene_color` into `gi_base`.
    copy: CachedComputePipelineId,
    /// `wsgi_composite` entry point: folds the gather into `scene_color`.
    fold: CachedComputePipelineId,
    /// copy layout: `scene_color` read + `gi_base` write.
    copy_layout: BindGroupLayout,
    /// fold layout: `gi_base` + `gi_out` + albedo + ambient reads,
    /// `scene_color` written in place.
    fold_layout: BindGroupLayout,
}

/// copy-pass layout mirroring `world_space_gi_composite.wesl`'s
/// `wsgi_copy_base`: one non-filterable float read (the shading-resolved
/// `scene_color`) then the write-only `gi_base` scratch.
fn copy_layout_entries() -> BindGroupLayoutEntries<2> {
    BindGroupLayoutEntries::sequential(
        ShaderStages::COMPUTE,
        (
            texture_2d(TextureSampleType::Float { filterable: false }),
            texture_storage_2d(SCENE_COLOR_FORMAT, StorageTextureAccess::WriteOnly),
        ),
    )
}

/// fold-pass layout mirroring `world_space_gi_composite.wesl`'s
/// `wsgi_composite`: the `gi_base` copy and the `gi_out` gather (both
/// `textureLoad`ed), the write-only `scene_color`, then the resolve's
/// Lambertian albedo and pre-albedo ambient exports.
fn fold_layout_entries() -> BindGroupLayoutEntries<5> {
    BindGroupLayoutEntries::sequential(
        ShaderStages::COMPUTE,
        (
            texture_2d(TextureSampleType::Float { filterable: false }),
            texture_2d(TextureSampleType::Float { filterable: false }),
            texture_storage_2d(SCENE_COLOR_FORMAT, StorageTextureAccess::WriteOnly),
            texture_2d(TextureSampleType::Float { filterable: false }),
            texture_2d(TextureSampleType::Float { filterable: false }),
        ),
    )
}

/// `RenderStartup` initializer for [`WorldSpaceGiCompositePipeline`].
pub(crate) fn init_world_space_gi_composite_pipeline(
    mut commands: Commands,
    device: Res<RenderDevice>,
    cache: Res<PipelineCache>,
    asset_server: Res<bevy_asset::AssetServer>,
) {
    let copy_entries = copy_layout_entries();
    let copy_descriptor =
        BindGroupLayoutDescriptor::new("prism world-space GI composite copy", &copy_entries);
    let copy_layout =
        device.create_bind_group_layout("prism world-space GI composite copy", &copy_entries);

    let fold_entries = fold_layout_entries();
    let fold_descriptor =
        BindGroupLayoutDescriptor::new("prism world-space GI composite fold", &fold_entries);
    let fold_layout =
        device.create_bind_group_layout("prism world-space GI composite fold", &fold_entries);

    let shader: Handle<Shader> = load_embedded_asset!(
        asset_server.as_ref(),
        "../shaders/world_space_gi_composite.wesl"
    );

    let copy = cache.queue_compute_pipeline(ComputePipelineDescriptor {
        label: Some("prism world-space GI composite copy".into()),
        layout: vec![copy_descriptor],
        immediate_size: size_of::<GpuWorldSpaceGiCompositeParams>() as u32,
        shader: shader.clone(),
        entry_point: Some("wsgi_copy_base".into()),
        ..Default::default()
    });
    let fold = cache.queue_compute_pipeline(ComputePipelineDescriptor {
        label: Some("prism world-space GI composite fold".into()),
        layout: vec![fold_descriptor],
        immediate_size: size_of::<GpuWorldSpaceGiCompositeParams>() as u32,
        shader,
        entry_point: Some("wsgi_composite".into()),
        ..Default::default()
    });

    commands.insert_resource(WorldSpaceGiCompositePipeline {
        copy,
        fold,
        copy_layout,
        fold_layout,
    });
}

/// The composite's two per-view bind groups. Present only when the visibility
/// buffer (for `scene_color` and the albedo / ambient exports) and the
/// world-space GI targets are resident.
#[derive(Component)]
pub(crate) struct ViewWorldSpaceGiCompositeBindGroups {
    /// group 0 for `wsgi_copy_base`: `scene_color` read + `gi_base` write.
    copy: BindGroup,
    /// group 0 for `wsgi_composite`: `gi_base` + `gi_out` + albedo + ambient
    /// reads and the write-only `scene_color`.
    fold: BindGroup,
}

/// `PrepareBindGroups` system building [`ViewWorldSpaceGiCompositeBindGroups`]
/// for every view that has both a visibility buffer and resident world-space
/// GI targets.
pub(crate) fn prepare_world_space_gi_composite_bind_groups(
    mut commands: Commands,
    pipeline: Res<WorldSpaceGiCompositePipeline>,
    device: Res<RenderDevice>,
    views: Query<(Entity, &ViewVisibilityBuffer, &ViewWorldSpaceGi)>,
) {
    for (entity, visibility, gi) in &views {
        let copy = device.create_bind_group(
            "prism world-space GI composite copy",
            &pipeline.copy_layout,
            &BindGroupEntries::sequential((visibility.scene_color_view(), gi.gi_base_view())),
        );
        let fold = device.create_bind_group(
            "prism world-space GI composite fold",
            &pipeline.fold_layout,
            &BindGroupEntries::sequential((
                gi.gi_base_view(),
                gi.gi_out_view(),
                visibility.scene_color_view(),
                visibility.ssgi_albedo_view(),
                visibility.ssgi_ambient_view(),
            )),
        );
        commands
            .entity(entity)
            .insert(ViewWorldSpaceGiCompositeBindGroups { copy, fold });
    }
}

/// `Core3d` scheduling system recording the world-space GI composite for every
/// view.
///
/// Runs after `world_space_gi_pass` (its `gi_out` input) and, being a
/// `scene_color` writer, after the SSGI composite in the `scene_color`-writer
/// chain, and before the main pass that presents `scene_color`. Records the
/// copy then the fold in a single encoder; wgpu inserts the storage barrier
/// between them. Dispatches one workgroup per 8x8 pixel tile; both shader
/// entry points bounds-check every invocation.
pub(crate) fn world_space_gi_composite_pass(
    settings: Res<PrismWorldSpaceGiSettings>,
    view: ViewQuery<(&ViewWorldSpaceGi, &ViewWorldSpaceGiCompositeBindGroups)>,
    pipeline: Res<WorldSpaceGiCompositePipeline>,
    cache: Res<PipelineCache>,
    mut ctx: RenderContext,
) {
    if !settings.enabled {
        return;
    }
    let (gi, groups) = view.into_inner();

    let (Some(copy), Some(fold)) = (
        cache.get_compute_pipeline(pipeline.copy),
        cache.get_compute_pipeline(pipeline.fold),
    ) else {
        return;
    };

    let size = gi.size;
    if size.x == 0 || size.y == 0 {
        return;
    }

    let params = GpuWorldSpaceGiCompositeParams::new(size.x, size.y);
    let workgroups_x = size.x.div_ceil(WORLD_SPACE_GI_WORKGROUP_SIZE);
    let workgroups_y = size.y.div_ceil(WORLD_SPACE_GI_WORKGROUP_SIZE);

    let encoder = ctx.command_encoder();
    {
        let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("prism world-space GI composite copy"),
            timestamp_writes: None,
        });
        pass.set_pipeline(copy);
        pass.set_bind_group(0, &groups.copy, &[]);
        pass.set_immediates(0, bytemuck::bytes_of(&params));
        pass.dispatch_workgroups(workgroups_x, workgroups_y, 1);
    }
    {
        let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("prism world-space GI composite fold"),
            timestamp_writes: None,
        });
        pass.set_pipeline(fold);
        pass.set_bind_group(0, &groups.fold, &[]);
        pass.set_immediates(0, bytemuck::bytes_of(&params));
        pass.dispatch_workgroups(workgroups_x, workgroups_y, 1);
    }
}
