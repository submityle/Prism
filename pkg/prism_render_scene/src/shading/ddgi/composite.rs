//! DDGI composite: the copy + fold pipelines, per-view bind groups, and the
//! `Core3d` dispatch that folds the resolved diffuse-GI irradiance volume back
//! over the shaded scene.
//!
//! The sample pass (`ddgi_sample.wesl`) leaves `rgb` = the pre-albedo
//! diffuse-GI irradiance (golden irradiance-volume evaluation, times the
//! artistic intensity) and `a` = a `[0, 1]` blend confidence in `gi_out`. This
//! is the *same* quantity the shading resolve exports at `ssgi_ambient`
//! (pre-albedo IBL / SH diffuse irradiance), so this stage performs the same
//! energy-conserving substitution SSGI and world-space GI do, swapping the flat
//! IBL ambient the resolve already folded into `scene_color` for the probe
//! gather under its confidence:
//!
//! ```text
//! scene = base + confidence * albedo * (gi_out - ambient)
//! ```
//!
//! A fully confident pixel (`a == 1`) replaces the flat ambient with the
//! irradiance-volume gather; a miss (`a == 0`) leaves the shaded colour
//! untouched, preserving the IBL ambient; a sky / unshaded pixel has
//! `albedo == 0`, so the substitution is a no-op there regardless of
//! confidence. This is UE's DDGI-replaces-sky-diffuse behaviour with an IBL
//! fallback (the chosen "option C"), and because `gi_out.rgb` is exactly the
//! golden irradiance (the CPU golden has no composite step), the fold never
//! creates energy.
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
//! `ddgi_sample_pass` (its `gi_out` input) and, being a `scene_color` writer,
//! is sequenced into the `scene_color`-writer chain after the world-space GI
//! composite and before the surface-cache gather / TAA.

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
use super::abi::{GpuDdgiCompositeParams, DDGI_WORKGROUP_SIZE};
use super::resources::ViewDdgi;
use super::settings::PrismDdgiSettings;

/// The two compute pipelines and their owned group-0 layouts for the DDGI
/// composite: the `scene_color` -> `gi_base` copy and the energy-conserving GI
/// fold, both entry points of `shaders/ddgi_composite.wesl`.
#[derive(Resource)]
pub(crate) struct DdgiCompositePipeline {
    /// `ddgi_copy_base` entry point: lifts `scene_color` into `gi_base`.
    copy: CachedComputePipelineId,
    /// `ddgi_composite` entry point: folds the gather into `scene_color`.
    fold: CachedComputePipelineId,
    /// copy layout: `scene_color` read + `gi_base` write.
    copy_layout: BindGroupLayout,
    /// fold layout: `gi_base` + `gi_out` + albedo + ambient reads,
    /// `scene_color` written in place.
    fold_layout: BindGroupLayout,
}

/// copy-pass layout mirroring `ddgi_composite.wesl`'s `ddgi_copy_base`: one
/// non-filterable float read (the shading-resolved `scene_color`) then the
/// write-only `gi_base` scratch.
fn copy_layout_entries() -> BindGroupLayoutEntries<2> {
    BindGroupLayoutEntries::sequential(
        ShaderStages::COMPUTE,
        (
            texture_2d(TextureSampleType::Float { filterable: false }),
            texture_storage_2d(SCENE_COLOR_FORMAT, StorageTextureAccess::WriteOnly),
        ),
    )
}

/// fold-pass layout mirroring `ddgi_composite.wesl`'s `ddgi_composite`: the
/// `gi_base` copy and the `gi_out` gather (both `textureLoad`ed), the
/// write-only `scene_color`, then the resolve's Lambertian albedo and
/// pre-albedo ambient exports.
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

/// `RenderStartup` initializer for [`DdgiCompositePipeline`].
pub(crate) fn init_ddgi_composite_pipeline(
    mut commands: Commands,
    device: Res<RenderDevice>,
    cache: Res<PipelineCache>,
    asset_server: Res<bevy_asset::AssetServer>,
) {
    let copy_entries = copy_layout_entries();
    let copy_descriptor =
        BindGroupLayoutDescriptor::new("prism DDGI composite copy", &copy_entries);
    let copy_layout = device.create_bind_group_layout("prism DDGI composite copy", &copy_entries);

    let fold_entries = fold_layout_entries();
    let fold_descriptor =
        BindGroupLayoutDescriptor::new("prism DDGI composite fold", &fold_entries);
    let fold_layout = device.create_bind_group_layout("prism DDGI composite fold", &fold_entries);

    let shader: Handle<Shader> =
        load_embedded_asset!(asset_server.as_ref(), "../shaders/ddgi_composite.wesl");

    let copy = cache.queue_compute_pipeline(ComputePipelineDescriptor {
        label: Some("prism DDGI composite copy".into()),
        layout: vec![copy_descriptor],
        immediate_size: size_of::<GpuDdgiCompositeParams>() as u32,
        shader: shader.clone(),
        entry_point: Some("ddgi_copy_base".into()),
        ..Default::default()
    });
    let fold = cache.queue_compute_pipeline(ComputePipelineDescriptor {
        label: Some("prism DDGI composite fold".into()),
        layout: vec![fold_descriptor],
        immediate_size: size_of::<GpuDdgiCompositeParams>() as u32,
        shader,
        entry_point: Some("ddgi_composite".into()),
        ..Default::default()
    });

    commands.insert_resource(DdgiCompositePipeline {
        copy,
        fold,
        copy_layout,
        fold_layout,
    });
}

/// The composite's two per-view bind groups. Present only when the visibility
/// buffer (for `scene_color` and the albedo / ambient exports) and the DDGI
/// targets are resident.
#[derive(Component)]
pub(crate) struct ViewDdgiCompositeBindGroups {
    /// group 0 for `ddgi_copy_base`: `scene_color` read + `gi_base` write.
    copy: BindGroup,
    /// group 0 for `ddgi_composite`: `gi_base` + `gi_out` + albedo + ambient
    /// reads and the write-only `scene_color`.
    fold: BindGroup,
}

/// `PrepareBindGroups` system building [`ViewDdgiCompositeBindGroups`] for every
/// view that has both a visibility buffer and resident DDGI targets.
pub(crate) fn prepare_ddgi_composite_bind_groups(
    mut commands: Commands,
    pipeline: Res<DdgiCompositePipeline>,
    device: Res<RenderDevice>,
    views: Query<(Entity, &ViewVisibilityBuffer, &ViewDdgi)>,
) {
    for (entity, visibility, gi) in &views {
        let copy = device.create_bind_group(
            "prism DDGI composite copy",
            &pipeline.copy_layout,
            &BindGroupEntries::sequential((visibility.scene_color_view(), gi.gi_base_view())),
        );
        let fold = device.create_bind_group(
            "prism DDGI composite fold",
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
            .insert(ViewDdgiCompositeBindGroups { copy, fold });
    }
}

/// `Core3d` scheduling system recording the DDGI composite for every view.
///
/// Runs after `ddgi_sample_pass` (its `gi_out` input) and, being a
/// `scene_color` writer, after the world-space GI composite in the
/// `scene_color`-writer chain, and before the surface-cache gather / TAA /
/// main pass that present `scene_color`. Records the copy then the fold in a
/// single encoder; wgpu inserts the storage barrier between them. Dispatches
/// one workgroup per 8x8 pixel tile; both shader entry points bounds-check
/// every invocation.
pub(crate) fn ddgi_composite_pass(
    settings: Res<PrismDdgiSettings>,
    view: ViewQuery<(&ViewDdgi, &ViewDdgiCompositeBindGroups)>,
    pipeline: Res<DdgiCompositePipeline>,
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

    let size = gi.size();
    if size.x == 0 || size.y == 0 {
        return;
    }

    let params = GpuDdgiCompositeParams::new(size.x, size.y);
    let workgroups_x = size.x.div_ceil(DDGI_WORKGROUP_SIZE);
    let workgroups_y = size.y.div_ceil(DDGI_WORKGROUP_SIZE);

    let encoder = ctx.command_encoder();
    {
        let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("prism DDGI composite copy"),
            timestamp_writes: None,
        });
        pass.set_pipeline(copy);
        pass.set_bind_group(0, &groups.copy, &[]);
        pass.set_immediates(0, bytemuck::bytes_of(&params));
        pass.dispatch_workgroups(workgroups_x, workgroups_y, 1);
    }
    {
        let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("prism DDGI composite fold"),
            timestamp_writes: None,
        });
        pass.set_pipeline(fold);
        pass.set_bind_group(0, &groups.fold, &[]);
        pass.set_immediates(0, bytemuck::bytes_of(&params));
        pass.dispatch_workgroups(workgroups_x, workgroups_y, 1);
    }
}
