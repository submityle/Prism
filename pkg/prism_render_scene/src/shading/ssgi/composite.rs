//! SSGI composite: the copy + fold pipelines, per-view bind groups, and the
//! `Core3d` dispatch node that folds the traced indirect-diffuse buffer back
//! over the shaded scene.
//!
//! The gather (`ssgi.wesl`) leaves `rgb` = the pre-albedo mean indirect
//! radiance and `a` = a `[0, 1]` blend confidence in its output. This stage
//! substitutes that gather for the flat IBL/SH ambient the resolve already
//! folded into `scene_color`, the GPU twin of the composite step in
//! [`prism_render_shading::screen_space::gi`], computing
//! `scene = base + confidence * albedo * (ssgi_out - ambient)`.
//!
//! `scene_color` is an `rgba16float` storage image, which is not read-write
//! storage-capable, so the node runs in two passes in one encoder (wgpu inserts
//! the barrier between them):
//!
//! * a **copy** pass lifts the SSR-composited `scene_color` into a scratch
//!   `gi_base` texture (`textureLoad(scene_color) -> gi_base`), and
//! * a **fold** pass reads that base plus the gather and the resolve's albedo /
//!   ambient exports and writes the substitution into `scene_color`.
//!
//! Reading the base from a distinct texture and writing `scene_color` keeps the
//! pass free of any read/write aliasing hazard. Running after the SSR composite
//! means the base already carries any screen-space reflection, so the two
//! composites do not clobber each other.

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
use super::abi::{GpuSsgiCompositeParams, SSGI_WORKGROUP_SIZE};
use super::resources::{ViewSsgiTextures, SSGI_BASE_FORMAT};

/// The two compute pipelines and their owned layouts for the SSGI composite:
/// the `scene_color` -> `gi_base` copy and the energy-conserving GI fold, both
/// entry points of `shaders/ssgi_composite.wesl`.
#[derive(Resource)]
pub(crate) struct SsgiCompositePipeline {
    /// `ssgi_copy_base` entry point: lifts `scene_color` into `gi_base`.
    copy: CachedComputePipelineId,
    /// `ssgi_composite` entry point: folds the gather into `scene_color`.
    fold: CachedComputePipelineId,
    /// copy layout: `scene_color` read + `gi_base` write.
    copy_layout: BindGroupLayout,
    /// fold layout: `gi_base` + gather + albedo + ambient reads, `scene_color`
    /// written in place.
    fold_layout: BindGroupLayout,
}

/// copy-pass layout mirroring `ssgi_composite.wesl`'s `ssgi_copy_base`: one
/// non-filterable float read (the composited `scene_color`) then the write-only
/// `gi_base` scratch.
fn copy_layout_entries() -> BindGroupLayoutEntries<2> {
    BindGroupLayoutEntries::sequential(
        ShaderStages::COMPUTE,
        (
            texture_2d(TextureSampleType::Float { filterable: false }),
            texture_storage_2d(SSGI_BASE_FORMAT, StorageTextureAccess::WriteOnly),
        ),
    )
}

/// fold-pass layout mirroring `ssgi_composite.wesl`'s `ssgi_composite`: the
/// `gi_base` copy and the gather output (both `textureLoad`ed), the write-only
/// `scene_color`, then the resolve's Lambertian albedo and pre-albedo ambient
/// exports.
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

/// `RenderStartup` initializer for [`SsgiCompositePipeline`].
pub(crate) fn init_ssgi_composite_pipeline(
    mut commands: Commands,
    device: Res<RenderDevice>,
    cache: Res<PipelineCache>,
    asset_server: Res<bevy_asset::AssetServer>,
) {
    let copy_entries = copy_layout_entries();
    let copy_descriptor = BindGroupLayoutDescriptor::new("prism SSGI composite copy", &copy_entries);
    let copy_layout = device.create_bind_group_layout("prism SSGI composite copy", &copy_entries);

    let fold_entries = fold_layout_entries();
    let fold_descriptor = BindGroupLayoutDescriptor::new("prism SSGI composite fold", &fold_entries);
    let fold_layout = device.create_bind_group_layout("prism SSGI composite fold", &fold_entries);

    let shader: Handle<Shader> =
        load_embedded_asset!(asset_server.as_ref(), "../shaders/ssgi_composite.wesl");

    let copy = cache.queue_compute_pipeline(ComputePipelineDescriptor {
        label: Some("prism SSGI composite copy".into()),
        layout: vec![copy_descriptor],
        immediate_size: size_of::<GpuSsgiCompositeParams>() as u32,
        shader: shader.clone(),
        entry_point: Some("ssgi_copy_base".into()),
        ..Default::default()
    });
    let fold = cache.queue_compute_pipeline(ComputePipelineDescriptor {
        label: Some("prism SSGI composite fold".into()),
        layout: vec![fold_descriptor],
        immediate_size: size_of::<GpuSsgiCompositeParams>() as u32,
        shader,
        entry_point: Some("ssgi_composite".into()),
        ..Default::default()
    });

    commands.insert_resource(SsgiCompositePipeline {
        copy,
        fold,
        copy_layout,
        fold_layout,
    });
}

/// The composite's two per-view bind groups. Present only when the visibility
/// buffer (for `scene_color` and the albedo/ambient exports) and the SSGI
/// targets are resident.
#[derive(Component)]
pub(crate) struct ViewSsgiCompositeBindGroups {
    copy: BindGroup,
    fold: BindGroup,
}

/// `PrepareBindGroups` system building [`ViewSsgiCompositeBindGroups`] for every
/// view that has both a visibility buffer and resident SSGI targets.
pub(crate) fn prepare_ssgi_composite_bind_groups(
    mut commands: Commands,
    pipeline: Res<SsgiCompositePipeline>,
    device: Res<RenderDevice>,
    views: Query<(Entity, &ViewVisibilityBuffer, &ViewSsgiTextures)>,
) {
    for (entity, visibility, textures) in &views {
        let copy = device.create_bind_group(
            "prism SSGI composite copy",
            &pipeline.copy_layout,
            &BindGroupEntries::sequential((
                visibility.scene_color_view(),
                textures.gi_base_view(),
            )),
        );
        let fold = device.create_bind_group(
            "prism SSGI composite fold",
            &pipeline.fold_layout,
            &BindGroupEntries::sequential((
                textures.gi_base_view(),
                textures.ssgi_out_view(),
                visibility.scene_color_view(),
                visibility.ssgi_albedo_view(),
                visibility.ssgi_ambient_view(),
            )),
        );
        commands
            .entity(entity)
            .insert(ViewSsgiCompositeBindGroups { copy, fold });
    }
}

/// `Core3d` node recording the SSGI composite for every view.
///
/// Runs after the SSR composite (so the base already carries any reflection)
/// and the SSGI trace (its gather input), and before the main pass that
/// presents `scene_color`. Records the copy then the fold in a single encoder;
/// wgpu inserts the storage barrier between them. Dispatches one workgroup per
/// 8x8 pixel tile; both shader entry points bounds-check every invocation.
pub(crate) fn ssgi_composite_pass(
    settings: Res<super::super::runtime::PrismShadingSettings>,
    view: ViewQuery<(&ViewSsgiTextures, &ViewSsgiCompositeBindGroups)>,
    pipeline: Res<SsgiCompositePipeline>,
    cache: Res<PipelineCache>,
    mut ctx: RenderContext,
) {
    if !settings.enable_ssgi {
        return;
    }
    let (textures, groups) = view.into_inner();

    let (Some(copy), Some(fold)) = (
        cache.get_compute_pipeline(pipeline.copy),
        cache.get_compute_pipeline(pipeline.fold),
    ) else {
        return;
    };

    let size = textures.size;
    if size.x == 0 || size.y == 0 {
        return;
    }

    let params = GpuSsgiCompositeParams::new(size.x, size.y);
    let workgroups_x = size.x.div_ceil(SSGI_WORKGROUP_SIZE);
    let workgroups_y = size.y.div_ceil(SSGI_WORKGROUP_SIZE);

    let encoder = ctx.command_encoder();
    {
        let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("prism SSGI composite copy"),
            timestamp_writes: None,
        });
        pass.set_pipeline(copy);
        pass.set_bind_group(0, &groups.copy, &[]);
        pass.set_immediates(0, bytemuck::bytes_of(&params));
        pass.dispatch_workgroups(workgroups_x, workgroups_y, 1);
    }
    {
        let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("prism SSGI composite fold"),
            timestamp_writes: None,
        });
        pass.set_pipeline(fold);
        pass.set_bind_group(0, &groups.fold, &[]);
        pass.set_immediates(0, bytemuck::bytes_of(&params));
        pass.dispatch_workgroups(workgroups_x, workgroups_y, 1);
    }
}
