//! Surface-cache composite: the copy + fold pipelines, per-view bind groups,
//! and the `Core3d` dispatch that folds the gathered surfel radiance cache back
//! over the shaded scene.
//!
//! The coverage pass (`surface_cache_coverage.wesl`) leaves `rgb` = the
//! pre-albedo diffuse `GI` irradiance (the golden coverage gather, times the
//! artistic intensity) and `a` = a `[0, 1]` coverage confidence in `gi_out`.
//! This is the *same* quantity the shading resolve exports at `ssgi_ambient`
//! (pre-albedo `IBL` / `SH` irradiance), so this stage performs the *same*
//! energy-conserving substitution world-space `GI` does, swapping the flat
//! `IBL` ambient the shading resolve already folded into `scene_color` for the
//! surfel gather under its confidence:
//!
//! ```text
//! scene = base + confidence * albedo * (gi_out - ambient)
//! ```
//!
//! A fully confident pixel (`a == 1`) replaces the flat ambient with the
//! cached indirect irradiance; a miss (`a == 0`) leaves the shaded colour
//! untouched, preserving the `IBL` ambient.
//!
//! `scene_color` is an `rgba16float` storage image, which is not read-write
//! storage-capable, so the pass runs two passes in one encoder (wgpu inserts
//! the barrier between them):
//!
//! * a **copy** pass lifts the shading-resolved `scene_color` into a scratch
//!   `gi_base` texture (`textureLoad(scene_color) -> gi_base`), and
//! * a **fold** pass reads that base plus the surfel gather and the resolve's
//!   albedo / ambient exports and writes the substitution into `scene_color`.
//!
//! Reading the base from a distinct texture and writing `scene_color` keeps the
//! pass free of any storage read/write aliasing hazard. It must run after
//! `surface_cache_pass` (its `gi_out` input) and, being a `scene_color`
//! writer, is sequenced into the `scene_color`-writer chain after the
//! world-space `GI` composite.

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
use super::abi::{GpuSurfaceCacheCompositeParams, SURFACE_CACHE_WORKGROUP_SIZE_2D};
use super::resources::ViewSurfaceCache;
use super::settings::PrismSurfaceCacheSettings;

/// The two compute pipelines and their owned group-0 layouts for the
/// surface-cache composite: the `scene_color` -> `gi_base` copy and the
/// energy-conserving `GI` fold, both entry points of
/// `shaders/surface_cache_composite.wesl`.
#[derive(Resource)]
pub(crate) struct SurfaceCacheCompositePipeline {
    /// `sc_copy_base` entry point: lifts `scene_color` into `gi_base`.
    copy: CachedComputePipelineId,
    /// `sc_composite` entry point: folds the gather into `scene_color`.
    fold: CachedComputePipelineId,
    /// copy layout: `scene_color` read + `gi_base` write.
    copy_layout: BindGroupLayout,
    /// fold layout: `gi_base` + `gi_out` + albedo + ambient reads,
    /// `scene_color` written in place.
    fold_layout: BindGroupLayout,
}

/// copy-pass layout mirroring `surface_cache_composite.wesl`'s `sc_copy_base`:
/// one non-filterable float read (the shading-resolved `scene_color`) then the
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

/// fold-pass layout mirroring `surface_cache_composite.wesl`'s `sc_composite`:
/// the `gi_base` copy and the `gi_out` gather (both `textureLoad`ed), the
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

/// `RenderStartup` initializer for [`SurfaceCacheCompositePipeline`].
pub(crate) fn init_surface_cache_composite_pipeline(
    mut commands: Commands,
    device: Res<RenderDevice>,
    cache: Res<PipelineCache>,
    asset_server: Res<bevy_asset::AssetServer>,
) {
    let copy_entries = copy_layout_entries();
    let copy_descriptor =
        BindGroupLayoutDescriptor::new("prism surface cache composite copy", &copy_entries);
    let copy_layout =
        device.create_bind_group_layout("prism surface cache composite copy", &copy_entries);

    let fold_entries = fold_layout_entries();
    let fold_descriptor =
        BindGroupLayoutDescriptor::new("prism surface cache composite fold", &fold_entries);
    let fold_layout =
        device.create_bind_group_layout("prism surface cache composite fold", &fold_entries);

    let shader: Handle<Shader> = load_embedded_asset!(
        asset_server.as_ref(),
        "../shaders/surface_cache_composite.wesl"
    );

    let copy = cache.queue_compute_pipeline(ComputePipelineDescriptor {
        label: Some("prism surface cache composite copy".into()),
        layout: vec![copy_descriptor],
        immediate_size: size_of::<GpuSurfaceCacheCompositeParams>() as u32,
        shader: shader.clone(),
        entry_point: Some("sc_copy_base".into()),
        ..Default::default()
    });
    let fold = cache.queue_compute_pipeline(ComputePipelineDescriptor {
        label: Some("prism surface cache composite fold".into()),
        layout: vec![fold_descriptor],
        immediate_size: size_of::<GpuSurfaceCacheCompositeParams>() as u32,
        shader,
        entry_point: Some("sc_composite".into()),
        ..Default::default()
    });

    commands.insert_resource(SurfaceCacheCompositePipeline {
        copy,
        fold,
        copy_layout,
        fold_layout,
    });
}

/// The composite's two per-view bind groups. Present only when the visibility
/// buffer (for `scene_color` and the albedo / ambient exports) and the
/// surface-cache targets are resident.
#[derive(Component)]
pub(crate) struct ViewSurfaceCacheCompositeBindGroups {
    /// group 0 for `sc_copy_base`: `scene_color` read + `gi_base` write.
    copy: BindGroup,
    /// group 0 for `sc_composite`: `gi_base` + `gi_out` + albedo + ambient
    /// reads and the write-only `scene_color`.
    fold: BindGroup,
}

/// `PrepareBindGroups` system building [`ViewSurfaceCacheCompositeBindGroups`]
/// for every view that has both a visibility buffer and resident surface-cache
/// targets.
pub(crate) fn prepare_surface_cache_composite_bind_groups(
    mut commands: Commands,
    pipeline: Res<SurfaceCacheCompositePipeline>,
    device: Res<RenderDevice>,
    views: Query<(Entity, &ViewVisibilityBuffer, &ViewSurfaceCache)>,
) {
    for (entity, visibility, sc) in &views {
        let copy = device.create_bind_group(
            "prism surface cache composite copy",
            &pipeline.copy_layout,
            &BindGroupEntries::sequential((visibility.scene_color_view(), sc.gi_base_view())),
        );
        let fold = device.create_bind_group(
            "prism surface cache composite fold",
            &pipeline.fold_layout,
            &BindGroupEntries::sequential((
                sc.gi_base_view(),
                sc.gi_out_view(),
                visibility.scene_color_view(),
                visibility.ssgi_albedo_view(),
                visibility.ssgi_ambient_view(),
            )),
        );
        commands
            .entity(entity)
            .insert(ViewSurfaceCacheCompositeBindGroups { copy, fold });
    }
}

/// `Core3d` scheduling system recording the surface-cache composite for every
/// view.
///
/// Runs after `surface_cache_pass` (its `gi_out` input) and, being a
/// `scene_color` writer, after the world-space `GI` composite in the
/// `scene_color`-writer chain, and before the main pass that presents
/// `scene_color`. Records the copy then the fold in a single encoder; wgpu
/// inserts the storage barrier between them. Dispatches one workgroup per 8x8
/// pixel tile; both shader entry points bounds-check every invocation.
pub(crate) fn surface_cache_composite_pass(
    settings: Res<PrismSurfaceCacheSettings>,
    view: ViewQuery<(&ViewSurfaceCache, &ViewSurfaceCacheCompositeBindGroups)>,
    pipeline: Res<SurfaceCacheCompositePipeline>,
    cache: Res<PipelineCache>,
    mut ctx: RenderContext,
) {
    if !settings.enabled {
        return;
    }
    let (sc, groups) = view.into_inner();

    let (Some(copy), Some(fold)) = (
        cache.get_compute_pipeline(pipeline.copy),
        cache.get_compute_pipeline(pipeline.fold),
    ) else {
        return;
    };

    let size = sc.size;
    if size.x == 0 || size.y == 0 {
        return;
    }

    let params = GpuSurfaceCacheCompositeParams::new(size.x, size.y);
    let workgroups_x = size.x.div_ceil(SURFACE_CACHE_WORKGROUP_SIZE_2D);
    let workgroups_y = size.y.div_ceil(SURFACE_CACHE_WORKGROUP_SIZE_2D);

    let encoder = ctx.command_encoder();
    {
        let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("prism surface cache composite copy"),
            timestamp_writes: None,
        });
        pass.set_pipeline(copy);
        pass.set_bind_group(0, &groups.copy, &[]);
        pass.set_immediates(0, bytemuck::bytes_of(&params));
        pass.dispatch_workgroups(workgroups_x, workgroups_y, 1);
    }
    {
        let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("prism surface cache composite fold"),
            timestamp_writes: None,
        });
        pass.set_pipeline(fold);
        pass.set_bind_group(0, &groups.fold, &[]);
        pass.set_immediates(0, bytemuck::bytes_of(&params));
        pass.dispatch_workgroups(workgroups_x, workgroups_y, 1);
    }
}
