//! The world-space `ReSTIR` composite's two compute pipelines, their owned
//! group-0 layouts, and the `RenderStartup` initializer that queues them.
//!
//! `scene_color` is an `rgba16float` storage image, which is not read-write
//! storage-capable, so the composite runs two passes in one encoder (wgpu
//! inserts the barrier between them):
//!
//! * `wr_copy_base` lifts the shading-resolved `scene_color` into a scratch
//!   `gi_base` texture (`textureLoad(scene_color) -> gi_base`), and
//! * `wr_composite` reads that base plus the resolve's `gi_out` export, the
//!   Lambertian albedo, and the clustered punctual direct export, then writes
//!   the energy-conserving direct substitution into `scene_color`.
//!
//! Both entry points live in `shaders/world_restir_composite.wesl`.

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
        BindGroupLayout, CachedComputePipelineId, ComputePipelineDescriptor, PipelineCache,
        ShaderStages, StorageTextureAccess, TextureSampleType,
    },
    renderer::RenderDevice,
};
use bevy_shader::Shader;

use super::super::super::resources::SCENE_COLOR_FORMAT;
use super::abi::GpuWorldRestirCompositeParams;

/// The two compute pipelines and their owned group-0 layouts for the
/// world-space `ReSTIR` composite: the `scene_color` -> `gi_base` copy and the
/// energy-conserving direct fold, both entry points of
/// `shaders/world_restir_composite.wesl`.
#[derive(Resource)]
pub(crate) struct WorldRestirCompositePipeline {
    /// `wr_copy_base` entry point: lifts `scene_color` into `gi_base`.
    copy: CachedComputePipelineId,
    /// `wr_composite` entry point: folds the `ReSTIR` estimate into
    /// `scene_color`.
    fold: CachedComputePipelineId,
    /// copy layout: `scene_color` read + `gi_base` write.
    copy_layout: BindGroupLayout,
    /// fold layout: `gi_base` + `gi_out` reads, `scene_color` write, then the
    /// Lambertian albedo + clustered punctual direct exports.
    fold_layout: BindGroupLayout,
}

impl WorldRestirCompositePipeline {
    /// The `wr_copy_base` compute pipeline id.
    pub(crate) fn copy(&self) -> CachedComputePipelineId {
        self.copy
    }

    /// The `wr_composite` compute pipeline id.
    pub(crate) fn fold(&self) -> CachedComputePipelineId {
        self.fold
    }

    /// group-0 layout for the `wr_copy_base` dispatch.
    pub(crate) fn copy_layout(&self) -> &BindGroupLayout {
        &self.copy_layout
    }

    /// group-0 layout for the `wr_composite` dispatch.
    pub(crate) fn fold_layout(&self) -> &BindGroupLayout {
        &self.fold_layout
    }
}

/// copy-pass layout mirroring `world_restir_composite.wesl`'s `wr_copy_base`:
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

/// fold-pass layout mirroring `world_restir_composite.wesl`'s `wr_composite`:
/// the `gi_base` copy and the `gi_out` export (both `textureLoad`ed), the
/// write-only `scene_color`, then the resolve's Lambertian albedo and the
/// shading resolve's clustered punctual direct export.
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

/// `RenderStartup` initializer for [`WorldRestirCompositePipeline`].
pub(crate) fn init_world_restir_composite_pipeline(
    mut commands: Commands,
    device: Res<RenderDevice>,
    cache: Res<PipelineCache>,
    asset_server: Res<bevy_asset::AssetServer>,
) {
    let copy_entries = copy_layout_entries();
    let copy_descriptor =
        BindGroupLayoutDescriptor::new("prism world-space ReSTIR composite copy", &copy_entries);
    let copy_layout =
        device.create_bind_group_layout("prism world-space ReSTIR composite copy", &copy_entries);

    let fold_entries = fold_layout_entries();
    let fold_descriptor =
        BindGroupLayoutDescriptor::new("prism world-space ReSTIR composite fold", &fold_entries);
    let fold_layout =
        device.create_bind_group_layout("prism world-space ReSTIR composite fold", &fold_entries);

    let shader: Handle<Shader> = load_embedded_asset!(
        asset_server.as_ref(),
        "../../../shaders/world_restir_composite.wesl"
    );

    let copy = cache.queue_compute_pipeline(ComputePipelineDescriptor {
        label: Some("prism world-space ReSTIR composite copy".into()),
        layout: vec![copy_descriptor],
        immediate_size: size_of::<GpuWorldRestirCompositeParams>() as u32,
        shader: shader.clone(),
        entry_point: Some("wr_copy_base".into()),
        ..Default::default()
    });
    let fold = cache.queue_compute_pipeline(ComputePipelineDescriptor {
        label: Some("prism world-space ReSTIR composite fold".into()),
        layout: vec![fold_descriptor],
        immediate_size: size_of::<GpuWorldRestirCompositeParams>() as u32,
        shader,
        entry_point: Some("wr_composite".into()),
        ..Default::default()
    });

    commands.insert_resource(WorldRestirCompositePipeline {
        copy,
        fold,
        copy_layout,
        fold_layout,
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn composite_immediate_block_is_the_abi_size() {
        // Both pipelines reserve exactly the composite immediate block; a drift
        // here would mismatch `set_immediates` against the shader's
        // `var<immediate>` block.
        assert_eq!(size_of::<GpuWorldRestirCompositeParams>() as u32, 16);
    }
}
