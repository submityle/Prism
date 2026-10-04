//! The world-space `ReSTIR` resolve compute pipeline, its owned group-0 layout,
//! and the `RenderStartup` initializer that queues it.
//!
//! `resolve_main` (`world_restir_resolve.wesl`) runs one invocation per screen
//! pixel. Its group-0 binds the SSR prepass reverse-Z device depth (sampled,
//! 0), the SSR packed view-space `normal_roughness` (sampled, 1), the resident
//! finalised reservoir table (storage read-only, 2) and the write-only
//! `rgba16float` direct-illumination export (storage, 3): each invocation
//! reconstructs the pixel's world shading point, re-hashes it into the resident
//! `SHARC` cell, probes the table and writes the reconnection-shifted direct
//! irradiance + confidence. The per-view matrices, the hash-grid tunables and
//! the framebuffer size arrive in the [`GpuWorldRestirResolveParams`] immediate
//! block, so no uniform buffer is bound.

use bevy_asset::{load_embedded_asset, Handle};
use bevy_ecs::prelude::*;
use bevy_material::{
    bind_group_layout_entries::{
        binding_types::{storage_buffer_read_only_sized, texture_2d, texture_storage_2d},
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
use super::abi::GpuWorldRestirResolveParams;

/// The world-space `ReSTIR` resolve compute pipeline and its owned group-0
/// layout.
#[derive(Resource)]
pub(crate) struct WorldRestirResolvePipeline {
    /// `resolve_main` entry: one invocation per screen pixel.
    pipeline: CachedComputePipelineId,
    /// group 0 for `resolve_main`: the SSR prepass depth (sampled, 0), the SSR
    /// packed `normal_roughness` (sampled, 1), the finalised reservoir table
    /// (storage read-only, 2) and the direct-illumination export (storage
    /// write-only, 3).
    layout: BindGroupLayout,
}

impl WorldRestirResolvePipeline {
    /// The `resolve_main` compute pipeline id.
    pub(crate) fn pipeline(&self) -> CachedComputePipelineId {
        self.pipeline
    }

    /// group-0 layout for the `resolve_main` dispatch.
    pub(crate) fn layout(&self) -> &BindGroupLayout {
        &self.layout
    }
}

/// `resolve_main` group-0 layout: the SSR prepass reverse-Z device depth
/// (sampled, 0) and the SSR packed view-space `normal_roughness` (sampled, 1),
/// both read with `textureLoad` (so non-filterable float is sufficient), the
/// resident finalised reservoir table (storage read-only, 2) and the write-only
/// `rgba16float` direct-illumination export (storage, 3).
fn layout_entries() -> BindGroupLayoutEntries<4> {
    BindGroupLayoutEntries::sequential(
        ShaderStages::COMPUTE,
        (
            texture_2d(TextureSampleType::Float { filterable: false }),
            texture_2d(TextureSampleType::Float { filterable: false }),
            storage_buffer_read_only_sized(false, None),
            texture_storage_2d(SCENE_COLOR_FORMAT, StorageTextureAccess::WriteOnly),
        ),
    )
}

/// `RenderStartup` initializer for [`WorldRestirResolvePipeline`].
pub(crate) fn init_world_restir_resolve_pipeline(
    mut commands: Commands,
    device: Res<RenderDevice>,
    cache: Res<PipelineCache>,
    asset_server: Res<bevy_asset::AssetServer>,
) {
    let entries = layout_entries();
    let descriptor = BindGroupLayoutDescriptor::new("prism world-space ReSTIR resolve", &entries);
    let layout = device.create_bind_group_layout("prism world-space ReSTIR resolve", &entries);

    let shader: Handle<Shader> = load_embedded_asset!(
        asset_server.as_ref(),
        "../../../shaders/world_restir_resolve.wesl"
    );

    let pipeline = cache.queue_compute_pipeline(ComputePipelineDescriptor {
        label: Some("prism world-space ReSTIR resolve".into()),
        layout: vec![descriptor],
        immediate_size: size_of::<GpuWorldRestirResolveParams>() as u32,
        shader,
        entry_point: Some("resolve_main".into()),
        ..Default::default()
    });

    commands.insert_resource(WorldRestirResolvePipeline { pipeline, layout });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolve_immediate_block_is_the_abi_size() {
        // The pipeline reserves exactly the frozen resolve immediate block; a
        // drift here would mismatch `set_immediates` against the shader's
        // `var<immediate>` block.
        assert_eq!(size_of::<GpuWorldRestirResolveParams>() as u32, 192);
    }
}
