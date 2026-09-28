//! Compute pipeline + bind-group layout for the DFG lookup-table precompute.
//!
//! The kernel (`shaders/brdf_lut.wesl`) reads no scene state: it derives
//! `n_dot_v` / `roughness` from the storage texture's dimensions and writes the
//! integrated `(scale, bias)` pair, so a single bind group holding the
//! write-only `Rg16Float` storage texture is all it needs.  The GGX sample
//! count arrives in the 16-byte immediate [`GpuBrdfLutConfig`] block.

use bevy_asset::{load_embedded_asset, Handle};
use bevy_ecs::prelude::*;
use bevy_material::{
    bind_group_layout_entries::{binding_types::texture_storage_2d, BindGroupLayoutEntries},
    descriptor::BindGroupLayoutDescriptor,
};
use bevy_render::{
    render_resource::{
        BindGroupLayout, CachedComputePipelineId, ComputePipelineDescriptor, PipelineCache,
        ShaderStages, StorageTextureAccess,
    },
    renderer::RenderDevice,
};
use bevy_shader::Shader;

use super::abi::GpuBrdfLutConfig;
use super::resources::DFG_LUT_FORMAT;

/// Compute pipeline and the single owned bind-group layout for the DFG table
/// precompute.
#[derive(Resource)]
pub(crate) struct BrdfLutPipeline {
    /// `integrate_brdf_lut` compute entry point, specialized against
    /// [`Self::layout`] and the 16-byte immediate config block.
    pub(crate) pipeline: CachedComputePipelineId,
    /// group 0: the write-only `Rg16Float` DFG storage texture.
    pub(crate) layout: BindGroupLayout,
}

/// group-0 layout: one write-only `rg16float` storage texture holding the
/// `(scale, bias)` split-sum pair.
fn layout_entries() -> BindGroupLayoutEntries<1> {
    BindGroupLayoutEntries::sequential(
        ShaderStages::COMPUTE,
        (texture_storage_2d(DFG_LUT_FORMAT, StorageTextureAccess::WriteOnly),),
    )
}

/// `RenderStartup` initializer for [`BrdfLutPipeline`].
pub(crate) fn init_brdf_lut_pipeline(
    mut commands: Commands,
    device: Res<RenderDevice>,
    cache: Res<PipelineCache>,
    asset_server: Res<bevy_asset::AssetServer>,
) {
    let entries = layout_entries();
    let descriptor = BindGroupLayoutDescriptor::new("prism DFG LUT", &entries);
    let layout = device.create_bind_group_layout("prism DFG LUT", &entries);

    let shader: Handle<Shader> =
        load_embedded_asset!(asset_server.as_ref(), "../shaders/brdf_lut.wesl");

    let pipeline = cache.queue_compute_pipeline(ComputePipelineDescriptor {
        label: Some("prism DFG LUT".into()),
        layout: vec![descriptor],
        immediate_size: size_of::<GpuBrdfLutConfig>() as u32,
        shader,
        entry_point: Some("integrate_brdf_lut".into()),
        ..Default::default()
    });

    commands.insert_resource(BrdfLutPipeline { pipeline, layout });
}
