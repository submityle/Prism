//! Frozen selection for the Bevy device that actually owns scene buffers.
//! This is not the independently created Prism RHI device in `PrismGpu`.
use bevy_ecs::prelude::*;
use bevy_render::renderer::RenderDevice;
use prism_render_architecture::{
    backend::capability::{Capability, CapabilitySet},
    render_fabric::{builtin_catalog, builtin_roots, DeviceFabric, RenderQualityTier},
};

#[derive(Resource)]
pub(crate) struct SceneFabric(pub(crate) DeviceFabric);

pub(crate) fn init_scene_fabric(mut commands: Commands, device: Res<RenderDevice>) {
    let mut caps = CapabilitySet::WEBGPU_BASELINE;
    let features = device.features();
    for (feature, capability) in [
        (
            wgpu_types::Features::MULTI_DRAW_INDIRECT_COUNT,
            Capability::IndirectDrawCount,
        ),
        (
            wgpu_types::Features::EXPERIMENTAL_MESH_SHADER,
            Capability::MeshShading,
        ),
        (
            wgpu_types::Features::EXPERIMENTAL_RAY_QUERY,
            Capability::RayQuery,
        ),
        (
            wgpu_types::Features::EXPERIMENTAL_RAY_TRACING_PIPELINES,
            Capability::RayTracingPipeline,
        ),
    ] {
        if features.contains(feature) {
            caps = caps.union(CapabilitySet::from_slice(&[capability]));
        }
    }
    if features.contains(
        wgpu_types::Features::TEXTURE_BINDING_ARRAY
            | wgpu_types::Features::SAMPLED_TEXTURE_AND_STORAGE_BUFFER_ARRAY_NON_UNIFORM_INDEXING,
    ) {
        caps = caps.union(CapabilitySet::from_slice(&[
            Capability::BindlessDescriptors,
        ]));
    }
    // The build profile is static, never derived from extracted/visible bodies.
    let catalog = builtin_catalog().expect("scene feature declarations");
    let roots = builtin_roots(&catalog).expect("scene build-profile roots");
    let fabric = DeviceFabric::new(catalog, roots, caps, RenderQualityTier::High)
        .expect("device cannot satisfy scene build-profile features");
    commands.insert_resource(SceneFabric(fabric));
}
