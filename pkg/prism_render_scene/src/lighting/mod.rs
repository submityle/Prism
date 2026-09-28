//! GPU light system: extraction, storage buffers, and bind group.
//!
//! This module owns the render-world half of Prism's lighting: it mirrors
//! Bevy's [`bevy_light`] components into the flat, `Pod` [`abi`] records, keeps
//! them in GPU storage buffers, and exposes a read-only bind group the shading
//! resolve pass consumes.  The lighting *math* lives in the CPU golden
//! reference (`prism_render_shading`) and its GPU mirror (the resolve shader);
//! this module only concerns itself with the light *inputs*.

mod abi;
mod bindings;
mod buffers;
mod cluster;
mod extract;
mod plugin;
mod probe;
mod stylized_config;
mod systems;

#[cfg(test)]
mod shader_tests;

pub use abi::{
    GpuDirectionalLight, GpuLightEnvironment, GpuPunctualLight,
    LIGHT_ENVIRONMENT_FLAG_IMAGE_BASED,
};
pub use bindings::LightBindGroup;
pub use cluster::{
    build_cluster_data, ClusterBindGroup, ClusterConfig, ClusterCpuData, ClusterGpuBuffers,
    ClusterViewFit, ExtractedClusterView, GpuClusterGrid,
};
pub use buffers::LightGpuBuffers;
pub use extract::ExtractedLights;
pub use probe::{
    cubemap_faces_from_image, project_image_to_sh, EnvironmentProbeCache,
};
pub use plugin::PrismLightingPlugin;
pub use stylized_config::StylizedLighting;
