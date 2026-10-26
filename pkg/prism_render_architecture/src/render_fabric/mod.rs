//! Static render-feature contracts and deterministic device-local selection.
//!
//! The catalog owns semantic declarations, not frame resources or material
//! parameters. Hard dependencies alone determine the closure. Ordering delegates
//! to the existing frame-graph compiler; this module is not a graph compiler.
//! Consumers read one frozen [`FeatureSelection`], rather than selecting again.

mod accesses;
mod builtins;
mod catalog;
mod compile;
mod decal;
mod device;
mod frame;
mod history;
mod instances;
mod payload;
mod pcm;
mod pipeline;
mod resources;
mod runtime;
mod selection;
mod topology;
mod types;

pub use accesses::VariantResourceUse;
pub use builtins::{
    builtin_catalog, builtin_resource_templates, builtin_roots, builtin_service_ids,
    volume_required_services, water_required_services, BUILTIN_RESOURCE_TEMPLATES,
    CUSTOM_FEATURE_ID, CUSTOM_VARIANT_ID, DECAL_CLUSTER_INDEX_RESOURCE,
    DECAL_CLUSTER_RANGE_RESOURCE, DECAL_FEATURE_ID, DECAL_INSTANCE_RESOURCE,
    DECAL_MATERIAL_RESOURCE, DECAL_RECEIVER_RESOURCE, DECAL_RESOLVE_PASS, DECAL_RESOLVE_RESOURCE,
    HAIR_DEEP_OPACITY_RESOURCE, HAIR_FEATURE_ID, HAIR_STRAND_RESOURCE, HAIR_VISIBILITY_RESOURCE,
    LIT_FEATURE_ID, LIT_VARIANT_ID, NPR_FEATURE_ID, NPR_VARIANT_ID, PARTICLE_ALIVE_RESOURCE,
    PARTICLE_BRIDGE_PASS, PARTICLE_CLUSTERED_LIGHTS, PARTICLE_FEATURE_ID,
    PARTICLE_GLOBAL_ILLUMINATION, PARTICLE_LIGHTING_CAPABILITIES, PARTICLE_LIGHTING_SERVICE_IDS,
    PARTICLE_RAY_TRACING, PARTICLE_RENDER_PASS, PARTICLE_SHADOW_MAPS, PARTICLE_SPRAY_RESOURCE,
    PARTICLE_STREAM_RESOURCE, SPRAY_FEEDBACK_FEATURE_ID, SPRAY_FEEDBACK_SERVICE,
    TAA_HISTORY_RESOURCE, VOLUME_APPLY_PASS, VOLUME_COPY_BACK_PASS, VOLUME_DENSITY_RESOURCE,
    VOLUME_EXTINCTION_RESOURCE, VOLUME_FEATURE_ID, VOLUME_FOG_APPLIED_RESOURCE,
    VOLUME_FROXEL_RESOURCE, VOLUME_INTEGRATED_SCATTERING_RESOURCE,
    VOLUME_INTEGRATED_TRANSMITTANCE_RESOURCE, VOLUME_INTEGRATE_PASS, VOLUME_SCATTERING_RESOURCE,
    VOLUME_SCATTER_PASS, VOLUME_SCENE_COLOR_RESOURCE, VOLUME_SCENE_DEPTH_RESOURCE,
    VOLUME_SERVICE_IDS, VOLUME_VELOCITY_RESOURCE, WATER_FEATURE_ID, WATER_FOAM_RESOURCE,
    WATER_SERVICE_IDS, WATER_SURFACE_RESOURCE, WATER_VARIANT_ID,
};
pub use catalog::{CatalogError, FeatureCatalog, RequirementKind, ServiceProvider};
pub use compile::*;
pub use decal::*;
pub use device::*;
pub use frame::{RenderFabricFrame, RenderFabricFrameError};
pub use history::*;
pub use instances::*;
pub use payload::*;
pub use pcm::*;
pub use pipeline::*;
pub use prism_render_graph::frame_graph::ResourceLifetime;
pub use resources::*;
pub use runtime::*;
pub use selection::{
    CandidateRejection, FeatureSelection, FrozenRoots, RootError, SelectedFeature, SelectionError,
};
pub use types::*;

#[cfg(test)]
mod alternatives_tests;
#[cfg(test)]
mod compile_tests;
#[cfg(test)]
mod pcm_tests;
#[cfg(test)]
mod pipeline_tests;
#[cfg(test)]
mod runtime_tests;
#[cfg(test)]
mod tests;
