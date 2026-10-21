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
mod device;
mod frame;
mod history;
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
    CUSTOM_FEATURE_ID, CUSTOM_VARIANT_ID, DECAL_FEATURE_ID, DECAL_INDIRECT_RESOURCE,
    DECAL_INSTANCE_RESOURCE, DECAL_RESOLVE_RESOURCE, HAIR_DEEP_OPACITY_RESOURCE, HAIR_FEATURE_ID,
    HAIR_STRAND_RESOURCE, HAIR_VISIBILITY_RESOURCE, LIT_FEATURE_ID, LIT_VARIANT_ID, NPR_FEATURE_ID,
    NPR_VARIANT_ID, PARTICLE_ALIVE_RESOURCE, PARTICLE_CLUSTERED_LIGHTS, PARTICLE_FEATURE_ID,
    PARTICLE_GLOBAL_ILLUMINATION, PARTICLE_LIGHTING_CAPABILITIES, PARTICLE_LIGHTING_SERVICE_IDS,
    PARTICLE_RAY_TRACING, PARTICLE_SHADOW_MAPS, PARTICLE_SPRAY_RESOURCE, PARTICLE_STREAM_RESOURCE,
    SPRAY_FEEDBACK_FEATURE_ID, SPRAY_FEEDBACK_SERVICE, VOLUME_DENSITY_RESOURCE, VOLUME_FEATURE_ID,
    VOLUME_FROXEL_RESOURCE, VOLUME_SERVICE_IDS, VOLUME_VELOCITY_RESOURCE, WATER_FEATURE_ID,
    WATER_FOAM_RESOURCE, WATER_SERVICE_IDS, WATER_SPRAY_RESOURCE, WATER_SURFACE_RESOURCE,
    WATER_VARIANT_ID,
};
pub use catalog::{CatalogError, FeatureCatalog, RequirementKind, ServiceProvider};
pub use compile::*;
pub use device::*;
pub use frame::{RenderFabricFrame, RenderFabricFrameError};
pub use history::*;
pub use payload::*;
pub use pcm::*;
pub use pipeline::*;
pub use resources::*;
pub use runtime::*;
pub use selection::{
    CandidateRejection, FeatureSelection, FrozenRoots, RootError, SelectedFeature, SelectionError,
};
pub use types::*;

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
