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
mod payload;
mod pcm;
mod resources;
mod runtime;
mod selection;
mod topology;
mod types;

pub use accesses::VariantResourceUse;
pub use builtins::{
    builtin_catalog, builtin_roots, builtin_service_ids, volume_required_services,
    water_required_services, VOLUME_SERVICE_IDS, WATER_SERVICE_IDS,
};
pub use catalog::{CatalogError, FeatureCatalog, RequirementKind};
pub use compile::*;
pub use payload::*;
pub use pcm::*;
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
mod runtime_tests;
#[cfg(test)]
mod tests;
