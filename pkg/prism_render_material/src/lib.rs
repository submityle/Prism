//! Versioned material IR and runtime ABI shared by every Prism renderer.

#![expect(missing_docs, reason = "The ABI is documented as it freezes.")]

extern crate alloc;

#[cfg(feature = "bevy")]
mod bevy_bridge;
mod handle;
mod ir;
mod record;
mod registry;
mod resources;
mod validation;

#[cfg(feature = "bevy")]
pub use bevy_bridge::{lower_standard_material, StandardMaterialTextureResolver, TextureSemantic};
pub use handle::{MaterialCapacityError, MaterialHandleAllocator, MaterialHandleError};
pub use ir::{
    ClosureKind, MaterialGraph, MaterialNode, MaterialNodeId, MaterialValue, NormalizedMaterial,
};
pub use record::{
    GpuMaterialHeader, GpuMaterialTexture, GpuSurfaceParameters, MaterialDomain,
    MaterialFeatureFlags, MaterialRecord, MaterialRenderClass, MaterialShadingModel,
    MATERIAL_ABI_VERSION,
};
pub use registry::{MaterialRegistry, MaterialRegistryError, MaterialSnapshot};
pub use resources::{MaterialResourceHandle, MaterialResourceKind, MaterialResourceTable};
pub use validation::{validate_graph, MaterialValidationError};

#[cfg(test)]
mod tests;
