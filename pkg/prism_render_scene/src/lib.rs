//! Bevy integration for Prism's retained GPU Scene.

#![expect(
    missing_docs,
    reason = "The experimental integration API is still evolving."
)]

extern crate alloc;

mod buffers;
mod compare;
mod completion;
mod consumer;
mod diagnostics;
mod extract;
mod geometry;
mod material;
mod opaque;
mod plugin;
mod scene;
mod shading;
mod visibility;

pub use buffers::{
    GpuSceneBuffers, RenderGpuSceneBounds, RenderGpuSceneInstance, RenderGpuSceneTransform,
};
pub use compare::GpuSceneParityDiagnostics;
pub use completion::GpuCompletionTracker;
pub use consumer::{GpuSceneBufferBindings, GpuSceneReader};
pub use diagnostics::{GpuSceneDiagnostics, GpuSceneUploadSettings};
pub use extract::{ExtractedSceneInstance, GpuSceneInstanceAddress, PrismGpuSceneEntity};
pub use geometry::{
    build_shading_geometry, GeometryBindGroup, RenderGeometryRegistry, RenderShadingGeometry,
    RenderShadingGeometryHeader, RenderShadingPrimitive, RenderShadingVertex,
    ShadingGeometryBuildError, SHADING_GEOMETRY_FLAG_ACTIVE, SHADING_GEOMETRY_FLAG_INVALID,
    SHADING_GEOMETRY_FLAG_MISSING_NORMAL, SHADING_GEOMETRY_FLAG_MISSING_UV,
};
pub use material::{
    MaterialBindGroup, MaterialBufferBindings, MaterialReader, PrismMaterialDiagnostics,
    PrismMaterialPlugin,
};
pub use opaque::{
    GpuSceneDebugView, GpuSceneOpaqueEnabled, GpuSceneOpaqueIndirectEnabled,
    PrismGpuSceneOpaquePlugin,
};
pub use plugin::{GpuSceneMode, PrismGpuScenePlugin};
pub use scene::RenderGpuScene;
pub use shading::{PrismShadingDiagnostics, PrismShadingPlugin, PrismShadingSettings};
pub use visibility::{
    PrismVisibilityDiagnostics, PrismVisibilityPlugin, UnifiedVisibilityEnabled,
    UnifiedVisibilityReader,
};

#[cfg(test)]
mod tests;
