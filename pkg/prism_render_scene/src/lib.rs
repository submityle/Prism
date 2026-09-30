//! Bevy integration for Prism's retained GPU Scene.

#![expect(
    missing_docs,
    reason = "The experimental integration API is still evolving."
)]

extern crate alloc;

mod buffers;
mod cloth;
mod compare;
mod completion;
mod consumer;
mod diagnostics;
mod extract;
mod geometry;
mod lighting;
mod material;
mod opaque;
mod plugin;
mod raytrace;
mod scene;
mod shading;
mod visibility;
mod water;

pub use buffers::{
    GpuSceneBuffers, RenderGpuSceneBounds, RenderGpuSceneInstance, RenderGpuSceneTransform,
};
pub use cloth::{ClothGarment, ClothGarmentBuilder};
pub use compare::GpuSceneParityDiagnostics;
pub use completion::GpuCompletionTracker;
pub use consumer::{GpuSceneBufferBindings, GpuSceneReader};
pub use diagnostics::{GpuSceneDiagnostics, GpuSceneUploadSettings};
pub use extract::{ExtractedSceneInstance, GpuSceneInstanceAddress, PrismGpuSceneEntity};
pub use geometry::{
    build_shading_geometry, GeometryBindGroup, RenderGeometryRegistry, RenderShadingGeometry,
    RenderShadingGeometryBuffers, RenderShadingGeometryEntry, RenderShadingGeometryHeader,
    RenderShadingGeometryRegistry, RenderShadingPrimitive, RenderShadingVertex,
    ShadingGeometryBuildError, SHADING_GEOMETRY_FLAG_ACTIVE, SHADING_GEOMETRY_FLAG_INVALID,
    SHADING_GEOMETRY_FLAG_MISSING_NORMAL, SHADING_GEOMETRY_FLAG_MISSING_UV,
};
pub use lighting::{
    build_cluster_data, cubemap_faces_from_image, project_image_to_sh, ClusterBindGroup,
    ClusterConfig, ClusterCpuData, ClusterGpuBuffers, ClusterViewFit, EnvironmentProbeCache,
    ExtractedClusterView, ExtractedLights, GpuClusterGrid, GpuDirectionalLight,
    GpuLightEnvironment, GpuPunctualLight, LightBindGroup, LightGpuBuffers, PrismLightingPlugin,
    StylizedLighting, LIGHT_ENVIRONMENT_FLAG_IMAGE_BASED,
};
pub use material::{
    BindlessHeapStats, BindlessSlot, BindlessTextureHeap, MaterialBindGroup,
    MaterialBufferBindings, MaterialReader, MaterialTextureArrays, PrismMaterialDiagnostics,
    PrismMaterialPlugin, MAX_BINDLESS_TEXTURES,
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
pub use water::{
    CoastlinePreset, FlipPoolPreset, LakeInflow, LakePreset, OceanPreset, PbfPoolPreset,
    RiverControlPoint, RiverPreset, ShallowWaterPreset, WaterBody,
};

#[cfg(test)]
mod tests;
