//! Bevy integration for Prism's retained GPU Scene.

#![expect(
    missing_docs,
    reason = "The experimental integration API is still evolving."
)]

extern crate alloc;

mod buffers;
mod completion;
mod consumer;
mod diagnostics;
mod extract;
mod plugin;
mod scene;

pub use buffers::{
    GpuSceneBuffers, RenderGpuSceneBounds, RenderGpuSceneInstance, RenderGpuSceneTransform,
};
pub use completion::GpuCompletionTracker;
pub use consumer::{GpuSceneBufferBindings, GpuSceneReader};
pub use diagnostics::{GpuSceneDiagnostics, GpuSceneUploadSettings};
pub use extract::{ExtractedSceneInstance, GpuSceneInstanceAddress, PrismGpuSceneEntity};
pub use plugin::{GpuSceneMode, PrismGpuScenePlugin};
pub use scene::RenderGpuScene;

#[cfg(test)]
mod tests;
