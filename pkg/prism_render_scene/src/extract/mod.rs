mod components;
pub(crate) mod lifecycle;
mod systems;

pub use components::{ExtractedSceneInstance, PrismGpuSceneEntity};
pub(crate) use lifecycle::retire_unused_geometry;
pub(crate) use systems::{apply_extracted_scene_changes, extract_scene_instances};
