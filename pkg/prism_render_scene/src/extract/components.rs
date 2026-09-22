use bevy_camera::primitives::Aabb;
use bevy_ecs::component::Component;
use bevy_mesh::Mesh3d;
use bevy_transform::components::GlobalTransform;
use prism_render_architecture::gpu_scene::{GeometryHandle, SceneHandle, SceneMaterialHandle};

/// Opt-in configuration for an entity mirrored into the Prism GPU Scene.
#[derive(Component, Clone, Copy, Debug, Default)]
pub struct PrismGpuSceneEntity {
    /// Optional stable geometry override. When absent the mesh asset is
    /// assigned a stable runtime geometry handle.
    pub geometry: Option<GeometryHandle>,
    /// Stable material reference consumed by future shading paths.
    pub material: SceneMaterialHandle,
    /// Renderer-defined instance flags.
    pub flags: u32,
    /// Fast-path layer mask. The default selects layer zero.
    pub render_layers: u32,
}

/// Render-world copy of the scene fields required by the retained mirror.
#[derive(Component, Clone, Debug)]
pub struct ExtractedSceneInstance {
    pub handle: Option<SceneHandle>,
    pub transform: GlobalTransform,
    pub bounds: Option<Aabb>,
    pub mesh: Mesh3d,
    pub geometry: Option<GeometryHandle>,
    pub material: SceneMaterialHandle,
    pub flags: u32,
    pub render_layers: u32,
}

/// Stable GPU Scene address copied onto the synchronized render entity.
#[derive(Component, Clone, Copy, Debug, Eq, PartialEq)]
pub struct GpuSceneInstanceAddress {
    pub index: u32,
    pub generation: u32,
}
