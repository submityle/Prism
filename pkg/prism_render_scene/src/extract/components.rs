use bevy_camera::primitives::Aabb;
use bevy_ecs::component::Component;
use bevy_mesh::Mesh3d;
use bevy_transform::components::GlobalTransform;
use prism_render_architecture::gpu_scene::SceneHandle;

/// Opt-in marker for an entity mirrored into the Prism GPU Scene.
#[derive(Component, Clone, Copy, Debug, Default)]
pub struct PrismGpuSceneEntity;

/// Render-world copy of the scene fields required by the retained mirror.
#[derive(Component, Clone, Debug)]
pub struct ExtractedSceneInstance {
    pub handle: Option<SceneHandle>,
    pub transform: GlobalTransform,
    pub bounds: Option<Aabb>,
    pub mesh: Mesh3d,
    pub flags: u32,
    pub render_layers: u32,
}
