use bevy_asset::{AssetId, Handle};
use bevy_camera::primitives::Aabb;
use bevy_ecs::component::Component;
use bevy_mesh::{Mesh, Mesh3d};
use bevy_pbr::StandardMaterial;
use bevy_transform::components::GlobalTransform;
use prism_render_architecture::gpu_scene::{GeometryHandle, SceneHandle, SceneMaterialHandle};

/// Opt-in configuration for an entity mirrored into the Prism GPU Scene.
#[derive(Component, Clone, Copy, Debug, Default)]
pub struct PrismGpuSceneEntity {
    /// Optional stable geometry override. When absent the mesh asset is
    /// assigned a stable runtime geometry handle.
    pub geometry: Option<GeometryHandle>,
    /// Optional stable material override. A non-zero value takes precedence
    /// over `MeshMaterial3d<StandardMaterial>`.
    pub material: SceneMaterialHandle,
    /// Renderer-defined instance flags.
    pub flags: u32,
    /// Fast-path layer mask. The default selects layer zero.
    pub render_layers: u32,
}

/// One optional prepared static-mesh LOD. Level one is the first entry in the
/// owning [`PrismGpuSceneLodSet`]; the entity's [`Mesh3d`] is level zero.
#[derive(Clone, Debug)]
pub struct PrismGpuSceneLod {
    pub mesh: Handle<Mesh>,
    pub screen_error: f32,
}

/// Explicit LOD assets bound to a retained GPU-Scene entity.
#[derive(Component, Clone, Debug, Default)]
pub struct PrismGpuSceneLodSet {
    pub lods: Vec<PrismGpuSceneLod>,
}

/// Render-world copy of the scene fields required by the retained mirror.
#[derive(Component, Clone, Debug)]
pub struct ExtractedSceneInstance {
    pub main_entity: bevy_render::sync_world::MainEntity,
    pub handle: Option<SceneHandle>,
    pub transform: GlobalTransform,
    pub bounds: Option<Aabb>,
    pub mesh: Mesh3d,
    pub lod_meshes: Vec<(AssetId<Mesh>, f32)>,
    pub geometry: Option<GeometryHandle>,
    pub material: SceneMaterialHandle,
    pub material_asset: Option<AssetId<StandardMaterial>>,
    pub flags: u32,
    pub render_layers: u32,
}

/// Stable GPU Scene address copied onto the synchronized render entity.
#[derive(Component, Clone, Copy, Debug, Eq, PartialEq)]
pub struct GpuSceneInstanceAddress {
    pub index: u32,
    pub generation: u32,
}
