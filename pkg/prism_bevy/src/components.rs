//! Public ECS components that drive and report Prism visibility.

use bevy_ecs::prelude::Component;
use bevy_math::Mat4;
use prism_render_architecture::gpu_scene::{GeometryHandle, SceneMaterialHandle};
use prism_render_visibility::ViewFlags;

/// Marks an entity that should be mirrored into [`crate::PrismRenderScene`] and
/// culled by Prism.
///
/// The entity must also carry a [`bevy_camera::primitives::Aabb`] (local-space
/// bounds) and a [`bevy_transform::components::GlobalTransform`]; the plugin
/// lowers those into a Prism instance record. The `geometry` and `material`
/// handles reference backend resources registered on the scene resource via
/// [`crate::PrismRenderScene::insert_geometry`] and
/// [`crate::PrismRenderScene::insert_material`].
///
/// Geometry and material are read when the entity is first observed; changing
/// them later currently requires despawning and respawning the entity.
#[derive(Component, Clone, Copy, Debug)]
pub struct PrismRenderable {
    /// Backend geometry (LOD chain) handle.
    pub geometry: GeometryHandle,
    /// Backend material handle.
    pub material: SceneMaterialHandle,
    /// Render-layer bitmask intersected with each view's `layer_mask`.
    pub render_layers: u32,
    /// Instance flags forwarded verbatim into the Prism instance record.
    pub flags: u32,
}

impl PrismRenderable {
    /// Builds a renderable with full render-layer visibility and no flags.
    #[must_use]
    pub fn new(geometry: GeometryHandle, material: SceneMaterialHandle) -> Self {
        Self {
            geometry,
            material,
            render_layers: u32::MAX,
            flags: 0,
        }
    }
}

/// Per-entity visibility result written by the plugin after culling.
///
/// `visible` is `true` when the entity survived culling in *any* [`PrismCamera`]
/// view this frame. Change detection only fires when the boolean flips.
#[derive(Component, Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PrismViewVisibility {
    /// Whether the entity is visible in at least one Prism view this frame.
    pub visible: bool,
}

impl PrismViewVisibility {
    /// Returns whether the entity is visible in any view.
    #[must_use]
    pub fn get(self) -> bool {
        self.visible
    }
}

/// Describes a Prism culling view attached to a camera entity.
///
/// The camera entity must also carry a [`bevy_camera::primitives::Frustum`]
/// (which supplies the six culling planes) and a
/// [`bevy_transform::components::GlobalTransform`] (which supplies the view's
/// world position for LOD selection). This component carries the remaining view
/// parameters that are not recoverable from the frustum alone.
#[derive(Component, Clone, Copy, Debug)]
pub struct PrismCamera {
    /// `clip_from_world` (view-projection) matrix for the view.
    pub clip_from_world: Mat4,
    /// `[x, y, width, height]` viewport in pixels (drives projected size).
    pub viewport: [u32; 4],
    /// Global LOD bias; `1.0` keeps authored screen-error thresholds.
    pub lod_scale: f32,
    /// Render-layer mask intersected with each instance's `render_layers`.
    pub layer_mask: u32,
    /// View behavior flags (reverse-Z, shadow, reflection, offline, cut, ...).
    pub flags: ViewFlags,
}

impl Default for PrismCamera {
    fn default() -> Self {
        Self {
            clip_from_world: Mat4::IDENTITY,
            viewport: [0, 0, 0, 0],
            lod_scale: 1.0,
            layer_mask: u32::MAX,
            flags: ViewFlags::default(),
        }
    }
}

impl PrismCamera {
    /// Builds a view from its `clip_from_world` matrix and pixel viewport,
    /// leaving the LOD scale, layer mask, and flags at their defaults.
    #[must_use]
    pub fn new(clip_from_world: Mat4, viewport: [u32; 4]) -> Self {
        Self {
            clip_from_world,
            viewport,
            ..Self::default()
        }
    }
}
