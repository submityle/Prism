use prism_ecs::component::Component;
use prism_math::{Frustum, Mat4};
use prism_render_architecture::gpu_scene::{GeometryHandle, SceneMaterialHandle};
use prism_render_visibility::ViewFlags;

/// Instance intent. All fields are synchronized on edits without replacing its
/// scene handle. An instance also needs `LocalBounds` and `GlobalTransform`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PrismRenderable {
    /// Registered backend geometry/LOD chain.
    pub geometry: GeometryHandle,
    /// Registered backend material.
    pub material: SceneMaterialHandle,
    /// Intersected with each camera's layer mask.
    pub render_layers: u32,
    /// Core instance flags (bit zero suppresses shadow work).
    pub flags: u32,
}
impl Component for PrismRenderable {}
impl PrismRenderable {
    /// Construct an instance visible to all layers, with no flags.
    pub fn new(geometry: GeometryHandle, material: SceneMaterialHandle) -> Self {
        Self {
            geometry,
            material,
            render_layers: u32::MAX,
            flags: 0,
        }
    }
}

/// The union of emitted work across all live cameras. Written only on a flip.
/// Without any valid camera, or after losing an instance prerequisite, it is false.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PrismViewVisibility {
    /// Whether any view emitted work for this entity in the current frame.
    pub visible: bool,
}
impl Component for PrismViewVisibility {}
impl PrismViewVisibility {
    /// Read the current union result.
    pub fn get(&self) -> bool {
        self.visible
    }
}

/// Native culling planes, in world space. Nonzero planes are normalized when
/// lowered for sphere tests. Zero-normal constant planes support infinite far.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PrismFrustum(pub Frustum);
impl Component for PrismFrustum {}

/// Camera parameters accompanying `PrismFrustum` and a native world transform.
/// Matrices and planes are supplied by the application; camera pose drives LOD.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PrismCamera {
    /// Column-major clip-from-world matrix.
    pub clip_from_world: Mat4,
    /// Pixel viewport `[x, y, width, height]`.
    pub viewport: [u32; 4],
    /// LOD bias, defaulting to one.
    pub lod_scale: f32,
    /// Visible instance layers.
    pub layer_mask: u32,
    /// Core view flags, including camera cuts.
    pub flags: ViewFlags,
    /// Bump to invalidate temporal history (for example after a projection reset).
    pub history_epoch: u64,
}
impl Component for PrismCamera {}
impl Default for PrismCamera {
    fn default() -> Self {
        Self {
            clip_from_world: Mat4::IDENTITY,
            viewport: [0; 4],
            lod_scale: 1.0,
            layer_mask: u32::MAX,
            flags: ViewFlags::default(),
            history_epoch: 0,
        }
    }
}
impl PrismCamera {
    /// Construct a camera with default LOD, flags, layers and history epoch.
    pub fn new(clip_from_world: Mat4, viewport: [u32; 4]) -> Self {
        Self {
            clip_from_world,
            viewport,
            ..Default::default()
        }
    }
}
