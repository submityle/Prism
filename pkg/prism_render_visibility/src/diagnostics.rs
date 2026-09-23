#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct VisibilityDiagnostics {
    pub input_instances: u32,
    pub visible_instances: u32,
    pub stale_handles: u32,
    pub layer_rejected: u32,
    pub frustum_rejected: u32,
    pub occlusion_rejected: u32,
    pub missing_geometry: u32,
    pub missing_material: u32,
    pub lod_fallbacks: u32,
    pub shadow_casters: u32,
    pub ray_scene_instances: u32,
    pub material_bins: u32,
    pub overflowed: bool,
}
