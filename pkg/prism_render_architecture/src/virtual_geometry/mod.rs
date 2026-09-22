//! Paged cluster geometry feature boundary.

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct GeometryPageKey {
    pub asset: u32,
    pub page: u32,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct GeometryLodPolicy {
    pub target_error_pixels: f32,
    pub hysteresis_pixels: f32,
    pub prefetch_velocity_scale: f32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GeometryRasterPath {
    MeshShader,
    ComputeSoftware,
    IndirectHardware,
    FallbackMesh,
}
