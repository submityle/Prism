//! Hardware and software ray-scene contracts.

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AccelerationUpdate {
    Reuse,
    Refit,
    Rebuild,
    BuildAndCompact,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TraceBackend {
    ScreenSpace,
    SoftwareBvh,
    HardwareRayQuery,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct RayFootprint {
    pub cone_width: f32,
    pub cone_spread_angle: f32,
    pub hit_distance: f32,
}
