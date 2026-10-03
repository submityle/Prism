//! Content-specific transparency strategies.

pub mod routing;
pub mod weighted_oit;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TransparencyPath {
    Sorted,
    WeightedOit,
    MomentOit,
    LayeredGlass,
    SingleLayerWater,
    HairVisibility,
    Volumetric,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct TransparencyOutputs {
    pub writes_reactive_mask: bool,
    pub writes_motion: bool,
    pub contributes_to_ray_scene: bool,
}
