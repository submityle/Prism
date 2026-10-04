//! Content-specific transparency strategies.

pub mod adaptive;
pub mod layered_glass;
pub mod moment_oit;
pub mod routing;
pub mod sorted_oit;
pub mod weighted_oit;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TransparencyPath {
    Sorted,
    WeightedOit,
    MomentOit,
    /// Order-independent resolve that keeps a bounded, explicit per-pixel
    /// visibility curve (adaptive transparency). Higher fidelity than weighted
    /// OIT with capped memory; approaches the exact `A-buffer` as its node
    /// budget grows.
    Adaptive,
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
