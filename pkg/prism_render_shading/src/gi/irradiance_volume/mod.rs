//! Dynamic diffuse global illumination (DDGI): octahedral irradiance + depth
//! probes with temporal blending, Chebyshev visibility, and probe relocation.
//!
//! This is the CPU golden reference for Majercik et al. 2019 ("Dynamic Diffuse
//! Global Illumination with Ray-Traced Irradiance Fields") and its RTXGI
//! follow-ups, split into three cooperating modules:
//!
//! * [`ddgi_probe`] — per-probe octahedral *irradiance* field: cosine-weighted
//!   ray accumulation, temporal hysteresis blending, and octahedral gutter
//!   border replication for seamless bilinear sampling.
//! * [`visibility`] — per-probe octahedral *depth / visibility* field: a
//!   sharpened-cosine two-moment depth map feeding the Chebyshev (variance)
//!   leak-suppression weight with a self-shadow bias.
//! * [`relocation`] — probe *management*: world↔grid mapping and toroidal
//!   scrolling, bounded probe relocation, active/inactive/newly-vacated
//!   classification, and the eight-probe trilinear + visibility + back-face
//!   interpolation weights.
//!
//! # Conventions
//! * Each probe stores an octahedral irradiance map and a depth/depth² map used
//!   for Chebyshev (variance) visibility weighting during interpolation.
//! * Octahedral mapping is shared with [`crate::gi::world_space::octahedral`]
//!   and the Chebyshev weight with [`crate::gi::world_space::visibility`], so
//!   the DDGI path and the Lumen-style world-space path agree numerically.
//! * All helpers are deterministic CPU golden pure functions (no RNG/IO/GPU/unsafe).

pub mod ddgi_probe;
pub mod relocation;
pub mod visibility;

pub use ddgi_probe::{cosine_weighted_irradiance, IrradianceOct};
pub use relocation::{
    classify_probe, interpolation_weights, normal_backface_weight, relocate_offset,
    transition_state, trilinear_weights, ProbeGrid, ProbeRayStats, ProbeSample, ProbeState,
    DEFAULT_RELOCATION_LIMIT,
};
pub use visibility::{sharpened_depth_moments, DdgiDepthOct, DEFAULT_DEPTH_SHARPNESS};
