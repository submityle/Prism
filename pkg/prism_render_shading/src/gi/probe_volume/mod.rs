//! Probe volumes: adaptive probe grid (APV) irradiance encoding + sky-visibility
//! occlusion, with trilinear + depth-aware (Chebyshev) probe weighting.
//!
//! * [`sh_irradiance`] — L1 (4-coefficient) / L2 (9-coefficient) spherical-
//!   harmonic irradiance encode/decode: `project_radiance`, `eval_irradiance`,
//!   `eval_radiance`, and the clamped-cosine convolution factors
//!   (`A0 = pi`, `A1 = 2*pi/3`, `A2 = pi/4`); a constant radiance field decodes
//!   to flat `pi * radiance` irradiance.
//! * [`probe_grid`] — eight-corner trilinear interpolation combined with a
//!   smooth normal/back-face weight and a Chebyshev depth weight (reusing
//!   `gi::world_space::visibility`), renormalised with a uniform fallback.
//! * [`sky_occlusion`] — octahedral depth-moment storage (reusing
//!   `gi::world_space::octahedral`) with Chebyshev directional visibility and a
//!   cosine-weighted hemisphere sky-visibility scalar.
//!
//! # Conventions
//! * Irradiance stored as L1/L2 spherical harmonics or octahedral depth moments.
//! * All helpers are deterministic CPU golden pure functions (no RNG/IO/GPU/unsafe).

pub mod probe_grid;
pub mod sh_irradiance;
pub mod sky_occlusion;

pub use probe_grid::{
    blend_probe_sh, normal_backface_weight, resolve_corner_weights, sample_probe_grid,
    trilinear_weights, ProbeCorner,
};
pub use sh_irradiance::{
    cosine_convolution_l1, cosine_convolution_l2, sh_basis_l1, sh_basis_l2, ShL1Irradiance,
    ShL2Irradiance, A0, A1, A2,
};
pub use sky_occlusion::SkyOcclusion;
