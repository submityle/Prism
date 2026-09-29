//! World-space global illumination CPU golden.
//!
//! A backend-neutral, GPU-free reference implementation of a Lumen-style
//! diffuse GI pipeline built from two cooperating structures:
//!
//! * **Screen probes** — one radiance probe per screen tile. [`probe_placement`]
//!   lays them out over the framebuffer, and [`octahedral`] parameterises the
//!   per-probe directional storage.
//! * **A world-space radiance cache** — a voxel grid of L1 SH irradiance probes
//!   ([`radiance_cache`]) that persists indirect light across frames and
//!   off-screen geometry.
//!
//! At shading time [`probe_interpolation`] gathers the four screen probes
//! around a pixel and blends their SH irradiance with bilinear plus
//! geometry-aware (normal/depth) weights.
//!
//! Every function here is a deterministic pure function with unit tests; these
//! outputs are the numerical reference the future GPU passes must reproduce.
//! The SH conventions (basis ordering, `pi` normalisation, cosine-lobe band
//! factors) intentionally match the `environment` module's L2 probe so the two
//! remain interoperable.

pub mod octahedral;
pub mod probe_interpolation;
pub mod probe_placement;
pub mod radiance_cache;

pub use octahedral::{dir_to_oct, oct_to_dir};
pub use probe_interpolation::{
    bilinear_weights, blend_sh, interpolate_irradiance, probe_bilinear_coords, resolve_weights,
    similarity_weight, InterpolationConfig, ProbeNeighbor,
};
pub use probe_placement::{
    probe_center_pixel, probe_coord, probe_count, probe_grid_dims, probe_index, probe_pixel_rect,
    PixelRect,
};
pub use radiance_cache::{
    cell_to_key, evaluate_irradiance, world_to_cell, RadianceCell, ShL1Rgb,
};
