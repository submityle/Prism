//! Sky LUT baking and sampling (Hillaire): transmittance LUT, multiple-scatter
//! LUT, and sky-view LUT, parameterized for cheap runtime lookup.
//!
//! # Conventions
//! * LUTs are backed by the physical atmosphere model in [`crate::gi::atmosphere`];
//!   this module only owns the UV<->physical parameterization and bilinear fetch.
//! * All helpers are deterministic CPU golden pure functions (no RNG/IO/GPU/unsafe).
//!
//! * [`transmittance_lut`] — spectral transmittance to the top boundary in the
//!   Bruneton/Hillaire `(r, mu)` boundary-distance warp.
//! * [`multiscatter_lut`] — Hillaire's isotropic multiple-scattering factor
//!   `L₂ / (1 - f)` over `(r, mu_sun)`.
//! * [`sky_view_lut`] — full-sky radiance (single + multiple scattering) in the
//!   horizon-centred latitude/longitude parameterization.

pub mod multiscatter_lut;
pub mod sky_view_lut;
pub mod transmittance_lut;
