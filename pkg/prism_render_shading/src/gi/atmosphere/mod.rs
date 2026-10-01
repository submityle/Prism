//! Physically based sky/atmosphere: Rayleigh + Mie transmittance, isotropic
//! multiple-scattering coupling, and aerial-perspective in-scattering (Hillaire
//! 2020, "A Scalable and Production Ready Sky and Atmosphere").
//!
//! * [`medium`] — planetary geometry, exponential Rayleigh/Mie density
//!   profiles, the optional ozone tent, and the spectral `σ_s` / `σ_t`
//!   coefficients they induce.
//! * [`phase`] — Rayleigh `3/(16π)(1+cos²θ)` and Cornette-Shanks (Mie) angular
//!   phase functions, energy-conserving and degenerating cleanly at `g = 0`.
//! * [`transmittance`] — ray/shell quadratic intersection and midpoint
//!   quadrature of `T = exp(-∫ σ_t ds)` to the first atmosphere boundary.
//! * [`scattering`] — single scattering along a view ray, Hillaire's isotropic
//!   multiple-scattering geometric coupling (`L_2 / (1 - f)`), and bounded
//!   aerial-perspective in-scattering + transmittance for compositing.
//!
//! # Conventions
//! * Densities are exponential in altitude; phase functions are Rayleigh and
//!   Cornette-Shanks (Mie). All integrals are deterministic ray-march quadrature.
//! * Lengths are kilometres, coefficients per-kilometre; radiance and
//!   transmittance are spectral linear-RGB with transmittance in `(0, 1]`.
//! * All helpers are deterministic CPU golden pure functions (no RNG/IO/GPU/unsafe),
//!   fully defensive (clamped, finite, never `NaN`), with transcendentals via
//!   [`bevy_math::ops`].

pub mod medium;
pub mod phase;
pub mod scattering;
pub mod transmittance;
