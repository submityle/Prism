//! Volumetric cloud CPU golden references.
//!
//! Backend-neutral references for a ray-marched volumetric cloud layer that
//! plugs into [`crate::gi::atmosphere`]: cloud density modelling, Beer-powder
//! extinction, Henyey-Greenstein multi-scatter, and layer raymarch integration.
//!
//! # Conventions
//! * Deterministic pure functions, no RNG / IO / GPU / unsafe.
//! * Transcendental math via [`bevy_math::ops`]; `sqrt` via the inherent method.
//! * Defensive clamping everywhere; never emit `NaN`.

pub mod density;
pub mod noise;
pub mod raymarch;
pub mod scattering;
