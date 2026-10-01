//! Environment-BRDF split-sum CPU golden references.
//!
//! Backend-neutral references for pre-integrated specular environment lighting:
//! the split-sum DFG LUT (scale/bias), multiscatter energy compensation, and
//! sheen / clearcoat environment response.
//!
//! # Conventions
//! * Deterministic pure functions, no RNG / IO / GPU / unsafe.
//! * Transcendental math via [`bevy_math::ops`]; `sqrt` via the inherent method.
//! * Defensive clamping everywhere; never emit `NaN`.

pub mod dfg_lut;
pub mod multiscatter;
pub mod sheen_clearcoat;
