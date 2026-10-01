//! Motion-vector CPU golden references.
//!
//! Backend-neutral references for the motion-vector stage feeding TAA, temporal
//! upscaling, and motion blur: camera + object reprojection velocity, velocity
//! dilation, and tile-wise max-velocity reduction.
//!
//! # Conventions
//! * Deterministic pure functions, no RNG / IO / GPU / unsafe.
//! * Transcendental math via [`bevy_math::ops`]; `sqrt` via the inherent method.
//! * Defensive clamping everywhere; never emit `NaN`.

pub mod reproject;
pub mod dilation;
pub mod tile;
