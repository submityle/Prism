//! Linearly Transformed Cosine (LTC) area-light CPU golden references.
//!
//! Backend-neutral references for real-time polygonal / disk / line / tube
//! area lights using Heitz et al. 2016 Linearly Transformed Cosines: the LTC
//! matrix LUT parameterized by (NdotV, roughness), clipped polygon edge
//! integration, and specular + diffuse area-light response.
//!
//! # Conventions
//! * Deterministic pure functions, no RNG / IO / GPU / unsafe.
//! * Transcendental math via [`bevy_math::ops`]; `sqrt` via the inherent method.
//! * Defensive clamping everywhere; never emit `NaN`.

pub mod ltc_lut;
pub mod polygon;
pub mod shapes;
