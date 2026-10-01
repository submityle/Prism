//! Next-event-estimation CPU golden references.
//!
//! Backend-neutral references for direct-light sampling: analytic solid-angle
//! sampling of spheres / rectangles / spot cones, light-sampling PDFs, and
//! multiple-importance-sampling (power/balance heuristic) between BSDF and
//! light strategies, plus resampled importance sampling (RIS) of light sets.
//!
//! # Conventions
//! * Deterministic pure functions, no RNG / IO / GPU / unsafe.
//! * Transcendental math via [`bevy_math::ops`]; `sqrt` via the inherent method.
//! * Defensive clamping everywhere; never emit `NaN`.

pub mod light_sampling;
pub mod mis;
pub mod ris;
