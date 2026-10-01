//! Micro-scale occlusion: normal-map-derived micro bent normal + cavity AO.
//!
//! # Conventions
//! * `no_std`: containers via `alloc`; math via `bevy_math`; transcendentals
//!   via `bevy_math::ops` (never `f32::exp`).
//! * All items are deterministic pure functions with `#[cfg(test)]` coverage.

pub mod micro_occlusion;
