//! Unified reflection CPU golden: stochastic HiZ SSR + reflection-probe blend.
//!
//! # Conventions
//! * `no_std`: containers via `alloc`; math via `bevy_math`; transcendentals
//!   via `bevy_math::ops` (never `f32::exp`).
//! * All items are deterministic pure functions with `#[cfg(test)]` coverage.

pub mod reflection_probe;
pub mod stochastic_ssr;
