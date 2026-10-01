//! Material-level GI responses: thin-film iridescence + water/wet-surface GI.
//!
//! # Conventions
//! * `no_std`: containers via `alloc`; math via `bevy_math`; transcendentals
//!   via `bevy_math::ops` (never `f32::exp`).
//! * All items are deterministic pure functions with `#[cfg(test)]` coverage.

pub mod thin_film;
pub mod water_gi;
