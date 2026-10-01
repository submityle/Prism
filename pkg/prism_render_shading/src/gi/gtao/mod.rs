//! Ground-Truth Ambient Occlusion (GTAO) CPU golden references.
//!
//! Backend-neutral references for the horizon-based GTAO integral (Jimenez et
//! al. 2016): per-slice horizon search, the cosine-weighted visibility
//! integral, multi-bounce lightening, a thickness heuristic, and bent-normal
//! accumulation across slices.
//!
//! # Conventions
//! * Deterministic pure functions, no RNG / IO / GPU / unsafe.
//! * Transcendental math via [`bevy_math::ops`]; `sqrt` via the inherent method.
//! * Defensive clamping everywhere; never emit `NaN`.

pub mod horizon;
pub mod integral;
pub mod multibounce;
