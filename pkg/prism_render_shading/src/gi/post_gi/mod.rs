//! Post-process GI pipeline CPU golden references.
//!
//! Backend-neutral, GPU-free numerical references for the tail of the GI
//! lighting pipeline: histogram-based auto-exposure, filmic tone mapping
//! (ACES / AgX), physically weighted bloom, and the final GI / AO composite.
//!
//! # Conventions
//! * Deterministic pure functions, no RNG / IO / GPU / unsafe.
//! * Transcendental math via [`bevy_math::ops`]; `sqrt` via the inherent method.
//! * Defensive clamping everywhere; never emit `NaN`.

pub mod auto_exposure;
pub mod bloom;
pub mod composite;
pub mod tonemap;
