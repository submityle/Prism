//! Contrast-adaptive sharpening and debanding (CPU golden reference).
//!
//! # Conventions
//! * Pure, deterministic functions: no RNG, IO, GPU, or `unsafe`.
//! * Transcendental functions via [`bevy_math::ops`]; never `f32::exp()` style.
//! * Storage layouts mirror the GPU twin (f32 fields, explicit alignment).
//! * Defensive clamping everywhere: guard divide-by-zero, enforce
//!   non-negativity and finiteness, and fall back gracefully on degenerate
//!   input so no `NaN`/`inf` ever escapes.
//!
//! Implements FidelityFX CAS, robust RCAS, and ordered-dither debanding.

pub mod cas;
pub mod deband;
pub mod rcas;

pub use cas::{cas, Neighborhood3x3};
pub use deband::{deband, ign};
pub use rcas::{rcas, Cross};
