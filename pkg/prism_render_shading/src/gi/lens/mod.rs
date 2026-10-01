//! Physical lens post-process effects (CPU golden reference).
//!
//! # Conventions
//! * Pure, deterministic functions: no RNG, IO, GPU, or `unsafe`.
//! * Transcendental functions via [`bevy_math::ops`]; never `f32::exp()` style.
//! * Storage layouts mirror the GPU twin (f32 fields, explicit alignment).
//! * Defensive clamping everywhere: guard divide-by-zero, enforce
//!   non-negativity and finiteness, and fall back gracefully on degenerate
//!   input so no `NaN`/`inf` ever escapes.
//!
//! Implements radial chromatic aberration, lens-flare ghosts and halos, and natural/optical vignetting with lens distortion.

pub mod chromatic;
pub mod flare;
pub mod vignette;

pub use chromatic::{ChromaticAberration, chromatic_offsets, resample_rgb};
pub use flare::{FlareConfig, GhostSample, prefilter, synthesize_ghosts};
pub use vignette::{BrownConrady, Vignette, distort, undistort};
