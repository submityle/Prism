//! Reconstruction-filter motion blur (CPU golden reference).
//!
//! # Conventions
//! * Pure, deterministic functions: no RNG, IO, GPU, or `unsafe`.
//! * Transcendental functions via [`bevy_math::ops`]; never `f32::exp()` style.
//! * Storage layouts mirror the GPU twin (f32 fields, explicit alignment).
//! * Defensive clamping everywhere: guard divide-by-zero, enforce
//!   non-negativity and finiteness, and fall back gracefully on degenerate
//!   input so no `NaN`/`inf` ever escapes.
//!
//! Implements the McGuire et al. (2012) tile-velocity reconstruction filter:
//! jittered along-velocity sampling with soft depth classification and
//! cone/cylinder foreground-background weighting.


pub mod weights;
pub mod sampling;
pub mod reconstruction;

pub use reconstruction::{reconstruct, PixelSample, ReconstructionParams};
