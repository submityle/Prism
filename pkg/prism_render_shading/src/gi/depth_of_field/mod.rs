//! Physically-based depth of field (CPU golden reference).
//!
//! # Conventions
//! * Pure, deterministic functions: no RNG, IO, GPU, or `unsafe`.
//! * Transcendental functions via [`bevy_math::ops`]; never `f32::exp()` style.
//! * Storage layouts mirror the GPU twin (f32 fields, explicit alignment).
//! * Defensive clamping everywhere: guard divide-by-zero, enforce
//!   non-negativity and finiteness, and fall back gracefully on degenerate
//!   input so no `NaN`/`inf` ever escapes.
//!
//! This module implements the thin-lens circle-of-confusion model, a
//! scatter-as-gather bokeh accumulator, and near/far layer compositing.


pub mod coc;
pub mod gather;
pub mod layers;

pub use coc::LensParams;
pub use gather::{gather, golden_angle_disk, GatherParams, GatherResult, GatherSample};
pub use layers::{composite, DofLayers, LayerParams};
