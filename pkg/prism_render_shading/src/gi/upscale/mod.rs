//! Temporal super-resolution and checkerboard reconstruction (classic, no ML):
//! sub-pixel jitter sequences, motion-compensated history accumulation with
//! confidence, and a sharpening resolve. Pure signal-processing upsampling.
//!
//! The pipeline mirrors a modern temporal upscaler (TAA/TSR) implemented purely
//! with classical numerics:
//!
//! * [`jitter`] — Halton (2, 3) sub-pixel camera-jitter offsets in `[-0.5, 0.5]`
//!   pixel units, with phase/length queries and low-discrepancy guarantees.
//! * [`accumulate`] — bilinear splatting of jittered low-res samples into a
//!   high-res grid, confidence- and sample-count-driven exponential moving
//!   averages, disocclusion resets, and luma-stability variance clipping.
//! * [`resolve`] — edge-aware checkerboard reconstruction, temporal/spatial
//!   blending, and Catmull-Rom / unsharp-mask sharpening with anti-overshoot
//!   clamping.
//!
//! # Conventions
//! * Reconstruction is a weighted temporal accumulation of jittered low-res
//!   samples into a high-res grid; no neural networks or learned priors are used.
//! * All helpers are deterministic CPU golden pure functions (no RNG/IO/GPU/unsafe).
//! * Colours are linear-RGB [`bevy_math::Vec3`], kept finite and non-negative;
//!   transcendental maths goes through [`bevy_math::ops`].

pub mod accumulate;
pub mod jitter;
pub mod resolve;
