//! Global-illumination CPU golden reference implementations.
//!
//! Houses backend-neutral, GPU-free numerical references for the render
//! engine's GI passes.  Every item here is a deterministic pure function with
//! unit tests; its output is the numerical reference the WESL/GPU twin passes
//! must reproduce under real-device Metal parity.
//!
//! * [`world_space`] — Lumen-style screen-probe + world-space radiance-cache
//!   pipeline, including DDGI-style Chebyshev visibility for leak suppression.
//! * [`sample`] — low-discrepancy / blue-noise-like sampling (Owen-scrambled
//!   Sobol' + R2) and BSDF-domain mappings shared by every GI ray budget.
//! * [`denoise`] — firefly clamping and running-variance estimation shared by
//!   the spatio-temporal denoisers.

pub mod denoise;
pub mod sample;
pub mod world_space;
