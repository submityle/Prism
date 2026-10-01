//! Global-illumination CPU golden reference implementations.
//!
//! Houses backend-neutral, GPU-free numerical references for the render
//! engine's GI passes.  Every item here is a deterministic pure function with
//! unit tests; its output is the numerical reference the WESL/GPU twin passes
//! must reproduce under real-device Metal parity.
//!
//! * [`world_space`] — Lumen-style screen-probe + world-space radiance-cache
//!   pipeline, including DDGI-style Chebyshev visibility for leak suppression
//!   and spherical-Gaussian glossy reconstruction.
//! * [`screen_probe`] — screen-space probe sampling and reuse: ReSTIR GI
//!   reservoirs/GRIS and path-guided importance sampling.
//! * [`scene`] — scene-level signed-distance structures: mesh distance fields
//!   and distance-field ambient occlusion.
//! * [`occlusion`] — bent-normal and specular-occlusion reconstruction from
//!   hemispherical visibility.
//! * [`sample`] — low-discrepancy / blue-noise-like sampling (Owen-scrambled
//!   Sobol' + R2) and BSDF-domain mappings shared by every GI ray budget.
//! * [`denoise`] — firefly clamping and running-variance estimation shared by
//!   the spatio-temporal denoisers.

pub mod denoise;
pub mod occlusion;
pub mod sample;
pub mod scene;
pub mod screen_probe;
pub mod world_space;
