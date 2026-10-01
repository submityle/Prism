//! Volumetric global illumination: froxel (frustum-voxel) radiance reservoirs
//! for participating-media single/multiple scattering (Volumetric ReSTIR).
//!
//! This subsystem is the backend-neutral, CPU-golden numerical reference for
//! the engine's volumetric lighting pass.  Light scattered by a participating
//! medium (fog, haze, god-rays) is integrated per *froxel* — a cell of a
//! view-frustum-aligned voxel grid — and the per-froxel light samples are
//! recycled across the frustum and across frames with reservoir resampling,
//! exactly as the WESL/GPU twin must reproduce.
//!
//! * [`froxel`] — the frustum-voxel grid with exponential depth slicing
//!   (`z_slice = near * (far / near)^(s / N)`) and clamped world↔froxel↔view
//!   coordinate mappings.
//! * [`scattering`] — Henyey-Greenstein phase function (isotropic as `g → 0`),
//!   Beer-Lambert transmittance, in-scattering, and analytic single / geometric
//!   multiple scattering.
//! * [`volumetric_reservoir`] — Volumetric ReSTIR: streaming RIS over froxel
//!   light candidates (reusing [`crate::gi::screen_probe::restir::Reservoir`])
//!   with temporal reprojection and an M-capped history.
//!
//! # Conventions
//! * Froxels use an exponential depth slicing; scattering uses Henyey-Greenstein
//!   phase + Beer-Lambert transmittance.
//! * All helpers are deterministic CPU golden pure functions (no RNG/IO/GPU/unsafe).
//! * Transcendental maths goes through [`bevy_math::ops`]; every result is
//!   finite, clamped, and never `NaN`.

pub mod froxel;
pub mod scattering;
pub mod volumetric_reservoir;
