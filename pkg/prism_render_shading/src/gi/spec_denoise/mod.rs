//! Specular / reflection denoiser (ReBLUR-spec / ReLAX-spec style): roughness-aware
//! history reprojection along the virtual (hit-distance) reflection point,
//! specular history clamping, and hit-distance-driven anisotropic spatial blur.
//!
//! # Conventions
//! * Complements the diffuse spatio-temporal filter in [`crate::gi::denoise`];
//!   this module owns the specular lobe (view-dependent, roughness-scaled) and
//!   never re-derives the diffuse kernels.
//! * All helpers are deterministic CPU golden pure functions (no RNG/IO/GPU/unsafe);
//!   transcendentals go through [`bevy_math::ops`] and every result is finite.
//! * The three stages compose: [`reproject`] produces a virtual sample position
//!   plus a confidence weight, [`history_clamp`] guards temporal accumulation
//!   with that confidence, and [`spatial`] cleans the residual noise with a
//!   hit-distance-driven bilateral blur.
//!
//! * [`reproject`] — virtual-reflection-point reprojection, parallax, and
//!   roughness-weighted reprojection confidence.
//! * [`history_clamp`] — colour-space history clipping, roughness-aware history
//!   length, GGX lobe-shift rejection, and fast-history response.
//! * [`spatial`] — anisotropic blur radius with contact hardening, bilateral
//!   edge-stopping, and normalised hit-distance reconstruction.

pub mod history_clamp;
pub mod reproject;
pub mod spatial;
