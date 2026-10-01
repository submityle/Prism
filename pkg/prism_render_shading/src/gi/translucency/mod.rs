//! Translucency and subsurface scattering GI: Burley normalized diffusion
//! profile, thickness-based transmission, and two-sided foliage forward
//! scattering for diffuse light transport through thin/soft media.
//!
//! # Conventions
//! * The diffusion profile is the Christensen-Burley normalized approximation;
//!   transmission uses Beer-Lambert attenuation over a signed thickness estimate.
//! * All helpers are deterministic CPU golden pure functions (no RNG/IO/GPU/unsafe)
//!   and route transcendental maths through [`bevy_math::ops`] for GPU-twin parity.
//!
//! * [`diffusion`] — Christensen-Burley normalized diffusion profile: albedo →
//!   shaping fit, dmfp → scaling length, radial profile / pdf / CDF, and exact
//!   analytic inverse-CDF radius importance sampling (per-channel RGB).
//! * [`transmission`] — thickness-based translucent transmission: Beer-Lambert
//!   attenuation over signed thickness, colour ↔ extinction inversion, and
//!   wrap-around back-lit transmitted radiance.
//! * [`foliage`] — two-sided foliage forward scattering: HG-style forward lobe
//!   on the view/light geometry, roughness → anisotropy, and a normal-independent
//!   two-sided wrap diffuse combined with thickness attenuation.

pub mod diffusion;
pub mod foliage;
pub mod transmission;
