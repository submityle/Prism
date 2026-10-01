//! Denoising CPU golden helpers (firefly clamping + variance estimation).
//!
//! Backend-neutral references for the spatio-temporal GI denoiser:
//! * [`firefly`] — luminance-aware outlier (firefly) suppression.
//! * [`variance`] — Welford running mean/variance for ReLAX-style guidance.
//! * [`reblur`] — ReBLUR/ReLAX spatio-temporal bilateral denoiser (edge-stopping
//!   à-trous + temporal reprojection/accumulation + specular virtual reprojection).

pub mod firefly;
pub mod reblur;
pub mod variance;
