//! Denoising CPU golden helpers (firefly clamping + variance estimation).
//!
//! Backend-neutral references for the spatio-temporal GI denoiser:
//! * [`firefly`] — luminance-aware outlier (firefly) suppression.
//! * [`variance`] — Welford running mean/variance for ReLAX-style guidance.

pub mod firefly;
pub mod variance;
