//! Color-grading CPU golden references.
//!
//! Backend-neutral references for the grading stage after tone mapping: ASC CDL
//! (slope / offset / power), lift-gamma-gain, white balance (von Kries chromatic
//! adaptation), saturation / contrast, and trilinear 3D-LUT application.
//!
//! # Conventions
//! * Deterministic pure functions, no RNG / IO / GPU / unsafe.
//! * Transcendental math via [`bevy_math::ops`]; `sqrt` via the inherent method.
//! * Defensive clamping everywhere; never emit `NaN`.

pub mod cdl;
pub mod lut3d;
pub mod white_balance;
