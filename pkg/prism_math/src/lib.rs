//! # prism_math
//!
//! Prism's numerical-truth kernel: the L0 root that every other Prism crate
//! depends on. It provides `f32` vectors, matrices, quaternions, and affine
//! transforms with a glam-shaped API.
//!
//! ## Conventions
//! - Column vectors; matrices are **column-major** and multiply on the left
//!   (`m * v`). Composition reads right-to-left: `a * b` applies `b` first.
//! - Right-handed coordinate system.
//! - Rotations compose as `a * b` = "apply `b`, then `a`".
//!
//! ## Milestone status (per the design doc roadmap)
//! - **M0 (this crate, done):** `Vec*/Mat*/Quat/Affine3` on a scalar reference
//!   backend, operators, helpers, and `prelude`, with round-trip tests.
//! - **M1+ (planned):** SIMD backends (SSE/AVX/NEON/WASM), `f64` big-world,
//!   `fixed` determinism, geometry/curve/color/rand/noise toolboxes.
//!
//! The scalar backend here is the behavioural reference that later SIMD
//! backends must match within documented tolerances.

#![cfg_attr(not(test), no_std)]
#![forbid(unsafe_code)]

mod affine;
mod float;
mod mat;
mod quat;
mod vec;

pub use affine::Affine3;
pub use mat::{Mat2, Mat3, Mat4};
pub use quat::Quat;
pub use vec::{Vec2, Vec3, Vec3A, Vec4, vec2, vec3, vec3a, vec4};

/// Mathematical constant helpers (`f32`).
pub mod consts {
    /// Pi.
    pub const PI: f32 = core::f32::consts::PI;
    /// Tau (2*pi).
    pub const TAU: f32 = core::f32::consts::TAU;
    /// Pi/2.
    pub const FRAC_PI_2: f32 = core::f32::consts::FRAC_PI_2;
    /// Degrees-to-radians factor.
    pub const DEG_TO_RAD: f32 = PI / 180.0;
    /// Radians-to-degrees factor.
    pub const RAD_TO_DEG: f32 = 180.0 / PI;
}

/// Convert degrees to radians.
#[inline]
pub fn to_radians(deg: f32) -> f32 {
    deg * consts::DEG_TO_RAD
}
/// Convert radians to degrees.
#[inline]
pub fn to_degrees(rad: f32) -> f32 {
    rad * consts::RAD_TO_DEG
}
/// Scalar linear interpolation.
#[inline]
pub fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

/// Detected SIMD capabilities of the running CPU.
///
/// M0 ships only the scalar backend, so every field is reported `false`. M1
/// will fill these in from `core::arch` feature detection and use them to pick
/// batch paths.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct MathCaps {
    /// SSE2 available.
    pub sse2: bool,
    /// AVX2 available.
    pub avx2: bool,
    /// FMA available.
    pub fma: bool,
    /// ARM NEON available.
    pub neon: bool,
    /// WebAssembly SIMD available.
    pub wasm_simd: bool,
}

impl MathCaps {
    /// Detect capabilities. In M0 this always returns the all-scalar profile.
    #[inline]
    pub fn detect() -> Self {
        Self::default()
    }
}

/// Glob-import the common types and helpers.
pub mod prelude {
    pub use crate::{
        Affine3, Mat2, Mat3, Mat4, MathCaps, Quat, Vec2, Vec3, Vec3A, Vec4, consts, lerp,
        to_degrees, to_radians, vec2, vec3, vec3a, vec4,
    };
}

#[cfg(test)]
mod tests;
