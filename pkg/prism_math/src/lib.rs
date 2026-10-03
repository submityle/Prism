//! # `prism_math`
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
//! - **M1 (done):** SIMD backends (SSE2/NEON) routed through [`Backend`] with
//!   scalar cross-checks and [`MathCaps`] runtime probing.
//! - **M2 (this crate, done):** geometry primitives ([`geom`]), intersection
//!   and culling queries ([`intersect`]), and interpolation/easing/spline
//!   helpers ([`curve`]), cross-checked against analytic references.
//! - **M3+ (planned):** `f64` big-world, `fixed` determinism, and the
//!   color/rand/noise toolboxes.
//!
//! The scalar backend here is the behavioural reference that later SIMD
//! backends must match within documented tolerances.

#![cfg_attr(not(test), no_std)]
// NOTE: `forbid(unsafe_code)` cannot be locally overridden, and the SIMD
// backends require `core::arch` intrinsics (which are `unsafe`). We therefore
// rely on the workspace lint `unsafe_code = "deny"` (which IS overridable) and
// grant narrowly-scoped `#[allow(unsafe_code, reason = ...)]` exceptions in the
// backend modules, each paired with `// SAFETY:` justifications.

mod affine;
mod backend;
mod float;
mod mat;
mod quat;
mod vec;

pub mod curve;
pub mod geom;
pub mod intersect;

pub use affine::Affine3;
pub use backend::Backend;
pub use mat::{Mat2, Mat3, Mat4};
pub use quat::Quat;
pub use vec::{Vec2, Vec3, Vec3A, Vec4, vec2, vec3, vec3a, vec4};

pub use geom::{Aabb3, BoundingSphere, Frustum, Plane, Ray3, Segment3};
pub use intersect::{Containment, RayHit};

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
/// Populated by [`MathCaps::detect`], which combines
/// [`prism_platform::CpuInfo`] (SSE2/AVX2/NEON probing) with a dedicated FMA
/// check. Under `std` the x86 flags come from runtime `cpuid` detection; under
/// `no_std` they fall back to compile-time `target_feature` configuration.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct MathCaps {
    /// SSE2 available (always true on `x86_64`).
    pub sse2: bool,
    /// AVX2 available.
    pub avx2: bool,
    /// FMA (fused multiply-add) available.
    pub fma: bool,
    /// ARM NEON available (always true on `aarch64`).
    pub neon: bool,
    /// WebAssembly SIMD available.
    pub wasm_simd: bool,
}

impl MathCaps {
    /// Probe the current CPU and report the capabilities relevant to backend
    /// selection.
    #[inline]
    pub fn detect() -> Self {
        let cpu = prism_platform::CpuInfo::detect();
        Self {
            sse2: cpu.sse2,
            avx2: cpu.avx2,
            fma: detect_fma(),
            neon: cpu.neon,
            wasm_simd: detect_wasm_simd(),
        }
    }

    /// Report which math [`Backend`] the dispatch layer uses on this CPU.
    #[inline]
    pub fn active_backend(self) -> Backend {
        backend::active()
    }
}

/// Detect FMA support (not surfaced by [`prism_platform::CpuInfo`]).
#[inline]
fn detect_fma() -> bool {
    #[cfg(all(target_arch = "x86_64", feature = "std"))]
    {
        std::is_x86_feature_detected!("fma")
    }
    #[cfg(not(all(target_arch = "x86_64", feature = "std")))]
    {
        // NEON always has fused multiply-add; otherwise trust the compile-time
        // feature set (covers `no_std` x86_64 and every other target).
        cfg!(target_arch = "aarch64") || cfg!(target_feature = "fma")
    }
}

/// Detect WebAssembly SIMD support (compile-time only).
#[inline]
fn detect_wasm_simd() -> bool {
    cfg!(all(target_arch = "wasm32", target_feature = "simd128"))
}

/// Glob-import the common types and helpers.
pub mod prelude {
    pub use crate::{
        Aabb3, Affine3, Backend, BoundingSphere, Containment, Frustum, Mat2, Mat3, Mat4, MathCaps,
        Plane, Quat, Ray3, RayHit, Segment3, Vec2, Vec3, Vec3A, Vec4, consts, curve, geom,
        intersect, lerp, to_degrees, to_radians, vec2, vec3, vec3a, vec4,
    };
}

#[cfg(test)]
mod tests;
#[cfg(test)]
mod m2_tests;
