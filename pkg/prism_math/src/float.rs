//! Scalar transcendental helpers, routed through `libm` so the core facade is
//! `no_std` and deterministic across `std`/`no_std` builds.
//!
//! M0 ships a scalar reference backend only; SIMD backends (M1) will mirror
//! these semantics within documented tolerances.

#![allow(missing_docs, reason = "scalar helpers are documented at module level; per-fn docs land with M1+ API stabilization")]
// Some helpers are the reference backend for later milestones (M1+ SIMD,
// M2 curves) and are not all exercised yet by the M0 facade.
#![allow(dead_code, reason = "reference backend helpers for later milestones (M1+ SIMD, M2 curves) not yet exercised by the facade")]

/// `f32` scalar math used by the facade. Kept in one place so the SIMD
/// backends can be audited against a single reference.
pub mod f32 {
    #[inline]
    pub fn sqrt(x: f32) -> f32 {
        libm::sqrtf(x)
    }
    #[inline]
    pub fn sin(x: f32) -> f32 {
        libm::sinf(x)
    }
    #[inline]
    pub fn cos(x: f32) -> f32 {
        libm::cosf(x)
    }
    #[inline]
    pub fn tan(x: f32) -> f32 {
        libm::tanf(x)
    }
    #[inline]
    pub fn asin(x: f32) -> f32 {
        libm::asinf(x)
    }
    #[inline]
    pub fn acos(x: f32) -> f32 {
        libm::acosf(x)
    }
    #[inline]
    pub fn atan2(y: f32, x: f32) -> f32 {
        libm::atan2f(y, x)
    }
    #[inline]
    pub fn abs(x: f32) -> f32 {
        libm::fabsf(x)
    }
    #[inline]
    pub fn copysign(x: f32, sign: f32) -> f32 {
        libm::copysignf(x, sign)
    }
    #[inline]
    pub fn floor(x: f32) -> f32 {
        libm::floorf(x)
    }
    #[inline]
    pub fn ceil(x: f32) -> f32 {
        libm::ceilf(x)
    }
    #[inline]
    pub fn round(x: f32) -> f32 {
        libm::roundf(x)
    }
    #[inline]
    pub fn powf(x: f32, y: f32) -> f32 {
        libm::powf(x, y)
    }

    /// `sin` and `cos` evaluated together (M0 computes them separately).
    #[inline]
    pub fn sin_cos(x: f32) -> (f32, f32) {
        (sin(x), cos(x))
    }
}
