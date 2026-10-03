//! SIMD backend selection and dispatch.
//!
//! The hot core operations (`Vec4`/`Vec3A` arithmetic, `Mat4` products,
//! `Quat` products) are implemented once per instruction set and routed here
//! through thin dispatch functions. Every backend operates on raw `[f32; 4]`
//! lanes so it is decoupled from the public facade types:
//!
//! - [`scalar`] is the behavioural ground truth and the fallback for targets
//!   without a dedicated backend.
//! - [`neon`] is used on `aarch64` (NEON is mandatory there).
//! - [`sse2`] is used on `x86_64` (SSE2 is part of the baseline), and the
//!   `mat4_mul` dispatcher opportunistically upgrades to an AVX2+FMA path when
//!   runtime detection (or compile-time `target_feature` under `no_std`)
//!   confirms the features.
//!
//! The compile-time-selected backend is aliased as `imp`. The SIMD backends are
//! algebraically identical to [`scalar`]; they diverge only in horizontal
//! reduction order and fused multiply-adds, which the cross-check tests bound
//! within a tight tolerance.

pub(crate) mod scalar;

#[cfg(target_arch = "aarch64")]
pub(crate) mod neon;

#[cfg(target_arch = "x86_64")]
pub(crate) mod sse2;

#[cfg(target_arch = "aarch64")]
use neon as imp;

#[cfg(target_arch = "x86_64")]
use sse2 as imp;

#[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
use scalar as imp;

/// The concrete SIMD backend selected for the current build/CPU.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Backend {
    /// Portable scalar reference backend.
    Scalar,
    /// x86-64 SSE2 backend.
    Sse2,
    /// x86-64 AVX2 (+FMA) backend.
    Avx2,
    /// `AArch64` NEON backend.
    Neon,
}

/// Report which backend the dispatch functions are using on this CPU.
#[inline]
pub fn active() -> Backend {
    active_impl()
}

#[cfg(target_arch = "aarch64")]
#[inline]
fn active_impl() -> Backend {
    Backend::Neon
}

#[cfg(target_arch = "x86_64")]
#[inline]
fn active_impl() -> Backend {
    x86_active()
}

#[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
#[inline]
fn active_impl() -> Backend {
    Backend::Scalar
}

#[cfg(all(target_arch = "x86_64", feature = "std"))]
#[inline]
fn x86_active() -> Backend {
    if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
        Backend::Avx2
    } else {
        Backend::Sse2
    }
}

#[cfg(all(target_arch = "x86_64", not(feature = "std")))]
#[inline]
fn x86_active() -> Backend {
    if cfg!(target_feature = "avx2") && cfg!(target_feature = "fma") {
        Backend::Avx2
    } else {
        Backend::Sse2
    }
}

/// Component-wise `a + b`.
#[inline]
pub fn vec4_add(a: [f32; 4], b: [f32; 4]) -> [f32; 4] {
    imp::vec4_add(a, b)
}

/// Component-wise `a - b`.
#[inline]
pub fn vec4_sub(a: [f32; 4], b: [f32; 4]) -> [f32; 4] {
    imp::vec4_sub(a, b)
}

/// Component-wise `a * b`.
#[inline]
pub fn vec4_mul(a: [f32; 4], b: [f32; 4]) -> [f32; 4] {
    imp::vec4_mul(a, b)
}

/// Component-wise `a / b`.
#[inline]
pub fn vec4_div(a: [f32; 4], b: [f32; 4]) -> [f32; 4] {
    imp::vec4_div(a, b)
}

/// Broadcast scalar multiply `a * s`.
#[inline]
pub fn vec4_scale(a: [f32; 4], s: f32) -> [f32; 4] {
    imp::vec4_scale(a, s)
}

/// 4-lane dot product.
#[inline]
pub fn vec4_dot(a: [f32; 4], b: [f32; 4]) -> f32 {
    imp::vec4_dot(a, b)
}

/// 4-lane Euclidean length.
#[inline]
pub fn vec4_length(a: [f32; 4]) -> f32 {
    imp::vec4_length(a)
}

/// 4-lane normalize (`a / |a|`).
#[inline]
pub fn vec4_normalize(a: [f32; 4]) -> [f32; 4] {
    imp::vec4_normalize(a)
}

/// 3-lane dot product (lane 3 ignored).
#[inline]
pub fn vec3_dot(a: [f32; 4], b: [f32; 4]) -> f32 {
    imp::vec3_dot(a, b)
}

/// 3-lane Euclidean length (lane 3 ignored).
#[inline]
pub fn vec3_length(a: [f32; 4]) -> f32 {
    imp::vec3_length(a)
}

/// 3-lane normalize; the padding lane is returned as `0.0`.
#[inline]
pub fn vec3_normalize(a: [f32; 4]) -> [f32; 4] {
    imp::vec3_normalize(a)
}

/// Column-major `4x4 * vec4`.
#[inline]
pub fn mat4_mul_vec4(m: &[[f32; 4]; 4], v: [f32; 4]) -> [f32; 4] {
    imp::mat4_mul_vec4(m, v)
}

/// Column-major `4x4 * 4x4` product (`a * b`).
///
/// On `x86_64` with `std` this upgrades to an AVX2+FMA kernel when the running
/// CPU supports both features; otherwise it uses the compile-time backend.
#[inline]
pub fn mat4_mul(a: &[[f32; 4]; 4], b: &[[f32; 4]; 4]) -> [[f32; 4]; 4] {
    #[cfg(all(target_arch = "x86_64", feature = "std"))]
    {
        if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
            // SAFETY: guarded by the runtime feature detection just above, which
            // is exactly the precondition documented on `mat4_mul_avx2`.
            return unsafe { sse2::mat4_mul_avx2(a, b) };
        }
    }
    imp::mat4_mul(a, b)
}

/// Hamilton product `a * b` on `[x, y, z, w]` quaternions.
#[inline]
pub fn quat_mul(a: [f32; 4], b: [f32; 4]) -> [f32; 4] {
    imp::quat_mul(a, b)
}

/// Rotate `v` (`[x, y, z, 0]`) by unit quaternion `q` (`[x, y, z, w]`).
#[inline]
pub fn quat_mul_vec3(q: [f32; 4], v: [f32; 4]) -> [f32; 4] {
    imp::quat_mul_vec3(q, v)
}

#[cfg(test)]
mod tests;
