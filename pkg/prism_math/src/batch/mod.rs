//! Batched (SIMD-accelerated) transforms (M5).
//!
//! These operate on whole slices of vectors, which lets the hot inner loop use
//! wide SIMD. On `x86_64`, [`transform_vec4`] uses an AVX2+FMA kernel
//! (processing two 4-wide vectors per 256-bit register) when runtime feature
//! detection confirms the CPU supports it; otherwise, and on every other
//! target, a portable scalar reference runs. The two paths are checked for
//! bit-for-bit-close parity by the test suite.
//!
//! The scalar reference is defined once here and is the behavioural ground
//! truth the AVX2 kernel must match within a tight tolerance (it differs only
//! in FMA rounding).

#[cfg(target_arch = "x86_64")]
mod avx2;

use crate::{Mat4, Vec3, Vec4};

/// Transform a single homogeneous vector by a column-major matrix.
///
/// This is the scalar reference used by both the fallback loop and the parity
/// tests: `out = c0*v.x + c1*v.y + c2*v.z + c3*v.w`.
#[inline]
pub fn transform_vec4_one(m: &Mat4, v: Vec4) -> Vec4 {
    m.x_axis * v.x + m.y_axis * v.y + m.z_axis * v.z + m.w_axis * v.w
}

/// Batch-transform `src` homogeneous vectors by `m`, writing to `dst`.
///
/// # Panics
/// Panics if `src.len() != dst.len()`.
#[inline]
pub fn transform_vec4(m: &Mat4, src: &[Vec4], dst: &mut [Vec4]) {
    assert_eq!(
        src.len(),
        dst.len(),
        "transform_vec4: src/dst length mismatch"
    );

    #[cfg(target_arch = "x86_64")]
    {
        if avx2::available() {
            // SAFETY: `available()` confirmed AVX2+FMA via runtime (std) or
            // compile-time (no_std) detection, which is this kernel's contract.
            unsafe { avx2::transform_vec4(m, src, dst) };
            return;
        }
    }

    transform_vec4_scalar(m, src, dst);
}

/// Portable scalar batch transform (the reference path).
#[inline]
pub(crate) fn transform_vec4_scalar(m: &Mat4, src: &[Vec4], dst: &mut [Vec4]) {
    for (o, &v) in dst.iter_mut().zip(src.iter()) {
        *o = transform_vec4_one(m, v);
    }
}

/// Batch-transform `src` points (implicit `w = 1`, perspective divide applied),
/// matching [`Mat4::transform_point3`].
///
/// # Panics
/// Panics if `src.len() != dst.len()`.
#[inline]
pub fn transform_points3(m: &Mat4, src: &[Vec3], dst: &mut [Vec3]) {
    assert_eq!(
        src.len(),
        dst.len(),
        "transform_points3: src/dst length mismatch"
    );
    for (o, &p) in dst.iter_mut().zip(src.iter()) {
        *o = m.transform_point3(p);
    }
}

/// Batch-transform `src` directions (implicit `w = 0`), matching
/// [`Mat4::transform_vector3`].
///
/// # Panics
/// Panics if `src.len() != dst.len()`.
#[inline]
pub fn transform_vectors3(m: &Mat4, src: &[Vec3], dst: &mut [Vec3]) {
    assert_eq!(
        src.len(),
        dst.len(),
        "transform_vectors3: src/dst length mismatch"
    );
    for (o, &v) in dst.iter_mut().zip(src.iter()) {
        *o = m.transform_vector3(v);
    }
}

/// Batch-normalize `src` vectors, writing unit-length results to `dst`.
///
/// # Panics
/// Panics if `src.len() != dst.len()`.
#[inline]
pub fn normalize3(src: &[Vec3], dst: &mut [Vec3]) {
    assert_eq!(src.len(), dst.len(), "normalize3: src/dst length mismatch");
    for (o, &v) in dst.iter_mut().zip(src.iter()) {
        *o = v.normalize();
    }
}
