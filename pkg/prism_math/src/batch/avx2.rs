//! x86-64 AVX2+FMA batch transform kernel.
//!
//! Processes two `Vec4`s per iteration in a single 256-bit register. The matrix
//! columns are broadcast into both 128-bit halves, and each pair of inputs has
//! its `x/y/z/w` lanes splatted into the matching half, so one FMA chain
//! computes `c0*x + c1*y + c2*z + c3*w` for both vectors at once.
//!
//! This mirrors [`super::transform_vec4_scalar`]; the only numeric difference is
//! the fused multiply-adds, which the parity tests bound with a tight tolerance.
#![allow(
    unsafe_code,
    reason = "core::arch AVX2/FMA intrinsics are unsafe fns; every entry point is \
              guarded by a runtime (std) or compile-time (no_std) avx2+fma check, and \
              all loads/stores target in-bounds 16-/32-byte buffers via unaligned ops."
)]

use core::arch::x86_64::{
    __m128, __m256, _mm256_fmadd_ps, _mm256_mul_ps, _mm256_set_m128,
    _mm256_storeu_ps, _mm_loadu_ps, _mm_shuffle_ps,
};

use crate::{Mat4, Vec4};

/// Report whether the AVX2+FMA kernel may be called on this CPU.
#[inline]
pub(crate) fn available() -> bool {
    #[cfg(feature = "std")]
    {
        std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma")
    }
    #[cfg(not(feature = "std"))]
    {
        cfg!(target_feature = "avx2") && cfg!(target_feature = "fma")
    }
}

/// Shuffle immediate that splats a single lane across all four of a `__m128`.
const fn splat_imm(lane: i32) -> i32 {
    lane | (lane << 2) | (lane << 4) | (lane << 6)
}

/// Broadcast one `__m128` into both halves of a `__m256`.
///
/// # Safety
/// Requires `avx` (implied by the caller's `avx2`); `c` must be a valid vector.
#[target_feature(enable = "avx2,fma")]
unsafe fn dup(c: __m128) -> __m256 {
    // SAFETY: `_mm256_set_m128` packs two copies of the valid vector `c`.
    unsafe { _mm256_set_m128(c, c) }
}

/// Build `[a.LANE ×4 | b.LANE ×4]` across a `__m256`.
///
/// # Safety
/// Requires `avx` (implied by the caller's `avx2`); `a`/`b` must be valid.
#[target_feature(enable = "avx2,fma")]
unsafe fn splat_pair<const LANE: i32>(a: __m128, b: __m128) -> __m256 {
    // SAFETY: shuffle immediates are compile-time constants in range; `a`/`b`
    // are valid 128-bit vectors and `_mm256_set_m128` packs the two splats.
    unsafe {
        let sa = _mm_shuffle_ps::<{ splat_imm(LANE) }>(a, a);
        let sb = _mm_shuffle_ps::<{ splat_imm(LANE) }>(b, b);
        _mm256_set_m128(sb, sa)
    }
}

/// AVX2+FMA batch transform. See module docs.
///
/// # Safety
/// The caller must ensure the CPU supports both `avx2` and `fma` (checked via
/// [`available`]) and that `src.len() == dst.len()`.
#[target_feature(enable = "avx2,fma")]
pub(crate) unsafe fn transform_vec4(m: &Mat4, src: &[Vec4], dst: &mut [Vec4]) {
    // SAFETY: the `#[target_feature]` contract guarantees avx2+fma here; all
    // loads/stores below address in-bounds buffers via unaligned intrinsics,
    // and the helpers share this function's feature set.
    unsafe {
        let c0 = dup(_mm_loadu_ps(m.x_axis.to_array().as_ptr()));
        let c1 = dup(_mm_loadu_ps(m.y_axis.to_array().as_ptr()));
        let c2 = dup(_mm_loadu_ps(m.z_axis.to_array().as_ptr()));
        let c3 = dup(_mm_loadu_ps(m.w_axis.to_array().as_ptr()));

        let n = src.len();
        let pairs = n / 2;

        for p in 0..pairs {
            let a = _mm_loadu_ps(src[2 * p].to_array().as_ptr());
            let b = _mm_loadu_ps(src[2 * p + 1].to_array().as_ptr());

            let vx = splat_pair::<0>(a, b);
            let vy = splat_pair::<1>(a, b);
            let vz = splat_pair::<2>(a, b);
            let vw = splat_pair::<3>(a, b);

            let mut acc = _mm256_mul_ps(c0, vx);
            acc = _mm256_fmadd_ps(c1, vy, acc);
            acc = _mm256_fmadd_ps(c2, vz, acc);
            acc = _mm256_fmadd_ps(c3, vw, acc);

            let mut out = [0.0f32; 8];
            _mm256_storeu_ps(out.as_mut_ptr(), acc);
            dst[2 * p] = Vec4::new(out[0], out[1], out[2], out[3]);
            dst[2 * p + 1] = Vec4::new(out[4], out[5], out[6], out[7]);
        }

        // Odd tail: one leftover vector via the scalar reference.
        if n & 1 == 1 {
            dst[n - 1] = super::transform_vec4_one(m, src[n - 1]);
        }
    }
}
