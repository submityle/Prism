//! SIMD-accelerated bulk kernels over contiguous `f32` component columns
//! (design §7 "脏块访问器 / SIMD 迭代", §17 "列对齐 + SIMD").
//!
//! Hot component columns are Structure-of-Arrays and contiguous (one table per
//! archetype), so a whole column of a numeric component (`[f32]`, obtained via
//! [`Column::as_slice`](crate::storage::Column::as_slice) /
//! [`Column::as_mut_slice`](crate::storage::Column::as_mut_slice)) can be driven
//! through vector instructions in one linear pass — the integration/scale/axpy
//! kernels that transform-propagation, particle, and skinning passes lean on.
//!
//! Backend selection mirrors `prism_math`: one kernel per instruction set,
//! routed through [`imp`] at compile time.
//! - [`scalar`] is the behavioural ground truth and the portable fallback.
//! - [`sse2`] runs on `x86_64` (SSE2 is in the baseline).
//! - [`neon`] runs on `aarch64` (NEON is mandatory there).
//!
//! The element-wise kernels ([`add_assign`], [`scale_assign`], [`axpy_assign`])
//! perform the *same* IEEE-754 single operations per lane as [`scalar`] — no
//! fused multiply-add and no lane reordering — so they are **bit-for-bit**
//! identical to the scalar reference. [`sum`] is a horizontal reduction whose
//! lane-pairing order differs, so it is only equal to the scalar reference
//! within a tight tolerance. The golden dual-run tests enforce both contracts.
//!
//! Enabled by the `simd` feature; no `portable_simd`/nightly is used.

/// The concrete SIMD backend selected for the current build.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Backend {
    /// Portable scalar reference backend (and fallback target).
    Scalar,
    /// `x86_64` SSE2 backend.
    Sse2,
    /// `AArch64` NEON backend.
    Neon,
}

/// Report which backend the kernels in this module compile to on this target.
#[inline]
pub fn active() -> Backend {
    #[cfg(target_arch = "x86_64")]
    {
        Backend::Sse2
    }
    #[cfg(target_arch = "aarch64")]
    {
        Backend::Neon
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        Backend::Scalar
    }
}

/// Element-wise `dst[i] += src[i]` over the whole column.
///
/// # Panics
/// Panics if `dst.len() != src.len()`.
#[inline]
pub fn add_assign(dst: &mut [f32], src: &[f32]) {
    assert_eq!(dst.len(), src.len(), "add_assign length mismatch");
    imp::add_assign(dst, src);
}

/// Element-wise `dst[i] *= k` over the whole column.
#[inline]
pub fn scale_assign(dst: &mut [f32], k: f32) {
    imp::scale_assign(dst, k);
}

/// Fused-free axpy: `dst[i] += k * src[i]` over the whole column.
///
/// Computed as a separate multiply then add (no FMA) so the result is
/// bit-identical to the scalar reference.
///
/// # Panics
/// Panics if `dst.len() != src.len()`.
#[inline]
pub fn axpy_assign(dst: &mut [f32], src: &[f32], k: f32) {
    assert_eq!(dst.len(), src.len(), "axpy_assign length mismatch");
    imp::axpy_assign(dst, src, k);
}

/// Horizontal sum of the column.
///
/// The SIMD backends pair lanes in a different order than a left-to-right
/// scalar fold, so the result matches [`scalar::sum`] only within a tight
/// tolerance (see the module docs).
#[inline]
pub fn sum(src: &[f32]) -> f32 {
    imp::sum(src)
}

#[cfg(target_arch = "x86_64")]
use sse2 as imp;

#[cfg(target_arch = "aarch64")]
use neon as imp;

#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
use scalar as imp;

/// Portable scalar reference kernels — the behavioural ground truth.
pub mod scalar {
    /// `dst[i] += src[i]` (caller guarantees equal lengths).
    #[inline]
    pub fn add_assign(dst: &mut [f32], src: &[f32]) {
        for (d, s) in dst.iter_mut().zip(src.iter()) {
            *d += *s;
        }
    }

    /// `dst[i] *= k`.
    #[inline]
    pub fn scale_assign(dst: &mut [f32], k: f32) {
        for d in dst.iter_mut() {
            *d *= k;
        }
    }

    /// `dst[i] += k * src[i]` (separate multiply then add, caller guarantees
    /// equal lengths).
    #[inline]
    pub fn axpy_assign(dst: &mut [f32], src: &[f32], k: f32) {
        for (d, s) in dst.iter_mut().zip(src.iter()) {
            *d += k * *s;
        }
    }

    /// Left-to-right fold sum.
    #[inline]
    pub fn sum(src: &[f32]) -> f32 {
        let mut acc = 0.0f32;
        for &s in src {
            acc += s;
        }
        acc
    }
}

#[cfg(target_arch = "x86_64")]
mod sse2 {
    use core::arch::x86_64::{
        __m128, _mm_add_ps, _mm_add_ss, _mm_cvtss_f32, _mm_loadu_ps, _mm_movehl_ps, _mm_mul_ps,
        _mm_set1_ps, _mm_setzero_ps, _mm_shuffle_ps, _mm_storeu_ps,
    };

    const LANES: usize = 4;

    #[inline]
    fn load(p: &[f32]) -> __m128 {
        // SAFETY: `p` has at least `LANES` elements at every call site (the
        // chunked loops below), so this unaligned load reads 4 in-bounds f32.
        unsafe { _mm_loadu_ps(p.as_ptr()) }
    }

    #[inline]
    fn store(p: &mut [f32], v: __m128) {
        // SAFETY: `p` has at least `LANES` elements at every call site, so this
        // unaligned store writes exactly those 4 f32.
        unsafe { _mm_storeu_ps(p.as_mut_ptr(), v) };
    }

    #[inline]
    pub fn add_assign(dst: &mut [f32], src: &[f32]) {
        let n = dst.len();
        let mut i = 0;
        while i + LANES <= n {
            let d = load(&dst[i..]);
            let s = load(&src[i..]);
            // SAFETY: register-only SSE2 add; SSE2 is baseline on x86_64.
            let r = unsafe { _mm_add_ps(d, s) };
            store(&mut dst[i..], r);
            i += LANES;
        }
        super::scalar::add_assign(&mut dst[i..], &src[i..]);
    }

    #[inline]
    pub fn scale_assign(dst: &mut [f32], k: f32) {
        let n = dst.len();
        // SAFETY: broadcast of a scalar into all 4 lanes; no memory access.
        let kv = unsafe { _mm_set1_ps(k) };
        let mut i = 0;
        while i + LANES <= n {
            let d = load(&dst[i..]);
            // SAFETY: register-only SSE2 multiply; SSE2 is baseline on x86_64.
            let r = unsafe { _mm_mul_ps(d, kv) };
            store(&mut dst[i..], r);
            i += LANES;
        }
        super::scalar::scale_assign(&mut dst[i..], k);
    }

    #[inline]
    pub fn axpy_assign(dst: &mut [f32], src: &[f32], k: f32) {
        let n = dst.len();
        // SAFETY: broadcast of a scalar into all 4 lanes; no memory access.
        let kv = unsafe { _mm_set1_ps(k) };
        let mut i = 0;
        while i + LANES <= n {
            let d = load(&dst[i..]);
            let s = load(&src[i..]);
            // Separate multiply then add (no FMA) to match scalar bit-for-bit.
            // SAFETY: register-only SSE2 multiply then add; SSE2 is baseline.
            let r = unsafe { _mm_add_ps(d, _mm_mul_ps(kv, s)) };
            store(&mut dst[i..], r);
            i += LANES;
        }
        super::scalar::axpy_assign(&mut dst[i..], &src[i..], k);
    }

    #[inline]
    pub fn sum(src: &[f32]) -> f32 {
        let n = src.len();
        // SAFETY: zeroed accumulator register.
        let mut acc = unsafe { _mm_setzero_ps() };
        let mut i = 0;
        while i + LANES <= n {
            let v = load(&src[i..]);
            // SAFETY: register-only SSE2 add.
            acc = unsafe { _mm_add_ps(acc, v) };
            i += LANES;
        }
        // Horizontal reduce the 4-lane accumulator.
        // SAFETY: all operands are initialized 128-bit registers.
        let folded = unsafe {
            let hi = _mm_movehl_ps(acc, acc);
            let lo = _mm_add_ps(acc, hi);
            let shuf = _mm_shuffle_ps(lo, lo, 0b01);
            _mm_cvtss_f32(_mm_add_ss(lo, shuf))
        };
        folded + super::scalar::sum(&src[i..])
    }
}

#[cfg(target_arch = "aarch64")]
mod neon {
    use core::arch::aarch64::{
        float32x4_t, vaddq_f32, vaddvq_f32, vdupq_n_f32, vld1q_f32, vmulq_f32, vst1q_f32,
    };

    const LANES: usize = 4;

    #[inline]
    fn load(p: &[f32]) -> float32x4_t {
        // SAFETY: `p` has at least `LANES` elements at every call site; the
        // unaligned NEON load reads exactly 4 in-bounds f32.
        unsafe { vld1q_f32(p.as_ptr()) }
    }

    #[inline]
    fn store(p: &mut [f32], v: float32x4_t) {
        // SAFETY: `p` has at least `LANES` elements at every call site.
        unsafe { vst1q_f32(p.as_mut_ptr(), v) };
    }

    #[inline]
    pub fn add_assign(dst: &mut [f32], src: &[f32]) {
        let n = dst.len();
        let mut i = 0;
        while i + LANES <= n {
            let d = load(&dst[i..]);
            let s = load(&src[i..]);
            // SAFETY: register-only NEON add on loaded lanes.
            store(&mut dst[i..], unsafe { vaddq_f32(d, s) });
            i += LANES;
        }
        super::scalar::add_assign(&mut dst[i..], &src[i..]);
    }

    #[inline]
    pub fn scale_assign(dst: &mut [f32], k: f32) {
        let n = dst.len();
        // SAFETY: broadcast scalar into all lanes; no memory access.
        let kv = unsafe { vdupq_n_f32(k) };
        let mut i = 0;
        while i + LANES <= n {
            let d = load(&dst[i..]);
            // SAFETY: register-only NEON multiply.
            store(&mut dst[i..], unsafe { vmulq_f32(d, kv) });
            i += LANES;
        }
        super::scalar::scale_assign(&mut dst[i..], k);
    }

    #[inline]
    pub fn axpy_assign(dst: &mut [f32], src: &[f32], k: f32) {
        let n = dst.len();
        // SAFETY: broadcast scalar into all lanes; no memory access.
        let kv = unsafe { vdupq_n_f32(k) };
        let mut i = 0;
        while i + LANES <= n {
            let d = load(&dst[i..]);
            let s = load(&src[i..]);
            // Separate multiply then add (no FMA) to match scalar bit-for-bit.
            // SAFETY: register-only NEON multiply then add.
            let prod = unsafe { vmulq_f32(kv, s) };
            // SAFETY: register-only NEON add.
            store(&mut dst[i..], unsafe { vaddq_f32(d, prod) });
            i += LANES;
        }
        super::scalar::axpy_assign(&mut dst[i..], &src[i..], k);
    }

    #[inline]
    pub fn sum(src: &[f32]) -> f32 {
        let n = src.len();
        // SAFETY: broadcast zero into all lanes.
        let mut acc = unsafe { vdupq_n_f32(0.0) };
        let mut i = 0;
        while i + LANES <= n {
            // SAFETY: register-only NEON add on loaded lanes.
            acc = unsafe { vaddq_f32(acc, load(&src[i..])) };
            i += LANES;
        }
        // SAFETY: horizontal add across the 4 lanes.
        let folded = unsafe { vaddvq_f32(acc) };
        folded + super::scalar::sum(&src[i..])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec::Vec;

    // A small deterministic LCG so the dual-run uses varied-but-reproducible
    // inputs without pulling in an RNG dependency.
    fn lcg(seed: &mut u64) -> f32 {
        *seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let bits = (*seed >> 33) as u32;
        // Map to a signed value in roughly [-128, 128).
        (bits as f32 / u32::MAX as f32) * 256.0 - 128.0
    }

    fn gen_vec(n: usize, seed: &mut u64) -> Vec<f32> {
        (0..n).map(|_| lcg(seed)).collect()
    }

    #[test]
    fn add_assign_matches_scalar_bit_for_bit() {
        let mut seed = 0x1234_5678u64;
        // Cover tails: lengths that are and are not multiples of the lane width.
        for n in [0usize, 1, 3, 4, 5, 7, 8, 15, 16, 17, 63, 64, 65, 1000] {
            let a = gen_vec(n, &mut seed);
            let b = gen_vec(n, &mut seed);
            let mut simd_dst = a.clone();
            let mut scalar_dst = a.clone();
            add_assign(&mut simd_dst, &b);
            scalar::add_assign(&mut scalar_dst, &b);
            assert_eq!(
                simd_dst.iter().map(|f| f.to_bits()).collect::<Vec<_>>(),
                scalar_dst.iter().map(|f| f.to_bits()).collect::<Vec<_>>(),
                "add_assign diverged at n={n}"
            );
        }
    }

    #[test]
    fn scale_assign_matches_scalar_bit_for_bit() {
        let mut seed = 0xdead_beefu64;
        for n in [0usize, 1, 3, 4, 7, 8, 17, 64, 65, 999] {
            let a = gen_vec(n, &mut seed);
            let k = lcg(&mut seed);
            let mut simd_dst = a.clone();
            let mut scalar_dst = a.clone();
            scale_assign(&mut simd_dst, k);
            scalar::scale_assign(&mut scalar_dst, k);
            assert_eq!(
                simd_dst.iter().map(|f| f.to_bits()).collect::<Vec<_>>(),
                scalar_dst.iter().map(|f| f.to_bits()).collect::<Vec<_>>(),
                "scale_assign diverged at n={n}"
            );
        }
    }

    #[test]
    fn axpy_assign_matches_scalar_bit_for_bit() {
        let mut seed = 0x0bad_f00du64;
        for n in [0usize, 1, 2, 3, 4, 5, 8, 9, 16, 31, 32, 33, 500] {
            let a = gen_vec(n, &mut seed);
            let b = gen_vec(n, &mut seed);
            let k = lcg(&mut seed);
            let mut simd_dst = a.clone();
            let mut scalar_dst = a.clone();
            axpy_assign(&mut simd_dst, &b, k);
            scalar::axpy_assign(&mut scalar_dst, &b, k);
            assert_eq!(
                simd_dst.iter().map(|f| f.to_bits()).collect::<Vec<_>>(),
                scalar_dst.iter().map(|f| f.to_bits()).collect::<Vec<_>>(),
                "axpy_assign diverged at n={n}"
            );
        }
    }

    #[test]
    fn active_reports_target_backend() {
        let backend = active();
        #[cfg(target_arch = "x86_64")]
        assert_eq!(backend, Backend::Sse2);
        #[cfg(target_arch = "aarch64")]
        assert_eq!(backend, Backend::Neon);
        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        assert_eq!(backend, Backend::Scalar);
    }

    #[test]
    fn sum_matches_scalar_within_tolerance() {
        let mut seed = 0xfeed_face_u64;
        for n in [0usize, 1, 4, 7, 8, 63, 64, 1000] {
            let a = gen_vec(n, &mut seed);
            let got = sum(&a);
            let want = scalar::sum(&a);
            let tol = 1e-2 * (n as f32).max(1.0);
            assert!(
                (got - want).abs() <= tol,
                "sum diverged at n={n}: {got} vs {want}"
            );
        }
    }
}
