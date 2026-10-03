//! x86-64 SSE2 (with optional AVX2+FMA) backend.
//!
//! SSE2 is part of the `x86_64` baseline, so these paths are always sound to
//! call on that target without a runtime probe. They mirror the
//! [`super::scalar`] reference semantics; the only numeric divergence is the
//! order of horizontal reductions (dot/length) and the fused multiply-adds in
//! the AVX2 matrix path, both of which the cross-check tests bound with a tight
//! tolerance.
#![allow(
    unsafe_code,
    reason = "core::arch SSE2/AVX2 intrinsics are unsafe fns; SSE2 is guaranteed by the \
              x86_64 baseline and every call site loads/stores in-bounds local [f32; 4] \
              buffers via the unaligned loadu/storeu intrinsics."
)]

use core::arch::x86_64::{
    __m128, _mm_add_ps, _mm_add_ss, _mm_cvtss_f32, _mm_div_ps, _mm_fmadd_ps, _mm_loadu_ps,
    _mm_movehl_ps, _mm_mul_ps, _mm_set1_ps, _mm_shuffle_ps, _mm_storeu_ps, _mm_sub_ps,
};

use crate::float::f32 as mf;

#[inline]
fn load(a: [f32; 4]) -> __m128 {
    // SAFETY: `a` is a 4-lane array (16 bytes); `_mm_loadu_ps` performs an
    // unaligned 128-bit read of exactly those bytes.
    unsafe { _mm_loadu_ps(a.as_ptr()) }
}

#[inline]
fn store(v: __m128) -> [f32; 4] {
    let mut out = [0.0f32; 4];
    // SAFETY: `out` holds 4 lanes (16 bytes); `_mm_storeu_ps` performs an
    // unaligned 128-bit write into exactly those bytes.
    unsafe { _mm_storeu_ps(out.as_mut_ptr(), v) };
    out
}

/// Horizontal sum of the four lanes of `v`, SSE2-only.
#[inline]
fn hsum(v: __m128) -> f32 {
    // SAFETY: all operands are valid 128-bit vectors; the shuffle immediates
    // are compile-time constants and the reduction stays within the register.
    unsafe {
        let hi = _mm_movehl_ps(v, v); // [v2, v3, v2, v3]
        let sum = _mm_add_ps(v, hi); // [v0+v2, v1+v3, _, _]
        let lane1 = _mm_shuffle_ps::<0b01_01_01_01>(sum, sum); // broadcast lane 1
        let s = _mm_add_ss(sum, lane1); // low lane = v0+v2 + v1+v3
        _mm_cvtss_f32(s)
    }
}

/// Component-wise `a + b`.
#[inline]
pub fn vec4_add(a: [f32; 4], b: [f32; 4]) -> [f32; 4] {
    // SAFETY: operands are valid SSE vectors produced by `load`.
    store(unsafe { _mm_add_ps(load(a), load(b)) })
}

/// Component-wise `a - b`.
#[inline]
pub fn vec4_sub(a: [f32; 4], b: [f32; 4]) -> [f32; 4] {
    // SAFETY: operands are valid SSE vectors produced by `load`.
    store(unsafe { _mm_sub_ps(load(a), load(b)) })
}

/// Component-wise `a * b`.
#[inline]
pub fn vec4_mul(a: [f32; 4], b: [f32; 4]) -> [f32; 4] {
    // SAFETY: operands are valid SSE vectors produced by `load`.
    store(unsafe { _mm_mul_ps(load(a), load(b)) })
}

/// Component-wise `a / b`.
#[inline]
pub fn vec4_div(a: [f32; 4], b: [f32; 4]) -> [f32; 4] {
    // SAFETY: operands are valid SSE vectors produced by `load`.
    store(unsafe { _mm_div_ps(load(a), load(b)) })
}

/// Broadcast scalar multiply `a * s`.
#[inline]
pub fn vec4_scale(a: [f32; 4], s: f32) -> [f32; 4] {
    // SAFETY: operands are valid SSE vectors; `_mm_set1_ps` broadcasts `s`.
    store(unsafe { _mm_mul_ps(load(a), _mm_set1_ps(s)) })
}

/// 4-lane dot product via multiply + horizontal add.
#[inline]
pub fn vec4_dot(a: [f32; 4], b: [f32; 4]) -> f32 {
    // SAFETY: operands are valid SSE vectors; `hsum` reduces across lanes.
    hsum(unsafe { _mm_mul_ps(load(a), load(b)) })
}

/// 4-lane length.
#[inline]
pub fn vec4_length(a: [f32; 4]) -> f32 {
    mf::sqrt(vec4_dot(a, a))
}

/// 4-lane normalize.
#[inline]
pub fn vec4_normalize(a: [f32; 4]) -> [f32; 4] {
    vec4_scale(a, 1.0 / vec4_length(a))
}

/// 3-lane dot product (lane 3 masked to zero before reducing).
#[inline]
pub fn vec3_dot(a: [f32; 4], b: [f32; 4]) -> f32 {
    // SAFETY: operands are valid SSE vectors; multiplying by [1,1,1,0] clears
    // lane 3 so any padding there cannot contribute to the reduction.
    let prod = unsafe { _mm_mul_ps(load(a), load(b)) };
    // SAFETY: both operands are valid 128-bit vectors.
    let masked = unsafe { _mm_mul_ps(prod, load([1.0, 1.0, 1.0, 0.0])) };
    hsum(masked)
}

/// 3-lane length.
#[inline]
pub fn vec3_length(a: [f32; 4]) -> f32 {
    mf::sqrt(vec3_dot(a, a))
}

/// 3-lane normalize; padding lane returned as `0.0`.
#[inline]
pub fn vec3_normalize(a: [f32; 4]) -> [f32; 4] {
    let inv = 1.0 / vec3_length(a);
    let r = vec4_scale(a, inv);
    [r[0], r[1], r[2], 0.0]
}

/// Column-major `4x4 * vec4` (`m.col0*v.x + .. + m.col3*v.w`).
#[inline]
pub fn mat4_mul_vec4(m: &[[f32; 4]; 4], v: [f32; 4]) -> [f32; 4] {
    // SAFETY: all operands are valid SSE vectors; `_mm_set1_ps` broadcasts each
    // scalar lane of `v`.
    unsafe {
        let c0 = load(m[0]);
        let c1 = load(m[1]);
        let c2 = load(m[2]);
        let c3 = load(m[3]);
        let acc = _mm_add_ps(
            _mm_add_ps(_mm_mul_ps(c0, _mm_set1_ps(v[0])), _mm_mul_ps(c1, _mm_set1_ps(v[1]))),
            _mm_add_ps(_mm_mul_ps(c2, _mm_set1_ps(v[2])), _mm_mul_ps(c3, _mm_set1_ps(v[3]))),
        );
        store(acc)
    }
}

/// Column-major `4x4 * 4x4` product (`a * b`).
#[inline]
pub fn mat4_mul(a: &[[f32; 4]; 4], b: &[[f32; 4]; 4]) -> [[f32; 4]; 4] {
    [
        mat4_mul_vec4(a, b[0]),
        mat4_mul_vec4(a, b[1]),
        mat4_mul_vec4(a, b[2]),
        mat4_mul_vec4(a, b[3]),
    ]
}

/// AVX2+FMA matrix product (`a * b`), used when runtime detection confirms the
/// features are present.
///
/// # Safety
/// The caller must ensure the CPU supports both `avx2` and `fma` (e.g. via
/// `std::is_x86_feature_detected!`). On `x86_64` the 128-bit lanes used here map
/// to the SSE register file, so only the FMA opcodes require the feature gate.
#[target_feature(enable = "avx2,fma")]
pub unsafe fn mat4_mul_avx2(a: &[[f32; 4]; 4], b: &[[f32; 4]; 4]) -> [[f32; 4]; 4] {
    // SAFETY: the `#[target_feature]` contract guarantees avx2+fma at this call;
    // all loads/stores target in-bounds 16-byte buffers and `_mm_set1_ps`
    // broadcasts valid scalars.
    unsafe {
        let c0 = _mm_loadu_ps(a[0].as_ptr());
        let c1 = _mm_loadu_ps(a[1].as_ptr());
        let c2 = _mm_loadu_ps(a[2].as_ptr());
        let c3 = _mm_loadu_ps(a[3].as_ptr());
        let mut out = [[0.0f32; 4]; 4];
        let mut j = 0;
        while j < 4 {
            let col = b[j];
            let mut acc = _mm_mul_ps(c0, _mm_set1_ps(col[0]));
            acc = _mm_fmadd_ps(c1, _mm_set1_ps(col[1]), acc);
            acc = _mm_fmadd_ps(c2, _mm_set1_ps(col[2]), acc);
            acc = _mm_fmadd_ps(c3, _mm_set1_ps(col[3]), acc);
            _mm_storeu_ps(out[j].as_mut_ptr(), acc);
            j += 1;
        }
        out
    }
}

/// Hamilton product `a * b` on `[x, y, z, w]` quaternions.
///
/// Decomposes the product into four scaled, sign-flipped shuffles of `b` and
/// accumulates them — algebraically identical to the scalar reference.
#[inline]
pub fn quat_mul(a: [f32; 4], b: [f32; 4]) -> [f32; 4] {
    // SAFETY: every value is a valid 128-bit vector; the shuffle immediates are
    // literal compile-time constants in 0..=255.
    unsafe {
        let va = load(a);
        let vb = load(b);

        // Shuffles of b (lane order [x, y, z, w]):
        //   reversed = [bw, bz, by, bx]
        //   roll2    = [bz, bw, bx, by]
        //   swap     = [by, bx, bw, bz]
        let reversed = _mm_shuffle_ps::<0b00_01_10_11>(vb, vb);
        let roll2 = _mm_shuffle_ps::<0b01_00_11_10>(vb, vb);
        let swap = _mm_shuffle_ps::<0b10_11_00_01>(vb, vb);

        // Apply sign masks to form the per-term b vectors:
        //   b1 = [ bw, -bz,  by, -bx]
        //   b2 = [ bz,  bw, -bx, -by]
        //   b3 = [-by,  bx,  bw, -bz]
        let b1 = _mm_mul_ps(reversed, load([1.0, -1.0, 1.0, -1.0]));
        let b2 = _mm_mul_ps(roll2, load([1.0, 1.0, -1.0, -1.0]));
        let b3 = _mm_mul_ps(swap, load([-1.0, 1.0, 1.0, -1.0]));

        // Broadcast each lane of a.
        let ax = _mm_shuffle_ps::<0b00_00_00_00>(va, va);
        let ay = _mm_shuffle_ps::<0b01_01_01_01>(va, va);
        let az = _mm_shuffle_ps::<0b10_10_10_10>(va, va);
        let aw = _mm_shuffle_ps::<0b11_11_11_11>(va, va);

        // result = aw*b + ax*b1 + ay*b2 + az*b3
        let acc = _mm_add_ps(
            _mm_add_ps(_mm_mul_ps(aw, vb), _mm_mul_ps(ax, b1)),
            _mm_add_ps(_mm_mul_ps(ay, b2), _mm_mul_ps(az, b3)),
        );
        store(acc)
    }
}

/// Rotate `v` (`[x, y, z, 0]`) by unit quaternion `q` via `q * v * q^-1`.
#[inline]
pub fn quat_mul_vec3(q: [f32; 4], v: [f32; 4]) -> [f32; 4] {
    let vq = [v[0], v[1], v[2], 0.0];
    let conj = [-q[0], -q[1], -q[2], q[3]];
    let r = quat_mul(quat_mul(q, vq), conj);
    [r[0], r[1], r[2], 0.0]
}
