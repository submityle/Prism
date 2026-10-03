//! `AArch64` NEON backend.
//!
//! NEON is mandatory on `aarch64`, so these paths are always available without
//! a runtime probe. They mirror the [`super::scalar`] reference semantics; the
//! only numeric divergence is the order of horizontal reductions (dot/length),
//! which the cross-check tests bound with a tight tolerance.
#![allow(
    unsafe_code,
    reason = "core::arch NEON intrinsics are unsafe fns; every call sites loads/stores \
              in-bounds local [f32; 4] buffers (4-byte aligned, enough for vld1q/vst1q)."
)]

use core::arch::aarch64::{
    float32x4_t, vaddq_f32, vaddvq_f32, vdivq_f32, vextq_f32, vfmaq_laneq_f32, vld1q_f32,
    vmulq_f32, vmulq_laneq_f32, vmulq_n_f32, vrev64q_f32, vsetq_lane_f32, vst1q_f32, vsubq_f32,
};

use crate::float::f32 as mf;

#[inline]
fn load(a: [f32; 4]) -> float32x4_t {
    // SAFETY: `a` is a 4-lane array, so the 128-bit load reads exactly its 16
    // bytes; vld1q only requires element (4-byte) alignment.
    unsafe { vld1q_f32(a.as_ptr()) }
}

#[inline]
fn store(v: float32x4_t) -> [f32; 4] {
    let mut out = [0.0f32; 4];
    // SAFETY: `out` has room for 4 lanes; vst1q writes exactly 16 bytes.
    unsafe { vst1q_f32(out.as_mut_ptr(), v) }
    out
}

/// Component-wise `a + b`.
#[inline]
pub fn vec4_add(a: [f32; 4], b: [f32; 4]) -> [f32; 4] {
    // SAFETY: operands are valid NEON vectors produced by `load`.
    store(unsafe { vaddq_f32(load(a), load(b)) })
}

/// Component-wise `a - b`.
#[inline]
pub fn vec4_sub(a: [f32; 4], b: [f32; 4]) -> [f32; 4] {
    // SAFETY: operands are valid NEON vectors produced by `load`.
    store(unsafe { vsubq_f32(load(a), load(b)) })
}

/// Component-wise `a * b`.
#[inline]
pub fn vec4_mul(a: [f32; 4], b: [f32; 4]) -> [f32; 4] {
    // SAFETY: operands are valid NEON vectors produced by `load`.
    store(unsafe { vmulq_f32(load(a), load(b)) })
}

/// Component-wise `a / b`.
#[inline]
pub fn vec4_div(a: [f32; 4], b: [f32; 4]) -> [f32; 4] {
    // SAFETY: operands are valid NEON vectors produced by `load`.
    store(unsafe { vdivq_f32(load(a), load(b)) })
}

/// Broadcast scalar multiply `a * s`.
#[inline]
pub fn vec4_scale(a: [f32; 4], s: f32) -> [f32; 4] {
    // SAFETY: operand is a valid NEON vector produced by `load`.
    store(unsafe { vmulq_n_f32(load(a), s) })
}

/// 4-lane dot product via NEON multiply + horizontal add.
#[inline]
pub fn vec4_dot(a: [f32; 4], b: [f32; 4]) -> f32 {
    // SAFETY: operands are valid NEON vectors; vaddvq reduces across lanes.
    unsafe { vaddvq_f32(vmulq_f32(load(a), load(b))) }
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

/// 3-lane dot product (lane 3 forced to zero before reducing).
#[inline]
pub fn vec3_dot(a: [f32; 4], b: [f32; 4]) -> f32 {
    // SAFETY: operands are valid NEON vectors; we clear lane 3 of the product
    // so any padding in lane 3 cannot contribute to the reduction.
    unsafe {
        let prod = vmulq_f32(load(a), load(b));
        vaddvq_f32(vsetq_lane_f32(0.0, prod, 3))
    }
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

/// Column-major `4x4 * vec4` using fused multiply-add across columns.
#[inline]
pub fn mat4_mul_vec4(m: &[[f32; 4]; 4], v: [f32; 4]) -> [f32; 4] {
    // SAFETY: all operands are valid NEON vectors; `vfmaq_laneq_f32` indexes
    // lanes 0..=3 of `vv`, all in range for a 4-lane vector.
    unsafe {
        let c0 = load(m[0]);
        let c1 = load(m[1]);
        let c2 = load(m[2]);
        let c3 = load(m[3]);
        let vv = load(v);
        let mut acc = vmulq_laneq_f32(c0, vv, 0);
        acc = vfmaq_laneq_f32(acc, c1, vv, 1);
        acc = vfmaq_laneq_f32(acc, c2, vv, 2);
        acc = vfmaq_laneq_f32(acc, c3, vv, 3);
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

/// Hamilton product `a * b` on `[x, y, z, w]` quaternions.
///
/// Decomposes the product into four scaled, sign-flipped shuffles of `b` and
/// accumulates them with lane-indexed FMAs — algebraically identical to the
/// scalar reference.
#[inline]
pub fn quat_mul(a: [f32; 4], b: [f32; 4]) -> [f32; 4] {
    // SAFETY: every value is a valid 4-lane NEON vector; lane indices passed to
    // the `_laneq_` intrinsics are literal 0..=3.
    unsafe {
        let va = load(a);
        let vb = load(b);

        // Sign masks applied to the shuffled `b` vectors.
        let s1 = load([1.0, -1.0, 1.0, -1.0]);
        let s2 = load([1.0, 1.0, -1.0, -1.0]);
        let s3 = load([-1.0, 1.0, 1.0, -1.0]);

        // Shuffles of b (see module docs for the derivation):
        //   rev64(b)       = [by, bx, bw, bz]
        //   ext(b,b,2)     = [bz, bw, bx, by]
        //   reverse(b)     = [bw, bz, by, bx]
        let rev64 = vrev64q_f32(vb);
        let ext2 = vextq_f32::<2>(vb, vb);
        let reverse = vextq_f32::<2>(rev64, rev64);

        let b1 = vmulq_f32(reverse, s1); // [ bw, -bz,  by, -bx]
        let b2 = vmulq_f32(ext2, s2); //    [ bz,  bw, -bx, -by]
        let b3 = vmulq_f32(rev64, s3); //   [-by,  bx,  bw, -bz]

        // result = aw*b + ax*b1 + ay*b2 + az*b3
        let mut acc = vmulq_laneq_f32(vb, va, 3);
        acc = vfmaq_laneq_f32(acc, b1, va, 0);
        acc = vfmaq_laneq_f32(acc, b2, va, 1);
        acc = vfmaq_laneq_f32(acc, b3, va, 2);
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
