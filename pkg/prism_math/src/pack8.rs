//! 8-bit-per-channel vertex-attribute packing (`unorm8x4` / `snorm8x4`).
//!
//! These fold a four-component value into one `u32` of four 8-bit channels and
//! widen it back, matching the WGSL built-ins `pack4x8unorm` / `pack4x8snorm`
//! (and their `unpack` inverses) **byte-for-byte** so a CPU-produced buffer can
//! be consumed verbatim by a shader and vice versa. This is the canonical
//! compact encoding for vertex colors (`unorm`), and for normals / tangents and
//! other direction-ish attributes (`snorm`), where a full `f32x4` would waste
//! three quarters of the bandwidth.
//!
//! # Conventions (identical to WGSL)
//!
//! Component 0 occupies the **low** byte of the result, component 3 the high
//! byte. The quantizers follow the WGSL spec exactly:
//!
//! - `unorm`: `byte = ⌊0.5 + 255 · clamp(c, 0, 1)⌋`, dequantized as `byte / 255`.
//! - `snorm`: `byte = ⌊0.5 + 127 · clamp(c, -1, 1)⌋` stored as a two's-complement
//!   `i8` (so the encoded range is `[-127, 127]`; `-128` never occurs), and
//!   dequantized as `max(i8 / 127, -1)`.
//!
//! The `⌊0.5 + x⌋` form rounds halves toward `+∞`. Rounding uses the crate's
//! deterministic `libm` floor so the result is identical on every target.
//!
//! No neural, learned, or data-driven components. No Unreal Engine or Unity
//! source or derived code.

use crate::float::f32 as mf;

/// Quantize one `[0, 1]` component to an 8-bit unsigned-normalized byte.
#[inline]
fn unorm8(c: f32) -> u8 {
    let c = c.clamp(0.0, 1.0);
    // c ∈ [0, 1] ⇒ 0.5 + 255·c ∈ [0.5, 255.5] ⇒ floor ∈ [0, 255], already an
    // exact non-negative integer, so the cast is lossless.
    mf::floor(0.5 + 255.0 * c) as u8
}

/// Quantize one `[-1, 1]` component to an 8-bit signed-normalized byte
/// (returned as its two's-complement `u8` bit pattern).
#[inline]
fn snorm8(c: f32) -> u8 {
    let c = c.clamp(-1.0, 1.0);
    // c ∈ [-1, 1] ⇒ floor(0.5 + 127·c) ∈ [-127, 127], an exact integer.
    let s = mf::floor(0.5 + 127.0 * c) as i32;
    (s as i8) as u8
}

/// Dequantize one `unorm` byte back to `[0, 1]`.
#[inline]
fn from_unorm8(b: u32) -> f32 {
    (b & 0xFF) as f32 / 255.0
}

/// Dequantize one `snorm` byte back to `[-1, 1]`.
#[inline]
fn from_snorm8(b: u32) -> f32 {
    let v = ((b & 0xFF) as u8 as i8) as f32 / 127.0;
    v.max(-1.0)
}

/// Pack four `[0, 1]` components into one `u32` of four `unorm8` channels
/// (component 0 in the low byte), mirroring WGSL `pack4x8unorm`.
#[inline]
#[must_use]
pub fn pack_unorm4x8(v: [f32; 4]) -> u32 {
    u32::from(unorm8(v[0]))
        | (u32::from(unorm8(v[1])) << 8)
        | (u32::from(unorm8(v[2])) << 16)
        | (u32::from(unorm8(v[3])) << 24)
}

/// Unpack a `u32` of four `unorm8` channels back to `[0, 1]` components,
/// mirroring WGSL `unpack4x8unorm`.
#[inline]
#[must_use]
pub fn unpack_unorm4x8(bits: u32) -> [f32; 4] {
    [
        from_unorm8(bits),
        from_unorm8(bits >> 8),
        from_unorm8(bits >> 16),
        from_unorm8(bits >> 24),
    ]
}

/// Pack four `[-1, 1]` components into one `u32` of four `snorm8` channels
/// (component 0 in the low byte), mirroring WGSL `pack4x8snorm`.
#[inline]
#[must_use]
pub fn pack_snorm4x8(v: [f32; 4]) -> u32 {
    u32::from(snorm8(v[0]))
        | (u32::from(snorm8(v[1])) << 8)
        | (u32::from(snorm8(v[2])) << 16)
        | (u32::from(snorm8(v[3])) << 24)
}

/// Unpack a `u32` of four `snorm8` channels back to `[-1, 1]` components,
/// mirroring WGSL `unpack4x8snorm`.
#[inline]
#[must_use]
pub fn unpack_snorm4x8(bits: u32) -> [f32; 4] {
    [
        from_snorm8(bits),
        from_snorm8(bits >> 8),
        from_snorm8(bits >> 16),
        from_snorm8(bits >> 24),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unorm_exact_byte_round_trip() {
        // Every byte dequantizes and re-quantizes to itself.
        for k in 0..=255u32 {
            let c = k as f32 / 255.0;
            assert_eq!(u32::from(unorm8(c)), k, "unorm requantize at byte {k}");
        }
    }

    #[test]
    fn snorm_exact_byte_round_trip() {
        // Every encodable signed byte in [-127, 127] requantizes to itself.
        for k in -127i32..=127 {
            let c = k as f32 / 127.0;
            let b = snorm8(c) as i8 as i32;
            assert_eq!(b, k, "snorm requantize at code {k}");
        }
    }

    #[test]
    fn unorm_endpoints_and_clamp() {
        assert_eq!(pack_unorm4x8([0.0, 1.0, 0.0, 1.0]), 0xFF00_FF00);
        // Out-of-range clamps.
        assert_eq!(pack_unorm4x8([-1.0, 2.0, 0.0, 1.0]) & 0xFFFF, 0xFF00);
    }

    #[test]
    fn snorm_endpoints_and_sign() {
        // +1 -> 127 (0x7F), -1 -> -127 (0x81 as u8), 0 -> 0.
        let bits = pack_snorm4x8([1.0, -1.0, 0.0, 0.5]);
        assert_eq!(bits & 0xFF, 0x7F);
        assert_eq!((bits >> 8) & 0xFF, 0x81);
        assert_eq!((bits >> 16) & 0xFF, 0x00);
        // 0.5 -> floor(0.5 + 63.5) = 64 = 0x40.
        assert_eq!((bits >> 24) & 0xFF, 0x40);
    }

    #[test]
    fn unpack_matches_known_bytes() {
        let v = unpack_unorm4x8(0xFF00_8000);
        assert_eq!(v[0].to_bits(), (0.0f32).to_bits());
        assert_eq!(v[2].to_bits(), (0.0f32).to_bits());
        assert_eq!(v[3].to_bits(), (1.0f32).to_bits());
        // snorm -1 clamp: byte 0x80 (-128) -> max(-128/127, -1) = -1.
        let s = unpack_snorm4x8(0x0000_0080);
        assert_eq!(s[0].to_bits(), (-1.0f32).to_bits());
    }

    #[test]
    fn full_round_trip_within_one_code() {
        // Arbitrary in-range values survive a pack/unpack within one code step.
        let mut seed = 0x1234_5678u32;
        let mut next = || {
            seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (seed >> 8) as f32 / (1u32 << 24) as f32 // [0, 1)
        };
        for _ in 0..1000 {
            let u = [next(), next(), next(), next()];
            let ru = unpack_unorm4x8(pack_unorm4x8(u));
            for i in 0..4 {
                assert!((ru[i] - u[i]).abs() <= 1.0 / 255.0 + 1e-6);
            }
            let s = [
                next() * 2.0 - 1.0,
                next() * 2.0 - 1.0,
                next() * 2.0 - 1.0,
                next() * 2.0 - 1.0,
            ];
            let rs = unpack_snorm4x8(pack_snorm4x8(s));
            for i in 0..4 {
                assert!((rs[i] - s[i]).abs() <= 1.0 / 127.0 + 1e-6);
            }
        }
    }
}
