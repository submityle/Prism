//! 16-bit-per-channel vertex-attribute packing (`unorm16x2` / `snorm16x2`).
//!
//! These fold a two-component value into one `u32` of two 16-bit channels and
//! widen it back, matching the WGSL built-ins `pack2x16unorm` / `pack2x16snorm`
//! (and their `unpack` inverses) **bit-for-bit** so a CPU-produced buffer can be
//! consumed verbatim by a shader and vice versa. This is the canonical
//! high-precision compact encoding for texture coordinates (`unorm16x2` UVs,
//! where 8-bit would visibly swim) and for packed tangent-frame or motion-vector
//! pairs (`snorm16x2`), where a full `f32x2` would waste half the bandwidth yet
//! 8-bit is too coarse.
//!
//! # Conventions (identical to WGSL)
//!
//! Component 0 occupies the **low** 16 bits of the result, component 1 the high
//! 16 bits. The quantizers follow the WGSL spec exactly:
//!
//! - `unorm`: `h = ⌊0.5 + 65535 · clamp(c, 0, 1)⌋`, dequantized as `h / 65535`.
//! - `snorm`: `h = ⌊0.5 + 32767 · clamp(c, -1, 1)⌋` stored as a two's-complement
//!   `i16` (so the encoded range is `[-32767, 32767]`; `-32768` never occurs),
//!   and dequantized as `max(i16 / 32767, -1)`.
//!
//! The `⌊0.5 + x⌋` form rounds halves toward `+∞`. Rounding uses the crate's
//! deterministic `libm` floor so the result is identical on every target.
//!
//! No neural, learned, or data-driven components. No Unreal Engine or Unity
//! source or derived code.

use crate::float::f32 as mf;

/// Quantize one `[0, 1]` component to a 16-bit unsigned-normalized half-word.
#[inline]
fn unorm16(c: f32) -> u16 {
    let c = c.clamp(0.0, 1.0);
    // c ∈ [0, 1] ⇒ 0.5 + 65535·c ∈ [0.5, 65535.5] ⇒ floor ∈ [0, 65535], an
    // exact non-negative integer, so the cast is lossless.
    mf::floor(0.5 + 65535.0 * c) as u16
}

/// Quantize one `[-1, 1]` component to a 16-bit signed-normalized half-word
/// (returned as its two's-complement `u16` bit pattern).
#[inline]
fn snorm16(c: f32) -> u16 {
    let c = c.clamp(-1.0, 1.0);
    // c ∈ [-1, 1] ⇒ floor(0.5 + 32767·c) ∈ [-32767, 32767], an exact integer.
    let s = mf::floor(0.5 + 32767.0 * c) as i32;
    (s as i16) as u16
}

/// Dequantize one `unorm` half-word back to `[0, 1]`.
#[inline]
fn from_unorm16(h: u32) -> f32 {
    (h & 0xFFFF) as f32 / 65535.0
}

/// Dequantize one `snorm` half-word back to `[-1, 1]`.
#[inline]
fn from_snorm16(h: u32) -> f32 {
    let v = ((h & 0xFFFF) as u16 as i16) as f32 / 32767.0;
    v.max(-1.0)
}

/// Pack two `[0, 1]` components into one `u32` of two `unorm16` channels
/// (component 0 in the low half-word), mirroring WGSL `pack2x16unorm`.
#[inline]
#[must_use]
pub fn pack_unorm2x16(v: [f32; 2]) -> u32 {
    u32::from(unorm16(v[0])) | (u32::from(unorm16(v[1])) << 16)
}

/// Unpack a `u32` of two `unorm16` channels back to `[0, 1]` components,
/// mirroring WGSL `unpack2x16unorm`.
#[inline]
#[must_use]
pub fn unpack_unorm2x16(bits: u32) -> [f32; 2] {
    [from_unorm16(bits), from_unorm16(bits >> 16)]
}

/// Pack two `[-1, 1]` components into one `u32` of two `snorm16` channels
/// (component 0 in the low half-word), mirroring WGSL `pack2x16snorm`.
#[inline]
#[must_use]
pub fn pack_snorm2x16(v: [f32; 2]) -> u32 {
    u32::from(snorm16(v[0])) | (u32::from(snorm16(v[1])) << 16)
}

/// Unpack a `u32` of two `snorm16` channels back to `[-1, 1]` components,
/// mirroring WGSL `unpack2x16snorm`.
#[inline]
#[must_use]
pub fn unpack_snorm2x16(bits: u32) -> [f32; 2] {
    [from_snorm16(bits), from_snorm16(bits >> 16)]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unorm_exact_halfword_round_trip() {
        // A spread of unorm half-words dequantizes and re-quantizes to itself.
        let mut h: u32 = 0;
        while h <= 0xFFFF {
            let c = h as f32 / 65535.0;
            assert_eq!(u32::from(unorm16(c)), h, "unorm requantize at hw {h}");
            h += 7;
        }
        // Endpoints exactly.
        assert_eq!(u32::from(unorm16(0.0)), 0);
        assert_eq!(u32::from(unorm16(1.0)), 65535);
    }

    #[test]
    fn snorm_exact_code_round_trip() {
        // A spread of encodable signed codes in [-32767, 32767] requantizes.
        let mut k: i32 = -32767;
        while k <= 32767 {
            let c = k as f32 / 32767.0;
            let b = snorm16(c) as i16 as i32;
            assert_eq!(b, k, "snorm requantize at code {k}");
            k += 7;
        }
        // Endpoints exactly.
        assert_eq!(snorm16(1.0) as i16, 32767);
        assert_eq!(snorm16(-1.0) as i16, -32767);
        assert_eq!(snorm16(0.0) as i16, 0);
    }

    #[test]
    fn unorm_endpoints_and_clamp() {
        assert_eq!(pack_unorm2x16([0.0, 1.0]), 0xFFFF_0000);
        // Out-of-range clamps to the endpoints.
        assert_eq!(pack_unorm2x16([-1.0, 2.0]), 0xFFFF_0000);
    }

    #[test]
    fn snorm_endpoints_and_sign() {
        // +1 -> 32767 (0x7FFF), -1 -> -32767 (0x8001 as u16).
        let bits = pack_snorm2x16([1.0, -1.0]);
        assert_eq!(bits & 0xFFFF, 0x7FFF);
        assert_eq!((bits >> 16) & 0xFFFF, 0x8001);
        // Out-of-range clamps.
        let c = pack_snorm2x16([-2.0, 3.0]);
        assert_eq!(c & 0xFFFF, 0x8001);
        assert_eq!((c >> 16) & 0xFFFF, 0x7FFF);
    }

    #[test]
    fn unorm_round_trip_values() {
        for &c in &[0.0f32, 0.25, 0.5, 0.75, 1.0, 0.123_456, 0.987_654] {
            let got = unpack_unorm2x16(pack_unorm2x16([c, 1.0 - c]));
            // Within half a quantization step of the input.
            assert!((got[0] - c).abs() <= 0.5 / 65535.0 + 1e-7, "unorm rt {c}");
        }
    }

    #[test]
    fn snorm_round_trip_values_and_clamp() {
        for &c in &[-1.0f32, -0.5, 0.0, 0.5, 1.0, -0.321, 0.654] {
            let got = unpack_snorm2x16(pack_snorm2x16([c, -c]));
            assert!((got[0] - c).abs() <= 0.5 / 32767.0 + 1e-7, "snorm rt {c}");
        }
        // The -32768 code never occurs on encode; but if a raw key carries it,
        // unpack clamps up to -1.0.
        let raw = 0x0000_8000u32; // low half-word = 0x8000 = -32768
        assert_eq!(unpack_snorm2x16(raw)[0], -1.0);
    }
}
