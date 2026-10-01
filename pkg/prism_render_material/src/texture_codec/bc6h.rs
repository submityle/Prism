//! BC6H (BPTC HDR) single-subset decode: mode 11, the simplest one-subset,
//! transform-free HDR mode of the `BC6H` block format.
//!
//! `BC6H` compresses *HDR* `RGB` (no alpha) into 16-byte blocks of half-float
//! texels. It has fourteen modes; ten carry two subsets (a 32-entry partition
//! table), four are single-subset (modes 11-14). **Mode 11** is the single
//! simplest: one subset, two `RGB` endpoints at 10 bits/channel stored
//! *directly* (no base+delta transform), and 4-bit indices for all sixteen
//! texels. Its field layout is fully contiguous, so it decodes from the block
//! bits alone -- no transcribed partition/anchor tables -- and a CPU golden
//! matches a GPU twin exactly (integer unquantize + integer interpolation, no
//! `AI/ML`).
//!
//! Only the **unsigned** (`UF16`) variant is decoded here; the signed (`SF16`)
//! variant and the delta-transform single-subset modes 12-14 (plus the
//! partitioned modes 1-10) are tracked as follow-ups so nothing is stubbed with
//! an unvalidated table.
//!
//! # Conventions
//! * The block is little-endian; bits are read LSB-first via [`BitReader`].
//! * The mode field is 2 bits when its low two bits are `00`/`01` (modes 1-2)
//!   and 5 bits otherwise; mode 11 is the 5-bit value `0b00011`.
//! * Output is row-major `RGB` `f32`, texel `t = y * 4 + x`, `t in [0, 16)`,
//!   converted from the decoded IEEE half-float bit pattern.
//! * Endpoint texels (index weight 0 or 64) decode bit-exactly; the half->f32
//!   widening is lossless, so `0.0` and `65504.0` compare exactly.
//!
//! # References
//! * Khronos Data Format Specification 1.3, BPTC `BC6H` block decode.
//! * Microsoft `DXGI_FORMAT_BC6H_UF16` / Vulkan `VK_FORMAT_BC6H_UFLOAT_BLOCK`.

use super::bitio::BitReader;

/// 4-bit index interpolation weights (Khronos `aWeight4`), in 1/64 units.
///
/// Local copy mirroring the `BC7` table; kept module-private to keep the two
/// BPTC decoders decoupled (the constant is tiny and never changes).
const WEIGHT4: [u32; 16] = [
    0, 4, 9, 13, 17, 21, 26, 30, 34, 38, 43, 47, 51, 56, 60, 64,
];

/// Convert an IEEE 754 binary16 (half) bit pattern to `f32`, losslessly.
///
/// Pure integer bit manipulation (no float transcendentals), handling zero,
/// subnormal, normal, and infinity/`NaN` half patterns. BC6H unsigned decode
/// only ever produces finite non-negative halves, but the full mapping keeps
/// the helper reusable and total.
#[must_use]
pub fn half_bits_to_f32(h: u16) -> f32 {
    let h = u32::from(h);
    let sign = (h & 0x8000) << 16;
    let exp = (h >> 10) & 0x1F;
    let mant = h & 0x3FF;
    let bits = if exp == 0 {
        if mant == 0 {
            sign // signed zero
        } else {
            // Subnormal half: renormalise into an f32 normal.
            let mut e = 0i32;
            let mut m = mant;
            while m & 0x400 == 0 {
                m <<= 1;
                e += 1;
            }
            m &= 0x3FF;
            // Half subnormal value is mant*2^-24; after renormalising the
            // leading 1 into bit 10 (e shifts), the unbiased exponent is
            // -14 - e, so the f32 biased exponent is 127 - 14 - e.
            let f32_exp = (127 - 14 - e) as u32;
            sign | (f32_exp << 23) | (m << 13)
        }
    } else if exp == 0x1F {
        // Infinity (mant 0) or NaN (mant != 0): max f32 exponent.
        sign | 0x7F80_0000 | (mant << 13)
    } else {
        // Normal: rebias exponent (15 -> 127) and left-align the mantissa.
        let f32_exp = exp + (127 - 15);
        sign | (f32_exp << 23) | (mant << 13)
    };
    f32::from_bits(bits)
}

/// Unquantize an unsigned `BC6H` endpoint component of `prec` bits to the
/// 16-bit intermediate range `[0, 0xFFFF]` (Khronos `Unquantize`, unsigned).
#[inline]
fn unquantize_unsigned(comp: u32, prec: u32) -> u32 {
    if prec >= 15 {
        comp
    } else if comp == 0 {
        0
    } else if comp == (1 << prec) - 1 {
        0xFFFF
    } else {
        ((comp << 16) + 0x8000) >> prec
    }
}

/// Interpolate one 16-bit intermediate channel between endpoints by a 4-bit
/// index weight, then apply the unsigned `finish_unquantize` scale to produce
/// the final half-float bit pattern.
#[inline]
fn interp_finish_unsigned(e0: u32, e1: u32, weight: u32) -> u16 {
    let q = ((64 - weight) * e0 + weight * e1 + 32) >> 6;
    // Unsigned finish scale: q * 31 / 64 maps the intermediate to a half.
    (((q * 31) >> 6) & 0xFFFF) as u16
}

/// Read the raw `BC6H` mode field: 2 bits when the low two bits select a
/// two-bit mode (`00`/`01`), otherwise the full 5-bit field.
#[inline]
#[must_use]
pub fn bc6h_mode_bits(block: &[u8; 16]) -> u8 {
    let low2 = block[0] & 0b11;
    if low2 < 0b10 {
        low2
    } else {
        block[0] & 0b1_1111
    }
}

/// Error returned by [`decode_bc6h_unsigned`] for a `BC6H` block whose mode is
/// not yet supported by this decoder.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Bc6hError {
    /// A valid but unsupported mode. Only the single-subset, transform-free
    /// unsigned mode 11 (`0b00011`) is decoded today; the delta-transform
    /// single-subset modes (12-14) and the partitioned modes are follow-ups.
    UnsupportedMode(u8),
}

/// Decode one 16-byte **BC6H mode 11 (unsigned)** block into sixteen `RGB`
/// `f32` texels.
///
/// The caller must have confirmed the block is mode 11 (see [`bc6h_mode_bits`]
/// `== 0b00011`). Field order (Khronos): mode(5); `rw gw bw` (endpoint 0) then
/// `rx gx bx` (endpoint 1), each 10 bits; the index block -- the anchor index
/// (texel 0) is 3 bits with an implicit high `0`, the remaining fifteen are
/// 4 bits each. Endpoints are used directly (no base+delta transform).
#[must_use]
pub fn decode_bc6h_mode11_unsigned(block: &[u8; 16]) -> [[f32; 3]; 16] {
    const PREC: u32 = 10;
    let mut r = BitReader::new(block);
    let _mode = r.read(5); // 5-bit mode field 0b00011.

    let rw = r.read(PREC);
    let gw = r.read(PREC);
    let bw = r.read(PREC);
    let rx = r.read(PREC);
    let gx = r.read(PREC);
    let bx = r.read(PREC);

    let e0 = [
        unquantize_unsigned(rw, PREC),
        unquantize_unsigned(gw, PREC),
        unquantize_unsigned(bw, PREC),
    ];
    let e1 = [
        unquantize_unsigned(rx, PREC),
        unquantize_unsigned(gx, PREC),
        unquantize_unsigned(bx, PREC),
    ];

    let mut indices = [0u8; 16];
    indices[0] = r.read(3) as u8; // anchor: implicit high bit 0.
    for idx in indices.iter_mut().skip(1) {
        *idx = r.read(4) as u8;
    }

    let mut out = [[0.0f32; 3]; 16];
    for (t, texel) in out.iter_mut().enumerate() {
        let w = WEIGHT4[indices[t] as usize];
        for c in 0..3 {
            texel[c] = half_bits_to_f32(interp_finish_unsigned(e0[c], e1[c], w));
        }
    }
    out
}

/// Decode an unsigned `BC6H` block, dispatching on its mode.
///
/// Only the single-subset, transform-free mode 11
/// ([`decode_bc6h_mode11_unsigned`]) is supported today. The delta-transform
/// single-subset modes (12-14) and the partitioned modes (1-10) return
/// [`Bc6hError::UnsupportedMode`] rather than a wrong decode -- they need the
/// validated Khronos endpoint-transform and partition/anchor tables (tracked as
/// a follow-up), and silently mis-decoding them would be worse than an error.
pub fn decode_bc6h_unsigned(block: &[u8; 16]) -> Result<[[f32; 3]; 16], Bc6hError> {
    match bc6h_mode_bits(block) {
        0b00011 => Ok(decode_bc6h_mode11_unsigned(block)),
        m => Err(Bc6hError::UnsupportedMode(m)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Minimal LSB-first bit writer for assembling test blocks.
    struct BitWriter {
        bytes: [u8; 16],
        pos: usize,
    }

    impl BitWriter {
        fn new() -> Self {
            Self {
                bytes: [0u8; 16],
                pos: 0,
            }
        }

        fn write(&mut self, value: u32, n: u32) {
            for i in 0..n {
                if (value >> i) & 1 == 1 {
                    self.bytes[self.pos / 8] |= 1 << (self.pos % 8);
                }
                self.pos += 1;
            }
        }
    }

    /// Assemble a mode-11 block: `e0`/`e1` are the two 10-bit `RGB` endpoints;
    /// `idx` the sixteen 4-bit indices (index 0 must be `<= 7`).
    fn make_block11(e0: [u32; 3], e1: [u32; 3], idx: [u8; 16]) -> [u8; 16] {
        let mut w = BitWriter::new();
        w.write(0b0_0011, 5); // mode 11
        w.write(e0[0], 10);
        w.write(e0[1], 10);
        w.write(e0[2], 10);
        w.write(e1[0], 10);
        w.write(e1[1], 10);
        w.write(e1[2], 10);
        w.write(u32::from(idx[0]), 3);
        for &i in idx.iter().skip(1) {
            w.write(u32::from(i), 4);
        }
        assert_eq!(w.pos, 128, "mode-11 fields must fill the block exactly");
        w.bytes
    }

    #[test]
    fn half_known_patterns_round_trip() {
        assert_eq!(half_bits_to_f32(0x0000), 0.0);
        assert_eq!(half_bits_to_f32(0x3C00), 1.0);
        assert_eq!(half_bits_to_f32(0x4000), 2.0);
        assert_eq!(half_bits_to_f32(0x3800), 0.5);
        assert_eq!(half_bits_to_f32(0xC000), -2.0);
        assert_eq!(half_bits_to_f32(0x7BFF), 65504.0);
        // Smallest positive subnormal half = 2^-24 (1.0 / 2^24, exact in f32).
        assert_eq!(half_bits_to_f32(0x0001), 1.0f32 / (1u32 << 24) as f32);
        // Signed zero preserves its sign bit.
        assert!(half_bits_to_f32(0x8000).is_sign_negative());
    }

    #[test]
    fn half_inf_and_nan() {
        assert!(half_bits_to_f32(0x7C00).is_infinite());
        assert!(half_bits_to_f32(0x7C00) > 0.0);
        assert!(half_bits_to_f32(0xFC00).is_infinite());
        assert!(half_bits_to_f32(0xFC00) < 0.0);
        assert!(half_bits_to_f32(0x7E00).is_nan());
    }

    #[test]
    fn mode11_is_detected() {
        let block = make_block11([0; 3], [1023; 3], [0u8; 16]);
        assert_eq!(bc6h_mode_bits(&block), 0b00011);
        assert!(decode_bc6h_unsigned(&block).is_ok());
    }

    #[test]
    fn mode11_endpoints_decode_exactly() {
        // Index 0 (weight 0) -> e0 (0 -> 0.0); index 15 (weight 64) -> e1
        // (1023 -> unquantize 0xFFFF -> finish 0x7BFF -> 65504.0).
        let mut idx = [0u8; 16];
        idx[1] = 15;
        let block = make_block11([0; 3], [1023; 3], idx);
        let out = decode_bc6h_mode11_unsigned(&block);
        assert_eq!(out[0], [0.0, 0.0, 0.0]); // e0 endpoint
        assert_eq!(out[1], [65504.0, 65504.0, 65504.0]); // e1 endpoint
    }

    #[test]
    fn mode11_is_monotonic_along_index_ramp() {
        // Flat-to-bright ramp: endpoints 0 and max, indices 0..=15 on R.
        let mut idx = [0u8; 16];
        for (t, slot) in idx.iter_mut().enumerate() {
            *slot = t.min(15) as u8;
        }
        idx[0] = idx[0].min(7); // anchor is 3-bit
        let block = make_block11([0; 3], [1023, 0, 0], idx);
        let out = decode_bc6h_mode11_unsigned(&block);
        // R channel is non-decreasing as the index weight grows.
        for t in 1..16 {
            assert!(out[t][0] >= out[t - 1][0], "R must be monotonic at {t}");
        }
        // G/B endpoints are both 0 -> flat 0 everywhere.
        assert!(out.iter().all(|p| p[1] == 0.0 && p[2] == 0.0));
    }

    #[test]
    fn mode11_midpoint_is_between_endpoints() {
        // Endpoints 0 and max; weight-32 index (index 8) sits between them.
        let mut idx = [0u8; 16];
        idx[1] = 8; // WEIGHT4[8] = 34, just over half
        let block = make_block11([0; 3], [1023; 3], idx);
        let out = decode_bc6h_mode11_unsigned(&block);
        assert!(out[1][0] > 0.0 && out[1][0] < 65504.0);
    }

    #[test]
    fn dispatch_rejects_unsupported_modes() {
        // Mode-1 block (low two bits 00) is a two-subset mode: unsupported.
        let m1 = [0u8; 16];
        assert_eq!(bc6h_mode_bits(&m1), 0);
        assert_eq!(decode_bc6h_unsigned(&m1), Err(Bc6hError::UnsupportedMode(0)));
        // Mode-12 block (5-bit 0b00111) is single-subset but delta: unsupported.
        let mut m12 = [0u8; 16];
        m12[0] = 0b0_0111;
        assert_eq!(bc6h_mode_bits(&m12), 0b00111);
        assert_eq!(
            decode_bc6h_unsigned(&m12),
            Err(Bc6hError::UnsupportedMode(0b00111))
        );
    }
}
