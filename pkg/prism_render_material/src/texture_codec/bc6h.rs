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
//! Both the **unsigned** (`UF16`) and **signed** (`SF16`) variants of mode 11
//! are decoded here: the mode-11 bit layout is identical for both, only the
//! endpoint interpretation differs (two's-complement + signed unquantize). The
//! delta-transform single-subset modes 12-14 (plus the partitioned modes 1-10)
//! are tracked as follow-ups so nothing is stubbed with an unvalidated table.
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
const WEIGHT4: [u32; 16] = [0, 4, 9, 13, 17, 21, 26, 30, 34, 38, 43, 47, 51, 55, 60, 64];

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

/// Sign-extend the low `bits` of `value` into a full `i32` (two's complement).
#[inline]
fn sign_extend(value: u32, bits: u32) -> i32 {
    let shift = 32 - bits;
    #[expect(
        clippy::cast_possible_wrap,
        reason = "deliberate two's-complement reinterpret"
    )]
    let widened = (value << shift) as i32;
    widened >> shift
}

/// Unquantize a **signed** `BC6H` endpoint component of `prec` bits into the
/// signed 16-bit intermediate range `[-0x7FFF, 0x7FFF]` (Khronos `Unquantize`,
/// signed). The magnitude saturates to `0x7FFF`; the sign is preserved.
#[inline]
fn unquantize_signed(comp: i32, prec: u32) -> i32 {
    if prec >= 16 {
        return comp;
    }
    let (neg, v) = if comp < 0 {
        (true, -comp)
    } else {
        (false, comp)
    };
    let unq = if v == 0 {
        0
    } else if v >= (1i32 << (prec - 1)) - 1 {
        0x7FFF
    } else {
        ((v << 15) + 0x4000) >> (prec - 1)
    };
    if neg {
        -unq
    } else {
        unq
    }
}

/// Interpolate one signed intermediate channel between endpoints by a 4-bit
/// index weight, apply the signed `finish_unquantize` scale (magnitude * 31/32),
/// and re-pack the result as a sign-magnitude IEEE half bit pattern (half-float
/// stores sign and magnitude separately, never two's complement).
#[inline]
fn interp_finish_signed(e0: i32, e1: i32, weight: i32) -> u16 {
    let q = ((64 - weight) * e0 + weight * e1 + 32) >> 6;
    #[expect(clippy::cast_sign_loss, reason = "magnitude is non-negative after abs")]
    let (sign, mag): (u16, u32) = if q < 0 {
        (0x8000, ((-q) as u32 * 31) >> 5)
    } else {
        (0, (q as u32 * 31) >> 5)
    };
    sign | (mag & 0x7FFF) as u16
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
    /// A valid but unsupported mode. All four single-subset modes are decoded
    /// for both the signed and unsigned variants: the transform-free mode 11
    /// (`0b00011`) and the delta-transform modes 12/13/14
    /// (`0b00111`/`0b01011`/`0b01111`). The two-subset partitioned modes are a
    /// follow-up and still report this error.
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

/// Decode one 16-byte **BC6H mode 11 (signed, `SF16`)** block into sixteen
/// `RGB` `f32` texels.
///
/// Identical field layout to [`decode_bc6h_mode11_unsigned`] -- the mode-11 bit
/// packing does not depend on signedness -- so only the endpoint interpretation
/// differs: each 10-bit field is read as a two's-complement signed value,
/// unquantized with the signed rule, interpolated in signed arithmetic, and
/// finished with the signed `31/32` scale before sign-magnitude half packing.
#[must_use]
pub fn decode_bc6h_mode11_signed(block: &[u8; 16]) -> [[f32; 3]; 16] {
    const PREC: u32 = 10;
    let mut r = BitReader::new(block);
    let _mode = r.read(5); // 5-bit mode field 0b00011.

    let rw = sign_extend(r.read(PREC), PREC);
    let gw = sign_extend(r.read(PREC), PREC);
    let bw = sign_extend(r.read(PREC), PREC);
    let rx = sign_extend(r.read(PREC), PREC);
    let gx = sign_extend(r.read(PREC), PREC);
    let bx = sign_extend(r.read(PREC), PREC);

    let e0 = [
        unquantize_signed(rw, PREC),
        unquantize_signed(gw, PREC),
        unquantize_signed(bw, PREC),
    ];
    let e1 = [
        unquantize_signed(rx, PREC),
        unquantize_signed(gx, PREC),
        unquantize_signed(bx, PREC),
    ];

    let mut indices = [0u8; 16];
    indices[0] = r.read(3) as u8; // anchor: implicit high bit 0.
    for idx in indices.iter_mut().skip(1) {
        *idx = r.read(4) as u8;
    }

    let mut out = [[0.0f32; 3]; 16];
    for (t, texel) in out.iter_mut().enumerate() {
        let w = WEIGHT4[indices[t] as usize] as i32;
        for c in 0..3 {
            texel[c] = half_bits_to_f32(interp_finish_signed(e0[c], e1[c], w));
        }
    }
    out
}

/// Decode a single-subset **delta-transform** `BC6H` block (modes 12/13/14).
///
/// The three single-subset transformed modes share one layout: `mode(5)`, the
/// base endpoint's low 10 bits inline (`rw gw bw`), then per channel the
/// `delta_bits`-wide two's-complement delta immediately followed by that
/// channel's relocated high base bits (`bit 10 .. base_prec-1`), giving the
/// stream `rx rw_hi gx gw_hi bx bw_hi` (each `*_hi` most-significant-bit first),
/// then the shared index block (anchor
/// texel 0 is 3 bits with an implicit high `0`, the other fifteen are 4 bits).
/// The inverse transform reconstructs endpoint 1 as
/// `(base + sign_extend(delta)) & ((1 << base_prec) - 1)`; both endpoints are
/// then unquantized at `base_prec`. `signed` selects the `SF16`/`UF16` rule.
///
/// | mode | bits    | `base_prec` | `delta_bits` |
/// |------|---------|-------------|--------------|
/// | 12   | `00111` | 11          | 9            |
/// | 13   | `01011` | 12          | 8            |
/// | 14   | `01111` | 16          | 4            |
fn decode_bc6h_single_subset_delta(
    block: &[u8; 16],
    base_prec: u32,
    delta_bits: u32,
    signed: bool,
) -> [[f32; 3]; 16] {
    let mut r = BitReader::new(block);
    let _mode = r.read(5);

    // BC6H packs the base endpoint's low 10 bits inline (`rw gw bw`), then
    // interleaves, per channel, the delta field followed by that channel's
    // relocated high base bits (`bit 10 .. base_prec-1`, ascending): the stream
    // is `rx rw_hi gx gw_hi bx bw_hi` (Khronos/DirectXTex one-region layout).
    let hi_bits = base_prec - 10;
    let lo = [r.read(10), r.read(10), r.read(10)];
    let mut delta = [0u32; 3];
    let mut base = [0u32; 3];
    for c in 0..3 {
        delta[c] = r.read(delta_bits);
        // High base bits are packed most-significant-first (bit base_prec-1
        // down to bit 10), immediately after this channel's delta.
        let mut hi = 0u32;
        for k in (0..hi_bits).rev() {
            hi |= r.read(1) << k;
        }
        base[c] = lo[c] | (hi << 10);
    }

    let mut indices = [0u8; 16];
    indices[0] = r.read(3) as u8; // anchor: implicit high bit 0.
    for idx in indices.iter_mut().skip(1) {
        *idx = r.read(4) as u8;
    }

    // Inverse transform: endpoint 1 = (base + signed delta) wrapped to the base
    // precision. `base_prec` is at most 16 here, so a 32-bit mask is exact.
    let mask: u32 = if base_prec >= 32 {
        u32::MAX
    } else {
        (1u32 << base_prec) - 1
    };
    let mut e1_bits = [0u32; 3];
    for c in 0..3 {
        let d = sign_extend(delta[c], delta_bits);
        #[expect(
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            reason = "wrapping add is then masked to base_prec bits"
        )]
        let sum = (i64::from(base[c]) + i64::from(d)) as u32;
        e1_bits[c] = sum & mask;
    }

    let mut out = [[0.0f32; 3]; 16];
    if signed {
        let e0: [i32; 3] =
            core::array::from_fn(|c| unquantize_signed(sign_extend(base[c], base_prec), base_prec));
        let e1: [i32; 3] = core::array::from_fn(|c| {
            unquantize_signed(sign_extend(e1_bits[c], base_prec), base_prec)
        });
        for (t, texel) in out.iter_mut().enumerate() {
            let w = WEIGHT4[indices[t] as usize] as i32;
            for c in 0..3 {
                texel[c] = half_bits_to_f32(interp_finish_signed(e0[c], e1[c], w));
            }
        }
    } else {
        let e0: [u32; 3] = core::array::from_fn(|c| unquantize_unsigned(base[c], base_prec));
        let e1: [u32; 3] = core::array::from_fn(|c| unquantize_unsigned(e1_bits[c], base_prec));
        for (t, texel) in out.iter_mut().enumerate() {
            let w = WEIGHT4[indices[t] as usize];
            for c in 0..3 {
                texel[c] = half_bits_to_f32(interp_finish_unsigned(e0[c], e1[c], w));
            }
        }
    }
    out
}

/// Decode a **BC6H mode 12 (unsigned)** block (one subset, 11-bit base + 9-bit
/// delta). See [`decode_bc6h_single_subset_delta`].
#[must_use]
pub fn decode_bc6h_mode12_unsigned(block: &[u8; 16]) -> [[f32; 3]; 16] {
    decode_bc6h_single_subset_delta(block, 11, 9, false)
}

/// Decode a **BC6H mode 12 (signed, `SF16`)** block (one subset, 11-bit base +
/// 9-bit delta). See [`decode_bc6h_single_subset_delta`].
#[must_use]
pub fn decode_bc6h_mode12_signed(block: &[u8; 16]) -> [[f32; 3]; 16] {
    decode_bc6h_single_subset_delta(block, 11, 9, true)
}

/// Decode a **BC6H mode 13 (unsigned)** block (one subset, 12-bit base + 8-bit
/// delta). See [`decode_bc6h_single_subset_delta`].
#[must_use]
pub fn decode_bc6h_mode13_unsigned(block: &[u8; 16]) -> [[f32; 3]; 16] {
    decode_bc6h_single_subset_delta(block, 12, 8, false)
}

/// Decode a **BC6H mode 13 (signed, `SF16`)** block (one subset, 12-bit base +
/// 8-bit delta). See [`decode_bc6h_single_subset_delta`].
#[must_use]
pub fn decode_bc6h_mode13_signed(block: &[u8; 16]) -> [[f32; 3]; 16] {
    decode_bc6h_single_subset_delta(block, 12, 8, true)
}

/// Decode a **BC6H mode 14 (unsigned)** block (one subset, 16-bit base + 4-bit
/// delta). See [`decode_bc6h_single_subset_delta`].
#[must_use]
pub fn decode_bc6h_mode14_unsigned(block: &[u8; 16]) -> [[f32; 3]; 16] {
    decode_bc6h_single_subset_delta(block, 16, 4, false)
}

/// Decode a **BC6H mode 14 (signed, `SF16`)** block (one subset, 16-bit base +
/// 4-bit delta). See [`decode_bc6h_single_subset_delta`].
#[must_use]
pub fn decode_bc6h_mode14_signed(block: &[u8; 16]) -> [[f32; 3]; 16] {
    decode_bc6h_single_subset_delta(block, 16, 4, true)
}

/// Decode an unsigned `BC6H` block, dispatching on its mode.
///
/// All four single-subset modes are supported: the transform-free mode 11
/// ([`decode_bc6h_mode11_unsigned`]) and the delta-transform modes 12/13/14
/// ([`decode_bc6h_mode12_unsigned`]/[`decode_bc6h_mode13_unsigned`]/
/// [`decode_bc6h_mode14_unsigned`]). The two-subset partitioned modes (0-10)
/// return [`Bc6hError::UnsupportedMode`] rather than a wrong decode -- they need
/// the validated Khronos partition/anchor and per-mode scrambled-bit tables
/// (tracked as a follow-up), and silently mis-decoding them would be worse than
/// an error.
pub fn decode_bc6h_unsigned(block: &[u8; 16]) -> Result<[[f32; 3]; 16], Bc6hError> {
    match bc6h_mode_bits(block) {
        0b00 => Ok(decode_bc6h_mode1_unsigned(block)),
        0b01 => Ok(decode_bc6h_mode2_unsigned(block)),
        0b00011 => Ok(decode_bc6h_mode11_unsigned(block)),
        0b00111 => Ok(decode_bc6h_mode12_unsigned(block)),
        0b01011 => Ok(decode_bc6h_mode13_unsigned(block)),
        0b01111 => Ok(decode_bc6h_mode14_unsigned(block)),
        m => Err(Bc6hError::UnsupportedMode(m)),
    }
}

/// Decode a **signed** (`SF16`) `BC6H` block, dispatching on its mode.
///
/// All four single-subset modes are supported (mode 11 transform-free plus the
/// delta-transform modes 12/13/14); every two-subset partitioned mode returns
/// [`Bc6hError::UnsupportedMode`] rather than risk a wrong decode.
pub fn decode_bc6h_signed(block: &[u8; 16]) -> Result<[[f32; 3]; 16], Bc6hError> {
    match bc6h_mode_bits(block) {
        0b00 => Ok(decode_bc6h_mode1_signed(block)),
        0b01 => Ok(decode_bc6h_mode2_signed(block)),
        0b00011 => Ok(decode_bc6h_mode11_signed(block)),
        0b00111 => Ok(decode_bc6h_mode12_signed(block)),
        0b01011 => Ok(decode_bc6h_mode13_signed(block)),
        0b01111 => Ok(decode_bc6h_mode14_signed(block)),
        m => Err(Bc6hError::UnsupportedMode(m)),
    }
}

// ---------------------------------------------------------------------------
// BC6H two-subset (partitioned) modes.
//
// Ten of BC6H's fourteen modes carry two subsets selected by a 5-bit partition
// index into the shared 2-subset partition table. Four RGB endpoints are
// stored (subset 0 endpoints A/B, subset 1 endpoints A/B); in the transformed
// modes endpoints B/A/B are per-channel two's-complement deltas off the single
// base endpoint (subset 0 A), sign-extended by their channel delta width, added
// to the base and wrapped to the base precision before unquantize. Mode 10 is
// the one non-transformed two-subset mode (four raw 6-bit endpoints).
//
// Each mode scatters its ~82 header bits in a mode-specific order (the Khronos
// / DirectXTex `ModeDescriptor`): a flat list of `(field, bit)` pairs in block
// bit-stream order. The generic decoder below replays that descriptor to
// reconstruct the fields, so adding a new mode is just adding its table + the
// per-channel precisions. Every mode is proved bit-exact against the GPU
// hardware oracle before being wired into the dispatcher.
// ---------------------------------------------------------------------------

use super::bptc_tables::{BPTC_ANCHORS_2, BPTC_PARTITIONS_2, WEIGHT3};

/// Header-field identifiers for the BC6H two-subset mode descriptor. `W` is
/// subset 0 endpoint A (the base in transformed modes), `X` subset 0 endpoint
/// B, `Y` subset 1 endpoint A, `Z` subset 1 endpoint B; `D` is the partition
/// index. `M` (mode bits) is implied by dispatch and ignored on decode.
#[derive(Clone, Copy)]
enum Bc6hField {
    Rw,
    Gw,
    Bw,
    Rx,
    Gx,
    Bx,
    Ry,
    Gy,
    By,
    Rz,
    Gz,
    Bz,
    D,
    M,
}

/// Static layout of one BC6H two-subset mode: whether endpoints B/A/B are
/// stored as deltas off the base (`transformed`), the shared endpoint
/// precision (`base_prec`), the per-channel delta widths (unused when not
/// transformed), and the 82-entry header-bit descriptor in bit-stream order.
struct TwoSubsetMode {
    transformed: bool,
    base_prec: u32,
    delta_bits: [u32; 3],
    descriptor: &'static [(Bc6hField, u8)],
}

/// BC6H mode 1 (`0b00`, 2-bit mode): two subsets, 10-bit base, 5/5/5 deltas,
/// transformed. Descriptor transcribed from the Khronos / `DirectXTex`
/// `ModeDescriptor` and proved bit-exact against the GPU oracle.
#[rustfmt::skip]
const BC6H_MODE1: TwoSubsetMode = {
    use Bc6hField::{Bw, Bx, By, Bz, D, Gw, Gx, Gy, Gz, M, Rw, Rx, Ry, Rz};
    TwoSubsetMode {
        transformed: true,
        base_prec: 10,
        delta_bits: [5, 5, 5],
        descriptor: &[
            (M, 0), (M, 1), (Gy, 4), (By, 4), (Bz, 4), (Rw, 0), (Rw, 1), (Rw, 2), (Rw, 3), (Rw, 4),
            (Rw, 5), (Rw, 6), (Rw, 7), (Rw, 8), (Rw, 9), (Gw, 0), (Gw, 1), (Gw, 2), (Gw, 3), (Gw, 4),
            (Gw, 5), (Gw, 6), (Gw, 7), (Gw, 8), (Gw, 9), (Bw, 0), (Bw, 1), (Bw, 2), (Bw, 3), (Bw, 4),
            (Bw, 5), (Bw, 6), (Bw, 7), (Bw, 8), (Bw, 9), (Rx, 0), (Rx, 1), (Rx, 2), (Rx, 3), (Rx, 4),
            (Gz, 4), (Gy, 0), (Gy, 1), (Gy, 2), (Gy, 3), (Gx, 0), (Gx, 1), (Gx, 2), (Gx, 3), (Gx, 4),
            (Bz, 0), (Gz, 0), (Gz, 1), (Gz, 2), (Gz, 3), (Bx, 0), (Bx, 1), (Bx, 2), (Bx, 3), (Bx, 4),
            (Bz, 1), (By, 0), (By, 1), (By, 2), (By, 3), (Ry, 0), (Ry, 1), (Ry, 2), (Ry, 3), (Ry, 4),
            (Bz, 2), (Rz, 0), (Rz, 1), (Rz, 2), (Rz, 3), (Rz, 4), (Bz, 3), (D, 0), (D, 1), (D, 2),
            (D, 3), (D, 4),
        ],
    }
};

/// Decode one BC6H two-subset block via its `mode` descriptor into sixteen RGB
/// `f32` texels. `signed` selects the `SF16` (two's-complement) vs `UF16`
/// endpoint interpretation. The caller must have confirmed the block's mode
/// matches `mode`.
fn decode_bc6h_two_subset(block: &[u8; 16], mode: &TwoSubsetMode, signed: bool) -> [[f32; 3]; 16] {
    let mut r = BitReader::new(block);

    // Replay the descriptor: each stream bit sets one bit of one field. Mode
    // bits (`M`) are implied by dispatch; their descriptor slots are skipped.
    let mut f = [0u32; 13];
    for &(field, bit) in mode.descriptor {
        let b = r.read(1);
        let i = field as usize;
        if i < 13 {
            f[i] |= b << bit;
        }
    }

    let partition = f[Bc6hField::D as usize] as usize;
    let base = [
        f[Bc6hField::Rw as usize],
        f[Bc6hField::Gw as usize],
        f[Bc6hField::Bw as usize],
    ];
    // Endpoints B (subset 0), A, B (subset 1) in channel order.
    let raw = [
        [
            f[Bc6hField::Rx as usize],
            f[Bc6hField::Gx as usize],
            f[Bc6hField::Bx as usize],
        ],
        [
            f[Bc6hField::Ry as usize],
            f[Bc6hField::Gy as usize],
            f[Bc6hField::By as usize],
        ],
        [
            f[Bc6hField::Rz as usize],
            f[Bc6hField::Gz as usize],
            f[Bc6hField::Bz as usize],
        ],
    ];

    // Reconstruct the four endpoints as `base_prec`-bit integers. In the
    // transformed modes the three non-base endpoints are deltas: sign-extend
    // by the channel delta width, add to the base, wrap to the base precision.
    let mask: u32 = (1u32 << mode.base_prec) - 1;
    let mut ep = [[0u32; 3]; 4]; // [s0A, s0B, s1A, s1B]
    ep[0] = base;
    for (e, raw_e) in raw.iter().enumerate() {
        for c in 0..3 {
            ep[e + 1][c] = if mode.transformed {
                let d = sign_extend(raw_e[c], mode.delta_bits[c]);
                #[expect(
                    clippy::cast_possible_truncation,
                    clippy::cast_sign_loss,
                    reason = "wrapping add is then masked to base_prec bits"
                )]
                let sum = (i64::from(base[c]) + i64::from(d)) as u32;
                sum & mask
            } else {
                raw_e[c]
            };
        }
    }

    let anchor1 = BPTC_ANCHORS_2[partition];
    let mut indices = [0u8; 16];
    for (t, slot) in indices.iter_mut().enumerate() {
        // Subset anchors (texel 0 for subset 0, `anchor1` for subset 1) carry
        // a 2-bit index with an implicit high zero; all others are 3-bit.
        let bits = if t == 0 || t == anchor1 { 2 } else { 3 };
        *slot = r.read(bits) as u8;
    }

    let mut out = [[0.0f32; 3]; 16];
    if signed {
        let q: [[i32; 3]; 4] = core::array::from_fn(|e| {
            core::array::from_fn(|c| {
                unquantize_signed(sign_extend(ep[e][c], mode.base_prec), mode.base_prec)
            })
        });
        for (t, texel) in out.iter_mut().enumerate() {
            let subset = BPTC_PARTITIONS_2[partition][t] as usize;
            let (a, b) = if subset == 0 { (0, 1) } else { (2, 3) };
            let w = WEIGHT3[indices[t] as usize] as i32;
            for c in 0..3 {
                texel[c] = half_bits_to_f32(interp_finish_signed(q[a][c], q[b][c], w));
            }
        }
    } else {
        let q: [[u32; 3]; 4] = core::array::from_fn(|e| {
            core::array::from_fn(|c| unquantize_unsigned(ep[e][c], mode.base_prec))
        });
        for (t, texel) in out.iter_mut().enumerate() {
            let subset = BPTC_PARTITIONS_2[partition][t] as usize;
            let (a, b) = if subset == 0 { (0, 1) } else { (2, 3) };
            let w = WEIGHT3[indices[t] as usize];
            for c in 0..3 {
                texel[c] = half_bits_to_f32(interp_finish_unsigned(q[a][c], q[b][c], w));
            }
        }
    }
    out
}

/// Decode a **BC6H mode 1 (unsigned)** block (two subsets, 10-bit base, 5-bit
/// deltas). See [`decode_bc6h_two_subset`].
#[must_use]
pub fn decode_bc6h_mode1_unsigned(block: &[u8; 16]) -> [[f32; 3]; 16] {
    decode_bc6h_two_subset(block, &BC6H_MODE1, false)
}

/// Decode a **BC6H mode 1 (signed, `SF16`)** block (two subsets, 10-bit base,
/// 5-bit deltas). See [`decode_bc6h_two_subset`].
#[must_use]
pub fn decode_bc6h_mode1_signed(block: &[u8; 16]) -> [[f32; 3]; 16] {
    decode_bc6h_two_subset(block, &BC6H_MODE1, true)
}

/// BC6H mode 2 (`0b01`, 2-bit mode): two subsets, 7-bit base, 6/6/6 deltas,
/// transformed. Descriptor transcribed from the Khronos / `DirectXTex`
/// `ModeDescriptor` and proved bit-exact against the GPU oracle.
#[rustfmt::skip]
const BC6H_MODE2: TwoSubsetMode = {
    use Bc6hField::{Bw, Bx, By, Bz, D, Gw, Gx, Gy, Gz, M, Rw, Rx, Ry, Rz};
    TwoSubsetMode {
        transformed: true,
        base_prec: 7,
        delta_bits: [6, 6, 6],
        descriptor: &[
            (M, 0), (M, 1), (Gy, 5), (Gz, 4), (Gz, 5), (Rw, 0), (Rw, 1), (Rw, 2), (Rw, 3), (Rw, 4),
            (Rw, 5), (Rw, 6), (Bz, 0), (Bz, 1), (By, 4), (Gw, 0), (Gw, 1), (Gw, 2), (Gw, 3), (Gw, 4),
            (Gw, 5), (Gw, 6), (By, 5), (Bz, 2), (Gy, 4), (Bw, 0), (Bw, 1), (Bw, 2), (Bw, 3), (Bw, 4),
            (Bw, 5), (Bw, 6), (Bz, 3), (Bz, 5), (Bz, 4), (Rx, 0), (Rx, 1), (Rx, 2), (Rx, 3), (Rx, 4),
            (Rx, 5), (Gy, 0), (Gy, 1), (Gy, 2), (Gy, 3), (Gx, 0), (Gx, 1), (Gx, 2), (Gx, 3), (Gx, 4),
            (Gx, 5), (Gz, 0), (Gz, 1), (Gz, 2), (Gz, 3), (Bx, 0), (Bx, 1), (Bx, 2), (Bx, 3), (Bx, 4),
            (Bx, 5), (By, 0), (By, 1), (By, 2), (By, 3), (Ry, 0), (Ry, 1), (Ry, 2), (Ry, 3), (Ry, 4),
            (Ry, 5), (Rz, 0), (Rz, 1), (Rz, 2), (Rz, 3), (Rz, 4), (Rz, 5), (D, 0), (D, 1), (D, 2),
            (D, 3), (D, 4),
        ],
    }
};

/// Decode a **BC6H mode 2 (unsigned)** block (two subsets, 7-bit base, 6-bit
/// deltas). See [`decode_bc6h_two_subset`].
#[must_use]
pub fn decode_bc6h_mode2_unsigned(block: &[u8; 16]) -> [[f32; 3]; 16] {
    decode_bc6h_two_subset(block, &BC6H_MODE2, false)
}

/// Decode a **BC6H mode 2 (signed, `SF16`)** block (two subsets, 7-bit base,
/// 6-bit deltas). See [`decode_bc6h_two_subset`].
#[must_use]
pub fn decode_bc6h_mode2_signed(block: &[u8; 16]) -> [[f32; 3]; 16] {
    decode_bc6h_two_subset(block, &BC6H_MODE2, true)
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

    /// Assemble a single-subset delta block (modes 12/13/14): `mode_bits` is
    /// the 5-bit mode field, `base` the three base-precision endpoint
    /// components, `delta` the three delta-precision fields, `idx` the sixteen
    /// indices (index 0 must be `<= 7`).
    fn make_block_delta(
        mode_bits: u32,
        base_prec: u32,
        delta_bits: u32,
        base: [u32; 3],
        delta: [u32; 3],
        idx: [u8; 16],
    ) -> [u8; 16] {
        let mut w = BitWriter::new();
        let hi_bits = base_prec - 10;
        w.write(mode_bits, 5);
        // Low 10 base bits inline, then per channel: delta then relocated high
        // base bits (matches `decode_bc6h_single_subset_delta`).
        w.write(base[0] & 0x3FF, 10);
        w.write(base[1] & 0x3FF, 10);
        w.write(base[2] & 0x3FF, 10);
        for c in 0..3 {
            w.write(delta[c], delta_bits);
            // High base bits most-significant-first.
            for k in (0..hi_bits).rev() {
                w.write((base[c] >> (10 + k)) & 1, 1);
            }
        }
        w.write(u32::from(idx[0]), 3);
        for &i in idx.iter().skip(1) {
            w.write(u32::from(i), 4);
        }
        assert_eq!(w.pos, 128, "delta-mode fields must fill the block exactly");
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
        // Mode-1 block (low two bits 00) is now a supported two-subset mode;
        // an all-zero block has zero base/deltas, so every texel decodes 0.
        let m1 = [0u8; 16];
        assert_eq!(bc6h_mode_bits(&m1), 0);
        assert_eq!(decode_bc6h_unsigned(&m1), Ok([[0.0f32; 3]; 16]));
        // Mode-3 block (5-bit 0b00010) is a two-subset partitioned mode that is
        // not wired up yet, so it must still report the error rather than guess.
        let mut m3 = [0u8; 16];
        m3[0] = 0b0_0010;
        assert_eq!(bc6h_mode_bits(&m3), 0b00010);
        assert_eq!(
            decode_bc6h_unsigned(&m3),
            Err(Bc6hError::UnsupportedMode(0b00010))
        );
    }

    #[test]
    fn signed_mode11_zero_decodes_to_zero() {
        // Both endpoints zero -> every texel is +0.0 regardless of index.
        let block = make_block11([0; 3], [0; 3], [0u8; 16]);
        let out = decode_bc6h_mode11_signed(&block);
        assert!(out.iter().all(|p| p == &[0.0, 0.0, 0.0]));
    }

    #[test]
    fn signed_mode11_saturates_to_plus_minus_max_half() {
        // 10-bit two's complement: 0x1FF = +511 (max positive) saturates to
        // +65504.0; 0x200 = -512 (max negative) saturates to -65504.0.
        let mut idx = [0u8; 16];
        idx[1] = 15; // weight 64 -> endpoint 1
        let block = make_block11([0x1FF; 3], [0x200; 3], idx);
        let out = decode_bc6h_mode11_signed(&block);
        assert_eq!(out[0], [65504.0, 65504.0, 65504.0]); // e0 (+max)
        assert_eq!(out[1], [-65504.0, -65504.0, -65504.0]); // e1 (-max)
    }

    #[test]
    fn signed_mode11_ramp_is_monotonic_decreasing() {
        // e0 = +max, e1 = -max on R: as the index weight grows the signed R
        // value decreases monotonically from +max toward -max.
        let mut idx = [0u8; 16];
        for (t, slot) in idx.iter_mut().enumerate() {
            *slot = t.min(15) as u8;
        }
        idx[0] = idx[0].min(7); // anchor is 3-bit
        let block = make_block11([0x1FF, 0, 0], [0x200, 0, 0], idx);
        let out = decode_bc6h_mode11_signed(&block);
        for t in 1..16 {
            assert!(
                out[t][0] <= out[t - 1][0],
                "signed R must be monotone at {t}"
            );
        }
        assert!(out.iter().all(|p| p[1] == 0.0 && p[2] == 0.0));
    }

    #[test]
    fn signed_and_unsigned_disagree_on_negative_endpoint() {
        // Raw field 0x200 reads as +512 unsigned but -512 signed: the two
        // decoders must disagree in sign for that endpoint.
        let mut idx = [0u8; 16];
        idx[1] = 15;
        let block = make_block11([0; 3], [0x200; 3], idx);
        let u = decode_bc6h_mode11_unsigned(&block);
        let si = decode_bc6h_mode11_signed(&block);
        assert!(u[1][0] > 0.0, "unsigned reads 0x200 as positive");
        assert!(si[1][0] < 0.0, "signed reads 0x200 as negative");
    }

    #[test]
    fn delta_modes_are_detected_and_dispatch_ok() {
        for &(bits, bp, db) in &[
            (0b00111u32, 11u32, 9u32),
            (0b01011, 12, 8),
            (0b01111, 16, 4),
        ] {
            let block = make_block_delta(bits, bp, db, [0; 3], [0; 3], [0u8; 16]);
            assert_eq!(u32::from(bc6h_mode_bits(&block)), bits);
            assert!(decode_bc6h_unsigned(&block).is_ok());
            assert!(decode_bc6h_signed(&block).is_ok());
        }
    }

    #[test]
    fn delta_zero_base_zero_delta_decodes_to_zero() {
        // Base 0 + delta 0 -> both endpoints 0 -> every texel is 0 regardless
        // of index, for every single-subset delta mode, both signednesses.
        for &(bits, bp, db) in &[
            (0b00111u32, 11u32, 9u32),
            (0b01011, 12, 8),
            (0b01111, 16, 4),
        ] {
            let mut idx = [0u8; 16];
            for (t, slot) in idx.iter_mut().enumerate() {
                *slot = (t % 16).min(15) as u8;
            }
            idx[0] = idx[0].min(7);
            let block = make_block_delta(bits, bp, db, [0; 3], [0; 3], idx);
            let u = decode_bc6h_unsigned(&block).unwrap();
            assert!(
                u.iter().all(|p| p == &[0.0, 0.0, 0.0]),
                "unsigned {bits:#07b}"
            );
            let si = decode_bc6h_signed(&block).unwrap();
            assert!(
                si.iter().all(|p| p == &[0.0, 0.0, 0.0]),
                "signed {bits:#07b}"
            );
        }
    }

    #[test]
    fn delta_positive_delta_raises_endpoint1_monotonically() {
        // Unsigned mode 12: base small, positive delta -> endpoint 1 brighter
        // than endpoint 0, so an index ramp is non-decreasing on R.
        let mut idx = [0u8; 16];
        for (t, slot) in idx.iter_mut().enumerate() {
            *slot = t.min(15) as u8;
        }
        idx[0] = idx[0].min(7);
        // base R = 100 (of 2^11), delta R = +200 -> endpoint1 R = 300.
        let block = make_block_delta(0b00111, 11, 9, [100, 0, 0], [200, 0, 0], idx);
        let out = decode_bc6h_mode12_unsigned(&block);
        for t in 1..16 {
            assert!(out[t][0] >= out[t - 1][0], "R must be monotone at {t}");
        }
        assert!(
            out[15][0] > out[0][0],
            "positive delta must brighten endpoint 1"
        );
    }

    #[test]
    fn delta_wraps_within_base_precision() {
        // Mode 13 (base 12 bits): base = 0, delta = -1 (0xFF in 8 bits) must
        // wrap to the top of the 12-bit range (0xFFF), i.e. endpoint 1 is the
        // brightest value, not an underflow. Index 15 selects endpoint 1.
        let mut idx = [0u8; 16];
        idx[1] = 15;
        let delta_neg1 = (1u32 << 8) - 1; // 0xFF == -1 as a signed 8-bit delta
        let block = make_block_delta(0b01011, 12, 8, [0, 0, 0], [delta_neg1, 0, 0], idx);
        let out = decode_bc6h_mode13_unsigned(&block);
        // (0 + (-1)) & 0xFFF == 0xFFF -> unquantize(0xFFF, 12) is near full range.
        let direct = make_block_delta(0b01011, 12, 8, [0xFFF, 0, 0], [0, 0, 0], idx);
        let want = decode_bc6h_mode13_unsigned(&direct);
        assert_eq!(
            out[1][0], want[1][0],
            "delta -1 from 0 must wrap to top of range"
        );
    }

    #[test]
    fn delta_signed_negative_delta_goes_negative() {
        // Signed mode 12 (base 11 bits, delta 9 bits): base 0, delta -256
        // (field 0x100 is the most-negative 9-bit value). Endpoint 1 wraps to
        // `(0 - 256) & 0x7FF == 0x700`, which sign-extended at 11 bits is -256,
        // a solidly negative unquantized endpoint. Index 15 picks endpoint 1.
        let mut idx = [0u8; 16];
        idx[1] = 15;
        let delta_neg256 = 1u32 << 8; // 0x100 == -256 as a signed 9-bit delta
        let block = make_block_delta(0b00111, 11, 9, [0, 0, 0], [delta_neg256, 0, 0], idx);
        let out = decode_bc6h_mode12_signed(&block);
        assert!(out[1][0] < 0.0, "wrapped endpoint sign-extends negative");
    }

    #[test]
    fn delta_signed_and_unsigned_disagree_on_wrapped_endpoint() {
        // Mode 12: base 0, delta that reconstructs endpoint1 = 0x7FF (top bit
        // of the 11-bit base set). Unsigned reads it as +2047, signed as -1.
        let mut idx = [0u8; 16];
        idx[1] = 15;
        let block = make_block_delta(0b00111, 11, 9, [0x7FF, 0, 0], [0, 0, 0], idx);
        let u = decode_bc6h_mode12_unsigned(&block);
        let si = decode_bc6h_mode12_signed(&block);
        assert!(u[1][0] > 0.0, "unsigned high endpoint is positive");
        assert!(si[1][0] < 0.0, "signed high endpoint is negative");
    }
}
