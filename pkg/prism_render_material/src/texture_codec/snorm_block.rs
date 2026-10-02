//! The 8-byte BC4-style **signed** single-channel block (RGTC2 signed / the
//! `BC4_SNORM` and `BC5_SNORM` D3D/Vulkan formats).
//!
//! This is the signed sibling of [`alpha_block`](super::alpha_block). The byte
//! layout and 3-bit index word are identical to the unsigned block; only the
//! endpoint *interpretation* differs: the two endpoint bytes are read as 8-bit
//! two's-complement values and the palette is built in the signed domain.
//! Signed data decodes to `i8` in `-127..=127` (mapping to the normalized
//! `-1.0..=1.0` range), which is why these formats are stored separately from
//! the `u8` unsigned path rather than reusing [`channel_palette`] with a cast.
//!
//! `BC4_SNORM` / `BC5_SNORM` are the AAA-standard encoding for signed tangent-
//! space data -- object-space normal maps, signed-distance / displacement
//! channels, and motion-vector packing -- where the unsigned `0..=255` block
//! would waste half its precision or need a bias.
//!
//! # Conventions
//! * Endpoints `r0 = block[0]`, `r1 = block[1]` are two's-complement `i8`. The
//!   bit pattern `0x80` (`-128`) is remapped to `-127` so the representable
//!   range is symmetric about zero, exactly as the D3D/Khronos spec requires
//!   (`-128` and `-127` would otherwise both map to `-1.0`).
//! * When `r0 > r1` (signed compare): eight-value mode (`r2..r7` interpolate in
//!   sevenths). Otherwise: six-value mode (`r2..r5` interpolate in fifths, with
//!   hard terminals `r6 = -127` and `r7 = 127`).
//! * The 48-bit index word is bytes `[2..8]`, 3 bits per texel, texel
//!   `t = y*4 + x` in bits `[3t, 3t+2]`, LSB = texel `(0,0)` -- identical to the
//!   unsigned block.
//! * Interpolation truncates toward zero; the interpolated LSBs are
//!   **implementation-defined within +/-1 LSB** across GPUs, so tests assert
//!   ordering / bounds / terminals, not exact interpolated values.
//!
//! # References
//! * Khronos Data Format Specification 1.3, RGTC signed block decode.
//! * Vulkan `VK_FORMAT_BC4_SNORM_BLOCK` / `VK_FORMAT_BC5_SNORM_BLOCK`;
//!   D3D `DXGI_FORMAT_BC4_SNORM` / `DXGI_FORMAT_BC5_SNORM`.

/// Read an endpoint byte as a two's-complement `i8`, remapping the reserved
/// `-128` pattern to `-127` so the signed range is symmetric about zero.
#[inline]
#[must_use]
pub fn signed_endpoint(byte: u8) -> i8 {
    let v = i8::from_le_bytes([byte]);
    if v == i8::MIN {
        -127
    } else {
        v
    }
}

/// Signed interpolation `(num0*r0 + num1*r1) / den`, clamped to `-127..=127`.
#[inline]
fn lerp_s(r0: i8, r1: i8, num0: i16, num1: i16, den: i16) -> i8 {
    let v = (num0 * i16::from(r0) + num1 * i16::from(r1)) / den;
    i8::try_from(v.clamp(-127, 127)).unwrap_or(0)
}

/// Build the 8-entry signed value palette from the two signed endpoints.
///
/// `r0`/`r1` must already be normalized through [`signed_endpoint`] (no raw
/// `-128`). Mirrors [`channel_palette`](super::alpha_block::channel_palette) in
/// the signed domain, with `-127`/`127` terminals in the six-value mode.
#[must_use]
pub fn signed_channel_palette(r0: i8, r1: i8) -> [i8; 8] {
    let mut p = [0i8; 8];
    p[0] = r0;
    p[1] = r1;
    if r0 > r1 {
        // Eight-value mode: r2..r7 interpolate in sevenths.
        for (i, slot) in p.iter_mut().enumerate().take(8).skip(2) {
            let k = i16::try_from(i).unwrap_or(0) - 1; // r2 -> 1 .. r7 -> 6
            *slot = lerp_s(r0, r1, 7 - k, k, 7);
        }
    } else {
        // Six-value mode: r2..r5 interpolate in fifths, then -127 and 127.
        for (i, slot) in p.iter_mut().enumerate().take(6).skip(2) {
            let k = i16::try_from(i).unwrap_or(0) - 1; // r2 -> 1 .. r5 -> 4
            *slot = lerp_s(r0, r1, 5 - k, k, 5);
        }
        p[6] = -127;
        p[7] = 127;
    }
    p
}

/// Decode a BC4-style **signed** single-channel block to 16 `i8` values
/// (row-major, texel `t = y*4 + x`).
#[must_use]
pub fn decode_signed_channel_block(block: &[u8; 8]) -> [i8; 16] {
    let palette = signed_channel_palette(signed_endpoint(block[0]), signed_endpoint(block[1]));
    // 48-bit little-endian index word from bytes [2..8].
    let mut bits: u64 = 0;
    for (i, &byte) in block[2..8].iter().enumerate() {
        bits |= u64::from(byte) << (8 * i);
    }
    let mut out = [0i8; 16];
    for (t, v) in out.iter_mut().enumerate() {
        let idx = ((bits >> (3 * t)) & 0x7) as usize;
        *v = palette[idx];
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn endpoint_remaps_reserved_minus_128() {
        assert_eq!(signed_endpoint(0x80), -127);
        assert_eq!(signed_endpoint(0x81), -127);
        assert_eq!(signed_endpoint(0x7F), 127);
        assert_eq!(signed_endpoint(0x00), 0);
    }

    #[test]
    fn eight_value_mode_endpoints_and_order() {
        // r0 > r1 -> eight-value mode; palette strictly spans [r1, r0].
        let p = signed_channel_palette(100, -100);
        assert_eq!(p[0], 100);
        assert_eq!(p[1], -100);
        // r2..r7 are strictly decreasing from near r0 toward near r1.
        for w in 2..7 {
            assert!(p[w] > p[w + 1], "palette {p:?} not monotone at {w}");
        }
        assert!(p.iter().all(|&v| (-127..=127).contains(&v)));
    }

    #[test]
    fn six_value_mode_has_hard_terminals() {
        // r0 <= r1 -> six-value mode with -127 / 127 terminals.
        let p = signed_channel_palette(-100, 100);
        assert_eq!(p[0], -100);
        assert_eq!(p[1], 100);
        assert_eq!(p[6], -127);
        assert_eq!(p[7], 127);
    }

    #[test]
    fn index_zero_and_one_read_endpoints() {
        // All indices 0 -> every texel = r0.
        let mut block = [0u8; 8];
        block[0] = 40; // r0 = 40
        block[1] = u8::from_le_bytes((-20i8).to_le_bytes()); // r1 = -20 (40 > -20 -> eight-value)
        let out = decode_signed_channel_block(&block);
        assert!(out.iter().all(|&v| v == 40));
        // All indices 1 -> every texel = r1.
        let mut bits: u64 = 0;
        for t in 0..16 {
            bits |= 1u64 << (3 * t);
        }
        let bb = bits.to_le_bytes();
        block[2..8].copy_from_slice(&bb[0..6]);
        let out = decode_signed_channel_block(&block);
        assert!(out.iter().all(|&v| v == -20));
    }

    #[test]
    fn constant_block_round_trips_through_both_endpoints() {
        // r0 == r1 -> six-value branch; palette[0..6] all equal, so any
        // non-terminal index decodes the constant value.
        let p = signed_channel_palette(55, 55);
        for v in &p[0..6] {
            assert_eq!(*v, 55);
        }
    }

    #[test]
    fn decode_is_deterministic() {
        let block = [0x7Fu8, 0x80, 0x12, 0x34, 0x56, 0x78, 0x9A, 0xBC];
        assert_eq!(
            decode_signed_channel_block(&block),
            decode_signed_channel_block(&block)
        );
    }
}
