//! The 8-byte BC4-style single-channel block shared by BC4, the alpha half of
//! BC3, and both channels of BC5.
//!
//! A single-channel block packs two `u8` endpoint values and sixteen 3-bit
//! palette indices (one per texel of a 4x4 tile). The decoder builds an
//! 8-entry palette -- either eight interpolated values, or six interpolated
//! plus hard `0`/`255` -- selected by endpoint ordering, then looks up each
//! texel. This is the classic RGTC/BC4 decode, matching the Vulkan/D3D
//! definition. Pure integer arithmetic, no AI/ML path.
//!
//! # Conventions
//! * Little-endian: `r0 = block[0]`, `r1 = block[1]`; the 48-bit index word is
//!   bytes `[2..8]`, 3 bits per texel, texel `t` (`t = y*4 + x`) in bits
//!   `[3t, 3t+2]`, LSB = texel `(0,0)`.
//! * When `r0 > r1`: eight-value mode (`r2..r7` interpolate in sevenths).
//!   Otherwise: six-value mode (`r2..r5` interpolate in fifths, `r6 = 0`,
//!   `r7 = 255`).
//! * Interpolation truncates (`((n-i)*r0 + i*r1)/n`); the thirds/fifths/
//!   sevenths rounding is **implementation-defined within +/-1 LSB** across
//!   GPUs, so tests assert ordering/bounds, not exact interpolated LSBs.
//!
//! # References
//! * Khronos Data Format Spec 1.3, RGTC/BC4 block decode.
//! * Vulkan `VK_FORMAT_BC4_*` / D3D `DXGI_FORMAT_BC4_*` definitions.

#[inline]
fn lerp(r0: u8, r1: u8, num0: u16, num1: u16, den: u16) -> u8 {
    ((num0 * u16::from(r0) + num1 * u16::from(r1)) / den) as u8
}

/// Build the 8-entry value palette from the two endpoints.
#[must_use]
pub fn channel_palette(r0: u8, r1: u8) -> [u8; 8] {
    let mut p = [0u8; 8];
    p[0] = r0;
    p[1] = r1;
    if r0 > r1 {
        // Eight-value mode: r2..r7 interpolate in sevenths.
        for (i, slot) in p.iter_mut().enumerate().take(8).skip(2) {
            let k = i as u16 - 1; // r2 -> 1 .. r7 -> 6
            *slot = lerp(r0, r1, 7 - k, k, 7);
        }
    } else {
        // Six-value mode: r2..r5 interpolate in fifths, then 0 and 255.
        for (i, slot) in p.iter_mut().enumerate().take(6).skip(2) {
            let k = i as u16 - 1; // r2 -> 1 .. r5 -> 4
            *slot = lerp(r0, r1, 5 - k, k, 5);
        }
        p[6] = 0;
        p[7] = 255;
    }
    p
}

/// Decode a BC4-style single-channel block to 16 values (row-major, texel
/// `t = y*4 + x`).
#[must_use]
pub fn decode_channel_block(block: &[u8; 8]) -> [u8; 16] {
    let palette = channel_palette(block[0], block[1]);
    // 48-bit little-endian index word from bytes [2..8].
    let mut bits: u64 = 0;
    for (i, &byte) in block[2..8].iter().enumerate() {
        bits |= u64::from(byte) << (8 * i);
    }
    let mut out = [0u8; 16];
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
    fn eight_value_mode_endpoints_and_order() {
        // r0 > r1 -> eight-value mode.
        let p = channel_palette(255, 0);
        assert_eq!(p[0], 255);
        assert_eq!(p[1], 0);
        // Interpolants strictly descend from r0 toward r1.
        for i in 2..7 {
            assert!(p[i] > p[i + 1], "p={p:?}");
        }
        assert!(p[2] < 255 && p[7] > 0);
    }

    #[test]
    fn six_value_mode_has_hard_zero_and_one() {
        // r0 <= r1 -> six-value mode with 0/255 terminals.
        let p = channel_palette(0, 255);
        assert_eq!(p[0], 0);
        assert_eq!(p[1], 255);
        assert_eq!(p[6], 0);
        assert_eq!(p[7], 255);
    }

    #[test]
    fn index_zero_reads_endpoint0() {
        // All indices 0 -> every texel is r0.
        let block = [200, 10, 0, 0, 0, 0, 0, 0];
        let out = decode_channel_block(&block);
        assert!(out.iter().all(|&v| v == 200));
    }

    #[test]
    fn index_one_reads_endpoint1() {
        // texel0 index 1 -> r1.
        let block = [200, 10, 0b001, 0, 0, 0, 0, 0];
        let out = decode_channel_block(&block);
        assert_eq!(out[0], 10);
    }

    #[test]
    fn six_value_mode_index6_and_7_are_hard_limits() {
        // r0 <= r1 so palette[6]=0, palette[7]=255. texel0 idx 6, texel1 idx 7.
        // indices: t0=6 (0b110), t1=7 (0b111) -> bits = 6 | (7<<3) = 0b111_110.
        let bits: u64 = 6 | (7 << 3);
        let b = bits.to_le_bytes();
        let block = [0, 255, b[0], b[1], b[2], b[3], b[4], b[5]];
        let out = decode_channel_block(&block);
        assert_eq!(out[0], 0);
        assert_eq!(out[1], 255);
    }
}
