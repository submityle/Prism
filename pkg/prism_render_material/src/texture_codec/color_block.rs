//! The 8-byte BC1-style colour block shared by BC1 and the colour half of BC3.
//!
//! A colour block packs two `RGB565` endpoint colours and sixteen 2-bit palette
//! indices (one per texel of a 4x4 tile). The decoder expands the endpoints to
//! `RGB888`, derives a 4-entry palette, and looks up each texel. This is the
//! classic S3TC/DXT1 colour decode, matching the Vulkan/D3D `BC1`/`BC3` colour
//! definition. Pure integer arithmetic, no AI/ML -- a CPU golden reproduces a
//! GPU twin within the format's documented interpolation tolerance.
//!
//! # Conventions
//! * Blocks are little-endian: endpoints are `u16` from bytes `[0..2]`,`[2..4]`;
//!   the 32-bit index word is bytes `[4..8]`, 2 bits per texel, texel `t`
//!   (`t = y*4 + x`) in bits `[2t, 2t+1]`, LSB = texel `(0,0)`.
//! * Interpolated palette entries use the truncating `(2a + b)/3` form. The
//!   exact rounding of the thirds is **implementation-defined within +/-1 LSB**
//!   across GPUs (a documented BC1 ambiguity), so tests assert ordering/bounds
//!   rather than exact interpolated LSBs.
//! * `punchthrough` selects BC1 semantics: when `true` and `color0 <= color1`,
//!   index 3 is transparent black (1-bit alpha). BC3 colour always passes
//!   `false` (always 4 opaque colours).
//!
//! # References
//! * Khronos Data Format Spec 1.3, S3TC/BC1 block decode.
//! * Vulkan `VK_FORMAT_BC1_*` / D3D `DXGI_FORMAT_BC1_*` definitions.

/// Expand a packed `RGB565` colour to `RGB888` via bit replication.
#[inline]
#[must_use]
pub fn rgb565_to_rgb888(c: u16) -> [u8; 3] {
    let r5 = ((c >> 11) & 0x1F) as u8;
    let g6 = ((c >> 5) & 0x3F) as u8;
    let b5 = (c & 0x1F) as u8;
    // Replicate the high bits into the low bits so 0x1F -> 0xFF exactly.
    let r8 = (r5 << 3) | (r5 >> 2);
    let g8 = (g6 << 2) | (g6 >> 4);
    let b8 = (b5 << 3) | (b5 >> 2);
    [r8, g8, b8]
}

#[inline]
fn read_u16_le(b: &[u8; 8], off: usize) -> u16 {
    u16::from(b[off]) | (u16::from(b[off + 1]) << 8)
}

#[inline]
fn third(a: u8, b: u8) -> u8 {
    // (2a + b)/3, truncating; inputs are u8 so the sum fits in u16.
    ((2 * u16::from(a) + u16::from(b)) / 3) as u8
}

#[inline]
fn half(a: u8, b: u8) -> u8 {
    ((u16::from(a) + u16::from(b)) / 2) as u8
}

/// Decode a BC1-style colour block to 16 RGBA texels (row-major, texel
/// `t = y*4 + x`).
///
/// `punchthrough == true` enables DXT1 1-bit alpha: in the `color0 <= color1`
/// branch, index 3 decodes to transparent black. BC3 colour passes `false`.
#[must_use]
pub fn decode_color_block(block: &[u8; 8], punchthrough: bool) -> [[u8; 4]; 16] {
    let c0 = read_u16_le(block, 0);
    let c1 = read_u16_le(block, 2);
    let e0 = rgb565_to_rgb888(c0);
    let e1 = rgb565_to_rgb888(c1);

    let mut palette = [[0u8; 4]; 4];
    palette[0] = [e0[0], e0[1], e0[2], 255];
    palette[1] = [e1[0], e1[1], e1[2], 255];

    if !punchthrough || c0 > c1 {
        // 4-colour opaque mode: two interpolated thirds.
        palette[2] = [third(e0[0], e1[0]), third(e0[1], e1[1]), third(e0[2], e1[2]), 255];
        palette[3] = [third(e1[0], e0[0]), third(e1[1], e0[1]), third(e1[2], e0[2]), 255];
    } else {
        // 3-colour + transparent-black mode.
        palette[2] = [half(e0[0], e1[0]), half(e0[1], e1[1]), half(e0[2], e1[2]), 255];
        palette[3] = [0, 0, 0, 0];
    }

    let indices = u32::from_le_bytes([block[4], block[5], block[6], block[7]]);
    let mut out = [[0u8; 4]; 16];
    for (t, texel) in out.iter_mut().enumerate() {
        let idx = ((indices >> (2 * t)) & 0x3) as usize;
        *texel = palette[idx];
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rgb565_extremes_bit_replicate() {
        assert_eq!(rgb565_to_rgb888(0x0000), [0, 0, 0]);
        assert_eq!(rgb565_to_rgb888(0xFFFF), [255, 255, 255]);
    }

    #[test]
    fn endpoints_decode_exactly_at_indices_0_and_1() {
        // color0 = white (0xFFFF), color1 = black (0x0000); c0 > c1 -> 4-colour.
        // indices: texel0 -> 0 (white), texel1 -> 1 (black).
        let block = [0xFF, 0xFF, 0x00, 0x00, 0b0000_0100, 0, 0, 0];
        let out = decode_color_block(&block, true);
        assert_eq!(out[0], [255, 255, 255, 255]);
        assert_eq!(out[1], [0, 0, 0, 255]);
    }

    #[test]
    fn interpolated_entries_lie_between_endpoints() {
        let block = [0xFF, 0xFF, 0x00, 0x00, 0, 0, 0, 0];
        let out = decode_color_block(&block, true);
        // index-2 and index-3 colours must sit inside [black, white].
        let two = decode_color_block(&[0xFF, 0xFF, 0x00, 0x00, 0b10, 0, 0, 0], true)[0];
        let three = decode_color_block(&[0xFF, 0xFF, 0x00, 0x00, 0b11, 0, 0, 0], true)[0];
        assert!(two[0] > 0 && two[0] < 255, "two={two:?}");
        assert!(three[0] > 0 && three[0] < 255, "three={three:?}");
        // 2/3-white should be brighter than 1/3-white.
        assert!(two[0] > three[0]);
        let _ = out;
    }

    #[test]
    fn punchthrough_index3_is_transparent_when_c0_le_c1() {
        // color0 = black, color1 = white -> c0 <= c1 -> 3-colour + transparent.
        let block = [0x00, 0x00, 0xFF, 0xFF, 0b11, 0, 0, 0];
        let out = decode_color_block(&block, true);
        assert_eq!(out[0], [0, 0, 0, 0]);
    }

    #[test]
    fn bc3_mode_never_transparent() {
        // Same bytes, punchthrough = false -> index 3 is an opaque interpolation.
        let block = [0x00, 0x00, 0xFF, 0xFF, 0b11, 0, 0, 0];
        let out = decode_color_block(&block, false);
        assert_eq!(out[0][3], 255);
    }
}
