//! The `EAC` single/dual-channel 11-bit codecs (`EAC_R11` and `EAC_RG11`).
//!
//! `EAC` ("Ericsson Alpha Codec") is the ETC2 family's high-precision scalar
//! codec. It is the AAA-standard GPU encoding for single- and dual-channel
//! linear data where 8 bits is not enough: height / displacement maps, scalar
//! masks, and -- most importantly -- two-channel tangent-space normal maps
//! (`RG11`, the ETC2 analogue of `BC5`). Each channel is an independent 8-byte
//! block; `RG11` simply concatenates the red block then the green block.
//!
//! # Block layout (per channel, 8 bytes, big-endian)
//! * `base` -- bits `63..56`, the 8-bit base codeword.
//! * `multiplier` -- bits `55..52`, 4-bit palette scale.
//! * `table_index` -- bits `51..48`, selects one of 16 intensity-modifier rows.
//! * `indices` -- bits `47..0`, sixteen 3-bit per-texel palette indices, texel
//!   `p = x*4 + y` (column-major, as in every ETC block) in bits
//!   `[45 - 3p ..= 47 - 3p]`, MSB first.
//!
//! # Decode (unsigned, 11-bit output in `0..=2047`)
//! ```text
//! m = MODIFIER[table_index][index]
//! value = base * 8 + 4 + m * multiplier * 8     (multiplier != 0)
//! value = base * 8 + 4 + m                       (multiplier == 0)
//! value = clamp(value, 0, 2047)
//! ```
//! The `multiplier == 0` fallback (adding the raw modifier) matches the Khronos
//! Data Format Specification exactly; without it a zero multiplier would make
//! the whole block collapse to the single base value.
//!
//! Pure integer arithmetic, no AI/ML path; every value is reproduced
//! bit-exactly by an Apple M2 Metal hardware decode (see the
//! `prism_render_material_gpu` EAC parity test).
//!
//! # References
//! * Khronos Data Format Specification 1.3, EAC block decode.
//! * Vulkan `VK_FORMAT_EAC_R11_UNORM_BLOCK` / `VK_FORMAT_EAC_R11G11_UNORM_BLOCK`.

/// The 16 EAC intensity-modifier rows (8 entries each), shared with the ETC2
/// alpha block. Index `0..=3` are the negative half, `4..=7` the positive half.
const MODIFIER: [[i32; 8]; 16] = [
    [-3, -6, -9, -15, 2, 5, 8, 14],
    [-3, -7, -10, -13, 2, 6, 9, 12],
    [-2, -5, -8, -13, 1, 4, 7, 12],
    [-2, -4, -6, -13, 1, 3, 5, 12],
    [-3, -6, -8, -12, 2, 5, 7, 11],
    [-3, -7, -9, -11, 2, 6, 8, 10],
    [-4, -7, -8, -11, 3, 6, 7, 10],
    [-3, -5, -8, -11, 2, 4, 7, 10],
    [-2, -6, -8, -10, 1, 5, 7, 9],
    [-2, -5, -8, -10, 1, 4, 7, 9],
    [-2, -4, -8, -10, 1, 3, 7, 9],
    [-2, -5, -7, -10, 1, 4, 6, 9],
    [-3, -4, -7, -10, 2, 3, 6, 9],
    [-1, -2, -3, -10, 0, 1, 2, 9],
    [-4, -6, -8, -9, 3, 5, 7, 8],
    [-3, -5, -7, -9, 2, 4, 6, 8],
];

/// Clamp an intermediate value to the unsigned 11-bit range.
#[inline]
fn clamp11(v: i32) -> u16 {
    v.clamp(0, 2047) as u16
}

/// Decode one 8-byte unsigned `EAC_R11` block into sixteen 11-bit values
/// (`0..=2047`), row-major with texel `t = y*4 + x`.
///
/// This is the single-channel core reused by [`decode_eac_rg11_unorm`].
#[must_use]
pub fn decode_eac_r11_unorm(block: &[u8; 8]) -> [u16; 16] {
    let bits = u64::from_be_bytes(*block);
    let base = ((bits >> 56) & 0xFF) as i32;
    let mult = ((bits >> 52) & 0x0F) as i32;
    let table = ((bits >> 48) & 0x0F) as usize;
    let row = &MODIFIER[table];

    let mut out = [0u16; 16];
    for x in 0..4u32 {
        for y in 0..4u32 {
            let p = x * 4 + y;
            let shift = 45 - 3 * p;
            let index = ((bits >> shift) & 0x7) as usize;
            let m = row[index];
            let value = if mult != 0 {
                base * 8 + 4 + m * mult * 8
            } else {
                base * 8 + 4 + m
            };
            out[(y * 4 + x) as usize] = clamp11(value);
        }
    }
    out
}

/// Decode one 16-byte unsigned `EAC_RG11` block into sixteen `[r, g]` pairs of
/// 11-bit values. The red channel occupies bytes `0..8`, green bytes `8..16`;
/// each is an independent [`decode_eac_r11_unorm`] block.
#[must_use]
pub fn decode_eac_rg11_unorm(block: &[u8; 16]) -> [[u16; 2]; 16] {
    let mut r_block = [0u8; 8];
    let mut g_block = [0u8; 8];
    r_block.copy_from_slice(&block[0..8]);
    g_block.copy_from_slice(&block[8..16]);
    let r = decode_eac_r11_unorm(&r_block);
    let g = decode_eac_r11_unorm(&g_block);
    let mut out = [[0u16; 2]; 16];
    for t in 0..16 {
        out[t] = [r[t], g[t]];
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_multiplier_adds_raw_modifier() {
        // base = 100, multiplier = 0, table 0; all indices 0 -> modifier -3.
        // value = 100*8 + 4 + (-3) = 801.
        let block = [100, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00];
        let out = decode_eac_r11_unorm(&block);
        for v in &out {
            assert_eq!(*v, 801);
        }
    }

    #[test]
    fn multiplier_scales_modifier_by_eight() {
        // base = 100, multiplier = 2 (byte1 high nibble), table 0; index 0 ->
        // modifier -3. value = 800 + 4 + (-3)*2*8 = 804 - 48 = 756.
        let block = [100, 0x20, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00];
        let out = decode_eac_r11_unorm(&block);
        for v in &out {
            assert_eq!(*v, 756);
        }
    }

    #[test]
    fn clamps_into_eleven_bit_range() {
        // base = 255, multiplier = 15, table 0, all indices 7 -> modifier +14.
        // value = 255*8 + 4 + 14*15*8 = 2044 + 1680 = 3724 -> clamps to 2047.
        let block = [255, 0xF0, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF];
        let out = decode_eac_r11_unorm(&block);
        for v in &out {
            assert_eq!(*v, 2047);
        }
    }

    #[test]
    fn rg11_splits_into_two_independent_channels() {
        // Red block = zero_multiplier case (-> 801); green block distinct base.
        let mut block = [0u8; 16];
        block[0] = 100; // red base
        block[8] = 50; // green base, multiplier 0, table 0, idx 0 -> 50*8+4-3=401
        let out = decode_eac_rg11_unorm(&block);
        for pair in &out {
            assert_eq!(*pair, [801, 401]);
        }
    }
}
