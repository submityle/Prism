//! ETC2 / EAC block decoders (the Ericsson texture-compression family).
//!
//! ETC2 is the mandatory baseline GPU compression on `OpenGL ES 3.0` / Vulkan
//! and the mobile-AAA counterpart to desktop BC: a 4x4 `RGB` block packs into
//! 8 bytes. This module decodes the `ETC2_RGB8` colour format. Each block
//! selects one of five sub-formats; the two **base** (ETC1-compatible) modes
//! are decoded here:
//!
//! * **Individual** -- two `RGB444` base colours (one per 2x4 / 4x2 subblock).
//! * **Differential** -- one `RGB555` base colour plus a signed 3-bit per-channel
//!   delta for the second subblock.
//!
//! The three ETC2-only extensions (`T`, `H`, planar), which a differential block
//! signals by letting a `base + delta` channel fall outside `0..=31`, are
//! classified by [`etc2_rgb8_mode`] and fully decoded by [`decode_etc2_rgb8`].
//! Every mode's bit layout was pinned bit-exactly against an M2 Metal hardware
//! decode (see the `prism_render_material_gpu` ETC2 parity test); the
//! [`Etc2Error`] type is retained for the format family's fallible decoders
//! (future `EAC` / punch-through-alpha variants).
//!
//! Pure integer decode, no AI/ML path; a GPU hardware decode reproduces every
//! texel bit-exactly (ETC has no interpolation tolerance -- outputs are exact).
//!
//! # Conventions
//! * Output is row-major `RGBA8`, texel `t = y*4 + x`, `t in [0, 16)`, with a
//!   constant opaque `A = 255` (`RGB8` carries no alpha).
//! * The block's 8 bytes are a big-endian 64-bit field; bit 63 is the MSB of
//!   byte 0, matching the Khronos bit numbering.
//!
//! # References
//! * Khronos Data Format Specification 1.3, section "ETC2 compressed texture
//!   image formats"; `OpenGL ES 3.0` specification, ETC2/EAC appendix.
//! * Strom & Pettersson, "iPACKMAN: High-Quality, Low-Complexity Texture
//!   Compression for Mobile Phones" (2007), the ETC1 base scheme.

/// Which of the five `ETC2_RGB8` sub-formats a block encodes.
///
/// The two base modes are ETC1-compatible; `T`, `H` and planar are the ETC2
/// extensions, reached when a differential block's `base + delta` leaves the
/// `0..=31` range on the red, green or blue channel respectively.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Etc2Mode {
    /// Two independent `RGB444` subblock colours (`diff` bit clear).
    Individual,
    /// One `RGB555` colour plus a signed 3-bit per-channel delta (`diff` set,
    /// every `base + delta` in range).
    Differential,
    /// `T` mode: two paint colours + a distance table (red channel overflow).
    T,
    /// `H` mode: two paint colours + a distance table (green channel overflow).
    H,
    /// Planar mode: three colours bilinearly interpolated (blue overflow).
    Planar,
}

/// Error returned by [`decode_etc2_rgb8`] for a block whose sub-format is not
/// yet decoded by this module.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Etc2Error {
    /// A valid but not-yet-implemented ETC2 extension (`T`, `H` or planar).
    /// Only the ETC1-compatible base modes are decoded today; the extensions
    /// require their colour-extraction and distance tables (tracked as a
    /// follow-up) and are reported here rather than mis-decoded.
    UnsupportedMode(Etc2Mode),
}

/// ETC1 intensity-modifier table (Khronos). Row = 3-bit table codeword, column
/// = 2-bit pixel index `(msb << 1) | lsb`: index 0/1 add a small/large positive
/// offset, 2/3 the negated pair, so the index MSB is the sign bit.
const ETC1_MODIFIER: [[i32; 4]; 8] = [
    [2, 8, -2, -8],
    [5, 17, -5, -17],
    [9, 29, -9, -29],
    [13, 42, -13, -42],
    [18, 60, -18, -60],
    [24, 80, -24, -80],
    [33, 106, -33, -106],
    [47, 183, -47, -183],
];

/// Extract the inclusive big-endian bit range `[lo, hi]` (bit 63 = MSB) as a
/// right-aligned value.
fn field(bits: u64, hi: u32, lo: u32) -> u32 {
    let width = hi - lo + 1;
    let mask = if width >= 32 {
        u32::MAX
    } else {
        (1u32 << width) - 1
    };
    ((bits >> lo) as u32) & mask
}

/// Interpret a 3-bit field as a two's-complement signed delta in `-4..=3`.
fn signed3(v: u32) -> i32 {
    let v = i32::try_from(v & 0x7).unwrap_or(0);
    if v >= 4 {
        v - 8
    } else {
        v
    }
}

/// Replicate a 4-bit channel to 8 bits (`RGB444` base colour expansion).
fn ext4(v: u32) -> i32 {
    let v = i32::try_from(v & 0xF).unwrap_or(0);
    (v << 4) | v
}

/// Replicate a 5-bit channel to 8 bits (`RGB555` base colour expansion).
fn ext5(v: i32) -> i32 {
    let v = v & 0x1F;
    (v << 3) | (v >> 2)
}

/// Clamp an `i32` colour sum into a `u8` channel.
fn clamp8(v: i32) -> u8 {
    (v.clamp(0, 255) & 0xFF) as u8
}

/// Classify an `ETC2_RGB8` block's sub-format without fully decoding it.
///
/// The `diff` bit (bit 33) separates individual from differential; a
/// differential block then falls into one of the ETC2 extensions when its
/// `base + delta` escapes `0..=31` on red (`T`), green (`H`) or blue (planar),
/// checked in that priority order (matching the Khronos decode).
#[must_use]
pub fn etc2_rgb8_mode(block: &[u8; 8]) -> Etc2Mode {
    let bits = u64::from_be_bytes(*block);
    if field(bits, 33, 33) == 0 {
        return Etc2Mode::Individual;
    }
    let r = i32::try_from(field(bits, 63, 59)).unwrap_or(0);
    let g = i32::try_from(field(bits, 55, 51)).unwrap_or(0);
    let b = i32::try_from(field(bits, 47, 43)).unwrap_or(0);
    let dr = signed3(field(bits, 58, 56));
    let dg = signed3(field(bits, 50, 48));
    let db = signed3(field(bits, 42, 40));
    if !(0..=31).contains(&(r + dr)) {
        Etc2Mode::T
    } else if !(0..=31).contains(&(g + dg)) {
        Etc2Mode::H
    } else if !(0..=31).contains(&(b + db)) {
        Etc2Mode::Planar
    } else {
        Etc2Mode::Differential
    }
}

/// Decode one 8-byte `ETC2_RGB8` block into sixteen opaque `RGBA8` texels.
///
/// All five sub-formats are decoded: the two ETC1-compatible base modes
/// (individual, differential) and the three ETC2 extensions (`T`, `H`, planar),
/// each selected by [`etc2_rgb8_mode`]. Output is bit-exact against hardware.
///
/// # Errors
/// Currently infallible for `ETC2_RGB8` (every bit pattern is a valid block);
/// the [`Result`] is kept for API symmetry with the format family's fallible
/// decoders (`EAC` / punch-through alpha).
#[allow(clippy::unnecessary_wraps)]
pub fn decode_etc2_rgb8(block: &[u8; 8]) -> Result<[[u8; 4]; 16], Etc2Error> {
    let bits = u64::from_be_bytes(*block);
    Ok(match etc2_rgb8_mode(block) {
        Etc2Mode::Individual | Etc2Mode::Differential => decode_base(bits),
        Etc2Mode::T => decode_t(bits),
        Etc2Mode::H => decode_h(bits),
        Etc2Mode::Planar => decode_planar(bits),
    })
}

/// Decode an ETC1-compatible base block (individual or differential) assuming
/// the caller has already confirmed a non-overflow mode.
fn decode_base(bits: u64) -> [[u8; 4]; 16] {
    let flip = field(bits, 32, 32) == 1;
    let differential = field(bits, 33, 33) == 1;

    let (c1, c2): ([i32; 3], [i32; 3]) = if differential {
        let r = i32::try_from(field(bits, 63, 59)).unwrap_or(0);
        let g = i32::try_from(field(bits, 55, 51)).unwrap_or(0);
        let b = i32::try_from(field(bits, 47, 43)).unwrap_or(0);
        let dr = signed3(field(bits, 58, 56));
        let dg = signed3(field(bits, 50, 48));
        let db = signed3(field(bits, 42, 40));
        (
            [ext5(r), ext5(g), ext5(b)],
            [ext5(r + dr), ext5(g + dg), ext5(b + db)],
        )
    } else {
        (
            [
                ext4(field(bits, 63, 60)),
                ext4(field(bits, 55, 52)),
                ext4(field(bits, 47, 44)),
            ],
            [
                ext4(field(bits, 59, 56)),
                ext4(field(bits, 51, 48)),
                ext4(field(bits, 43, 40)),
            ],
        )
    };

    let cw1 = field(bits, 39, 37) as usize;
    let cw2 = field(bits, 36, 34) as usize;

    let mut out = [[0u8; 4]; 16];
    for p in 0..16u32 {
        let x = p >> 2;
        let y = p & 3;
        let in_sub1 = if flip { y < 2 } else { x < 2 };
        let (base, cw) = if in_sub1 { (c1, cw1) } else { (c2, cw2) };

        let lsb = field(bits, p, p);
        let msb = field(bits, p + 16, p + 16);
        let idx = ((msb << 1) | lsb) as usize;
        let m = ETC1_MODIFIER[cw][idx];

        let t = (y * 4 + x) as usize;
        out[t] = [
            clamp8(base[0] + m),
            clamp8(base[1] + m),
            clamp8(base[2] + m),
            255,
        ];
    }
    out
}

/// ETC2 `T`/`H` distance table (Khronos). The 3-bit distance selector indexes
/// the per-channel `+/-` offset applied to the paint colours.
const ETC2_DISTANCE: [i32; 8] = [3, 6, 11, 16, 23, 32, 41, 64];

/// Replicate a 6-bit channel to 8 bits (planar `R`/`B` expansion).
fn ext6(v: u32) -> i32 {
    let v = i32::try_from(v & 0x3F).unwrap_or(0);
    (v << 2) | (v >> 4)
}

/// Replicate a 7-bit channel to 8 bits (planar `G` expansion).
fn ext7(v: u32) -> i32 {
    let v = i32::try_from(v & 0x7F).unwrap_or(0);
    (v << 1) | (v >> 6)
}

/// Select the 2-bit paint index for texel `(x, y)` from the 32 pixel-index
/// bits (identical layout to the ETC1 base modes).
fn pixel_index(bits: u64, x: u32, y: u32) -> usize {
    let p = x * 4 + y;
    let lsb = field(bits, p, p);
    let msb = field(bits, p + 16, p + 16);
    ((msb << 1) | lsb) as usize
}

/// Decode an ETC2 `T`-mode block: two `RGB444` paint anchors plus a distance.
///
/// Paint palette is `{C0, C1 + d, C1, C1 - d}`; each texel's 2-bit index
/// selects an entry. Reached when the differential red channel overflows.
fn decode_t(bits: u64) -> [[u8; 4]; 16] {
    let r1 = (field(bits, 60, 59) << 2) | field(bits, 57, 56);
    let g1 = field(bits, 55, 52);
    let b1 = field(bits, 51, 48);
    let r2 = field(bits, 47, 44);
    let g2 = field(bits, 43, 40);
    let b2 = field(bits, 39, 36);
    let dist = ((field(bits, 35, 34) << 1) | field(bits, 32, 32)) as usize;
    let d = ETC2_DISTANCE[dist & 7];

    let c0 = [ext4(r1), ext4(g1), ext4(b1)];
    let c1 = [ext4(r2), ext4(g2), ext4(b2)];
    let paint = [
        [clamp8(c0[0]), clamp8(c0[1]), clamp8(c0[2])],
        [clamp8(c1[0] + d), clamp8(c1[1] + d), clamp8(c1[2] + d)],
        [clamp8(c1[0]), clamp8(c1[1]), clamp8(c1[2])],
        [clamp8(c1[0] - d), clamp8(c1[1] - d), clamp8(c1[2] - d)],
    ];

    let mut out = [[0u8; 4]; 16];
    for p in 0..16u32 {
        let x = p >> 2;
        let y = p & 3;
        let c = paint[pixel_index(bits, x, y)];
        out[(y * 4 + x) as usize] = [c[0], c[1], c[2], 255];
    }
    out
}

/// Decode an ETC2 `H`-mode block: two `RGB444` paint anchors, each split by a
/// distance into `+d`/`-d`. The distance's low bit is derived from the ordering
/// of the two 12-bit packed colours. Reached when the green channel overflows.
fn decode_h(bits: u64) -> [[u8; 4]; 16] {
    let r1 = field(bits, 62, 59);
    let g1 = (field(bits, 58, 56) << 1) | field(bits, 52, 52);
    let b1 = (field(bits, 51, 51) << 3) | field(bits, 49, 47);
    let r2 = field(bits, 46, 43);
    let g2 = field(bits, 42, 39);
    let b2 = field(bits, 38, 35);

    let c0_444 = (r1 << 8) | (g1 << 4) | b1;
    let c1_444 = (r2 << 8) | (g2 << 4) | b2;
    // Distance index: bit34 and bit32 give the upper two bits (bit33 is the
    // mode-selection diff flag and is skipped); the low bit is derived from the
    // ordering of the two packed 12-bit colours. Confirmed bit-exactly against
    // an M2 Metal hardware decode via a single-bit distance probe.
    let mut dist = ((field(bits, 34, 34) << 2) | (field(bits, 32, 32) << 1)) as usize;
    if c0_444 >= c1_444 {
        dist |= 1;
    }
    let d = ETC2_DISTANCE[dist & 7];

    let c0 = [ext4(r1), ext4(g1), ext4(b1)];
    let c1 = [ext4(r2), ext4(g2), ext4(b2)];
    let paint = [
        [clamp8(c0[0] + d), clamp8(c0[1] + d), clamp8(c0[2] + d)],
        [clamp8(c0[0] - d), clamp8(c0[1] - d), clamp8(c0[2] - d)],
        [clamp8(c1[0] + d), clamp8(c1[1] + d), clamp8(c1[2] + d)],
        [clamp8(c1[0] - d), clamp8(c1[1] - d), clamp8(c1[2] - d)],
    ];

    let mut out = [[0u8; 4]; 16];
    for p in 0..16u32 {
        let x = p >> 2;
        let y = p & 3;
        let c = paint[pixel_index(bits, x, y)];
        out[(y * 4 + x) as usize] = [c[0], c[1], c[2], 255];
    }
    out
}

/// Decode an ETC2 planar block: three `RGB676` corner colours (origin `O`,
/// horizontal `H`, vertical `V`) bilinearly interpolated across the 4x4 tile.
/// Reached when the blue channel overflows; carries no per-texel indices.
fn decode_planar(bits: u64) -> [[u8; 4]; 16] {
    // RGB676 corner colours. Bit layout per Khronos ETC2 planar (raw block).
    // Corner colours in RGB676. Bit positions confirmed bit-exactly against an
    // M2 Metal hardware decode (single-bit GPU probe); several fields are
    // scattered because the planar block reuses the differential-mode overflow
    // bit slots. `R`/`B` expand 6->8 by replication (`ext6`), `G` 7->8 (`ext7`).
    let ro = field(bits, 62, 57);
    let go = (field(bits, 56, 56) << 6) | field(bits, 54, 49);
    let bo = (field(bits, 48, 48) << 5) | (field(bits, 44, 43) << 3) | field(bits, 41, 39);
    let rh = (field(bits, 38, 34) << 1) | field(bits, 32, 32);
    let gh = field(bits, 31, 25);
    let bh = field(bits, 24, 19);
    let rv = field(bits, 18, 13);
    let gv = field(bits, 12, 6);
    let bv = field(bits, 5, 0);

    let o = [ext6(ro), ext7(go), ext6(bo)];
    let h = [ext6(rh), ext7(gh), ext6(bh)];
    let v = [ext6(rv), ext7(gv), ext6(bv)];

    let mut out = [[0u8; 4]; 16];
    for yy in 0..4i32 {
        for xx in 0..4i32 {
            let mut px = [0u8; 4];
            for c in 0..3 {
                let val = (xx * (h[c] - o[c]) + yy * (v[c] - o[c]) + 4 * o[c] + 2) >> 2;
                px[c] = clamp8(val);
            }
            px[3] = 255;
            out[(yy * 4 + xx) as usize] = px;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flat_differential_block_known_answer() {
        // diff=1, flip=0, base R=G=B=16 (-> ext5 = 132), delta 0, codewords 0,
        // all pixel indices 0 (modifier +2). Every texel must be 132+2 = 134.
        let block = [0x80, 0x80, 0x80, 0x02, 0x00, 0x00, 0x00, 0x00];
        assert_eq!(etc2_rgb8_mode(&block), Etc2Mode::Differential);
        let out = decode_etc2_rgb8(&block).unwrap();
        for texel in &out {
            assert_eq!(*texel, [134, 134, 134, 255]);
        }
    }

    #[test]
    fn differential_negative_modifier_known_answer() {
        // Same base, but all pixel indices 3 -> modifier -8 -> 132-8 = 124.
        let block = [0x80, 0x80, 0x80, 0x02, 0xFF, 0xFF, 0xFF, 0xFF];
        let out = decode_etc2_rgb8(&block).unwrap();
        for texel in &out {
            assert_eq!(*texel, [124, 124, 124, 255]);
        }
    }

    #[test]
    fn individual_block_splits_into_two_colours() {
        // diff=0, flip=0: left subblock colour1 = R444=15 (->255) red,
        // right subblock colour2 = B444=15 (->255) blue, codewords 0, idx 0
        // (+2). Left texels ~= (255,2,2), right ~= (2,2,255).
        let block = [0xF0, 0x00, 0x0F, 0x00, 0x00, 0x00, 0x00, 0x00];
        assert_eq!(etc2_rgb8_mode(&block), Etc2Mode::Individual);
        let out = decode_etc2_rgb8(&block).unwrap();
        for y in 0..4 {
            for x in 0..4 {
                let t = y * 4 + x;
                if x < 2 {
                    assert_eq!(out[t], [255, 2, 2, 255], "left@{x},{y}");
                } else {
                    assert_eq!(out[t], [2, 2, 255, 255], "right@{x},{y}");
                }
            }
        }
    }

    #[test]
    fn flip_bit_switches_split_orientation() {
        // Individual, flip=1 (horizontal split): top rows use colour1,
        // bottom rows colour2. Reuse the red/blue colours from above.
        let block = [0xF0, 0x00, 0x0F, 0x01, 0x00, 0x00, 0x00, 0x00];
        let out = decode_etc2_rgb8(&block).unwrap();
        for y in 0..4 {
            for x in 0..4 {
                let t = y * 4 + x;
                if y < 2 {
                    assert_eq!(out[t], [255, 2, 2, 255], "top@{x},{y}");
                } else {
                    assert_eq!(out[t], [2, 2, 255, 255], "bottom@{x},{y}");
                }
            }
        }
    }

    #[test]
    fn red_overflow_selects_and_decodes_t_mode() {
        // diff=1, R=31, dR=+1 -> 32 > 31 overflows red -> T mode.
        // byte0 = 11111 001 = 0xF9; diff bit set in byte3 (0x02). The extended
        // modes now decode rather than returning an error; exact-texel parity
        // against the GPU hardware decoder lives in the `prism_render_material_gpu`
        // ETC2 parity test.
        // T colours are RGB444: C0 = R1=0b1101=13 -> ext4 = 221 red, C1 = 0.
        // dist index 0 -> d = 3; all texel indices 0 select paint[0] = C0.
        let block = [0xF9, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x00];
        assert_eq!(etc2_rgb8_mode(&block), Etc2Mode::T);
        let out = decode_etc2_rgb8(&block).unwrap();
        for texel in &out {
            assert_eq!(*texel, [221, 0, 0, 255]);
        }
    }

    #[test]
    fn green_overflow_selects_h_mode() {
        // R in range (0,d0), G=31 dG=+1 overflow, B in range -> H mode.
        // byte0: R=00000 dR=000 -> 0x00; byte1: G=11111 dG=001 -> 0xF9.
        // C0 = RGB444(0,1,10) -> (0,17,170); C1 = 0. c0 >= c1 so the distance
        // LSB is set -> index 1 -> d = 6. All texel indices 0 select paint[0] =
        // C0 + d = (6, 23, 176).
        let block = [0x00, 0xF9, 0x00, 0x02, 0x00, 0x00, 0x00, 0x00];
        assert_eq!(etc2_rgb8_mode(&block), Etc2Mode::H);
        let out = decode_etc2_rgb8(&block).unwrap();
        for texel in &out {
            assert_eq!(*texel, [6, 23, 176, 255]);
        }
    }

    #[test]
    fn blue_overflow_selects_planar_mode() {
        // R,G in range, B=31 dB=+1 overflow -> planar.
        // byte2 holds B(47..43) and dB(42..40): 11111 001 = 0xF9.
        // Only the origin blue is non-zero (O = (0,0,105); H = V = 0), so blue
        // ramps down bilinearly from the top-left corner while red/green stay 0.
        let block = [0x00, 0x00, 0xF9, 0x02, 0x00, 0x00, 0x00, 0x00];
        assert_eq!(etc2_rgb8_mode(&block), Etc2Mode::Planar);
        let out = decode_etc2_rgb8(&block).unwrap();
        assert_eq!(out[0], [0, 0, 105, 255]); // (x=0, y=0)
        assert_eq!(out[3], [0, 0, 26, 255]); // (x=3, y=0)
        assert_eq!(out[12], [0, 0, 26, 255]); // (x=0, y=3)
        assert_eq!(out[15], [0, 0, 0, 255]); // (x=3, y=3) clamps to 0
    }

    #[test]
    fn per_subblock_codeword_and_large_modifier() {
        // Differential, flip=0, base R=G=B=16 -> ext5 = 132 on both subblocks
        // (delta 0). byte3 = 0x1E packs cw1=0 (bits 39..37) and cw2=7
        // (bits 36..34, table row {47,183,-47,-183}), with diff=1 (bit33) and
        // flip=0 (bit32). Every pixel LSB=1 (bytes 6,7) and MSB=0 (bytes 4,5)
        // selects index 1.
        //
        // Left subblock (x<2) uses cw1=0: ETC1_MODIFIER[0][1] = +8 -> 132+8=140.
        // Right subblock (x>=2) uses cw2=7: +183 -> 132+183=315 clamps to 255.
        let block = [0x80, 0x80, 0x80, 0x1E, 0x00, 0x00, 0xFF, 0xFF];
        assert_eq!(etc2_rgb8_mode(&block), Etc2Mode::Differential);
        let out = decode_etc2_rgb8(&block).unwrap();
        for y in 0..4 {
            for x in 0..4 {
                let t = y * 4 + x;
                if x < 2 {
                    assert_eq!(out[t], [140, 140, 140, 255], "left@{x},{y}");
                } else {
                    assert_eq!(out[t], [255, 255, 255, 255], "right@{x},{y}");
                }
            }
        }
    }

    #[test]
    fn decode_is_deterministic() {
        let block = [0x12, 0x34, 0x56, 0x7A, 0x9A, 0xBC, 0xDE, 0xF0];
        if let Ok(a) = decode_etc2_rgb8(&block) {
            assert_eq!(a, decode_etc2_rgb8(&block).unwrap());
        }
    }
}
