//! BC7 mode 6 decode: the single-subset, full-`RGBA`, highest-quality opaque
//! and smooth-alpha mode of the BPTC (`BC7`) format.
//!
//! BC7 packs eight modes into a 16-byte block; the active mode is a unary code
//! in the low bits of byte 0 (mode `m` = `m` zero bits then a `1`). The other
//! seven modes add 2/3-subset partitioning (a 64-entry partition table), index
//! rotation, and p-bit sharing variants. **Mode 6** is the one mode with *no*
//! partition table: a single subset, two `RGBA` endpoints at 7 bits/channel
//! each plus a unique p-bit per endpoint (giving 8-bit endpoints), and 4-bit
//! indices for all sixteen texels. That makes it fully reproducible from the
//! block bits alone -- no transcribed partition/anchor tables -- so a CPU
//! golden matches a GPU twin exactly (integer interpolation, no `AI/ML`).
//!
//! Mode 6 is also the mode AAA compressors fall back to for high-fidelity
//! `RGBA` where a single subset already captures the block, so decoding it
//! covers a large fraction of shipped BC7 content. The remaining partitioned
//! modes (0-5, 7) are tracked as a follow-up: they require the Khronos
//! partition and anchor-index tables, which must be validated against a
//! reference decoder before they can be claimed, and are therefore kept out of
//! this module rather than stubbed.
//!
//! # Conventions
//! * The block is little-endian: bit `i` is `block[i / 8] >> (i % 8) & 1`.
//!   Fields are read LSB-first in Khronos field order.
//! * Output is row-major `RGBA8`, texel `t = y * 4 + x`, `t in [0, 16)`.
//! * Endpoint channels expand 7-bit + p-bit to 8-bit as `(v << 1) | p`; this is
//!   exact, so endpoint texels (`index` weight 0 or 64) decode bit-exactly and
//!   interior texels are deterministic (no `+/-1 LSB` ambiguity, unlike the
//!   truncating S3TC palette).
//!
//! # References
//! * Khronos Data Format Specification 1.3, BPTC (`BC7`) block decode.
//! * Microsoft `DXGI_FORMAT_BC7_*` / Vulkan `VK_FORMAT_BC7_*` BPTC spec.

/// 4-bit index interpolation weights (Khronos `aWeight4`), in 1/64 units.
const WEIGHT4: [u32; 16] = [
    0, 4, 9, 13, 17, 21, 26, 30, 34, 38, 43, 47, 51, 56, 60, 64,
];

/// 2-bit index interpolation weights (Khronos `aWeight2`), in 1/64 units.
const WEIGHT2: [u32; 4] = [0, 21, 43, 64];

/// 3-bit index interpolation weights (Khronos `aWeight3`), in 1/64 units.
const WEIGHT3: [u32; 8] = [0, 9, 18, 27, 37, 46, 55, 64];

/// LSB-first bit cursor over a 16-byte BC7 block.
struct BitReader<'a> {
    bytes: &'a [u8; 16],
    pos: usize,
}

impl<'a> BitReader<'a> {
    #[inline]
    fn new(bytes: &'a [u8; 16]) -> Self {
        Self { bytes, pos: 0 }
    }

    /// Read `n` bits (`n <= 32`) LSB-first and advance the cursor.
    ///
    /// Reads past the 128-bit block are impossible for mode 6 (its fields sum
    /// to exactly 128 bits); the index guard below keeps the function total
    /// even for a malformed caller by treating out-of-range bits as `0`.
    #[inline]
    fn read(&mut self, n: u32) -> u32 {
        let mut v = 0u32;
        for i in 0..n {
            let bit = self
                .bytes
                .get(self.pos / 8)
                .map_or(0, |byte| (byte >> (self.pos % 8)) & 1);
            v |= u32::from(bit) << i;
            self.pos += 1;
        }
        v
    }
}

/// Return the BC7 mode (`0..=7`) encoded in the block's unary prefix, or
/// `None` when byte 0 is zero (a reserved/invalid encoding).
#[inline]
#[must_use]
pub fn bc7_mode(block: &[u8; 16]) -> Option<u8> {
    if block[0] == 0 {
        None
    } else {
        // A nonzero `u8` has `trailing_zeros` in `0..=7`, exactly the mode.
        Some(block[0].trailing_zeros() as u8)
    }
}

/// Expand a 7-bit endpoint channel plus its p-bit to 8 bits (`(v << 1) | p`).
#[inline]
fn expand7(v: u32, p: u32) -> u8 {
    (((v << 1) | (p & 1)) & 0xFF) as u8
}

/// Expand a `bits`-wide endpoint channel (no p-bit) to 8 bits by MSB
/// replication, matching the Khronos BPTC `unquantize` for precisions below
/// 8. For a 7-bit value this is `(v << 1) | (v >> 6)`.
#[inline]
fn expand_rep(v: u32, bits: u32) -> u8 {
    debug_assert!((1..=8).contains(&bits));
    let shifted = v << (8 - bits);
    let replicate = v >> (2 * bits).saturating_sub(8).min(bits);
    ((shifted | replicate) & 0xFF) as u8
}

/// Interpolate one 8-bit channel between endpoints by a 4-bit index weight.
#[inline]
fn interp(e0: u8, e1: u8, weight: u32) -> u8 {
    let r = ((64 - weight) * u32::from(e0) + weight * u32::from(e1) + 32) >> 6;
    (r & 0xFF) as u8
}

/// Decode one 16-byte **BC7 mode 6** block into sixteen `RGBA8` texels.
///
/// The caller must have confirmed the block is mode 6 (see [`bc7_mode`]); the
/// seven mode bits are consumed and ignored here. Field order follows the
/// Khronos spec: `R0 R1 G0 G1 B0 B1 A0 A1` (7 bits each), `P0 P1` (1 bit each),
/// then the index block -- the anchor index (texel 0) is 3 bits with an
/// implicit high `0`, the remaining fifteen are 4 bits each.
#[must_use]
pub fn decode_bc7_mode6(block: &[u8; 16]) -> [[u8; 4]; 16] {
    let mut r = BitReader::new(block);
    let _mode = r.read(7); // unary mode-6 marker: six 0s then a 1.

    let r0 = r.read(7);
    let r1 = r.read(7);
    let g0 = r.read(7);
    let g1 = r.read(7);
    let b0 = r.read(7);
    let b1 = r.read(7);
    let a0 = r.read(7);
    let a1 = r.read(7);
    let p0 = r.read(1);
    let p1 = r.read(1);

    let e0 = [
        expand7(r0, p0),
        expand7(g0, p0),
        expand7(b0, p0),
        expand7(a0, p0),
    ];
    let e1 = [
        expand7(r1, p1),
        expand7(g1, p1),
        expand7(b1, p1),
        expand7(a1, p1),
    ];

    let mut indices = [0u8; 16];
    indices[0] = r.read(3) as u8; // anchor: implicit high bit 0.
    for idx in indices.iter_mut().skip(1) {
        *idx = r.read(4) as u8;
    }

    let mut out = [[0u8; 4]; 16];
    for (t, texel) in out.iter_mut().enumerate() {
        let w = WEIGHT4[indices[t] as usize];
        for c in 0..4 {
            texel[c] = interp(e0[c], e1[c], w);
        }
    }
    out
}

/// Decode one 16-byte **BC7 mode 5** block into sixteen `RGBA8` texels.
///
/// Mode 5 is the other single-subset mode without a partition table: `RGB`
/// endpoints at 7 bits/channel (no p-bit, MSB-replicated to 8), a *separate*
/// 8-bit alpha endpoint pair, independent 2-bit colour and 2-bit alpha index
/// blocks (each with a 1-bit anchor at texel 0), and a 2-bit **rotation** that
/// selects which output channel swaps with alpha. It suits content where
/// colour and alpha want different interpolation directions (e.g. masks over a
/// gradient), which a shared index (mode 6) cannot represent.
///
/// The caller must have confirmed the block is mode 5 (see [`bc7_mode`]).
/// Field order (Khronos): rotation(2); `R0 R1 G0 G1 B0 B1` (7 bits each);
/// `A0 A1` (8 bits each); colour indices (anchor 1 bit, rest 2 bits); alpha
/// indices (anchor 1 bit, rest 2 bits).
#[must_use]
pub fn decode_bc7_mode5(block: &[u8; 16]) -> [[u8; 4]; 16] {
    let mut r = BitReader::new(block);
    let _mode = r.read(6); // unary mode-5 marker: five 0s then a 1.
    let rotation = r.read(2);

    let r0 = r.read(7);
    let r1 = r.read(7);
    let g0 = r.read(7);
    let g1 = r.read(7);
    let b0 = r.read(7);
    let b1 = r.read(7);
    let a0 = r.read(8);
    let a1 = r.read(8);

    let c0 = [expand_rep(r0, 7), expand_rep(g0, 7), expand_rep(b0, 7)];
    let c1 = [expand_rep(r1, 7), expand_rep(g1, 7), expand_rep(b1, 7)];
    let ae0 = (a0 & 0xFF) as u8;
    let ae1 = (a1 & 0xFF) as u8;

    let mut color_idx = [0u8; 16];
    color_idx[0] = r.read(1) as u8; // colour anchor: implicit high bit 0.
    for idx in color_idx.iter_mut().skip(1) {
        *idx = r.read(2) as u8;
    }
    let mut alpha_idx = [0u8; 16];
    alpha_idx[0] = r.read(1) as u8; // alpha anchor: implicit high bit 0.
    for idx in alpha_idx.iter_mut().skip(1) {
        *idx = r.read(2) as u8;
    }

    let mut out = [[0u8; 4]; 16];
    for (t, texel) in out.iter_mut().enumerate() {
        let cw = WEIGHT2[color_idx[t] as usize];
        let aw = WEIGHT2[alpha_idx[t] as usize];
        let mut rgba = [
            interp(c0[0], c1[0], cw),
            interp(c0[1], c1[1], cw),
            interp(c0[2], c1[2], cw),
            interp(ae0, ae1, aw),
        ];
        // Rotation un-swaps the channel that carried the alpha-index value.
        match rotation {
            1 => rgba.swap(0, 3),
            2 => rgba.swap(1, 3),
            3 => rgba.swap(2, 3),
            _ => {}
        }
        *texel = rgba;
    }
    out
}

/// Decode one 16-byte **BC7 mode 4** block into sixteen `RGBA8` texels.
///
/// Mode 4 is the third partition-free single-subset mode. It carries `RGB`
/// endpoints at 5 bits/channel and a *separate* 6-bit alpha endpoint pair, plus
/// **two** index blocks of different precision -- a 2-bit set and a 3-bit set.
/// A 1-bit *index-selection* flag (`idxMode`) chooses which precision drives
/// colour and which drives alpha: `idxMode = 0` gives colour the 2-bit indices
/// and alpha the 3-bit indices; `idxMode = 1` swaps them. A 2-bit **rotation**
/// then selects which output channel swaps with alpha, exactly as in mode 5.
/// Giving alpha the higher-precision index set suits smooth alpha gradients
/// over blocky colour (or vice-versa), which a shared index cannot represent.
///
/// The caller must have confirmed the block is mode 4 (see [`bc7_mode`]).
/// Field order (Khronos): rotation(2); `idxMode`(1); `R0 R1 G0 G1 B0 B1`
/// (5 bits each); `A0 A1` (6 bits each); the 2-bit index block (anchor 1 bit,
/// rest 2 bits); the 3-bit index block (anchor 2 bits, rest 3 bits).
#[must_use]
pub fn decode_bc7_mode4(block: &[u8; 16]) -> [[u8; 4]; 16] {
    let mut r = BitReader::new(block);
    let _mode = r.read(5); // unary mode-4 marker: four 0s then a 1.
    let rotation = r.read(2);
    let idx_mode = r.read(1);

    let r0 = r.read(5);
    let r1 = r.read(5);
    let g0 = r.read(5);
    let g1 = r.read(5);
    let b0 = r.read(5);
    let b1 = r.read(5);
    let a0 = r.read(6);
    let a1 = r.read(6);

    let c0 = [expand_rep(r0, 5), expand_rep(g0, 5), expand_rep(b0, 5)];
    let c1 = [expand_rep(r1, 5), expand_rep(g1, 5), expand_rep(b1, 5)];
    let ae0 = expand_rep(a0, 6);
    let ae1 = expand_rep(a1, 6);

    // Index set 0 (2-bit) is stored first; its anchor (texel 0) is 1 bit.
    let mut idx2 = [0u8; 16];
    idx2[0] = r.read(1) as u8;
    for idx in idx2.iter_mut().skip(1) {
        *idx = r.read(2) as u8;
    }
    // Index set 1 (3-bit) follows; its anchor is 2 bits (implicit high 0).
    let mut idx3 = [0u8; 16];
    idx3[0] = r.read(2) as u8;
    for idx in idx3.iter_mut().skip(1) {
        *idx = r.read(3) as u8;
    }

    let mut out = [[0u8; 4]; 16];
    for (t, texel) in out.iter_mut().enumerate() {
        // idxMode routes which precision drives colour vs alpha.
        let (cw, aw) = if idx_mode == 0 {
            (WEIGHT2[idx2[t] as usize], WEIGHT3[idx3[t] as usize])
        } else {
            (WEIGHT3[idx3[t] as usize], WEIGHT2[idx2[t] as usize])
        };
        let mut rgba = [
            interp(c0[0], c1[0], cw),
            interp(c0[1], c1[1], cw),
            interp(c0[2], c1[2], cw),
            interp(ae0, ae1, aw),
        ];
        // Rotation un-swaps the channel that carried the alpha-index value.
        match rotation {
            1 => rgba.swap(0, 3),
            2 => rgba.swap(1, 3),
            3 => rgba.swap(2, 3),
            _ => {}
        }
        *texel = rgba;
    }
    out
}

/// Error returned by [`decode_bc7`] for a BC7 block whose mode is not yet
/// supported by this decoder.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Bc7Error {
    /// Byte 0 was zero: a reserved/invalid mode encoding.
    ReservedMode,
    /// A valid but unsupported mode (`0..=3`, `7`); only the partition-free
    /// single-subset modes 4, 5 and 6 are decoded today.
    UnsupportedMode(u8),
}

/// Decode a BC7 block, dispatching on its mode.
///
/// Only the partition-table-free single-subset modes are supported today:
/// mode 4 ([`decode_bc7_mode4`]), mode 5 ([`decode_bc7_mode5`]), and mode 6
/// ([`decode_bc7_mode6`]). The partitioned modes (0-3, 7) return
/// [`Bc7Error::UnsupportedMode`] rather than a wrong decode -- they require the
/// validated Khronos partition/anchor tables (tracked as a follow-up), and
/// silently mis-decoding them would be worse than an explicit error.
pub fn decode_bc7(block: &[u8; 16]) -> Result<[[u8; 4]; 16], Bc7Error> {
    match bc7_mode(block) {
        None => Err(Bc7Error::ReservedMode),
        Some(4) => Ok(decode_bc7_mode4(block)),
        Some(5) => Ok(decode_bc7_mode5(block)),
        Some(6) => Ok(decode_bc7_mode6(block)),
        Some(m) => Err(Bc7Error::UnsupportedMode(m)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Minimal LSB-first bit writer mirroring [`BitReader`] for test blocks.
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

    /// Assemble a mode-6 block from field values; `idx` holds the sixteen
    /// 4-bit indices (index 0 must be `<= 7` so its high bit is implicit 0).
    #[allow(clippy::too_many_arguments)]
    fn make_block(
        rgba0: [u32; 4],
        rgba1: [u32; 4],
        p0: u32,
        p1: u32,
        idx: [u8; 16],
    ) -> [u8; 16] {
        let mut w = BitWriter::new();
        w.write(0b100_0000, 7); // mode 6 marker
        w.write(rgba0[0], 7);
        w.write(rgba1[0], 7);
        w.write(rgba0[1], 7);
        w.write(rgba1[1], 7);
        w.write(rgba0[2], 7);
        w.write(rgba1[2], 7);
        w.write(rgba0[3], 7);
        w.write(rgba1[3], 7);
        w.write(p0, 1);
        w.write(p1, 1);
        w.write(u32::from(idx[0]), 3);
        for &i in idx.iter().skip(1) {
            w.write(u32::from(i), 4);
        }
        assert_eq!(w.pos, 128, "mode-6 fields must fill the block exactly");
        w.bytes
    }

    #[test]
    fn mode_detection_reads_unary_prefix() {
        let mut b = [0u8; 16];
        b[0] = 0b0100_0000; // bit 6 set, bits 0-5 clear => mode 6
        assert_eq!(bc7_mode(&b), Some(6));
        b[0] = 0b0000_0001; // bit 0 set => mode 0
        assert_eq!(bc7_mode(&b), Some(0));
        b[0] = 0b1000_0000; // bit 7 set => mode 7
        assert_eq!(bc7_mode(&b), Some(7));
        b[0] = 0; // reserved / invalid
        assert_eq!(bc7_mode(&b), None);
    }

    #[test]
    fn make_block_is_tagged_mode6() {
        let block = make_block([0; 4], [0x7F; 4], 0, 1, [0u8; 16]);
        assert_eq!(bc7_mode(&block), Some(6));
    }

    #[test]
    fn endpoint_indices_decode_exactly() {
        // e0 = (0x7F<<1)|0 = 0xFE on all channels; e1 = (0<<1)|1 = 1.
        let block = make_block(
            [0x7F; 4],
            [0; 4],
            0,
            1,
            [0, 15, 0, 15, 0, 15, 0, 15, 0, 15, 0, 15, 0, 15, 0, 15],
        );
        let out = decode_bc7_mode6(&block);
        // Weight 0 -> pure e0; weight 64 -> pure e1 (exact, no rounding slack).
        assert_eq!(out[0], [0xFE, 0xFE, 0xFE, 0xFE]);
        assert_eq!(out[1], [1, 1, 1, 1]);
        assert_eq!(out[2], [0xFE, 0xFE, 0xFE, 0xFE]);
        assert_eq!(out[3], [1, 1, 1, 1]);
    }

    #[test]
    fn constant_block_is_flat_regardless_of_index() {
        // e0 == e1 => every texel equals that colour for any index weight.
        let c = [0x55, 0x2A, 0x7F, 0x10];
        let mut idx = [0u8; 16];
        for (t, i) in idx.iter_mut().enumerate() {
            *i = (t as u8) & 0x0F;
        }
        idx[0] = 5; // anchor must stay <= 7
        let block = make_block(c, c, 1, 1, idx);
        let expected = [
            expand7(c[0], 1),
            expand7(c[1], 1),
            expand7(c[2], 1),
            expand7(c[3], 1),
        ];
        let out = decode_bc7_mode6(&block);
        for texel in &out {
            assert_eq!(*texel, expected);
        }
    }

    #[test]
    fn interior_indices_are_monotonic_between_endpoints() {
        // Red ramps 0 -> 254 as the index weight grows; check monotonicity and
        // bracketing on the red channel across all sixteen weight steps.
        let idx = [0u8, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15];
        let block = make_block([0, 0, 0, 0x7F], [0x7F, 0, 0, 0x7F], 0, 0, idx);
        let out = decode_bc7_mode6(&block);
        let e0r = out[0][0];
        let e1r = out[15][0];
        assert_eq!(e0r, 0);
        assert_eq!(e1r, 0xFE);
        for t in 1..16 {
            assert!(
                out[t][0] >= out[t - 1][0],
                "red must be non-decreasing in index"
            );
            assert!(out[t][0] >= e0r && out[t][0] <= e1r, "red stays bracketed");
        }
        // Alpha endpoints are equal (0xFE) so alpha is flat.
        for texel in &out {
            assert_eq!(texel[3], 0xFE);
        }
    }

    #[test]
    fn pbit_sets_endpoint_low_bit() {
        // Same 7-bit value, differing p-bit, must differ only in the LSB.
        let block = make_block([0x40, 0, 0, 0x7F], [0x40, 0, 0, 0x7F], 0, 1, [0u8; 16]);
        let out = decode_bc7_mode6(&block);
        // index 0 (anchor, weight 0) -> e0 with p=0 => 0x80.
        assert_eq!(out[0][0], 0x80);
        // A weight-64 texel would give e1 with p=1 => 0x81; verify via a block
        // whose anchor still weight-0 but texel 1 is weight 64.
        let mut idx = [0u8; 16];
        idx[1] = 15;
        let block2 = make_block([0x40, 0, 0, 0x7F], [0x40, 0, 0, 0x7F], 0, 1, idx);
        let out2 = decode_bc7_mode6(&block2);
        assert_eq!(out2[1][0], 0x81);
    }

    #[test]
    fn midpoint_index_is_near_average() {
        // Weight 30 (index 7) and 34 (index 8) bracket the true midpoint of a
        // 0..254 ramp (127); check the decode lands within the integer step.
        let block = make_block([0, 0, 0, 0x7F], [0x7F, 0, 0, 0x7F], 0, 0, {
            let mut idx = [0u8; 16];
            idx[1] = 7;
            idx[2] = 8;
            idx
        });
        let out = decode_bc7_mode6(&block);
        assert!(out[1][0] <= 127 && out[2][0] >= 127);
    }

    /// Assemble a mode-5 block: `rgb0/rgb1` are 7-bit, `a0/a1` 8-bit; `cidx`
    /// and `aidx` are the sixteen 2-bit colour/alpha indices (index 0 <= 1).
    #[allow(clippy::too_many_arguments)]
    fn make_block5(
        rgb0: [u32; 3],
        rgb1: [u32; 3],
        a0: u32,
        a1: u32,
        rotation: u32,
        cidx: [u8; 16],
        aidx: [u8; 16],
    ) -> [u8; 16] {
        let mut w = BitWriter::new();
        w.write(0b10_0000, 6); // mode 5 marker
        w.write(rotation, 2);
        w.write(rgb0[0], 7);
        w.write(rgb1[0], 7);
        w.write(rgb0[1], 7);
        w.write(rgb1[1], 7);
        w.write(rgb0[2], 7);
        w.write(rgb1[2], 7);
        w.write(a0, 8);
        w.write(a1, 8);
        w.write(u32::from(cidx[0]), 1);
        for &i in cidx.iter().skip(1) {
            w.write(u32::from(i), 2);
        }
        w.write(u32::from(aidx[0]), 1);
        for &i in aidx.iter().skip(1) {
            w.write(u32::from(i), 2);
        }
        assert_eq!(w.pos, 128, "mode-5 fields must fill the block exactly");
        w.bytes
    }

    #[test]
    fn mode5_block_is_tagged_mode5() {
        let block = make_block5([0; 3], [0x7F; 3], 0, 255, 0, [0u8; 16], [0u8; 16]);
        assert_eq!(bc7_mode(&block), Some(5));
    }

    #[test]
    fn mode5_color_endpoints_replicate_to_8bit() {
        // 7-bit 0x7F -> (0x7F<<1)|(0x7F>>6) = 0xFF; 0x00 -> 0x00.
        let block = make_block5(
            [0x7F; 3],
            [0; 3],
            200,
            50,
            0,
            [0, 3, 0, 3, 0, 3, 0, 3, 0, 3, 0, 3, 0, 3, 0, 3],
            [0u8; 16],
        );
        let out = decode_bc7_mode5(&block);
        // colour weight 0 -> c0 (0xFF); weight 64 (index 3) -> c1 (0x00).
        assert_eq!([out[0][0], out[0][1], out[0][2]], [0xFF, 0xFF, 0xFF]);
        assert_eq!([out[1][0], out[1][1], out[1][2]], [0, 0, 0]);
        // alpha endpoints are exact 8-bit; index 0 everywhere -> a0 = 200.
        assert_eq!(out[0][3], 200);
    }

    #[test]
    fn mode5_alpha_index_is_independent_of_color() {
        // Flat colour, alpha ramps via its own index block.
        let mut aidx = [0u8; 16];
        aidx[1] = 3; // weight 64 -> a1
        let block = make_block5([0x40; 3], [0x40; 3], 0, 255, 0, [0u8; 16], aidx);
        let out = decode_bc7_mode5(&block);
        assert_eq!(out[0][3], 0); // a0
        assert_eq!(out[1][3], 255); // a1
        // Colour identical on both texels (flat endpoints).
        assert_eq!([out[0][0], out[0][1], out[0][2]], [out[1][0], out[1][1], out[1][2]]);
    }

    #[test]
    fn mode5_rotation_swaps_alpha_channel() {
        // rotation 1 swaps R<->A. Colour R=0xFF via c0, alpha=0 via a0.
        let block = make_block5([0x7F, 0, 0], [0x7F, 0, 0], 0, 0, 1, [0u8; 16], [0u8; 16]);
        let out = decode_bc7_mode5(&block);
        // After swap(0,3): channel 0 carries the (0) alpha, channel 3 carries R=0xFF.
        assert_eq!(out[0][0], 0);
        assert_eq!(out[0][3], 0xFF);
    }

    #[test]
    fn dispatch_rejects_unsupported_and_reserved_modes() {
        let mode6 = make_block([0; 4], [0x7F; 4], 0, 1, [0u8; 16]);
        assert!(decode_bc7(&mode6).is_ok());
        let mode5 = make_block5([0; 3], [0x7F; 3], 0, 255, 0, [0u8; 16], [0u8; 16]);
        assert!(decode_bc7(&mode5).is_ok());

        let mut m0 = [0u8; 16];
        m0[0] = 0b0000_0001; // mode 0
        assert_eq!(decode_bc7(&m0), Err(Bc7Error::UnsupportedMode(0)));
        let mut m7 = [0u8; 16];
        m7[0] = 0b1000_0000; // mode 7
        assert_eq!(decode_bc7(&m7), Err(Bc7Error::UnsupportedMode(7)));
        let reserved = [0u8; 16];
        assert_eq!(decode_bc7(&reserved), Err(Bc7Error::ReservedMode));
    }

    #[test]
    fn expand_rep_matches_known_replication() {
        assert_eq!(expand_rep(0x7F, 7), 0xFF);
        assert_eq!(expand_rep(0, 7), 0);
        // 0x40 = 0b100_0000; (<<1)=0x80, (>>6)=1, OR = 0x81.
        assert_eq!(expand_rep(0x40, 7), 0x81);
        assert_eq!(expand_rep(0xFF, 8), 0xFF); // 8-bit identity
        assert_eq!(expand_rep(0x5A, 8), 0x5A);
    }

    /// Assemble a mode-4 block: `rgb0/rgb1` are 5-bit, `a0/a1` 6-bit; `idx2`
    /// are the sixteen 2-bit indices (index 0 <= 1), `idx3` the sixteen 3-bit
    /// indices (index 0 <= 3); `idx_mode` selects colour/alpha index routing.
    #[allow(clippy::too_many_arguments)]
    fn make_block4(
        rgb0: [u32; 3],
        rgb1: [u32; 3],
        a0: u32,
        a1: u32,
        rotation: u32,
        idx_mode: u32,
        idx2: [u8; 16],
        idx3: [u8; 16],
    ) -> [u8; 16] {
        let mut w = BitWriter::new();
        w.write(0b1_0000, 5); // mode 4 marker
        w.write(rotation, 2);
        w.write(idx_mode, 1);
        w.write(rgb0[0], 5);
        w.write(rgb1[0], 5);
        w.write(rgb0[1], 5);
        w.write(rgb1[1], 5);
        w.write(rgb0[2], 5);
        w.write(rgb1[2], 5);
        w.write(a0, 6);
        w.write(a1, 6);
        w.write(u32::from(idx2[0]), 1);
        for &i in idx2.iter().skip(1) {
            w.write(u32::from(i), 2);
        }
        w.write(u32::from(idx3[0]), 2);
        for &i in idx3.iter().skip(1) {
            w.write(u32::from(i), 3);
        }
        assert_eq!(w.pos, 128, "mode-4 fields must fill the block exactly");
        w.bytes
    }

    #[test]
    fn mode4_block_is_tagged_mode4() {
        let block = make_block4([0; 3], [0x1F; 3], 0, 0x3F, 0, 0, [0u8; 16], [0u8; 16]);
        assert_eq!(bc7_mode(&block), Some(4));
    }

    #[test]
    fn mode4_color_endpoints_replicate_to_8bit() {
        // 5-bit 0x1F -> (0x1F<<3)|(0x1F>>2) = 0xF8|0x07 = 0xFF; 0x00 -> 0x00.
        // idx_mode 0: colour uses the 2-bit index set.
        let mut idx2 = [0u8; 16];
        idx2[1] = 3; // weight 64 -> c1
        let block = make_block4([0x1F; 3], [0; 3], 0x3F, 0, 0, 0, idx2, [0u8; 16]);
        let out = decode_bc7_mode4(&block);
        assert_eq!([out[0][0], out[0][1], out[0][2]], [0xFF, 0xFF, 0xFF]); // c0
        assert_eq!([out[1][0], out[1][1], out[1][2]], [0, 0, 0]); // c1
    }

    #[test]
    fn mode4_alpha_6bit_replicates_and_uses_3bit_index() {
        // 6-bit 0x3F -> (0x3F<<2)|(0x3F>>4) = 0xFC|0x03 = 0xFF.
        // idx_mode 0: alpha uses the 3-bit index set.
        let mut idx3 = [0u8; 16];
        idx3[1] = 7; // weight 64 -> a1
        let block = make_block4([0x10; 3], [0x10; 3], 0, 0x3F, 0, 0, [0u8; 16], idx3);
        let out = decode_bc7_mode4(&block);
        assert_eq!(out[0][3], 0); // a0
        assert_eq!(out[1][3], 0xFF); // a1 replicated
    }

    #[test]
    fn mode4_idx_mode_routes_precision() {
        // idx_mode 1: colour driven by the 3-bit set, alpha by the 2-bit set.
        // colour c0=0, c1=0xFF; alpha a0=0, a1=0xFF.
        let mut idx2 = [0u8; 16];
        let mut idx3 = [0u8; 16];
        idx3[1] = 7; // colour at texel1 -> c1 (0xFF); its 2-bit alpha stays a0.
        idx2[2] = 3; // alpha at texel2 -> a1 (0xFF); its 3-bit colour stays c0.
        let block = make_block4([0; 3], [0x1F; 3], 0, 0x3F, 0, 1, idx2, idx3);
        let out = decode_bc7_mode4(&block);
        assert_eq!([out[1][0], out[1][1], out[1][2]], [0xFF, 0xFF, 0xFF]);
        assert_eq!(out[1][3], 0);
        assert_eq!([out[2][0], out[2][1], out[2][2]], [0, 0, 0]);
        assert_eq!(out[2][3], 0xFF);
    }

    #[test]
    fn mode4_rotation_swaps_alpha_channel() {
        // rotation 1 swaps R<->A. idx_mode 0: colour via 2-bit, alpha via 3-bit.
        // R=0xFF via c0 (idx2 all 0), alpha=0 via a0 (idx3 all 0).
        let block = make_block4([0x1F, 0, 0], [0x1F, 0, 0], 0, 0, 1, 0, [0u8; 16], [0u8; 16]);
        let out = decode_bc7_mode4(&block);
        assert_eq!(out[0][0], 0); // channel 0 now carries the (0) alpha
        assert_eq!(out[0][3], 0xFF); // channel 3 now carries R=0xFF
    }

    #[test]
    fn dispatch_decodes_mode4() {
        let block = make_block4([0; 3], [0x1F; 3], 0, 0x3F, 0, 0, [0u8; 16], [0u8; 16]);
        assert!(decode_bc7(&block).is_ok());
    }
}
