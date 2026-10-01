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
}
