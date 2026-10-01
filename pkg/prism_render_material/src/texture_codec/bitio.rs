//! LSB-first bit cursor over a fixed 16-byte block, shared by the BPTC
//! decoders ([`super::bc7`] and [`super::bc6h`]).
//!
//! BPTC blocks (`BC6H`, `BC7`) are 128-bit little-endian records whose fields
//! are packed LSB-first: field bit `i` of a value lives at absolute block bit
//! `base + i`, and absolute block bit `b` is `bytes[b / 8] >> (b % 8) & 1`.
//! A single forward cursor decodes every contiguous-layout mode exactly.
//!
//! # Conventions
//! * [`BitReader::read`] consumes `n <= 32` bits LSB-first and advances.
//! * Reads past bit 128 are impossible for well-formed single-subset modes
//!   (their fields sum to exactly 128 bits); out-of-range bits read as `0` so
//!   the reader stays total even for a malformed caller.
//!
//! # References
//! * Khronos Data Format Specification 1.3, BPTC block layout.

/// LSB-first bit cursor over a 16-byte BPTC block.
pub(crate) struct BitReader<'a> {
    bytes: &'a [u8; 16],
    pos: usize,
}

impl<'a> BitReader<'a> {
    /// Create a cursor positioned at bit 0 of `bytes`.
    #[inline]
    pub(crate) fn new(bytes: &'a [u8; 16]) -> Self {
        Self { bytes, pos: 0 }
    }

    /// Read `n` bits (`n <= 32`) LSB-first and advance the cursor.
    ///
    /// Out-of-range bits (cursor past the 128-bit block) read as `0`, keeping
    /// the function total for malformed callers.
    #[inline]
    pub(crate) fn read(&mut self, n: u32) -> u32 {
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

    /// Current cursor position in bits from the start of the block.
    ///
    /// Only the self-tests consume this today; scattered-layout BPTC modes
    /// (e.g. partitioned `BC6H`) that must seek will use it once they land, so
    /// it is gated to avoid a dead-code warning rather than removed.
    #[cfg(test)]
    #[inline]
    pub(crate) fn pos(&self) -> usize {
        self.pos
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_lsb_first_across_byte_boundaries() {
        // byte0 = 0b1010_0101, byte1 = 0b0000_0011.
        let mut b = [0u8; 16];
        b[0] = 0b1010_0101;
        b[1] = 0b0000_0011;
        let mut r = BitReader::new(&b);
        assert_eq!(r.read(4), 0b0101); // low nibble of byte0, LSB-first
        assert_eq!(r.read(4), 0b1010); // high nibble of byte0
        assert_eq!(r.read(2), 0b11); // low two bits of byte1
        assert_eq!(r.pos(), 10);
    }

    #[test]
    fn reads_past_block_end_as_zero() {
        let b = [0xFFu8; 16];
        let mut r = BitReader::new(&b);
        assert_eq!(r.read(32), 0xFFFF_FFFF);
        assert_eq!(r.read(32), 0xFFFF_FFFF);
        assert_eq!(r.read(32), 0xFFFF_FFFF);
        assert_eq!(r.read(31), 0x7FFF_FFFF); // bits 96..127 all set
        assert_eq!(r.read(8), 1); // bit 127 set, bits 128..134 => 0
        assert_eq!(r.pos(), 135);
    }
}
