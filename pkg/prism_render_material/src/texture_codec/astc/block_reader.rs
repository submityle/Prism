//! Little-endian 128-bit block bit extraction shared by every ASTC decoder.
//!
//! An ASTC block is 16 bytes. Bit `0` is the least-significant bit of byte `0`;
//! bit `127` is the most-significant bit of byte `15`. Fields are packed
//! LSB-first, so [`read_bits`] returns the little-endian integer formed by
//! `count` consecutive bits starting at `lo`.

/// Extract `count` bits (`count <= 32`) starting at bit index `lo` from the
/// 16-byte little-endian `block`, returned as a right-justified `u32`.
///
/// # Panics (debug only)
/// Panics in debug builds if `count > 32` or the field runs past bit 128.
#[must_use]
pub fn read_bits(block: &[u8; 16], lo: u32, count: u32) -> u32 {
    debug_assert!(count <= 32, "ASTC field wider than u32");
    debug_assert!(lo + count <= 128, "ASTC field past end of block");
    let mut v = 0u32;
    for i in 0..count {
        let bit = lo + i;
        let byte = (bit >> 3) as usize;
        let b = (block[byte] >> (bit & 7)) & 1;
        v |= (b as u32) << i;
    }
    v
}

#[cfg(test)]
mod tests {
    use super::read_bits;

    #[test]
    fn single_bits_are_lsb_first() {
        let mut b = [0u8; 16];
        b[0] = 0b0000_0101; // bits 0 and 2 set
        assert_eq!(read_bits(&b, 0, 1), 1);
        assert_eq!(read_bits(&b, 1, 1), 0);
        assert_eq!(read_bits(&b, 2, 1), 1);
        assert_eq!(read_bits(&b, 0, 3), 0b101);
    }

    #[test]
    fn fields_cross_byte_boundaries() {
        let mut b = [0u8; 16];
        b[7] = 0xFF;
        b[8] = 0x01;
        // bits 56..64 are byte 7 (all ones), bit 64 is low bit of byte 8.
        assert_eq!(read_bits(&b, 56, 9), 0b1_1111_1111);
    }

    #[test]
    fn high_color_fields_round_trip() {
        let mut b = [0u8; 16];
        // Place 0xBEEF at bits 64..80 (byte 8 = 0xEF, byte 9 = 0xBE).
        b[8] = 0xEF;
        b[9] = 0xBE;
        assert_eq!(read_bits(&b, 64, 16), 0xBEEF);
    }
}
