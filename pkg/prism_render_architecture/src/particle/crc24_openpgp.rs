//! `CRC-24/OPENPGP` checksum as defined by `RFC 4880`.
//!
//! Parameters: width = 24, poly = `0x864CFB`, init = `0xB704CE`,
//! `refin` = false, `refout` = false, `xorout` = 0. All arithmetic is pure
//! integer math on `u32` values that carry the low 24 bits of the running
//! checksum; no floating point and no `unsafe` are used.
//!
//! This module is self-contained and `no_std` friendly: it relies only on
//! `core` operations and does not touch any other module in the crate.
//!
//! Two algorithms are provided: a straightforward bitwise implementation
//! (the public API) and a 256-entry table-driven implementation (test-only)
//! that cross-validates the bitwise results.

/// `CRC-24/OPENPGP` generator polynomial (low 24 bits, bit 24 implicit).
const POLY: u32 = 0x0086_4CFB;
/// Initial register value for `CRC-24/OPENPGP`.
const INIT: u32 = 0x00B7_04CE;
/// Mask selecting the low 24 bits of a `u32`.
const MASK: u32 = 0x00FF_FFFF;
/// Bit 24, used to detect the high bit shifted out of the 24-bit register.
const TOP_BIT: u32 = 0x0100_0000;

/// Computes the `CRC-24/OPENPGP` checksum of `data` in one shot.
///
/// Returns the checksum in the low 24 bits of the result; the upper 8 bits
/// are always zero.
pub fn crc24(data: &[u8]) -> u32 {
    let mut crc = INIT;
    for &b in data {
        crc ^= u32::from(b) << 16;
        for _ in 0..8 {
            crc <<= 1;
            if crc & TOP_BIT != 0 {
                crc ^= POLY;
            }
        }
    }
    crc & MASK
}

/// Splits a 24-bit `CRC` value into its three big-endian bytes.
///
/// The input is masked to 24 bits first, so callers may pass a value that
/// still has stray upper bits set.
pub fn crc24_be_bytes(crc: u32) -> [u8; 3] {
    let crc = crc & MASK;
    [
        ((crc >> 16) & 0xFF) as u8,
        ((crc >> 8) & 0xFF) as u8,
        (crc & 0xFF) as u8,
    ]
}

/// Incremental `CRC-24/OPENPGP` calculator.
///
/// Feed data with [`Crc24::update`] across any number of chunks, then read
/// the result with [`Crc24::finalize`]. These are plain methods (not trait
/// operator implementations), so the arithmetic stays explicit.
pub struct Crc24 {
    state: u32,
}

impl Default for Crc24 {
    fn default() -> Self {
        Self::new()
    }
}

impl Crc24 {
    /// Creates a fresh calculator primed with the `CRC-24/OPENPGP` init value.
    pub fn new() -> Self {
        Self { state: INIT }
    }

    /// Feeds another chunk of bytes into the running checksum.
    pub fn update(&mut self, data: &[u8]) {
        let mut crc = self.state;
        for &b in data {
            crc ^= u32::from(b) << 16;
            for _ in 0..8 {
                crc <<= 1;
                if crc & TOP_BIT != 0 {
                    crc ^= POLY;
                }
            }
            crc &= MASK;
        }
        self.state = crc & MASK;
    }

    /// Returns the current checksum value in the low 24 bits.
    pub fn finalize(&self) -> u32 {
        self.state & MASK
    }
}

/// Builds the 256-entry lookup table for the table-driven implementation.
#[cfg(test)]
const fn build_table() -> [u32; 256] {
    let mut table = [0u32; 256];
    let mut i = 0usize;
    while i < 256 {
        let mut crc = (i as u32) << 16;
        let mut j = 0usize;
        while j < 8 {
            crc <<= 1;
            if crc & TOP_BIT != 0 {
                crc ^= POLY;
            }
            j += 1;
        }
        table[i] = crc & MASK;
        i += 1;
    }
    table
}

/// Precomputed `CRC-24/OPENPGP` lookup table (test-only cross-check).
#[cfg(test)]
const TABLE: [u32; 256] = build_table();

/// Table-driven `CRC-24/OPENPGP` used only to validate the bitwise code.
#[cfg(test)]
fn crc24_table(data: &[u8]) -> u32 {
    let mut crc = INIT;
    for &b in data {
        let idx = (((crc >> 16) ^ u32::from(b)) & 0xFF) as usize;
        crc = ((crc << 8) ^ TABLE[idx]) & MASK;
    }
    crc & MASK
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn check_vector_123456789() {
        assert_eq!(crc24(b"123456789"), 0x0021_CF02);
    }

    #[test]
    fn empty_input_returns_init() {
        assert_eq!(crc24(b""), 0x00B7_04CE);
    }

    #[test]
    fn empty_input_equals_init_const() {
        assert_eq!(crc24(&[]), INIT);
    }

    #[test]
    fn be_bytes_known_vector() {
        assert_eq!(crc24_be_bytes(0x0021_CF02), [0x21, 0xCF, 0x02]);
    }

    #[test]
    fn be_bytes_zero() {
        assert_eq!(crc24_be_bytes(0x0000_0000), [0x00, 0x00, 0x00]);
    }

    #[test]
    fn be_bytes_max() {
        assert_eq!(crc24_be_bytes(0x00FF_FFFF), [0xFF, 0xFF, 0xFF]);
    }

    #[test]
    fn be_bytes_masks_high_bits() {
        // Stray upper bits must be ignored.
        assert_eq!(crc24_be_bytes(0xFF21_CF02), [0x21, 0xCF, 0x02]);
    }

    #[test]
    fn be_bytes_middle_value() {
        assert_eq!(crc24_be_bytes(0x00B7_04CE), [0xB7, 0x04, 0xCE]);
    }

    #[test]
    fn be_bytes_roundtrip() {
        let crc = crc24(b"123456789");
        let bytes = crc24_be_bytes(crc);
        let rebuilt =
            (u32::from(bytes[0]) << 16) | (u32::from(bytes[1]) << 8) | u32::from(bytes[2]);
        assert_eq!(rebuilt, crc);
    }

    #[test]
    fn table_len_is_256() {
        assert_eq!(TABLE.len(), 256);
    }

    #[test]
    fn table_entry_zero_is_zero() {
        assert_eq!(TABLE[0], 0);
    }

    #[test]
    fn table_matches_bitwise_check_vector() {
        assert_eq!(crc24_table(b"123456789"), 0x0021_CF02);
    }

    #[test]
    fn table_matches_bitwise_empty() {
        assert_eq!(crc24_table(b""), crc24(b""));
    }

    #[test]
    fn table_matches_bitwise_all_single_bytes() {
        for b in 0u8..=u8::MAX {
            let data = [b];
            assert_eq!(crc24(&data), crc24_table(&data));
        }
    }

    #[test]
    fn table_matches_bitwise_various() {
        let samples: [&[u8]; 6] = [
            b"",
            b"a",
            b"abc",
            b"The quick brown fox jumps over the lazy dog",
            b"\x00\x01\x02\x03\x04\x05\x06\x07",
            b"\xFF\xFE\xFD\xFC\xFB",
        ];
        for s in samples {
            assert_eq!(crc24(s), crc24_table(s));
        }
    }

    #[test]
    fn table_matches_bitwise_long_pattern() {
        let mut data = [0u8; 512];
        for (i, d) in data.iter_mut().enumerate() {
            *d = (i & 0xFF) as u8;
        }
        assert_eq!(crc24(&data), crc24_table(&data));
    }

    #[test]
    fn incremental_two_chunks() {
        let data = b"123456789";
        let (a, b) = data.split_at(4);
        let mut crc = Crc24::new();
        crc.update(a);
        crc.update(b);
        assert_eq!(crc.finalize(), crc24(data));
    }

    #[test]
    fn incremental_two_chunks_equals_check_vector() {
        let mut crc = Crc24::new();
        crc.update(b"1234");
        crc.update(b"56789");
        assert_eq!(crc.finalize(), 0x0021_CF02);
    }

    #[test]
    fn incremental_three_chunks() {
        let data = b"The quick brown fox";
        let mut crc = Crc24::new();
        crc.update(&data[0..5]);
        crc.update(&data[5..11]);
        crc.update(&data[11..]);
        assert_eq!(crc.finalize(), crc24(data));
    }

    #[test]
    fn incremental_byte_by_byte() {
        let data = b"incremental-check";
        let mut crc = Crc24::new();
        for &b in data {
            crc.update(&[b]);
        }
        assert_eq!(crc.finalize(), crc24(data));
    }

    #[test]
    fn incremental_matches_oneshot_at_every_split() {
        let data = b"The quick brown fox jumps";
        for split in 0..=data.len() {
            let (a, b) = data.split_at(split);
            let mut crc = Crc24::new();
            crc.update(a);
            crc.update(b);
            assert_eq!(crc.finalize(), crc24(data), "split at {split}");
        }
    }

    #[test]
    fn incremental_empty_then_data() {
        let mut crc = Crc24::new();
        crc.update(b"");
        crc.update(b"123456789");
        assert_eq!(crc.finalize(), 0x0021_CF02);
    }

    #[test]
    fn incremental_data_then_empty() {
        let mut crc = Crc24::new();
        crc.update(b"123456789");
        crc.update(b"");
        assert_eq!(crc.finalize(), 0x0021_CF02);
    }

    #[test]
    fn finalize_without_update_is_init() {
        let crc = Crc24::new();
        assert_eq!(crc.finalize(), INIT);
    }

    #[test]
    fn update_empty_slice_keeps_init() {
        let mut crc = Crc24::new();
        crc.update(&[]);
        assert_eq!(crc.finalize(), INIT);
    }

    #[test]
    fn default_equals_new() {
        let a = Crc24::default();
        let b = Crc24::new();
        assert_eq!(a.finalize(), b.finalize());
    }

    #[test]
    fn single_byte_zero() {
        let data = [0x00u8];
        assert_eq!(crc24(&data), crc24_table(&data));
    }

    #[test]
    fn single_byte_ff() {
        let data = [0xFFu8];
        assert_eq!(crc24(&data), crc24_table(&data));
    }

    #[test]
    fn single_byte_one() {
        let data = [0x01u8];
        assert_eq!(crc24(&data), crc24_table(&data));
    }

    #[test]
    fn single_byte_high_bit() {
        let data = [0x80u8];
        assert_eq!(crc24(&data), crc24_table(&data));
    }

    #[test]
    fn all_zeros_short() {
        let data = [0u8; 8];
        assert_eq!(crc24(&data), crc24_table(&data));
    }

    #[test]
    fn all_zeros_long() {
        let data = [0u8; 1024];
        let expected = crc24_table(&data);
        assert_eq!(crc24(&data), expected);
    }

    #[test]
    fn all_ones_short() {
        let data = [0xFFu8; 8];
        assert_eq!(crc24(&data), crc24_table(&data));
    }

    #[test]
    fn all_ones_long() {
        let data = [0xFFu8; 1024];
        let expected = crc24_table(&data);
        assert_eq!(crc24(&data), expected);
    }

    #[test]
    fn long_input_pattern_incremental_consistency() {
        let mut data = [0u8; 777];
        for (i, d) in data.iter_mut().enumerate() {
            *d = (i.wrapping_mul(31) & 0xFF) as u8;
        }
        let oneshot = crc24(&data);
        let mut crc = Crc24::new();
        for chunk in data.chunks(64) {
            crc.update(chunk);
        }
        assert_eq!(crc.finalize(), oneshot);
    }

    #[test]
    fn result_within_24_bits() {
        assert_eq!(crc24(b"hello world") >> 24, 0);
        assert_eq!(crc24(b"") >> 24, 0);
        assert_eq!(crc24(b"123456789") >> 24, 0);
    }

    #[test]
    fn different_inputs_differ() {
        assert_ne!(crc24(b"foo"), crc24(b"bar"));
    }

    #[test]
    fn length_sensitivity() {
        assert_ne!(crc24(b"\x00"), crc24(b"\x00\x00"));
    }

    #[test]
    fn single_zero_byte_is_not_init() {
        // Feeding a byte must change the register away from the init value.
        assert_ne!(crc24(b"\x00"), INIT);
    }

    #[test]
    fn abc_matches_across_methods_and_incremental() {
        let data = b"abc";
        let oneshot = crc24(data);
        let mut inc = Crc24::new();
        inc.update(data);
        assert_eq!(crc24_table(data), oneshot);
        assert_eq!(inc.finalize(), oneshot);
    }

    #[test]
    fn quick_brown_fox_cross_check() {
        let data = b"The quick brown fox jumps over the lazy dog";
        let oneshot = crc24(data);
        assert_eq!(crc24_table(data), oneshot);
        let mut inc = Crc24::new();
        inc.update(data);
        assert_eq!(inc.finalize(), oneshot);
        assert_eq!(oneshot >> 24, 0);
    }

    #[test]
    fn table_cross_check_long_input() {
        let data = [0xA5u8; 2048];
        assert_eq!(crc24(&data), crc24_table(&data));
    }
}
