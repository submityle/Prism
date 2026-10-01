//! `CRC-3`/`GSM` checksum: width=3, poly=0x3, init=0x0, refin=false, refout=false, xorout=0x7.
//!
//! Pure-integer (`u8`) implementation using a bit-at-a-time, MSB-first
//! algorithm. Because the register width (3) is smaller than a byte, each
//! input byte is consumed one bit at a time from bit 7 down to bit 0
//! (`refin` = false). The result is reflected neither on input nor output
//! (`refout` = false) and is finally combined with `xorout` (0x7).
//!
//! Reference vectors enshrined in the test module:
//! - `checksum(b"123456789")` == 0x4 (the standard `CRC` check value)
//! - `checksum(&[])`          == 0x7
//! - `checksum(&[0x41])`      == 0x0
//! - `checksum(&[0x00])`      == 0x7

/// Low 3 bits mask for the register.
const MASK: u8 = 0b111;

/// Top bit of the 3-bit register.
const TOP: u8 = 0b100;

/// Generator polynomial (low 3 bits of 0x3).
const POLY: u8 = 0x3;

/// Initial register value.
const INIT: u8 = 0x0;

/// Final XOR value applied to the register.
const XOROUT: u8 = 0x7;

/// Advance the running `CRC` register by a single byte.
///
/// This does NOT apply `xorout`; it only folds `byte` into the register so it
/// can be used as a streaming/incremental interface. Call [`finalize`] (or
/// XOR with the public finalization value) once all bytes are consumed.
#[must_use]
pub fn update(crc: u8, byte: u8) -> u8 {
    let mut crc = crc & MASK;
    for i in (0..8u32).rev() {
        let inbit = (byte >> i) & 1;
        let msb = if (crc & TOP) != 0 { 1u8 } else { 0u8 };
        crc = (crc << 1) & MASK;
        if (msb ^ inbit) == 1 {
            crc ^= POLY;
        }
    }
    crc
}

/// Apply the final reflection (none) and `xorout` to a register value.
#[must_use]
pub fn finalize(crc: u8) -> u8 {
    (crc & MASK) ^ XOROUT
}

/// The register value used to seed a streaming computation.
#[must_use]
pub fn init() -> u8 {
    INIT
}

/// Compute the `CRC-3`/`GSM` checksum over `data`.
#[must_use]
pub fn checksum(data: &[u8]) -> u8 {
    let mut crc = INIT;
    for &byte in data {
        crc = update(crc, byte);
    }
    finalize(crc)
}

#[cfg(test)]
mod tests {
    use super::{checksum, finalize, init, update, INIT, XOROUT};

    /// Fold a slice using the streaming interface (mirrors `checksum`).
    fn fold(data: &[u8]) -> u8 {
        let mut crc = INIT;
        for &b in data {
            crc = update(crc, b);
        }
        finalize(crc)
    }

    /// Fold a slice processed as two chunks via the streaming interface.
    fn fold_split(a: &[u8], b: &[u8]) -> u8 {
        let mut crc = INIT;
        for &byte in a {
            crc = update(crc, byte);
        }
        for &byte in b {
            crc = update(crc, byte);
        }
        finalize(crc)
    }

    // ---- Hard reference vectors (must all hit) ----

    #[test]
    fn hard_vector_standard_check() {
        assert_eq!(checksum(b"123456789"), 0x4);
    }

    #[test]
    fn hard_vector_empty() {
        assert_eq!(checksum(&[]), 0x7);
    }

    #[test]
    fn hard_vector_single_0x41() {
        assert_eq!(checksum(&[0x41]), 0x0);
    }

    #[test]
    fn hard_vector_single_0x00() {
        assert_eq!(checksum(&[0x00]), 0x7);
    }

    // ---- Derived single-byte vectors (hand-verified) ----

    #[test]
    fn single_0x01_is_4() {
        assert_eq!(checksum(&[0x01]), 0x4);
    }

    #[test]
    fn single_0xff_is_4() {
        assert_eq!(checksum(&[0xff]), 0x4);
    }

    // ---- Structural identities ----

    #[test]
    fn empty_equals_init_xor_xorout() {
        assert_eq!(checksum(&[]), INIT ^ XOROUT);
    }

    #[test]
    fn init_is_zero() {
        assert_eq!(init(), 0x0);
    }

    #[test]
    fn finalize_of_init_matches_empty() {
        assert_eq!(finalize(init()), checksum(&[]));
    }

    #[test]
    fn finalize_applies_xorout() {
        assert_eq!(finalize(0x0), 0x7);
        assert_eq!(finalize(0x7), 0x0);
    }

    // ---- Range invariants ----

    #[test]
    fn checksum_fits_in_three_bits_for_many_inputs() {
        let samples: [&[u8]; 6] = [
            b"",
            b"a",
            b"123456789",
            &[0x00, 0xff, 0x55],
            &[0xde, 0xad, 0xbe, 0xef],
            b"The quick brown fox",
        ];
        for s in samples {
            assert_eq!(checksum(s) & 0xf8, 0);
        }
    }

    #[test]
    fn update_fits_in_three_bits() {
        for byte in 0u16..=255 {
            for crc in 0u8..=7 {
                assert_eq!(update(crc, byte as u8) & 0xf8, 0);
            }
        }
    }

    // ---- Incremental vs one-shot consistency ----

    #[test]
    fn incremental_matches_oneshot_standard() {
        assert_eq!(fold(b"123456789"), checksum(b"123456789"));
    }

    #[test]
    fn incremental_matches_oneshot_abc() {
        assert_eq!(fold(b"abc"), checksum(b"abc"));
    }

    #[test]
    fn incremental_matches_oneshot_hello() {
        let data = b"Hello, world!";
        assert_eq!(fold(data), checksum(data));
    }

    #[test]
    fn incremental_matches_oneshot_sequence() {
        let data: &[u8] = &[0, 1, 2, 3, 4, 5, 6, 7, 8, 9];
        assert_eq!(fold(data), checksum(data));
    }

    #[test]
    fn incremental_matches_oneshot_deadbeef() {
        let data: &[u8] = &[0xde, 0xad, 0xbe, 0xef];
        assert_eq!(fold(data), checksum(data));
    }

    #[test]
    fn incremental_matches_oneshot_all_ones() {
        let data: &[u8] = &[0xff; 8];
        assert_eq!(fold(data), checksum(data));
    }

    #[test]
    fn incremental_matches_oneshot_alternating() {
        let data: &[u8] = &[0x55, 0xaa, 0x55, 0xaa];
        assert_eq!(fold(data), checksum(data));
    }

    #[test]
    fn incremental_matches_oneshot_empty() {
        assert_eq!(fold(&[]), checksum(&[]));
    }

    #[test]
    fn incremental_matches_oneshot_single() {
        assert_eq!(fold(&[0x41]), checksum(&[0x41]));
    }

    // ---- Split-chunk consistency ----

    #[test]
    fn split_matches_whole_standard() {
        assert_eq!(fold_split(b"1234", b"56789"), checksum(b"123456789"));
    }

    #[test]
    fn split_matches_whole_at_zero() {
        assert_eq!(fold_split(b"", b"123456789"), checksum(b"123456789"));
    }

    #[test]
    fn split_matches_whole_at_end() {
        assert_eq!(fold_split(b"123456789", b""), checksum(b"123456789"));
    }

    #[test]
    fn split_matches_whole_hello() {
        let whole = b"Hello, world!";
        assert_eq!(fold_split(b"Hello, ", b"world!"), checksum(whole));
    }

    #[test]
    fn split_matches_whole_bytes() {
        let a: &[u8] = &[0xde, 0xad];
        let b: &[u8] = &[0xbe, 0xef];
        let whole: &[u8] = &[0xde, 0xad, 0xbe, 0xef];
        assert_eq!(fold_split(a, b), checksum(whole));
    }

    #[test]
    fn split_every_boundary_matches_whole() {
        let data: &[u8] = &[0x12, 0x34, 0x56, 0x78, 0x9a, 0xbc];
        let full = checksum(data);
        for i in 0..=data.len() {
            let (a, b) = data.split_at(i);
            assert_eq!(fold_split(a, b), full);
        }
    }

    // ---- update() building blocks ----

    #[test]
    fn update_from_init_single_byte_matches_checksum() {
        let crc = update(INIT, 0x41);
        assert_eq!(finalize(crc), checksum(&[0x41]));
    }

    #[test]
    fn update_zero_byte_from_init() {
        let crc = update(INIT, 0x00);
        assert_eq!(finalize(crc), checksum(&[0x00]));
    }

    #[test]
    fn update_two_bytes_matches_checksum() {
        let crc = update(update(INIT, 0x01), 0x02);
        assert_eq!(finalize(crc), checksum(&[0x01, 0x02]));
    }

    #[test]
    fn update_masks_high_bits_of_input_crc() {
        // High bits of the incoming register should not affect the result.
        assert_eq!(update(0xf8 | 0x5, 0x41), update(0x5, 0x41));
    }

    // ---- Different data yielding different checksums ----

    #[test]
    fn different_data_different_crc_empty_vs_a() {
        assert_ne!(checksum(&[]), checksum(&[0x41]));
    }

    #[test]
    fn different_data_different_crc_zero_vs_one() {
        assert_ne!(checksum(&[0x00]), checksum(&[0x01]));
    }

    #[test]
    fn different_data_different_crc_a_vs_one() {
        assert_ne!(checksum(&[0x41]), checksum(&[0x01]));
    }

    #[test]
    fn different_data_different_crc_zero_is_seven() {
        assert_ne!(checksum(&[0x41]), checksum(&[0x00]));
    }

    // ---- Determinism ----

    #[test]
    fn checksum_is_deterministic() {
        let data = b"determinism check";
        assert_eq!(checksum(data), checksum(data));
    }

    #[test]
    fn update_is_deterministic() {
        assert_eq!(update(0x3, 0xa5), update(0x3, 0xa5));
    }

    // ---- Multi-byte broad coverage ----

    #[test]
    fn multibyte_fold_matches_for_all_single_bytes() {
        for byte in 0u16..=255 {
            let data = [byte as u8];
            assert_eq!(fold(&data), checksum(&data));
        }
    }

    #[test]
    fn multibyte_fold_matches_for_all_byte_pairs_sample() {
        // Sweep a representative grid of pairs.
        let probes: [u8; 6] = [0x00, 0x01, 0x7f, 0x80, 0xaa, 0xff];
        for &a in &probes {
            for &b in &probes {
                let data = [a, b];
                assert_eq!(fold(&data), checksum(&data));
            }
        }
    }

    #[test]
    fn checksum_of_text_runs() {
        let data = b"The quick brown fox jumps over the lazy dog";
        assert_eq!(fold(data), checksum(data));
        assert_eq!(checksum(data) & 0xf8, 0);
    }

    #[test]
    fn repeated_byte_lengths_consistent() {
        for len in 0usize..=16 {
            let buf = [0xa5u8; 16];
            let data = &buf[..len];
            assert_eq!(fold(data), checksum(data));
        }
    }

    #[test]
    fn leading_zeros_change_result_vs_bare() {
        // Prepending a zero byte generally alters the register state.
        let bare = checksum(&[0x41]);
        let padded = checksum(&[0x00, 0x41]);
        // Both must be valid 3-bit values regardless of equality.
        assert_eq!(bare & 0xf8, 0);
        assert_eq!(padded & 0xf8, 0);
    }

    #[test]
    fn triple_chunk_matches_whole() {
        let whole: &[u8] = &[0x11, 0x22, 0x33, 0x44, 0x55, 0x66];
        let mut crc = INIT;
        for &b in &whole[0..2] {
            crc = update(crc, b);
        }
        for &b in &whole[2..4] {
            crc = update(crc, b);
        }
        for &b in &whole[4..6] {
            crc = update(crc, b);
        }
        assert_eq!(finalize(crc), checksum(whole));
    }

    #[test]
    fn finalize_round_trip_consistency() {
        // finalize is an XOR with a constant, so applying it twice is identity.
        for crc in 0u8..=7 {
            assert_eq!(finalize(finalize(crc)), crc);
        }
    }
}
