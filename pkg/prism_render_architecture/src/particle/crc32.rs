//! Standard `CRC-32`/`ISO-HDLC` checksum (the `zlib`/`PNG`/`gzip` variant): a
//! pure-integer, table-driven cyclic redundancy check for verifying the
//! integrity of `GPU` resource blobs and network packet payloads (design §
//! integrity checks).
//!
//! This module implements exactly one well-known parameterisation: the
//! reflected `CRC-32`/`ISO-HDLC` polynomial `0xEDB88320` (the bit-reversed form
//! of the `IEEE` 802.3 generator), an initial register value of `0xFFFFFFFF`,
//! reflected input and output, and a final exclusive-or of `0xFFFFFFFF`. That
//! is the same checksum produced by `zlib`'s `crc32`, by `PNG` chunk `CRC`
//! fields, and by `gzip` trailers, so a value computed here can be compared
//! directly against those toolchains. The canonical *check* value for the
//! `ASCII` string `123456789` is `0xCBF43926`.
//!
//! The implementation is the classic byte-at-a-time table method. A `256`-entry
//! lookup table is built at compile time by a `const fn`, one entry per possible
//! low byte of the register; each [`crc32_update`] step folds one input byte
//! into the register with a table lookup, a right shift, and an exclusive-or.
//! The running time is linear in the input length, `O(n)`, with a small
//! constant per byte. Every operation is an integer shift, exclusive-or, or
//! wrapping mask; there are no floating-point or transcendental operations.
//!
//! Two surfaces are exposed. [`crc32`] computes the checksum of a whole slice in
//! one call. The streaming trio [`crc32_init`], [`crc32_update`], and
//! [`crc32_finalize`] lets a caller fold data in arbitrarily many chunks:
//! `crc32_init` returns the internal register seed, `crc32_update` folds bytes
//! into that register, and `crc32_finalize` applies the trailing output
//! reflection and exclusive-or. By construction
//! `crc32(x) == crc32_finalize(crc32_update(crc32_init(), x))` for any slice
//! `x`, and folding a byte stream in one call or in several `crc32_update`
//! calls yields the same register.
//!
//! Scope: this is an error-detection code, not a hash and not a message
//! authentication code. `CRC-32` is *not* cryptographically secure — it is
//! trivially invertible and collisions are easy to construct on purpose — so it
//! must never be used to authenticate data or guard against a malicious
//! adversary. It is meant only for catching accidental corruption in transit or
//! storage. For content-addressing or security use a real hash instead.

/// The reflected `CRC-32`/`ISO-HDLC` generator polynomial (`0xEDB88320`).
const POLYNOMIAL: u32 = 0xEDB8_8320;

/// The register seed and the final output mask (`0xFFFFFFFF`).
const ALL_ONES: u32 = 0xFFFF_FFFF;

/// Builds the `256`-entry byte-at-a-time lookup table at compile time.
///
/// Entry `i` holds the register contribution of folding the byte `i` into an
/// otherwise-zero register, using the reflected polynomial.
const fn build_table() -> [u32; 256] {
    let mut table = [0u32; 256];
    let mut i = 0usize;
    while i < 256 {
        let mut crc = i as u32;
        let mut bit = 0u32;
        while bit < 8 {
            if crc & 1 == 1 {
                crc = (crc >> 1) ^ POLYNOMIAL;
            } else {
                crc >>= 1;
            }
            bit += 1;
        }
        table[i] = crc;
        i += 1;
    }
    table
}

/// The precomputed byte-at-a-time lookup table.
const TABLE: [u32; 256] = build_table();

/// Returns the internal register seed for a streaming computation.
///
/// This is the pre-reflection initial value `0xFFFFFFFF`, not a finished
/// checksum. Feed it to [`crc32_update`] and pass the result to
/// [`crc32_finalize`].
#[must_use]
pub const fn crc32_init() -> u32 {
    ALL_ONES
}

/// Folds every byte of `data` into the running register `crc`.
///
/// `crc` must be a register value obtained from [`crc32_init`] or a previous
/// `crc32_update`; it is *not* a finished checksum. The result is again a raw
/// register value and must be passed through [`crc32_finalize`] before it can
/// be compared against a standard `CRC-32` value.
#[must_use]
pub fn crc32_update(crc: u32, data: &[u8]) -> u32 {
    let mut crc = crc;
    for &byte in data {
        let index = ((crc ^ u32::from(byte)) & 0xFF) as usize;
        crc = (crc >> 8) ^ TABLE[index];
    }
    crc
}

/// Applies the trailing output reflection and exclusive-or to a register.
///
/// Turns a raw register value produced by [`crc32_update`] into the finished
/// `CRC-32`/`ISO-HDLC` checksum by xoring with `0xFFFFFFFF`.
#[must_use]
pub const fn crc32_finalize(crc: u32) -> u32 {
    crc ^ ALL_ONES
}

/// Computes the `CRC-32`/`ISO-HDLC` checksum of `data` in a single call.
///
/// Equivalent to
/// `crc32_finalize(crc32_update(crc32_init(), data))`. The empty slice yields
/// `0`, and `crc32(b"123456789")` is the canonical check value `0xCBF43926`.
#[must_use]
pub fn crc32(data: &[u8]) -> u32 {
    crc32_finalize(crc32_update(crc32_init(), data))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A tiny deterministic linear congruential generator for test data.
    struct Lcg {
        state: u64,
    }

    impl Lcg {
        const fn new(seed: u64) -> Self {
            Self { state: seed }
        }

        fn next_u8(&mut self) -> u8 {
            // Numerical Recipes constants; wrapping keeps it in-range.
            self.state = self
                .state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            (self.state >> 56) as u8
        }

        fn bytes(&mut self, len: usize) -> alloc::vec::Vec<u8> {
            (0..len).map(|_| self.next_u8()).collect()
        }
    }

    #[test]
    fn empty_input_is_zero() {
        assert_eq!(crc32(b""), 0);
    }

    #[test]
    fn standard_check_vector() {
        assert_eq!(crc32(b"123456789"), 0xCBF4_3926);
    }

    #[test]
    fn single_byte_a() {
        assert_eq!(crc32(b"a"), 0xE8B7_BE43);
    }

    #[test]
    fn single_byte_zero() {
        assert_eq!(crc32(&[0u8]), 0xD202_EF8D);
    }

    #[test]
    fn short_ascii_abc() {
        assert_eq!(crc32(b"abc"), 0x3524_41C2);
    }

    #[test]
    fn known_vector_the_quick_brown_fox() {
        // zlib crc32 of the classic pangram.
        assert_eq!(
            crc32(b"The quick brown fox jumps over the lazy dog"),
            0x414F_A339
        );
    }

    #[test]
    fn known_vector_all_zeros_32() {
        let data = [0u8; 32];
        // Matches zlib crc32 of 32 zero bytes.
        assert_eq!(crc32(&data), 0x190A_55AD);
    }

    #[test]
    fn known_vector_all_ones_32() {
        let data = [0xFFu8; 32];
        assert_eq!(crc32(&data), 0xFF6C_AB0B);
    }

    #[test]
    fn init_then_finalize_is_zero() {
        assert_eq!(crc32_finalize(crc32_init()), 0);
    }

    #[test]
    fn streaming_equals_oneshot_empty() {
        let crc = crc32_finalize(crc32_update(crc32_init(), b""));
        assert_eq!(crc, crc32(b""));
    }

    #[test]
    fn streaming_equals_oneshot_check_vector() {
        let crc = crc32_finalize(crc32_update(crc32_init(), b"123456789"));
        assert_eq!(crc, crc32(b"123456789"));
        assert_eq!(crc, 0xCBF4_3926);
    }

    #[test]
    fn two_chunk_split_matches_whole() {
        let whole = b"Hello, world!";
        let (left, right) = whole.split_at(5);
        let mut reg = crc32_init();
        reg = crc32_update(reg, left);
        reg = crc32_update(reg, right);
        assert_eq!(crc32_finalize(reg), crc32(whole));
    }

    #[test]
    fn byte_by_byte_matches_whole() {
        let data = b"streaming byte by byte should match";
        let mut reg = crc32_init();
        for &b in data {
            reg = crc32_update(reg, &[b]);
        }
        assert_eq!(crc32_finalize(reg), crc32(data));
    }

    #[test]
    fn many_chunk_boundaries_match_whole() {
        let mut lcg = Lcg::new(0x1234_5678_9ABC_DEF0);
        let data = lcg.bytes(1000);
        for split in [0usize, 1, 7, 63, 64, 65, 256, 511, 999, 1000] {
            let (left, right) = data.split_at(split);
            let mut reg = crc32_init();
            reg = crc32_update(reg, left);
            reg = crc32_update(reg, right);
            assert_eq!(crc32_finalize(reg), crc32(&data));
        }
    }

    #[test]
    fn three_way_split_matches_whole() {
        let data = b"abcdefghijklmnopqrstuvwxyz0123456789";
        let a = &data[..10];
        let b = &data[10..20];
        let c = &data[20..];
        let mut reg = crc32_init();
        reg = crc32_update(reg, a);
        reg = crc32_update(reg, b);
        reg = crc32_update(reg, c);
        assert_eq!(crc32_finalize(reg), crc32(data));
    }

    #[test]
    fn concatenation_associativity() {
        let left = b"first part ";
        let right = b"second part";
        let mut joined = alloc::vec::Vec::new();
        joined.extend_from_slice(left);
        joined.extend_from_slice(right);

        let mut reg = crc32_init();
        reg = crc32_update(reg, left);
        reg = crc32_update(reg, right);
        assert_eq!(crc32_finalize(reg), crc32(&joined));
    }

    #[test]
    fn long_uniform_data() {
        let data = alloc::vec![0x5Au8; 100_000];
        let oneshot = crc32(&data);
        let mut reg = crc32_init();
        reg = crc32_update(reg, &data);
        assert_eq!(crc32_finalize(reg), oneshot);
    }

    #[test]
    fn random_data_streaming_matches_oneshot() {
        let mut lcg = Lcg::new(0xDEAD_BEEF_CAFE_F00D);
        let data = lcg.bytes(4096);
        let oneshot = crc32(&data);
        let mut reg = crc32_init();
        for chunk in data.chunks(37) {
            reg = crc32_update(reg, chunk);
        }
        assert_eq!(crc32_finalize(reg), oneshot);
    }

    #[test]
    fn random_lengths_streaming_matches_oneshot() {
        let mut lcg = Lcg::new(0x0F0F_0F0F_1234_5678);
        for len in [0usize, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144] {
            let data = lcg.bytes(len);
            let oneshot = crc32(&data);
            let mut reg = crc32_init();
            for chunk in data.chunks(4) {
                reg = crc32_update(reg, chunk);
            }
            assert_eq!(crc32_finalize(reg), oneshot);
        }
    }

    #[test]
    fn empty_update_leaves_register_unchanged() {
        let reg = crc32_init();
        assert_eq!(crc32_update(reg, b""), reg);
    }

    #[test]
    fn table_entry_zero_is_zero() {
        assert_eq!(TABLE[0], 0);
    }

    #[test]
    fn table_entries_are_distinct() {
        // Distinctness of a handful of entries as a smoke test.
        assert_ne!(TABLE[1], TABLE[2]);
        assert_ne!(TABLE[2], TABLE[255]);
        assert_ne!(TABLE[128], TABLE[64]);
    }

    #[test]
    fn different_data_differs() {
        assert_ne!(crc32(b"abc"), crc32(b"abd"));
        assert_ne!(crc32(b"abc"), crc32(b"acb"));
    }

    #[test]
    fn order_sensitivity() {
        assert_ne!(crc32(b"ab"), crc32(b"ba"));
    }

    #[test]
    fn length_prefix_changes_result() {
        assert_ne!(crc32(b"12345678"), crc32(b"123456789"));
    }

    #[test]
    fn appending_zero_changes_result() {
        let base = crc32(b"data");
        let extended = crc32(b"data\0");
        assert_ne!(base, extended);
    }

    #[test]
    fn incremental_extension_matches_full() {
        let prefix = b"prefix-";
        let suffix = b"suffix";
        let mut reg = crc32_init();
        reg = crc32_update(reg, prefix);
        let extended = crc32_finalize(crc32_update(reg, suffix));

        let mut whole = alloc::vec::Vec::new();
        whole.extend_from_slice(prefix);
        whole.extend_from_slice(suffix);
        assert_eq!(extended, crc32(&whole));
    }

    #[test]
    fn repeated_calls_are_pure() {
        let data = b"idempotent";
        assert_eq!(crc32(data), crc32(data));
    }

    #[test]
    fn every_split_of_medium_buffer_matches() {
        let mut lcg = Lcg::new(0xABCD_1234_5678_9F00);
        let data = lcg.bytes(300);
        let oneshot = crc32(&data);
        for split in 0..=data.len() {
            let (left, right) = data.split_at(split);
            let mut reg = crc32_init();
            reg = crc32_update(reg, left);
            reg = crc32_update(reg, right);
            assert_eq!(crc32_finalize(reg), oneshot);
        }
    }
}
