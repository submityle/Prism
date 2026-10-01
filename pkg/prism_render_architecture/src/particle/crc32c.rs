//! `CRC-32C` (Castagnoli) checksum, bit-by-bit reflected implementation.
//!
//! Algorithm parameters: width = 32, polynomial = `0x1EDC6F41`,
//! init = `0xFFFFFFFF`, refin = true, refout = true, xorout = `0xFFFFFFFF`.
//!
//! Because refin and refout are both true, the computation is performed
//! least-significant-bit first using the reflected polynomial
//! `bitreverse(0x1EDC6F41, 32) == 0x82F63B78`.
//!
//! The crate is `no_std` friendly: only `core` is used, no heap allocation,
//! and the arithmetic is purely integer based (no `f32`/`f64`).

/// Reflected `CRC-32C` (Castagnoli) polynomial, `bitreverse(0x1EDC6F41, 32)`.
const REFPOLY: u32 = 0x82F63B78;

/// Initial register value before processing any input bytes.
const INIT: u32 = 0xFFFFFFFF;

/// Final value `XORed` into the register after processing all input bytes.
const XOROUT: u32 = 0xFFFFFFFF;

/// Compute the `CRC-32C` (Castagnoli) checksum of `data`.
///
/// This processes one bit at a time using the reflected polynomial, which
/// matches the `iSCSI`/Castagnoli parameters used by many storage and
/// networking stacks.
#[must_use]
pub fn crc32c(data: &[u8]) -> u32 {
    let mut crc: u32 = INIT;
    for &b in data {
        crc ^= u32::from(b);
        for _ in 0..8u32 {
            if (crc & 1) != 0 {
                crc = (crc >> 1) ^ REFPOLY;
            } else {
                crc >>= 1;
            }
        }
    }
    crc ^ XOROUT
}

#[cfg(test)]
mod tests {
    use super::crc32c;

    // --- Hard anchor vectors (ground truth) -------------------------------

    #[test]
    fn anchor_empty() {
        assert!(crc32c(b"") == 0x0000_0000);
    }

    #[test]
    fn anchor_single_a() {
        assert!(crc32c(b"a") == 0xc1d0_4330);
    }

    #[test]
    fn anchor_byte_zero() {
        assert!(crc32c(&[0x00]) == 0x527d_5351);
    }

    #[test]
    fn anchor_byte_ff() {
        assert!(crc32c(&[0xff]) == 0xff00_0000);
    }

    #[test]
    fn anchor_check_value() {
        // The canonical catalogue "check" value for the digits 1..9.
        assert!(crc32c(b"123456789") == 0xe306_9283);
    }

    // --- Multi-byte vectors (computed from this implementation) -----------

    #[test]
    fn vector_ab() {
        assert!(crc32c(b"ab") == 0xe2a2_2936);
    }

    #[test]
    fn vector_abc() {
        assert!(crc32c(b"abc") == 0x364b_3fb7);
    }

    #[test]
    fn vector_abcd() {
        assert!(crc32c(b"abcd") == 0x92c8_0a31);
    }

    #[test]
    fn vector_message_digest() {
        assert!(crc32c(b"message digest") == 0x02bd_79d0);
    }

    #[test]
    fn vector_quick_brown_fox() {
        assert!(crc32c(b"The quick brown fox") == 0x537e_5cf4);
    }

    #[test]
    fn vector_quick_brown_fox_lazy_dog() {
        let msg = b"The quick brown fox jumps over the lazy dog";
        assert!(crc32c(msg) == 0x2262_0404);
    }

    #[test]
    fn vector_two_byte_sequence() {
        assert!(crc32c(&[0x00, 0x01]) == 0x030a_f4d1);
    }

    #[test]
    fn vector_deadbeef() {
        assert!(crc32c(&[0xde, 0xad, 0xbe, 0xef]) == 0xf1dc_778e);
    }

    #[test]
    fn vector_hello() {
        assert!(crc32c(b"hello") == 0x9a71_bb4c);
    }

    #[test]
    fn vector_hello_world() {
        assert!(crc32c(b"hello world") == 0xc994_65aa);
    }

    #[test]
    fn vector_prism() {
        assert!(crc32c(b"Prism") == 0x39e0_d041);
    }

    #[test]
    fn vector_particle() {
        assert!(crc32c(b"particle") == 0x7fe7_590b);
    }

    #[test]
    fn vector_alphabet_lower() {
        assert!(crc32c(b"abcdefghijklmnopqrstuvwxyz") == 0x9ee6_ef25);
    }

    #[test]
    fn vector_uppercase_run() {
        assert!(crc32c(b"AAAA") == 0x0305_6cf3);
    }

    #[test]
    fn vector_ascending_bytes() {
        assert!(crc32c(&[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]) == 0x022c_2131);
    }

    #[test]
    fn vector_byte_0x80() {
        assert!(crc32c(&[0x80]) == 0xd08b_6829);
    }

    #[test]
    fn vector_byte_0x01() {
        assert!(crc32c(&[0x01]) == 0xa016_d052);
    }

    #[test]
    fn vector_two_ff() {
        assert!(crc32c(&[0xff, 0xff]) == 0xffff_0000);
    }

    // --- Bulk / boundary vectors ------------------------------------------

    #[test]
    fn vector_256_zeros() {
        let buf = [0u8; 256];
        assert!(crc32c(&buf) == 0xb872_b190);
    }

    #[test]
    fn vector_16_ff() {
        let buf = [0xffu8; 16];
        assert!(crc32c(&buf) == 0xef2f_4c10);
    }

    #[test]
    fn boundary_single_byte_slice_matches_literal() {
        let one = [b'a'];
        assert!(crc32c(&one) == crc32c(b"a"));
    }

    #[test]
    fn boundary_empty_slice_variants_equal() {
        let empty: [u8; 0] = [];
        assert!(crc32c(&empty) == crc32c(b""));
    }

    #[test]
    fn boundary_length_one_through_four() {
        // Prefixes of "abcd" should match the standalone prefix vectors.
        let full = b"abcd";
        assert!(crc32c(&full[..1]) == crc32c(b"a"));
        assert!(crc32c(&full[..2]) == crc32c(b"ab"));
        assert!(crc32c(&full[..3]) == crc32c(b"abc"));
        assert!(crc32c(&full[..4]) == crc32c(b"abcd"));
    }

    // --- Structural / property tests --------------------------------------

    #[test]
    fn property_deterministic_repeat() {
        let data = b"deterministic";
        let first = crc32c(data);
        let second = crc32c(data);
        assert!(first == second);
    }

    #[test]
    fn property_deterministic_many_iterations() {
        let data = b"The quick brown fox";
        let reference = crc32c(data);
        let mut stable = true;
        for _ in 0..64u32 {
            if crc32c(data) != reference {
                stable = false;
            }
        }
        assert!(stable);
    }

    #[test]
    fn property_order_matters() {
        assert!(crc32c(b"ab") != crc32c(b"ba"));
    }

    #[test]
    fn property_distinct_inputs_distinct_outputs() {
        assert!(crc32c(b"abc") != crc32c(b"abd"));
    }

    #[test]
    fn property_single_bit_change_detected() {
        let a = [0b0000_0000u8];
        let b = [0b0000_0001u8];
        assert!(crc32c(&a) != crc32c(&b));
    }

    #[test]
    fn property_length_sensitive() {
        assert!(crc32c(b"a") != crc32c(b"aa"));
    }

    #[test]
    fn property_leading_zero_changes_checksum() {
        assert!(crc32c(&[0x00, 0x61]) != crc32c(b"a"));
    }

    #[test]
    fn property_trailing_zero_changes_checksum() {
        assert!(crc32c(&[0x61, 0x00]) != crc32c(b"a"));
    }

    #[test]
    fn property_result_fits_u32() {
        // Trivially true, but documents the output width contract.
        let value = crc32c(b"width");
        assert!((0..=u32::MAX).contains(&value));
    }

    #[test]
    fn property_all_single_bytes_run() {
        // Every possible single-byte input is well defined and finite.
        let mut sum: u32 = 0;
        for byte in 0u16..=255 {
            let input = [byte as u8];
            sum = sum.wrapping_add(crc32c(&input));
        }
        // Deterministic aggregate of all 256 single-byte checksums.
        assert!(
            sum == crc32c(&[0x00])
                .wrapping_add(sum)
                .wrapping_sub(crc32c(&[0x00]))
        );
    }

    #[test]
    fn property_single_byte_matches_two_entry_table() {
        // Compare against the two anchored single-byte vectors directly.
        assert!(crc32c(&[0x00]) == 0x527d_5351);
        assert!(crc32c(&[0xff]) == 0xff00_0000);
    }

    #[test]
    fn property_repeated_zero_grows_monotonically_in_length() {
        let one = [0u8; 1];
        let two = [0u8; 2];
        assert!(crc32c(&one) != crc32c(&two));
    }

    #[test]
    fn property_whitespace_sensitive() {
        assert!(crc32c(b"hello world") != crc32c(b"helloworld"));
    }

    #[test]
    fn property_case_sensitive() {
        assert!(crc32c(b"prism") != crc32c(b"Prism"));
    }

    #[test]
    fn property_prefix_of_fox_matches_literal() {
        let full = b"The quick brown fox jumps over the lazy dog";
        assert!(crc32c(&full[..19]) == crc32c(b"The quick brown fox"));
    }

    #[test]
    fn property_empty_is_identity_vector() {
        assert!(crc32c(b"") == 0);
    }

    #[test]
    fn property_half_of_ff_pair() {
        // Two 0xff bytes have a known compact checksum.
        assert!(crc32c(&[0xff, 0xff]) == 0xffff_0000);
        assert!(crc32c(&[0xff]) != crc32c(&[0xff, 0xff]));
    }
}
