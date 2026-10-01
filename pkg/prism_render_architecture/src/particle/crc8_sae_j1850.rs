//! `CRC-8/SAE-J1850` byte checksum: a width-`8`, pure-integer `CRC` over a byte
//! slice, used as a tiny tamper-evident tag for compact particle control words
//! and short `GPU` command headers where a one-byte guard is enough.
//!
//! Register parameters (the full `CRC` definition): `width = 8`,
//! `poly = 0x1D`, `init = 0xFF`, `refin = false`, `refout = false`,
//! `xorout = 0xFF`. Because both `refin` and `refout` are `false` this is the
//! non-reflected, `MSB`-first form: each input byte is folded into the top of
//! the register and the polynomial is applied from the most significant bit
//! downward, with no bit reversal on input or output.
//!
//! Processing walks the input byte by byte. Each byte is combined into the
//! register with an `XOR`, then the register is shifted left eight times; on
//! every shift whose outgoing top bit is set, the polynomial `0x1D` is folded
//! back in with another `XOR`. After all bytes are consumed the register is
//! `XOR`-ed with `xorout` (`0xFF`) to produce the final one-byte checksum. The
//! register is a plain `u8`, so the left shifts naturally discard bits above
//! bit seven; there is no floating-point or transcendental work anywhere.
//!
//! Reference check values (verified): the empty input hashes to `0x00`, a
//! single `0x00` byte to `0x3B`, a single `0xFF` byte to `0xFF`, the one-byte
//! input `b"a"` to `0xB2`, and the canonical `b"123456789"` check string to
//! `0x4B`.
//!
//! Scope: this is an error-detection code for catching accidental corruption,
//! not a hash and not a message authentication code. An `8`-bit tag collides
//! trivially and offers no protection against a deliberate adversary, so it
//! must never be used to authenticate data. For content-addressing or security
//! use a real hash instead.

/// The `CRC-8/SAE-J1850` generator polynomial (the `x^8` term is implicit).
pub const POLY: u8 = 0x1D;

/// The register seed value applied before any input bytes are folded in.
pub const INIT: u8 = 0xFF;

/// The final value `XOR`-ed into the register to produce the checksum.
pub const XOROUT: u8 = 0xFF;

/// Computes the `CRC-8/SAE-J1850` checksum of `data`.
///
/// The register starts at [`INIT`] (`0xFF`); each byte is folded in with an
/// `XOR` and the register is shifted left eight times, folding in [`POLY`]
/// whenever the outgoing top bit is set. The result is the register `XOR`-ed
/// with [`XOROUT`] (`0xFF`). The empty slice yields `0x00`.
#[must_use]
pub fn crc8_sae_j1850(data: &[u8]) -> u8 {
    let mut reg: u8 = INIT;
    for &b in data {
        reg ^= b;
        let mut i: u8 = 0;
        while i < 8 {
            let top_set = (reg & 0x80) != 0;
            reg <<= 1;
            if top_set {
                reg ^= POLY;
            }
            i += 1;
        }
    }
    reg ^ XOROUT
}

#[cfg(test)]
mod tests {
    use super::{crc8_sae_j1850, INIT, POLY, XOROUT};

    // --- Four required anchor vectors -------------------------------------

    #[test]
    fn anchor_empty_is_0x00() {
        assert!(crc8_sae_j1850(b"") == 0x00);
    }

    #[test]
    fn anchor_single_zero_is_0x3b() {
        assert!(crc8_sae_j1850(&[0x00]) == 0x3b);
    }

    #[test]
    fn anchor_single_ff_is_0xff() {
        assert!(crc8_sae_j1850(&[0xff]) == 0xff);
    }

    #[test]
    fn anchor_check_string_is_0x4b() {
        assert!(crc8_sae_j1850(b"123456789") == 0x4b);
    }

    // --- Single-byte hardcoded vectors ------------------------------------

    #[test]
    fn single_byte_a_is_0xb2() {
        assert!(crc8_sae_j1850(b"a") == 0xb2);
    }

    #[test]
    fn single_byte_0x01_is_0x26() {
        assert!(crc8_sae_j1850(&[0x01]) == 0x26);
    }

    #[test]
    fn single_byte_empty_matches_literal() {
        let value = crc8_sae_j1850(&[]);
        assert!(value == 0x00);
    }

    // --- Multi-byte hardcoded vectors -------------------------------------

    #[test]
    fn multi_byte_ab_is_0xc3() {
        assert!(crc8_sae_j1850(b"ab") == 0xc3);
    }

    #[test]
    fn multi_byte_abc_is_0x9a() {
        assert!(crc8_sae_j1850(b"abc") == 0x9a);
    }

    #[test]
    fn multi_byte_hello_is_0xa9() {
        assert!(crc8_sae_j1850(b"hello") == 0xa9);
    }

    #[test]
    fn multi_byte_hello_world_is_0xbe() {
        assert!(crc8_sae_j1850(b"Hello, World!") == 0xbe);
    }

    #[test]
    fn multi_byte_four_zeros_is_0x59() {
        assert!(crc8_sae_j1850(&[0x00, 0x00, 0x00, 0x00]) == 0x59);
    }

    #[test]
    fn multi_byte_two_ff_is_0x3b() {
        assert!(crc8_sae_j1850(&[0xff, 0xff]) == 0x3b);
    }

    #[test]
    fn multi_byte_sequence_is_0x67() {
        assert!(crc8_sae_j1850(&[0x01, 0x02, 0x03, 0x04]) == 0x67);
    }

    #[test]
    fn multi_byte_deadbeef_is_0xb3() {
        assert!(crc8_sae_j1850(&[0xde, 0xad, 0xbe, 0xef]) == 0xb3);
    }

    #[test]
    fn multi_byte_zero_then_one_is_0xa3() {
        assert!(crc8_sae_j1850(&[0x00, 0x01]) == 0xa3);
    }

    #[test]
    fn multi_byte_one_then_zero_is_0xf2() {
        assert!(crc8_sae_j1850(&[0x01, 0x00]) == 0xf2);
    }

    #[test]
    fn multi_byte_eight_ff_is_0xc1() {
        assert!(crc8_sae_j1850(&[0xff; 8]) == 0xc1);
    }

    #[test]
    fn multi_byte_triple_0x55_is_0x88() {
        assert!(crc8_sae_j1850(&[0x55, 0x55, 0x55]) == 0x88);
    }

    #[test]
    fn multi_byte_aa_bb_cc_is_0x21() {
        assert!(crc8_sae_j1850(&[0xaa, 0xbb, 0xcc]) == 0x21);
    }

    #[test]
    fn multi_byte_range_0_to_15_is_0xfb() {
        let mut buf = [0u8; 16];
        let mut i = 0usize;
        while i < buf.len() {
            buf[i] = i as u8;
            i += 1;
        }
        assert!(crc8_sae_j1850(&buf) == 0xfb);
    }

    // --- Order sensitivity -------------------------------------------------

    #[test]
    fn order_matters_zero_one_differs_from_one_zero() {
        let forward = crc8_sae_j1850(&[0x00, 0x01]);
        let reversed = crc8_sae_j1850(&[0x01, 0x00]);
        assert!(forward != reversed);
    }

    #[test]
    fn order_matters_abc_differs_from_cba() {
        assert!(crc8_sae_j1850(b"abc") != crc8_sae_j1850(b"cba"));
    }

    // --- Determinism -------------------------------------------------------

    #[test]
    fn deterministic_check_string_repeats() {
        let first = crc8_sae_j1850(b"123456789");
        let second = crc8_sae_j1850(b"123456789");
        assert!(first == second);
    }

    #[test]
    fn deterministic_many_repeats_identical() {
        let expected = crc8_sae_j1850(b"Hello, World!");
        let mut i = 0u32;
        while i < 64 {
            assert!(crc8_sae_j1850(b"Hello, World!") == expected);
            i += 1;
        }
    }

    #[test]
    fn deterministic_empty_repeats() {
        let mut i = 0u32;
        while i < 32 {
            assert!(crc8_sae_j1850(b"") == 0x00);
            i += 1;
        }
    }

    // --- Long input stability ---------------------------------------------

    #[test]
    fn long_input_256_counting_is_0x05() {
        let mut buf = [0u8; 256];
        let mut i = 0usize;
        while i < buf.len() {
            buf[i] = i as u8;
            i += 1;
        }
        assert!(crc8_sae_j1850(&buf) == 0x05);
    }

    #[test]
    fn long_input_100_zeros_is_0xa6() {
        let buf = [0u8; 100];
        assert!(crc8_sae_j1850(&buf) == 0xa6);
    }

    #[test]
    fn long_input_100_ff_is_0x80() {
        let buf = [0xffu8; 100];
        assert!(crc8_sae_j1850(&buf) == 0x80);
    }

    #[test]
    fn long_input_stable_across_calls() {
        let mut buf = [0u8; 512];
        let mut i = 0usize;
        while i < buf.len() {
            buf[i] = (i & 0xff) as u8;
            i += 1;
        }
        let first = crc8_sae_j1850(&buf);
        let second = crc8_sae_j1850(&buf);
        assert!(first == second);
    }

    // --- Structural / incremental consistency -----------------------------

    #[test]
    fn prefix_changes_checksum() {
        let base = crc8_sae_j1850(b"payload");
        let extended = crc8_sae_j1850(b"payload!");
        assert!(base != extended);
    }

    #[test]
    fn single_bit_flip_changes_checksum() {
        let clean = crc8_sae_j1850(&[0x00, 0x00, 0x00, 0x00]);
        let dirty = crc8_sae_j1850(&[0x00, 0x00, 0x00, 0x01]);
        assert!(clean != dirty);
    }

    #[test]
    fn result_fits_u8_range() {
        let value = crc8_sae_j1850(b"range probe");
        assert!((0x00..=0xff).contains(&value));
    }

    #[test]
    fn trailing_zero_byte_changes_checksum() {
        let short = crc8_sae_j1850(b"abc");
        let padded = crc8_sae_j1850(&[b'a', b'b', b'c', 0x00]);
        assert!(short != padded);
    }

    // --- Parameter sanity --------------------------------------------------

    #[test]
    fn parameters_have_expected_constants() {
        assert!(POLY == 0x1d);
        assert!(INIT == 0xff);
        assert!(XOROUT == 0xff);
    }

    #[test]
    fn empty_equals_init_xor_xorout() {
        assert!(crc8_sae_j1850(b"") == (INIT ^ XOROUT));
    }

    #[test]
    fn length_multiple_of_four_is_handled() {
        let buf = [0x12u8, 0x34, 0x56, 0x78, 0x9a, 0xbc, 0xde, 0xf0];
        assert!(buf.len().is_multiple_of(4));
        let value = crc8_sae_j1850(&buf);
        let again = crc8_sae_j1850(&buf);
        assert!(value == again);
    }

    #[test]
    fn concatenation_differs_from_single_segment() {
        let whole = crc8_sae_j1850(b"abcdef");
        let left = crc8_sae_j1850(b"abc");
        assert!(whole != left);
    }
}
