//! `CRC-5/G-704` (also catalogued as `CRC-5/ITU`) checksum over byte slices.
//!
//! This module implements the reflected, bitwise `CRC` algorithm for the
//! `CRC-5/G-704` catalog entry. The algorithm parameters are fixed: width 5,
//! polynomial `0x15`, initial value `0x00`, `refin` true, `refout` true, and
//! `xorout` `0x00`. The computed checksum is a 5-bit value returned in the low
//! bits of a `u8`; the upper three bits are always clear.
//!
//! Because `refin` and `refout` are both true, the register is processed
//! least-significant-bit first using the reflected polynomial. The reflected
//! polynomial of `0x15` over five bits is itself `0x15` because `0b10101` is a
//! palindrome. Each input byte is combined into the register with `XOR` and the
//! register is reduced eight times; the low five bits then hold the reflected
//! output directly, so no separate final reflection step is required.
//!
//! The implementation is intended for `no_std` + `alloc` crates: it performs
//! pure integer arithmetic with no floating-point, transcendental, or heap
//! operations, and the public surface accepts `&[u8]` and returns `u8`.

/// Reflected polynomial for the width-5 `CRC-5/G-704` algorithm.
///
/// The forward polynomial `0x15` (`0b10101`) is a palindrome, so its five-bit
/// reflection is identical.
const REFPOLY: u8 = 0x15;

/// Mask retaining the low five bits of the `CRC` register.
const MASK: u8 = 0x1f;

/// Number of bit reductions performed per input byte.
const BITS_PER_BYTE: u8 = 8;

/// Computes the `CRC-5/G-704` checksum of `data`.
///
/// The register starts at the initial value `0x00`. For every input byte the
/// byte is folded into the register with `XOR` and the register is reduced one
/// bit at a time, eight times, using the reflected polynomial `REFPOLY`. The
/// returned value occupies the low five bits of the `u8`; bits five through
/// seven are always zero.
///
/// # Examples
///
/// ```ignore
/// let check = crc5_g704(b"123456789");
/// assert!(check == 0x07);
/// ```
pub fn crc5_g704(data: &[u8]) -> u8 {
    let mut crc: u8 = 0;
    for &byte in data {
        crc ^= byte;
        for _ in 0..BITS_PER_BYTE {
            if (crc & 1) != 0 {
                crc = (crc >> 1) ^ REFPOLY;
            } else {
                crc >>= 1;
            }
        }
    }
    crc & MASK
}

#[cfg(test)]
mod tests {
    use super::crc5_g704;

    #[test]
    fn anchor_empty_input_is_zero() {
        assert!(crc5_g704(b"") == 0x00);
    }

    #[test]
    fn anchor_single_letter_a() {
        assert!(crc5_g704(b"a") == 0x02);
    }

    #[test]
    fn anchor_single_zero_byte() {
        assert!(crc5_g704(&[0x00]) == 0x00);
    }

    #[test]
    fn anchor_single_full_byte() {
        assert!(crc5_g704(&[0xff]) == 0x1b);
    }

    #[test]
    fn anchor_catalog_check_value() {
        assert!(crc5_g704(b"123456789") == 0x07);
    }

    #[test]
    fn single_byte_0x01() {
        assert!(crc5_g704(&[0x01]) == 0x07);
    }

    #[test]
    fn single_byte_0x02() {
        assert!(crc5_g704(&[0x02]) == 0x0e);
    }

    #[test]
    fn single_byte_0x80() {
        assert!(crc5_g704(&[0x80]) == 0x15);
    }

    #[test]
    fn single_byte_digit_one() {
        assert!(crc5_g704(b"1") == 0x10);
    }

    #[test]
    fn single_byte_0x55() {
        assert!(crc5_g704(&[0x55]) == 0x09);
    }

    #[test]
    fn single_byte_0xaa() {
        assert!(crc5_g704(&[0xaa]) == 0x12);
    }

    #[test]
    fn single_byte_0x7f() {
        assert!(crc5_g704(&[0x7f]) == 0x0e);
    }

    #[test]
    fn prefix_two_digits() {
        assert!(crc5_g704(b"12") == 0x14);
    }

    #[test]
    fn prefix_three_digits() {
        assert!(crc5_g704(b"123") == 0x0f);
    }

    #[test]
    fn prefix_four_digits() {
        assert!(crc5_g704(b"1234") == 0x0d);
    }

    #[test]
    fn prefix_five_digits() {
        assert!(crc5_g704(b"12345") == 0x04);
    }

    #[test]
    fn prefix_six_digits() {
        assert!(crc5_g704(b"123456") == 0x19);
    }

    #[test]
    fn prefix_seven_digits() {
        assert!(crc5_g704(b"1234567") == 0x1b);
    }

    #[test]
    fn prefix_eight_digits() {
        assert!(crc5_g704(b"12345678") == 0x13);
    }

    #[test]
    fn double_letter_aa() {
        assert!(crc5_g704(b"aa") == 0x0c);
    }

    #[test]
    fn pair_letters_ab() {
        assert!(crc5_g704(b"ab") == 0x05);
    }

    #[test]
    fn pair_letters_ba() {
        assert!(crc5_g704(b"ba") == 0x18);
    }

    #[test]
    fn byte_pair_01_02() {
        assert!(crc5_g704(&[0x01, 0x02]) == 0x1b);
    }

    #[test]
    fn byte_pair_02_01() {
        assert!(crc5_g704(&[0x02, 0x01]) == 0x06);
    }

    #[test]
    fn two_zero_bytes_stay_zero() {
        assert!(crc5_g704(&[0x00, 0x00]) == 0x00);
    }

    #[test]
    fn three_zero_bytes_stay_zero() {
        assert!(crc5_g704(&[0x00, 0x00, 0x00]) == 0x00);
    }

    #[test]
    fn leading_zero_before_letter_a() {
        assert!(crc5_g704(&[0x00, 0x61]) == 0x02);
    }

    #[test]
    fn leading_zero_before_catalog() {
        let data = [0x00u8, b'1', b'2', b'3', b'4', b'5', b'6', b'7', b'8', b'9'];
        assert!(crc5_g704(&data) == 0x07);
    }

    #[test]
    fn order_sensitivity_byte_pair() {
        let forward = crc5_g704(&[0x01, 0x02]);
        let reverse = crc5_g704(&[0x02, 0x01]);
        assert!(forward != reverse);
    }

    #[test]
    fn order_sensitivity_letters() {
        let forward = crc5_g704(b"ab");
        let reverse = crc5_g704(b"ba");
        assert!(forward != reverse);
    }

    #[test]
    fn result_bounded_for_empty() {
        assert!(crc5_g704(b"") <= 0x1f);
    }

    #[test]
    fn result_bounded_for_catalog() {
        assert!(crc5_g704(b"123456789") <= 0x1f);
    }

    #[test]
    fn single_byte_results_within_five_bits() {
        for &byte in &[0x00u8, 0x01, 0x02, 0x55, 0x7f, 0x80, 0xaa, 0xff] {
            assert!(crc5_g704(&[byte]) <= 0x1f);
        }
    }

    #[test]
    fn high_bits_clear_for_full_byte() {
        assert!((crc5_g704(&[0xff]) >> 5) == 0);
    }

    #[test]
    fn high_bits_clear_for_catalog() {
        assert!((crc5_g704(b"123456789") >> 5) == 0);
    }

    #[test]
    fn high_bits_clear_across_samples() {
        for &byte in &[0x00u8, 0x3c, 0x81, 0xff] {
            let crc = crc5_g704(&[byte]);
            assert!((crc >> 5) == 0);
        }
    }

    #[test]
    fn repeated_calls_are_deterministic() {
        let first = crc5_g704(b"123456789");
        let second = crc5_g704(b"123456789");
        assert!(first == second);
        assert!(first == 0x07);
    }

    #[test]
    fn distinct_single_bytes_differ() {
        let one = crc5_g704(&[0x01]);
        let two = crc5_g704(&[0x02]);
        assert!(one != two);
    }

    #[test]
    fn catalog_differs_from_letter_a() {
        let catalog = crc5_g704(b"123456789");
        let letter = crc5_g704(b"a");
        assert!(catalog != letter);
    }

    #[test]
    fn single_byte_0x01_is_nonzero() {
        assert!(crc5_g704(&[0x01]) != 0x00);
    }

    #[test]
    fn masked_result_matches_raw_result() {
        let crc = crc5_g704(b"1234567");
        assert!(crc == (crc & 0x1f));
    }

    #[test]
    fn appending_zero_byte_changes_nonzero_state() {
        let base = crc5_g704(b"a");
        let extended = crc5_g704(&[0x61, 0x00]);
        assert!(base != extended);
    }
}
