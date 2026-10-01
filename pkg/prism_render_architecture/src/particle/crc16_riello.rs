//! `CRC`-16/RIELLO checksum computed `LSB`-first (reflected) over a byte slice.
//!
//! Parameters: width = 16, polynomial = `0x1021`, `init` = `0xB2AA`
//! (reflected-domain starting `crc` value `0x554D`), `refin` = true,
//! `refout` = true, `xorout` = `0x0000`. Because the algorithm is reflected,
//! the loop processes each bit `LSB`-first using the reflected polynomial
//! `REFPOLY` = `0x8408`. No final `XOR` stage is applied since `xorout` is
//! zero. The accumulator register is a `u16`. The check value for the ASCII
//! input `b"123456789"` is `0x63D0`.

/// Reflected form of the `0x1021` polynomial, used by the `LSB`-first loop.
const REFPOLY: u16 = 0x8408;

/// Reflected-domain starting value, equal to `reflect(0xB2AA)`.
const INIT: u16 = 0x554D;

/// Computes the `CRC`-16/RIELLO checksum of `data`.
///
/// The register is seeded with the reflected initial value `INIT` (`0x554D`),
/// each input byte is folded in `LSB`-first, and the final value is returned
/// directly because `xorout` is zero.
#[must_use]
pub fn crc16_riello(data: &[u8]) -> u16 {
    let mut crc: u16 = INIT;
    for &b in data {
        crc ^= b as u16;
        let mut i: u8 = 0;
        while i < 8 {
            if (crc & 1) != 0 {
                crc = (crc >> 1) ^ REFPOLY;
            } else {
                crc >>= 1;
            }
            i += 1;
        }
    }
    crc
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- Anchor vectors (hard references) ---

    #[test]
    fn anchor_empty() {
        assert!(crc16_riello(b"") == 0x554D);
    }

    #[test]
    fn anchor_single_zero() {
        assert!(crc16_riello(&[0x00]) == 0x99B4);
    }

    #[test]
    fn anchor_single_ff() {
        assert!(crc16_riello(&[0xFF]) == 0x96CC);
    }

    #[test]
    fn anchor_lowercase_a() {
        assert!(crc16_riello(b"a") == 0xEB3B);
    }

    #[test]
    fn anchor_check_vector() {
        assert!(crc16_riello(b"123456789") == 0x63D0);
    }

    #[test]
    fn anchor_empty_equals_init() {
        assert!(crc16_riello(b"") == INIT);
    }

    // --- Multi-byte hard-coded vectors ---

    #[test]
    fn multibyte_abc() {
        assert!(crc16_riello(b"abc") == 0x0CAF);
    }

    #[test]
    fn multibyte_prism() {
        assert!(crc16_riello(b"Prism") == 0xEB5A);
    }

    #[test]
    fn multibyte_one_two_three() {
        assert!(crc16_riello(&[0x01, 0x02, 0x03]) == 0x0FB1);
    }

    #[test]
    fn multibyte_quick_fox() {
        assert!(crc16_riello(b"The quick brown fox") == 0x3EEE);
    }

    #[test]
    fn multibyte_sixteen_ff() {
        let data = [0xFFu8; 16];
        assert!(crc16_riello(&data) == 0xBB71);
    }

    #[test]
    fn multibyte_sixteen_zero() {
        let data = [0x00u8; 16];
        assert!(crc16_riello(&data) == 0x3971);
    }

    #[test]
    fn multibyte_two_zero() {
        assert!(crc16_riello(&[0x00, 0x00]) != crc16_riello(&[0x00]));
    }

    #[test]
    fn multibyte_zero_then_ff() {
        assert!(crc16_riello(&[0x00, 0xFF]) == 0xFC4E);
    }

    #[test]
    fn multibyte_ff_then_zero() {
        assert!(crc16_riello(&[0xFF, 0x00]) == 0x0CF6);
    }

    #[test]
    fn multibyte_upper_aa() {
        assert!(crc16_riello(b"AA") == 0xFF05);
    }

    #[test]
    fn multibyte_lower_aa() {
        assert!(crc16_riello(b"aa") == 0xFD34);
    }

    #[test]
    fn multibyte_incrementing_256() {
        let mut data = [0u8; 256];
        let mut i = 0usize;
        while i < data.len() {
            data[i] = i as u8;
            i += 1;
        }
        assert!(crc16_riello(&data) == 0x8543);
    }

    // --- Order sensitivity ---

    #[test]
    fn order_matters_two_bytes() {
        assert!(crc16_riello(&[0x00, 0xFF]) != crc16_riello(&[0xFF, 0x00]));
    }

    #[test]
    fn order_matters_strings() {
        assert!(crc16_riello(b"ab") != crc16_riello(b"ba"));
    }

    #[test]
    fn case_sensitivity() {
        assert!(crc16_riello(b"AA") != crc16_riello(b"aa"));
    }

    // --- Determinism ---

    #[test]
    fn deterministic_empty() {
        assert!(crc16_riello(b"") == crc16_riello(b""));
    }

    #[test]
    fn deterministic_check_vector() {
        let a = crc16_riello(b"123456789");
        let b = crc16_riello(b"123456789");
        assert!(a == b);
    }

    #[test]
    fn deterministic_repeated_many() {
        let data = [0x3Cu8; 64];
        let first = crc16_riello(&data);
        let mut n = 0u32;
        while n < 50 {
            assert!(crc16_riello(&data) == first);
            n += 1;
        }
    }

    #[test]
    fn deterministic_prism() {
        assert!(crc16_riello(b"Prism") == crc16_riello(b"Prism"));
    }

    // --- Range / structural invariants ---

    #[test]
    fn result_within_u16_range() {
        let v = crc16_riello(b"range-check");
        assert!((0x0000u16..=0xFFFFu16).contains(&v));
    }

    #[test]
    fn single_byte_values_within_range() {
        let mut byte = 0u16;
        while byte <= 0xFF {
            let v = crc16_riello(&[byte as u8]);
            assert!((0x0000u16..=0xFFFFu16).contains(&v));
            byte += 1;
        }
    }

    #[test]
    fn all_single_bytes_differ_from_init_sometimes() {
        // At least one single-byte input must change the register away from INIT.
        let mut byte = 0u16;
        let mut changed = false;
        while byte <= 0xFF {
            if crc16_riello(&[byte as u8]) != INIT {
                changed = true;
            }
            byte += 1;
        }
        assert!(changed);
    }

    // --- Incremental / prefix behavior ---

    #[test]
    fn prefix_changes_result() {
        assert!(crc16_riello(b"12345") != crc16_riello(b"123456789"));
    }

    #[test]
    fn appending_zero_changes_result() {
        let base = crc16_riello(b"data");
        let extended = crc16_riello(b"data\0");
        assert!(base != extended);
    }

    // --- Long input stability ---

    #[test]
    fn long_input_fixed_value() {
        let data = [0x5Au8; 1000];
        assert!(crc16_riello(&data) == 0x9B10);
    }

    #[test]
    fn long_input_deterministic() {
        let data = [0xA5u8; 2048];
        let first = crc16_riello(&data);
        assert!(crc16_riello(&data) == first);
    }

    #[test]
    fn long_input_within_range() {
        let data = [0x7Eu8; 4096];
        let v = crc16_riello(&data);
        assert!((0x0000u16..=0xFFFFu16).contains(&v));
    }

    #[test]
    fn long_zero_run_distinct_lengths() {
        let short = [0x00u8; 100];
        let long = [0x00u8; 101];
        assert!(crc16_riello(&short) != crc16_riello(&long));
    }

    #[test]
    fn long_input_blocked_matches_whole() {
        // The whole-slice computation is independent of how we view the array.
        let data = [0x11u8; 512];
        let whole = crc16_riello(&data);
        let again = crc16_riello(&data[..]);
        assert!(whole == again);
    }

    // --- Constant sanity ---

    #[test]
    fn refpoly_constant_value() {
        assert!(REFPOLY == 0x8408);
    }

    #[test]
    fn init_constant_value() {
        assert!(INIT == 0x554D);
    }

    #[test]
    fn init_nonzero() {
        assert!(INIT != 0);
    }

    #[test]
    fn block_length_multiple_check() {
        // A 12-byte buffer length is divisible by 4; use the clippy-preferred
        // `is_multiple_of` helper on an unsigned length while exercising the
        // checksum over that buffer.
        let data = [0x24u8; 12];
        assert!(data.len().is_multiple_of(4));
        let v = crc16_riello(&data);
        assert!((0x0000u16..=0xFFFFu16).contains(&v));
    }
}
